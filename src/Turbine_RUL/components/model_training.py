import os
import numpy as np
import pandas as pd
import pickle
import warnings
from datetime import datetime
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from ngboost.distns import Poisson

from src.Turbine_RUL.config.configuration import ConfigurationManager
from src.Turbine_RUL.utils.common import calculate_RUL

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
RUL_THRESHOLD = 135

class CustomGroupKFold(GroupKFold):
    """CV Splitter which drops validation records with RUL values outside of test set ranges"""
    def split(self, X, y, groups):
        splits = super().split(X, y, groups)
        for train_ind, val_ind in splits:
            # Filter validation indices to keep only RUL values in range [6, 135]
            yield train_ind, val_ind[(y[val_ind] > 6) & (y[val_ind] < RUL_THRESHOLD)]

class ModelTrainer:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_model_training_config()
        
    def prepare_training_data(self, train_features_path):
        """Prepare data for training by loading features and calculating RUL"""
        print("Loading engineered features...")
        train_ftrs = pd.read_csv(train_features_path)
        
        # Handle index - if saved with index, set it properly
        if 'unit_id' in train_ftrs.columns and 'time_cycles' in train_ftrs.columns:
            train_ftrs = train_ftrs.set_index(['unit_id', 'time_cycles'])
        
        # Extract features (X) and prepare for training
        X_train = train_ftrs.reset_index().drop(columns=['unit_id'])
        
        # Get unit information for calculating RUL
        train_units_df = train_ftrs.index.to_frame(index=False)
        
        # Calculate RUL (y)
        print("Calculating RUL values...")
        y_train = calculate_RUL(train_units_df, upper_threshold=RUL_THRESHOLD)
        
        return X_train, y_train, train_units_df
    
    def create_ngboost_model(self):
        """Create NGBoost model with best parameters from experiments"""
        # Decision tree base learner with best parameters
        ngb_base = DecisionTreeRegressor(
            criterion='friedman_mse',
            max_depth=5,
            max_features=0.8,
            min_samples_leaf=50
        )
        
        # NGBoost regressor with Poisson distribution
        ngb = NGBRegressor(
            Dist=Poisson,
            Base=ngb_base,
            learning_rate=0.01,
            col_sample=0.8,
            minibatch_frac=0.5,
            n_estimators=400,
            verbose=False
        )
        
        return ngb
    
    def evaluate_model(self, model, X, y, groups, cv, n_jobs=None):
        """Evaluate a model with Cross-Validation"""
        print("\nPerforming cross-validation...")
        cv_results = cross_validate(
            model, X=X, y=y, groups=groups,
            scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error'],
            cv=cv, 
            return_train_score=True, 
            return_estimator=True,
            n_jobs=n_jobs
        )
        
        # Print results
        for k, v in cv_results.items():
            if k.startswith('train_') or k.startswith('test_'):
                k_sp = k.split('_')
                metric_name = " ".join(k_sp[2:])
                print(f'[{k_sp[0]}] :: {metric_name} : {np.abs(v.mean()):.2f} +- {v.std():.2f}')
        
        return cv_results
    
    def extract_feature_importance(self, model, feature_names):
        """Extract and sort feature importances"""
        importances = model.feature_importances_.ravel()
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def initiate_model_training(self):
        """Main model training process"""
        print("Starting Model Training...")
        start_time = datetime.now()
        
        # 1. Load and prepare data
        X_train, y_train, train_units_df = self.prepare_training_data(
            self.config.engineered_features_path
        )
        
        # Filter out samples with RUL <= 0
        valid_mask = y_train > 0
        X_train_valid = X_train[valid_mask]
        y_train_valid = y_train[valid_mask]
        train_units_valid = train_units_df[valid_mask]
        
        print(f"Training data shape: {X_train_valid.shape}")
        print(f"Target values range: [{y_train_valid.min()}, {y_train_valid.max()}]")
        
        # 2. Create model
        ngb_model = self.create_ngboost_model()
        
        # 3. Cross-validation evaluation
        cv_splitter = CustomGroupKFold(n_splits=4)
        cv_results = self.evaluate_model(
            ngb_model,
            X=X_train_valid.values,
            y=y_train_valid,
            groups=train_units_valid['unit_id'],
            cv=cv_splitter,
            n_jobs=4
        )
        
        # 4. Train final model on all data
        print("\nTraining final model on all data...")
        selected_features = X_train.columns.tolist()
        ngb_model.fit(X_train_valid[selected_features], y_train_valid)
        
        # 5. Extract feature importance
        feature_importance = self.extract_feature_importance(ngb_model, selected_features)
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        # 6. Save model and artifacts
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        
        # Save the trained model
        with open(self.config.model_path, 'wb') as f:
            pickle.dump(ngb_model, f)
        
        # Save selected features list
        with open(self.config.selected_features_path, 'wb') as f:
            pickle.dump(selected_features, f)
        
        # Save feature importance
        feature_importance.to_csv(self.config.feature_importance_path, index=False)
        
        # Save cross-validation results
        cv_results_df = pd.DataFrame({
            'train_rmse': -cv_results['train_neg_root_mean_squared_error'],
            'test_rmse': -cv_results['test_neg_root_mean_squared_error'],
            'train_mae': -cv_results['train_neg_mean_absolute_error'],
            'test_mae': -cv_results['test_neg_mean_absolute_error']
        })
        cv_results_df.to_csv(self.config.cv_results_path, index=False)
        
        # Calculate and save final metrics
        train_time = (datetime.now() - start_time).total_seconds()
        metrics = {
            'train_rmse_mean': cv_results_df['train_rmse'].mean(),
            'train_rmse_std': cv_results_df['train_rmse'].std(),
            'test_rmse_mean': cv_results_df['test_rmse'].mean(),
            'test_rmse_std': cv_results_df['test_rmse'].std(),
            'train_mae_mean': cv_results_df['train_mae'].mean(),
            'train_mae_std': cv_results_df['train_mae'].std(),
            'test_mae_mean': cv_results_df['test_mae'].mean(),
            'test_mae_std': cv_results_df['test_mae'].std(),
            'n_features': len(selected_features),
            'n_samples': len(X_train_valid),
            'training_time_seconds': train_time
        }
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.config.metrics_path, index=False)
        
        print(f"\nModel training completed in {train_time:.2f} seconds!")
        print(f"Model saved at: {self.config.model_path}")
        print(f"Selected features saved at: {self.config.selected_features_path}")
        print(f"Feature importance saved at: {self.config.feature_importance_path}")
        print(f"Cross-validation results saved at: {self.config.cv_results_path}")
        print(f"Training metrics saved at: {self.config.metrics_path}")
        
        print("\nFinal Model Performance:")
        print(f"Test RMSE: {metrics['test_rmse_mean']:.2f} ± {metrics['test_rmse_std']:.2f}")
        print(f"Test MAE: {metrics['test_mae_mean']:.2f} ± {metrics['test_mae_std']:.2f}")
        
        return self.config.model_path, metrics