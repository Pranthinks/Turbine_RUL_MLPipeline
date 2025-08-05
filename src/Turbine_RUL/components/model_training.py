import os
import numpy as np
import pandas as pd
import pickle
import warnings
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import GroupKFold, cross_validate, ParameterGrid
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from ngboost import NGBRegressor
from ngboost.distns import Poisson
from dotenv import load_dotenv
load_dotenv()

from src.Turbine_RUL.config.configuration import ConfigurationManager
from src.Turbine_RUL.utils.common import calculate_RUL

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
RUL_THRESHOLD = 135

# Setup MLflow tracking
if os.getenv("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
if os.getenv("MLFLOW_TRACKING_USERNAME"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
if os.getenv("MLFLOW_TRACKING_PASSWORD"):
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

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
        
        # Setup MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        
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
    
    def evaluate_model(self, model, X, y, groups, cv, n_jobs=None):
        """Evaluate a model with Cross-Validation"""
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
        """Main model training process with hyperparameter tuning"""
        print("Starting Model Training with Hyperparameter Tuning...")
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
        
        # 2. Define parameter grids (4 combinations each)
        ngboost_params = [
            {'learning_rate': 0.01, 'n_estimators': 200},
            {'learning_rate': 0.02, 'n_estimators': 200},
            {'learning_rate': 0.02, 'n_estimators': 300},
            {'learning_rate': 0.01, 'n_estimators': 400}
        ]
        
        rf_params = [
            {'n_estimators': 100, 'max_depth': 10},
            {'n_estimators': 100, 'max_depth': None},
            {'n_estimators': 200, 'max_depth': 10},
            {'n_estimators': 200, 'max_depth': None}
        ]
        
        # 3. Set MLflow experiment
        mlflow.set_experiment("Turbine_RUL_Hyperparameter_Tuning")
        
        cv_splitter = CustomGroupKFold(n_splits=4)
        best_model = None
        best_score = float('inf')
        best_params = None
        best_model_name = None
        best_cv_results = None
        
        # 4. NGBoost hyperparameter tuning
        print(f"\n{'='*50}")
        print("TUNING NGBOOST PARAMETERS")
        print(f"{'='*50}")
        
        for i, params in enumerate(ngboost_params, 1):
            print(f"\nNGBoost {i}/4: {params}")
            
            with mlflow.start_run():
                # Create model
                ngb_base = DecisionTreeRegressor(
                    criterion='friedman_mse', max_depth=5, 
                    max_features=0.8, min_samples_leaf=50
                )
                model = NGBRegressor(
                    Dist=Poisson, Base=ngb_base,
                    learning_rate=params['learning_rate'],
                    n_estimators=params['n_estimators'],
                    col_sample=0.8, minibatch_frac=0.5, verbose=False
                )
                
                # Set tags and log parameters
                mlflow.set_tag("model_type", "ngboost")
                mlflow.log_params(params)
                mlflow.log_param("max_depth", 5)
                mlflow.log_param("max_features", 0.8)
                mlflow.log_param("col_sample", 0.8)
                
                # Cross-validation
                cv_results = self.evaluate_model(
                    model, X_train_valid.values, y_train_valid,
                    train_units_valid['unit_id'], cv_splitter, n_jobs=2
                )
                
                # Train model and get feature importance
                model.fit(X_train_valid, y_train_valid)
                feature_importance = self.extract_feature_importance(model, X_train_valid.columns)
                
                # Calculate and log metrics
                test_rmse = np.abs(cv_results['test_neg_root_mean_squared_error'].mean())
                test_mae = np.abs(cv_results['test_neg_mean_absolute_error'].mean())
                train_rmse = np.abs(cv_results['train_neg_root_mean_squared_error'].mean())
                
                mlflow.log_metric("test_rmse", test_rmse)
                mlflow.log_metric("test_mae", test_mae)
                mlflow.log_metric("train_rmse", train_rmse)
                
                # Log feature importance
                importance_path = "temp_feature_importance.csv"
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)
                
                # Check if best
                if test_rmse < best_score:
                    best_score = test_rmse
                    best_model = model
                    best_params = params.copy()
                    best_params.update({'max_depth': 5, 'max_features': 0.8, 'col_sample': 0.8})
                    best_model_name = "ngboost"
                    best_cv_results = cv_results
        
        # 5. Random Forest hyperparameter tuning
        print(f"\n{'='*50}")
        print("TUNING RANDOM FOREST PARAMETERS")
        print(f"{'='*50}")
        
        for i, params in enumerate(rf_params, 1):
            print(f"\nRandomForest {i}/4: {params}")
            
            with mlflow.start_run():
                # Create model
                model = RandomForestRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=5, min_samples_leaf=2,
                    max_features=0.8, random_state=42, n_jobs=-1
                )
                
                # Set tags and log parameters
                mlflow.set_tag("model_type", "random_forest")
                mlflow.log_params(params)
                mlflow.log_param("min_samples_split", 5)
                mlflow.log_param("min_samples_leaf", 2)
                mlflow.log_param("max_features", 0.8)
                
                # Cross-validation
                cv_results = self.evaluate_model(
                    model, X_train_valid.values, y_train_valid,
                    train_units_valid['unit_id'], cv_splitter, n_jobs=2
                )
                
                # Train model and get feature importance
                model.fit(X_train_valid, y_train_valid)
                feature_importance = self.extract_feature_importance(model, X_train_valid.columns)
                
                # Calculate and log metrics
                test_rmse = np.abs(cv_results['test_neg_root_mean_squared_error'].mean())
                test_mae = np.abs(cv_results['test_neg_mean_absolute_error'].mean())
                train_rmse = np.abs(cv_results['train_neg_root_mean_squared_error'].mean())
                
                mlflow.log_metric("test_rmse", test_rmse)
                mlflow.log_metric("test_mae", test_mae)
                mlflow.log_metric("train_rmse", train_rmse)
                
                # Log feature importance
                importance_path = "temp_feature_importance.csv"
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)
                
                # Log sklearn model
                # --- MODIFIED PART ---
        # Save the model locally first
                local_model_path = "rf_model.pkl"
                with open(local_model_path, 'wb') as f:
                    pickle.dump(model, f)
        
        # Log the file as an artifact
                mlflow.log_artifact(local_model_path, "model")
        
        # Clean up the local file
                os.remove(local_model_path)
        # --- END MODIFIED PART ---
                
                # Check if best
                if test_rmse < best_score:
                    best_score = test_rmse
                    best_model = model
                    best_params = params.copy()
                    best_params.update({'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.8})
                    best_model_name = "random_forest"
                    best_cv_results = cv_results
        
        # 6. Final best model logging
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_model_name.upper()} (RMSE: {best_score:.3f})")
        print(f"BEST PARAMETERS: {best_params}")
        print(f"{'='*60}")
        
        # Log best model to separate experiment
        mlflow.set_experiment("Turbine_RUL_Best_Model")
        with mlflow.start_run() as run:
            mlflow.set_tag("model_type", best_model_name)
            mlflow.set_tag("experiment_type", "best_model")
            
            # Log best parameters
            mlflow.log_params(best_params)
            
            # Log best metrics
            mlflow.log_metric("best_test_rmse", best_score)
            mlflow.log_metric("best_test_mae", np.abs(best_cv_results['test_neg_mean_absolute_error'].mean()))
            
            # Log best model
            if best_model_name == "ngboost":
                model_path = "best_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(best_model, f)
                mlflow.log_artifact(model_path, "model")
                os.remove(model_path)
            else:
                model_path = "best_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(best_model, f)
                mlflow.log_artifact(model_path, "model")
                os.remove(model_path)
            
            print(f"✅ Best model logged to MLflow")
            print(f"🔗 Model URI: runs:/{run.info.run_id}/model")
        
        # 7. Extract final feature importance and save locally
        final_feature_importance = self.extract_feature_importance(best_model, X_train_valid.columns)
        print("\nTop 10 most important features:")
        print(final_feature_importance.head(10))
        
        # Save artifacts locally
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        
        # Save the best model
        with open(self.config.model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save selected features list
        selected_features = X_train_valid.columns.tolist()
        with open(self.config.selected_features_path, 'wb') as f:
            pickle.dump(selected_features, f)
        
        # Save feature importance
        final_feature_importance.to_csv(self.config.feature_importance_path, index=False)
        
        # Save cross-validation results
        cv_results_df = pd.DataFrame({
            'train_rmse': -best_cv_results['train_neg_root_mean_squared_error'],
            'test_rmse': -best_cv_results['test_neg_root_mean_squared_error'],
            'train_mae': -best_cv_results['train_neg_mean_absolute_error'],
            'test_mae': -best_cv_results['test_neg_mean_absolute_error']
        })
        cv_results_df.to_csv(self.config.cv_results_path, index=False)
        
        # Calculate final metrics
        total_time = (datetime.now() - start_time).total_seconds()
        metrics = {
            'best_model': best_model_name,
            'best_params': str(best_params),
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
            'total_training_time_seconds': total_time
        }
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.config.metrics_path, index=False)
        
        print(f"\nModel training completed in {total_time:.2f} seconds!")
        print(f"Best model: {best_model_name}")
        print(f"Best parameters: {best_params}")
        print(f"Model saved at: {self.config.model_path}")
        print(f"Selected features saved at: {self.config.selected_features_path}")
        print(f"Feature importance saved at: {self.config.feature_importance_path}")
        print(f"Cross-validation results saved at: {self.config.cv_results_path}")
        print(f"Training metrics saved at: {self.config.metrics_path}")
        
        print("\nFinal Model Performance:")
        print(f"Test RMSE: {metrics['test_rmse_mean']:.2f} ± {metrics['test_rmse_std']:.2f}")
        print(f"Test MAE: {metrics['test_mae_mean']:.2f} ± {metrics['test_mae_std']:.2f}")
        
        return self.config.model_path, metrics