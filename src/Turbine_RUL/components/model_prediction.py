import os
import numpy as np
import pandas as pd
import pickle
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from src.Turbine_RUL.config.configuration import ConfigurationManager
from src.Turbine_RUL.utils.common import calculate_RUL

# Suppress warnings
warnings.filterwarnings("ignore")

class ModelPrediction:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_model_prediction_config()
        
    def load_artifacts(self):
        """Load all trained artifacts"""
        print("Loading trained artifacts...")
        
        # Load trained model
        with open(self.config.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✅ Model loaded from: {self.config.model_path}")
        
        # Load feature engineering pipelines
        with open(self.config.long_term_pipeline_path, 'rb') as f:
            self.features_long_pipe = pickle.load(f)
        print(f"✅ Long-term pipeline loaded from: {self.config.long_term_pipeline_path}")
        
        with open(self.config.short_term_pipeline_path, 'rb') as f:
            self.features_short_pipe = pickle.load(f)
        print(f"✅ Short-term pipeline loaded from: {self.config.short_term_pipeline_path}")
        
        # Load preprocessor
        with open(self.config.preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        print(f"✅ Preprocessor loaded from: {self.config.preprocessor_path}")
        
        # Load selected features
        with open(self.config.selected_features_path, 'rb') as f:
            self.selected_features = pickle.load(f)
        print(f"✅ Selected features loaded: {len(self.selected_features)} features")
    
    def extract_and_save_true_rul(self, raw_test_data):
        """Extract true RUL values before preprocessing and save them"""
        true_rul_per_engine = None
        
        if 'rul' in raw_test_data.columns:
            print("Extracting true RUL values...")
            # Get the last RUL value for each engine (true remaining useful life at test time)
            true_rul_per_engine = raw_test_data.groupby('unit_id')['rul'].last()
            
            # Create directory and save
            os.makedirs(os.path.dirname(self.config.evaluation_rul_path), exist_ok=True)
            true_rul_per_engine.to_csv(self.config.evaluation_rul_path)
            print(f"✅ True RUL values saved: {len(true_rul_per_engine)} engines")
            print(f"RUL range: {true_rul_per_engine.min():.1f} to {true_rul_per_engine.max():.1f}")
        else:
            print("⚠️ No RUL column found in test data - evaluation will be skipped")
            
        return true_rul_per_engine
    
    def preprocess_test_data(self, test_data):
        """Apply preprocessing to test data (this will remove RUL column)"""
        print("Preprocessing test data...")
        
        # The preprocessor includes ColumnDropper which removes 'rul' column
        test_preprocessed = self.preprocessor.transform(test_data)
        print(f"✅ Test data preprocessed: {test_preprocessed.shape}")
        
        return test_preprocessed
    
    def extract_features(self, test_data_preprocessed):
        """Extract features using trained pipelines"""
        print("Extracting features from test data...")
        
        # Extract long-term features
        print("Extracting long-term features...")
        test_long_h_fts = self.features_long_pipe.transform(test_data_preprocessed)
        
        # Extract short-term features  
        print("Extracting short-term features...")
        test_short_h_fts = self.features_short_pipe.transform(test_data_preprocessed)
        
        # Set proper index names before merging
        print("Setting proper index names...")
        test_long_h_fts.index = test_long_h_fts.index.set_names(['unit_id', 'time_cycles'])
        test_short_h_fts.index = test_short_h_fts.index.set_names(['unit_id', 'time_cycles'])
        
        # Merge features with error handling
        print("Merging features...")
        try:
            test_ftrs = test_long_h_fts.merge(test_short_h_fts, how='inner',
                                              right_index=True, left_index=True)
            print(f"✅ Merge successful! Combined features shape: {test_ftrs.shape}")
        except Exception as e:
            print(f"Direct merge failed: {e}")
            print("Trying alternative merge approach...")
            
            # Alternative approach: reset index and merge on columns
            long_reset = test_long_h_fts.reset_index()
            short_reset = test_short_h_fts.reset_index()
            merged_reset = long_reset.merge(short_reset, on=['unit_id', 'time_cycles'], how='inner')
            test_ftrs = merged_reset.set_index(['unit_id', 'time_cycles'])
            print(f"✅ Alternative merge successful! Shape: {test_ftrs.shape}")
        
        # Debug: Print feature information
        print(f"Features after merge: {test_ftrs.shape}")
        print(f"Feature columns: {test_ftrs.columns.tolist()}")
        
        return test_ftrs
    
    def prepare_prediction_data(self, test_features):
        """Prepare features for final prediction"""
        print("Preparing data for prediction...")
        
        # Prepare features for prediction
        X_test_features = test_features.reset_index().drop(columns=['unit_id'])
        test_units = test_features.index.to_frame(index=False)
        
        print(f"Total test samples: {len(X_test_features)}")
        print(f"Unique test engines: {test_units['unit_id'].nunique()}")
        
        # Get only the last cycle for each engine (most predictive for RUL)
        X_test_last = X_test_features.groupby(test_units['unit_id']).last()
        X_test_last.columns = [str(col) for col in X_test_last.columns]
        engine_ids = X_test_last.index.values  # These are the unit_ids
        print(f"Last cycle data shape: {X_test_last.shape}")
        print(f"Last cycle columns: {X_test_last.columns.tolist()}")
        
        # Debug: Print selected features that were saved during training
        print(f"Required features from training: {self.selected_features}")
        
        # Check feature alignment - CRITICAL FIX
        available_features = set(X_test_last.columns)
        required_features = set(self.selected_features)
        missing_features = required_features - available_features
        extra_features = available_features - required_features
        
        if missing_features:
            print(f"❌ ERROR: Missing {len(missing_features)} required features!")
            print(f"Missing features: {list(missing_features)}")
            print(f"Available features: {list(available_features)}")
            print(f"Required features: {list(required_features)}")
            
            # This is a critical error - we cannot proceed with different features
            # than what the model was trained on
            raise ValueError(
                f"Feature mismatch! Model requires {len(self.selected_features)} features "
                f"but only {len(available_features)} are available. "
                f"Missing features: {missing_features}. "
                f"This indicates a pipeline inconsistency between training and prediction."
            )
        else:
            working_features = self.selected_features
            print(f"✅ All {len(self.selected_features)} required features available!")
        
        if extra_features:
            print(f"ℹ️ Note: {len(extra_features)} extra features in test data (will be ignored)")
        
        # Ensure features are in the exact same order as training
        X_test_final = X_test_last[working_features]
        print(f"Final feature matrix shape: {X_test_final.shape}")
        print(f"Expected features: {len(self.selected_features)}")
        
        return X_test_final, engine_ids
    
    def make_predictions(self, X_test_prepared):
        """Make RUL predictions"""
        print("Making RUL predictions...")
        
        pred_rul = self.model.predict(X_test_prepared.values)
        print(f"✅ Predictions completed! Shape: {pred_rul.shape}")
        print(f"Prediction range: {pred_rul.min():.1f} to {pred_rul.max():.1f}")
        
        return pred_rul
    
    def calculate_metrics(self, true_rul_series, pred_rul, engine_ids):
        """Calculate evaluation metrics"""
        print("Calculating evaluation metrics...")
        
        # Align predictions with true RUL values
        # true_rul_series is indexed by unit_id, pred_rul is array, engine_ids maps array positions to unit_ids
        true_rul_aligned = []
        pred_rul_aligned = []
        
        for i, engine_id in enumerate(engine_ids):
            if engine_id in true_rul_series.index:
                true_rul_aligned.append(true_rul_series[engine_id])
                pred_rul_aligned.append(pred_rul[i])
        
        true_rul_array = np.array(true_rul_aligned)
        pred_rul_array = np.array(pred_rul_aligned)
        
        print(f"Aligned {len(true_rul_array)} engines for evaluation")
        
        if len(true_rul_array) == 0:
            print("⚠️ No matching engines found between predictions and true RUL")
            return None, None, None
        
        # Calculate basic metrics
        rmse = np.sqrt(mean_squared_error(true_rul_array, pred_rul_array))
        mae = mean_absolute_error(true_rul_array, pred_rul_array)
        
        # Handle MAPE calculation (avoid division by zero)
        non_zero_mask = true_rul_array != 0
        if np.sum(non_zero_mask) > 0:
            mape = mean_absolute_percentage_error(true_rul_array[non_zero_mask], pred_rul_array[non_zero_mask])
        else:
            mape = float('inf')
            print("⚠️ WARNING: All true RUL values are zero, MAPE set to infinity")
        
        # Custom RUL score function (domain-specific metric)
        def rul_score_f(err):
            if err >= 0:  # Late prediction (more dangerous)
                return np.exp(err / 10) - 1
            else:  # Early prediction (safer)
                return np.exp(-err / 13) - 1
        
        def rul_score(true_rul, estimated_rul):
            err = estimated_rul - true_rul
            return np.sum([rul_score_f(x) for x in err])
        
        custom_score = rul_score(true_rul_array, pred_rul_array)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'accuracy': (1 - mape) * 100 if mape != float('inf') else 0,
            'custom_score': custom_score,
            'n_engines': len(true_rul_array),
            'pred_range': (pred_rul_array.min(), pred_rul_array.max()),
            'true_range': (true_rul_array.min(), true_rul_array.max())
        }
        
        return metrics, true_rul_array, pred_rul_array
    
    def create_evaluation_plots(self, true_rul, pred_rul, metrics):
        """Create evaluation plots"""
        print("Creating evaluation plots...")
        
        try:
            plt.figure(figsize=(15, 10))
            
            # Subplot 1: Predictions vs True values
            plt.subplot(2, 3, 1)
            plt.scatter(true_rul, pred_rul, alpha=0.6, s=30)
            min_val = min(true_rul.min(), pred_rul.min())
            max_val = max(true_rul.max(), pred_rul.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            plt.xlabel('True RUL')
            plt.ylabel('Predicted RUL')
            plt.title('Predicted vs True RUL')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Residuals plot
            plt.subplot(2, 3, 2)
            residuals = pred_rul - true_rul
            plt.scatter(true_rul, residuals, alpha=0.6, s=30)
            plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
            plt.xlabel('True RUL')
            plt.ylabel('Residuals (Predicted - True)')
            plt.title('Residuals vs True RUL')
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: Error distribution
            plt.subplot(2, 3, 3)
            plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
            plt.axvline(x=residuals.mean(), color='orange', linestyle='-', linewidth=2,
                       label=f'Mean Error: {residuals.mean():.2f}')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 4: Absolute Error vs True RUL
            plt.subplot(2, 3, 4)
            abs_errors = np.abs(residuals)
            plt.scatter(true_rul, abs_errors, alpha=0.6, s=30)
            plt.xlabel('True RUL')
            plt.ylabel('Absolute Error')
            plt.title('Absolute Error vs True RUL')
            plt.grid(True, alpha=0.3)
            
            # Subplot 5: Prediction accuracy by RUL range
            plt.subplot(2, 3, 5)
            # Bin true RUL values and calculate average absolute error in each bin
            bins = np.linspace(true_rul.min(), true_rul.max(), 10)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_errors = []
            
            for i in range(len(bins)-1):
                mask = (true_rul >= bins[i]) & (true_rul < bins[i+1])
                if np.sum(mask) > 0:
                    bin_errors.append(abs_errors[mask].mean())
                else:
                    bin_errors.append(0)
            
            plt.bar(bin_centers, bin_errors, width=(bins[1]-bins[0])*0.8, alpha=0.7)
            plt.xlabel('True RUL Range')
            plt.ylabel('Mean Absolute Error')
            plt.title('Prediction Error by RUL Range')
            plt.grid(True, alpha=0.3)
            
            # Subplot 6: Performance metrics text
            plt.subplot(2, 3, 6)
            plt.axis('off')
            metrics_text = f"""
EVALUATION METRICS

RMSE: {metrics['rmse']:.2f}
MAE: {metrics['mae']:.2f}
MAPE: {metrics['mape']:.3f}
Accuracy: {metrics['accuracy']:.1f}%
Custom Score: {metrics['custom_score']:.2f}

Engines Tested: {metrics['n_engines']}
Pred Range: [{metrics['pred_range'][0]:.1f}, {metrics['pred_range'][1]:.1f}]
True Range: [{metrics['true_range'][0]:.1f}, {metrics['true_range'][1]:.1f}]
            """
            plt.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            
            # Save plot
            os.makedirs(os.path.dirname(self.config.evaluation_plots_path), exist_ok=True)
            plt.savefig(self.config.evaluation_plots_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Evaluation plots saved at: {self.config.evaluation_plots_path}")
            
        except Exception as e:
            print(f"⚠️ Plotting failed: {e}")
    
    def save_results(self, predictions, engine_ids, true_rul_series=None, metrics=None):
        """Save prediction results and metrics"""
        print("Saving prediction results...")
        
        # Create results DataFrame
        results_data = {
            'engine_id': engine_ids,
            'predicted_rul': predictions
        }
        
        # Add true RUL and error calculations if available
        if true_rul_series is not None:
            true_rul_values = []
            errors = []
            abs_errors = []
            percentage_errors = []
            
            for engine_id, pred in zip(engine_ids, predictions):
                if engine_id in true_rul_series.index:
                    true_val = true_rul_series[engine_id]
                    error = pred - true_val
                    abs_error = abs(error)
                    perc_error = abs_error / true_val * 100 if true_val != 0 else float('inf')
                else:
                    true_val = np.nan
                    error = np.nan
                    abs_error = np.nan
                    perc_error = np.nan
                
                true_rul_values.append(true_val)
                errors.append(error)
                abs_errors.append(abs_error)
                percentage_errors.append(perc_error)
            
            results_data.update({
                'true_rul': true_rul_values,
                'error': errors,
                'abs_error': abs_errors,
                'percentage_error': percentage_errors
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Create directories
        os.makedirs(os.path.dirname(self.config.predictions_path), exist_ok=True)
        
        # Save predictions
        results_df.to_csv(self.config.predictions_path, index=False)
        print(f"✅ Predictions saved at: {self.config.predictions_path}")
        
        # Save metrics if available
        if metrics is not None:
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(self.config.evaluation_metrics_path, index=False)
            print(f"✅ Evaluation metrics saved at: {self.config.evaluation_metrics_path}")
        
        return results_df
    
    def print_evaluation_summary(self, metrics):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("🎯 MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Number of test engines: {metrics['n_engines']}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        if metrics['mape'] != float('inf'):
            print(f"MAPE: {metrics['mape']:.3f}")
            print(f"Accuracy: {metrics['accuracy']:.1f}%")
        else:
            print("MAPE: inf (zero true values)")
        print(f"Custom RUL Score: {metrics['custom_score']:.2f}")
        print(f"Prediction range: {metrics['pred_range'][0]:.1f} to {metrics['pred_range'][1]:.1f}")
        print(f"True RUL range: {metrics['true_range'][0]:.1f} to {metrics['true_range'][1]:.1f}")
        print("="*60)
    
    def initiate_model_prediction(self):
        """Complete prediction pipeline with evaluation"""
        print("Starting Model Prediction Pipeline...")
        start_time = datetime.now()
        
        # 1. Load raw test data (with RUL column)
        print("Loading raw test data...")
        raw_test_data = pd.read_csv(self.config.test_data_path)
        print(f"Raw test data shape: {raw_test_data.shape}")
        
        # 2. Extract and store true RUL values BEFORE preprocessing
        true_rul_per_engine = self.extract_and_save_true_rul(raw_test_data)
        
        # 3. Load all trained artifacts
        self.load_artifacts()
        
        # 4. Preprocess test data (this removes RUL column via ColumnDropper)
        test_data_preprocessed = self.preprocess_test_data(raw_test_data)
        
        # 5. Extract features
        test_features = self.extract_features(test_data_preprocessed)
        
        # 6. Prepare data for prediction
        X_test_prepared, engine_ids = self.prepare_prediction_data(test_features)
        
        # 7. Make predictions
        predictions = self.make_predictions(X_test_prepared)
        
        # 8. Evaluate if true RUL values are available
        metrics = None
        true_rul_array = None
        pred_rul_array = None
        
        if true_rul_per_engine is not None:
            metrics, true_rul_array, pred_rul_array = self.calculate_metrics(
                true_rul_per_engine, predictions, engine_ids
            )
            
            if metrics is not None:
                # Print evaluation summary
                self.print_evaluation_summary(metrics)
                
                # Create evaluation plots
                self.create_evaluation_plots(true_rul_array, pred_rul_array, metrics)
        
        # 9. Save results
        results_df = self.save_results(predictions, engine_ids, true_rul_per_engine, metrics)
        
        # Calculate total time
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Model prediction completed in {total_time:.2f} seconds!")
        
        # Return summary
        return {
            'predictions_path': self.config.predictions_path,
            'evaluation_rul_path': self.config.evaluation_rul_path,
            'evaluation_metrics_path': self.config.evaluation_metrics_path,
            'evaluation_plots_path': self.config.evaluation_plots_path,
            'total_time': total_time,
            'n_predictions': len(predictions),
            'metrics': metrics,
            'n_engines_evaluated': len(true_rul_array) if true_rul_array is not None else 0
        }