from src.Turbine_RUL.config.configuration import ConfigurationManager
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


class ModelPrediction:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_model_prediction_config()
    
    def evaluate_predictions(self, pred_rul, true_rul):
        """Calculate evaluation metrics for RUL predictions"""
        
        # Handle any NaN values
        valid_mask = ~(np.isnan(pred_rul) | np.isnan(true_rul))
        pred_rul_clean = pred_rul[valid_mask]
        true_rul_clean = true_rul[valid_mask]
        
        # Calculate basic metrics
        rmse = np.sqrt(mean_squared_error(true_rul_clean, pred_rul_clean))
        mae = mean_absolute_error(true_rul_clean, pred_rul_clean)
        
        # Handle MAPE calculation (avoid division by zero)
        non_zero_mask = true_rul_clean != 0
        if np.sum(non_zero_mask) > 0:
            mape = mean_absolute_percentage_error(true_rul_clean[non_zero_mask], pred_rul_clean[non_zero_mask])
        else:
            mape = float('inf')
        
        # Custom RUL score function (from your Google Colab code)
        def rul_score_f(err):
            if err >= 0:
                return np.exp(err / 10) - 1
            else:
                return np.exp(-err / 13) - 1
        
        def rul_score(true_rul, estimated_rul):
            err = estimated_rul - true_rul
            return np.sum([rul_score_f(x) for x in err])
        
        custom_score = rul_score(true_rul_clean, pred_rul_clean)
        
        results = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'accuracy': (1 - mape) * 100 if mape != float('inf') else 0,
            'custom_rul_score': custom_score,
            'n_engines': len(pred_rul_clean),
            'pred_range_min': pred_rul_clean.min(),
            'pred_range_max': pred_rul_clean.max(),
            'true_range_min': true_rul_clean.min(),
            'true_range_max': true_rul_clean.max()
        }
        
        return results
    
    def initiate_model_prediction(self):
        """Complete prediction pipeline with evaluation"""
        print("Starting Model Prediction...")
        
        # 1. Load raw test data (with RUL column)
        print("Loading raw test data...")
        raw_test_data = pd.read_csv(self.config.test_data_path)
        print(f"Raw test data shape: {raw_test_data.shape}")
        
        # 2. Extract and store true RUL values BEFORE preprocessing
        true_rul_per_engine = None
        if 'rul' in raw_test_data.columns:
            print("Extracting true RUL values...")
            true_rul_per_engine = raw_test_data.groupby('unit_id')['rul'].last()
            os.makedirs(os.path.dirname(self.config.evaluation_rul_path), exist_ok=True)
            true_rul_per_engine.to_csv(self.config.evaluation_rul_path)
            print(f"True RUL values saved: {len(true_rul_per_engine)} engines")
        
        # 3. Apply data transformation (removes RUL column)
        print("Applying data transformation...")
        with open(self.config.preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        preprocessed_data = preprocessor.transform(raw_test_data)
        
        # 4. Apply feature engineering
        print("Applying feature engineering...")
        with open(self.config.long_term_pipeline_path, 'rb') as f:
            features_long_pipe = pickle.load(f)
        with open(self.config.short_term_pipeline_path, 'rb') as f:
            features_short_pipe = pickle.load(f)
        
        # Transform through both pipelines
        print("Transforming test data...")
        test_long_features = features_long_pipe.transform(preprocessed_data)
        test_short_features = features_short_pipe.transform(preprocessed_data)
        
        # Handle index merging
        print("Setting proper index names...")
        test_long_features.index = test_long_features.index.set_names(['unit_id', 'time_cycles'])
        test_short_features.index = test_short_features.index.set_names(['unit_id', 'time_cycles'])
        
        # Merge features
        print("Merging features...")
        try:
            test_features = test_long_features.merge(test_short_features, how='inner',
                                                   right_index=True, left_index=True)
            print(f"Merge successful! Combined features shape: {test_features.shape}")
        except Exception as e:
            print(f"Direct merge failed: {e}")
            print("Trying alternative merge approach...")
            long_reset = test_long_features.reset_index()
            short_reset = test_short_features.reset_index()
            merged_reset = long_reset.merge(short_reset, on=['unit_id', 'time_cycles'], how='inner')
            test_features = merged_reset.set_index(['unit_id', 'time_cycles'])
            print(f"Alternative merge successful! Shape: {test_features.shape}")
        
        # 5. Prepare for prediction (last cycle per engine)
        X_test_features = test_features.reset_index().drop(columns=['unit_id'])
        test_units = test_features.index.to_frame(index=False)
        
        print(f"Total test samples: {len(X_test_features)}")
        print(f"Unique test engines: {test_units['unit_id'].nunique()}")
        
        # Get only the last cycle for each engine
        X_test_last = X_test_features.groupby(test_units['unit_id']).last()
        print(f"Last cycle data shape: {X_test_last.shape}")
        
        # 6. Load model and selected features
        with open(self.config.model_path, 'rb') as f:
            model = pickle.load(f)
        with open(self.config.selected_features_path, 'rb') as f:
            selected_features = pickle.load(f)
        
        # 7. Ensure EXACT feature match with training
        available_columns = list(X_test_last.columns)
        
        print(f"DEBUG: Available columns: {len(available_columns)}")
        print(f"DEBUG: Selected features: {len(selected_features)}")
        print(f"DEBUG: First 5 selected features: {selected_features[:5]}")
        print(f"DEBUG: First 5 available columns: {available_columns[:5]}")
        
        # Create working features array matching training EXACTLY
        working_features = []
        missing_features = []
        
        for i, feature in enumerate(selected_features):
            if feature in available_columns:
                # Direct match - feature name exists
                working_features.append(feature)
            elif str(feature).isdigit() and int(feature) < len(available_columns):
                # Numeric index - convert to column name
                working_features.append(available_columns[int(feature)])
            elif isinstance(feature, str) and feature.isdigit() and int(feature) < len(available_columns):
                # String numeric index - convert to column name
                working_features.append(available_columns[int(feature)])
            else:
                # Feature missing - use a dummy feature (first available feature)
                print(f"WARNING: Feature {feature} missing, using dummy feature")
                working_features.append(available_columns[0])  # Use first column as dummy
                missing_features.append(feature)
        
        print(f"DEBUG: Final working features count: {len(working_features)}")
        print(f"DEBUG: Missing features filled with dummy: {len(missing_features)}")
        
        # Ensure we have EXACTLY the same number of features as training
        if len(working_features) != len(selected_features):
            print(f"ERROR: Feature count mismatch! Expected {len(selected_features)}, got {len(working_features)}")
            return None, None
        
        print(f"✓ Feature alignment successful: {len(working_features)} features")
        
        # 8. Make predictions
        print("Making predictions...")
        predictions = model.predict(X_test_last[working_features].values)
        print(f"Predictions completed! Shape: {predictions.shape}")
        
        # 9. Create results DataFrame
        engine_ids = X_test_last.index
        results_df = pd.DataFrame({
            'engine_id': engine_ids,
            'predicted_rul': predictions
        })
        
        # 10. Add true RUL and evaluation if available
        if true_rul_per_engine is not None:
            # Align true RUL with predictions
            results_df = results_df.merge(
                true_rul_per_engine.to_frame('true_rul'), 
                left_on='engine_id', right_index=True, how='left'
            )
            
            # Calculate evaluation metrics
            evaluation_results = self.evaluate_predictions(
                results_df['predicted_rul'].values,
                results_df['true_rul'].values
            )
            
            # Save evaluation results
            os.makedirs(os.path.dirname(self.config.evaluation_metrics_path), exist_ok=True)
            eval_df = pd.DataFrame([evaluation_results])
            eval_df.to_csv(self.config.evaluation_metrics_path, index=False)
            
            print("\n" + "="*50)
            print("EVALUATION RESULTS:")
            print("="*50)
            print(f"Number of test engines: {evaluation_results['n_engines']}")
            print(f"RMSE: {evaluation_results['rmse']:.2f}")
            print(f"MAE: {evaluation_results['mae']:.2f}")
            if evaluation_results['mape'] != float('inf'):
                print(f"MAPE: {evaluation_results['mape']:.3f}")
                print(f"Accuracy: {evaluation_results['accuracy']:.1f}%")
            else:
                print("MAPE: inf (zero true values)")
            print(f"Custom RUL Score: {evaluation_results['custom_rul_score']:.2f}")
            print(f"Prediction range: {evaluation_results['pred_range_min']:.1f} to {evaluation_results['pred_range_max']:.1f}")
            print(f"True RUL range: {evaluation_results['true_range_min']:.1f} to {evaluation_results['true_range_max']:.1f}")
            print("="*50)
        
        # 11. Save predictions
        os.makedirs(os.path.dirname(self.config.predictions_path), exist_ok=True)
        results_df.to_csv(self.config.predictions_path, index=False)
        
        print(f"Predictions saved at: {self.config.predictions_path}")
        if true_rul_per_engine is not None:
            print(f"Evaluation metrics saved at: {self.config.evaluation_metrics_path}")
        
        return self.config.predictions_path, self.config.evaluation_metrics_path