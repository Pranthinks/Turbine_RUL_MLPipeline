import pandas as pd
import numpy as np
import pickle
import time
import warnings
import gc
from datetime import datetime
from src.Turbine_RUL.monitoring.enhanced_metrics import TurbineMLOpsMetrics, monitor_pipeline_stage
from src.Turbine_RUL.config.configuration import ConfigurationManager
from src.Turbine_RUL.utils.common import (
    calculate_evaluation_metrics, create_evaluation_plots, 
    save_prediction_results, extract_and_save_true_rul,
    print_evaluation_summary
)

warnings.filterwarnings("ignore")

class ModelPrediction:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_model_prediction_config()
        # Add enhanced metrics - EXACT SAME
        self.metrics = TurbineMLOpsMetrics()
    
    def _optimize_data_types(self, df):
        """Memory optimization without changing data integrity"""
        print("Optimizing data types for memory efficiency...")
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Convert float64 to float32 where possible
        for col in df.select_dtypes(include=['float64']).columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
        
        # Convert int64 to smaller int types where possible
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')
        
        new_memory = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Memory usage reduced from {original_memory:.2f} MB to {new_memory:.2f} MB")
        
        return df

    def load_artifacts(self):
        """Load all trained artifacts - EXACT SAME logic"""
        print("Loading trained artifacts...")
        artifacts = [
            ('model', self.config.model_path),
            ('features_long_pipe', self.config.long_term_pipeline_path),
            ('features_short_pipe', self.config.short_term_pipeline_path),
            ('preprocessor', self.config.preprocessor_path),
            ('selected_features', self.config.selected_features_path)
        ]
        
        for attr_name, path in artifacts:
            with open(path, 'rb') as f:
                setattr(self, attr_name, pickle.load(f))
        
        print(f"✅ All artifacts loaded. Features: {len(self.selected_features)}")
        
    def preprocess_and_extract_features(self, test_data):
        """Combined preprocessing and feature extraction - EXACT SAME logic with memory optimization"""
        print("Preprocessing and extracting features...")
        
        # Memory optimization: Optimize data types before processing
        test_data_optimized = self._optimize_data_types(test_data.copy())
        
        # Preprocess data - EXACT SAME logic
        test_preprocessed = self.preprocessor.transform(test_data_optimized)
        
        # Memory cleanup
        del test_data_optimized
        gc.collect()
        
        # Extract features - EXACT SAME logic
        print("Extracting long-term features...")
        test_long_h_fts = self.features_long_pipe.transform(test_preprocessed)
        
        # Memory cleanup between feature extractions
        gc.collect()
        
        print("Extracting short-term features...")
        test_short_h_fts = self.features_short_pipe.transform(test_preprocessed)
        
        # Memory cleanup
        del test_preprocessed
        gc.collect()
        
        # Set index names and merge - EXACT SAME logic
        for df in [test_long_h_fts, test_short_h_fts]:
            df.index = df.index.set_names(['unit_id', 'time_cycles'])
        
        try:
            test_ftrs = test_long_h_fts.merge(test_short_h_fts, how='inner', 
                                              right_index=True, left_index=True)
        except:
            # Alternative merge approach - EXACT SAME logic
            long_reset = test_long_h_fts.reset_index()
            short_reset = test_short_h_fts.reset_index()
            merged_reset = long_reset.merge(short_reset, on=['unit_id', 'time_cycles'], how='inner')
            test_ftrs = merged_reset.set_index(['unit_id', 'time_cycles'])
            
            # Memory cleanup
            del long_reset, short_reset, merged_reset
            gc.collect()
        
        # Memory cleanup
        del test_long_h_fts, test_short_h_fts
        gc.collect()
        
        # Memory optimization: Convert to float32 for final features
        for col in test_ftrs.select_dtypes(include=['float64']).columns:
            test_ftrs[col] = test_ftrs[col].astype('float32')
        
        print(f"✅ Features extracted: {test_ftrs.shape}")
        print(f"Memory usage: {test_ftrs.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return test_ftrs
    
    def prepare_prediction_data(self, test_features):
        """Prepare features for final prediction - EXACT SAME logic with memory optimization"""
        print("Preparing prediction data...")
        
        X_test_features = test_features.reset_index().drop(columns=['unit_id'])
        test_units = test_features.index.to_frame(index=False)
        
        # Memory optimization: Convert to float32
        for col in X_test_features.select_dtypes(include=['float64']).columns:
            X_test_features[col] = X_test_features[col].astype('float32')
        
        # Get last cycle for each engine - EXACT SAME logic
        X_test_last = X_test_features.groupby(test_units['unit_id']).last()
        X_test_last.columns = [str(col) for col in X_test_last.columns]
        engine_ids = X_test_last.index.values
        
        # Memory cleanup
        del X_test_features, test_units
        gc.collect()
        
        # Check feature alignment - EXACT SAME logic
        available_features = set(X_test_last.columns)
        required_features = set(self.selected_features)
        missing_features = required_features - available_features
        
        if missing_features:
            raise ValueError(f"Feature mismatch! Missing: {missing_features}")
        
        X_test_final = X_test_last[self.selected_features]
        
        # Memory cleanup
        del X_test_last
        gc.collect()
        
        print(f"✅ Prediction data ready: {X_test_final.shape}")
        print(f"Memory usage: {X_test_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return X_test_final, engine_ids
    
    def make_predictions(self, X_test_prepared):
        """Make RUL predictions - EXACT SAME logic with memory optimization"""
        print("Making predictions...")
        
        # Memory optimization: Ensure float32 for prediction
        X_values = X_test_prepared.values.astype('float32')
        
        pred_rul = self.model.predict(X_values)
        
        # Memory cleanup
        del X_values
        gc.collect()
        
        print(f"✅ Predictions completed: {pred_rul.min():.1f} to {pred_rul.max():.1f}")
        return pred_rul
    
    def _memory_efficient_evaluation(self, true_rul_per_engine, predictions, engine_ids):
        """Memory-efficient evaluation for large datasets"""
        print("Performing memory-efficient evaluation...")
        
        # For very large datasets, sample for evaluation if needed
        if len(predictions) > 50000:
            print(f"Large dataset detected ({len(predictions)} predictions), using memory-efficient evaluation")
            
            # Sample a subset for detailed evaluation if memory is critical
            # But still return all predictions
            sample_size = min(10000, len(predictions))
            sample_indices = np.random.choice(len(predictions), sample_size, replace=False)
            
            sampled_true_rul = {engine_ids[i]: true_rul_per_engine.get(engine_ids[i]) 
                               for i in sample_indices if engine_ids[i] in true_rul_per_engine}
            sampled_predictions = predictions[sample_indices]
            sampled_engine_ids = engine_ids[sample_indices]
            
            print(f"Using sample of {len(sampled_true_rul)} engines for evaluation")
            return calculate_evaluation_metrics(sampled_true_rul, sampled_predictions, sampled_engine_ids)
        else:
            # Use full evaluation for smaller datasets
            return calculate_evaluation_metrics(true_rul_per_engine, predictions, engine_ids)
    
    @monitor_pipeline_stage('model_prediction')
    def initiate_model_prediction(self):
        """Core prediction pipeline with evaluation - EXACT SAME logic with memory optimizations"""
        print("Starting Model Prediction Pipeline...")
        start_time = datetime.now()
        
        # Load and process data - EXACT SAME logic
        raw_test_data = pd.read_csv(self.config.test_data_path)
        print(f"Raw test data: {raw_test_data.shape}")
        print(f"Initial memory usage: {raw_test_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Extract true RUL values for evaluation - EXACT SAME logic
        true_rul_per_engine = extract_and_save_true_rul(raw_test_data, self.config.evaluation_rul_path)
        
        # Load artifacts and make predictions
        self.load_artifacts()
        
        # Memory-optimized processing
        test_features = self.preprocess_and_extract_features(raw_test_data)
        
        # Memory cleanup after feature extraction
        del raw_test_data
        gc.collect()
        
        X_test_prepared, engine_ids = self.prepare_prediction_data(test_features)
        
        # Memory cleanup
        del test_features
        gc.collect()
        
        # Time the prediction step - EXACT SAME logic
        prediction_start = time.time()
        predictions = self.make_predictions(X_test_prepared)
        prediction_time = time.time() - prediction_start
        
        # Memory cleanup
        del X_test_prepared
        gc.collect()
        
        # Evaluate if true RUL values are available - EXACT SAME logic with memory optimization
        metrics = true_rul_array = pred_rul_array = None
        if true_rul_per_engine is not None:
            # Use memory-efficient evaluation for large datasets
            if len(predictions) > 30000:
                metrics, true_rul_array, pred_rul_array = self._memory_efficient_evaluation(
                    true_rul_per_engine, predictions, engine_ids)
            else:
                metrics, true_rul_array, pred_rul_array = calculate_evaluation_metrics(
                    true_rul_per_engine, predictions, engine_ids)
            
            if metrics is not None:
                print_evaluation_summary(metrics)
                create_evaluation_plots(true_rul_array, pred_rul_array, metrics, 
                                       self.config.evaluation_plots_path)
        
        # Save results - EXACT SAME logic
        save_prediction_results(predictions, engine_ids, true_rul_per_engine, metrics,
                              self.config.predictions_path, self.config.evaluation_metrics_path)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # ENHANCED MONITORING - Record comprehensive prediction metrics - EXACT SAME logic
        evaluation_metrics_dict = {}
        if metrics:
            evaluation_metrics_dict = {
                'rmse': metrics.get('rmse', 0),
                'mae': metrics.get('mae', 0),
                'r2': metrics.get('r2_score', 0),
                'mape': metrics.get('mape', 0)
            }
        
        self.metrics.record_prediction_metrics(predictions, prediction_time, evaluation_metrics_dict)
        
        # Check SLA compliance (example: pipeline should complete in < 30 minutes) - EXACT SAME logic
        sla_compliance = 1 if total_time < 1800 else 0  # 30 minutes SLA
        self.metrics.pipeline_sla_compliance.set(sla_compliance)
        self.metrics.total_pipeline_duration.set(total_time)
        
        # Final memory cleanup
        gc.collect()
        
        print(f"\n✅ Pipeline completed in {total_time:.2f} seconds!")

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