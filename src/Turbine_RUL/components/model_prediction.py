import pandas as pd
import numpy as np
import pickle
import time
import warnings
import gc
import psutil
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
        # Add enhanced metrics
        self.metrics = TurbineMLOpsMetrics()

    def log_memory_usage(self, stage_name):
        """Log current memory usage"""
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"[{stage_name}] Memory usage: {memory_mb:.1f} MB")
        return memory_mb
        
    def load_artifacts(self):
        """Load all trained artifacts"""
        print("Loading trained artifacts...")
        self.log_memory_usage("Load Start")
        
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
        
        self.log_memory_usage("Load Complete")
        print(f"✅ All artifacts loaded. Features: {len(self.selected_features)}")
        
    def preprocess_and_extract_features(self, test_data):
        """Combined preprocessing and feature extraction - MEMORY OPTIMIZED"""
        print("Preprocessing and extracting features...")
        self.log_memory_usage("Feature Extraction Start")
        
        # Preprocess data
        test_preprocessed = self.preprocessor.transform(test_data)
        
        # Free original test data from memory
        del test_data
        gc.collect()
        self.log_memory_usage("After Preprocessing")
        
        # Extract long-term features
        test_long_h_fts = self.features_long_pipe.transform(test_preprocessed)
        gc.collect()  # Clean up intermediate objects from pipeline
        self.log_memory_usage("After Long-term Features")
        
        # Extract short-term features
        test_short_h_fts = self.features_short_pipe.transform(test_preprocessed)
        
        # Free preprocessed data since we don't need it anymore
        del test_preprocessed
        gc.collect()
        self.log_memory_usage("After Short-term Features")
        
        # Set index names and merge
        for df in [test_long_h_fts, test_short_h_fts]:
            df.index = df.index.set_names(['unit_id', 'time_cycles'])
        
        try:
            test_ftrs = test_long_h_fts.merge(test_short_h_fts, how='inner', 
                                              right_index=True, left_index=True)
        except:
            # Alternative merge approach
            long_reset = test_long_h_fts.reset_index()
            short_reset = test_short_h_fts.reset_index()
            merged_reset = long_reset.merge(short_reset, on=['unit_id', 'time_cycles'], how='inner')
            test_ftrs = merged_reset.set_index(['unit_id', 'time_cycles'])
            
            # Clean up intermediate DataFrames
            del long_reset, short_reset, merged_reset
        
        # Free the individual feature sets
        del test_long_h_fts, test_short_h_fts
        gc.collect()
        self.log_memory_usage("After Feature Merge")
        
        print(f"✅ Features extracted: {test_ftrs.shape}")
        return test_ftrs
    
    def prepare_prediction_data(self, test_features):
        """Prepare features for final prediction - MEMORY OPTIMIZED"""
        print("Preparing prediction data...")
        self.log_memory_usage("Prep Start")
        
        X_test_features = test_features.reset_index().drop(columns=['unit_id'])
        test_units = test_features.index.to_frame(index=False)
        
        # Free original test_features since we have what we need
        del test_features
        gc.collect()
        
        # Get last cycle for each engine
        X_test_last = X_test_features.groupby(test_units['unit_id']).last()
        X_test_last.columns = [str(col) for col in X_test_last.columns]
        engine_ids = X_test_last.index.values
        
        # Free intermediate DataFrames
        del X_test_features, test_units
        gc.collect()
        
        # Check feature alignment
        available_features = set(X_test_last.columns)
        required_features = set(self.selected_features)
        missing_features = required_features - available_features
        
        if missing_features:
            raise ValueError(f"Feature mismatch! Missing: {missing_features}")
        
        X_test_final = X_test_last[self.selected_features]
        
        # Free the larger DataFrame
        del X_test_last
        gc.collect()
        self.log_memory_usage("Prep Complete")
        
        print(f"✅ Prediction data ready: {X_test_final.shape}")
        return X_test_final, engine_ids
    
    def make_predictions(self, X_test_prepared):
        """Make RUL predictions"""
        self.log_memory_usage("Prediction Start")
        pred_rul = self.model.predict(X_test_prepared.values)
        self.log_memory_usage("Prediction Complete")
        print(f"✅ Predictions completed: {pred_rul.min():.1f} to {pred_rul.max():.1f}")
        return pred_rul
    
    @monitor_pipeline_stage('model_prediction')
    def initiate_model_prediction(self):
        """Core prediction pipeline with evaluation - MEMORY OPTIMIZED"""
        print("Starting Model Prediction Pipeline...")
        start_time = datetime.now()
        self.log_memory_usage("Pipeline Start")
        
        # Load and process data
        raw_test_data = pd.read_csv(self.config.test_data_path)
        print(f"Raw test data: {raw_test_data.shape}")
        self.log_memory_usage("Data Loaded")
        
        # Extract true RUL values for evaluation
        true_rul_per_engine = extract_and_save_true_rul(raw_test_data, self.config.evaluation_rul_path)
        
        # Load artifacts and make predictions
        self.load_artifacts()
        
        # Process in stages with memory cleanup
        test_features = self.preprocess_and_extract_features(raw_test_data)  # raw_test_data freed inside
        X_test_prepared, engine_ids = self.prepare_prediction_data(test_features)  # test_features freed inside
        
        # Time the prediction step
        prediction_start = time.time()
        predictions = self.make_predictions(X_test_prepared)
        prediction_time = time.time() - prediction_start
        
        # Free prediction data
        del X_test_prepared
        gc.collect()
        self.log_memory_usage("After Predictions")
        
        # Evaluate if true RUL values are available
        metrics = true_rul_array = pred_rul_array = None
        if true_rul_per_engine is not None:
            metrics, true_rul_array, pred_rul_array = calculate_evaluation_metrics(
                true_rul_per_engine, predictions, engine_ids)
            
            if metrics is not None:
                print_evaluation_summary(metrics)
                create_evaluation_plots(true_rul_array, pred_rul_array, metrics, 
                                       self.config.evaluation_plots_path)
        
        # Save results
        save_prediction_results(predictions, engine_ids, true_rul_per_engine, metrics,
                              self.config.predictions_path, self.config.evaluation_metrics_path)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # ENHANCED MONITORING - Record comprehensive prediction metrics
        evaluation_metrics_dict = {}
        if metrics:
            evaluation_metrics_dict = {
                'rmse': metrics.get('rmse', 0),
                'mae': metrics.get('mae', 0),
                'r2': metrics.get('r2_score', 0),
                'mape': metrics.get('mape', 0)
            }
        
        self.metrics.record_prediction_metrics(predictions, prediction_time, evaluation_metrics_dict)
        
        # Check SLA compliance (example: pipeline should complete in < 30 minutes)
        sla_compliance = 1 if total_time < 1800 else 0  # 30 minutes SLA
        self.metrics.pipeline_sla_compliance.set(sla_compliance)
        self.metrics.total_pipeline_duration.set(total_time)
        
        # Final cleanup
        del predictions, engine_ids, true_rul_per_engine
        if true_rul_array is not None:
            del true_rul_array, pred_rul_array
        gc.collect()
        self.log_memory_usage("Pipeline Complete")
        
        print(f"\n✅ Pipeline completed in {total_time:.2f} seconds!")

        return {
            'predictions_path': self.config.predictions_path,
            'evaluation_rul_path': self.config.evaluation_rul_path,
            'evaluation_metrics_path': self.config.evaluation_metrics_path,
            'evaluation_plots_path': self.config.evaluation_plots_path,
            'total_time': total_time,
            'n_predictions': len(predictions) if 'predictions' in locals() else 0,
            'metrics': metrics,
            'n_engines_evaluated': len(true_rul_array) if true_rul_array is not None else 0
        }