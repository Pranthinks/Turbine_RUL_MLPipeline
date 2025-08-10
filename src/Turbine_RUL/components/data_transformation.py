import os
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from src.Turbine_RUL.config.configuration import ConfigurationManager
from src.Turbine_RUL.components.preprocessor import ColumnDropper, LowVarianceFeaturesRemover, ScalePerEngine, SENSOR_COLUMNS
from src.Turbine_RUL.monitoring.enhanced_metrics import TurbineMLOpsMetrics, monitor_pipeline_stage


class DataTransformation:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_data_transformation_config()
        self.metrics = TurbineMLOpsMetrics()

    
    def get_preprocessor(self):
        """Create complete preprocessing pipeline including column dropping"""
        preprocess_pipe = Pipeline([
            ('drop-columns', ColumnDropper(columns_to_drop=['rul', 'data_type', 'dataset', 'created_at'])),
            ('drop-low-variance', LowVarianceFeaturesRemover(threshold=0)),
            ('scale-per-engine', ScalePerEngine(n_first_cycles=10, sensors_columns=SENSOR_COLUMNS))  # Changed from 15 to 10
        ])
        return preprocess_pipe
    

    @monitor_pipeline_stage('data_transformation')
    def initiate_data_transformation(self):
        """Main data transformation process - only process training data"""
        # Read only training data from previous stage (RAW data with all columns)
        train_data = pd.read_csv(self.config.train_data_path)
        original_shape = train_data.shape
        # Create complete preprocessor (includes column dropping)
        preprocessor = self.get_preprocessor()
        train_preprocessed = preprocessor.fit_transform(train_data)
        transformed_shape = train_preprocessed.shape
        
        # ENHANCED MONITORING - Record transformation metrics
        dropped_features = {
            'low_variance': original_shape[1] - transformed_shape[1],
            'manual_drops': 4  # rul, data_type, dataset, created_at
        }
        self.metrics.record_transformation_metrics(original_shape, transformed_shape, dropped_features)
        
        
        # Create directories
        os.makedirs(os.path.dirname(self.config.train_preprocessed_path), exist_ok=True)
        
        # Save transformed training data
        train_preprocessed.to_csv(self.config.train_preprocessed_path, index=False)
        
        # Save complete preprocessor for future use
        with open(self.config.preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        
        print(f"Training data transformed: {train_preprocessed.shape}")
        print(f"Preprocessor saved at: {self.config.preprocessor_path}")
        
        return self.config.train_preprocessed_path, self.config.preprocessor_path