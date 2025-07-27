import yaml
from src.Turbine_RUL.entity import DataIngestionConfig, DataTransformationConfig, FeatureEngineeringConfig

class ConfigurationManager:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def get_data_ingestion_config(self):
        config = self.config['data_ingestion']
        return DataIngestionConfig(
            train_data_path=config['train_data_path'],
            test_data_path=config['test_data_path']
        )
    
    def get_data_transformation_config(self):
        config = self.config['data_transformation']
        return DataTransformationConfig(
            train_data_path=config['train_data_path'],
            train_preprocessed_path=config['train_preprocessed_path'],
            preprocessor_path=config['preprocessor_path']
        )
    def get_feature_engineering_config(self):
        config = self.config['feature_engineering']
        return FeatureEngineeringConfig(
            train_preprocessed_path=config['train_preprocessed_path'],
            engineered_features_path=config['engineered_features_path'],
            long_term_pipeline_path=config['long_term_pipeline_path'],
            short_term_pipeline_path=config['short_term_pipeline_path']
    )