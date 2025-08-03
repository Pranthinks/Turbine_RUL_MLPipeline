import yaml
from src.Turbine_RUL.entity import DataIngestionConfig, DataTransformationConfig, FeatureEngineeringConfig, ModelTrainingConfig, ModelPredictionConfig

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

    def get_model_training_config(self):
        config = self.config['model_training']
        return ModelTrainingConfig(
            engineered_features_path=config['engineered_features_path'],
            model_path=config['model_path'],
            selected_features_path=config['selected_features_path'],
            feature_importance_path=config['feature_importance_path'],
            cv_results_path=config['cv_results_path'],
            metrics_path=config['metrics_path']
        )
    
    def get_model_prediction_config(self):
        config = self.config['model_prediction']
        return ModelPredictionConfig(
            test_data_path=config['test_data_path'],
            preprocessor_path=config['preprocessor_path'],
            long_term_pipeline_path=config['long_term_pipeline_path'],
            short_term_pipeline_path=config['short_term_pipeline_path'],
            model_path=config['model_path'],
            selected_features_path=config['selected_features_path'],
            predictions_path=config['predictions_path'],
            evaluation_rul_path=config['evaluation_rul_path'],
            evaluation_metrics_path=config['evaluation_metrics_path'],
            evaluation_plots_path=config['evaluation_plots_path']
        )