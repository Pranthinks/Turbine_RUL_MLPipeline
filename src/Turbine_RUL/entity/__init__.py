from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str
    test_data_path: str

@dataclass
class DriftDetectionConfig:
    root_dir: str
    train_data_path: str
    test_data_path: str
    reference_data_path: str
    reference_stats_path: str
    drift_report_path: str

@dataclass
class DataTransformationConfig:
    train_data_path: str                         
    train_preprocessed_path: str                          
    preprocessor_path: str

@dataclass
class FeatureEngineeringConfig:
    train_preprocessed_path: str
    engineered_features_path: str
    long_term_pipeline_path: str
    short_term_pipeline_path: str

@dataclass
class ModelTrainingConfig:
    engineered_features_path: str
    model_path: str
    selected_features_path: str
    feature_importance_path: str
    cv_results_path: str
    metrics_path: str

@dataclass
class ModelPredictionConfig:
    test_data_path: str
    preprocessor_path: str
    long_term_pipeline_path: str
    short_term_pipeline_path: str
    model_path: str
    selected_features_path: str
    predictions_path: str
    evaluation_rul_path: str
    evaluation_metrics_path: str
    evaluation_plots_path: str