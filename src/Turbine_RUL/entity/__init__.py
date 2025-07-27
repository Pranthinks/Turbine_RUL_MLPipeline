from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str
    test_data_path: str

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