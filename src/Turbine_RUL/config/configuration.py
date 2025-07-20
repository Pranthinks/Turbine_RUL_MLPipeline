import yaml
from src.Turbine_RUL.entity import DataIngestionConfig

class ConfigurationManager:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    def get_data_ingestion_config(self):
        config = self.config['data_ingestion']
        return DataIngestionConfig(
            raw_data_path=config['raw_data_path'],
            train_data_path=config['train_data_path'],
            test_data_path=config['test_data_path'],
            test_size=config['test_size']
        )