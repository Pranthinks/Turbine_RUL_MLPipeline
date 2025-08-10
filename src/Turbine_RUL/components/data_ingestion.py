import os
import yaml
import pandas as pd
import psycopg2
from src.Turbine_RUL.config.configuration import ConfigurationManager
from src.Turbine_RUL.monitoring.docker_metrics import DockerMLOpsMetrics, monitor_pipeline_stage

class DataIngestion:
    def __init__(self):
        # Get configs inside the class
        config_manager = ConfigurationManager()
        self.config = config_manager.get_data_ingestion_config()
        
        # Load DB config
        with open("config/config.yaml") as f:
            self.db_config = yaml.safe_load(f)['database']
        self.metrics = DockerMLOpsMetrics()

    def get_data_from_postgres(self, table_name):
        """Extract data from PostgreSQL for a specific table"""
        conn_str = f"postgresql://{self.db_config['username']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn_str)
        return df

    def save_data(self, train_df, test_df):
        """Save train and test data directly"""
        # Create directories
        os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
        
        # Save train and test data
        train_df.to_csv(self.config.train_data_path, index=False)
        test_df.to_csv(self.config.test_data_path, index=False)
        
        print(f"Data saved: Train({train_df.shape}), Test({test_df.shape})")

    
    @monitor_pipeline_stage('data_ingestion')
    def initiate_data_ingestion(self):
        """Main data ingestion process"""
        # Get data from both tables
        train_df = self.get_data_from_postgres("train_turbofan_data")
        test_df = self.get_data_from_postgres("test_turbofan_data")

        # ADD DATA QUALITY METRICS
        if self.metrics.monitoring_enabled:
            train_quality = 1 - (train_df.isnull().sum().sum() / (train_df.shape[0] * train_df.shape[1]))
            test_quality = 1 - (test_df.isnull().sum().sum() / (test_df.shape[0] * test_df.shape[1]))
            
            self.metrics.data_quality.labels(dataset_type='train').set(train_quality)
            self.metrics.data_quality.labels(dataset_type='test').set(test_quality)
        
        
        # Save both datasets
        self.save_data(train_df, test_df)
        
        return self.config.train_data_path, self.config.test_data_path