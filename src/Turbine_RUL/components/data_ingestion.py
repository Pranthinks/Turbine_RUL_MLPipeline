import os
import yaml
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from src.Turbine_RUL.config.configuration import ConfigurationManager

class DataIngestion:
    def __init__(self):
        # Get configs inside the class
        config_manager = ConfigurationManager()
        self.config = config_manager.get_data_ingestion_config()
        
        # Load DB config
        with open("config/config.yaml") as f:
            self.db_config = yaml.safe_load(f)['database']
    
    def get_data_from_postgres(self):
        """Extract data from PostgreSQL"""
        conn_str = f"postgresql://{self.db_config['username']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        
        query = f"SELECT * FROM {self.db_config['table_name']}"
        df = pd.read_sql(query, conn_str)
        return df
    
    def save_data(self, df):
        """Save raw data and split into train/test"""
        # Create directories
        os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
        
        # Save raw data
        df.to_csv(self.config.raw_data_path, index=False)
        
        # Split data
        train, test = train_test_split(df, test_size=self.config.test_size, random_state=42)
        
        # Save train/test
        train.to_csv(self.config.train_data_path, index=False)
        test.to_csv(self.config.test_data_path, index=False)
        
        print(f"Data saved: Raw({df.shape}), Train({train.shape}), Test({test.shape})")
    
    def initiate_data_ingestion(self):
        """Main data ingestion process"""
        df = self.get_data_from_postgres()
        self.save_data(df)
        return self.config.train_data_path, self.config.test_data_path