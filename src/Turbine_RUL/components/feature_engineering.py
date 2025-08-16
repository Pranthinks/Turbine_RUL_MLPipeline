import os
import pandas as pd
import pickle
import warnings
import time
import gc
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from src.Turbine_RUL.monitoring.enhanced_metrics import TurbineMLOpsMetrics, monitor_pipeline_stage
from src.Turbine_RUL.config.configuration import ConfigurationManager
from src.Turbine_RUL.utils.common import calculate_RUL

# Suppress pandas FutureWarnings from TSFresh
warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply operated on the grouping columns")
warnings.filterwarnings("ignore", category=FutureWarning)

# TSFresh feature calculations
tsfresh_calc = {
    'mean_change': None,
    'mean': None,
    'standard_deviation': None,
    'root_mean_square': None,
    'last_location_of_maximum': None,
    'first_location_of_maximum': None,
    'last_location_of_minimum': None,
    'first_location_of_minimum': None,
    'maximum': None,
    'minimum': None,
    'time_reversal_asymmetry_statistic': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
    'c3': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
    'cid_ce': [{'normalize': True}, {'normalize': False}],
    'autocorrelation': [{'lag': 0}, {'lag': 1}, {'lag': 2}, {'lag': 3}],
    'partial_autocorrelation': [{'lag': 0}, {'lag': 1}, {'lag': 2}, {'lag': 3}],
    'linear_trend': [{'attr': 'intercept'}, {'attr': 'slope'}, {'attr': 'stderr'}],
    'augmented_dickey_fuller': [{'attr': 'teststat'}, {'attr': 'pvalue'}, {'attr': 'usedlag'}],
    'fft_coefficient': [{'coeff': i, 'attr': 'abs'} for i in range(11)],
    'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance'},
                      {'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
}

class RollTimeSeries(BaseEstimator, TransformerMixin):
    def __init__(self, min_timeshift, max_timeshift, rolling_direction):
        self.min_timeshift = min_timeshift
        self.max_timeshift = max_timeshift
        self.rolling_direction = rolling_direction

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        _start = datetime.now()
        print('Start Rolling TS')
        X_t = roll_time_series(
            X, column_id='unit_id', column_sort='time_cycles',
            rolling_direction=self.rolling_direction,
            min_timeshift=self.min_timeshift,
            max_timeshift=self.max_timeshift,
            n_jobs=1)  # Disable multiprocessing
        print(f'Done Rolling TS in {datetime.now() - _start}')
        return X_t

class TSFreshFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, calc=tsfresh_calc, batch_size=50):  # ADD BATCH SIZE
        self.calc = calc
        self.batch_size = batch_size  # Process 50 units at a time

    def _clean_features(self, X):
        old_shape = X.shape
        X_t = X.T.drop_duplicates().T
        print(f'Dropped {old_shape[1] - X_t.shape[1]} duplicate features')

        old_shape = X_t.shape
        X_t = X_t.dropna(axis=1)
        print(f'Dropped {old_shape[1] - X_t.shape[1]} features with NA values')
        
        return X_t

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        _start = datetime.now()
        print('Start Extracting Features')
        
        # Get unique unit IDs for batching
        unique_ids = X['id'].unique()
        total_units = len(unique_ids)
        print(f'Processing {total_units} units in batches of {self.batch_size}')
        
        all_features = []
        
        # Process in batches
        for i in range(0, total_units, self.batch_size):
            batch_ids = unique_ids[i:i + self.batch_size]
            batch_data = X[X['id'].isin(batch_ids)]
            
            print(f'Processing batch {i//self.batch_size + 1}/{(total_units + self.batch_size - 1)//self.batch_size} '
                  f'(units {i+1}-{min(i+self.batch_size, total_units)})')
            
            # Extract features for this batch
            batch_features = extract_features(
                batch_data[['id', 'time_cycles'] +
                          batch_data.columns[batch_data.columns.str.startswith('sensor')].tolist()],
                column_id='id',
                column_sort='time_cycles',
                default_fc_parameters=self.calc
            )
            
            all_features.append(batch_features)
            
            # CRITICAL: Free memory after each batch
            del batch_data, batch_features
            gc.collect()  # Force garbage collection
            
            print(f'Batch {i//self.batch_size + 1} completed. Memory freed.')
        
        # Combine all batches
        print('Combining all feature batches...')
        X_t = pd.concat(all_features, axis=0)
        
        # Clean up batch list
        del all_features
        gc.collect()
            
        print(f'Done Extracting Features in {datetime.now() - _start}')
        X_t = self._clean_features(X_t)
        return X_t

class CustomPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        assert 'unit_id' not in X.columns, "columns should be only features"
        self.ftr_columns = X.columns

        self.scaler = StandardScaler()
        X_sc = self.scaler.fit_transform(X[self.ftr_columns].values)

        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit_transform(X_sc)
        
        # Free memory
        del X_sc
        gc.collect()
        
        return self

    def transform(self, X):
        # Store original index to preserve it
        original_index = X.index
        
        X_sc = self.scaler.transform(X[self.ftr_columns].values)
        X_pca = self.pca.transform(X_sc)
        
        # Create DataFrame with original index preserved
        result = pd.DataFrame(X_pca, index=original_index)
        
        # Free memory
        del X_sc, X_pca
        gc.collect()
        
        return result

class TSFreshFeaturesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, fdr_level=0.001, upper_threshold=135):
        self.fdr_level = fdr_level
        self.upper_threshold = upper_threshold

    def fit(self, X, y=None):
        # Simplified approach like Google Colab
        index_df = X.index.to_frame().reset_index(drop=True)
        
        # Handle different index structures
        if len(index_df.columns) == 2:
            index_df.columns = ['unit_id', 'time_cycles']
        else:
            print("Warning: Unexpected index structure")
            # Use all features as fallback
            self.selected_ftr = X.columns
            return self
        
        # Calculate RUL
        rul = calculate_RUL(index_df, upper_threshold=self.upper_threshold)
        
        # Feature selection
        try:
            X_t = select_features(X, rul, fdr_level=self.fdr_level)
            self.selected_ftr = X_t.columns
            print(f'Selected {len(self.selected_ftr)} out of {X.shape[1]} features: '
                  f'{self.selected_ftr.to_list()}')
            
            # Free memory
            del X_t
            gc.collect()
            
        except Exception as e:
            print(f"Feature selection failed: {e}. Using all features.")
            self.selected_ftr = X.columns
            
        return self

    def transform(self, X):
        return X[self.selected_ftr]

class FeatureEngineering:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_feature_engineering_config()
        self.metrics = TurbineMLOpsMetrics()

    def create_long_term_pipeline(self):
        """Create pipeline for long-term features (19 time steps) with memory optimization"""
        return Pipeline([
            ('roll-time-series', RollTimeSeries(min_timeshift=19, max_timeshift=19, rolling_direction=1)),
            ('extract-tsfresh-features', TSFreshFeaturesExtractor(calc=tsfresh_calc, batch_size=30)),  # SMALLER BATCHES
            ('PCA', CustomPCA(n_components=20)),
            ('features-selection', TSFreshFeaturesSelector(fdr_level=0.001)),
        ])

    def create_short_term_pipeline(self):
        """Create pipeline for short-term features (4 time steps) with memory optimization"""
        return Pipeline([
            ('roll-time-series', RollTimeSeries(min_timeshift=4, max_timeshift=4, rolling_direction=1)),
            ('extract-tsfresh-features', TSFreshFeaturesExtractor(calc={'mean': None}, batch_size=50)),  # LARGER BATCHES FOR SIMPLE FEATURES
            ('features-selection', TSFreshFeaturesSelector(fdr_level=0.0002)),
        ])

    @monitor_pipeline_stage('feature_engineering')
    def initiate_feature_engineering(self):
        """Main feature engineering process with memory optimization"""
        print("Starting Feature Engineering...")
        
        # Load preprocessed data from previous stage
        train_data = pd.read_csv(self.config.train_preprocessed_path)
        print(f"Loaded training data shape: {train_data.shape}")
        
        # Create feature engineering pipelines
        features_long_h_pipe = self.create_long_term_pipeline()
        features_short_h_pipe = self.create_short_term_pipeline()
        
        # Extract long-term features WITH TIMING AND MEMORY MONITORING
        print("Extracting long-term features...")
        start_long = time.time()
        train_long_h_ftrs = features_long_h_pipe.fit_transform(train_data)
        long_duration = time.time() - start_long
        print(f"Long-term features completed in {long_duration:.2f}s, shape: {train_long_h_ftrs.shape}")
        
        # Force garbage collection between stages
        gc.collect()
        
        # Extract short-term features WITH TIMING AND MEMORY MONITORING
        print("Extracting short-term features...")
        start_short = time.time()
        train_short_h_ftrs = features_short_h_pipe.fit_transform(train_data)
        short_duration = time.time() - start_short
        print(f"Short-term features completed in {short_duration:.2f}s, shape: {train_short_h_ftrs.shape}")
        
        # Free the original data since we don't need it anymore
        del train_data
        gc.collect()
        
        # Merge both feature sets
        print("Merging feature sets...")
        train_ftrs = train_long_h_ftrs.merge(
            train_short_h_ftrs, 
            how='inner',
            right_index=True, 
            left_index=True
        )
        
        # Free intermediate feature sets
        del train_long_h_ftrs, train_short_h_ftrs
        gc.collect()

        # ENHANCED MONITORING - Record feature engineering metrics
        extraction_times = {
            'long_term': long_duration,
            'short_term': short_duration
        }
        
        feature_counts = {
            'long_term': train_ftrs.shape[1] - (train_short_h_ftrs.shape[1] if 'train_short_h_ftrs' in locals() else 0),
            'short_term': 0,  # Will be updated if available
            'final': train_ftrs.shape[1]
        }
        
        # Calculate selection ratio
        original_features = 21  # Approximate sensor count
        final_features = train_ftrs.shape[1]
        selection_ratio = final_features / original_features if original_features > 0 else 1.0
        
        # Record monitoring metrics
        self.metrics.record_feature_engineering_metrics(extraction_times, feature_counts, selection_ratio)
        
        # Set proper index names if it's a MultiIndex
        if hasattr(train_ftrs.index, 'names') and len(train_ftrs.index.names) == 2:
            train_ftrs.index = train_ftrs.index.set_names(['unit_id', 'time_cycles'])
        
        print(f'Final training features shape: {train_ftrs.shape}')
        
        # Create directories
        os.makedirs(os.path.dirname(self.config.engineered_features_path), exist_ok=True)
        
        # Save engineered features
        print("Saving engineered features...")
        train_ftrs.to_csv(self.config.engineered_features_path)
        
        # Save both pipelines for future use
        with open(self.config.long_term_pipeline_path, 'wb') as f:
            pickle.dump(features_long_h_pipe, f)
            
        with open(self.config.short_term_pipeline_path, 'wb') as f:
            pickle.dump(features_short_h_pipe, f)
        
        # Final cleanup
        del train_ftrs, features_long_h_pipe, features_short_h_pipe
        gc.collect()
        
        print(f"Feature engineering completed!")
        print(f"Engineered features saved at: {self.config.engineered_features_path}")
        print(f"Long-term pipeline saved at: {self.config.long_term_pipeline_path}")
        print(f"Short-term pipeline saved at: {self.config.short_term_pipeline_path}")
        
        return (self.config.engineered_features_path, 
                self.config.long_term_pipeline_path, 
                self.config.short_term_pipeline_path)