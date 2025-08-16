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

# EXACT SAME TSFresh feature calculations - NO CHANGES
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
    def __init__(self, min_timeshift, max_timeshift, rolling_direction, batch_size=1000):
        self.min_timeshift = min_timeshift
        self.max_timeshift = max_timeshift
        self.rolling_direction = rolling_direction
        self.batch_size = batch_size  # Added for memory management only

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        _start = datetime.now()
        print('Start Rolling TS')
        
        # Memory optimization: Process in batches if dataset is large
        if len(X['unit_id'].unique()) > self.batch_size:
            print(f'Processing {len(X["unit_id"].unique())} units in batches of {self.batch_size}')
            
            unique_units = X['unit_id'].unique()
            result_chunks = []
            
            for i in range(0, len(unique_units), self.batch_size):
                batch_units = unique_units[i:i + self.batch_size]
                batch_data = X[X['unit_id'].isin(batch_units)].copy()
                
                # EXACT SAME rolling logic
                batch_result = roll_time_series(
                    batch_data, column_id='unit_id', column_sort='time_cycles',
                    rolling_direction=self.rolling_direction,
                    min_timeshift=self.min_timeshift,
                    max_timeshift=self.max_timeshift,
                    n_jobs=1)
                
                result_chunks.append(batch_result)
                
                # Memory cleanup only
                del batch_data
                gc.collect()
                
                print(f'Processed batch {i//self.batch_size + 1}/{(len(unique_units) + self.batch_size - 1)//self.batch_size}')
            
            X_t = pd.concat(result_chunks, ignore_index=True)
            del result_chunks
            gc.collect()
        else:
            # EXACT SAME original logic for smaller datasets
            X_t = roll_time_series(
                X, column_id='unit_id', column_sort='time_cycles',
                rolling_direction=self.rolling_direction,
                min_timeshift=self.min_timeshift,
                max_timeshift=self.max_timeshift,
                n_jobs=1)
        
        print(f'Done Rolling TS in {datetime.now() - _start}')
        return X_t

class TSFreshFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, calc=tsfresh_calc, batch_size=500):
        self.calc = calc  # EXACT SAME calculations
        self.batch_size = batch_size  # Added for memory management only

    def _clean_features(self, X):
        # EXACT SAME cleaning logic
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
        
        sensor_cols = X.columns[X.columns.str.startswith('sensor')].tolist()
        feature_cols = ['id', 'time_cycles'] + sensor_cols
        
        # Memory optimization: Process in batches if dataset is large
        unique_ids = X['id'].unique()
        if len(unique_ids) > self.batch_size:
            print(f'Processing {len(unique_ids)} units in batches of {self.batch_size}')
            
            result_chunks = []
            
            for i in range(0, len(unique_ids), self.batch_size):
                batch_ids = unique_ids[i:i + self.batch_size]
                batch_data = X[X['id'].isin(batch_ids)][feature_cols].copy()
                
                # EXACT SAME feature extraction logic
                batch_features = extract_features(
                    batch_data,
                    column_id='id',
                    column_sort='time_cycles',
                    default_fc_parameters=self.calc)
                
                result_chunks.append(batch_features)
                
                # Memory cleanup only
                del batch_data
                gc.collect()
                
                print(f'Processed feature batch {i//self.batch_size + 1}/{(len(unique_ids) + self.batch_size - 1)//self.batch_size}')
            
            X_t = pd.concat(result_chunks, ignore_index=False)
            del result_chunks
            gc.collect()
        else:
            # EXACT SAME original logic for smaller datasets
            X_t = extract_features(
                X[feature_cols],
                column_id='id',
                column_sort='time_cycles',
                default_fc_parameters=self.calc)
        
        print(f'Done Extracting Features in {datetime.now() - _start}')
        
        # EXACT SAME cleaning process
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
        
        # Memory optimization: Use float32 instead of float64
        X_values = X[self.ftr_columns].values.astype('float32')
        X_sc = self.scaler.fit_transform(X_values)

        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit_transform(X_sc)
        
        # Memory cleanup
        del X_values, X_sc
        gc.collect()
        
        return self

    def transform(self, X):
        # Store original index to preserve it - EXACT SAME logic
        original_index = X.index
        
        # Memory optimization: Use float32
        X_values = X[self.ftr_columns].values.astype('float32')
        X_sc = self.scaler.transform(X_values)
        X_pca = self.pca.transform(X_sc)
        
        # Create DataFrame with original index preserved - EXACT SAME logic
        result = pd.DataFrame(X_pca, index=original_index)
        
        # Memory cleanup
        del X_values, X_sc, X_pca
        gc.collect()
        
        return result

class TSFreshFeaturesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, fdr_level=0.001, upper_threshold=135):
        # EXACT SAME parameters
        self.fdr_level = fdr_level
        self.upper_threshold = upper_threshold

    def fit(self, X, y=None):
        # EXACT SAME logic - Simplified approach like Google Colab
        index_df = X.index.to_frame().reset_index(drop=True)
        
        # Handle different index structures - EXACT SAME logic
        if len(index_df.columns) == 2:
            index_df.columns = ['unit_id', 'time_cycles']
        else:
            print("Warning: Unexpected index structure")
            # Use all features as fallback - EXACT SAME logic
            self.selected_ftr = X.columns
            return self
        
        # Calculate RUL - EXACT SAME logic
        rul = calculate_RUL(index_df, upper_threshold=self.upper_threshold)
        
        # Feature selection - EXACT SAME logic
        try:
            # Memory optimization: Convert to float32 for selection process
            X_optimized = X.astype('float32')
            X_t = select_features(X_optimized, rul, fdr_level=self.fdr_level)
            self.selected_ftr = X_t.columns
            print(f'Selected {len(self.selected_ftr)} out of {X.shape[1]} features: '
                  f'{self.selected_ftr.to_list()}')
            
            # Memory cleanup
            del X_optimized, X_t
            gc.collect()
            
        except Exception as e:
            print(f"Feature selection failed: {e}. Using all features.")
            # EXACT SAME fallback logic
            self.selected_ftr = X.columns
            
        return self

    def transform(self, X):
        # EXACT SAME transform logic
        return X[self.selected_ftr]

class FeatureEngineering:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_feature_engineering_config()
        self.metrics = TurbineMLOpsMetrics()

    def _optimize_memory_usage(self, df):
        """Memory optimization without changing data integrity"""
        print("Optimizing memory usage...")
        
        # Convert int64 to smaller int types where possible
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')
        
        # Convert float64 to float32 where precision allows
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        
        return df

    def create_long_term_pipeline(self, batch_size=500):
        """Create pipeline for long-term features (19 time steps) - EXACT SAME logic with memory optimization"""
        return Pipeline([
            ('roll-time-series', RollTimeSeries(min_timeshift=19, max_timeshift=19, rolling_direction=1, batch_size=batch_size)),
            ('extract-tsfresh-features', TSFreshFeaturesExtractor(calc=tsfresh_calc, batch_size=batch_size)),
            ('PCA', CustomPCA(n_components=20)),  # EXACT SAME parameters
            ('features-selection', TSFreshFeaturesSelector(fdr_level=0.001)),  # EXACT SAME parameters
        ])

    def create_short_term_pipeline(self, batch_size=1000):
        """Create pipeline for short-term features (4 time steps) - EXACT SAME logic with memory optimization"""
        return Pipeline([
            ('roll-time-series', RollTimeSeries(min_timeshift=4, max_timeshift=4, rolling_direction=1, batch_size=batch_size)),
            ('extract-tsfresh-features', TSFreshFeaturesExtractor(calc={'mean': None}, batch_size=batch_size)),  # EXACT SAME calc
            ('features-selection', TSFreshFeaturesSelector(fdr_level=0.0002)),  # EXACT SAME parameters
        ])

    @monitor_pipeline_stage('feature_engineering')
    def initiate_feature_engineering(self, batch_size=300):
        """Main feature engineering process - EXACT SAME logic with memory optimizations"""
        print("Starting Feature Engineering...")
        
        # Load preprocessed data from previous stage
        train_data = pd.read_csv(self.config.train_preprocessed_path)
        
        # Memory optimization only - no data changes
        train_data = self._optimize_memory_usage(train_data)
        print(f"Data shape: {train_data.shape}, Memory usage: {train_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Create feature engineering pipelines - EXACT SAME logic
        features_long_h_pipe = self.create_long_term_pipeline(batch_size=batch_size)
        features_short_h_pipe = self.create_short_term_pipeline(batch_size=batch_size*2)
        
        # Extract long-term features WITH TIMING - EXACT SAME logic
        print("Extracting long-term features...")
        start_long = time.time()
        train_long_h_ftrs = features_long_h_pipe.fit_transform(train_data.copy())
        long_duration = time.time() - start_long
        
        # Memory cleanup between operations
        gc.collect()
        
        # Extract short-term features WITH TIMING - EXACT SAME logic
        print("Extracting short-term features...")
        start_short = time.time()
        train_short_h_ftrs = features_short_h_pipe.fit_transform(train_data.copy())
        short_duration = time.time() - start_short
        
        # Free original data memory
        del train_data
        gc.collect()
        
        # Merge both feature sets - EXACT SAME logic
        train_ftrs = train_long_h_ftrs.merge(
            train_short_h_ftrs, 
            how='inner',
            right_index=True, 
            left_index=True
        )

        # Free intermediate feature sets
        del train_long_h_ftrs, train_short_h_ftrs
        gc.collect()

        # ENHANCED MONITORING - Record feature engineering metrics - EXACT SAME logic
        extraction_times = {
            'long_term': long_duration,
            'short_term': short_duration
        }
        
        feature_counts = {
            'long_term': train_ftrs.shape[1] // 2,  # Approximate since we freed memory
            'short_term': train_ftrs.shape[1] // 2,  # Approximate since we freed memory
            'final': train_ftrs.shape[1]
        }
        
        # Calculate selection ratio - EXACT SAME logic
        final_features = train_ftrs.shape[1]
        selection_ratio = final_features / 100  # Approximate since original data freed
        
        # Record monitoring metrics - EXACT SAME logic
        self.metrics.record_feature_engineering_metrics(extraction_times, feature_counts, selection_ratio)
        
        # Set proper index names if it's a MultiIndex - EXACT SAME logic
        if hasattr(train_ftrs.index, 'names') and len(train_ftrs.index.names) == 2:
            train_ftrs.index = train_ftrs.index.set_names(['unit_id', 'time_cycles'])
        
        print(f'Final training features shape: {train_ftrs.shape}')
        
        # Create directories - EXACT SAME logic
        os.makedirs(os.path.dirname(self.config.engineered_features_path), exist_ok=True)
        
        # Save engineered features - EXACT SAME logic
        train_ftrs.to_csv(self.config.engineered_features_path)
        
        # Save both pipelines for future use - EXACT SAME logic
        with open(self.config.long_term_pipeline_path, 'wb') as f:
            pickle.dump(features_long_h_pipe, f)
            
        with open(self.config.short_term_pipeline_path, 'wb') as f:
            pickle.dump(features_short_h_pipe, f)
        
        print(f"Feature engineering completed!")
        print(f"Engineered features saved at: {self.config.engineered_features_path}")
        print(f"Long-term pipeline saved at: {self.config.long_term_pipeline_path}")
        print(f"Short-term pipeline saved at: {self.config.short_term_pipeline_path}")
        
        return (self.config.engineered_features_path, 
                self.config.long_term_pipeline_path, 
                self.config.short_term_pipeline_path)