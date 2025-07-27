import os
import pandas as pd
import pickle
import warnings
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series

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
    'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance',
                      'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
}

class RollTimeSeries(BaseEstimator, TransformerMixin):
    def __init__(self, min_timeshift, max_timeshift, rolling_direction):
        self.min_timeshift = min_timeshift
        self.max_timeshift = max_timeshift
        self.rolling_direction = rolling_direction

    def fit(self, X, y=None):  # ← Fixed: Added y=None
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
    def __init__(self, calc=tsfresh_calc):
        self.calc = calc

    def _clean_features(self, X):
        print(f"Before cleaning - Index type: {type(X.index)}, names: {X.index.names}")
        
        old_shape = X.shape
        X_t = X.T.drop_duplicates().T  # ← This might be destroying the MultiIndex!
        print(f'Dropped {old_shape[1] - X_t.shape[1]} duplicate features')
        print(f"After drop_duplicates - Index type: {type(X_t.index)}, names: {X_t.index.names}")

        old_shape = X_t.shape
        X_t = X_t.dropna(axis=1)  # ← This might also be destroying the MultiIndex!
        print(f'Dropped {old_shape[1] - X_t.shape[1]} features with NA values')
        print(f"After dropna - Index type: {type(X_t.index)}, names: {X_t.index.names}")
        
        return X_t

    def fit(self, X, y=None):  # ← Fixed: Added y=None
        return self

    def transform(self, X):
        _start = datetime.now()
        print('Start Extracting Features')
        
        # Debug: Check input structure
        print(f"TSFresh input shape: {X.shape}")
        print(f"TSFresh input index: {type(X.index)}")
        print(f"TSFresh input columns: {X.columns.tolist()}")
        
        X_t = extract_features(
            X[['id', 'time_cycles'] +  # ← Use 'id' like Google Colab
              X.columns[X.columns.str.startswith('sensor')].tolist()],
            column_id='id',  # ← Use 'id' like Google Colab
            column_sort='time_cycles',
            default_fc_parameters=self.calc)
            
        print(f'Done Extracting Features in {datetime.now() - _start}')
        
        # Debug: Check TSFresh output structure  
        print(f"TSFresh output shape: {X_t.shape}")
        print(f"TSFresh output index type: {type(X_t.index)}")
        print(f"TSFresh output index names: {X_t.index.names}")
        print(f"TSFresh output sample index: {X_t.index[:5]}")
        
        X_t = self._clean_features(X_t)
        
        # Debug: Check after cleaning
        print(f"After cleaning shape: {X_t.shape}")
        print(f"After cleaning index type: {type(X_t.index)}")
        print(f"After cleaning index names: {X_t.index.names}")
        
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
        return self

    def transform(self, X):
        # Store original index to preserve it
        original_index = X.index
        
        X_sc = self.scaler.transform(X[self.ftr_columns].values)
        X_pca = self.pca.transform(X_sc)
        
        # Create DataFrame with original index preserved
        result = pd.DataFrame(X_pca, index=original_index)
        
        return result

class TSFreshFeaturesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, fdr_level=0.001, upper_threshold=135):
        self.fdr_level = fdr_level
        self.upper_threshold = upper_threshold

    def fit(self, X, y=None):
        # Check index structure first
        print(f"TSFreshFeaturesSelector - Index type: {type(X.index)}")
        print(f"TSFreshFeaturesSelector - Index names: {X.index.names}")
        print(f"TSFreshFeaturesSelector - Sample index: {X.index[:5].tolist()}")
        
        # Check if we have proper MultiIndex
        if hasattr(X.index, 'names') and len(X.index.names) == 2 and X.index.names != [None, None]:
            # We have a proper MultiIndex
            index_df = X.index.to_frame().reset_index(drop=True)
            index_df.columns = ['unit_id', 'time_cycles']
        elif hasattr(X.index, 'levels'):
            # MultiIndex but unnamed
            index_df = X.index.to_frame().reset_index(drop=True)
            index_df.columns = ['unit_id', 'time_cycles']
        else:
            # Single index - this is the problem! 
            print("ERROR: Expected MultiIndex but got single index. This breaks RUL calculation.")
            print("Falling back to using all features...")
            self.selected_ftr = X.columns
            return self
        
        print(f"Index DataFrame shape: {index_df.shape}")
        print(f"Unique unit_ids: {index_df['unit_id'].nunique()}")
        print(f"Time cycles per unit (sample): {index_df.groupby('unit_id')['time_cycles'].count().head()}")
        
        # Calculate RUL
        rul = calculate_RUL(index_df, upper_threshold=self.upper_threshold)
        
        # Debug RUL
        print(f"RUL unique values: {len(pd.Series(rul).unique())}")
        print(f"RUL range: {pd.Series(rul).min()} to {pd.Series(rul).max()}")
        
        # Check if RUL has enough variation
        if len(pd.Series(rul).unique()) <= 1:
            print("WARNING: RUL has no variation. Using all features.")
            self.selected_ftr = X.columns
            return self
        
        try:
            X_t = select_features(X, rul, fdr_level=self.fdr_level)
            self.selected_ftr = X_t.columns
            print(f'Selected {len(self.selected_ftr)} out of {X.shape[1]} features: '
                  f'{self.selected_ftr.to_list()}')
        except Exception as e:
            print(f"Feature selection failed: {e}")
            print("Using all features as fallback.")
            self.selected_ftr = X.columns
            
        return self

    def transform(self, X):
        return X[self.selected_ftr]

class FeatureEngineering:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_feature_engineering_config()

    def create_long_term_pipeline(self):
        """Create pipeline for long-term features (19 time steps)"""
        return Pipeline([
            # Basic preprocessing already done in previous stage
            ('roll-time-series', RollTimeSeries(min_timeshift=19, max_timeshift=19, rolling_direction=1)),
            ('extract-tsfresh-features', TSFreshFeaturesExtractor(calc=tsfresh_calc)),
            ('PCA', CustomPCA(n_components=20)),
            ('features-selection', TSFreshFeaturesSelector(fdr_level=0.001)),
        ])

    def create_short_term_pipeline(self):
        """Create pipeline for short-term features (4 time steps)"""
        return Pipeline([
            # Basic preprocessing already done in previous stage
            ('roll-time-series', RollTimeSeries(min_timeshift=4, max_timeshift=4, rolling_direction=1)),
            ('extract-tsfresh-features', TSFreshFeaturesExtractor(calc={'mean': None})),
            ('features-selection', TSFreshFeaturesSelector(fdr_level=0.0002)),
        ])

    def initiate_feature_engineering(self):
        """Main feature engineering process"""
        print("Starting Feature Engineering...")
        
        # Load preprocessed data from previous stage
        train_data = pd.read_csv(self.config.train_preprocessed_path)
        
        # Create feature engineering pipelines
        features_long_h_pipe = self.create_long_term_pipeline()
        features_short_h_pipe = self.create_short_term_pipeline()
        
        # Extract long-term features
        print("Extracting long-term features...")
        train_long_h_ftrs = features_long_h_pipe.fit_transform(train_data)
        
        # Extract short-term features
        print("Extracting short-term features...")
        train_short_h_ftrs = features_short_h_pipe.fit_transform(train_data)
        
        # Merge both feature sets
        train_ftrs = train_long_h_ftrs.merge(
            train_short_h_ftrs, 
            how='inner',
            right_index=True, 
            left_index=True
        )
        
        # Debug: Check merged result structure
        print(f"After merge - Index type: {type(train_ftrs.index)}")
        print(f"After merge - Index names: {train_ftrs.index.names}")
        print(f"After merge - Shape: {train_ftrs.shape}")
        
        # Handle index naming based on structure
        if hasattr(train_ftrs.index, 'names') and len(train_ftrs.index.names) == 2:
            # We have a MultiIndex
            train_ftrs.index = train_ftrs.index.set_names(['unit_id', 'time_cycles'])
        else:
            # We have a single index - convert to MultiIndex if needed
            print("Warning: Merged result has single index instead of MultiIndex")
            # For now, just keep the single index
        
        print(f'Final training features shape: {train_ftrs.shape}')
        
        # Create directories
        os.makedirs(os.path.dirname(self.config.engineered_features_path), exist_ok=True)
        
        # Save engineered features
        train_ftrs.to_csv(self.config.engineered_features_path)
        
        # Save both pipelines for future use
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