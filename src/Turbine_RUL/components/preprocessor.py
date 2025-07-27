from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

SENSOR_COLUMNS = ['sensor_{}'.format(x) for x in range(1, 22)]

class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops specified columns from dataset"""
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Only drop columns that actually exist
        existing_columns_to_drop = [col for col in self.columns_to_drop if col in X.columns]
        return X.drop(columns=existing_columns_to_drop)

class LowVarianceFeaturesRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=threshold)
    
    def fit(self, X, y=None):
        self.selector.fit(X)
        return self
    
    def transform(self, X):
        X_t = self.selector.transform(X)
        dropped_features = X.columns[~self.selector.get_support()]
        print(f'Dropped low variance features: {dropped_features.to_list()}')
        return pd.DataFrame(X_t, columns=self.selector.get_feature_names_out())

class ScalePerEngine(BaseEstimator, TransformerMixin):
    '''Scale individual engines time series with respect to its start'''
    def __init__(self, n_first_cycles=10, sensors_columns=SENSOR_COLUMNS):
        self.n_first_cycles = n_first_cycles
        self.sensors_columns = sensors_columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.sensors_columns = [x for x in X.columns if x in self.sensors_columns]
        
        init_sensors_avg = X[X['time_cycles'] <= self.n_first_cycles] \
            .groupby(by=['unit_id'])[self.sensors_columns] \
            .mean() \
            .reset_index()
        
        X_t = X[X['time_cycles'] > self.n_first_cycles].merge(
            init_sensors_avg,
            on=['unit_id'], how='left', suffixes=('', '_init_v')
        )
        
        for SENSOR in self.sensors_columns:
            X_t[SENSOR] = X_t[SENSOR] - X_t['{}_init_v'.format(SENSOR)]
        
        drop_columns = X_t.columns.str.endswith('init_v')
        return X_t[X_t.columns[~drop_columns]]