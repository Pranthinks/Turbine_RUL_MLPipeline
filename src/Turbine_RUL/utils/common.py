import pandas as pd
import numpy as np

def calculate_RUL(X, upper_threshold=None):
    """Calculate Remaining Useful Life per unit - Google Colab version"""
    lifetime = X.groupby(['unit_id'])['time_cycles'].transform(max)
    rul = lifetime - X['time_cycles']

    if upper_threshold:
        rul = np.where(rul > upper_threshold, upper_threshold, rul)

    return rul