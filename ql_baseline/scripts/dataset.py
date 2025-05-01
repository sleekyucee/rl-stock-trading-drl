#scripts/dataset.py

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

def load_stock_data(csv_path):
    """Loads and returns the raw TSLA dataset."""
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    return df

def discretize_features(df, n_bins=5):
    """Discretizes the features using quantile-based binning."""
    features = df.columns.tolist()
    feature_data = df[features].copy()

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    discretized_array = discretizer.fit_transform(feature_data)

    discretized_df = pd.DataFrame(discretized_array, columns=features, index=feature_data.index)

    return discretized_df, feature_data

