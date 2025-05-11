#dataset

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

def load_stock_data(csv_path):
    """Loads and returns the raw dataset."""
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    return df

def fit_discretizer(df, n_bins=5):
    """Fits a discretizer on the provided data and returns the transformer."""
    features = df.columns.tolist()
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    discretizer.fit(df[features])
    return discretizer

def apply_discretizer(df, discretizer):
    """Applies a previously fitted discretizer to new data."""
    features = df.columns.tolist()
    transformed = discretizer.transform(df[features])
    discretized_df = pd.DataFrame(transformed, columns=features, index=df.index)
    return discretized_df
