#train

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_path: str, window_size: int = 30, scaler: MinMaxScaler = None):
    """
    Load preprocessed stock data, normalize it, and return windowed states,
    close prices, scaler, and corresponding aligned raw data for analysis.

    Args:
        csv_path (str): Path to the dataset CSV file.
        window_size (int): Number of days to use in each state window.
        scaler (MinMaxScaler, optional): If provided, use this scaler instead of fitting a new one.

    Returns:
        states (np.ndarray): Shape (num_samples, window_size, num_features)
        prices (np.ndarray): Close prices aligned with each state (for reward calc)
        scaler (MinMaxScaler): Fitted MinMaxScaler instance
        raw_df (pd.DataFrame): Original (non-normalized) data aligned with states
    """
    df = pd.read_csv(csv_path, index_col='Date')
    df.dropna(inplace=True)

    features = df.columns.tolist()

    if scaler is None:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
    else:
        scaled_data = scaler.transform(df)

    df_scaled = pd.DataFrame(scaled_data, columns=features, index=df.index)

    states = []
    for t in range(window_size, len(df_scaled)):
        state = df_scaled.iloc[t - window_size:t].values
        states.append(state)

    states = np.array(states)
    prices = df['Close'].values[window_size:]
    raw_df = df.iloc[window_size:]

    return states, prices, scaler, raw_df