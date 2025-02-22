# feature_engineering.py
import pandas as pd
import ta

def add_technical_indicators(df):
    """
    Add technical indicators using the 'ta' library.
    For example: SMA, EMA, RSI, and MACD.
    """
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    # Fill missing values created by indicators
    df.fillna(method='bfill', inplace=True)
    return df

def create_lag_features(df, lags=[1, 2, 3]):
    """
    Create lag features for the 'Close' price.
    These will serve as predictors in the model.
    """
    for lag in lags:
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)
    return df
