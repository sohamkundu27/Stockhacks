# pip install yfinance pandas numpy PyWavelets statsmodels tensorflow scikit-learn xgboost
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import pywt
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Global ticker and forecast horizon (2-day ahead)
ticker = "FUBO"
FORECAST_HORIZON = 2

#####################################
# FUNDAMENTAL ANALYSIS SECTION
#####################################
def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    fundamentals = {
        "trailingPE": info.get("trailingPE", np.nan),
        "dividendYield": info.get("dividendYield", np.nan),
        "priceToBook": info.get("priceToBook", np.nan),
        "profitMargins": info.get("profitMargins", np.nan)
    }
    # Replace NaNs with zero for production (or choose an alternative imputation)
    return {k: (np.nan_to_num(v, nan=0)) for k, v in fundamentals.items()}

#####################################
# MARKOV CHAIN ANALYSIS SECTION
#####################################
def markov_chain_analysis(ticker, start_date="2021-01-01",
                          end_date=datetime.date.today().strftime('%Y-%m-%d')):
    data = yf.download(ticker, start=start_date, end=end_date)
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    data["daily_return"] = data[price_col].pct_change()
    data["state"] = np.where(data["daily_return"] >= 0, "up", "down")
    
    numerator = len(data[
        (data["state"] == "up") &
        (data["state"].shift(-1) == "down") &
        (data["state"].shift(-2) == "down") &
        (data["state"].shift(-3) == "down") &
        (data["state"].shift(-4) == "down") &
        (data["state"].shift(-5) == "down")
    ])
    denominator = len(data[
        (data["state"].shift(1) == "down") &
        (data["state"].shift(2) == "down") &
        (data["state"].shift(3) == "down") &
        (data["state"].shift(4) == "down") &
        (data["state"].shift(5) == "down")
    ])
    pattern_probability = numerator / denominator if denominator != 0 else 0.0
    return pattern_probability

#####################################
# OPTIONS CONTRACTS ANALYSIS SECTION
#####################################
def get_options_data(ticker, expiration):
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiration)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    calls['expiration'] = expiration
    puts['expiration'] = expiration
    return calls, puts

def filter_outliers(df, std_multiplier=2):
    mean_volume = df['volume'].mean()
    std_volume = df['volume'].std()
    lower_threshold = max(mean_volume - std_multiplier * std_volume, 0)
    upper_threshold = mean_volume + std_multiplier * std_volume
    outliers = df[(df['volume'] < lower_threshold) | (df['volume'] > upper_threshold)]
    return outliers

def compute_side_averages(calls_outliers, puts_outliers):
    avg_calls_volume = calls_outliers['volume'].mean() if not calls_outliers.empty else np.nan
    avg_puts_volume = puts_outliers['volume'].mean() if not puts_outliers.empty else np.nan
    if np.isnan(avg_calls_volume) and np.isnan(avg_puts_volume):
        return None, None, None
    elif np.nan_to_num(avg_calls_volume) >= np.nan_to_num(avg_puts_volume):
        side = "calls"
        avg_strike = calls_outliers['strike'].mean() if not calls_outliers.empty else np.nan
    else:
        side = "puts"
        avg_strike = puts_outliers['strike'].mean() if not puts_outliers.empty else np.nan
    return side, avg_calls_volume if side=="calls" else avg_puts_volume, avg_strike

def options_contracts_signal(ticker, std_multiplier=2):
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        return None, None, None
    expiration = expirations[0]
    calls, puts = get_options_data(ticker, expiration)
    calls_outliers = filter_outliers(calls, std_multiplier)
    puts_outliers = filter_outliers(puts, std_multiplier)
    side, avg_volume, avg_strike = compute_side_averages(calls_outliers, puts_outliers)
    return side, avg_volume, avg_strike

#####################################
# TECHNICAL INDICATORS SECTION
#####################################
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_vwap(df, period=14):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].rolling(window=period, min_periods=1).sum()
    epsilon = 1e-8
    return (typical * df["Volume"]).rolling(window=period, min_periods=1).sum() / (vol + epsilon)

def compute_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = series.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = series.ewm(span=slowperiod, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
    return macd_line, signal_line

def compute_stoch_osc(high_series, low_series, close_series, k_period=14, d_period=3):
    lowest_low = high_series.rolling(k_period, min_periods=1).min()
    highest_high = high_series.rolling(k_period, min_periods=1).max()
    stoch_k = 100 * (close_series - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(d_period, min_periods=1).mean()
    return stoch_k, stoch_d

def compute_bollinger_bands(df, window=20, num_std=2):
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band, lower_band

#####################################
# PREPROCESSING: WAVELET DENOISING
#####################################
def wavelet_denoise(series, wavelet='db4', level=1):
    coeff = pywt.wavedec(series, wavelet, mode="per")
    # Zero-out detail coefficients
    coeff[1:] = [np.zeros_like(detail) for detail in coeff[1:]]
    return pywt.waverec(coeff, wavelet, mode="per")[:len(series)]

#####################################
# XGBOOST FORECASTING SECTION WITH HYPERPARAMETER TUNING
#####################################
def create_lag_features(df, n_lags=5):
    df = df.copy()
    # Denoise the Close series
    df['Close_denoised'] = wavelet_denoise(df['Close'])
    for i in range(1, n_lags+1):
        df[f'lag_{i}'] = df['Close_denoised'].shift(i)
    df.dropna(inplace=True)
    return df

#####################################
# HELPER: LINEAR REGRESSION SLOPE
#####################################
def compute_linreg(s, window):
    x = np.arange(len(s))
    try:
        slope, _ = np.polyfit(x, s, 1)
    except np.linalg.LinAlgError:
        slope = np.nan
    return slope

#####################################
# FINAL COMBINER MODEL (ENSEMBLE)
#####################################
def final_combiner_model(df, base_model, options_strike, fundamentals, linreg_length=14):
    df = df.copy()
    # Ensure "Close_denoised" exists; if not, recompute it
    if "Close_denoised" not in df.columns:
        df["Close_denoised"] = wavelet_denoise(df["Close"])
    
    # Ensure "target" exists; if not, compute it
    if "target" not in df.columns:
        df["target"] = df["Close"].shift(-FORECAST_HORIZON)
    
    # Compute technical indicators
    df["sma_200"] = df["Close"].rolling(200, min_periods=1).mean()
    df["linreg_14"] = df["Close"].rolling(linreg_length, min_periods=1).apply(lambda s: compute_linreg(s, linreg_length), raw=False)
    df["vwap_14"] = compute_vwap(df, period=14)
    df["rsi_14"] = compute_rsi(df["Close"], period=14)
    macd_line, signal_line = compute_macd(df["Close"])
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    stoch_k, stoch_d = compute_stoch_osc(df["High"], df["Low"], df["Close"])
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d
    upper_bb, lower_bb = compute_bollinger_bands(df)
    df["upper_bb"] = upper_bb
    df["lower_bb"] = lower_bb

    # Add Markov chain output
    df["markov_prob"] = markov_chain_analysis(ticker, start_date="2021-01-01",
                                              end_date=datetime.date.today().strftime('%Y-%m-%d'))
    
    # Use only rows with valid target
    final_data = df[df["target"].notna()].copy()
    final_data.reset_index(drop=True, inplace=True)
    
    # Impute missing technical features
    tech_features = ["sma_200", "linreg_14", "vwap_14", "rsi_14", "macd_line", "macd_signal", "stoch_k", "stoch_d", "upper_bb", "lower_bb"]
    final_data[tech_features] = final_data[tech_features].ffill().fillna(final_data[tech_features].median()).fillna(0)
    
    # Sanitize infinite values and replace any leftover NaNs
    final_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_data = final_data.fillna(0)
    
    if final_data.empty:
        raise ValueError("After filling missing values, no data is available for training the combiner model.")
    
    # Define columns used by the base model; include "Close_denoised"
    xgb_cols = ["High", "Low", "Volume", "Close_denoised"] + [f'lag_{i}' for i in range(1, 6)]
    final_data["base_pred"] = base_model.predict(final_data[xgb_cols])
    
    if options_strike is None:
        options_strike = 0.0
    final_data["opt_strike"] = options_strike
    final_data["trailingPE"] = fundamentals.get("trailingPE", 0)
    final_data["dividendYield"] = fundamentals.get("dividendYield", 0)
    final_data["priceToBook"] = fundamentals.get("priceToBook", 0)
    final_data["profitMargins"] = fundamentals.get("profitMargins", 0)
    
    features = final_data[["base_pred", "opt_strike", "linreg_14", "sma_200", "vwap_14", "rsi_14",
                           "macd_line", "macd_signal", "stoch_k", "stoch_d", "upper_bb", "lower_bb",
                           "trailingPE", "dividendYield", "priceToBook", "profitMargins", "markov_prob"]]
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    target = final_data["target"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features (convert to numpy arrays to avoid feature name warnings)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.values)
    
    # Build two combiner models and average their predictions
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    lr_model = LinearRegression()
    
    rf_model.fit(scaled_features, target.values)
    lr_model.fit(scaled_features, target.values)
    
    # For meta prediction, use the last row of df
    latest_row = df.iloc[-1]
    meta_input = {
        "base_pred": base_model.predict(latest_row[xgb_cols].values.reshape(1, -1))[0],
        "opt_strike": options_strike,
        "linreg_14": latest_row.get("linreg_14", 0),
        "sma_200": latest_row.get("sma_200", 0),
        "vwap_14": latest_row.get("vwap_14", 0),
        "rsi_14": latest_row.get("rsi_14", 0),
        "macd_line": latest_row.get("macd_line", 0),
        "macd_signal": latest_row.get("macd_signal", 0),
        "stoch_k": latest_row.get("stoch_k", 0),
        "stoch_d": latest_row.get("stoch_d", 0),
        "upper_bb": latest_row.get("upper_bb", 0),
        "lower_bb": latest_row.get("lower_bb", 0),
        "trailingPE": fundamentals.get("trailingPE", 0),
        "dividendYield": fundamentals.get("dividendYield", 0),
        "priceToBook": fundamentals.get("priceToBook", 0),
        "profitMargins": fundamentals.get("profitMargins", 0),
        "markov_prob": latest_row.get("markov_prob", 0)
    }
    feature_order = ["base_pred", "opt_strike", "linreg_14", "sma_200", "vwap_14", "rsi_14",
                     "macd_line", "macd_signal", "stoch_k", "stoch_d", "upper_bb", "lower_bb",
                     "trailingPE", "dividendYield", "priceToBook", "profitMargins", "markov_prob"]
    meta_input_df = pd.DataFrame([meta_input], columns=feature_order).fillna(0)
    meta_scaled = scaler.transform(meta_input_df.values)
    
    # Average the predictions of both models
    final_pred = (rf_model.predict(meta_scaled)[0] + lr_model.predict(meta_scaled)[0]) / 2.0
    return final_pred

#####################################
# PREDICTION FOR A SINGLE FORECAST
#####################################
def predict_forecast(df_train, options_strike, fundamentals):
    df_train = df_train.copy()
    df_train = create_lag_features(df_train)
    df_train["target"] = df_train["Close"].shift(-FORECAST_HORIZON)
    df_train.dropna(inplace=True)
    if len(df_train) < 50:
        return None
    
    train_size = int(len(df_train) * 0.8)
    train_df = df_train.iloc[:train_size]
    X_train = train_df.drop(["Close", "target"], axis=1)
    y_train = train_df["target"]
    
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train.values)
    
    xgb = XGBRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(scaled_X_train, y_train)
    base_model = grid_search.best_estimator_
    
    forecast = final_combiner_model(df_train, base_model, options_strike, fundamentals, linreg_length=14)
    return forecast

#####################################
# MAIN EXECUTION (PRODUCTION FORECAST)
#####################################
def main():
    stock_data = yf.download(ticker, start="2021-01-01", end=datetime.date.today().strftime('%Y-%m-%d'))
    stock_data.to_csv("stock_data.csv")
    
    stock = yf.Ticker(ticker)
    options_list = stock.options
    if options_list:
        expiration = options_list[0]
        calls, puts = get_options_data(ticker, expiration)
        calls.to_csv("options_calls.csv", index=False)
        puts.to_csv("options_puts.csv", index=False)
    
    _, _, options_strike = options_contracts_signal(ticker, std_multiplier=2)
    fundamentals = get_fundamentals(ticker)
    
    print("Production forecast:")
    prod_data = yf.download(ticker, start="2021-01-01", end=datetime.date.today().strftime('%Y-%m-%d'))[['Close','High','Low','Volume']].copy()
    prod_data = create_lag_features(prod_data)
    prod_data["target"] = prod_data["Close"].shift(-FORECAST_HORIZON)
    prod_data.dropna(inplace=True)
    
    if len(prod_data) < 50:
        print("Not enough data for production forecast.")
    else:
        train_size = int(len(prod_data) * 0.8)
        train_df = prod_data.iloc[:train_size]
        X_train = train_df.drop(["Close", "target"], axis=1)
        y_train = train_df["target"]
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train.values)
        
        xgb = XGBRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
        grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(scaled_X_train, y_train)
        base_model = grid_search.best_estimator_
        
        prod_forecast = final_combiner_model(prod_data, base_model, options_strike, fundamentals, linreg_length=14)
        print(f"FINAL PRICE PREDICTION: {prod_forecast:.2f}")

if __name__ == "__main__":
    main()
