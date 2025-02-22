# config.py
import os

# List of stock tickers to forecast
COMPANIES = ['CELH', 'CVNA', 'UPST', 'ALT', 'FUBO']

# Date parameters for historical data
START_DATE = '2018-01-01'
END_DATE = '2023-12-31'

# Forecast horizon (in days)
FORECAST_DAYS = 2

# Model parameters for XGBoost (example)
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': 42
}

# Directories for saving data and models
DATA_DIR = "data"
MODEL_DIR = "models"
