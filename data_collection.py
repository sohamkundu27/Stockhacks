# data_collection.py
import os
import yfinance as yf
import pandas as pd
from config import COMPANIES, START_DATE, END_DATE, DATA_DIR

def fetch_stock_data(ticker):
    """Fetch historical stock data using yfinance."""
    data = yf.download(ticker, start=START_DATE, end=END_DATE)
    return data

def save_data_to_csv(ticker, df):
    """Save stock data to CSV."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    df.index.name = 'Date'  # Set the index name to 'Date'
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(file_path)
    return file_path


def load_data_from_csv(ticker):
    """Load stock data from CSV."""
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    return pd.read_csv(file_path, index_col='Date', parse_dates=True)

def fetch_and_save_all():
    """Fetch and save data for all companies."""
    for ticker in COMPANIES:
        print(f"Fetching data for {ticker}...")
        df = fetch_stock_data(ticker)
        save_data_to_csv(ticker, df)
    print("Data fetching complete.")

