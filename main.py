# main.py
import pandas as pd
from data_collection import fetch_and_save_all, load_data_from_csv
from feature_engineering import add_technical_indicators, create_lag_features
from modeling import train_and_evaluate
from forecasting import forecast_next_days
from config import COMPANIES, FORECAST_DAYS

def process_stock(ticker):
    print(f"\nProcessing {ticker}...")
    # Load data (ensure data has been fetched already)
    df = load_data_from_csv(ticker)
    # Add technical indicators and lag features
    df = add_technical_indicators(df)
    df = create_lag_features(df)
    # Train the model and evaluate performance
    model = train_and_evaluate(df)
    # Forecast the next 'FORECAST_DAYS' days
    preds = forecast_next_days(model, df, days=FORECAST_DAYS)
    print(f"Forecast for {ticker} for next {FORECAST_DAYS} days: {preds}")
    return preds

def main():
    # Uncomment the following line to fetch data if not already saved
    #fetch_and_save_all()

    results = {}
    for ticker in COMPANIES:
        preds = process_stock(ticker)
        results[ticker] = preds

    # Save all predictions to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv("predictions.csv", index=False)
    print("\nPredictions saved to predictions.csv")

if __name__ == "__main__":
    main()
