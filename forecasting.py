# forecasting.py
import pandas as pd

def forecast_next_days(model, df, days=2):
    """
    Generate a forecast for the next 'days' days using an iterative approach.
    This is a simplified example. For each forecasted day, the previous
    prediction is used to update lag features.
    """
    last_row = df.iloc[-1].copy()
    predictions = []

    for _ in range(days):
        # Extract features for prediction (assumes 'Close' and lag features exist)
        features = last_row.drop('Close').values.reshape(1, -1)
        pred = model.predict(features)[0]
        predictions.append(pred)
        
        # Update lag features: shift lag values and use current prediction
        for lag in range(3, 1, -1):
            last_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}']
        last_row['lag_1'] = last_row['Close']
        last_row['Close'] = pred  # Update the 'Close' value with the prediction

    return predictions
