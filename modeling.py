# modeling.py
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from config import MODEL_PARAMS

def prepare_data(df):
    """
    Prepare features and target from the DataFrame.
    Here, we drop the 'Close' column from the features.
    """
    features = df.drop(['Close'], axis=1)
    target = df['Close']
    return features, target

def train_model(X_train, y_train):
    """Train an XGBoost regression model."""
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using RMSE."""
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"RMSE: {rmse:.4f}")
    return rmse, preds

def train_and_evaluate(df):
    """
    Prepare the data, split into training and testing portions,
    train the model, and evaluate it.
    """
    features, target = prepare_data(df)
    # Simple time-series split (80% train, 20% test)
    split = int(len(df) * 0.8)
    X_train, y_train = features.iloc[:split], target.iloc[:split]
    X_test, y_test = features.iloc[split:], target.iloc[split:]
    model = train_model(X_train, y_train)
    rmse, preds = evaluate_model(model, X_test, y_test)
    return model
