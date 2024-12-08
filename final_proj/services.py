import xgboost as xgb
from sklearn.metrics import mean_squared_error
from .utils import (
    get_stock_data,
    prepare_data_for_model,
    split_data,
)
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np


def train_model(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost model.
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1500,
        learning_rate=0.01,
        max_depth=6,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    return model


def train_sarima(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """
    Train a SARIMA model.
    """
    model = SARIMAX(data['Close'], order=order, seasonal_order=seasonal_order)
    sarima_model = model.fit(disp=False)
    return sarima_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return predictions, RMSE, and MSE.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    return y_pred, rmse, mse


def predict_recursive(model, X_start, steps):
    """
    Recursive prediction for multi-step forecasting.
    """
    predictions = []
    X_current = X_start.copy()
    for _ in range(steps):
        prediction = model.predict(X_current)
        predictions.append(prediction[0])
        X_current.iloc[0, -3:] = [X_current.iloc[0, -2], X_current.iloc[0, -1], prediction[0]]
    return predictions


def train_multi_step_models(data, target, horizons):
    """
    Train a separate XGBoost model for each prediction horizon.
    """
    models = {}
    for horizon in horizons:
        print(f"Training model for {horizon}-step horizon...")
        X, y = prepare_data_for_model(data, target, horizon)
        X_train, X_test, y_train, y_test = split_data(X, y)
        model = train_model(X_train, y_train, X_test, y_test)
        models[horizon] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test
        }
    return models


def evaluate_multi_step_models(models):
    """
    Evaluate each trained model on its respective test data.
    """
    results = {}
    for horizon, data in models.items():
        model = data["model"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        y_pred, rmse, mse = evaluate_model(model, X_test, y_test)
        results[horizon] = {
            "y_pred": y_pred,
            "rmse": rmse,
            "mse": mse,
            "actual": y_test.tolist(),
            "dates": X_test.index.tolist()
        }
        print(f"Horizon {horizon}: RMSE = {rmse:.2f}, MSE = {mse:.2f}")
    return results


def predict_stock_price(symbol, target, horizon):
    """
    Predict stock prices using the trained model and calculate performance metrics.
    """
    # Fetch and engineer data
    data = get_stock_data(symbol, '2y')
    X, y = prepare_data_for_model(data, target, horizon)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = train_model(X_train, y_train, X_test, y_test)

    # Evaluate the model
    y_pred, rmse, mse = evaluate_model(model, X_test, y_test)

    # Calculate RMSE as a percentage of the Close price range
    close_min = data['Close'].min()
    close_max = data['Close'].max()
    close_range = close_max - close_min
    rmse_range_percentage = (rmse / close_range) * 100 if close_range != 0 else None

    # Calculate RMSE as a percentage of the average Close price
    close_avg = data['Close'].mean()
    rmse_avg_percentage = (rmse / close_avg) * 100 if close_avg != 0 else None

    # Fetch stock information using yfinance
    stock = yf.Ticker(symbol)
    stock_info = stock.info
    current_price = stock_info.get('currentPrice', None)
    market_cap = stock_info.get('marketCap', None)
    price_target = stock_info.get('targetMeanPrice', None)

    # Calculate upside/downside percentage
    if current_price is not None and price_target is not None:
        upside_downside = ((price_target - current_price) / current_price) * 100
    else:
        upside_downside = None

    # Generate a recommendation based on upside/downside
    if upside_downside is not None:
        if upside_downside > 10:
            recommendation = "Strong Buy"
        elif 0 < upside_downside <= 10:
            recommendation = "Buy"
        elif -10 <= upside_downside < 0:
            recommendation = "Hold"
        else:
            recommendation = "Sell"
    else:
        recommendation = "No Recommendation Available"

    # Predict the next day’s closing price
    next_day_prediction = None
    if len(X) > 0:
        last_row = X.iloc[[-1]]  # Get the most recent data row
        next_day_prediction = model.predict(last_row)[0]

    # Print results for debugging (optional)
    print(f"Current Price: {current_price}")
    print(f"Market Cap: {market_cap}")
    print(f"Price Target: {price_target}")
    print(f"Upside/Downside: {upside_downside}")
    print(f"Recommendation: {recommendation}")
    print(f"Next Day Prediction: {next_day_prediction}")

    return {
        "dates": data["Date"].iloc[-len(y_test):].dt.strftime("%Y-%m-%d").tolist(),
        "actual": y_test.tolist(),
        "predicted": y_pred.tolist(),
        "mse": mse,
        "rmse": rmse,
        "rmse_range_percentage": rmse_range_percentage,
        "rmse_avg_percentage": rmse_avg_percentage,
        "current_price": current_price,
        "market_cap": market_cap,
        "price_target": price_target,
        "upside_downside": upside_downside,
        "recommendation": recommendation,
        "next_day_prediction": next_day_prediction,
    }


def ensemble_predictions(data, horizons):
    """
    Combine predictions from XGBoost and SARIMA using a weighted ensemble.
    """
    # Train SARIMA
    sarima_model = train_sarima(data)

    # Train XGBoost for each horizon
    models = train_multi_step_models(data, 'Close', horizons)

    # Combine predictions
    results = {}
    for horizon in horizons:
        xgb_pred = models[horizon]["model"].predict(models[horizon]["X_test"])
        sarima_pred = sarima_model.predict(start=0, end=len(models[horizon]["X_test"]) - 1)

        combined_pred = 0.7 * xgb_pred + 0.3 * sarima_pred

        results[horizon] = {
            "combined_pred": combined_pred,
            "rmse": mean_squared_error(models[horizon]["y_test"], combined_pred) ** 0.5
        }
    return results