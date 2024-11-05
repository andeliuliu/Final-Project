# final_proj/services.py

import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from .utils import (
    get_stock_data,
    engineer_features,
    prepare_data_for_model,
    split_data,
)
import yfinance as yf

def train_model(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.01,
        early_stopping_rounds=10,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return y_pred, rmse


def predict_stock_price(symbol, horizon):
    # Fetch historical stock data
    data = get_stock_data(symbol, period='2y')
    stock = yf.Ticker(symbol)

    # Engineer features and prepare data
    data = engineer_features(data)
    X, y = prepare_data_for_model(data, horizon)  # Use horizon to set the target

    # Split data and train model
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, X_test, y_test)
    y_pred, rmse = evaluate_model(model, X_test, y_test)

    # Feature importance
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importance = sorted(zip(feature_importances, feature_names), reverse=True)

    # Date labels for predictions
    dates = data['Date'].iloc[-len(y_test):].dt.strftime('%Y-%m-%d').tolist()

    # Fetch additional stock details
    stock_info = stock.info
    current_price = stock_info.get('currentPrice', None)
    market_cap = stock_info.get('marketCap', None)
    price_target = stock_info.get('targetMeanPrice', None)

    # Calculate upside/downside and recommendation
    if current_price is not None and price_target is not None:
        upside_downside = ((price_target - current_price) / current_price) * 100
    else:
        upside_downside = None

    recommendation = (
        "Buy" if upside_downside > 15 else
        "Sell" if upside_downside < -15 else
        "Hold"
    ) if upside_downside is not None else "No Recommendation"

    return {
        'dates': dates,
        'actual': y_test.tolist(),
        'predicted': y_pred.tolist(),
        'rmse': rmse,
        'feature_importance': feature_importance,
        'current_price': current_price,
        'market_cap': market_cap,
        'price_target': price_target,
        'upside_downside': upside_downside,
        'recommendation': recommendation,
    }