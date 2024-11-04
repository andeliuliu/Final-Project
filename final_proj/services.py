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

def predict_stock_price(symbol):
    data = get_stock_data(symbol, period='2y')
    data = engineer_features(data)
    X, y = prepare_data_for_model(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, X_test, y_test)
    y_pred, rmse = evaluate_model(model, X_test, y_test)
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importance = sorted(
        zip(feature_importances, feature_names),
        reverse=True
    )
    dates = data['Date'].iloc[-len(y_test):].dt.strftime('%Y-%m-%d').tolist()
    return {
        'dates': dates,
        'actual': y_test.tolist(),
        'predicted': y_pred.tolist(),
        'rmse': rmse,
        'feature_importance': feature_importance,
    }