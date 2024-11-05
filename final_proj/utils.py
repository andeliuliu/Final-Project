# final_proj/utils.py

import yfinance as yf
import pandas as pd

def get_stock_data(symbol, period='2y', interval='1d'):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        data.reset_index(inplace=True)
        print("Raw Data:", data.head(720))  # Debug line to check if data is loaded correctly
        return data
    except Exception as e:
        raise ValueError(f"Failed to retrieve data for symbol {symbol}: {e}")
    
def calculate_upside_downside(stock_info):
    target_price = stock_info.get('targetMeanPrice')
    current_price = stock_info.get('currentPrice')
    if target_price and current_price:
        return ((target_price - current_price) / current_price) * 100
    return None

def get_recommendation(stock_data):
    if stock_data.upside_downside is not None:
        if stock_data.upside_downside > 15:
            return "Buy"
        elif stock_data.upside_downside < -15:
            return "Sell"
        else:
            return "Hold"
    return "No Recommendation"

def calculate_moving_averages(data, windows=[10, 50]):
    for window in windows:
        data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    return data

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    data['Bollinger_High'] = rolling_mean + (rolling_std * num_std)
    data['Bollinger_Low'] = rolling_mean - (rolling_std * num_std)
    return data

def engineer_features(data):
    data = calculate_moving_averages(data)
    data = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data.dropna(inplace=True)
    return data

# utils.py

def prepare_data_for_model(data, horizon=1):
    # Define features for the model
    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA_10', 'MA_50', 'RSI', 'Bollinger_High', 'Bollinger_Low']]

    # Define the target variable (shifted by `horizon` days into the future)
    y = data['Close'].shift(-horizon)  # Shift the target variable by the specified horizon

    # Drop rows with NaN values that result from the shift
    X = X.iloc[:-horizon]
    y = y.iloc[:-horizon]

    return X, y

def split_data(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    return X_train, X_test, y_train, y_test