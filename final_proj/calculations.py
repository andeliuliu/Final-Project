import numpy as np

def calculate_vwap(data):
    """
    Calculate the Volume Weighted Average Price (VWAP).
    """
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    return data

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

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculate the MACD and MACD Signal Line.
    """
    data['MACD'] = data['Close'].ewm(span=short_window, adjust=False).mean() - data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

def calculate_atr(data, period=14):
    """
    Calculate the Average True Range (ATR).
    """
    data['True_Range'] = np.maximum(data['High'] - data['Low'],
                                    np.maximum(abs(data['High'] - data['Close'].shift(1)),
                                               abs(data['Low'] - data['Close'].shift(1))))
    data['ATR'] = data['True_Range'].rolling(window=period).mean()
    return data

def calculate_adx(data, period=14):
    """
    Calculate the Average Directional Index (ADX).
    """
    data['+DM'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']),
                           np.maximum(data['High'] - data['High'].shift(1), 0), 0)
    data['-DM'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)),
                           np.maximum(data['Low'].shift(1) - data['Low'], 0), 0)
    data['+DI'] = 100 * (data['+DM'].rolling(window=period).mean() / data['ATR'])
    data['-DI'] = 100 * (data['-DM'].rolling(window=period).mean() / data['ATR'])
    data['ADX'] = 100 * abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])
    return data

def calculate_stochastic_oscillator(data, period=14):
    """
    Calculate the Stochastic Oscillator.
    """
    data['Stochastic'] = ((data['Close'] - data['Low'].rolling(window=period).min()) /
                          (data['High'].rolling(window=period).max() - data['Low'].rolling(window=period).min())) * 100
    return data

def calculate_additional_features(data):
    """
    Add additional features: Daily Variation, High-Close, Low-Open.
    """
    data['Daily_Variation'] = (data['High'] - data['Low']) / data['Open']
    data['High_Close'] = (data['High'] - data['Close']) / data['Open']
    data['Low_Open'] = (data['Low'] - data['Open']) / data['Open']
    return data