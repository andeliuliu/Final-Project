import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from .calculations import (
    calculate_vwap,
    calculate_moving_averages,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_atr,
    calculate_adx,
    calculate_stochastic_oscillator,
    calculate_additional_features,
)



def get_stock_data(symbol, period, interval='1d'):
    """
    Fetch historical stock data using yfinance.
    """
    print(f"Fetching stock data for symbol: {symbol}, period: {period}, interval: {interval}")
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        data.reset_index(inplace=True)
        print(f"Fetched data shape: {data.shape}")
        return data
    except Exception as e:
        raise ValueError(f"Failed to retrieve data for symbol {symbol}: {e}")


def feature_selection(data, target, significance_level=0.1):
    """
    Perform feature selection based on mutual information and p-values.
    """
    print(f"Starting feature selection with target: {target} and significance level: {significance_level}")
    
    # Separate features and target
    X = data.drop(columns=[target])
    y = data[target]
    print(f"Data shape before feature selection: Features - {X.shape}, Target - {y.shape}")

    # Ensure only numerical features are used
    X = X.select_dtypes(include=[np.number])
    print(f"Filtered numerical features shape: {X.shape}")

    # Compute mutual information
    mutual_info = mutual_info_regression(X, y)
    mutual_info_series = pd.Series(mutual_info, index=X.columns)
    mutual_info_series = mutual_info_series.sort_values(ascending=False)
    print(f"Mutual information scores: {mutual_info_series}")

    # Compute p-values
    p_values = []
    for col in X.columns:
        X_with_const = sm.add_constant(X[col])
        model = sm.OLS(y, X_with_const).fit()
        p_values.append(model.pvalues[1])  # Get p-value of the feature

    p_values_series = pd.Series(p_values, index=X.columns)
    print(f"P-values for features: {p_values_series}")

    # Select features with p-value < significance_level and high mutual information
    selected_features = mutual_info_series[
        (p_values_series < significance_level)
    ].index.tolist()
    print(f"Selected features: {selected_features}")

    return selected_features


def feature_transformation(data, selected_features):
    """
    Perform feature transformation: Standardization and Normalization.
    """
    print(f"Performing feature transformation on selected features: {selected_features}")
    
    # Standardization
    standard_scaler = StandardScaler()
    data[selected_features] = standard_scaler.fit_transform(data[selected_features])
    print(f"Features after standardization: {data[selected_features].head()}")

    # Normalization
    min_max_scaler = MinMaxScaler()
    data[selected_features] = min_max_scaler.fit_transform(data[selected_features])
    print(f"Features after normalization: {data[selected_features].head()}")

    return data


def add_date_features(data):
    """
    Add features extracted from the Date column: Year, Month, Day, and TimeStamp.
    """
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['TimeStamp'] = data['Date'].view('int64') / 1e9  # Convert to seconds since epoch as float
    return data

def add_lagged_features(data, lags=[1, 2, 3]):
    """
    Add lagged features for the Close price.
    """
    for lag in lags:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
    return data

def engineer_features(data):
    """
    Engineer all features: date-based, returns, technical indicators, and more.
    """
    print("Starting feature engineering...")
    data = add_date_features(data)
    data = calculate_vwap(data)
    data = calculate_moving_averages(data)
    data = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = calculate_macd(data)
    data = calculate_atr(data)
    data = calculate_adx(data)
    data = calculate_stochastic_oscillator(data)
    data = calculate_additional_features(data)
    data = add_lagged_features(data)
    data.dropna(inplace=True)  # Drop rows with NaN caused by rolling calculations
    print(f"Feature engineering complete. Final data shape: {data.shape}")
    return data


def prepare_data_for_model(data, target, horizon):
    """
    Prepare the data by selecting and transforming features.
    """
    print(f"Preparing data for the model. Target: {target}, Horizon: {horizon}")
    
    # Step 1: Engineer features
    print("Step 1: Engineering features...")
    data = engineer_features(data)

    # Step 2: Perform feature selection
    print("Step 2: Performing feature selection...")
    selected_features = feature_selection(data, target)

    # Step 3: Perform feature transformation
    print("Step 3: Performing feature transformation...")
    data = feature_transformation(data, selected_features)

    # Step 4: Prepare feature matrix and target variable
    print("Step 4: Preparing feature matrix and target variable...")
    X = data[selected_features]
    y = data[target].shift(-horizon)  # Shift target variable by horizon
    X = X.iloc[:-horizon]
    y = y.iloc[:-horizon]
    print(f"Prepared data shapes: Features - {X.shape}, Target - {y.shape}")

    return X, y


def split_data(X, y, test_size=0.2):
    """
    Split the data into training and test sets.
    """
    print(f"Splitting data into training and test sets with test size: {test_size}")
    split_index = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    print(f"Data split complete. Training set: {X_train.shape}, {y_train.shape}. Test set: {X_test.shape}, {y_test.shape}")
    return X_train, X_test, y_train, y_test