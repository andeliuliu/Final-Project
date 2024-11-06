# Stock Price Prediction Model Midterm Report 

## Presentation Link
[View the presentation on Google Slides](https://docs.google.com/presentation/d/1ldVlhIxLv9Ws7QH0tb8xTBQSvn2ezauIiC9Rqwx9vVc/edit#slide=id.g3122b41325a_0_88)

## Overview
This project uses the XGBoost algorithm to predict stock prices based on historical data and technical indicators. The model focuses on structured data and is designed to capture complex relationships within time-series data for more accurate short-term predictions.

## Features Used for Prediction

The following features were selected due to their known impact on stock prices:
- **Moving Averages**: 10-day and 50-day intervals help smooth out price trends.
- **RSI (Relative Strength Index)**: Measures stock momentum to determine if it’s overbought or oversold.
- **Bollinger Bands**: Provides insight into the volatility of the stock.
- **Trading Volume**: Indicates the level of activity in the stock, potentially signaling trends.

Each feature contributes a unique perspective on stock behavior, giving the model a well-rounded view of past trends for future price prediction.

## Modeling Method

We used the `XGBRegressor` model from the XGBoost library, chosen for its ability to capture non-linear relationships and feature interactions, which is well-suited for time-series predictions.

To ensure robustness:
- We applied an 80/20 train-test split.
- Used cross-validation to prevent overfitting.
- Leveraged features such as previous closing prices, moving averages, and other technical indicators.

## Model Evaluation and Insights

Performance was visualized by plotting both actual and predicted prices on a time-series line graph, allowing us to see where the model aligns closely with real values and where it diverges, highlighting potential areas for improvement.

### Example Results

- **Apple (AAPL)**: Achieved an RMSE of 23.15 for a 1-day prediction horizon. However, as the horizon increases (e.g., 1 week or 1 month), the RMSE rises, and the predicted and actual price lines diverge, indicating reduced accuracy over longer time frames.
- **Tesla (TSLA)**: The model accurately captures short-term trends, but similarly to AAPL, longer-term predictions exhibit increased RMSE.

This increase in RMSE over longer horizons is due to the model's limited capacity to account for unpredictable external factors, such as market news and economic changes, which influence stock prices but aren’t captured in purely technical data.

## Services and Utilities

### Services (`services.py`)

- **train_model**: Trains an XGBoost regression model with early stopping based on test performance, setting a maximum of 1000 trees and a learning rate of 0.01.
- **evaluate_model**: Generates predictions and calculates Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
- **predict_stock_price**: Fetches and processes stock data, trains and evaluates the model, and returns key metrics, predictions, and a forecasted next-day closing price.

### Utilities (`utils.py`)

- **get_stock_data**: Fetches historical stock data for a given symbol and period, resetting the index for easier manipulation.
- **calculate_upside_downside**: Computes the upside/downside percentage, indicating potential gain or loss.
- **get_recommendation**: Provides a buy, sell, or hold recommendation based on upside/downside percentage.
- **calculate_moving_averages**: Adds moving averages over specified windows to the dataset.
- **calculate_rsi**: Calculates the RSI over a specified period for stock momentum.
- **calculate_bollinger_bands**: Computes Bollinger Bands for volatility assessment.
- **engineer_features**: Adds technical indicators to the dataset, preparing it for model input by dropping empty values.
- **prepare_data_for_model**: Prepares features (X) and target (y) by shifting the target for the specified horizon and removing empty rows.
- **split_data**: Splits features and target data into training and testing sets based on a specified test size.

## Challenges and Improvements

### Challenges
Managing the model's Root Mean Squared Error (RMSE) was a primary challenge, as reducing RMSE would improve the model's reliability.

### Potential Improvements
1. **Feature Engineering**: Incorporate additional features, such as economic indicators or sentiment analysis from news headlines, to capture external factors affecting stock prices.
2. **Hyperparameter Tuning**: Run extensive grid searches to optimize parameters, aiming to reduce RMSE and enhance prediction stability.

---

By addressing these improvements, we hope to further increase the model's accuracy and robustness for stock price predictions.
