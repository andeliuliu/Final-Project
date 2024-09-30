# Predicting Stock Prices Using XGBoost

## Description of the Project
The goal of this project is to predict daily stock prices using the **XGBoost** algorithm, which is a gradient boosting technique known for its high predictive accuracy. We aim to utilize historical stock data to predict the closing price for a selected stock. This project will introduce us to advanced machine learning methods, data collection, feature engineering, and predictive modeling.

## Clear Goal(s)
Our main goal is to accurately predict the next day’s closing price of a selected stock (e.g., Apple, Tesla) using historical price data and technical indicators, applying **XGBoost** for better predictive performance.

## Data to Be Collected and How You Will Collect It
### Data Needs
- **Historical Stock Price Data**: Includes daily opening, closing, high, and low prices, as well as trading volume.
- **Technical Indicators**: Features such as moving averages (e.g., 10-day and 50-day moving averages), **RSI (Relative Strength Index)**, and **Bollinger Bands** will be engineered from the collected data.

### Collection Method
- We will use the **Yahoo Finance API** or **Alpha Vantage API** to collect historical price data.
- **Technical indicators** will be derived from this data using Python libraries like `Pandas`.

## How You Plan on Modeling the Data
### Model Type
We plan to use **XGBoost** (`XGBRegressor`), which is a powerful boosting algorithm suitable for time series prediction. **XGBoost** will allow us to capture complex relationships in the stock price data by creating an ensemble of decision trees.

### Features
The model will use features such as:
- Previous closing prices
- Moving averages (e.g., 10-day, 50-day)
- RSI (Relative Strength Index)
- Trading volume

These features will be used to improve prediction accuracy.

## How Do You Plan on Visualizing the Data?
- **Stock Price Trends**: We will use **line plots** to visualize historical and predicted stock prices over time, allowing us to evaluate the model’s performance visually.
- **Feature Importance**: XGBoost provides **feature importance scores**, which we will visualize using **bar plots** to show which features contribute most to the model’s predictions.
- **Residual Analysis**: We will use **scatter plots** to analyze the residuals (errors between predicted and actual prices) and identify any potential biases in the model.

## What is Your Test Plan?
### Data Splitting
- We will withhold **20% of the data for testing**, and use the remaining **80%** for training the model.
- We will apply **cross-validation** during the training phase to reduce overfitting and ensure the model generalizes well.

### Evaluation Metrics
- We will use **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** to evaluate model accuracy.
- Additionally, we will use **backtesting** to evaluate how well the model would have performed in predicting stock prices over a historical time period, simulating real-world performance.
