# Stock Price Prediction Model Midterm Report

## Presentation Link
[View the report on Google Slides with visualizations of data](https://docs.google.com/presentation/d/1ldVlhIxLv9Ws7QH0tb8xTBQSvn2ezauIiC9Rqwx9vVc/edit#slide=id.g3122b41325a_0_88)

## Demo Video Link
[Demo Video](https://drive.google.com/file/d/17jLHJ15lHiLnFOFMQxTxZ178IyBICTaJ/view?usp=sharing)

## Overview
This report presents a comprehensive study on predicting stock prices using two advanced models: **XGBoost**, a machine learning algorithm, and **SARIMA**, a time-series statistical model. The primary goal is to understand short-term price movements while incorporating long-term seasonal trends. We leverage extensive data exploration, advanced feature engineering, hyperparameter tuning, and a robust evaluation framework. By comparing the strengths and weaknesses of these models, we aim to identify their suitability for financial forecasting and propose methods for further improvement.

---

## 1. Introduction
Stock price prediction remains one of the most challenging tasks in financial analytics due to market volatility, non-linearity, and the influence of unpredictable external factors. Accurate forecasting can provide traders and investors with critical insights, enabling informed decision-making and risk management.

This project combines machine learning and time-series modeling techniques:
1. **XGBoost** is well-suited for capturing non-linear relationships and interactions between features.
2. **SARIMA** specializes in modeling seasonality and trends in time-series data.

The report focuses on:
- Data exploration and feature selection.
- Model design and tuning.
- Performance evaluation and comparison.
- Insights into the impact of prediction horizons.

Additionally, the report discusses improvements made over a prior midterm implementation and outlines future enhancements.

---

## 2. Data Exploration

### 2.1 Data Collection
The dataset comprises two years of daily stock price data for companies like Apple (AAPL) and Tesla (TSLA), sourced using the `yfinance` API. The data includes:
- **Date**: Timestamps of observations.
- **Open, High, Low, Close Prices**: Intraday price data.
- **Volume**: Number of shares traded.

### 2.2 Key Observations
1. **Seasonality**: Periodic patterns, such as monthly trends, were observed in many stocks, making SARIMA an appropriate model for capturing these effects.
2. **Volatility**: Stocks like Tesla exhibited high intraday variations, requiring models that can handle noise and non-linearity.
3. **Price Trends**: Stocks displayed consistent trends over shorter intervals, making technical indicators like Moving Averages highly relevant.
4. **Volume Trends**: Peaks in trading volume often coincided with significant price movements, reinforcing its importance as a feature.

### 2.3 Challenges
1. **Noise**: Stock prices are influenced by external factors (e.g., news, economic policies) that are difficult to model directly.
2. **Non-Linearity**: Complex interactions between variables necessitate advanced algorithms like XGBoost.
3. **Temporal Dependencies**: Accurate predictions require models that account for past values, such as SARIMA and lagged features.

---

## 3. Feature Engineering and Selection

Feature engineering plays a pivotal role in improving model performance by incorporating domain knowledge into the dataset. Below are the features used and the reasoning for their inclusion:

### 3.1 Core Variables
1. **Close Price (Target Variable)**:
   - Represents the final price at which the stock was traded each day.
   - Chosen as the target variable due to its significance in financial decision-making.
2. **Open, High, Low Prices**:
   - Provide insights into intraday price behavior.
   - Contribute to indicators like Bollinger Bands and ATR.
3. **Volume**:
   - Indicates market activity, often correlated with price trends.
   - High trading volume typically precedes significant price movements.

### 3.2 Technical Indicators
Technical indicators are mathematical calculations based on price, volume, or a combination of both. They help identify trends, momentum, and volatility.

1. **VWAP (Volume Weighted Average Price)**:
   - Formula: `(Close × Volume).cumsum() / Volume.cumsum()`
   - Importance: Reflects the average price weighted by volume, commonly used by institutional traders.
2. **Moving Averages (10-day, 50-day)**:
   - Capture short-term (10-day) and long-term (50-day) price trends.
   - Useful for identifying trend reversals when short- and long-term averages intersect (e.g., golden/death cross).
3. **RSI (Relative Strength Index)**:
   - Formula: `100 - (100 / (1 + RS))`, where `RS = Avg Gain / Avg Loss`.
   - Importance: Measures momentum, identifying overbought (>70) or oversold (<30) conditions.
4. **Bollinger Bands**:
   - Upper Band = Rolling Mean + 2 × Rolling Std
   - Lower Band = Rolling Mean - 2 × Rolling Std
   - Importance: Highlights price deviations and volatility.
5. **MACD (Moving Average Convergence Divergence)**:
   - Tracks the difference between short-term (12-day) and long-term (26-day) exponential moving averages.
   - Importance: Identifies trend shifts.
6. **ATR (Average True Range)**:
   - Measures daily price volatility based on the range of high, low, and close prices.
   - Importance: Used for risk management.
7. **ADX (Average Directional Index)**:
   - Evaluates trend strength; values >25 indicate strong trends.
8. **Stochastic Oscillator**:
   - Formula: `(Close - Lowest Low) / (Highest High - Lowest Low) × 100`
   - Importance: Tracks momentum by comparing closing prices to historical ranges.

### 3.3 Lagged and Derived Features
1. **Lagged Close Prices (1-day, 2-day, 3-day)**:
   - Introduce temporal dependencies for recursive forecasting.
2. **Daily Variation**:
   - Formula: `(High - Low) / Open`
   - Importance: Reflects market volatility.
3. **High-Close and Low-Open Ratios**:
   - Capture intraday price pressure, indicating upward/downward trends.

---

## Services (`services.py`)

- **train_model**: Trains an XGBoost regression model with early stopping based on test performance, setting a maximum of 1000 trees and a learning rate of 0.01.
- **evaluate_model**: Generates predictions and calculates Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
- **predict_stock_price**: Fetches and processes stock data, trains and evaluates the model, and returns key metrics, predictions, and a forecasted next-day closing price.

---

## Utilities (`utils.py`)

- **get_stock_data**: Fetches historical stock data for a given symbol and period, resetting the index for easier manipulation.
- **calculate_upside_downside**: Computes the upside/downside percentage, indicating potential gain or loss.
- **get_recommendation**: Provides a buy, sell, or hold recommendation based on upside/downside percentage.
- **calculate_moving_averages**: Adds moving averages over specified windows to the dataset.
- **calculate_rsi**: Calculates the RSI over a specified period for stock momentum.
- **calculate_bollinger_bands**: Computes Bollinger Bands for volatility assessment.
- **engineer_features**: Adds technical indicators to the dataset, preparing it for model input by dropping empty values.
- **prepare_data_for_model**: Prepares features (X) and target (y) by shifting the target for the specified horizon and removing empty rows.
- **split_data**: Splits features and target data into training and testing sets based on a specified test size.

---

## 4. Model Selection

### 4.1 XGBoost
XGBoost was chosen for its ability to handle:
1. **Non-Linear Interactions**: Captures relationships among multiple technical indicators.
2. **High-Dimensional Data**: Efficiently processes numerous features.
3. **Regularization**: Reduces overfitting through parameters like `reg_alpha` and `reg_lambda`.

### 4.2 SARIMA
SARIMA is ideal for:
1. **Seasonality and Trends**: Models recurring patterns like monthly price fluctuations.
2. **Explainability**: Provides interpretable components, such as trend and seasonal effects.

### 4.3 Ensemble Approach
The models were combined to leverage their complementary strengths:
- **XGBoost**: Captures short-term, non-linear dynamics.
- **SARIMA**: Focuses on long-term trends and seasonality.

---

## 5. Model Tuning

### 5.1 XGBoost
- **Learning Rate**: Set to 0.01 for gradual convergence.
- **Max Depth**: Tuned to 6 for a balance between complexity and overfitting.
- **Subsample**: Set to 80% to improve generalization.
- **Early Stopping**: Stops training after 10 rounds without improvement.

### 5.2 SARIMA
- **Order**: (1, 1, 1) for non-seasonal components.
- **Seasonal Order**: (1, 1, 1, 12) for monthly seasonality.

---

## 6. Model Validation

### 6.1 Metrics Used
1. **Root Mean Squared Error (RMSE)**:
   - Measures prediction error.
2. **Relative RMSE (% of Close Price Range)**:
   - Contextualizes RMSE based on price volatility.
3. **Relative RMSE (% of Average Close Price)**:
   - Allows comparisons across stocks with different price levels.

### 6.2 Validation Approach
- **Train-Test Split**: 80/20 split for model evaluation.
- **Time-Series Validation**: Ensures models are tested on unseen future data.
- **Cross-Validation**: Reduces overfitting and improves generalization.

---

## 7. Results

### 1. Accuracy vs. Prediction Horizon:
- For shorter prediction horizons (1 day), the RMSE is relatively low, indicating that the model can capture short-term trends effectively.
- As the prediction horizon increases (e.g., 1 week or 1 month), the RMSE increases significantly, reflecting the model’s struggle to predict long-term trends accurately. This is consistent with financial market unpredictability over extended periods.

### 2. Actual vs. Predicted Trends:
- The actual prices (blue lines) show significant variability and volatility in the data, which is characteristic of stock price movements.
- Predicted prices (red lines) are smoother, indicating the model’s limitations in capturing sharp, sudden price changes influenced by external factors like news or market events.

### 3. Evaluation Metrics:
- **RMSE**: Provides an absolute measure of the prediction error, which is particularly useful for comparing performance across stocks with similar price ranges.
- **RMSE as % of Close Price Range**: Contextualizes the RMSE relative to the stock’s volatility, highlighting how well the model adapts to stocks with different price variances.
- **RMSE as % of Average Close Price**: Allows cross-comparison between stocks of varying price levels.

---

# 1. AAPL (Apple)

- **Horizon**: 1 Day
- **RMSE**: 6.50
- **RMSE as % of Close Price Range**: 6.32%
- **RMSE as % of Average Close Price**: 3.41%
- **Next Day Prediction**: $228.20 (Recommendation: Hold)

### Analysis:
- The model performs reasonably well for Apple, with a low RMSE indicating accurate short-term predictions.
- The “Hold” recommendation aligns with minimal deviation in predicted vs. actual prices.
- The model successfully tracks overall trends but underestimates short-term spikes or dips.

---

# 2. MSFT (Microsoft)

- **Horizon**:
  - **1 Day**: RMSE = 7.06, RMSE as % of Close Price Range = 3.17%, RMSE as % of Average Close Price = 1.90%.
  - **1 Week**: RMSE = 9.91, RMSE as % of Close Price Range = 4.45%, RMSE as % of Average Close Price = 2.67%.
- **Next Day Prediction (1 Day Horizon)**: $432.80 (Recommendation: Strong Buy)
- **Next Day Prediction (1 Week Horizon)**: $424.24 (Recommendation: Strong Buy)

### Analysis:
- The model shows strong performance with low RMSE percentages for both 1-day and 1-week horizons, reflecting its ability to predict short- to medium-term movements for Microsoft accurately.
- The recommendation of “Strong Buy” is backed by the upward trend in predicted prices, aligning with actual price movements.
- The 1-week horizon demonstrates slightly higher error due to increased market uncertainty over time.

---

# Insights and Possible Improvements

## 1. Short-Term vs. Long-Term Predictions:
- The model performs well for short-term predictions (1 day) across all stocks, with low RMSE percentages indicating accurate predictions.
- For longer horizons (1 week, 1 month), performance degrades due to the model’s inability to capture long-term volatility and external factors.

## 2. Stock-Specific Challenges:
- Stocks like NVIDIA and Coca-Cola exhibit higher volatility or long-term trend shifts, leading to higher RMSE percentages for their predictions.
- Stable stocks like Johnson & Johnson show consistently lower RMSE, making them more suitable for the current model.

## 3. Feature Engineering Improvements:
- Incorporate additional features like macroeconomic indicators (e.g., interest rates) or sentiment analysis to improve long-term prediction accuracy.
- Enhance the model to capture sudden price changes by integrating news-based events or other external signals.

## 4. Model Adjustments:
- For longer horizons, explore ensemble approaches combining SARIMA with deep learning models (e.g., LSTMs) to better handle sequential data and volatility.
- Refine hyperparameters for specific stocks with higher volatility.

---

### Overall:
The model demonstrates strong predictive power for short-term horizons, particularly for stable stocks, but further improvements are necessary to handle long-term forecasts and volatile stock behaviors effectively.

---

## 8. Comparison of Midterm and Final Models

The transition from the midterm to the final model marks a significant improvement in terms of data processing, feature engineering, model design, evaluation, and overall predictive performance. This section provides a detailed comparison of the two models and highlights the key improvements.

### 8.1 Feature Engineering
**Midterm Model**:
- Limited Features:
  - Used basic technical indicators such as simple Moving Averages and Relative Strength Index (RSI).
  - Focused solely on price-based features, ignoring volume, volatility, and momentum indicators.
- No Advanced Derived Features:
  - Did not include lagged variables or features derived from intraday price movements, such as Daily Variation or High-Close Ratios.
- No Recursive Features:
  - No consideration for the temporal dependencies required for recursive forecasting.

**Final Model**:
- Extensive Feature Set:
  - Integrated advanced technical indicators, including Bollinger Bands, MACD, VWAP, ATR, ADX, and the Stochastic Oscillator.
  - Captured volatility and trend strength, which are crucial for market prediction.
- Lagged Features:
  - Introduced lagged variables (Close_Lag_1, Close_Lag_2, Close_Lag_3) to capture temporal dependencies and enhance recursive predictions.
- Derived Features:
  - Added custom metrics like Daily Variation, High-Close Ratio, and Low-Open Ratio, which provide nuanced insights into intraday price behavior.
- Volume-Weighted Features:
  - Included VWAP to reflect institutional trading behavior and its impact on price movements.

**Impact**:
The expanded feature set significantly enhanced the final model’s ability to capture non-linear relationships and long-term trends, making it more robust across different stocks and market conditions.

### 8.2 Model Tuning
**Midterm Model**:
- Minimal Hyperparameter Tuning:
  - Used default parameters for XGBoost, resulting in suboptimal performance.
  - No systematic exploration of key hyperparameters like learning rate, maximum depth, or subsampling ratios.
- SARIMA Absent:
  - Did not include any time-series model to handle seasonality or trends.

**Final Model**:
- Systematic Hyperparameter Tuning for XGBoost:
  - **Learning Rate**: Tuned to 0.01 for gradual optimization, preventing overfitting while improving convergence.
  - **Maximum Depth**: Optimized at 6 to balance complexity and overfitting.
  - **Subsample and Column Subsample**: Set to 80% to enhance generalization.
  - **Early Stopping**: Incorporated early stopping rounds to prevent overfitting.
- SARIMA Added:
  - Configured with `order = (1, 1, 1)` and `seasonal_order = (1, 1, 1, 12)` to capture monthly seasonality and long-term trends.
  - Differenced data to achieve stationarity and align with SARIMA’s assumptions.

**Impact**:
The tuning process improved model accuracy and generalization, while the inclusion of SARIMA enabled the capture of long-term patterns that were previously ignored.

### 8.3 Validation Approach
**Midterm Model**:
- Basic Validation:
  - Relied on simple train-test splits without considering the temporal structure of the data.
  - No cross-validation or out-of-sample testing, increasing the risk of overfitting.

**Final Model**:
- Robust Validation Framework:
  - **Time-Series Split**: Ensured that the training and testing data were chronologically separated to reflect real-world forecasting scenarios.
  - **Cross-Validation**: Applied k-fold cross-validation to minimize overfitting and assess performance consistency across different subsets of data.
  - **Multiple Metrics**:
    - RMSE to measure error magnitude.
    - RMSE as a percentage of Close Price Range to contextualize performance relative to stock volatility.
    - RMSE as a percentage of Average Close Price to enable cross-stock comparisons.

**Impact**:
The improved validation approach provided a more accurate assessment of the models’ real-world performance, enhancing confidence in the predictions.

### 8.4 Model Design
**Midterm Model**:
- Single Model:
  - Relied solely on XGBoost, limiting its ability to capture long-term trends or seasonal effects.
- Simplistic Architecture:
  - Did not incorporate ensemble techniques or consider the unique strengths of different models.

**Final Model**:
- Hybrid Model Architecture:
  - Combined XGBoost for short-term, non-linear dynamics and SARIMA for long-term seasonal trends.
- Ensemble Approach:
  - Weighted predictions from XGBoost (70%) and SARIMA (30%) to balance short- and long-term forecasting accuracy.

**Impact**:
The hybrid architecture improved flexibility and performance, allowing the model to adapt to different market conditions and prediction horizons.

---

## 9. Assumptions and Decisions
1. **Stationarity**: Assumed SARIMA’s data could be differenced to achieve stationarity.
2. **Feature Selection**: Chosen based on domain knowledge and statistical relevance.
3. **Prediction Horizon**: Focused on short-term accuracy due to increased uncertainty in long-term forecasts.

---

## 10. Improvements and Future Work
1. **Feature Expansion**:
   - Incorporate sentiment analysis from news and social media.
   - Add macroeconomic variables (e.g., interest rates, GDP growth).
2. **Advanced Models**:
   - Explore RNNs and Transformers for sequential forecasting.
3. **Real-Time Integration**:
   - Deploy models in real-time trading systems for dynamic predictions.

---

### Challenges
Managing the model's Root Mean Squared Error (RMSE) was a primary challenge, as reducing RMSE would improve the model's reliability.

### Potential Improvements
1. **Feature Engineering**: Incorporate additional features, such as economic indicators or sentiment analysis from news headlines, to capture external factors affecting stock prices.
2. **Hyperparameter Tuning**: Run extensive grid searches to optimize parameters, aiming to reduce RMSE and enhance prediction stability.

---

By addressing these improvements, we hope to further increase the model's accuracy and robustness for stock price predictions.
