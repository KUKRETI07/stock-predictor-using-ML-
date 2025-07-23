## Stock Predictor: Bitcoin Price Forecasting

# Overview
This project implements an LSTM-based deep learning model to forecast Bitcoin prices using historical market data.
The model analyzes technical indicators and volume patterns to predict future closing prices,
providing a 10-day forecast to help investors make more informed decisions.

# Problem Statement
Predicting cryptocurrency prices is challenging due to:

* Extreme market volatility

* Complex non-linear patterns

* Multiple influencing factors (market sentiment, news, regulations)

* Lack of fundamental valuation metrics

* Traditional technical analysis methods often fail to capture these complexities, creating a need for more sophisticated forecasting approaches.

# Solution
 I developed a deep learning solution that:

1. Uses historical price/volume data from Yahoo Finance (5,000+ Bitcoin records)

2. Incorporates technical indicators as predictive features

3. Implements a Long Short-Term Memory (LSTM) neural network

4. Generates 10-day price forecasts

# Key Features Used
* Closing price (primary target)

* Trading volume

* 50-day moving average

* 200-day moving average

# Technical Implementation

A[Yahoo Finance API] --> B[Raw Data]
B --> C[Preprocessing]
C --> D[Feature Engineering]
D --> E[Technical Indicators]
E --> F[Train/Test Split]
F --> G[Scaling]
G --> H[LSTM Model]

# Feature Engineering
1. Calculated 50-day and 200-day moving averages

2. Normalized volume using Min-Max scaling

3. Created sequential time windows (150 time steps) for LSTM input

# Model Architecture
model = Sequential()
model.add(LSTM(150, return_sequences=True, input_shape=(150, 4)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# Training Process
1. Dataset split: 70% training, 30% testing

2. Time-series cross-validation to prevent lookahead bias

3. MinMaxScaler (0-1 normalization) applied separately to train/test sets

4. 100 training epochs with early stopping

5. Adam optimizer with Mean Absolute Error loss

# Model Performance
The model achieved strong predictive accuracy:

Metric	Value 
* MAE	0.0506
* R²	0.9402
* RMSE	0.0751
