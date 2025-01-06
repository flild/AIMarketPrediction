import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging

def fetch_gas_prices():
    gas_prices = yf.download("NG=F", period="1mo", interval="1h")[['Close', 'Volume']]
    gas_prices.rename(columns={'Close': 'Gas_Price', 'Volume': 'Trade_Volume'}, inplace=True)
    if gas_prices.empty:
        logging.error("No gas price data fetched.")
    return gas_prices

def preprocess_data(data):
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    return data


def create_sequences(data, seq_length, target_col):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length, target_col]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_results(gas_prices, predictions_rescaled, y_test_rescaled, forecast_rescaled, seq_length):
    test_index = gas_prices.index[-len(y_test_rescaled):]
    forecast_index = pd.date_range(start=test_index[-1], periods=24, freq='H')

    plt.figure(figsize=(14, 7))
    plt.plot(gas_prices.index[-seq_length*2:], gas_prices['Gas_Price'][-seq_length*2:], label='Actual Prices', color='blue')
    plt.plot(test_index, y_test_rescaled, label='True Test Prices', color='green')
    plt.plot(test_index, predictions_rescaled, label='Predicted Test Prices', color='red')
    plt.plot(forecast_index, forecast_rescaled, label='Forecast', linestyle='--', color='orange')
    plt.title('Gas Price Prediction and Forecast')
    plt.xlabel('Time')
    plt.ylabel('Gas Price')
    plt.legend()
    plt.show()

def main():
    # Fetch and preprocess data
    gas_prices = fetch_gas_prices()
    if gas_prices.empty:
        return

    gas_prices = preprocess_data(gas_prices)

    # Define features and target
    features = ['Gas_Price', 'Trade_Volume']
    target_col = 0  # Index of 'Gas_Price'

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(gas_prices[features])

    # Split into train, validation, and test sets
    seq_length = 24
    train_size = int(len(scaled_data) * 0.7)
    val_size = int(len(scaled_data) * 0.2)
    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:train_size + val_size]
    test_data = scaled_data[train_size + val_size:]

    X_train, y_train = create_sequences(train_data, seq_length, target_col)
    X_val, y_val = create_sequences(val_data, seq_length, target_col)
    X_test, y_test = create_sequences(test_data, seq_length, target_col)

    # Reshape for LSTM
    input_shape = (seq_length, len(features))
    X_train = X_train.reshape(-1, *input_shape)
    X_val = X_val.reshape(-1, *input_shape)
    X_test = X_test.reshape(-1, *input_shape)

    # Build and train model
    model = build_lstm_model(input_shape)
    model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=30, batch_size=32, verbose=1
    )

    # Predict
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], len(features)- 1))))
    )[:, 0]
    y_test_rescaled = scaler.inverse_transform(
        np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(features) - 1))))
    )[:, 0]

    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    print(f"Mean Squared Error: {mse}")

    # Forecast for next 24 hours
    forecast = []
    last_seq = test_data[-seq_length:].reshape(1, seq_length, len(features))
    for _ in range(24):
        pred = model.predict(last_seq)
        forecast.append(pred[0, 0])
        new_seq = np.hstack((pred, last_seq[0, -1, 1:].reshape(1, -1)))
        last_seq = np.vstack((last_seq[0, 1:], new_seq)).reshape(1, seq_length, len(features))

    forecast_rescaled = scaler.inverse_transform(
        np.hstack((np.array(forecast).reshape(-1, 1), np.zeros((24, len(features) - 1))))
    )[:, 0]

    # Plot results
    plot_results(gas_prices, predictions_rescaled, y_test_rescaled, forecast_rescaled, seq_length)

if __name__ == "__main__":
    main()