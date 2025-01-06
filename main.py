import yfinance as yf
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

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
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length, :]
        target = data[i+seq_length, target_col]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def main():
    # Fetch data
    gas_prices = fetch_gas_prices()
    if gas_prices.empty:
        return

    # Preprocess data
    gas_prices = preprocess_data(gas_prices)
    
    # Define features and target
    features = ['Gas_Price', 'Trade_Volume']
    target_col = 0  # Gas_Price column index
    
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(gas_prices[features])
    
    # Split data
    seq_length = 24
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    X_train, y_train = create_sequences(train_data, seq_length, target_col)
    X_test, y_test = create_sequences(test_data, seq_length, target_col)
    
    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], seq_length, len(features))
    X_test = X_test.reshape(X_test.shape[0], seq_length, len(features))
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, len(features))),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    # Predict
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(
        np.hstack((predictions, np.zeros((predictions.shape[0], len(features) - 1))))
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
    
    # Plot
    forecast_index = pd.date_range(start=gas_prices.index[-1], periods=24, freq='H')
    forecast_df = pd.DataFrame(forecast_rescaled, index=forecast_index, columns=['Predicted_Gas_Price'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(gas_prices.index[-168:], gas_prices['Gas_Price'][-168:], label='Actual Prices')
    plt.plot(forecast_df.index, forecast_df['Predicted_Gas_Price'], label='Forecast', linestyle='--')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()