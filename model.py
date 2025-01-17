import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def forecast_future_prices(model, last_sequence, scaler, features, seq_length, hours=10):
    forecast = []
    current_seq = last_sequence.copy()

    for _ in range(hours):
        pred = model.predict(current_seq)
        forecast.append(pred[0, 0])
        new_seq = np.hstack((pred, current_seq[0, -1, 1:].reshape(1, -1)))
        current_seq = np.vstack((current_seq[0, 1:], new_seq)).reshape(1, seq_length, len(features))

    forecast_rescaled = scaler.inverse_transform(
        np.hstack((np.array(forecast).reshape(-1, 1), np.zeros((hours, len(features)-1))))
    )[:, 0]

    return forecast_rescaled