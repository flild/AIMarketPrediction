import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from api_client import fetch_gas_prices,fetch_today_profit
from model import preprocess_data, build_lstm_model, create_sequences, forecast_future_prices
from visualization import plot_results_with_forecast
from sklearn.preprocessing import MinMaxScaler

def calculate_chart():
    gas_prices = fetch_gas_prices()
    if gas_prices.empty:
        return

    gas_prices = preprocess_data(gas_prices)
    features = ['Gas_Price', 'Trade_Volume']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(gas_prices[features])

    seq_length = 24
    train_size = int(len(scaled_data) * 0.7)
    test_data = scaled_data[train_size:]

    X_train, y_train = create_sequences(scaled_data[:train_size], seq_length, 0)
    X_test, y_test = create_sequences(test_data, seq_length, 0)

    model = build_lstm_model((seq_length, len(features)))
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(
        np.hstack((predictions, np.zeros((predictions.shape[0], len(features)-1))))
    )[:, 0]

    y_test_rescaled = scaler.inverse_transform(
        np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(features)-1))))
    )[:, 0]

    # Прогноз на 10 часов
    last_seq = scaled_data[-seq_length:].reshape(1, seq_length, len(features))
    future_forecast = forecast_future_prices(model, last_seq, scaler, features, seq_length, hours=10)

    # Передаем реальные данные в график
    plot_results_with_forecast(gas_prices, predictions_rescaled, y_test_rescaled, [], seq_length, future_forecast)

def calculate_profit():
    profit = fetch_today_profit()
    result_label.config(text=f"Прибыль за сегодня: {profit:.2f} ₽")

# Создание окна
window = tk.Tk()
window.title("Gas Price Prediction")
window.geometry("400x300")
window.configure(bg="#f0f0f0")  # Устанавливаем цвет фона окна

# Шрифты
custom_font = tkfont.Font(family="Helvetica", size=12, weight="bold")

# Кнопка для прогноза графика
chart_button = tk.Button(
    window, 
    text="Просчитать график", 
    command=calculate_chart,
    bg="#4CAF50",  # Зеленый цвет фона
    fg="white",    # Белый цвет текста
    font=custom_font,
    padx=20,
    pady=10,
    borderwidth=0,
    relief="flat"
)
chart_button.pack(pady=20)

# Кнопка для расчета прибыли
profit_button = tk.Button(
    window, 
    text="Посчитать сделки", 
    command=calculate_profit,
    bg="#2196F3",  # Синий цвет фона
    fg="white",    # Белый цвет текста
    font=custom_font,
    padx=20,
    pady=10,
    borderwidth=0,
    relief="flat"
)
profit_button.pack(pady=20)

# Лейбл для вывода прибыли
result_label = tk.Label(
    window, 
    text="Прибыль: 0 ₽", 
    bg="#f0f0f0",  # Цвет фона как у окна
    fg="#333333",  # Темно-серый цвет текста
    font=custom_font
)
result_label.pack(pady=20)

# Запуск приложения
window.mainloop()