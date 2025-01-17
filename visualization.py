import matplotlib.pyplot as plt
import pandas as pd

def plot_results_with_forecast(gas_prices, predictions_rescaled, y_test_rescaled, forecast_rescaled, seq_length, future_forecast):
    test_index = gas_prices.index[-len(y_test_rescaled):]
    future_index = pd.date_range(start=test_index[-1], periods=len(future_forecast) + 1, freq='h')[1:]  # Заменил 'H' на 'h'

    plt.figure(figsize=(14, 7))

    # Фактические цены за последние 30 дней
    plt.plot(gas_prices.index[-seq_length*2:], gas_prices['Gas_Price'][-seq_length*2:], label='Actual Prices', color='blue')

    # Проверка, чтобы не было ошибки с пустыми данными
    if len(y_test_rescaled) > 0 and len(predictions_rescaled) > 0:
        plt.plot(test_index, y_test_rescaled, label='Test Predictions', color='green')
        plt.plot(test_index, predictions_rescaled, label='Predicted Test Prices', color='red')

    # Прогноз на 10 часов
    plt.plot(future_index, future_forecast, label='Future Forecast (10h)', linestyle='--', color='purple')

    plt.xlabel('Time')
    plt.ylabel('Gas Price')
    plt.title('Gas Price Prediction and 10-Hour Forecast')
    plt.legend()
    plt.grid(True)
    plt.show()