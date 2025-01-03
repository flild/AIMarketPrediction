import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging
import matplotlib.pyplot as plt
from collections import deque

def fetch_gas_prices():
    gas_prices = yf.download("NG=F", period="1mo", interval="1h")[['Close', 'Volume']]
    gas_prices.rename(columns={'Close': 'Gas_Price', 'Volume': 'Trade_Volume'}, inplace=True)
    gas_prices.columns = ['Gas_Price', 'Trade_Volume']
    if gas_prices.empty:
        logging.error("No gas price data fetched.")
    return gas_prices

def fetch_oil_prices():
    oil_prices = yf.download("CL=F", period="1mo", interval="1h")[['Close']]
    oil_prices.rename(columns={'Close': 'Oil_Price'}, inplace=True)
    oil_prices.columns = ['Oil_Price']
    if oil_prices.empty:
        logging.error("No oil price data fetched.")
    return oil_prices

def fetch_currency_rates():
    currency_rates = yf.download("RUB=X", period="1mo", interval="1h")[['Close']]
    currency_rates.rename(columns={'Close': 'USD_RUB'}, inplace=True)
    currency_rates.columns = ['USD_RUB']
    if currency_rates.empty:
        logging.error("No currency rate data fetched.")
    return currency_rates

def add_lag_features(data, columns, lags):
    for col in columns:
        for lag in lags:
            data[f"{col}_lag{lag}"] = data[col].shift(lag)
    return data

def preprocess_data(gas_prices, oil_prices, currency_rates):
    data = gas_prices.join(oil_prices, how='outer').join(currency_rates, how='outer')
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    lag_columns = ['Gas_Price', 'Oil_Price', 'USD_RUB']
    data = add_lag_features(data, lag_columns, lags=[1, 3, 6])
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data.dropna(inplace=True)
    return data

def main():
    # Получаем данные о ценах на газ, нефть и курсы валют
    gas_prices = fetch_gas_prices()
    oil_prices = fetch_oil_prices()
    currency_rates = fetch_currency_rates()
    
    # Предобрабатываем данные
    data = preprocess_data(gas_prices, oil_prices, currency_rates)
    print("Размер предобработанных данных:", data.shape)
    
    # Определяем признаки и целевую переменную
    features = [col for col in data.columns if col != 'Gas_Price']
    target = 'Gas_Price'
    
    # Разделяем данные на обучающую и тестовую выборки
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print("Размер обучающей выборки:", train_data.shape)
    print("Размер тестовой выборки:", test_data.shape)
    
    # Проверяем, есть ли данные для обучения
    if len(train_data) == 0:
        logging.error("Нет данных для обучения.")
        return
    
    # Масштабируем данные
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data[features])
    test_scaled = scaler.transform(test_data[features])
    
    # Определяем целевые переменные для обучения и тестирования
    y_train = train_data[target]
    y_test = test_data[target]
    
    # Создаем и обучаем модель случайного леса
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_scaled, y_train)
    
    # Делаем предсказания на тестовых данных и вычисляем среднеквадратичную ошибку
    predictions = model.predict(test_scaled)
    mse = mean_squared_error(y_test, predictions)
    print("Предсказания на тестовых данных:", predictions)
    print("Среднеквадратичная ошибка на тестовых данных:", mse)
    logging.info(f"Среднеквадратичная ошибка на тестовых данных: {mse}")
    
    # Инициализируем исторические данные для лаговых признаков
    gas_history = deque(test_data['Gas_Price'].tail(6).tolist(), maxlen=6)
    oil_history = deque(test_data['Oil_Price'].tail(6).tolist(), maxlen=6)
    rub_history = deque(test_data['USD_RUB'].tail(6).tolist(), maxlen=6)
    
    # Прогнозируем на следующие 10 часов
    forecast = []
    last_data_point = test_data.iloc[-1].copy()
    
    for _ in range(10):
        # Обновляем лаговые признаки для Gas_Price
        if len(gas_history) >= 1:
            last_data_point['Gas_Price_lag1'] = gas_history[-1]
        if len(gas_history) >= 3:
            last_data_point['Gas_Price_lag3'] = gas_history[-3]
        if len(gas_history) >= 6:
            last_data_point['Gas_Price_lag6'] = gas_history[-6]
        
        # Обновляем лаговые признаки для Oil_Price
        if len(oil_history) >= 1:
            last_data_point['Oil_Price_lag1'] = oil_history[-1]
        if len(oil_history) >= 3:
            last_data_point['Oil_Price_lag3'] = oil_history[-3]
        if len(oil_history) >= 6:
            last_data_point['Oil_Price_lag6'] = oil_history[-6]
        
        # Обновляем лаговые признаки для USD_RUB
        if len(rub_history) >= 1:
            last_data_point['USD_RUB_lag1'] = rub_history[-1]
        if len(rub_history) >= 3:
            last_data_point['USD_RUB_lag3'] = rub_history[-3]
        if len(rub_history) >= 6:
            last_data_point['USD_RUB_lag6'] = rub_history[-6]
        
        # Обновляем временные признаки
        last_time = last_data_point.name
        next_time = last_time + pd.Timedelta(hours=1)
        last_data_point['hour'] = next_time.hour
        last_data_point['day_of_week'] = next_time.dayofweek
        last_data_point.name = next_time
        
        # Прогнозируем цену на газ на следующий час
        X_forecast = scaler.transform(last_data_point[features].values.reshape(1, -1))
        pred = model.predict(X_forecast)
        forecast.append(pred[0])
        
        # Обновляем исторические данные
        gas_history.append(pred[0])
        oil_history.append(oil_history[-1])
        rub_history.append(rub_history[-1])
        
        # Обновляем Gas_Price для следующей итерации
        last_data_point['Gas_Price'] = pred[0]
    
    # Создаем DataFrame для прогнозируемых цен
    forecast_index = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=10, freq='H')
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Predicted_Gas_Price'])
    
    # Строим график исторических и прогнозируемых цен на газ
    plt.figure(figsize=(12, 6))
    plt.plot(data['Gas_Price'], label='Исторические цены на газ')
    plt.plot(forecast_df['Predicted_Gas_Price'], label='Прогнозируемые цены на газ', linestyle='--')
    plt.title('Прогноз цен на газ на следующие 10 часов')
    plt.xlabel('Дата')
    plt.ylabel('Цена на газ')
    plt.legend()
    plt.show()
    
    # Выводим прогнозируемые цены
    print("Прогнозируемые цены на газ на следующие 10 часов:")
    print(forecast_df)

if __name__ == "__main__":
    # Настраиваем логирование
    logging.basicConfig(level=logging.INFO)
    main()
