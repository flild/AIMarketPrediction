import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from textblob import TextBlob
import asyncio
import aiohttp
import logging
import os
from dotenv import load_dotenv


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
load_dotenv()

# 1. Асинхронный сбор данных 
async def fetch_data_async(url, session):
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        else:
            logging.error(f"Ошибка при получении данных: {response.status}")
            return None

# 2. Синхронные функции для сбора данных
def fetch_gas_prices():
    return yf.download("NG=F", period="1mo", interval="1h")[['Close', 'Volume']].rename(
        columns={'Close': 'Gas_Price', 'Volume': 'Trade_Volume'})

def fetch_oil_prices():
    return yf.download("CL=F", period="1mo", interval="1h")[['Close']].rename(columns={'Close': 'Oil_Price'})

def fetch_currency_rates():
    return yf.download("RUB=X", period="1mo", interval="1h")[['Close']].rename(columns={'Close': 'USD_RUB'})

# 3. Обработка данных
def add_lag_features(data, columns, lags):
    for col in columns:
        for lag in lags:
            data[f"{col}_lag{lag}"] = data[col].shift(lag)
    return data

def preprocess_data(gas_prices, oil_prices, currency_rates):
    # Объединение данных
    data = gas_prices.join(oil_prices).join(currency_rates)

    # Добавление временных признаков
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek

    # Добавление лаговых переменных
    lag_columns = ['Gas_Price', 'Oil_Price', 'USD_RUB']
    data = add_lag_features(data, lag_columns, lags=[1, 3, 6])

    # Заполнение пропусков
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)

    # Нормализация
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    return data_scaled

# Основная функция
async def main():
    # Сбор данных
    gas_prices = fetch_gas_prices()
    oil_prices = fetch_oil_prices()
    currency_rates = fetch_currency_rates()

    processed_data = preprocess_data(gas_prices, oil_prices, currency_rates)

    # Сохранение данных
    processed_data.to_csv("processed_gas_data.csv")
    logging.info("Данные успешно собраны и сохранены.")

if __name__ == "__main__":
    asyncio.run(main())