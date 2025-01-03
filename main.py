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

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# 1. Асинхронный сбор данных
async def fetch_data_async(url, session):
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        else:
            logging.error(f"Ошибка при получении данных: {response.status}")
            return None

async def fetch_weather_data_async(api_key, location):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}&units=metric"
    async with aiohttp.ClientSession() as session:
        return await fetch_data_async(url, session)

async def fetch_news_async(api_key, query="natural gas", language="en", sort_by="publishedAt", page_size=50):
    url = (f"https://newsapi.org/v2/everything?q={query}&language={language}"
           f"&sortBy={sort_by}&pageSize={page_size}&apiKey={api_key}")
    async with aiohttp.ClientSession() as session:
        return await fetch_data_async(url, session)

# 2. Синхронные функции для сбора данных
def fetch_gas_prices():
    return yf.download("NG=F", period="1mo", interval="1h")[['Close', 'Volume']].rename(
        columns={'Close': 'Gas_Price', 'Volume': 'Trade_Volume'})

def fetch_oil_prices():
    return yf.download("CL=F", period="1mo", interval="1h")[['Close']].rename(columns={'Close': 'Oil_Price'})

def fetch_currency_rates():
    return yf.download("RUB=X", period="1mo", interval="1h")[['Close']].rename(columns={'Close': 'USD_RUB'})

# 3. Обработка данных
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def add_lag_features(data, columns, lags):
    for col in columns:
        for lag in lags:
            data[f"{col}_lag{lag}"] = data[col].shift(lag)
    return data

def preprocess_data(gas_prices, weather_data, oil_prices, currency_rates, news):
    # Объединение данных
    data = gas_prices.join(oil_prices).join(currency_rates)

    # Обработка погодных данных
    weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
    weather_data.set_index('timestamp', inplace=True)
    data = data.join(weather_data)

    # Обработка новостей
    news_df = pd.DataFrame(news)
    news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
    news_df.set_index('publishedAt', inplace=True)
    news_df['sentiment'] = news_df['title'].apply(analyze_sentiment)
    news_df = news_df.resample('1H').mean().ffill()
    data = data.join(news_df[['sentiment']], how='left')

    # Добавление временных признаков
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek

    # Добавление лаговых переменных
    lag_columns = ['Gas_Price', 'Oil_Price', 'USD_RUB', 'sentiment']
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
    api_key_weather = "your_openweathermap_api_key"
    api_key_news = "your_newsapi_key"
    location = "Moscow, RU"

    # Сбор данных
    gas_prices = fetch_gas_prices()
    oil_prices = fetch_oil_prices()
    currency_rates = fetch_currency_rates()

    weather_data_future = fetch_weather_data_async(api_key_weather, location)
    news_future = fetch_news_async(api_key_news)

    weather_data = await weather_data_future
    news = await news_future

    weather_data_df = pd.DataFrame([{
        'timestamp': entry['dt_txt'],
        'temperature': entry['main']['temp'],
        'humidity': entry['main']['humidity']
    } for entry in weather_data['list']])

    processed_data = preprocess_data(
        gas_prices, weather_data_df, oil_prices, currency_rates, news
    )

    # Сохранение данных
    processed_data.to_csv("processed_gas_data.csv")
    logging.info("Данные успешно собраны и сохранены.")

if __name__ == "__main__":
    asyncio.run(main())