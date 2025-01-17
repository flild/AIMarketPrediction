from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.utils import now
from datetime import timedelta
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import os

# Загружаем переменные из .env файла
load_dotenv()

# Ваш токен доступа Tinkoff Invest API
TOKEN = os.getenv('TINKOF_API')

# FIGI фьючерса на газ (пример)
GAS_FIGI = "FUTNG0125000"

def fetch_gas_prices():
    """
    Получает данные по фьючерсу на газ (цены и объемы) через Tinkoff API.
    """
    with Client(TOKEN) as client:
        response = client.market_data.get_candles(
            figi=GAS_FIGI,
            from_=now() - timedelta(days=30),  # Данные за последние 30 дней
            to=now(),
            interval=CandleInterval.CANDLE_INTERVAL_HOUR  # Часовые свечи
        )

        # Собираем данные в DataFrame
        data = []
        for candle in response.candles:
            data.append({
                "time": candle.time,
                "Gas_Price": float(candle.close.units + candle.close.nano / 1e9),  # Цена закрытия
                "Trade_Volume": candle.volume  # Объем торгов
            })

        gas_prices = pd.DataFrame(data)
        gas_prices.set_index("time", inplace=True)

        if gas_prices.empty:
            logging.error("No gas price data fetched.")
        return gas_prices

def calculate_indicators(df):
    """
    Рассчитывает индикаторы: Stochastic RSI и MACD.
    """
    # Stochastic RSI
    df.ta.stochrsi(append=True)

    # MACD
    df.ta.macd(append=True)

    return df

def save_to_csv(df, filename="gas_data.csv"):
    """
    Сохраняет данные в CSV-файл.
    """
    df.to_csv(filename, index=True)
    print(f"Данные сохранены в файл: {filename}")


if __name__ == "__main__":
    main()