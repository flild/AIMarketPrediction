from tinkoff.invest import Client, CandleInterval, OperationState
from tinkoff.invest.utils import now
from datetime import timedelta
import pandas as pd
import os
from dotenv import load_dotenv
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Загружаем переменные из .env файла
load_dotenv()
TOKEN = os.getenv('TINKOF_API')
tin_acc_id = os.getenv('TIN_ACC_ID')
GAS_FIGI = "FUTNGM012500"

def fetch_gas_prices():
    try:
        with Client(TOKEN) as client:
            response = client.market_data.get_candles(
                figi=GAS_FIGI,
                from_=now() - timedelta(days=30),
                to=now(),
                interval=CandleInterval.CANDLE_INTERVAL_HOUR
            )

            data = [{
                "time": candle.time,
                "Gas_Price": candle.close.units + candle.close.nano * 1e-9,
                "Trade_Volume": candle.volume
            } for candle in response.candles]

            gas_prices = pd.DataFrame(data)
            gas_prices.set_index("time", inplace=True)
            return gas_prices
    except Exception as e:
        logging.error(f"Error fetching gas prices: {e}")
        return pd.DataFrame()

def fetch_today_profit():
    try:
        with Client(TOKEN) as client:
            operations = client.operations.get_operations(
                account_id=tin_acc_id,
                figi=GAS_FIGI,
                from_=now() - timedelta(days=1),
                to=now(),
                state=OperationState.OPERATION_STATE_EXECUTED
            )
            
            # Преобразуем MoneyValue в рубли и суммируем
            profit = sum(op.payment.units + op.payment.nano / 1e9 for op in operations.operations)
            return profit
    except Exception as e:
        logging.error(f"Error fetching profit data: {e}")
        return 0