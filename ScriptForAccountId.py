import requests
import os
from dotenv import load_dotenv
# Загружаем переменные из .env файла
load_dotenv()
# Ваш токен
token = os.getenv('TINKOF_API')

# Заголовки запроса
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Эндпоинт для получения счетов
url = "https://invest-public-api.tinkoff.ru/rest/tinkoff.public.invest.api.contract.v1.UsersService/GetAccounts"

# Отправка запроса
response = requests.post(url, headers=headers, json={})
accounts = response.json()["accounts"]

# Вывод идентификаторов счетов
for account in accounts:
    print(f"ID счета: {account['id']}, Тип: {account['type']}, Название: {account['name']}")