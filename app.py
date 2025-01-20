import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import font as tkfont
import numpy as np
import threading
from api_client import fetch_gas_prices, fetch_today_profit
from model import preprocess_data, build_lstm_model, create_sequences, forecast_future_prices
from visualization import plot_results_with_forecast
from sklearn.preprocessing import MinMaxScaler

# Конфигурационные параметры
class Config:
    SEQ_LENGTH = 24
    TRAIN_TEST_RATIO = 0.7
    EPOCHS = 30
    BATCH_SIZE = 32
    FORECAST_HOURS = 10
    FEATURES = ['Gas_Price', 'Trade_Volume']
    TARGET_FEATURE = 'Gas_Price'

class GasPricePredictorApp:
    def __init__(self, master):
        self.master = master
        self.scaler = MinMaxScaler()
        self.model = None
        self.setup_ui()
        self.running = False

    def setup_ui(self):
        self.master.title("Gas Price Prediction")
        self.master.geometry("400x350")
        self.master.configure(bg="#f0f0f0")
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        custom_font = tkfont.Font(family="Helvetica", size=12, weight="bold")

        # Кнопки
        self.chart_button = self.create_button(
            "Просчитать график", 
            self.start_calculate_chart,
            "#4CAF50"
        )
        self.profit_button = self.create_button(
            "Посчитать сделки", 
            self.calculate_profit,
            "#2196F3"
        )

        # Лейбл результата
        self.result_label = tk.Label(
            self.master,
            text="Прибыль: 0 ₽",
            bg="#f0f0f0",
            fg="#333333",
            font=custom_font
        )
        self.result_label.pack(pady=10)

        # Индикатор прогресса
        self.progress = ttk.Progressbar(
            self.master,
            orient='horizontal',
            mode='indeterminate',
            length=280
        )

    def create_button(self, text, command, color):
        button = tk.Button(
            self.master,
            text=text,
            command=command,
            bg=color,
            fg="white",
            font=tkfont.Font(family="Helvetica", size=12, weight="bold"),
            padx=20,
            pady=10,
            borderwidth=0,
            relief="flat",
            state=tk.NORMAL
        )
        button.pack(pady=10)
        return button

    def start_calculate_chart(self):
        if not self.running:
            self.running = True
            self.progress.pack(pady=5)
            self.progress.start()
            self.chart_button.config(state=tk.DISABLED)
            threading.Thread(target=self.calculate_chart).start()

    def rescale_values(self, values, feature_index):
        dummy_data = np.zeros((len(values), len(Config.FEATURES)))
        dummy_data[:, feature_index] = values.ravel()
        return self.scaler.inverse_transform(dummy_data)[:, feature_index]

    def handle_error(self, message):
        self.master.after(0, lambda: messagebox.showerror("Ошибка", message))
        self.reset_ui()

    def reset_ui(self):
        self.master.after(0, lambda: self.progress.stop())
        self.master.after(0, lambda: self.progress.pack_forget())
        self.master.after(0, lambda: self.chart_button.config(state=tk.NORMAL))
        self.running = False

    def calculate_chart(self):
        try:
            gas_prices = fetch_gas_prices()
            if gas_prices.empty:
                self.handle_error("Не удалось получить данные о ценах")
                return

            gas_prices = preprocess_data(gas_prices)
            if not all(f in gas_prices.columns for f in Config.FEATURES):
                self.handle_error("Отсутствуют необходимые колонки данных")
                return

            # Масштабирование данных
            scaled_data = self.scaler.fit_transform(gas_prices[Config.FEATURES])

            # Создание последовательностей
            train_size = int(len(scaled_data) * Config.TRAIN_TEST_RATIO)
            test_data = scaled_data[train_size:]

            X_train, y_train = create_sequences(scaled_data[:train_size], Config.SEQ_LENGTH, 0)
            X_test, y_test = create_sequences(test_data, Config.SEQ_LENGTH, 0)

            # Построение и обучение модели
            self.model = build_lstm_model((Config.SEQ_LENGTH, len(Config.FEATURES)))
            self.model.fit(
                X_train, 
                y_train, 
                epochs=Config.EPOCHS, 
                batch_size=Config.BATCH_SIZE, 
                verbose=0,
                validation_split=0.1
            )

            # Предсказание
            predictions = self.model.predict(X_test)
            target_index = Config.FEATURES.index(Config.TARGET_FEATURE)
            
            predictions_rescaled = self.rescale_values(predictions, target_index)
            y_test_rescaled = self.rescale_values(y_test, target_index)

            # Прогнозирование
            last_seq = scaled_data[-Config.SEQ_LENGTH:].reshape(1, Config.SEQ_LENGTH, len(Config.FEATURES))
            future_forecast = forecast_future_prices(
                self.model,
                last_seq,
                self.scaler,
                Config.FEATURES,
                Config.SEQ_LENGTH,
                hours=Config.FORECAST_HOURS
            )

            # Визуализация
            self.master.after(0, lambda: plot_results_with_forecast(
                gas_prices,
                predictions_rescaled,
                y_test_rescaled,
                [],
                Config.SEQ_LENGTH,
                future_forecast
            ))

        except Exception as e:
            self.handle_error(f"Ошибка при расчете: {str(e)}")
        finally:
            self.reset_ui()

    def calculate_profit(self):
        try:
            profit = fetch_today_profit()
            self.result_label.config(text=f"Прибыль за сегодня: {profit:.2f} ₽")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось получить прибыль: {str(e)}")

    def on_close(self):
        if self.running:
            messagebox.showinfo("Информация", "Пожалуйста, дождитесь завершения операции")
        else:
            self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GasPricePredictorApp(root)
    root.mainloop()