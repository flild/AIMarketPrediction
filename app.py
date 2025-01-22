import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import font as tkfont
from tkcalendar import DateEntry
from ttkthemes import ThemedStyle
import threading
from datetime import datetime, timedelta
import numpy as np
from tinkoff.invest.utils import now
from api_client import fetch_gas_prices, fetch_gaz_profit
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
    COLORS = {
        'primary': '#2c3e50',
        'secondary': '#3498db',
        'success': '#27ae60',
        'danger': '#e74c3c',
        'light': '#ecf0f1',
        'dark': '#2c3e50'
    }
    FONTS = {
        'title': ('Helvetica', 16, 'bold'),
        'body': ('Arial', 10),
        'button': ('Arial', 10, 'bold')
    }

class GasPricePredictorApp:
    def __init__(self, master):
        self.master = master
        self.scaler = MinMaxScaler()
        self.model = None
        self.setup_style()
        self.setup_ui()

        self.running = False
        self.setup_date_pickers()
    def setup_style(self):
        self.style = ThemedStyle(self.master)
        self.style.set_theme('arc')
        self.style.configure('TButton', 
            font=Config.FONTS['button'],
            padding=6,
            foreground=Config.COLORS['dark']
        )
        self.style.configure('TFrame', background=Config.COLORS['light'])
        self.style.configure('TLabel', 
            background=Config.COLORS['light'],
            font=Config.FONTS['body']
        )
    def setup_date_pickers(self):
        self.date_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.date_frame.pack(pady=5)

        self.start_date_label = tk.Label(
            self.date_frame, 
            text="Начало периода:", 
            bg="#f0f0f0",
            font=tkfont.Font(family="Helvetica", size=10)
        )
        self.start_date_label.grid(row=0, column=0, padx=5)

        self.start_date = DateEntry(
            self.date_frame,
            date_pattern='yyyy-mm-dd',
            background='darkblue',
            foreground='white',
            borderwidth=2
        )
        self.start_date.grid(row=0, column=1, padx=5)
        self.start_date.set_date(now() - timedelta(days=1))

        self.end_date_label = tk.Label(
            self.date_frame, 
            text="Конец периода:", 
            bg="#f0f0f0",
            font=tkfont.Font(family="Helvetica", size=10)
        )
        self.end_date_label.grid(row=0, column=2, padx=5)

        self.end_date = DateEntry(
            self.date_frame,
            date_pattern='yyyy-mm-dd',
            background='darkblue',
            foreground='white',
            borderwidth=2
        )
        self.end_date.grid(row=0, column=3, padx=5)
        self.end_date.set_date(now())

    def setup_ui(self):
        self.master.title("Gas Price Prediction")
        self.master.geometry("600x400")
        self.master.configure(bg=Config.COLORS['light'])
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        # Header
        header_frame = ttk.Frame(self.master)
        header_frame.pack(pady=20, fill='x')
        
        title_label = ttk.Label(
            header_frame,
            text="Анализ цен на газ",
            font=Config.FONTS['title'],
            foreground=Config.COLORS['primary']
        )
        title_label.pack()

        # Main Content
        main_frame = ttk.Frame(self.master)
        main_frame.pack(pady=10, padx=20, fill='both', expand=True)


        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20, fill='x')

        self.chart_button = ttk.Button(
            button_frame,
            text="Построить прогноз",
            command=self.start_calculate_chart,
            style='primary.TButton'
        )
        self.chart_button.pack(side='left', padx=10, fill='x', expand=True)

        self.profit_button = ttk.Button(
            button_frame,
            text="Рассчитать прибыль",
            command=self.calculate_gaz_profit,
            style='success.TButton'
        )
        self.profit_button.pack(side='left', padx=10, fill='x', expand=True)

        # Лейбл результата
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill='x', pady=10)

        self.result_label = ttk.Label(
            result_frame,
            text="Прибыль: 0.00 ₽",
            font=Config.FONTS['body'],
            foreground=Config.COLORS['success'],
            anchor='center'
        )
        self.result_label.pack(fill='x')

        # Индикатор прогресса
        self.progress = ttk.Progressbar(
            main_frame,
            orient='horizontal',
            mode='indeterminate',
            length=400,
            style='success.Horizontal.TProgressbar'
        )
        # Custom Styles
        self.style.configure('primary.TButton', 
            background=Config.COLORS['secondary'],
            bordercolor=Config.COLORS['secondary']
        )
        self.style.configure('success.TButton', 
            background=Config.COLORS['success'],
            bordercolor=Config.COLORS['success']
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

    def calculate_gaz_profit(self):
        try:
            # Преобразуем date в datetime
            start = datetime.combine(self.start_date.get_date(), datetime.min.time())
            end = datetime.combine(self.end_date.get_date(), datetime.min.time()) + timedelta(days=1)
            
            if start > end:
                messagebox.showerror("Ошибка", "Некорректный период")
                return

            profit = fetch_gaz_profit(start, end)
            self.result_label.config(
                text=f"Прибыль за период: {profit:.2f} ₽",
                foreground=Config.COLORS['success']
            )
        except Exception as e:
            self.result_label.config(
                text="Ошибка расчета!",
                foreground=Config.COLORS['danger']
            )
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
