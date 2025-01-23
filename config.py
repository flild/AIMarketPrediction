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