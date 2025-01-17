# Gas Price Prediction Using LSTM

### Description
This project predicts natural gas futures prices using an LSTM (Long Short-Term Memory) neural network.  
The model is trained on historical gas price and trade volume data fetched from Tinkoff Invest API.  
It forecasts future gas prices for the next **10 hours** and visualizes the predictions alongside actual prices for the last **30 days**.  
Additionally, the app calculates daily trading profits based on recent operations.

### Example:
![image](https://github.com/user-attachments/assets/f3d5d168-e226-43ff-b0f5-1e17bf9b31a9)

---

### Table of Contents

- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Data](#data)  
- [Model](#model)  
- [Results](#results)  
- [Contributing](#contributing)  

---

### Features

- Fetches real-time gas price data from **Tinkoff Invest API**.  
- Preprocesses data to handle missing values.  
- Builds and trains an **LSTM model** for time series prediction.  
- Predicts gas prices for the next **10 hours**.  
- Visualizes model predictions and forecasts:  
  - **30-day historical data**  
  - **2-day test predictions**  
  - **10-hour forecast**  
- Calculates **daily trading profit** and percentage based on recent gas trades.

---

### Requirements

- Python 3.11 or higher  
- tinkoff-investments  
- numpy  
- pandas  
- scikit-learn  
- matplotlib  
- tensorflow  
- python-dotenv  
- tkinter  

---

### Installation

1. **Clone the repository:**  
bash
git clone https://github.com/flild/AIMarketPrediction.git
cd AIMarketPrediction


2. **Install the required libraries:**  
bash
pip install -r requirements.txt


3. **Create a `.env` file** in the project directory and enter your Tinkoff API token and account ID:  
bash
TINKOF_API="your_token"
TIN_ACC_ID="your_account_id"


4. **Find your account ID** using the script:  
bash
python ScriptForAccountId.py


---

### Usage

1. **Run the application:**  
bash
python app.py


2. **App Features:**  
   - **"Просчитать график"** – fetches data, trains the model, predicts prices, and displays the 30-day chart with predictions.  
   - **"Посчитать сделки"** – calculates today's profit from gas trades in rubles and percentage.

---

### Data

- **Source:** Tinkoff Invest API  
- **Attributes:**  
  - `Gas_Price`: Closing price of natural gas futures.  
  - `Trade_Volume`: Volume of trades.  

---

### Model

- **Architecture:** Sequential **LSTM model** with two LSTM layers and dropout regularization.  
- **Loss Function:** Mean Squared Error (MSE).  
- **Optimizer:** Adam optimizer.  
- **Training:** Trained for **30 epochs** with a batch size of **32**.  
- **Prediction:**  
  - Predicts prices for the last **2 days** (test data).  
  - Forecasts gas prices for the next **10 hours**.  

---

### Results

- **Visualizations include:**  
  - Actual gas prices for the last **30 days**.  
  - **2-day test predictions** compared with actual prices.  
  - Forecasted gas prices for the next **10 hours**.  

- **Profit Calculation:**  
  - Displays the daily trading profit in rubles and percentage.  

- **Model Performance:**  
  - Mean Squared Error (MSE) is printed after evaluation.

---

### Contributing

Contributions are welcome!  
Please open an issue or submit a pull request.

---

### Future Improvements

- Extend forecast beyond **10 hours**.  
- Add support for other commodities.  
- Optimize the model for better accuracy.


---
