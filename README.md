Gas Price Prediction Using LSTM
Description
This project predicts natural gas futures prices using an LSTM (Long Short-Term Memory) neural network. The model is trained on historical gas price data and volume data fetched from Yahoo Finance. It then forecasts future gas prices for the next 24 hours and visualizes the predictions alongside actual prices.

Table of Contents
Features

Requirements

Installation

Usage

Data

Model

Results

Contributing

License

Contact

Features
Fetches real-time gas price data.

Preprocesses data to handle missing values.

Builds and trains an LSTM model for time series prediction.

Generates a 24-hour forecast of gas prices.

Visualizes model predictions and forecasts.

Requirements
Python 3.7 or higher

yfinance

numpy

pandas

scikit-learn

matplotlib

tensorflow

Installation
Clone the repository:

bash
Copy
git clone https://github.com/yourusername/gas-price-prediction.git
Install the required libraries:

bash
Copy
pip install -r requirements.txt
(Note: Create a requirements.txt file with the listed libraries if not provided.)

Usage
Run the script:

bash
Copy
python main.py
The script will fetch data, train the model, make predictions, and display a plot of the results.

Data
Source: Yahoo Finance for natural gas futures (NG=F).

Attributes:

Gas_Price: Closing price of natural gas futures.

Trade_Volume: Volume of trades.

Model
Architecture: Sequential LSTM model with two LSTM layers and dropout regularization.

Loss Function: Mean Squared Error (MSE).

Optimizer: Adam optimizer.

Training: Trained for 30 epochs with a batch size of 32.

Results
The model predicts gas prices based on historical data.

A plot is generated showing:

Actual gas prices for the last 48 hours.

True test prices for the most recent data.

Predicted test prices.

Forecast for the next 24 hours.

The Mean Squared Error (MSE) of the predictions is printed.

Contributing
Contributions are welcome! Please open an issue or submit a pull request.

