import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


api_key = 'PKQFU4RVZL8BPWYUEC87'
api_secret = 'w18iFqsFi2dkGmteUJaVeWkAIBfFBgAFWt7loPSe'
base_url = 'https://paper-api.alpaca.markets'  # Use the paper trading URL or change to the live trading URL if you are ready for live trading.

# Initialize a session variable
aapl_bars_data = None

# No keys required for crypto data
crpto_client = CryptoHistoricalDataClient()
data_client = StockHistoricalDataClient(api_key, api_secret)

def fetch_aapl_data():
    global aapl_bars_data  # Referencing the global variable
    if aapl_bars_data is None:
        # Data not loaded yet, fetch and store it
        request_params = StockBarsRequest(
            symbol_or_symbols=["AAPL"],
            timeframe=TimeFrame.Day,
            start="2019-12-10T00:00:00Z",
            end="2023-12-28T00:00:00Z"
        )
        aapl_bars = data_client.get_stock_bars(request_params)
        aapl_bars_data = aapl_bars.df.reset_index()
    return aapl_bars_data

def create_windowed_dataset(dataset, window_size):
    dataX, dataY = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        dataX.append(a)
        dataY.append(dataset[i + window_size, 0])
    return np.array(dataX), np.array(dataY)

# Choose a window size
window_size = 60

# Convert to dataframe
df = fetch_aapl_data()

# Preprocess data (example: using only 'close' price)
dataset = df.filter(['close']).values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Assuming df is your DataFrame with columns like 'timestamp' and 'close'
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

X, y = create_windowed_dataset(scaled_data, window_size)

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
trainX, testX = X[:train_size], X[train_size:]
trainY, testY = y[:train_size], y[train_size:]

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(trainX, trainY, batch_size=1, epochs=1)

# Making predictions
predictions = model.predict(testX)
predictions = scaler.inverse_transform(predictions)

# Assuming 'testY' is your actual stock prices from the test set
# Note: You need to reshape testY and predicted_stock_prices if they are not in the same shape
testY_actual = scaler.inverse_transform(testY.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(testY_actual, predictions))
print("Root Mean Square Error (RMSE):", rmse)

plt.figure(figsize=(10, 6))
plt.plot(testY_actual, label='Actual Values', color='blue')
plt.plot(predictions, label='Predicted Values', color='red', linestyle='--')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
