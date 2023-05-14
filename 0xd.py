import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, IchimokuIndicator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
import telegram

# Define the currency pair and the time interval
currency_pair = "EURUSD"
time_interval = "1d"

# Download historical data from Alpha Vantage API
api_key = "S3KGFDC61GY4IPA1"
url = f"https://www.alphavantage.co/query?function=FX_{currency_pair}&interval={time_interval}&apikey={api_key}"
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data["Time Series FX ({})".format(time_interval)]).T
df.columns = ["open", "high", "low", "close"]
df.index = pd.to_datetime(df.index)

# Calculate the Relative Strength Index (RSI)
rsi_window = 14
df["rsi"] = RSIIndicator(df["close"], window=rsi_window).rsi()

# Calculate the Moving Average (MA)
ma_window = 20
df["MA"] = df["close"].rolling(window=ma_window).mean()

# Calculate the Highs and Lows
bb_window = 10
bb_deviation = 2
df["upper_envelope"], df["lower_envelope"] = BollingerBands(df["close"], window=bb_window, window_dev=bb_deviation, window_type="sma").bollinger_hband(), BollingerBands(df["close"], window=bb_window, window_dev=bb_deviation, window_type="sma").bollinger_lband()
df["highs"] = df["high"].rolling(window=ma_window).max()
df["lows"] = df["low"].rolling(window=ma_window).min()

# Calculate the Fibonacci Retracement Levels
fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
price_range = df["high"].max() - df["low"].min()
for level in fib_levels:
    df[f"fib_{level}"] = df["high"] - (price_range * level)

# Determine the Trend
df["trend"] = np.where(df["MA"] > df["MA"].shift(1), 1, 0)

# Determine the Strength of the Trend
df["strength"] = np.where(df["trend"] == df["trend"].shift(1), df["strength"].shift(1) + 1, 1)

# Determine the Entry and Exit Points
df["entry"] = np.where(
    (df["trend"] == 1) &
    (df["rsi"] < 30) &
    (df["close"] < df["lows"].shift(1)) &
    (df["low"] > df["fib_0.618"]) &
    (df["low"] < df["fib_0.786"]) &
    (df["close"] < df["lower_envelope"]), 1, 0)
df["exit"] = np.where(
    (df["strength"] > 1) &
    (df["trend"] == 0) &
    (df["close"] > df["MA"]) &
    (df["high"] < df["fib_0.236"]) &
    (df["close"] > df["upper_envelope"]), 1, 0)

# Use Bollinger Bands to identify overbought and oversold conditions
df["bb_signal"] = np.where(df["close"] < df["lower_envelope"], 1, np.where(df["close"] > df["upper_envelope"], -1, 0))

# Use the Average Directional Index (ADX) to determine the strength of the trend
adx_window = 14
df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=adx_window).adx()

# Use the Ichimoku Cloud indicator to identify trend direction and momentum
ichimoku_window1 = 9
ichimoku_window2 = 26
ichimoku_window3 = 52
df["ichimoku_conv_line"] = IchimokuIndicator(df["high"], df["low"], window1=ichimoku_window1, window2=ichimoku_window2).ichimoku_conversion_line()
df["ichimoku_base_line"] = IchimokuIndicator(df["high"], df["low"], window1=ichimoku_window2, window2=ichimoku_window3).ichimoku_base_line()
df["ichimoku_a"] = IchimokuIndicator(df["high"], df["low"], window1=ichimoku_window1, window2=ichimoku_window2, window3=ichimoku_window3).ichimoku_a()
df["ichimoku_b"] = IchimokuIndicator(df["high"], df["low"], window1=ichimoku_window1, window2=ichimoku_window2, window3=ichimoku_window3).ichimoku_b()

# Use the Stochastic Oscillator to identify overbought and oversold conditions
stoch_window = 14
stoch_smooth_window = 3
df["stoch_k"] = StochasticOscillator(df["high"], df["low"], df["close"], window=stoch_window, smooth_window=stoch_smooth_window).stoch()
df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

# Use the Average True Range (ATR) to determine the volatility of the market
atr_window = 14
df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=atr_window).average_true_range()

# Use the On-Balance Volume (OBV) and Chaikin Money Flow (CMF) indicators to confirm trends
df["obv"] = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
df["cmf"] = ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"]).chaikin_money_flow()

# Create a Support Vector Machine (SVM) model to predict future price movements
svm_window = 14
svm = SVC()
X = df[["rsi", "bb_signal", "adx", "stoch_k", "stoch_d", "atr", "obv", "cmf"]]
y = np.where(df["close"].shift(-svm_window) > df["close"], 1, -1)
scores = cross_val_score(svm, X, y, cv=5)
accuracy = np.mean(scores)
svm.fit(X, y)
predicted_movement = svm.predict(X.iloc[[-1]])
if predicted_movement == 1:
    message = f"The {currency_pair} market is expected to go up."
else:
    message = f"The {currency_pair} market is expected to go down."

# Send a message through Telegram with the prediction
bot_token = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
bot_chatID = "1234567890"
bot = telegram.Bot(token=bot_token)
bot.sendMessage(chat_id=bot_chatID, text=message)

# Visualize the data
plt.plot(df.index, df["close"], label="Closing Price")
plt.plot(df.index, df["MA"], label="Moving Average")
plt.plot(df.index, df["upper_envelope"], label="Upper Envelope")
plt.plot(df.index, df["lower_envelope"], label="Lower Envelope")
plt.legend()
plt.show()
