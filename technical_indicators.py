import pandas as pd
import talib
import yfinance as yf
import pickle
import telegram
import time
import requests
from bs4 import BeautifulSoup

# Define the URL for the EUR/USD pair
url = "https://www.investing.com/currencies/eur-usd"

# Define the headers for the HTTP request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

# Download the page content and parse it using BeautifulSoup
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

# Find the current price of the EUR/USD pair
price = float(soup.find('span', {'class': 'instrument-price_last__KQzyA'}).text)

# Download historical data for EUR/USD from Yahoo Finance
symbol = "EURUSD=X"
data = yf.download(symbol, start="2022-01-01", end="2022-05-13")

# Calculate technical indicators using talib
macd, macdsignal, macdhist = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
rsi = talib.RSI(data['Close'], timeperiod=14)
bbands_upper, bbands_middle, bbands_lower = talib.BBANDS(data['Close'], timeperiod=20)
slowk, slowd = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
adx = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)

# Load the machine learning models from saved files
with open('model_1.pkl', 'rb') as f:
    model_1 = pickle.load(f)
with open('model_2.pkl', 'rb') as f:
    model_2 = pickle.load(f)
with open('model_3.pkl', 'rb') as f:
    model_3 = pickle.load(f)

# Set up Telegram bot
bot = telegram.Bot(token='YOUR_TELEGRAM_BOT_TOKEN')
chat_id = 'YOUR_TELEGRAM_CHAT_ID'

# Set up risk management parameters
stop_loss = 0.01  # 1% stop loss per trade
position_size = 0.02  # 2% position size per trade
trailing_stop = 0.005  # 0.5% trailing stop

# Set up initial balance
balance = 10000.0

# Loop to continuously check the price and send recommendations
while True:
    # Download the page content and parse it using BeautifulSoup
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the current price of the EUR/USD pair
    new_price = float(soup.find('span', {'class': 'instrument-price_last__KQzyA'}).text)

    # Download historical data for EUR/USD from Yahoo Finance
    data = yf.download(symbol, start="2022-01-01", end="2022-05-13")

    # Calculate technical indicators using talib
    macd, macdsignal, macdhist = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    rsi = talib.RSI(data['Close'], timeperiod=14)
    bbands_upper, bbands_middle, bbands_lower = talib.BBANDS(data['Close'], timeperiod=20)
    slowk, slowd = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    adx = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)

    # Add more technical indicators
    rvi = talib.RVI(data['High'], data['Low'], data['Close'], timeperiod=14)
    macd_hist = macd - macdsignal
    chaikin = talib.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], fastperiod=3, slowperiod=10)
    cci = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)

    # Use sentiment analysis to inform trading decisions
    news_sentiment = get_news_sentiment(symbol)
    social_sentiment = get_social_sentiment(symbol)
    overall_sentiment = (news_sentiment + social_sentiment) / 2

    # Incorporate news events into trading strategy
    economic_events = get_economic_events()
    for event in economic_events:
        if event.currency == symbol[:3]:
            if event.impact == 'high':
                if event.actual > event.forecast:
                    overall_sentiment += 0.5
                else:
                    overall_sentiment -= 0.5
            elif event.impact == 'medium':
                if event.actual > event.forecast:
                    overall_sentiment += 0.3
                else:
                    overall_sentiment -= 0.3
            elif event.impact == 'low':
                if event.actual > event.forecast:
                    overall_sentiment += 0.1
                else:
                    overall_sentiment -= 0.1

    # Use machine learning to optimize parameters
    optimal_params = optimize_parameters(model_1, model_2, model_3, macd_hist, rsi, bbands_upper, bbands_middle, bbands_lower, slowk, slowd, adx, rvi, chaikin, cci, overall_sentiment)
    recommended_action = determine_action(new_price, data, optimal_params, stop_loss, position_size, trailing_stop, balance)

    # Send recommendation via Telegram
    bot.send_message(chat_id=chat_id, text=f"Current price: {new_price}\nRecommended action: {recommended_action}")

    # Sleep for 5 minutes before checking the price again
    time.sleep(300)