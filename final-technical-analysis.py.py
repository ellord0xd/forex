import ccxt
import telegram
import pandas as pd
import yfinance as yf
import configparser
import argparse
import backtrader as bt
import sqlite3
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from ta.momentum import RSIIndicator, StochasticOscillator  
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, IchimokuIndicator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
import requests
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pickle
import time
from bs4 import BeautifulSoup

# Define configuration variables  
config = configparser.ConfigParser()
config.read('config.ini')   

# Define command-line arguments
parser = argparse.ArgumentParser(description='Trading Bot')    
parser.add_argument('--symbol', type=str, required=True, help='The trading symbol to use')
parser.add_argument('--timeframes', type=str, required=True, help='The timeframes to use, separated by commas')   
parser.add_argument('--strategy', type=str, required=True, help='The trading strategy to use')
args = parser.parse_args()

# Define Telegram bot
bot = telegram.Bot(token=config['TELEGRAM']['ACCESS_TOKEN'])
chat_id = config['TELEGRAM']['CHAT_ID']

# Define CCXT exchange
exchange = ccxt.binance({
    'apiKey': config['BINANCE']['API_KEY'],
    'secret': config['BINANCE']['SECRET_KEY'],
    'enableRateLimit': True
})

# Define SQLite database
conn = sqlite3.connect('tradingbot.db')
c = conn.cursor()

# Define trading symbol and timeframes
symbol = args.symbol
timeframes = args.timeframes.split(',')

# Define trading strategy
if args.strategy == 'rsi':
    class RSIStrategy(bt.Strategy):
        params = (('rsi_period', 14), ('rsi_upper', 70), ('rsi_lower', 30), ('sma_period', 20), ('tp_sl_ratio', 3))

        def __init__(self):
            self.rsi = RSIIndicator(self.data.close, self.params.rsi_period)
            self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_period)
            self.tp_sl_ratio = self.params.tp_sl_ratio

        def next(self):
            if self.position.size == 0:
                if self.rsi < self.params.rsi_lower:
                    self.buy(size=self.broker.getcash() // self.data.close * self.tp_sl_ratio)
            elif self.position.size > 0:
                if self.rsi > self.params.rsi_upper:
                    self.close()

    cerebro = bt.Cerebro()
    cerebro.addstrategy(RSIStrategy)
elif args.strategy == 'stochastic':
    class StochasticStrategy(bt.Strategy):
        params = (('stoch_period', 14), ('stoch_upper', 80), ('stoch_lower', 20), ('sma_period', 20), ('tp_sl_ratio', 3))

        def __init__(self):
            self.stoch = StochasticOscillator(self.data.high, self.data.low, self.data.close, self.params.stoch_period)
            self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_period)
            self.tp_sl_ratio = self.params.tp_sl_ratio

        def next(self):
            if self.position.size == 0:
                if self.stoch.oscillator_k < self.params.stoch_lower:
                    self.buy(size=self.broker.getcash() // self.data.close * self.tp_sl_ratio)
            elif self.position.size > 0:
                if self.stoch.oscillator_k > self.params.stoch_upper:
                    self.close()

    cerebro = bt.Cerebro()
    cerebro.addstrategy(StochasticStrategy)
else:
    bot.send_message(chat_id=chat_id, text='Invalid strategy')
    exit()

# Define data feed
data = {}
for timeframe in timeframes:
    data[timeframe] = bt.feeds.PandasData(dataname=exchange.fetch_ohlcv(symbol, timeframe))

# Add data feed to cerebro
for timeframe in timeframes:
    cerebro.adddata(data[timeframe])

# Set broker settings
cerebro.broker.setcash(1000)
cerebro.broker.setcommission(commission=0.001)

# Run cerebro engine
cerebro.run()

# Get trading results
strat = cerebro.runstrats[0][0]
pnl = cerebro.broker.getvalue() - cerebro.broker.getcash()
percent_pnl = pnl / cerebro.broker.getcash() * 100

# Print trading results
print('PnL: ${:.2f}'.format(pnl))
print('Percent PnL: {:.2f}%'.format(percent_pnl))
bot.send_message(chat_id=chat_id, text='PnL: ${:.2f}\nPercent PnL: {:.2f}%'.format(pnl, percent_pnl))

# Save trading results to SQLite database
c.execute('CREATE TABLE IF NOT EXISTS trading_results (symbol TEXT, timeframes TEXT, strategy TEXT, pnl REAL, percent_pnl REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
c.execute('INSERT INTO trading_results (symbol, timeframes, strategy, pnl, percent_pnl) VALUES (?, ?, ?, ?, ?)', (symbol, args.timeframes, args.strategy, pnl, percent_pnl))
conn.commit()

# Close database connection
conn.close()
# Get Reddit sentiment data
    titles = get_reddit_sentiment_data()

    # Load SVM model and vectorizer
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Predict sentiment
    sentiment = predict_sentiment(svm_model, titles)

    # Send Telegram message with sentiment
    bot.send_message(chat_id=chat_id, text='Reddit sentiment for {}: {}'.format(symbol, sentiment[0]))

# Define function to plot stock data
def plot_stock_data(symbol, start_date, end_date):
    data = get_stock_data(symbol, start_date, end_date)
    plt.plot(data)
    plt.title(symbol)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Define function to plot technical indicators
def plot_technical_indicators(data):
    rsi, stoch, bb, atr, adx, ichimoku, obv, cmf = calculate_indicators(data)
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))
    axs[0, 0].plot(data)
    axs[0, 0].set_title('Price')
    axs[0, 1].plot(rsi)
    axs[0, 1].axhline(y=30, color='r', linestyle='-')
    axs[0, 1].axhline(y=70, color='r', linestyle='-')
    axs[0, 1].set_title('RSI')
    axs[1, 0].plot(stoch.oscillator_k)
    axs[1, 0].axhline(y=20, color='r', linestyle='-')
    axs[1, 0].axhline(y=80, color='r', linestyle='-')
    axs[1, 0].set_title('Stochastic Oscillator %K')
    axs[1, 1].plot(bb.bollinger_mavg)
    axs[1, 1].fill_between(data.index, bb.bollinger_hband, bb.bollinger_lband, alpha=0.1)
    axs[1, 1].set_title('Bollinger Bands')
    axs[2, 0].plot(atr.average_true_range)
    axs[2, 0].set_title('Average True Range')
    axs[2, 1].plot(adx.adx)
    axs[2, 1].axhline(y=25, color='r', linestyle='-')
    axs[2, 1].set_title('ADX')
    axs[3, 0].plot(ichimoku.ichimoku_a)
    axs[3, 0].plot(ichimoku.ichimoku_b)
    axs[3, 0].fill_between(data.index, ichimoku.ichimoku_a, ichimoku.ichimoku_b, alpha=0.1)
    axs[3, 0].set_title('Ichimoku Cloud')
    axs[3, 1].plot(obv.on_balance_volume)
    axs[3, 1].set_title('On Balance Volume')
    plt.tight_layout()
    plt.show()

# Define function to run the trading bot
def run_trading_bot():
    # Get command-line arguments
    symbol = args.symbol
    timeframes = args.timeframes.split(',')
    strategy = args.strategy

    # Run trading bot
    run_backtesting(symbol, timeframes, strategy)
    plot_stock_data(symbol, '2020-01-01', '2022-05-01')
    plot_technical_indicators(data[symbol]['2020-01-01':'2022-05-01'])
    make_stock_predictions(symbol, '2020-01-01', '2022-05-01')
    run_sentiment_analysis()

# Execute trading bot
run_trading_bot()