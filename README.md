# Lord Productions
This is a Python script for a trading bot that uses various technical indicators and trading strategies to backtest and trade on the Binance exchange. It uses the CCXT library to connect to the exchange, backtrader for backtesting, and various other Python libraries for technical analysis, sentiment analysis, and visualization.

The script starts by defining configuration variables, command-line arguments, and initializing the Telegram bot, CCXT exchange, and SQLite database. It then parses the command-line arguments to get the trading symbol, timeframes, and trading strategy.

Based on the strategy selected by the user, the script defines a backtrader strategy and adds it to the cerebro engine. It then fetches the historical OHLCV data for the trading symbol and timeframes from the exchange, adds the data to the cerebro engine, and runs the engine to backtest the strategy.

After the backtesting is complete, the script calculates the trading results, sends a message to the Telegram bot with the results, and saves the results to the SQLite database.

The script also includes functions to get Reddit sentiment data, plot stock data and technical indicators, and make stock price predictions using machine learning.

Finally, the script executes the trading bot by calling the `run_trading_bot()` function, which runs the backtesting, plots the stock data and technical indicators, makes stock price predictions, and performs sentiment analysis.
