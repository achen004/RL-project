# Script for scraping data from Yahoo Finance
import pandas as pd
import yfinance as yf

SPY = yf.Ticker("SPY")

period = "1y"
interval = "1h"

data = SPY.history(period=period, interval=interval, start="2021-04-14")
print(data.head())

data.to_csv("SPY_prices_{}_{}.csv".format(period, interval))

tickers = pd.read_csv("constituents_csv.csv")

for tick in tickers["Symbol"]:
    ticker = yf.Ticker(tick)
    data = ticker.history(period=period, interval=interval)
    data.to_csv(f"data/{tick}.csv")