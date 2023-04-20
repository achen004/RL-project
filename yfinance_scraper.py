# Script for scraping data from Yahoo Finance
import yfinance as yf

SPY = yf.Ticker("SPY")

period = "1y"
interval = "1h"

data = SPY.history(period=period, interval=interval, start="2021-04-14")
print(data.head())

data.to_csv("SPY_prices_{}_{}.csv".format(period, interval))