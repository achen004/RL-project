# Script for scraping data from Yahoo Finance
import pandas as pd
import yfinance as yf
import pandas as pd

def get_data(ticker, period, interval):
    ticker = yf.Ticker(ticker)
    data = ticker.history(period=period, interval=interval)
    return data

stocks = pd.read_csv("spy_constituents.csv")
tickers = stocks[stocks["Sector"] == "Information Technology"]["Symbol"].tolist()

success_count = 0
for tick in tickers:
    period = "1y"
    interval = "1h"
    try:
        data = get_data(tick, period, interval)
        print(f"Got data for {tick} with shape {data.shape}")
        data.to_csv(f"data/{tick}.csv")
        success_count += 1
    except:
        print(f"Error with {tick}")

print(f"{success_count} tickers retrieved successfully")
