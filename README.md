# Reinforcement Learning Stock Trading Agent

This project contains a reinforcement learning (RL) agent designed to trade stocks in a custom stock trading environment. The agent is trained using historical stock data with the goal of maximizing profit. The code provided in `main.py` demonstrates how to train the agent, execute trades, and visualize the results.

Additionally, this project includes a script for scraping stock data from Yahoo Finance (`yfinance_scraper.py`) and an example of an ensemble approach for trading using multiple RL algorithms (`Ensemble.py`).

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Scraping Stock Data](#scraping)
5. [Ensemble Trading](#ensemble)
6. [Customizing the Training](#customizing)
7. [License](#license)

## Requirements

- Python 3.x
- pandas
- torch
- ta (Technical Analysis Library)
- numpy
- gym-anytrading
- stable-baselines3
- enum
- yfinance

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/rl-stock-trading-agent.git
    cd rl-stock-trading-agent
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your stock data in CSV format with columns: `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`.

2. Train the agent and visualize the results using the provided `main.py` script:
    ```
    python main.py --stock path/to/your/stock_data.csv
    ```

3. The training results and performance visualization will be displayed upon completion.

## Scraping Stock Data

To scrape stock data from Yahoo Finance, use the `yfinance_scraper.py` script. The script saves stock data for Information Technology companies in the S&P 500 index in the `data/` directory.

To run the script:

```
python yfinance_scraper.py
```

## Ensemble Trading

The `Ensemble.py` script demonstrates an ensemble approach for stock trading using multiple reinforcement learning algorithms: PPO, A2C, and DQN. The script trains the models and performs trading based on a majority vote of their actions.

To run the ensemble script:

```
python Ensemble.py --stock path/to/your/stock_data.csv
```

## Customizing the Training

You can modify the training parameters by passing additional arguments to `main.py` or `Ensemble.py`. These arguments should be defined in the `parse_args` function located in the `parse_args.py` module.

Example:

```
python main.py --stock path/to/your/stock_data.csv --window-size 60 --epochs 150 --hidden-nodes 240 --max-shares 200 --learning-rate 5e-4 --use-lr-scheduler False
```