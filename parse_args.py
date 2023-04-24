import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent to trade a stock.")

    # The path to the CSV file containing the prices for the stock we want to trade
    parser.add_argument("--stock", type=str, default="SPY_prices_2y_1h.csv")

    # Number of days of past stock prices for the agent to look at when it's deciding whether to buy/sell/hold
    parser.add_argument("--window-size", type=int, default=50)

    # Number of epochs to train the agent
    parser.add_argument("--epochs", type=int, default=100)
    
    # Number of hidden nodes in the policy function (neural network) that the agent learns
    # As a rule of thumb, this should be 2 or 3 times the window size
    parser.add_argument("--hidden-nodes", type=int, default=200)

    # Maximum number of shares we allow the agent to own at any given time
    parser.add_argument("--max-shares", type=int, default=100)

    # Learning rate for the policy gradient algorithm
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    # Whether or not to use the learning rate scheduler
    parser.add_argument("--use-lr-scheduler", type=bool, default=True)

    return parser.parse_args()