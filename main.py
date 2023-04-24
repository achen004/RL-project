from ta import add_all_ta_features
import pandas as pd
import torch
from MyCustomEnv import MyCustomEnv
from Agent import Agent
import argparse

torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent to trade a stock.")

    # The path to the CSV file containing the prices for the stock we want to trade
    parser.add_argument("--stock", type=str)
    
    # Number of days of past stock prices for the agent to look at when it's deciding whether to buy/sell/hold
    parser.add_argument("--window-size", type=int)

    # Number of epochs to train the agent
    parser.add_argument("--epochs", type=int)
    
    # Number of hidden nodes in the policy function (neural network) that the agent learns
    parser.add_argument("--hidden-nodes", type=int)
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Get S&P 500 data from Yahoo Finance
    df = pd.read_csv(args.stock)
    print("df =", df.shape)

    # Engineer financial indicators using the method imported above from the "TA" library
    df2 = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
    N = df2.shape[0]

    # Initialize an environment for the agent to execute trades
    env = MyCustomEnv(df2, window_size=args.window_size, frame_bound=(args.window_size, N))

    # Initialize and train an agent on this environment by updating the policy function
    agent = Agent(action_space_dim=env.action_space.n,
                  observation_space_dim=env.observation_space.shape,
                  learning_rate=1e-4,
                  num_hidden_nodes=args.hidden_nodes)

    agent.train(env, n_epochs=args.epochs)

    # Output results
    env.render_all()