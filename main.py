from ta import add_all_ta_features
import pandas as pd
import torch
from MyCustomEnv import MyCustomEnv
from parse_args import parse_args
from Agent import Agent

torch.manual_seed(0)

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
    env = MyCustomEnv(df2,
                      window_size=args.window_size,
                      frame_bound=(args.window_size, N),
                      MAX_SHARES=args.max_shares)

    # Initialize and train our custom agent on this environment by updating the policy function
    agent = Agent(action_space_dim=env.action_space.n,
                  observation_space_dim=env.observation_space.shape,
                  learning_rate=args.learning_rate,
                  num_hidden_nodes=args.hidden_nodes,
                  use_lr_scheduler=args.use_lr_scheduler)

    agent.train(env, n_epochs=args.epochs)

    # Output results
    env.render_all()
