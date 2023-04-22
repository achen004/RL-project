from ta import add_all_ta_features
import pandas as pd
import torch
from MyCustomEnv import MyCustomEnv
from Agent import Agent

torch.manual_seed(0)

# Get S&P 500 data from Yahoo Finance
df = pd.read_csv("data/ADBE.csv")
#df = pd.read_csv("SPY_prices_2y_1h.csv")
print("df =", df.shape)

# Engineer financial indicators using the method imported above from the "TA" library
df2 = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
print(df2.columns)
N = df2.shape[0]

# Initialize an environment for the agent to execute trades
env_window_size = 5
env = MyCustomEnv(df2, window_size=env_window_size, frame_bound=(env_window_size, N))

# Initialize and train an agent on this environment by updating the policy function
agent = Agent(action_space_dim=env.action_space.n,
              observation_space_dim=env.observation_space.shape,
              learning_rate=1e-4) #TODO adjust learning_rate; maybe annealize it 

agent.train(env, n_epochs=10)

env.render_all()