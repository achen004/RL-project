import gym
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import A2C, PPO, DDPG
import yfinance as yf
from pandas_datareader import data as pdr
from ta import add_all_ta_features
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def DecisionTransformer(rewards, states, action,  t):
    pass
    return  

# Get S&P 500 data from Yahoo Finance
df = pd.read_csv("SPY.csv")
print("df =", df.shape)

# Engineer financial indicators using the method imported above from the "TA" library
df2 = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)

# Configure an environment for training, and train the agent on it
train_env = StocksEnv(df2, window_size=5, frame_bound=(5, 200))
agent = PPO('MlpPolicy', train_env, verbose=1)
agent.learn(total_timesteps=1000)

# Configure an environment for testing, and run the trained agent on it
test_env = StocksEnv(df2, window_size=5, frame_bound=(200, 273))
obs = test_env.reset()
while True:
    obs = obs[np.newaxis, ...]
    action, states = agent.predict(obs)
    obs, rewards, done, info = test_env.step(action)
    
    if done:
        print("info =", info)
        break

plt.figure(figsize=(15, 6))
plt.cla()
test_env.render_all()
plt.show()

# Compute performance of our trading strategy vs. S&P 500
perf_SP500 = df2.loc[df2.shape[0]-1, "Close"] / df2.loc[0, "Close"] - 1
perf_SP500 = np.round(perf_SP500 * 100, 2)
print("S&P 500 performance = {}%".format(perf_SP500))
perf_agent = info["total_profit"] - 1
perf_agent = np.round(perf_agent * 100, 2)
print("Agent performance   = {}%".format(perf_agent))


"""
Some considerations:
- Instead of discrete action space (buy, sell) have a continuous action space (buy x shares, sell x shares) where x is any real number
- Agent 1: Look at 500 C 1 + 500 C 2 + 500 C 3 groupings of stocks. Pick the group that does the best by having the agent buy a fixed number of shares
and sell a fixed number of shares.
- Agent 2: Take the best group of stocks, and figure out what % allocation towards each stock is best.


Notes:
"""

"""
Previous things we talked about - might not work:
~~Train a model that at each time step predicts which stock is the best out of the 500 to trade. "Best" = the stock with
the biggest percent jump the next day. We sell all other stocks and buy that one.~~

"""