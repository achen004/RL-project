import gym
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import A2C, PPO, DDPG
import yfinance as yf
from pandas_datareader import data as pdr
from ta import add_all_ta_features
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#!git clone https://github.com/amr10073/RL-project.git
import sys
sys.path.append('RL-project')
#from models.model import TrajectoryModel
#from models import decisionTransformer, model, trajectory_gpt2

# Get S&P 500 data from Yahoo Finance
df = pd.read_csv("SPY-1hr-2yr.csv")

print("df =", df.shape)

# Engineer financial indicators using the method imported above from the "TA" library
df2 = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
df2 = df2[['Close', 'Volume', 'momentum_rsi', 'volume_obv', 'trend_macd_diff']]
print("Data has features", df2.columns)
N = df2.shape[0]

# Ratio for train/test split (what % of data to hold out as test)
test_ratio = 0.2
train_ratio = 1 - test_ratio

# Configure an environment for training, and train the agent on it
train_env = StocksEnv(df2, window_size=5, frame_bound=(5, int(train_ratio*N)))
agent = A2C('MlpPolicy', train_env, verbose=1)
agent.learn(total_timesteps=100_000)

# Configure an environment for testing, and run the trained agent on it
test_env = StocksEnv(df2, window_size=5, frame_bound=(int(train_ratio*N), N))
obs = test_env.reset()
while True:
    obs = obs[np.newaxis, ...]
    action, states = agent.predict(obs)
    obs, rewards, done, info = test_env.step(action)
    
    if done:
        print("info =", info)
        break

# Compute performance of our trading strategy vs. S&P 500
perf_SP500 = df2.loc[df2.shape[0]-1, "Close"] / df2.loc[0, "Close"] - 1
perf_SP500 = np.round(perf_SP500 * 100, 2)
print("S&P 500 performance = {}%".format(perf_SP500))
perf_agent = info["total_profit"] - 1
perf_agent = np.round(perf_agent * 100, 2)
print("Agent performance   = {}%".format(perf_agent))

plt.figure(figsize=(15, 6))
plt.cla()
test_env.render_all()
plt.show()

"""
Some considerations:
- Instead of discrete action space (buy, sell) have a continuous action space (buy x shares, sell x shares) where x is any real number
"""

"""
Previous things we talked about - might not work:
~~Train a model that at each time step predicts which stock is the best out of the 500 to trade. "Best" = the stock with
the biggest percent jump the next day. We sell all other stocks and buy that one.~~

"""