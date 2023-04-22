from ta import add_all_ta_features
import pandas as pd
import torch
from MyCustomEnv import MyCustomEnv
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

torch.manual_seed(0)

# Get S&P 500 data from Yahoo Finance
#df = pd.read_csv("data/ADBE.csv")
df = pd.read_csv("SPY_prices_2y_1h.csv")
print("df =", df.shape)

# Engineer financial indicators using the method imported above from the "TA" library
df2 = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
print(df2.columns)
N = df2.shape[0]

# Initialize an environment for the agent to execute trades
env_window_size = 5
env = MyCustomEnv(df2, window_size=env_window_size, frame_bound=(env_window_size, N))
# env = DummyVecEnv([lambda: env])

# Create the PPO agent
model = PPO("MlpPolicy", env, verbose=0)
# model = A2C("MlpPolicy", env, verbose=0)
# model = DQN("MlpPolicy", env, verbose=0)

# Train the agent
model.learn(total_timesteps=10000)
    
env.render_all()
