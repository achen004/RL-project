from ta import add_all_ta_features
import pandas as pd
import torch
from MyCustomEnv import MyCustomEnv
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import Counter

torch.manual_seed(0)

df = pd.read_csv("SPY_prices_2y_1h.csv")
print("df =", df.shape)

df2 = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
print(df2.columns)
N = df2.shape[0]

env_window_size = 5
env = MyCustomEnv(df2, window_size=env_window_size, frame_bound=(env_window_size, N))#, MAX_SHARES=1000)
obs = env.reset()
# Train the models
ppo_model = PPO("MlpPolicy", env, verbose=0)
ppo_model.learn(total_timesteps=1000)

obs = env.reset()
a2c_model = A2C("MlpPolicy", env, verbose=0)
a2c_model.learn(total_timesteps=1000)

obs = env.reset()
dqn_model = DQN("MlpPolicy", env, verbose=0)
dqn_model.learn(total_timesteps=1000)

# Reset the environment
env = MyCustomEnv(df2, window_size=env_window_size, frame_bound=(env_window_size, N))
obs = env.reset()

# Majority vote
for _ in range(N - env_window_size):
    ppo_action, _ = ppo_model.predict(obs)
    a2c_action, _ = a2c_model.predict(obs)
    dqn_action, _ = dqn_model.predict(obs)

    actions = [int(ppo_action), int(a2c_action), int(dqn_action)]
    majority_action = Counter(actions).most_common(1)[0][0]
    
    obs, rewards, dones, info = env.step(majority_action)

env.render_all()