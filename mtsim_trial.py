import pytz
from datetime import datetime, timedelta
import numpy as np
from gym_mtsim import (MtSimulator, OrderType, Order, SymbolNotFound, OrderNotFound, MtEnv,
    FOREX_DATA_PATH, STOCKS_DATA_PATH, CRYPTO_DATA_PATH, MIXED_DATA_PATH)
from stable_baselines3 import A2C, PPO, DDPG 
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise 

import gym
#print(gym.__version__)

env = gym.make('stocks-unhedge-v0')

sim = MtSimulator(
    unit='USD',
    balance=10000.,
    leverage=100.,
    stop_out_level=0.2,
    hedge=True,
    symbols_filename=FOREX_DATA_PATH
)

# env = MtEnv(
#     original_simulator=sim,
#     trading_symbols=['GBPCAD', 'EURUSD', 'USDJPY'],
#     window_size=10,
#     # time_points=[desired time points ...],
#     hold_threshold=0.5,
#     close_threshold=0.5,
#     fee=lambda symbol: {
#         'GBPCAD': max(0., np.random.normal(0.0007, 0.00005)),
#         'EURUSD': max(0., np.random.normal(0.0002, 0.00003)),
#         'USDJPY': max(0., np.random.normal(0.02, 0.003)),
#     }[symbol],
#     symbol_max_orders=2,
#     multiprocessing_processes=2
# )

print("env information:")

for symbol in env.prices:
    print(f"> prices[{symbol}].shape:", env.prices[symbol].shape)

print("> signal_features.shape:", env.signal_features.shape)
print("> features_shape:", env.features_shape)

observation = env.reset()

while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        # print(info)
        print(
            f"balance: {info['balance']}, equity: {info['equity']}, margin: {info['margin']}\n"
            f"free_margin: {info['free_margin']}, margin_level: {info['margin_level']}\n"
            f"step_reward: {info['step_reward']}"
        )
        break

state = env.render()

print(
    f"balance: {state['balance']}, equity: {state['equity']}, margin: {state['margin']}\n"
    f"free_margin: {state['free_margin']}, margin_level: {state['margin_level']}\n"
)
state['orders']

#stable baselines application
model = DDPG(
    'MlpPolicy', env, verbose=1,
             actor_lr=0.001, critic_lr=0.001, gamma=0.99, tau=0.001, batch_size=64, buffer_size=int(1e6),
             learning_starts=10000, action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape),
                                                                               sigma=float(0.5) * np.ones(env.action_space.shape)))  #PPO('MultiInputPolicy', env, verbose=0)
model.learn(total_timesteps=1000)

observation = env.reset()
while True:
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    if done:
        break

env.render('advanced_figure', time_format="%Y-%m-%d")
# The green/red triangles show successful buy/sell actions.
# The gray triangles indicate that the buy/sell action has encountered an error.
# The black vertical bars specify close actions.