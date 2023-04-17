import pytz
from datetime import datetime, timedelta
import numpy as np
from gym_mtsim import MtEnv, OrderType, Timeframe, MtSimulator, FOREX_DATA_PATH

import gym
import gym_mtsim

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

env.render('simple_figure')