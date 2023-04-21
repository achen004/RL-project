from ta import add_all_ta_features
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from MyCustomEnv import MyCustomEnv
from Agent import Agent
from torch.distributions.categorical import Categorical

torch.manual_seed(0)

# Get S&P 500 data from Yahoo Finance
df = pd.read_csv("SPY_prices_2y_1h.csv")
print("df =", df.shape)

# Engineer financial indicators using the method imported above from the "TA" library
df2 = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
N = df2.shape[0]

# Ratio for train/test split (what % of data to hold out as test)
test_ratio = 0.2
train_ratio = 1 - test_ratio

# Initialize an environment for training the agent, and train the agent on it
env_window_size = 5
train_env = MyCustomEnv(df2, window_size=env_window_size, frame_bound=(env_window_size, int(train_ratio*N)))

#epsilon=0,  full exploration; TODO what if we implemented GREEDY approach? 
# Initialize and train an agent on this environment by updating the policy function
agent = Agent(train_env, epsilon=0, learning_rate=1e-4) #TODO adjust learning_rate; maybe annealize it 
agent.train(n_epochs=2)

train_env.render_all()

# Initialize an environment for testing, and run the trained agent on it
test_env = MyCustomEnv(df2, window_size=env_window_size, frame_bound=(int(train_ratio*N), N))
state = test_env.reset()
done = False
while not done:
    state = torch.tensor(state)
    state = torch.unsqueeze(state, dim=0)

    policy = agent.policy_function(state)
    action = Categorical(policy).sample().item()
    
    state, rewards, done, info = test_env.step(action)

    print("info =", info)

# Compute performance of our trading strategy vs. S&P 500
# Percentage change in SP500 close price from start to end; benchmark 
perf_SP500 = df2.loc[df2.shape[0]-1, "Close"] / df2.loc[0, "Close"] - 1
perf_SP500 = np.round(perf_SP500 * 100, 2)
#print("S&P 500 performance = {}%".format(perf_SP500))

#percent increase or decrease in investments 
#perf_agent = info["total_profit"] - 1 #info is a dictionary
#perf_agent = np.round(perf_agent * 100, 2)
#print("Agent performance   = {}%".format(perf_agent))

test_env.render_all()