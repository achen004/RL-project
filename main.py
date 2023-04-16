import gym
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import A2C, PPO, DDPG
import yfinance as yf
from pandas_datareader import data as pdr
from ta import add_all_ta_features
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.distributions.categorical import Categorical

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
N = df2.shape[0]

# Ratio for train/test split (what % of data to hold out as test)
test_ratio = 0.2
train_ratio = 1 - test_ratio

# Configure an environment for training, and train the agent on it
class MyCustomEnv(StocksEnv):
    def __init__(self, df, window_size, frame_bound):
        super().__init__(df, window_size, frame_bound)
    
    # Override the StocksEnv _process_data function with our own function
    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        
        # Prices over the window size
        prices = self.df.loc[:, 'Close']
        prices = prices.to_numpy()[start:end]

        # Features to use as signal
        signal_features = self.df.loc[:, ['Close', 'Volume', 'momentum_rsi', 'volume_obv']]#, 'trend_macd_diff']]
        signal_features = signal_features.to_numpy()[start:end]
        print("signal_features =", signal_features.shape)
        return prices, signal_features

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# We define a deep RL agent
class Agent(torch.nn.Module):
    def __init__(self, env, learning_rate, epsilon):
        super().__init__()
        self.env = env
        self.epsilon = epsilon

        # The number of rows and columns in the array representation of a state
        state_num_rows, state_num_cols = env.observation_space.shape

        # Number of nodes in hidden layer
        num_hidden_nodes = 5

        # A function (represented by a neural network) that takes in a state as input,
        # and outputs - for each possible action - the probability of taking that action
        self.policy_function = torch.nn.Sequential(layer_init(torch.nn.Linear(in_features=state_num_rows * state_num_cols,
                                                                   out_features=num_hidden_nodes,
                                                                   dtype=torch.float64)),
                                                   layer_init(torch.nn.Linear(in_features=num_hidden_nodes,
                                                                   out_features=env.action_space.n,
                                                                   dtype=torch.float64)),
                                                   torch.nn.Softmax()
                                                   )
        
        # Optimizer for training
        self.optimizer = torch.optim.Adam(self.policy_function.parameters(), lr=learning_rate, eps=1e-5)

    # Actor (policy function)
    def policy(self, state):
        state = torch.flatten(state)
        policy_ = self.policy_function(state)
        
        if policy_[0].item() == 0 and policy_[1].item() == 0:
            return torch.Tensor([0.5, 0.5])
        return policy_
    
    # Epsilon-greedy approach
    def select_action(self, state):
        # Compute probabilities of taking each action
        probs = self.policy(state)

        # Take the action corresponding to the highest probability
        # (the greedy action) with probability 1-epsilon, and take
        # a random action with probability epsilon
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return torch.argmax(probs)

    # Compute policy loss for a mini-batch of states and actions
    def policy_loss(self, states_batch, actions_batch, weights):
        # For each state in this mini-batch, compute the log probability of taking the action
        # that we took in that state
        logp = torch.zeros_like(weights)
        for i in range(states_batch.shape[0]):
            state = states_batch[i]
            action = actions_batch[i]
            logp[i] = Categorical(self.policy(state)).log_prob(action)
        
        loss = -(logp * weights).mean()
        print("loss =", loss)
        return loss

    # Train for one epoch
    def train(self, batch_size=32):
        # Make some empty lists for saving mini-batches of observations
        batch_obs = []          # for states
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # Reset episode-specific variables
        state = self.env.reset()  # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout the episode

        # A flag as to whether or not the agent is taking an action for the first time
        first_action = True

        # Collect experience by executing trades in the environment, using the
        # current policy
        while True:
            # Save the current state in our minibatch
            batch_obs.append(state.copy())

            # Take an action, collect reward, and observe new state
            state = torch.DoubleTensor(state)
            if first_action:
                # Always buy the first time, just to get the algo started
                action = 1
                first_action = False
            else:
                action = self.select_action(state)
                #print("action =", action)
            
            state, rew, done, info = self.env.step(action)

            # Save in memory the action we took and reward we collected
            batch_acts.append(action)
            ep_rews.append(rew)

            # If the episode is over
            if done:
                print("info =", info)
                # Compute the episodic return (the total reward we accumulated during this episode)
                ep_ret = sum(ep_rews)
                print("Episodic return =", ep_ret)

                # Compute how many rewards we collected
                ep_len = len(ep_rews)

                # Save in memory the values we computed above
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                state, done, ep_rews = self.env.reset(), False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # source: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#implementing-the-simplest-policy-gradient

        # Take a single policy gradient update step
        self.optimizer.zero_grad()
        
        batch_loss = self.policy_loss(states_batch=torch.as_tensor(np.stack(batch_obs), dtype=torch.float64),
                                      actions_batch=torch.as_tensor(batch_acts, dtype=torch.int32),
                                      weights=torch.as_tensor(batch_weights, dtype=torch.float64)
                                      )
        batch_loss.backward()
        self.optimizer.step()
        
        return batch_loss, batch_rets, batch_lens

# Initialize an environment for training the agent, and train the agent on it
# by updating the policy and value functions
train_env = MyCustomEnv(df2, window_size=5, frame_bound=(5, int(train_ratio*N)))
agent = Agent(train_env, learning_rate=1.5, epsilon=0)

n_epochs = 3
for i in range(n_epochs):
    print("EPOCH {} =============================================================".format(i))
    agent.train(batch_size=32)

"""
# Configure an environment for testing, and run the trained agent on it
test_env = MyCustomEnv(df2, window_size=5, frame_bound=(int(train_ratio*N), N))
state = test_env.reset()
while True:
    state = torch.DoubleTensor(state)
    action = agent.select_action(state)
    state, rewards, done, info = test_env.step(action)
    
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