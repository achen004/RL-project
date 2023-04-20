import gym
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import A2C, PPO, DDPG
import regularizedPPO
import yfinance as yf
from pandas_datareader import data as pdr
from ta import add_all_ta_features
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
import pytz
from datetime import datetime, timedelta
from gym_mtsim import MtSimulator, OrderType, Timeframe, FOREX_DATA_PATH
#import torchviz

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
#TODO implement MtSim
class MyCustomEnv(StocksEnv):
    def __init__(self, df, window_size, frame_bound):
        super().__init__(df, window_size, frame_bound)
    
    # Override the StocksEnv _process_data function with our own function
    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        
        # Prices over the window size
        prices = self.df.loc[:, 'Close'] #close prices
        prices = prices.to_numpy()[start:end]

        # Features to use as signal
        signal_features = self.df.loc[:, ['Close', 'Volume', 'momentum_rsi', 'volume_obv']]#, 'trend_macd_diff']]
        #TODO cash at hand value; incorporate other features to set constraints amounts to buy/sell 
        signal_features = signal_features.to_numpy()[start:end]
        print("signal_features =", signal_features.shape)
        return prices, signal_features

#initialize layer with xavier initiliazed weights ~(0 mean, sqrt(2/(inputs_outputs) standard deviation)
def layer_init(layer):
    torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
    torch.nn.init.constant_(layer.bias, val=0.0)
    return layer

# We define a deep RL agent
class Agent(torch.nn.Module):
    def __init__(self, env, epsilon, learning_rate):
        super().__init__()
        self.env = env
        self.epsilon = epsilon

        # The number of rows and columns in the array representation of a state
        state_num_rows, state_num_cols = env.observation_space.shape

        # Number of nodes in hidden layer
        #TODO: experiment 
        num_hidden_nodes = 5 #originally 5; changing this doesn't impact outputs
        #what if we increase number of epochs along with this? 

        # A function (represented by a neural network) that takes in a state as input,
        # and outputs - for each possible action - the probability of taking that action
        self.policy_function = torch.nn.Sequential(torch.nn.Flatten(start_dim=1, end_dim=-1),
                                                   layer_init(torch.nn.Linear(in_features=state_num_rows * state_num_cols,
                                                                   out_features=num_hidden_nodes,
                                                                   dtype=torch.float64)),
                                                   layer_init(torch.nn.Linear(in_features=num_hidden_nodes,
                                                                   out_features=env.action_space.n,
                                                                   dtype=torch.float64)),
                                                   torch.nn.LogSoftmax()
                                                   )
                
        for tensor in self.policy_function.parameters():
            tensor.requires_grad_(True) #record operations on tensor
        
        # Optimizer for training; TODO learning rate
        self.policy_optimizer = torch.optim.Adam(self.policy_function.parameters(), lr=learning_rate, eps=1e-5)

        self.scheduler=lr_scheduler.StepLR(self.policy_optimizer, step_size=2, gamma=0.1)

    # Compute policy loss for a mini-batch of states and actions; action vectors of policy probabilties for each state
    def policy_loss(self, states_batch, weights):
        """
        states_batch:  tensor format of stack of states from stored batch
        weights: weights of policy model
        """
        #TODO fix 
        # A tensor containing the policy for each state in the mini-batch
        policies_batch = self.policy_function(states_batch)
        return -(policies_batch * weights).mean()
    
    # Train for one epoch
    def train_one_epoch(self, batch_size=32):
        # Make some empty lists for saving mini-batches of observations
        batch_obs = []          # for states
        batch_acts = []         # for actions
        batch_probs = []        # for probabilities of taking the actions we took
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # Reset episode-specific variables
        state = self.env.reset()  # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout the episode

        # Collect experience by executing trades in the environment, using the
        # current policy
        while True:
            # Save the current state in our minibatch
            batch_obs.append(state.copy())

            # Before passing in the state array to the policy function, we have
            # to convert it to a tensor in a specific format
            state = torch.tensor(state) #window size x signal features, where the window size corresponds to the weekdays, and the num of signal features are our features 

            # Converts the array from shape (window_size, num_features) to (1, window_size, num_features)
            state = torch.unsqueeze(state, dim=0)

            # Compute the action we need to take
            policy = self.policy_function(state) #[stochastic] policy aka agent 
            #sampling actions from a probability density guarantees exploration
            action = Categorical(policy).sample().item() #categorical policies are used in discrete action spaces
            prob = policy[0, action]

            # Take that action, collect a reward, and observe the new state
            state, rew, done, info = self.env.step(action)

            # Save in memory the action we took, its probability, and the reward we collected
            batch_acts.append(action)
            batch_probs.append(prob)

            #taking action: buy or sell 
            if action == 0:
                ep_rews.append([rew, 0])
            elif action == 1:
                ep_rews.append([0, rew])

            # If the episode is over
            if done:
                print("info =", info)
                break
        
        # source: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#implementing-the-simplest-policy-gradient

        # Take a single policy gradient update step
        self.policy_optimizer.zero_grad()  #sets gradients of all optimized torch tensors to zero
        states_batch = torch.as_tensor(np.stack(batch_obs), dtype=torch.float64)
        ep_rews = torch.as_tensor(ep_rews, dtype=torch.float64) # the weight for each logprob(a|s) is R(tau)

        batch_loss = self.policy_loss(states_batch, ep_rews)
        print("loss =", batch_loss.item())
        batch_loss.backward()

        self.policy_optimizer.step()
        self.scheduler.step() #learning rate optimizer step
        self.batch_loss = batch_loss
        return batch_loss, batch_rets, batch_lens
    
    def train(self, n_epochs):
        for _ in range(n_epochs):
            self.train_one_epoch()

# Initialize an environment for training the agent, and train the agent on it
# by updating the policy and value functions
train_env = MyCustomEnv(df2, window_size=5, frame_bound=(5, int(train_ratio*N)))
#epsilon=0,  full exploration; TODO what if we implemented GREEDY approach? 
agent = Agent(train_env, epsilon=0, learning_rate=1e-4) #TODO adjust learning_rate; maybe annealize it 
agent.train(n_epochs=50)

# Configure an environment for testing, and run the trained agent on it
test_env = MyCustomEnv(df2, window_size=5, frame_bound=(int(train_ratio*N), N))
state = test_env.reset()
while True:
    state = torch.tensor(state)
    state = torch.unsqueeze(state, dim=0)

    policy = agent.policy_function(state) 
    action = Categorical(policy).sample().item()
    
    state, rewards, done, info = test_env.step(action)
    
    if done:
        print("info =", info)
        break

# Compute performance of our trading strategy vs. S&P 500
# Percentage change in SP500 close price from start to end; benchmark 
perf_SP500 = df2.loc[df2.shape[0]-1, "Close"] / df2.loc[0, "Close"] - 1
perf_SP500 = np.round(perf_SP500 * 100, 2)
print("S&P 500 performance = {}%".format(perf_SP500))
#percent increase or decrease in investments 
perf_agent = info["total_profit"] - 1 #info is a dictionary
perf_agent = np.round(perf_agent * 100, 2)
print("Agent performance   = {}%".format(perf_agent))

plt.figure(figsize=(15, 6))
plt.cla()
test_env.render_all()
plt.show()