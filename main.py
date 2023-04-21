from gym_anytrading.envs import StocksEnv, TradingEnv
from gym import spaces
from ta import add_all_ta_features
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.distributions.categorical import Categorical
import torch.optim.lr_scheduler as lr_scheduler 
from enum import Enum
torch.manual_seed(0)

"""
Algorithm for Environment:
Our Agent will be buying or selling 1 stock at every timestep
We will keep track of the total number of shares it owns and the price it paid for each share

At every tick, Timestep T, the Agent will be rewarded based on the following formula:
step_reward = (Average Bought_Price of shares at T) - (Price of share sold at T) * (Number of shares sold at T) /// Number of Shares sold at T = 1

Profit at each step based on signals of market data 
Total profit at T += step_reward

Algorithm for Agent:
Our Agent Can Decide to Take 1 of 3 Actions:
1. Buy 1 share
2. Sell 1 share
3. Hold

The Agent will be rewarded based on the following formula:
step_reward = (Average Bought_Price of shares at T) - (Price of share sold at T) * (Number of shares sold at T) /// Number of Shares sold at T = 1

#TODO:
Need to Calculate Market Value of Agent's Portfolio at the end of each epoch, and reward the agent based on that

Punish the agent for unrealized losses, and reward it for unrealized gains
"""

# Get S&P 500 data from Yahoo Finance
df = pd.read_csv("SPY_prices_2y_1h.csv")

print("df =", df.shape)

# Engineer financial indicators using the method imported above from the "TA" library
df2 = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
N = df2.shape[0]

# Ratio for train/test split (what % of data to hold out as test)
test_ratio = 0.2
train_ratio = 1 - test_ratio
class Actions(Enum):
    Sell = 0
    Hold = 1
    Buy = 2

# Configure an environment for training - specifically we take StocksEnv
# and customize it according to the problem we're trying to solve
#
# We assume unlimited margin (the agent can buy as much stock as it wants)
class MyCustomEnv(StocksEnv, TradingEnv):
    # Call the StocksEnv constructor, and on top of that, initialize our own variables
    def __init__(self, df, window_size, frame_bound, MAX_SHARES=50):
        super().__init__(df, window_size, frame_bound)

        # We override StocksEnv's action space with our own action space, according to
        # the Actions enum class we wrote above
        self.action_space = spaces.Discrete(len(Actions))

        # The total number of shares we currently own
        self.total_shares = 0
        
        # The profit we've made so far
        self.profit_at_T = 0

        # Maximum number of shares an agent can hold at any given time
        # It keeps margin trading from getting too out of control
        self.MAX_SHARES = MAX_SHARES
        
        # The prices of the shares we bought
        self.prices_of_shares_we_bought = [] 
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0

        #limit % of shares to buy 

        # List of actions we've taken over the course of an episode
        self.action_history = []
    
    # ====================================================================
    # Override the StocksEnv functions with our own functions
    # ====================================================================
    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        
        # Prices over the window size
        prices = self.df.loc[:, 'Close'] #close prices
        prices = prices.to_numpy()[start:end]
        self.total_profit = 0
        # Features to use as signal
        signal_features = self.df.loc[:, ['Close', 'Volume', 'momentum_rsi', 'volume_obv']]#, 'trend_macd_diff']]
        #TODO cash at hand value; incorporate other features to set constraints amounts to buy/sell 
        signal_features = signal_features.to_numpy()[start:end]
        print("signal_features =", signal_features.shape)
        return prices, signal_features
    
    def _calculate_reward(self, action):
        # Initialize a variable for the reward the agent receives from taking this action
        step_reward = 0
        
        if action == Actions.Hold.value:
            self.action_history.append(action)
            return step_reward
        
        elif action == Actions.Buy.value and self.total_shares < self.MAX_SHARES:
            self.action_history.append(action)

            current_price = self.prices[self._current_tick]
            self.prices_of_shares_we_bought.append(current_price)
            self.total_shares += 1
            self.buy_count += 1
            
            # We don't reward buying, because we don't know yet if we'll profit from this
            # trade or not. If we rewarded buying, then the agent might keep buying forever,
            # and never selling.
            return step_reward
        
        elif action == Actions.Sell.value and self.total_shares > 0:
            self.action_history.append(action)
            
            current_price = self.prices[self._current_tick]
            average_buy_price = np.mean(self.prices_of_shares_we_bought)
            self.total_shares -=1
            self.prices_of_shares_we_bought = [average_buy_price for _ in range(self.total_shares)]
            step_reward = (current_price - average_buy_price)
            self._total_reward += step_reward
            self.sell_count += 1
            return step_reward

        # This only runs when we try to sell but we don't have any shares to sell
        else:
            self.action_history.append(-99)
            return step_reward

    # ====================================================================
    # Override the TradingEnv functions with our own functions
    # ====================================================================
    def reset(self):
        print("\n\n\n\n ***** RESET ******* \n\n\n\n")
        # Call TradingEnv.reset()
        super().reset()
        

        self.action_history = []
        self.total_shares = 0
        print("[reset] total reward =", self._total_reward, self.total_shares)

        return self._get_observation()

    def step(self, action):
        # Take a step in the environment
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        # Calculate the reward received in this action
        step_reward = self._calculate_reward(action)

        #print("[training] action =", action)
        
        # Get the next state
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            action = action
            #total_profit = self._total_profit,
        )
        self._update_history(info)

        return observation, step_reward, self._done, info
    
    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self.action_history))

        plt.plot(self.prices)

        buy_ticks = []
        sell_ticks = []
        hold_ticks = []
        denied_ticks = [] # this is when the agent tries to buy or sell but the environment doesn't allow it to do so
        for i, tick in enumerate(window_ticks):
            if self.action_history[i] == Actions.Buy.value:
                buy_ticks.append(tick)
            elif self.action_history[i] == Actions.Sell.value:
                sell_ticks.append(tick)
            elif self.action_history[i] == Actions.Hold.value:
                hold_ticks.append(tick)
            else:
                denied_ticks.append(tick)
        
        plt.plot(buy_ticks, self.prices[buy_ticks], 'go')
        plt.plot(sell_ticks, self.prices[sell_ticks], 'ro')
        plt.legend()
        # plt.plot(hold_ticks, self.prices[hold_ticks], 'bo', alpha=0.25)
        # plt.plot(denied_ticks, self.prices[denied_ticks], 'bo', alpha=0.25)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

#initialize layer with xavier initialized weights ~(0 mean, sqrt(2/(inputs_outputs) standard deviation)
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
        
        print("policy func shapes=", [param.shape for param in self.policy_function.parameters()])
        
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
        buy_count = 0
        sell_count = 0
        hold_count = 0
        while not done:
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
            state, reward, done, info = self.env.step(action)
            #print("info =", info)

            # Save in memory the action we took, its probability, and the reward we collected
            batch_acts.append(action)
            batch_probs.append(prob)

            #taking action: buy or sell or do nothing
            if action == Actions.Sell.value:
                ep_rews.append([reward, 0, 0])
                sell_count += 1
            elif action == Actions.Hold.value:
                ep_rews.append([0, reward, 0])
                hold_count += 1
            elif action == Actions.Buy.value:
                ep_rews.append([0, 0, reward])
                buy_count += 1
    
        # print(f"final info = {info}")    
        # print(f"[training] buy count = {self.env.buy_count}, sell count = {self.env.sell_count}, hold count = {self.env.hold_count}")
        
        # source: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#implementing-the-simplest-policy-gradient

        # Take a single policy gradient update step
        self.policy_optimizer.zero_grad()  #sets gradients of all optimized torch tensors to zero
        states_batch = torch.as_tensor(np.stack(batch_obs), dtype=torch.float64)
        ep_rews = torch.as_tensor(ep_rews, dtype=torch.float64) # the weight for each logprob(a|s) is R(tau)

        batch_loss = self.policy_loss(states_batch, ep_rews)
        batch_loss.backward()

        self.policy_optimizer.step()
        self.scheduler.step() #learning rate optimizer step
        self.batch_loss = batch_loss
        return batch_loss, batch_rets, batch_lens
    
    def train(self, n_epochs):
        for _ in range(n_epochs):
            self.train_one_epoch()

# Initialize an environment for training the agent, and train the agent on it
# by updating the policy function
env_window_size = 5
train_env = MyCustomEnv(df2, window_size=env_window_size, frame_bound=(env_window_size, int(train_ratio*N)))

print("bid =", train_env.trade_fee_bid_percent)
print("ask =", train_env.trade_fee_ask_percent)

#epsilon=0,  full exploration; TODO what if we implemented GREEDY approach? 
agent = Agent(train_env, epsilon=0, learning_rate=1e-4) #TODO adjust learning_rate; maybe annealize it 
agent.train(n_epochs=20)

# plt.figure(figsize=(15, 6))
# plt.cla()
# train_env.render_all()
# plt.show()

# Configure an environment for testing, and run the trained agent on it
test_env = MyCustomEnv(df2, window_size=env_window_size, frame_bound=(int(train_ratio*N), N))
state = test_env.reset()
while True:
    state = torch.tensor(state)
    state = torch.unsqueeze(state, dim=0)

    policy = agent.policy_function(state)
    action = Categorical(policy).sample().item()
    #print("[testing] action =", action)
    
    state, rewards, done, info = test_env.step(action)
    
    if done:
        print("info =", info)
        break

# Compute performance of our trading strategy vs. S&P 500
# Percentage change in SP500 close price from start to end; benchmark 
perf_SP500 = df2.loc[df2.shape[0]-1, "Close"] / df2.loc[0, "Close"] - 1
perf_SP500 = np.round(perf_SP500 * 100, 2)
#print("S&P 500 performance = {}%".format(perf_SP500))

#percent increase or decrease in investments 
#perf_agent = info["total_profit"] - 1 #info is a dictionary
#perf_agent = np.round(perf_agent * 100, 2)
#print("Agent performance   = {}%".format(perf_agent))

plt.figure(figsize=(15, 6))
plt.cla()
test_env.render_all()
plt.show()
