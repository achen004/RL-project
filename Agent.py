import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.optim.lr_scheduler as lr_scheduler 
from MyCustomEnv import Actions

#initialize layer with xavier initialized weights ~(0 mean, sqrt(2/(inputs_outputs) standard deviation)
def layer_init(layer):
    torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
    torch.nn.init.constant_(layer.bias, val=0.0)
    return layer

# We define a deep RL agent
class Agent(torch.nn.Module):
    def __init__(self, action_space_dim, observation_space_dim, learning_rate, num_hidden_nodes, use_lr_scheduler):
        super().__init__()

        # The number of rows and columns in the array representation of a state
        state_num_rows, state_num_cols = observation_space_dim

        # Number of nodes in hidden layer
        self.num_hidden_nodes = num_hidden_nodes

        # A function (represented by a neural network) that takes in a state as input,
        # and outputs - for each possible action - the probability of taking that action
        self.policy_function = torch.nn.Sequential(torch.nn.Flatten(start_dim=1, end_dim=-1),
                                                   layer_init(torch.nn.Linear(in_features=state_num_rows * state_num_cols,
                                                                   out_features=num_hidden_nodes,
                                                                   dtype=torch.float64)),
                                                                   torch.nn.ReLU(),
                                                   layer_init(torch.nn.Linear(in_features=num_hidden_nodes,
                                                                   out_features=action_space_dim,
                                                                   dtype=torch.float64)),
                                                   torch.nn.LogSoftmax() # use this instead of softmax for better numerical stability
                                                   )
        #include critic function: value(state, action)
        
        for tensor in self.policy_function.parameters():
            tensor.requires_grad_(True) #record operations on tensor
        
        # Optimizer for training
        self.policy_optimizer = torch.optim.Adam(self.policy_function.parameters(), lr=learning_rate, eps=1e-5)

        self.use_lr_scheduler = use_lr_scheduler
        if self.use_lr_scheduler:
            self.scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=2, gamma=0.1)

    # Compute policy loss for a mini-batch of states and actions; action vectors of policy probabilties for each state
    def policy_loss(self, states_batch, weights):
        """
        states_batch:  tensor format of stack of states from stored batch
        weights: weights of policy model
        """ 
        # A tensor containing the policy for each state in the mini-batch
        policies_batch = self.policy_function(states_batch)
        return -(policies_batch * weights).mean()
    
    # Train for one epoch
    def train_one_epoch(self, env, epoch_num):
        # Make some empty lists for saving mini-batches of observations
        batch_obs = []          # for states
        batch_acts = []         # for actions
        batch_probs = []        # for probabilities of taking the actions we took
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # Reset episode-specific variables
        state = env.reset()  # first obs comes from starting distribution
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

            # Sample an action from the probability distribution (policy)
            # This ensures exploration instead of just always choosing the greedy action
            # Categorical policies are used in discrete action spaces
            action = Categorical(policy).sample().item()
            prob = policy[0, action]

            # Take that action, collect a reward, and observe the new state
            state, reward, done, info = env.step(action)

            # Save in memory the action we took, its probability, and the reward we collected
            batch_acts.append(action)
            batch_probs.append(prob)

            #taking action: buy or sell or do nothing
            if action == Actions.Sell.value:
                ep_rews.append([env._total_reward, 0, 0])
                sell_count += 1
            elif action == Actions.Hold.value:
                ep_rews.append([0, env._total_reward, 0])
                hold_count += 1
            elif action == Actions.Buy.value:
                ep_rews.append([0, 0, env._total_reward])
                buy_count += 1

            # if reward > 0:
            #     print("[training epoch {}] info = {}".format(epoch_num, info))
        
        print(f"[training epoch {epoch_num}] realized gain (total reward) = {np.round(env._total_reward, 2)}")
        # print(f"[training epoch {epoch_num}] buy count = {env.buy_count}, sell count = {env.sell_count}, hold count = {env.hold_count}")    

        # source: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#implementing-the-simplest-policy-gradient

        # Take a single policy gradient update step
        self.policy_optimizer.zero_grad()  #sets gradients of all optimized torch tensors to zero
        states_batch = torch.as_tensor(np.stack(batch_obs), dtype=torch.float64)
        ep_rews = torch.as_tensor(ep_rews, dtype=torch.float64) # the weight for each logprob(a|s) is R(tau)

        batch_loss = self.policy_loss(states_batch, ep_rews)
        batch_loss.backward()

        self.policy_optimizer.step()
        if self.use_lr_scheduler:
            self.scheduler.step() # learning rate optimizer step
        
        return batch_loss, batch_rets, batch_lens
    
    def train(self, env, n_epochs):
        for i in range(1, n_epochs+1):
            self.train_one_epoch(env, i)