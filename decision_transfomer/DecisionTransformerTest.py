from random import random
import sys

import gym
import gym_mtsim
import pandas as pd
from ta import add_all_ta_features
from decision_transfomer.models.decisionTransformer import DecisionTransformer
# from decision_transfomer.models.decisionTransformer import DecisionTransformer
from MyCustomEnv import MyCustomEnv

import numpy as np
import torch
import time
import torch.nn as nn
from torch.distributions.categorical import Categorical
from decision_transfomer.lamb import Lamb
from decision_transfomer.seq_trainer import SequenceTrainer

MAX_EPISODE_LEN =1000

scale = 1000
action_range = [0,1]
K = 10
embed_dim =128
n_layer = 4
n_head = 4
activation_function = "relu"
dropout = 0.1
eval_context_length=5

ordering = 0

# shared evaluation options
eval_rtg = 3600
num_eval_episodes = 10

# shared training options
init_temperature=0.1
batch_size = 256
learning_rate = 1e-4
weight_decay = 5e-4
warmup_steps = 10000

# pretraining options
max_pretrain_iters = 1
num_updates_per_pretrain_iter = 5000

# finetuning options
max_online_iters = 1500
online_rtg = 7200
num_online_rollouts = 1
replay_size = 1000
num_updates_per_online_iter = 300
eval_interval = 10

# environment options
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("SPY_prices_2y_1h.csv")
df_test = pd.read_csv("data/AAPL.csv")
#df = pd.read_csv("SPY_prices_2y_1h.csv")

# Engineer financial indicators using the method imported above from the "TA" library
df2 = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
df2_test = add_all_ta_features(df_test, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
# print(df2.columns)
N = df2.shape[0]

# Initialize an environment for the agent to execute trades
env_window_size = 1

dt_train_env = MyCustomEnv(df2, window_size=env_window_size, frame_bound=(env_window_size, N))
state = dt_train_env.reset()
state_dim = dt_train_env.observation_space.shape[1]
act_dim = 3
target_entropy = -act_dim

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    action_range= action_range,
    max_length=K,
    eval_context_length=eval_context_length,
    max_ep_len=MAX_EPISODE_LEN,
    hidden_size=embed_dim,
    n_layer=n_layer,
    n_head=n_head,
    n_inner=4 * embed_dim,
    activation_function=activation_function,
    n_positions=1024,
    resid_pdrop=dropout,
    attn_pdrop=dropout,
    stochastic_policy=False,
    ordering=ordering,
    init_temperature=init_temperature,
    target_entropy= target_entropy,
).to(device=device)

done = False
def randomInit(context,env = dt_train_env):
    init_actions= []
    init_rewards = []
    init_states = []
    init_dones = []
    state = env.reset()
    for i in range(context):
        if random() < 0.5:
            action = np.array([0,0,1])
        else:
            action = np.array([1,0,0])
        
        state, rew, done, info = env.step(action=Categorical(torch.tensor(action, dtype=torch.float)).sample().item())
        init_states.append(state)
        init_rewards.append(rew)
        init_actions.append(action)
        init_dones.append(done)

    init_rtg = np.array([[sum(init_rewards[:i+1])] for i in range(len(init_rewards))])
    init_states = torch.tensor(np.array(init_states), dtype=torch.float).squeeze(dim=1)
    init_actions = torch.tensor(np.array(init_actions), dtype=torch.float)
    init_rewards = torch.tensor(np.array(init_rewards), dtype=torch.float)
    init_rtg = torch.tensor(np.array(init_rtg), dtype=torch.float)
    timesteps = torch.range(start=0,step=1,end=K, dtype=torch.float)

    return init_states, init_actions, init_rewards, init_rtg, timesteps
states, actions, rewards, rtg, timesteps = randomInit(K)


optimizer = Lamb(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    eps=1e-8,
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda steps: min((steps + 1) /warmup_steps, 1)
)


online_iter = 0
total_transitions_sampled = 0
reward_scale = 1

# trainer = SequenceTrainer(
#     model=model,
#     optimizer=optimizer,
#     log_temperature_optimizer=log_temperature_optimizer,
#     scheduler=scheduler,
#     device=device,
# )

def policy_loss(self, model, states_batch, weights):
        """
        states_batch:  tensor format of stack of states from stored batch
        weights: weights of policy model
        """
        policies_batch = model(states_batch)
        return -(policies_batch * weights).mean()
    

def train_step_stochastic(self, trajs, env):
    (
        states,
        actions,
        rewards,
        rtg,
        timesteps,
        ordering,
        padding_mask
    ) = trajs

    action_target = torch.clone(actions)

    # forward pass
    _, action_preds, _ = model.forward(
        states[-K:],
        actions[-K:],
        rewards[-K:],
        rtg[-K:],
        timesteps[-K:],
        ordering=ordering[-K:],
        padding_mask=padding_mask,
    )
    
    state, rew, done, info = dt_train_env.step(action=Categorical(action_preds[-1]).sample().item())
    
    loss = policy_loss(
        action_preds,  # a_hat_dist
        rtg,
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()

    if scheduler is not None:
        scheduler.step()
    
    states = torch.cat((states, state), dim=0)
    actions = torch.cat((actions, action_preds[-1]), dim=0)
    rewards = torch.cat((rewards, rew), dim=0)
    rtg = torch.cat((rtg, rtg[-1]+ rew), dim=0)
    timesteps = torch.cat((timesteps, timesteps[-1]+1), dim=0)

def train_epoch(env, model, optimizer, scheduler, device, padding_mask, ordering):
    states, actions, rewards, rtg, timesteps = randomInit(K,env)
    while not done:

        train_step_stochastic(model, (states, actions, rewards, rtg, timesteps, ordering, torch.ones((env_window_size, act_dim), dtype=torch.long)))
num_epochs=2

def train(num_epochs):
    for i in range(num_epochs):
        train_epoch(model, optimizer, scheduler, device, None, ordering)

# loss, nll, entropy = train_step_stochastic(model, (states, actions, rewards, rtg, timesteps, ordering, torch.ones((env_window_size, act_dim), dtype=torch.long)))
