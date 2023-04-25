from ta import add_all_ta_features
import pandas as pd
import torch
from MyCustomEnv import MyCustomEnv, Actions
from parse_args import parse_args
from stable_baselines3 import PPO, A2C, DQN
from collections import Counter
from Agent import Agent

torch.manual_seed(0)

def check_tie(actions):
    count_actions = Counter(actions)
    most_common = count_actions.most_common()[0:2]
    if len(most_common) < 2:
        return False
    
    pair1, pair2 = most_common
    return (pair1[1] == pair2[1])

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Get S&P 500 data from Yahoo Finance
    df = pd.read_csv(args.stock)
    print("df =", df.shape)

    # Engineer financial indicators using the method imported above from the "TA" library
    df2 = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
    N = df2.shape[0]

    # Initialize an environment for the agent to execute trades
    env = MyCustomEnv(df2,
                      window_size=args.window_size,
                      frame_bound=(args.window_size, N),
                      MAX_SHARES=args.max_shares)

    # Train the models
    obs = env.reset()
    ppo_model = PPO("MlpPolicy", env, verbose=0)
    ppo_model.learn(total_timesteps=10_000)

    obs = env.reset()
    a2c_model = A2C("MlpPolicy", env, verbose=0)
    a2c_model.learn(total_timesteps=10_000)

    obs = env.reset()
    dqn_model = DQN("MlpPolicy", env, verbose=0)
    dqn_model.learn(total_timesteps=10_000)

    obs = env.reset()
    custom_agent = Agent(action_space_dim=env.action_space.n,
                observation_space_dim=env.observation_space.shape,
                learning_rate=args.learning_rate,
                num_hidden_nodes=args.hidden_nodes,
                use_lr_scheduler=args.use_lr_scheduler)
    custom_agent.train(env, n_epochs=args.epochs)
    
    env.reset()

    # Majority vote
    done = False

    while not done:
        ppo_action, _ = ppo_model.predict(obs)
        a2c_action, _ = a2c_model.predict(obs)
        dqn_action, _ = dqn_model.predict(obs)
        custom_action, _ = custom_agent.predict(obs)

        actions = [int(ppo_action), int(a2c_action), int(dqn_action), int(custom_action)]

        # If we have a tie (no majority):
        if check_tie(actions):
            majority_action = Actions.Buy.value
        else:
            majority_action = Counter(actions).most_common(1)[0][0]
        
        obs, reward, done, info = env.step(majority_action)
        
    env.render_all()