import torch
from decision_transfomer.evaluate_episodes import evaluate_episode_rtg
from models.decisionTransformer import DecisionTransformer

state_dim = 5
act_dim = 2
K = 30
max_ep_len = 1000
embed_dim = 128
n_layer = 3
n_head = 1
activation_function = 'relu'
dropout = 0.1
learning_rate = 1e-4
weight_decay = 1e-4
warmup_steps = 10000
num_eval_episodes = 100
mode = 'normal'

scale = 1000

device =  'cuda' if torch.cuda.is_available() else 'cpu'

def eval_episodes(target_rew,env,mode = 'norma'):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew/scale,
                        mode=mode,
                        state_mean=state_mean, # TODO: add these to the model
                        state_std=state_std, # TODO: add these to the model
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn


model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=embed_dim,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4*embed_dim,
        activation_function=activation_function,
        n_positions=1024,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
    )

model = model.to(device=device)

optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

trainer = SequenceTrainer(
    model=model,
    optimizer=optimizer,
    batch_size=batch_size,
    get_batch=get_batch,
    scheduler=scheduler,
    loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
    eval_fns=[eval_episodes(tar) for tar in env_targets],
)


if __name__ == '__main__':
    