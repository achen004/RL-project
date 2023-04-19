import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class RegularizedPPO(PPO):
    def init(
        self,
        policy: ActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.0,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        n_epochs=10,
        target_kl=None,
        tensorboard_log=None,
        create_eval_env=False,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device='auto',
        _init_setup_model=True,
        l2_reg_coef=0.001  # Add L2 regularization coefficient as a parameter
    ):
        super().init(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            ent_coef,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            n_epochs,
            target_kl,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model
        )

        self.l2_reg_coef = l2_reg_coef

    def compute_loss(self, data, return_backward_elements=False, return_logs=False):
            # Call the original compute_loss function
            loss, policy_loss, value_loss, entropy_loss, approx_kl_div, clip_fraction, logs = super().compute_loss(
                data, return_backward_elements=True, return_logs=True
            )

            # Add L2 regularization to the policy_loss
            l2_reg = torch.tensor(0.0).to(self.device)
            for param in self.policy.actor.parameters():
                l2_reg += torch.sum(param*2)
            policy_loss += self.l2_reg_coef

            # Combine losses
            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            if return_backward_elements:
                return loss, policy_loss, value_loss, entropy_loss, approx_kl_div, clip_fraction
            if return_logs:
                return loss, logs
            return loss

# Example usage:
# env = ...
# model = RegularizedPPO("MlpPolicy", env, l2_reg_coef=0.001)
# model.learn(total_timesteps=100000)