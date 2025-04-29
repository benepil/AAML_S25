# train_ppo.py

#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.wrappers import TimeLimit

from gym_snake import GymSnakeEnv

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]
        self.post = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.post(x)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--random-start", action="store_true",
                   help="randomize snake initial position and direction")
    p.add_argument("--steps", type=int, default=1_000_000,
                   help="total timesteps for training")
    return p.parse_args()

def make_env_fn(random_start: bool):
    def _init():
        env = GymSnakeEnv(
            render_mode=False,
            step_penalty=-0.01,
            death_penalty=-100.0,
            proximity_bonus=1.0,
            random_start=random_start,
        )
        env = TimeLimit(env, max_episode_steps=1000)
        return Monitor(env)
    return _init

def main():
    args = parse_args()

    # 4 parallel envs
    env_fns = [make_env_fn(args.random_start) for _ in range(4)]
    train_env = SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
    )

    model = PPO(
        "CnnPolicy", train_env,
        verbose=1,
        tensorboard_log="./ppo_snake_tb/",
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=3,
        ent_coef=0.01,
        device="cuda",
        policy_kwargs=policy_kwargs,
    )

    # print eval stats periodically
    eval_env = SubprocVecEnv([make_env_fn(args.random_start)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path="./eval_logs/",
        eval_freq=250_000,
        n_eval_episodes=10,
        verbose=1,
    )

    model.learn(
        total_timesteps=args.steps,
        tb_log_name="cnn_run",
        callback=[ProgressBarCallback(), eval_callback],
    )

    model.save("ppo_snake_cnn_final.zip")
    train_env.close()

if __name__ == "__main__":
    main()
