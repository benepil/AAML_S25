#!/usr/bin/env python3
# train_ppo.py

import argparse
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from pacman_env import PacmanEnv

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_c = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_c, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.post = nn.Sequential(
            nn.Linear(n_flat, features_dim), nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.post(self.cnn(obs))

def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on Pac-Man Env")
    p.add_argument("--steps",  type=int, default=3_000_000, help="Total timesteps")
    p.add_argument("--n-envs", type=int, default=8,         help="Parallel envs")
    p.add_argument("--seed",   type=int, default=0,         help="Random seed")
    return p.parse_args()

def make_env(seed):
    def _init():
        # headless + cap at 1000 steps
        env = TimeLimit(PacmanEnv(), max_episode_steps=1000)
        return env
    torch.manual_seed(seed)
    return _init

def main():
    args = parse_args()

    # vectorized train env
    train_env = SubprocVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
    train_env = VecMonitor(train_env)

    # vectorized eval env
    eval_env = SubprocVecEnv([make_env(args.seed + args.n_envs)])
    eval_env = VecMonitor(eval_env)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=200_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
    )

    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log="./ppo_pacman_tb/",
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        ent_coef=0.02,                               # **higher entropy bonus**
        learning_rate=lambda f: 2.5e-4 * f,          # linear decay
        clip_range=0.1,
        policy_kwargs=policy_kwargs,
    )

    model.learn(
        total_timesteps=args.steps,
        tb_log_name="pacman_run_shaped",
        callback=[ProgressBarCallback(), eval_cb],
    )

    model.save("ppo_pacman_final")
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
