#!/usr/bin/env python3
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym_snake import GymSnakeEnv
from pacman_env import PacmanEnv

# 1) Shared conv body
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=128):
        super().__init__(obs_space, features_dim)
        n_c = obs_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32,64, 3, padding=1),   nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *obs_space.shape)).shape[1]
        self.post = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, x):
        return self.post(self.cnn(x))

# 2) Multi-head policy/value on top
class MultiHeadCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=kwargs.pop("features_extractor_kwargs"),
            **kwargs,
        )
        latent = self.mlp_extractor.latent_dim_pi
        self.policy_snake  = nn.Linear(latent, self.action_space.n)
        self.policy_pacman = nn.Linear(latent, self.action_space.n)
        self.value_snake   = nn.Linear(latent, 1)
        self.value_pacman  = nn.Linear(latent, 1)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # tag channel: 0=Snake, 1=Pac-Man
        is_pm = obs[:, 6, 0, 0].bool()

        # build logits
        logits = torch.zeros((obs.size(0), self.action_space.n), device=obs.device)
        logits[~is_pm] = self.policy_snake(latent_pi[~is_pm])
        logits[ is_pm] = self.policy_pacman(latent_pi[ is_pm])

        # <<< REPLACED HERE >>> use the correct dist builder
        dist = self._get_action_dist_from_latent(logits)

        actions  = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)

        values = torch.zeros(obs.size(0), device=obs.device)
        values[~is_pm] = self.value_snake(latent_vf[~is_pm]).view(-1)
        values[ is_pm] = self.value_pacman(latent_vf[ is_pm]).view(-1)

        return actions, values, log_prob

    def _predict(self, obs, deterministic=False):
        actions, _, _ = self.forward(obs, deterministic)
        return actions, None

# 3) Training script
def parse_args():
    p = argparse.ArgumentParser("Multi-head PPO Snake+Pac-Man")
    p.add_argument("--steps",           type=int, default=3_000_000)
    p.add_argument("--pacman-pretrain", type=int, default=250_000)
    p.add_argument("--n-envs",          type=int, default=8)
    p.add_argument("--seed",            type=int, default=0)
    return p.parse_args()

def make_env(seed, cls):
    def _init():
        env = cls()
        return TimeLimit(env, max_episode_steps=1000)
    torch.manual_seed(seed)
    return _init

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_kwargs = dict(
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
    )

    # Phase 1: Pac-Man only
    pm_envs = [make_env(args.seed + i, PacmanEnv) for i in range(args.n_envs)]
    pm_vec  = DummyVecEnv(pm_envs); pm_vec = VecMonitor(pm_vec)
    model = PPO(
        MultiHeadCnnPolicy, pm_vec,
        verbose=1, seed=args.seed, device=device,
        n_steps=512, batch_size=256, n_epochs=4,
        ent_coef=0.02,
        learning_rate=lambda f: 2.5e-4 * f,
        clip_range=0.1,
        policy_kwargs=policy_kwargs,
    )

    print(f"=== Pac-Man pretrain: {args.pacman_pretrain} steps ===")
    model.learn(
        total_timesteps=args.pacman_pretrain,
        tb_log_name="pm_pretrain",
        callback=[ProgressBarCallback()]
    )
    pm_vec.close()

    # Phase 2: Mixed Snake + Pac-Man
    remain = args.steps - args.pacman_pretrain
    if remain > 0:
        tasks = [GymSnakeEnv, PacmanEnv]
        mix_envs = [
            make_env(args.seed + args.n_envs + i, tasks[i % 2])
            for i in range(args.n_envs)
        ]
        mix_vec = DummyVecEnv(mix_envs); mix_vec = VecMonitor(mix_vec)

        eval_envs = [make_env(args.seed + 100, GymSnakeEnv),
                     make_env(args.seed + 101, PacmanEnv)]
        eval_vec  = DummyVecEnv(eval_envs); eval_vec = VecMonitor(eval_vec)
        eval_cb   = EvalCallback(
            eval_vec,
            best_model_save_path="./best_model/",
            log_path="./eval_logs/",
            eval_freq=200_000,
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        )

        print(f"=== Multi-game training: {remain} steps ===")
        model.set_env(mix_vec)
        model.learn(
            total_timesteps=remain,
            tb_log_name="mix_run",
            callback=[ProgressBarCallback(), eval_cb]
        )
        mix_vec.close()
        eval_vec.close()
    else:
        print("=== No mixed phase (pacman-pretrain == total steps) ===")

    model.save("ppo_multi_final")
    print("Saved final model → ppo_multi_final.zip")

if __name__ == "__main__":
    main()
