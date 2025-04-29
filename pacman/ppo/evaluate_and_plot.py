#!/usr/bin/env python3
# evaluate_and_plot.py

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from pacman_env import PacmanEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO Pac-Man agent")
    parser.add_argument(
        "--model_path", type=str, default="ppo_pacman_final.zip",
        help="Path to the trained PPO model"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=100,
        help="Number of episodes to evaluate over"
    )
    parser.add_argument(
        "--save_plot", type=str, default="ppo_pacman_eval.png",
        help="Filename for the rewards plot"
    )
    return parser.parse_args()


def make_env(seed):
    def _init():
        # headless Pac-Man + cap at 1000 steps
        env = TimeLimit(PacmanEnv(), max_episode_steps=1000)
        return env
    np.random.seed(seed)
    return _init


def evaluate(model, env, n_episodes):
    rewards = []
    for _ in range(n_episodes):
        # reset may return (obs, info)
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs = reset_out[0]
        else:
            obs = reset_out
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            # step may return 4- or 5-tuple
            if len(step_out) == 4:
                obs, reward, done, _ = step_out
            else:
                obs, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            total_reward += float(reward)
        rewards.append(total_reward)
    return np.array(rewards)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(args.model_path, device=device)

    # create single-env VecMonitor
    eval_env = SubprocVecEnv([make_env(0)])
    eval_env = VecMonitor(eval_env)

    rewards = evaluate(model, eval_env, args.n_episodes)
    mean, std = rewards.mean(), rewards.std()
    print(f"Mean±std reward over {args.n_episodes} episodes: {mean:.2f} ± {std:.2f}")

    episodes = np.arange(1, args.n_episodes + 1)
    window = min(10, args.n_episodes)
    running_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")

    plt.figure()
    plt.plot(episodes, rewards, label="Episode reward")
    plt.plot(episodes[window-1:], running_avg, label=f"{window}-ep running avg")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Pacman Evaluation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_plot)
    print(f"Saved plot to {args.save_plot}")

    eval_env.close()

if __name__ == "__main__":
    main()
