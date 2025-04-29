#!/usr/bin/env python3
import argparse
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np
import matplotlib.pyplot as plt

from gym_snake import GymSnakeEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO Snake agent")
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="randomize snake initial position and direction during evaluation",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=100,
        help="number of evaluation episodes",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ppo_snake_cnn_final.zip",
        help="path to the trained PPO model",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="ppo_snake_evaluation.png",
        help="filename to save the evaluation plot",
    )
    return parser.parse_args()

def make_env(random_start: bool):
    """Create a headless Snake env capped at 1000 steps, with optional random start."""
    env = GymSnakeEnv(
        render_mode=False,
        step_penalty=-0.01,
        death_penalty=-100.0,
        proximity_bonus=1.0,
        random_start=random_start,
    )
    env = TimeLimit(env, max_episode_steps=1000)
    return Monitor(env)

def evaluate(model: PPO, env: gym.Env, n_episodes: int):
    """Run `n_episodes` with the PPO policy, return array of total rewards."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
        rewards.append(total_reward)
    return np.array(rewards)

def main():
    args = parse_args()

    # 1. Load model & make eval env
    model = PPO.load(args.model_path, device="cuda")
    eval_env = make_env(random_start=args.random_start)

    # 2. Evaluate
    rewards = evaluate(model, eval_env, n_episodes=args.n_episodes)
    mean, std = rewards.mean(), rewards.std()
    print(f"[PPO]   Mean±std reward: {mean:.2f} ± {std:.2f}")

    # 3. Plot returns + running average
    episodes = np.arange(1, args.n_episodes + 1)
    window = min(10, args.n_episodes)
    running_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')

    plt.figure()
    plt.plot(episodes, rewards, label="Episode reward")
    plt.plot(episodes[window-1:], running_avg, label=f"{window}-ep running avg")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Snake Evaluation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_plot)
    print(f"Saved evaluation plot to {args.save_plot}")
    # plt.show()  # uncomment if running with a display

    eval_env.close()

if __name__ == "__main__":
    main()