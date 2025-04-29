#!/usr/bin/env python3
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import matplotlib.pyplot as plt

from gym_snake import GymSnakeEnv

def make_env():
    """Headless Snake env capped at 1000 steps."""
    env = GymSnakeEnv(render_mode=False)
    return TimeLimit(env, max_episode_steps=1000)

def evaluate_random(env: gym.Env, n_episodes: int = 100):
    """Run `n_episodes` with random actions, return array of total rewards."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
        rewards.append(total_reward)
    return np.array(rewards)

def main():
    env = make_env()
    rewards = evaluate_random(env, n_episodes=100)
    mean, std = rewards.mean(), rewards.std()
    print(f"[Random] Mean±std reward: {mean:.2f} ± {std:.2f}")

    plt.figure()
    plt.hist(rewards, bins=20, edgecolor='k')
    plt.xlabel("Total Reward")
    plt.ylabel("Count")
    plt.title("Random Policy Returns")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
