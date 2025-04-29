#!/usr/bin/env python3
# evaluate_comparison.py

import numpy as np
import matplotlib.pyplot as plt

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from gym_snake import GymSnakeEnv

def make_env():
    """Create a headless Snake env capped at 1000 steps, monitored for stats."""
    env = GymSnakeEnv(render_mode=False)
    env = TimeLimit(env, max_episode_steps=1000)
    return Monitor(env)

def evaluate_ppo(model, env, n_episodes=100):
    """Run deterministic PPO policy for n_episodes."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += float(reward)
        rewards.append(total_r)
    return np.array(rewards)

def evaluate_random(env, n_episodes=100):
    """Run uniform-random policy for n_episodes."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += float(reward)
        rewards.append(total_r)
    return np.array(rewards)

def main():
    # 1) Prepare env & model
    eval_env = make_env()
    ppo = PPO.load("ppo_snake_cnn_final.zip", device="cuda")

    # 2) Collect rewards
    n_eps = 100
    ppo_rewards = evaluate_ppo(ppo, eval_env, n_episodes=n_eps)
    rnd_rewards = evaluate_random(eval_env, n_episodes=n_eps)

    # 3) Print stats
    print(f"[PPO]    Mean±std reward: {ppo_rewards.mean():.2f} ± {ppo_rewards.std():.2f}")
    print(f"[Random] Mean±std reward: {rnd_rewards.mean():.2f} ± {rnd_rewards.std():.2f}")

    # 4) Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram comparison
    bins = np.linspace(min(rnd_rewards.min(), ppo_rewards.min()),
                       max(rnd_rewards.max(), ppo_rewards.max()), 20)
    ax1.hist(rnd_rewards, bins=bins, alpha=0.6, label='Random', edgecolor='k')
    ax1.hist(ppo_rewards, bins=bins, alpha=0.6, label='PPO',    edgecolor='k')
    ax1.set_title("Return Distribution")
    ax1.set_xlabel("Total Reward")
    ax1.set_ylabel("Count")
    ax1.legend()

    # Episode-wise + running avg
    episodes = np.arange(1, n_eps+1)
    window   = 10
    ppo_ra   = np.convolve(ppo_rewards, np.ones(window)/window, mode='valid')
    rnd_ra   = np.convolve(rnd_rewards, np.ones(window)/window, mode='valid')

    ax2.plot(episodes,     ppo_rewards,     label='PPO reward',    linestyle='-')
    ax2.plot(episodes,     rnd_rewards,     label='Random reward', linestyle='--')
    ax2.plot(episodes[window-1:], ppo_ra,   label=f'PPO {window}-ep avg',    linewidth=2)
    ax2.plot(episodes[window-1:], rnd_ra,   label=f'Rand {window}-ep avg',   linewidth=2)
    ax2.set_title("Episode Rewards & Running Averages")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
