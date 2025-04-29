# evaluate_comparison_debug.py

import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gym_snake import GymSnakeEnv

def evaluate_random(env, n_episodes=100):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += float(r)
        rewards.append(total_r)
    return np.array(rewards)

def evaluate_ppo(model, env, n_episodes=100):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += float(r)
        rewards.append(total_r)
    return np.array(rewards)

def main():
    # 1) Random baseline
    rnd_env = Monitor(TimeLimit(GymSnakeEnv(render_mode=False), max_episode_steps=1000))
    rnd_rewards = evaluate_random(rnd_env, n_episodes=100)
    print(f"[Random] Mean±std reward: {rnd_rewards.mean():.2f} ± {rnd_rewards.std():.2f}")

    # 2) PPO baseline
    ppo_model = PPO.load("ppo_snake_cnn_final.zip", device="cuda")
    ppo_env = Monitor(TimeLimit(GymSnakeEnv(render_mode=False), max_episode_steps=1000))
    ppo_rewards = evaluate_ppo(ppo_model, ppo_env, n_episodes=100)
    print(f"[PPO]    Mean±std reward: {ppo_rewards.mean():.2f} ± {ppo_rewards.std():.2f}")

    # 3) Histogram — for headless: save to file
    plt.figure()
    bins = np.linspace(min(rnd_rewards.min(), ppo_rewards.min()),
                       max(rnd_rewards.max(), ppo_rewards.max()), 20)
    plt.hist(rnd_rewards, bins=bins, alpha=0.6, label='Random', edgecolor='k')
    plt.hist(ppo_rewards, bins=bins, alpha=0.6, label='PPO',    edgecolor='k')
    plt.legend()
    plt.title("Random vs PPO Returns")
    plt.xlabel("Total Reward")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("hist.png")
    print("Histogram saved to hist.png")
    # plt.show()  # enable if running with a display

if __name__ == "__main__":
    main()
