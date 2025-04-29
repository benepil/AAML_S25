#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from gym_snake import GymSnakeEnv
from pacman_env import PacmanEnv

def parse_args():
    p = argparse.ArgumentParser("Manual eval for Multi‐Game PPO")
    p.add_argument("--model_path", type=str, default="ppo_multi_final.zip")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--save_plot", type=str, default="ppo_multi_eval.png")
    return p.parse_args()

def evaluate_task(model, TaskEnv, n_episodes, device):
    env = TimeLimit(TaskEnv(), max_episode_steps=1000)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            # turn obs into a torch tensor, batch size=1
            obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                # forward returns (actions, values, log_prob)
                actions_tensor, _, _ = model.policy.forward(obs_tensor, deterministic=True)
            action = int(actions_tensor[0].cpu().numpy())
            out = env.step(action)
            # support both Gym and Gymnasium 4/5‐tuples
            if len(out) == 5:
                obs, reward, term, trunc, _ = out
                done = term or trunc
            else:
                obs, reward, done, _ = out
            total += float(reward)
        rewards.append(total)
    env.close()
    return np.array(rewards)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(args.model_path, device=device)

    snake_rs   = evaluate_task(model, GymSnakeEnv,  args.n_episodes, device)
    pacman_rs  = evaluate_task(model, PacmanEnv,    args.n_episodes, device)

    s_mean, s_std = snake_rs.mean(),  snake_rs.std()
    p_mean, p_std = pacman_rs.mean(), pacman_rs.std()
    print(f"Snake   Mean±std: {s_mean:.2f} ± {s_std:.2f}")
    print(f"Pac-Man Mean±std: {p_mean:.2f} ± {p_std:.2f}")

    # plot
    eps = np.arange(1, args.n_episodes + 1)
    win = min(10, args.n_episodes)
    run_s = np.convolve(snake_rs,  np.ones(win)/win, mode="valid")
    run_p = np.convolve(pacman_rs, np.ones(win)/win, mode="valid")

    plt.figure()
    plt.plot(eps, snake_rs,  label="Snake")
    plt.plot(eps[win-1:], run_s, label=f"Snake {win}-ep avg")
    plt.plot(eps, pacman_rs, label="Pac-Man")
    plt.plot(eps[win-1:], run_p,label=f"Pac-Man {win}-ep avg")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Multi‐Game Evaluation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_plot)
    print(f"Saved plot to {args.save_plot}")

if __name__ == "__main__":
    main()
