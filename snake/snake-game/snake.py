# snake.py
import os
import random
import argparse
import torch
import numpy as np

from snake_env import SnakeGameEnv
from agent import CNNQNetwork, ReplayMemory, train_q_network, update_target_network

# counters for exploration debugging
random_choice = 0
not_random_choice = 0

def select_action(q_net, grid, eps, device):
    global random_choice, not_random_choice
    if random.random() < eps:
        random_choice += 1
        return random.choice([0, 1, 2])
    else:
        not_random_choice += 1
        with torch.no_grad():
            tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(device)
            return q_net(tensor).argmax().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--update_frequency", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="snake_rl_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    score_file = os.path.join(args.output_dir, "episode_scores.txt")
    with open(score_file, "w") as f:
        f.write("episode,score\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # headless training: render_mode=False
    game = SnakeGameEnv(teleport_walls=False, render_mode=False)
    grid_size = game.HEIGHT // game.BLOCK_SIZE

    q_net = CNNQNetwork(3, grid_size, 3).to(device)
    tgt_net = CNNQNetwork(3, grid_size, 3).to(device)
    update_target_network(q_net, tgt_net)
    q_net.train(); tgt_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    memory = ReplayMemory(10000)

    eps = 1.0
    eps_decay = 0.995
    eps_min = 0.1
    batch_size = 32

    episode = 1
    total_reward = 0
    state_dict = game.reset()

    while episode <= args.num_episodes:
        old_head = state_dict['snake_head']
        old_food = state_dict['food']
        old_dist = game._get_distance(old_head, old_food)

        grid = game.get_grid_state()
        action = select_action(q_net, grid, eps, device)
        next_state, reward, done, _ = game.step(action)

        new_dist = game._get_distance(next_state['snake_head'], next_state['food'])
        shaped = reward + 0.1 * (old_dist - new_dist)

        next_grid = game.get_grid_state()
        memory.push((grid, action, shaped, next_grid, done))

        total_reward += reward
        print(f"Ep{episode}: A={action} R={reward:.1f} S={shaped:.2f} Score={game.score}")

        # only train once buffer is “warmed up”
        if len(memory) > 1000:
            for _ in range(4):
                train_q_network(q_net, tgt_net, optimizer, memory, batch_size, device)

        if done:
            with open(score_file, "a") as f:
                f.write(f"{episode},{game.score}\n")
            print(f"--> End Ep{episode}: TotalR={total_reward}, eps={eps:.3f}\n")

            if episode % args.update_frequency == 0:
                update_target_network(q_net, tgt_net)
                torch.save(
                    q_net.state_dict(),
                    os.path.join(args.output_dir, f"qn_ep{episode}.pth")
                )
                print(f"[Checkpoint] Ep{episode}")

            episode += 1
            total_reward = 0
            eps = max(eps_min, eps * eps_decay)
            state_dict = game.reset()
        else:
            state_dict = next_state

if __name__ == "__main__":
    main()
