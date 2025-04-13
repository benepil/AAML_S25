import os
import random
import pygame
import torch
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

from snake_env import SnakeGameEnv
from agent import CNNQNetwork, ReplayMemory, train_q_network, update_target_network

# Global counters for debugging.
random_choice = 0
not_random_choice = 0

def select_action(q_network, grid_state, epsilon, device):
    """
    Selects an action using an epsilon-greedy policy.
    
    Args:
      q_network: The CNN Q-network model.
      grid_state: The current state as a 3-channel grid (numpy array).
      epsilon: The exploration rate.
      device: torch device.
    
    Returns:
      An action: 0 (straight), 1 (turn right), or 2 (turn left).
    """
    global random_choice, not_random_choice
    if random.random() < epsilon:
        random_choice += 1
        return random.choice([0, 1, 2])
    else:
        with torch.no_grad():
            not_random_choice += 1
            state_tensor = torch.tensor(grid_state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = q_network(state_tensor)
            return q_values.argmax().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Total number of episodes to run")
    parser.add_argument("--update_frequency", type=int, default=10,
                        help="Update target network and save checkpoint every this many episodes")
    parser.add_argument("--output_dir", type=str, default="../snake_rl_output",
                        help="Directory to save checkpoints and scores")
    parser.add_argument("--plot", action='store_true',
                        help="Display an interactive plot of episode scores")
    args = parser.parse_args()

    # Create output directory if it doesn't exist.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    score_file = os.path.join(args.output_dir, "episode_scores.txt")
    # Create (or overwrite) the score file with a header.
    with open(score_file, "w") as f:
        f.write("episode,score\n")

    # Define the device (GPU if available).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize the game environment.
    game = SnakeGameEnv(teleport_walls=True)
    grid_size = game.HEIGHT // game.BLOCK_SIZE

    # Instantiate the CNN Q-Network and the target network, and move them to GPU if available.
    q_network = CNNQNetwork(input_channels=3, grid_size=grid_size, num_actions=3).to(device)
    target_network = CNNQNetwork(input_channels=3, grid_size=grid_size, num_actions=3).to(device)
    update_target_network(q_network, target_network)

    q_network.train()
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    memory = ReplayMemory(capacity=1000)

    # Get the initial grid state.
    state = game.get_grid_state()
    
    # Epsilon parameters.
    epsilon = 0.95  # Slightly higher starting value.
    epsilon_decay = 0.995  # Slower decay.
    min_epsilon = 0.01

    episode = 1
    batch_size = 32
    episode_reward = 0
    running = True

    if args.plot:
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], marker='o', linestyle='-')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        ax.set_title("Episode Scores Over Time")

    while running and episode <= args.num_episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                break
        
        grid_state = game.get_grid_state()
        action = select_action(q_network, grid_state, epsilon, device)
        next_state_dict, reward, done, _ = game.step(action)
        next_grid_state = game.get_grid_state()
        transition = (grid_state, action, reward, next_grid_state, done)
        memory.push(transition)

        episode_reward += reward
        print(f"Episode {episode}: Action: {action}, Reward: {reward}, Score: {game.get_state()['score']}")

        # Commented out rendering for headless cluster training.
        # game.render()  # [DISABLED for speed on headless cluster]

        state = next_state_dict

        if done:
            with open(score_file, "a") as f:
                f.write(f"{episode},{game.get_state()['score']}\n")
            print(f"\nEpisode {episode} finished. Total Reward: {episode_reward}, Epsilon: {epsilon}\n")
            
            if args.plot:
                try:
                    with open(score_file, "r") as f:
                        lines = f.readlines()[1:]  # Skip header
                    episode_scores = [int(line.split(",")[1].strip()) for line in lines]
                except Exception as e:
                    episode_scores = []
                line.set_xdata(range(1, len(episode_scores) + 1))
                line.set_ydata(episode_scores)
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.01)
            
            state = game.reset()
            memory = ReplayMemory(capacity=1000)
            episode_reward = 0
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            
            if episode % args.update_frequency == 0:
                update_target_network(q_network, target_network)
                checkpoint_path = os.path.join(args.output_dir, f"cnn_q_network_episode_{episode}.pth")
                torch.save(q_network.state_dict(), checkpoint_path)
                print(f"Target network updated and model checkpoint saved at episode {episode}.")
            episode += 1

        train_q_network(q_network, target_network, optimizer, memory, batch_size, device)

    if args.plot:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
