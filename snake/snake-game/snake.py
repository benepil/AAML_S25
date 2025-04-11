import random
import pygame
import torch
import torch.optim as optim
from snake_env import SnakeGameEnv
from agent import CNNQNetwork, ReplayMemory, train_q_network, update_target_network

def select_action(q_network, grid_state, epsilon):
    if random.random() < epsilon:
        return random.choice([0, 1, 2])
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(grid_state, dtype=torch.float32).unsqueeze(0)
            q_values = q_network(state_tensor)
            return q_values.argmax().item()


def main():
    game = SnakeGameEnv(teleport_walls=True)
    grid_size = game.HEIGHT // game.BLOCK_SIZE
    q_network = CNNQNetwork(input_channels=3, grid_size=grid_size, num_actions=3)
    target_network = CNNQNetwork(input_channels=3, grid_size=grid_size, num_actions=3)
    update_target_network(q_network, target_network)

    q_network.train()
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    memory = ReplayMemory(capacity=1000)

    state = game.get_grid_state()

    epsilon = 0.9
    min_epsilon = 0.01
    epsilon_decay = 0.990

    episode = 1
    batch_size = 32
    update_frequency = 5
    episode_scores = []  # To store end-of-episode scores.
    episode_reward = 0
    running = True

    # (Interactive plot setup omitted for brevity if not needed on cluster.)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

        grid_state = game.get_grid_state()
        action = select_action(q_network, grid_state, epsilon)

        next_state_dict, reward, done, _ = game.step(action)
        next_grid_state = game.get_grid_state()
        memory.push((grid_state, action, reward, next_grid_state, done))

        # Accumulate reward, but do NOT print every move.
        episode_reward += reward

        game.render()
        state = next_state_dict

        if done:
            final_score = game.get_state()['score']
            # Log only at the end of the episode:
            print(
                f"\nEpisode {episode} finished. Total Reward: {episode_reward}, Score: {final_score}, Epsilon: {epsilon}\n")
            episode_scores.append(final_score)

            state = game.reset()
            memory = ReplayMemory(capacity=1000)
            episode_reward = 0
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

            if episode % update_frequency == 0:
                update_target_network(q_network, target_network)
                torch.save(q_network.state_dict(), f"cnn_q_network_episode_{episode}.pth")
                print(f"Checkpoint saved at episode {episode}.")

            episode += 1

        train_q_network(q_network, target_network, optimizer, memory, batch_size)


if __name__ == '__main__':
    main()
