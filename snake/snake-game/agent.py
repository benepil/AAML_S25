# agent.py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# CNN Q-Network with one pooling layer
class CNNQNetwork(nn.Module):
    def __init__(self, input_channels=3, grid_size=30, num_actions=3):
        super(CNNQNetwork, self).__init__()
        # convolutional feature extractor
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # reduce spatial dims by 2×
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # compute flattened size after pooling
        pooled_size = (grid_size // 2)
        self.fc_input_dim = 64 * pooled_size * pooled_size
        # final Q-value head
        self.fc = nn.Linear(self.fc_input_dim, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Simple Replay Memory
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)

# Training with Double DQN
def train_q_network(q_network, target_network, optimizer, memory, batch_size, device, gamma=0.99):
    if len(memory) < batch_size:
        return

    # sample a batch
    transitions = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)

    state_batch = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    reward_batch = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_state_batch = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    done_batch = torch.tensor(dones, dtype=torch.float32).to(device)

    # current Q values
    q_values = q_network(state_batch)
    state_action_values = q_values.gather(1, action_batch).squeeze(1)

    # Double DQN: online net picks next action, target net evaluates it
    with torch.no_grad():
        next_q_online = q_network(next_state_batch)
        next_actions = next_q_online.argmax(dim=1, keepdim=True)
        next_q_target = target_network(next_state_batch)
        max_next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
        expected_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    # Huber loss for stability
    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(state_action_values, expected_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # log training info
    #print(f"[TRAIN] loss={loss.item():.4f}  buffer={len(memory)}")

def update_target_network(q_network, target_network):
    target_network.load_state_dict(q_network.state_dict())
