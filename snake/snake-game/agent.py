import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Define a CNN-based Q-Network for grid inputs.
class CNNQNetwork(nn.Module):
    def __init__(self, input_channels=3, grid_size=30, num_actions=3):
        super(CNNQNetwork, self).__init__()
        # Increase network capacity with an extra convolutional layer.
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # We assume grid_size remains the same.
        self.fc_input_dim = 64 * grid_size * grid_size
        self.fc = nn.Linear(self.fc_input_dim, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Simple Replay Memory for storing transitions.
class ReplayMemory:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        """
        Saves a transition (state, action, reward, next_state, done).
        If memory is full, remove the oldest transition.
        """
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size):
        """
        Returns a random sample of transitions.
        """
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)

def train_q_network(q_network, target_network, optimizer, memory, batch_size, device, gamma=0.99):
    """
    Samples a mini-batch from replay memory and performs a training update.
    
    Args:
      q_network: The current CNN Q-network model.
      target_network: The target network used for computing next Q-values.
      optimizer: Optimizer for updating the network.
      memory: ReplayMemory instance.
      batch_size: Number of transitions to sample.
      device: torch device ("cuda" or "cpu").
      gamma: Discount factor.
    """
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    done_batch = []

    for state, action, reward, next_state, done in transitions:
        state_batch.append(state)         # state: a 3-channel grid (numpy array)
        action_batch.append(action)
        reward_batch.append(reward)
        next_state_batch.append(next_state)
        done_batch.append(done)

    state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32).to(device)
    action_batch = torch.tensor(action_batch, dtype=torch.int64).to(device)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(device)
    next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32).to(device)
    done_batch = torch.tensor(done_batch, dtype=torch.float32).to(device)

    q_values = q_network(state_batch)  # Shape: [batch_size, num_actions]
    state_action_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

    next_q_values = target_network(next_state_batch)
    max_next_q_values = torch.max(next_q_values, dim=1)[0]

    expected_state_action_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    loss_fn = nn.MSELoss()
    loss = loss_fn(state_action_values, expected_state_action_values.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Training loss:", loss.item())

def update_target_network(q_network, target_network):
    """
    Updates the target network by copying parameters from the Q-network.
    """
    target_network.load_state_dict(q_network.state_dict())
