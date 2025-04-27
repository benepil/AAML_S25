import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv_QNet(nn.Module):
    def __init__(self, grid_size=20, output_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * grid_size * grid_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))     # [B, 32, 20, 20]
        x = F.relu(self.conv2(x))     # [B, 64, 20, 20]
        x = x.view(x.size(0), -1)     # flatten
        x = F.relu(self.fc1(x))       # fully connected
        return self.fc2(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    

class QTrainer:
    def __init__(self, model, lr, gamma, target_model=None):
        self.model = model
        self.target_model = target_model or model
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        # Single sample?
        is_single = isinstance(state, np.ndarray)

        if is_single:
            state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)
            action = torch.tensor([action], dtype=torch.long).to(DEVICE)
            reward = torch.tensor([reward], dtype=torch.float32).to(DEVICE)
            done = torch.tensor([done], dtype=torch.bool).to(DEVICE)
        else:
            # Fix: convert from [B, 1, 1, 20, 20] to [B, 1, 20, 20]
            state = np.array(state)
            next_state = np.array(next_state)

            if state.ndim == 5:
                state = state.squeeze(2)
                next_state = next_state.squeeze(2)

            state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)
            action = torch.tensor(action, dtype=torch.long).to(DEVICE)
            reward = torch.tensor(reward, dtype=torch.float32).to(DEVICE)
            done = torch.tensor(done, dtype=torch.bool).to(DEVICE)

        # Q(s)
        pred = self.model(state)

        # Q(s')
        with torch.no_grad():
            next_Q = self.target_model(next_state)  # instead of self.model
            max_next_Q = torch.max(next_Q, dim=1)[0]

        # Q target
        target = pred.clone().detach()
        Q_new = reward + self.gamma * max_next_Q * (~done)
        target[range(len(action)), action] = Q_new

        # Train
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
