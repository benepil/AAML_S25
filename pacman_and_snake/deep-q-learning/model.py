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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(128, output_size)

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    

class QTrainer:
    def __init__(self, model, lr, gamma, target_model=None):
        self.model = model
        self.target_model = target_model or model  # fallback to current model
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        self.model.train()
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
            next_Q = self.target_model(next_state)
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
        return loss.item()  
