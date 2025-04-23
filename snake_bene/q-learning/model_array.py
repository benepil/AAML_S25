import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

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
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float).to(DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float).to(DEVICE)
        action = torch.tensor(action, dtype=torch.long).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float).to(DEVICE)  # This could be a batch tensor
        done = torch.tensor(done, dtype=torch.bool).to(DEVICE)

        # Handle batch dimensions (if state is 1D, add a batch dimension)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        elif len(state.shape) == 5:
            state = state.squeeze(1)

        # Predict Q values for the current state
        pred = self.model(state)

        # Create target Q values (clone of pred)
        target = pred.clone()

        # Check if 'reward' and 'done' are scalar tensors (0-dim) and handle accordingly
        if reward.dim() == 0:  # scalar case
            reward_value = reward.item()  # Use .item() to get a scalar value
            done_value = done.item()      # Use .item() to get a scalar value
            Q_new = reward_value if done_value else reward_value + self.gamma * torch.max(self.model(next_state))
            target[0][action] = Q_new
        else:  # batch case
            for idx in range(len(reward)):  # Iterate over the batch
                reward_value = reward[idx].item()  # Extract value from tensor
                done_value = done[idx].item()      # Extract value from tensor

                # Compute the new Q value
                if done_value:
                    Q_new = reward_value 
                else:
                    Q_new = reward_value + self.gamma * torch.max(self.model(next_state[idx]))
                target[idx][action[idx]] = Q_new

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()