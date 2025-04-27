import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from helper import plot
from model_array import Conv_QNet, QTrainer
from copy import deepcopy

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BOARD_RATIO = (18,18)
CELL_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Conv_QNet(grid_size=BOARD_RATIO[0]+2, output_size=3).to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.target_model = deepcopy(self.model)
        self.target_update_freq = 100  # update every 100 games


    def get_state(self, game):
        map_data = np.zeros((BOARD_RATIO[0]+2)*(BOARD_RATIO[1]+2))
        for pt in game.snake:
            map_data[int(pt.x/CELL_SIZE)+int(pt.y/CELL_SIZE)*(BOARD_RATIO[0]+1)]=3/3
        map_data[int(game.snake[0].x/CELL_SIZE)+int(game.snake[0].y/CELL_SIZE)*(BOARD_RATIO[0]+1)]=1/3
        map_data[int(game.food.x/CELL_SIZE)+int(game.food.y/CELL_SIZE)*(BOARD_RATIO[0]+1)]=2/3
        grid = np.reshape(map_data, ((BOARD_RATIO[0]+2),(BOARD_RATIO[1]+2)))  # now shape is (20, 20)
        grid = np.expand_dims(grid, axis=0)  # [1, 20, 20]
        grid = np.expand_dims(grid, axis=0)  # [1, 1, 20, 20]

        return grid

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE * 5:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        actions = [np.argmax(a) for a in actions]  
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        action = np.argmax(action)
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(10, 80 - self.n_games // 10)
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(DEVICE)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    total_reward = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(w=BOARD_RATIO[0]*CELL_SIZE, h=BOARD_RATIO[1]*CELL_SIZE)
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        total_reward += reward
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            train_repeats = 1 + len(agent.memory) // (BATCH_SIZE * 5)
            for _ in range(min(train_repeats, 10)):
                agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Reward:', total_reward)
            total_reward=0
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

        if agent.n_games % agent.target_update_freq == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())


if __name__ == '__main__':
    train()