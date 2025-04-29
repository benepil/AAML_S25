import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from helper import plot, plot_reward
from model import Conv_QNet, QTrainer
from copy import deepcopy
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
ITERATIONS = 10000
LR = 0.001
BOARD_RATIO = (19,17)
CELL_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Conv_QNet(grid_size=BOARD_RATIO[0]+2, output_size=4).to(DEVICE)
        self.target_model = deepcopy(self.model)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, target_model=self.target_model)
        self.N_STEP = 3  # or 5
        self.n_step_buffer = deque(maxlen=self.N_STEP)


    def get_state(self, game):

        cols = BOARD_RATIO[0] + 2
        rows = BOARD_RATIO[1] + 2
        map_data = np.zeros((rows, cols))

        for pt in game.snake:
            map_data[int(pt.y / CELL_SIZE), int(pt.x / CELL_SIZE)] = 1.0

        map_data[int(game.food.y / CELL_SIZE), int(game.food.x / CELL_SIZE)] = 2.0
        map_data[int(game.snake[0].y / CELL_SIZE), int(game.snake[0].x / CELL_SIZE)] = 3.0

        # Normalize and reshape:
        map_data /= 3.0
        grid = map_data[np.newaxis, np.newaxis, :, :]

        return grid

    def remember(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.N_STEP:
            state_n, action_n, _, _, _ = self.n_step_buffer[0]
            R, next_s, done_flag = self._get_n_step_info()
            self.memory.append((state_n, action_n, R, next_s, done_flag))


    def train_long_memory(self):
        if len(self.memory) > max(BATCH_SIZE, 10):
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        actions = [np.argmax(a) for a in actions]  
        return self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        action = np.argmax(action)
        return self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, eval=False):
        # Slow decay with initial exploration freeze
        if eval:
            epsilon = 0  # No randomness in eval
        elif self.n_games < 500:
            epsilon = 1.0
        else:
            epsilon = max(0.05, 0.995 ** (self.n_games - 500))

        final_move = [0, 0, 0, 0]
        if random.random() < epsilon:
            move = random.randint(0, 2)
        else:
            q_vals = self.model(torch.tensor(state, dtype=torch.float).to(DEVICE))[0]
            probs = torch.softmax(q_vals, dim=0).cpu().detach().numpy()
            move = np.random.choice(4, p=probs)
        final_move = [0, 0, 0, 0]
        final_move[move] = 1

        return final_move

    def _get_n_step_info(self):
        R = 0
        next_state, done = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for idx, (_, _, reward, _, _) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * reward
        return R, next_state, done

def plot_eval_scores(scores, agent_name):
    plt.plot(scores, label='Eval Scores')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.title(f"Evaluation Scores: {agent_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"eval/{agent_name}_score_plot.png")
    plt.close()

def evaluate(agent, model_path="model/model.pth", n_games=10):
    agent_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\n[ Evaluation Mode ] \n {agent_name} \n")
    eval_file = f"eval/{agent_name}_eval.txt"
    os.makedirs("eval", exist_ok=True)
    open(eval_file, "w").close()
    
    agent.model.eval()
    total_score = 0
    eval_scores = []
    all_actions = []

    for game_id in range(1, n_games + 1):
        game = SnakeGameAI(w=BOARD_RATIO[0]*CELL_SIZE, h=BOARD_RATIO[1]*CELL_SIZE)
        done = False
        steps = 0
        score = 0

        while not done:
            state = agent.get_state(game)
            final_move = agent.get_action(state, eval=True)
            while game.check_action(final_move) != True:
                final_move = agent.get_action(state, eval=True)
            state_tensor = torch.tensor(state, dtype=torch.float).to(DEVICE)
            with torch.no_grad():
                q_vals = agent.model(state_tensor).cpu().numpy().flatten()
            move = np.argmax(final_move)
            all_actions.append(move)

            with open(eval_file, "a") as f:
                f.write(f"[Game {game_id}] Step {steps} - Q-values: {q_vals} - Move: {final_move}\n")

            reward, done, score = game.play_step(final_move)
            steps += 1

        eval_scores.append(score)
        total_score += score
        with open(eval_file, "a") as f:
            f.write(f"Game {game_id} finished. Score: {score}, Steps Survived: {steps}\n")

    avg_score = total_score / n_games
    action_dist = Counter(all_actions)

    # Write final stats
    with open(eval_file, "a") as f:
        f.write(f"\nAverage Score over {n_games} games: {avg_score}\n")
        f.write(f"Action Distribution: {dict(action_dist)}\n")

    # Save plot
    plot_eval_scores(eval_scores, agent_name)

    # Save CSV summary
    summary = {
        "model": agent_name,
        "avg_score": avg_score,
        "min_score": min(eval_scores),
        "max_score": max(eval_scores),
        "std_dev": np.std(eval_scores),
        "action_0": action_dist.get(0, 0),
        "action_1": action_dist.get(1, 0),
        "action_2": action_dist.get(2, 0),
        "action_3": action_dist.get(2, 0),
    }
    df = pd.DataFrame([summary])
    csv_path = "eval/eval_summary.csv"
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    agent.model.train()


def train(model=None):
    plot_mean_scores = []
    plot_losses = []
    total_score = 0
    plot_total_rewards = []
    plot_mean_rewards = []
    plot_scores = []
    total_reward = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(w=BOARD_RATIO[0]*CELL_SIZE, h=BOARD_RATIO[1]*CELL_SIZE)
    x=0

    if model:
        resume_from = f"model/model_{model}.pth"
        if os.path.exists(resume_from):
            agent.model.load_state_dict(torch.load(resume_from, map_location=DEVICE))
            agent.target_model.load_state_dict(agent.model.state_dict())  # sync target model
            agent.n_games = int(model)
            print(f"✅ Loaded model from {resume_from}")

    for _ in range(1000):
        state = agent.get_state(game)
        action = random.choice([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
        while game.check_action(action) != True:
            action = random.choice([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
        reward, done, score = game.play_step(action, agent.n_games)
        next_state = agent.get_state(game)
        agent.remember(state, action, reward, next_state, done)
        if done:
            game.reset()

    with open("eval/log_file", "w") as f:
        f.write("game,score,record,mean_score,total_reward,loss\n")

    while x < ITERATIONS:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        while game.check_action(final_move) != True:
            final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move, agent.n_games)
        total_reward += reward
        state_new = agent.get_state(game)

        # train short memory
        loss = agent.train_short_memory(state_old, final_move, reward, state_new, done)
        if loss is not None:
            plot_losses.append(loss)


        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if agent.n_games % 50 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        if done:
            agent.n_step_buffer.clear()
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            for _ in range(3):
                loss = agent.train_long_memory()
                if loss is not None:
                    plot_losses.append(loss)

            if score > record:
                record = score
                agent.model.save()
            
            plot_total_rewards.append(total_reward)
            mean_reward = sum(plot_total_rewards) / len(plot_total_rewards)
            plot_mean_rewards.append(mean_reward)
            
            plot_scores.append(score)
            total_score += score
            if model:
                mean_score = total_score / (agent.n_games-int(model))
            else:
                mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            loss_str = f"{loss:.4f}" if loss is not None else "NA"
            log_entry = f"{agent.n_games},{score},{record},{mean_score:.2f},{total_reward:.2f},{loss_str}\n"
            with open("log_file", "a") as f:
                f.write(log_entry)
            if agent.n_games % 500 == 0:
                model_path = f"model/model_{agent.n_games}.pth"
                torch.save(agent.model.state_dict(), model_path)
                print(f"[Model Saved] → {model_path}")  
                evaluate(agent, f"model_{agent.n_games}")
            x+=1

    plot_reward(plot_total_rewards, plot_mean_rewards)
    


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--games")
    parser.add_argument("--mode")
    parser.add_argument("--model")
    args=parser.parse_args()

    if args.mode:
        if args.mode=="train":
            if args.games:
                ITERATIONS = int(args.games)
            if args.model:
                train(args.model)
            train()
        elif args.mode=="eval":
            games = 100
            if args.games:
                games = int(args.games)
            if args.model:
                agent = Agent()
                agent.model.load_state_dict(torch.load(os.path.join(f"model/model_{args.model}.pth"), map_location=DEVICE))
                evaluate(agent, os.path.splitext(args.model)[0],games)
            else:
                for root, _, files in os.walk("model/"):  
                    for filename in files:  # loop through files in the current directory
                        agent = Agent()
                        agent.model.load_state_dict(torch.load(os.path.join(root, filename), map_location=DEVICE))
                        evaluate(agent, os.path.splitext(filename)[0],100)

    else:
        train()