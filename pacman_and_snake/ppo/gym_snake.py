# gym_snake.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from snake_env import SnakeGameEnv

class GymSnakeEnv(gym.Env):
    """
    Snake wrapper that emits a 7×30×30 observation:
      - channels 0–2: snake head/body/food
      - channels 3–4: unused (reserved for Pac-Man)
      - channel 5: unused (reserved for Pac-Man)
      - channel 6: task tag = 0.0 for Snake
    Action space is 4 discrete absolute headings (E,S,W,N), remapped to relative.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=False, **kwargs):
        super().__init__()
        self.env = SnakeGameEnv(render_mode=render_mode, **kwargs)

        # absolute directions in same order as SnakeGameEnv.directions
        self.dirs = [
            (self.env.BLOCK_SIZE, 0),    # East
            (0, self.env.BLOCK_SIZE),    # South
            (-self.env.BLOCK_SIZE, 0),   # West
            (0, -self.env.BLOCK_SIZE),   # North
        ]

        H = self.env.HEIGHT // self.env.BLOCK_SIZE  # 30
        W = self.env.WIDTH  // self.env.BLOCK_SIZE  # 30

        self.observation_space = spaces.Box(0.0, 1.0, (7, H, W), dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self, **kwargs):
        grid3 = self.env.reset()  # shape (3,30,30)
        obs7  = np.zeros((7, *grid3.shape[1:]), dtype=np.float32)
        obs7[:3] = grid3          # snake head/body/food
        obs7[6]  = 0.0            # tag=0 for Snake
        return obs7, {}

    def step(self, action: int):
        # Map 4-way absolute action to 3-way relative {straight,right,left}
        ci = self.dirs.index(self.env.direction)
        ti = action
        diff = (ti - ci) % 4
        if diff == 0:
            rel = 0
        elif diff == 1:
            rel = 1
        elif diff == 3:
            rel = 2
        else:
            rel = 1

        grid3, reward, done, info = self.env.step(rel)
        obs7 = np.zeros((7, *grid3.shape[1:]), dtype=np.float32)
        obs7[:3] = grid3
        obs7[6]  = 0.0
        return obs7, reward, done, False, info

    def render(self, mode="human"):
        self.env.render()
