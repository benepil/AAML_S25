# gym_snake.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from snake_env import SnakeGameEnv

class GymSnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: bool = False,
        step_penalty: float = -0.01,
        death_penalty: float = -100.0,
        proximity_bonus: float = 1.0,
        random_start: bool = False,
    ):
        super().__init__()
        self.env = SnakeGameEnv(
            teleport_walls=False,
            render_mode=render_mode,
            step_penalty=step_penalty,
            death_penalty=death_penalty,
            proximity_bonus=proximity_bonus,
            random_start=random_start,
        )

        H = self.env.HEIGHT // self.env.BLOCK_SIZE
        W = self.env.WIDTH  // self.env.BLOCK_SIZE

        self.observation_space = spaces.Box(0.0, 1.0, (3, H, W), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=None)
        # Simply call reset on the underlying env (it returns a dict),
        # then return our grid observation + empty info
        self.env.reset()
        return self.env.get_grid_state(), {}

    def step(self, action):
        state_dict, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        return self.env.get_grid_state(), reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        pass
