import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from game import PacmanGameAI, maze
from settings import GRID_WIDTH, GRID_HEIGHT, UP, DOWN, LEFT, RIGHT

# We pad Pac-Man's 21×19 grid to Snake's 30×30
TARGET_H = 600 // 20  # 30 rows
TARGET_W = 600 // 20  # 30 columns

class PacmanEnv(gym.Env):
    """
    Gym wrapper for Pac-Man that returns a 7×30×30 observation:
      - channels 0: walls
      - channel 1: pellets
      - channel 2: energizers
      - channel 3: Pac-Man
      - channel 4: normal ghosts
      - channel 5: frightened ghosts
      - channel 6: task tag = 1.0 for Pac-Man
    Action space: 4 discrete actions (UP, DOWN, LEFT, RIGHT)
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()
        # underlying game
        self.game = PacmanGameAI(render_mode=False)
        self.prev_score = 0
        self.prev_lives = self.game.lives
        self.prev_dist = None
        self.step_count = 0
        self.max_steps = 1000

        # define spaces
        self.observation_space = spaces.Box(
            0.0, 1.0, (7, TARGET_H, TARGET_W), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # reset game
        self.game.reset()
        self.prev_score = self.game.score
        self.prev_lives = self.game.lives
        self.prev_dist = self._dist_to_pellet()
        self.step_count = 0
        # return padded obs
        return self._get_obs(), {}

    def step(self, action):
        # apply action
        dirs = [UP, DOWN, LEFT, RIGHT]
        self.game.pacman.direction = dirs[action]
        self.step_count += 1
        self.game.update()

        # reward shaping
        base = self.game.score - self.prev_score
        self.prev_score = self.game.score

        step_pen = -0.01 * (self.step_count / self.max_steps)
        new_dist = self._dist_to_pellet()
        pellet_bonus = (self.prev_dist - new_dist) * 0.1
        self.prev_dist = new_dist

        ghost_dist = self._dist_to_ghost()
        ghost_pen = -0.05 / (ghost_dist + 1.0)

        reward = base + step_pen + pellet_bonus + ghost_pen

        # done conditions
        died = (self.game.lives < self.prev_lives)
        cleared = not (self.game.pellets.pellets or self.game.pellets.energizers)
        done = died or cleared
        self.prev_lives = self.game.lives

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # build raw 6×H×W
        raw = np.zeros((6, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
        # walls
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if maze[y][x] == '1':
                    raw[0, y, x] = 1.0
        # pellets
        for x, y in self.game.pellets.pellets:
            raw[1, y, x] = 1.0
        # energizers
        for x, y in self.game.pellets.energizers:
            raw[2, y, x] = 1.0
        # pac-man
        px, py = self.game.pacman.x, self.game.pacman.y
        raw[3, py, px] = 1.0
        # ghosts
        for g in self.game.ghosts:
            gx, gy = g.x, g.y
            idx = 5 if g.frightened else 4
            raw[idx, gy, gx] = 1.0
        # pad to 7×30×30
        padded = np.zeros((7, TARGET_H, TARGET_W), dtype=np.float32)
        padded[:6, :GRID_HEIGHT, :GRID_WIDTH] = raw
        # task tag channel
        padded[6, :, :] = 1.0
        return padded

    def _dist_to_pellet(self):
        head = (self.game.pacman.x, self.game.pacman.y)
        pellets = self.game.pellets.pellets
        if not pellets:
            return 0.0
        return float(min(abs(head[0]-x) + abs(head[1]-y) for x, y in pellets))

    def _dist_to_ghost(self):
        head = (self.game.pacman.x, self.game.pacman.y)
        normals = [(g.x, g.y) for g in self.game.ghosts if not g.frightened]
        if not normals:
            return float('inf')
        return float(min(abs(head[0]-x) + abs(head[1]-y) for x, y in normals))

    def render(self, mode="human"):
        self.game.draw()
        if mode == "human":
            import pygame; pygame.display.flip()
