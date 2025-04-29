# pacman_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from game import PacmanGameAI, maze
from settings import GRID_WIDTH, GRID_HEIGHT, UP, DOWN, LEFT, RIGHT

class PacmanEnv(gym.Env):
    """
    6×H×W grid, with:
      • dynamic step penalty: –0.01 * (step_count / max_steps)
      • ghost‐proximity penalty: –0.1 / (dist_to_nearest_normal_ghost + 1)
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()
        self.game        = PacmanGameAI(render_mode=False)
        self.prev_score  = 0
        self.prev_lives  = self.game.lives
        self.prev_dist   = None

        # track steps per episode
        self.step_count      = 0
        self.max_episode_steps = 1000

        H, W = GRID_HEIGHT, GRID_WIDTH
        self.observation_space = spaces.Box(0.0, 1.0, (6, H, W), dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.game.reset()
        self.prev_score = self.game.score
        self.prev_lives = self.game.lives
        self.prev_dist  = self._dist_to_nearest_pellet()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # apply action
        dirs = [UP, DOWN, LEFT, RIGHT]
        self.game.pacman.direction = dirs[action]
        self.game.update()

        # base reward = Δscore
        base = self.game.score - self.prev_score
        self.prev_score = self.game.score

        # dynamic step penalty
        step_pen = -0.01 * (self.step_count / self.max_episode_steps)

        # proximity shaping to pellets
        new_dist = self._dist_to_nearest_pellet()
        pellet_bonus = (self.prev_dist - new_dist) * 0.05
        self.prev_dist = new_dist

        # ghost‐proximity penalty
        ghost_dist = self._dist_to_nearest_normal_ghost()
        ghost_pen  = -0.1 / (ghost_dist + 1.0)

        # total reward
        reward = base + step_pen + pellet_bonus + ghost_pen

        # termination
        died    = (self.game.lives < self.prev_lives)
        cleared = not (self.game.pellets.pellets or self.game.pellets.energizers)
        done    = died or cleared
        self.prev_lives = self.game.lives

        return self._get_obs(), reward, done, False, {}

    def render(self, mode="human"):
        self.game.draw()
        if mode == "human":
            import pygame; pygame.display.flip()

    def _get_obs(self):
        H, W = GRID_HEIGHT, GRID_WIDTH
        obs = np.zeros((6, H, W), dtype=np.float32)

        # C0: walls
        for y in range(H):
            for x in range(W):
                if maze[y][x] == '1':
                    obs[0, y, x] = 1.0

        # C1: pellets
        for x, y in self.game.pellets.pellets:
            obs[1, y, x] = 1.0

        # C2: energizers
        for x, y in self.game.pellets.energizers:
            obs[2, y, x] = 1.0

        # C3: Pac-Man
        px, py = self.game.pacman.x, self.game.pacman.y
        obs[3, py, px] = 1.0

        # C4 & C5: ghosts
        for ghost in self.game.ghosts:
            gx, gy = ghost.x, ghost.y
            if ghost.frightened:
                obs[5, gy, gx] = 1.0
            else:
                obs[4, gy, gx] = 1.0

        return obs

    def _dist_to_nearest_pellet(self):
        head = (self.game.pacman.x, self.game.pacman.y)
        pellets = self.game.pellets.pellets
        if not pellets:
            return 0.0
        dists = [abs(head[0]-x) + abs(head[1]-y) for x, y in pellets]
        return float(min(dists))

    def _dist_to_nearest_normal_ghost(self):
        head = (self.game.pacman.x, self.game.pacman.y)
        # only non‐frightened ghosts
        normals = [(g.x, g.y) for g in self.game.ghosts if not g.frightened]
        if not normals:
            return float('inf')
        dists = [abs(head[0]-x) + abs(head[1]-y) for x, y in normals]
        return float(min(dists))

    def close(self):
        pass
