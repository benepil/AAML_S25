# snake_env.py

import pygame
import random
import numpy as np

class SnakeGameEnv:
    def __init__(
        self,
        width=600,
        height=600,
        block_size=20,
        teleport_walls=False,
        render_mode=False,
        step_penalty: float = -0.01,
        death_penalty: float = -100.0,
        proximity_bonus: float = 1.0,
        random_start: bool = False,    # new flag
    ):
        pygame.init()
        pygame.font.init()

        self.WIDTH = width
        self.HEIGHT = height
        self.BLOCK_SIZE = block_size
        self.teleport_walls = teleport_walls
        self.render_mode = render_mode

        self.step_penalty = step_penalty
        self.death_penalty = death_penalty
        self.proximity_bonus = proximity_bonus
        self.random_start = random_start    # store flag

        if self.render_mode:
            self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        else:
            self.win = pygame.Surface((self.WIDTH, self.HEIGHT))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.game_over_font = pygame.font.SysFont("consolas", 50)

        self.directions = [
            (self.BLOCK_SIZE, 0),
            (0, self.BLOCK_SIZE),
            (-self.BLOCK_SIZE, 0),
            (0, -self.BLOCK_SIZE)
        ]

        self.reset()

    def reset(self):
        self.score = 0

        if self.random_start:
            # random position anywhere on grid
            grid_W = self.WIDTH // self.BLOCK_SIZE
            grid_H = self.HEIGHT // self.BLOCK_SIZE
            start_x = random.randint(0, grid_W - 1) * self.BLOCK_SIZE
            start_y = random.randint(0, grid_H - 1) * self.BLOCK_SIZE
            self.snake_pos = [[start_x, start_y]]
            # random initial heading
            self.direction = random.choice(self.directions)
        else:
            # original center+down start
            self.snake_pos = [[self.WIDTH // 2, self.HEIGHT // 2]]
            self.direction = self.directions[1]

        self.food = self.generate_food()
        return self.get_state()

    def generate_food(self):
        while True:
            x = random.randint(0, (self.WIDTH - self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
            y = random.randint(0, (self.HEIGHT - self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
            if [x, y] not in self.snake_pos:
                return [x, y]

    def _get_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def get_state(self):
        return {
            'snake_head': self.snake_pos[0],
            'food': self.food,
            'score': self.score,
            'snake': self.snake_pos,
            'direction': self.direction
        }

    def get_grid_state(self):
        grid_H = self.HEIGHT // self.BLOCK_SIZE
        grid_W = self.WIDTH  // self.BLOCK_SIZE
        grid = np.zeros((3, grid_H, grid_W), dtype=np.float32)

        head = self.snake_pos[0]
        grid[0, head[1] // self.BLOCK_SIZE, head[0] // self.BLOCK_SIZE] = 1.0
        for pos in self.snake_pos[1:]:
            grid[1, pos[1] // self.BLOCK_SIZE, pos[0] // self.BLOCK_SIZE] = 1.0
        grid[2, self.food[1] // self.BLOCK_SIZE, self.food[0] // self.BLOCK_SIZE] = 1.0

        return grid

    def render(self):
        if not self.render_mode:
            return
        self.win.fill((0, 0, 0))
        for pos in self.snake_pos:
            pygame.draw.rect(self.win, (255, 255, 255),
                             pygame.Rect(pos[0], pos[1], self.BLOCK_SIZE, self.BLOCK_SIZE))
        pygame.draw.rect(self.win, (255, 0, 0),
                         pygame.Rect(self.food[0], self.food[1], self.BLOCK_SIZE, self.BLOCK_SIZE))
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.win.blit(score_text, (10, 10))
        pygame.display.update()

    def step(self, action):
        old_head = self.snake_pos[0].copy()
        old_dist = self._get_distance(old_head, self.food)

        idx = self.directions.index(self.direction)
        if action == 1:
            idx = (idx + 1) % 4
        elif action == 2:
            idx = (idx - 1) % 4
        self.direction = self.directions[idx]

        new_head = [
            old_head[0] + self.direction[0],
            old_head[1] + self.direction[1]
        ]

        if not self.teleport_walls:
            if (new_head[0] < 0 or new_head[0] >= self.WIDTH or
                new_head[1] < 0 or new_head[1] >= self.HEIGHT):
                return self.get_state(), self.death_penalty, True, {}
        else:
            new_head[0] %= self.WIDTH
            new_head[1] %= self.HEIGHT

        if new_head in self.snake_pos:
            return self.get_state(), self.death_penalty, True, {}

        new_dist = self._get_distance(new_head, self.food)
        shape_reward = (old_dist - new_dist) * 0.1

        self.snake_pos.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            reward = 100.0 + shape_reward
            self.food = self.generate_food()
        else:
            self.snake_pos.pop()
            reward = shape_reward + self.step_penalty
            if shape_reward > 0:
                reward += self.proximity_bonus

        return self.get_state(), float(reward), False, {}

    def game_over_screen(self):
        if not self.render_mode:
            return
        self.win.fill((0, 0, 0))
        go_text = self.game_over_font.render(f"Game Over! Score: {self.score}", True, (255, 255, 255))
        rect = go_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.win.blit(go_text, rect)
        pygame.display.update()
        pygame.time.delay(2000)
