import pygame
import random
import numpy as np


class SnakeGameEnv:
    def __init__(self, width=600, height=600, block_size=20, teleport_walls=True):
        pygame.init()
        pygame.font.init()
        self.WIDTH = width
        self.HEIGHT = height
        self.BLOCK_SIZE = block_size
        self.teleport_walls = teleport_walls

        self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
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
        self.snake_pos = [[self.WIDTH // 2, self.HEIGHT // 2]]
        self.direction = self.directions[1]
        self.food = self.generate_food()
        return self.get_state()

    def generate_food(self):
        while True:
            x = random.randint(0, (self.WIDTH - self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
            y = random.randint(0, (self.HEIGHT - self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
            food_pos = [x, y]
            if food_pos not in self.snake_pos:
                return food_pos

    def _get_distance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

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
        grid_W = self.WIDTH // self.BLOCK_SIZE
        grid = np.zeros((3, grid_H, grid_W), dtype=np.float32)

        head = self.snake_pos[0]
        head_x = head[0] // self.BLOCK_SIZE
        head_y = head[1] // self.BLOCK_SIZE
        grid[0, head_y, head_x] = 1.0

        for pos in self.snake_pos[1:]:
            bx = pos[0] // self.BLOCK_SIZE
            by = pos[1] // self.BLOCK_SIZE
            grid[1, by, bx] = 1.0

        fx = self.food[0] // self.BLOCK_SIZE
        fy = self.food[1] // self.BLOCK_SIZE
        grid[2, fy, fx] = 1.0

        return grid

    def render(self):
        self.win.fill((0, 0, 0))
        for pos in self.snake_pos:
            pygame.draw.rect(self.win, (255, 255, 255),
                             pygame.Rect(pos[0], pos[1], self.BLOCK_SIZE, self.BLOCK_SIZE))
        pygame.draw.rect(self.win, (255, 0, 0),
                         pygame.Rect(self.food[0], self.food[1], self.BLOCK_SIZE, self.BLOCK_SIZE))
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.win.blit(score_text, (10, 10))
        pygame.display.update()
        # self.clock.tick(5)

    def step(self, action):
        current_index = self.directions.index(self.direction)
        if action == 1:
            new_index = (current_index + 1) % 4
            self.direction = self.directions[new_index]
        elif action == 2:
            new_index = (current_index - 1) % 4
            self.direction = self.directions[new_index]

        new_head = [self.snake_pos[0][0] + self.direction[0],
                    self.snake_pos[0][1] + self.direction[1]]

        if self.teleport_walls:
            new_head[0] %= self.WIDTH
            new_head[1] %= self.HEIGHT
        else:
            if new_head[0] < 0 or new_head[0] >= self.WIDTH or new_head[1] < 0 or new_head[1] >= self.HEIGHT:
                return self.get_state(), -100, True, {}

        if new_head in self.snake_pos:
            return self.get_state(), -100, True, {}

        self.snake_pos.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 100
            self.food = self.generate_food()
        else:
            self.snake_pos.pop()
            reward = -1

        return self.get_state(), reward, False, {}

    def game_over_screen(self):
        self.win.fill((0, 0, 0))
        game_over_text = self.game_over_font.render(f"Game Over! Score: {self.score}", True, (255, 255, 255))
        text_rect = game_over_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.win.blit(game_over_text, text_rect)
        pygame.display.update()
        pygame.time.delay(2000)