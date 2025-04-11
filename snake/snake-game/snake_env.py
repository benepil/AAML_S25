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

        # Setup display, clock, and fonts.
        self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.game_over_font = pygame.font.SysFont("consolas", 50)

        # Define direction vectors in clockwise order: [Right, Down, Left, Up]
        self.directions = [
            (self.BLOCK_SIZE, 0),
            (0, self.BLOCK_SIZE),
            (-self.BLOCK_SIZE, 0),
            (0, -self.BLOCK_SIZE)
        ]

        self.reset()

    def reset(self):
        """
        Resets the game to the initial state and returns the state.
        """
        self.score = 0
        self.snake_pos = [[self.WIDTH // 2, self.HEIGHT // 2]]
        # Start moving downward (index 1 in self.directions is (0, BLOCK_SIZE)).
        self.direction = self.directions[1]
        self.food = self.generate_food()
        return self.get_state()

    def generate_food(self):
        """
        Generates a new food position not occupied by the snake.
        """
        while True:
            x = random.randint(0, (self.WIDTH - self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
            y = random.randint(0, (self.HEIGHT - self.BLOCK_SIZE) // self.BLOCK_SIZE) * self.BLOCK_SIZE
            food_pos = [x, y]
            if food_pos not in self.snake_pos:
                return food_pos

    def _get_distance(self, point1, point2):
        """
        Computes Manhattan distance between two points.
        """
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def get_state(self):
        """
        Returns a dictionary representing the current state.
        """
        return {
            'snake_head': self.snake_pos[0],
            'food': self.food,
            'score': self.score,
            'snake': self.snake_pos,
            'direction': self.direction
        }

    def get_grid_state(self):
        """
        Converts the current state into a 3-channel grid representation.

        Channels:
          - Channel 0: Snake head (1 where the head is, 0 otherwise).
          - Channel 1: Snake body (excluding head).
          - Channel 2: Food.

        The grid dimensions are (grid_H, grid_W), where:
            grid_H = HEIGHT // BLOCK_SIZE,
            grid_W = WIDTH // BLOCK_SIZE.
        """
        grid_H = self.HEIGHT // self.BLOCK_SIZE
        grid_W = self.WIDTH // self.BLOCK_SIZE
        grid = np.zeros((3, grid_H, grid_W), dtype=np.float32)

        # Place snake head in channel 0.
        head = self.snake_pos[0]
        head_x = head[0] // self.BLOCK_SIZE
        head_y = head[1] // self.BLOCK_SIZE
        grid[0, head_y, head_x] = 1.0

        # Place snake body in channel 1.
        for pos in self.snake_pos[1:]:
            bx = pos[0] // self.BLOCK_SIZE
            by = pos[1] // self.BLOCK_SIZE
            grid[1, by, bx] = 1.0

        # Place food in channel 2.
        fx = self.food[0] // self.BLOCK_SIZE
        fy = self.food[1] // self.BLOCK_SIZE
        grid[2, fy, fx] = 1.0

        return grid

    def render(self):
        """
        Renders the game state to the screen.
        """
        self.win.fill((0, 0, 0))
        for pos in self.snake_pos:
            pygame.draw.rect(self.win, (255, 255, 255),
                             pygame.Rect(pos[0], pos[1], self.BLOCK_SIZE, self.BLOCK_SIZE))
        pygame.draw.rect(self.win, (255, 0, 0),
                         pygame.Rect(self.food[0], self.food[1], self.BLOCK_SIZE, self.BLOCK_SIZE))
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.win.blit(score_text, (10, 10))
        pygame.display.update()
        self.clock.tick(10)

    def step(self, action):
        """
        Applies the given action (0=straight, 1=turn right, 2=turn left), updates the game state,
        and returns a tuple: (state, reward, done, info).
        Reward shaping is applied based on the Manhattan distance to food.
        """
        prev_distance = self._get_distance(self.snake_pos[0], self.food)

        # Update direction according to the action.
        current_index = self.directions.index(self.direction)
        if action == 1:
            new_index = (current_index + 1) % 4
            self.direction = self.directions[new_index]
        elif action == 2:
            new_index = (current_index - 1) % 4
            self.direction = self.directions[new_index]
        # Action 0: no change in direction.

        new_head = [
            self.snake_pos[0][0] + self.direction[0],
            self.snake_pos[0][1] + self.direction[1]
        ]

        # Handle wall teleportation.
        if self.teleport_walls:
            new_head[0] %= self.WIDTH
            new_head[1] %= self.HEIGHT
        else:
            if new_head[0] < 0 or new_head[0] >= self.WIDTH or new_head[1] < 0 or new_head[1] >= self.HEIGHT:
                reward = -10
                done = True
                return self.get_state(), reward, done, {}

        # Check self-collision.
        if new_head in self.snake_pos:
            reward = -10
            done = True
            return self.get_state(), reward, done, {}

        new_distance = self._get_distance(new_head, self.food)
        shaped_reward = 0.1 * (prev_distance - new_distance)

        # Insert new head.
        self.snake_pos.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10 + shaped_reward
            self.food = self.generate_food()
        else:
            self.snake_pos.pop()
            reward = shaped_reward

        done = False
        return self.get_state(), reward, done, {}

    def game_over_screen(self):
        """
        Displays a game-over screen.
        """
        self.win.fill((0, 0, 0))
        game_over_text = self.game_over_font.render(f"Game Over! Score: {self.score}", True, (255, 255, 255))
        text_rect = game_over_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.win.blit(game_over_text, text_rect)
        pygame.display.update()
        pygame.time.delay(2000)
