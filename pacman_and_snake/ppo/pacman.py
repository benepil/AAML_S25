# pacman.py  (pacman_player.py)
import pygame
from settings import CELL_SIZE, GRID_WIDTH, GRID_HEIGHT, YELLOW, UP, DOWN, LEFT, RIGHT

class PacMan:
    def __init__(self):
        self.start_x = 9
        self.start_y = 15
        self.x = self.start_x
        self.y = self.start_y
        self.direction = (0, 0)

    def move(self, maze):
        new_x = self.x + self.direction[0]
        new_y = self.y + self.direction[1]

        if new_x < 0:
            new_x = GRID_WIDTH - 1
        elif new_x >= GRID_WIDTH:
            new_x = 0

        if 0 <= new_y < GRID_HEIGHT:
            if maze[new_y][new_x] == '0':
                self.x = new_x
                self.y = new_y

    def draw(self, screen):
        pygame.draw.circle(
            screen, YELLOW,
            (self.x*CELL_SIZE+CELL_SIZE//2, self.y*CELL_SIZE+CELL_SIZE//2),
            CELL_SIZE//2 - 2
        )
