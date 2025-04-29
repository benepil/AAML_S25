# ghost.py
import pygame
import random
from settings import CELL_SIZE, GRID_WIDTH, GRID_HEIGHT, UP, DOWN, LEFT, RIGHT, FPS

class Ghost:
    def __init__(self, x, y, color, ghost_type="random", scatter_target=(0,0), exit_delay=0):
        self.start_x, self.start_y = x, y
        self.x, self.y = x, y
        self.color = color
        self.ghost_type = ghost_type
        self.scatter_target = scatter_target
        self.exit_delay = exit_delay
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.frightened = False
        self.frightened_timer = 0
        self.revive_timer = 0
        self.in_house = True

    def move(self, pacman, scatter_mode, ghosts, maze, gate_open, gate_timer):
        if self.frightened:
            self.frightened_timer -= 1
            if self.frightened_timer <= 0:
                self.frightened = False

        if self.exit_delay > 0:
            self.exit_delay -= 1
            return gate_open, gate_timer

        if self.revive_timer > 0:
            self.revive_timer -= 1
            return gate_open, gate_timer

        if self.in_house:
            if maze[self.y-1][self.x] in ('0', 'G'):
                self.y -= 1
                if maze[self.y][self.x] == '0':
                    self.in_house = False
                gate_open = True
                gate_timer = FPS
            return gate_open, gate_timer

        if scatter_mode:
            tx, ty = self.scatter_target
        else:
            if self.ghost_type == "blinky":
                tx, ty = pacman.x, pacman.y
            elif self.ghost_type == "pinky":
                if pacman.direction == UP:
                    tx, ty = pacman.x - 4, pacman.y - 4
                else:
                    dx, dy = pacman.direction
                    tx, ty = pacman.x + 4*dx, pacman.y + 4*dy
            elif self.ghost_type == "inky":
                blinky = next((g for g in ghosts if g.ghost_type=="blinky"), None)
                if blinky:
                    ax, ay = pacman.x + 2*pacman.direction[0], pacman.y + 2*pacman.direction[1]
                    vx, vy = ax - blinky.x, ay - blinky.y
                    tx, ty = blinky.x + 2*vx, blinky.y + 2*vy
                else:
                    tx, ty = pacman.x, pacman.y
            elif self.ghost_type == "clyde":
                d = abs(self.x - pacman.x) + abs(self.y - pacman.y)
                if d > 8:
                    tx, ty = pacman.x, pacman.y
                else:
                    tx, ty = 1, GRID_HEIGHT - 2
            else:
                tx, ty = pacman.x, pacman.y

        options = []
        for d in [UP, LEFT, DOWN, RIGHT]:
            nx, ny = self.x + d[0], self.y + d[1]
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and maze[ny][nx] in ('0','2','G'):
                if d != (-self.direction[0], -self.direction[1]):
                    dist = (nx - tx)**2 + (ny - ty)**2
                    options.append((dist, d))
        if options:
            self.direction = min(options, key=lambda x: x[0])[1]

        self.x = (self.x + self.direction[0]) % GRID_WIDTH
        self.y = (self.y + self.direction[1]) % GRID_HEIGHT

        return gate_open, gate_timer

    def draw(self, screen):
        color = (0, 0, 255) if self.frightened else self.color
        pygame.draw.circle(
            screen, color,
            (self.x*CELL_SIZE+CELL_SIZE//2, self.y*CELL_SIZE+CELL_SIZE//2),
            CELL_SIZE//2 - 2
        )

    def respawn(self):
        self.x, self.y = self.start_x, self.start_y
        self.in_house = True
        self.frightened = False
        self.revive_timer = FPS * 2
