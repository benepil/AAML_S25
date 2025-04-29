# ghost.py

import pygame
import random
from settings import CELL_SIZE, GRID_WIDTH, GRID_HEIGHT, UP, DOWN, LEFT, RIGHT, FPS

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Maze is imported dynamically via pacman.py passing it in (not needed to redefine here)

class Ghost:
    def __init__(self, x, y, color, ghost_type="random", scatter_target=(0,0), exit_delay=0):
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.color = color
        self.ghost_type = ghost_type
        self.scatter_target = scatter_target
        self.exit_delay = exit_delay
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.frightened = False
        self.revive_timer = 0
        self.in_house = True
        self.frightened_timer = 0  # 🔥 How long they stay frightened


    def move(self, pacman, scatter_mode, ghosts, maze, ghost_gate_open, ghost_gate_timer):
        if self.frightened:
            self.frightened_timer -= 1
            if self.frightened_timer <= 0:
                self.frightened = False


        if self.exit_delay > 0:
            self.exit_delay -= 1
            return ghost_gate_open, ghost_gate_timer

        if self.revive_timer > 0:
            self.revive_timer -= 1
            return ghost_gate_open, ghost_gate_timer

        if self.in_house:
            if maze[self.y - 1][self.x] in ('0', 'G'):
                self.y -= 1
                if maze[self.y][self.x] == '0':
                    self.in_house = False
                ghost_gate_open = True
                ghost_gate_timer = 30
            elif maze[self.y][self.x-1] in ('0', 'G'):
                self.x -= 1
            elif maze[self.y][self.x+1] in ('0', 'G'):
                self.x += 1
            return ghost_gate_open, ghost_gate_timer

        # 🧠 Always define target_x, target_y
        target_x, target_y = pacman.x, pacman.y

        if self.frightened:
            # Random movement if frightened
            possible_dirs = [UP, DOWN, LEFT, RIGHT]
            random.shuffle(possible_dirs)
            for d in possible_dirs:
                if self.valid_move(d, maze):
                    if self.opposite_direction(d) != self.direction:
                        self.direction = d
                        break
        else:
            if scatter_mode:
                target_x, target_y = self.scatter_target
            else:
                if self.ghost_type == "blinky":
                    target_x, target_y = pacman.x, pacman.y
                elif self.ghost_type == "pinky":
                    # Real Pinky bug fix: if Pac-Man is moving UP, shift also LEFT
                    if pacman.direction == UP:
                        target_x = pacman.x - 4
                        target_y = pacman.y - 4
                    else:
                        target_x = pacman.x + 4 * pacman.direction[0]
                        target_y = pacman.y + 4 * pacman.direction[1]
                elif self.ghost_type == "inky":
                    blinky_list = [g for g in ghosts if g.ghost_type == "blinky"]
                    if blinky_list:
                        blinky = blinky_list[0]
                        tile_ahead_x = pacman.x + 2 * pacman.direction[0]
                        tile_ahead_y = pacman.y + 2 * pacman.direction[1]
                        vec_x = tile_ahead_x - blinky.x
                        vec_y = tile_ahead_y - blinky.y
                        target_x = blinky.x + 2 * vec_x
                        target_y = blinky.y + 2 * vec_y
                    else:
                        target_x, target_y = pacman.x, pacman.y
                elif self.ghost_type == "clyde":
                    distance = abs(self.x - pacman.x) + abs(self.y - pacman.y)
                    if distance > 8:
                        target_x, target_y = pacman.x, pacman.y
                    else:
                        target_x, target_y = 1, GRID_HEIGHT - 2

            # Find best direction (shortest distance to target), but no reverse
            options = []
            for d in [UP, LEFT, DOWN, RIGHT]:  # Arcade priority order
                if self.valid_move(d, maze):
                    if self.opposite_direction(d) != self.direction:
                        nx = self.x + d[0]
                        ny = self.y + d[1]
                        dist = (nx - target_x) ** 2 + (ny - target_y) ** 2
                        options.append((dist, d))

            if options:
                options.sort()
                self.direction = options[0][1]

        # Move one step
        new_x = self.x + self.direction[0]
        new_y = self.y + self.direction[1]

        if new_x < 0:
            new_x = GRID_WIDTH - 1
        elif new_x >= GRID_WIDTH:
            new_x = 0

        if 0 <= new_y < GRID_HEIGHT:
            if maze[new_y][new_x] == '0' or (maze[new_y][new_x] in ('2', 'G') and self.revive_timer > 0):
                self.x = new_x
                self.y = new_y

        return ghost_gate_open, ghost_gate_timer

    def valid_move(self, direction, maze):
        nx = self.x + direction[0]
        ny = self.y + direction[1]
        if nx < 0:
            nx = GRID_WIDTH - 1
        elif nx >= GRID_WIDTH:
            nx = 0
        if 0 <= ny < GRID_HEIGHT:
            return maze[ny][nx] == '0' or maze[ny][nx] in ('2', 'G')
        return False

    def opposite_direction(self, direction):
        return (-direction[0], -direction[1])

    def draw(self, screen):
        color = (0, 0, 255) if self.frightened else self.color
        pygame.draw.circle(screen, color, (self.x * CELL_SIZE + CELL_SIZE//2, self.y * CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2 - 2)

    def respawn(self):
        self.x = self.start_x
        self.y = self.start_y
        self.in_house = True
        self.frightened = False
        self.revive_timer = FPS*2  # 🔥 Wait 2 seconds before moving out again (120 frames at 60FPS)

