# game.py

import pygame
import sys
from ghost import Ghost
from pellet_manager import PelletManager
from pacman import PacMan
from settings import CELL_SIZE, GRID_WIDTH, GRID_HEIGHT, UP, DOWN, LEFT, RIGHT, FPS

# Constants
WIDTH = GRID_WIDTH * CELL_SIZE
HEIGHT = GRID_HEIGHT * CELL_SIZE
YELLOW = (255, 255, 0)
BLUE   = (0,   0, 255)

# Maze layout
maze = [
    ['1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1'],
    ['1','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','1'],
    ['1','0','1','1','0','1','1','1','0','1','0','1','1','1','0','1','1','0','1'],
    ['1','0','0','0','0','1','0','0','0','1','0','0','0','1','0','0','0','0','1'],
    ['1','1','0','1','0','1','0','1','0','1','0','1','0','1','0','1','0','1','1'],
    ['1','0','0','1','0','0','0','1','0','0','0','1','0','0','0','1','0','0','1'],
    ['1','0','1','1','1','1','0','1','1','1','1','1','0','1','1','1','1','0','1'],
    ['1','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','1'],
    ['1','1','0','1','1','1','0','1','1','G','1','1','0','1','1','1','1','0','1'],
    ['0','0','0','0','0','1','0','1','G','G','G','1','0','1','0','0','0','0','0'],
    ['1','1','0','1','0','1','0','1','1','1','1','1','0','1','0','1','0','1','1'],
    ['1','0','0','1','0','0','0','0','0','0','0','0','0','0','0','1','0','0','1'],
    ['1','0','1','1','1','1','0','1','1','1','1','1','0','1','1','1','1','0','1'],
    ['1','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','1'],
    ['1','1','1','0','1','1','1','0','1','1','1','0','1','1','1','0','1','1','1'],
    ['1','0','0','0','1','0','0','0','0','0','0','0','0','0','1','0','0','0','1'],
    ['1','0','1','0','1','0','1','0','1','1','1','0','1','0','1','0','1','0','1'],
    ['1','0','1','0','0','0','1','0','0','0','0','0','1','0','0','0','1','0','1'],
    ['1','0','1','1','1','0','1','1','1','0','1','1','1','0','1','1','1','0','1'],
    ['1','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','1'],
    ['1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1']
]

class PacmanGameAI:
    def __init__(self, render_mode=False):
        pygame.init()
        pygame.font.init()
        self.render_mode = render_mode
        if self.render_mode:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Pacman Game AI")
        else:
            # headless mode: render to off-screen Surface
            self.screen = pygame.Surface((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.game_over_font = pygame.font.SysFont("consolas", 50)
        self.running = True
        self.reset()

    def reset(self):
        self.lives = 3
        self.score = 0
        self.pellets = PelletManager(maze)
        self.reset_positions()

    def reset_positions(self):
        self.scatter_mode = False
        self.scatter_timer = 600
        self.ghost_gate_open = False
        self.ghost_gate_timer = 0
        self.pacman = PacMan()
        self.ghosts = [
            Ghost(9,  8, (255,   0,   0), "blinky", scatter_target=(17, 1),  exit_delay=0),
            Ghost(9,  9, (255, 184, 255), "pinky",  scatter_target=(1,  1),  exit_delay=60),
            Ghost(8,  9, (0,   255, 255), "inky",   scatter_target=(17, 19), exit_delay=120),
            Ghost(10, 9, (255, 184,  82), "clyde",  scatter_target=(1,  19), exit_delay=180),
        ]

    def update(self):
        # exactly your existing update logic
        self.pacman.move(maze)
        result = self.pellets.eat(self.pacman)
        if result == "pellet":
            self.score += 10
        elif result == "energizer":
            self.score += 50
            for ghost in self.ghosts:
                ghost.frightened = True
                ghost.frightened_timer = FPS * 10

        for ghost in self.ghosts:
            self.ghost_gate_open, self.ghost_gate_timer = ghost.move(
                self.pacman, self.scatter_mode, self.ghosts,
                maze, self.ghost_gate_open, self.ghost_gate_timer
            )

        for ghost in self.ghosts:
            if self.pacman.x == ghost.x and self.pacman.y == ghost.y and ghost.revive_timer == 0:
                if ghost.frightened:
                    ghost.respawn()
                    self.score += 200
                else:
                    self.lives -= 1
                    if self.lives == 0:
                        self.reset()
                        return
                    else:
                        self.reset_positions()

        if not self.pellets.pellets and not self.pellets.energizers:
            self.reset()

    def draw(self):
        self.screen.fill((0, 0, 0))
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if maze[y][x] == '1':
                    pygame.draw.rect(self.screen, BLUE, (x*CELL_SIZE,y*CELL_SIZE,CELL_SIZE,CELL_SIZE))
                elif maze[y][x] == 'G':
                    color = (0,255,0) if self.ghost_gate_open else (0,128,255)
                    pygame.draw.rect(self.screen, color, (x*CELL_SIZE,y*CELL_SIZE,CELL_SIZE,CELL_SIZE))
        self.pellets.draw(self.screen)
        self.pacman.draw(self.screen)
        for ghost in self.ghosts:
            ghost.draw(self.screen)
        # lives
        for i in range(self.lives):
            x_pos = WIDTH - (i + 1) * 30
            y_pos = HEIGHT - 15
            pygame.draw.circle(self.screen, YELLOW, (x_pos,y_pos), 10)
        # score
        score_text = self.game_over_font.render(f"Score: {self.score}", True, (255,255,255))
        self.screen.blit(score_text, (10,10))

        if self.render_mode:
            pygame.display.flip()
