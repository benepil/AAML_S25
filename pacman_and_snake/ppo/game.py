# game.py  (pacman_game_ai.py renamed)
import pygame
import sys
from ghost import Ghost
from pellet_manager import PelletManager
from pacman import PacMan
from settings import CELL_SIZE, GRID_WIDTH, GRID_HEIGHT, UP, DOWN, LEFT, RIGHT, FPS

# Constants
WIDTH, HEIGHT = GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE
YELLOW = (255, 255, 0)
BLUE   = (0, 0, 255)

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
    def __init__(self, render_mode=True):
        pygame.init()
        pygame.font.init()

        self.render_mode = render_mode
        if self.render_mode:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Pacman Game AI")
        else:
            self.screen = pygame.Surface((WIDTH, HEIGHT))

        self.clock = pygame.time.Clock()
        self.running = True
        self.reset()

    def reset(self):
        self.lives = 3
        self.score = 0
        self.pellets = PelletManager(maze)
        self.reset_positions()
        return self.get_state()

    def reset_positions(self):
        self.scatter_mode = False
        self.scatter_timer = FPS * 10
        self.ghost_gate_open = False
        self.ghost_gate_timer = 0

        self.pacman = PacMan()
        self.ghosts = [
            Ghost(9, 8,   (255, 0,   0),   "blinky", scatter_target=(17,1),  exit_delay=0),
            Ghost(9, 9,   (255,184,255),"pinky",  scatter_target=(1,1),   exit_delay=60),
            Ghost(8, 9,   (0,  255,255),"inky",   scatter_target=(17,19), exit_delay=120),
            Ghost(10,9,   (255,184,82), "clyde",  scatter_target=(1,19),  exit_delay=180)
        ]

    def get_state(self):
        return {
            'snake_head': self.pacman.x,
            'food': self.pellets,
            'score': self.score,
            'snake': None,
            'direction': self.pacman.direction
        }

    def update(self):
        self.pacman.move(maze)
        result = self.pellets.eat(self.pacman)
        if result == "pellet":
            self.score += 10
        elif result == "energizer":
            self.score += 50
            for g in self.ghosts:
                g.frightened = True
                g.frightened_timer = FPS * 10

        for g in self.ghosts:
            self.ghost_gate_open, self.ghost_gate_timer = g.move(
                self.pacman, self.scatter_mode, self.ghosts,
                maze, self.ghost_gate_open, self.ghost_gate_timer
            )

        for g in self.ghosts:
            if self.pacman.x == g.x and self.pacman.y == g.y and g.revive_timer == 0:
                if g.frightened:
                    g.respawn()
                    self.score += 200
                else:
                    self.lives -= 1
                    if self.lives == 0:
                        self.running = False
                    else:
                        self.reset_positions()

        if not self.pellets.pellets and not self.pellets.energizers:
            self.running = False

    def draw(self):
        if not self.render_mode:
            return
        self.screen.fill((0,0,0))
        for y,row in enumerate(maze):
            for x,ch in enumerate(row):
                if ch == '1':
                    pygame.draw.rect(self.screen, BLUE,
                                     (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif ch == 'G':
                    col = (0,255,0) if self.ghost_gate_open else (0,128,255)
                    pygame.draw.rect(self.screen, col,
                                     (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        self.pellets.draw(self.screen)
        self.pacman.draw(self.screen)
        for g in self.ghosts:
            g.draw(self.screen)

        for i in range(self.lives):
            pygame.draw.circle(self.screen, YELLOW,
                               (WIDTH - 30*(i+1), HEIGHT - 15), 10)

        font = pygame.font.SysFont(None,36)
        txt = font.render(f"Score: {self.score}", True, (255,255,255))
        self.screen.blit(txt, (10,10))
        pygame.display.flip()

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:    self.pacman.direction = UP
                    elif event.key == pygame.K_DOWN:  self.pacman.direction = DOWN
                    elif event.key == pygame.K_LEFT:  self.pacman.direction = LEFT
                    elif event.key == pygame.K_RIGHT: self.pacman.direction = RIGHT

            self.update()
            self.draw()

        pygame.quit()
        sys.exit()
