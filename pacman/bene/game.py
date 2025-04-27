# pacman_game_ai.py

import pygame
import sys
from ghost import Ghost
from pellet_manager import PelletManager
from pacman import PacMan
from settings import CELL_SIZE, GRID_WIDTH, GRID_HEIGHT, UP, DOWN, LEFT, RIGHT, FPS

# Constants
WIDTH, HEIGHT = GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

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
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pacman Game AI")
        self.clock = pygame.time.Clock()
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

        self.pacman = self.PacMan()
        self.ghosts = [
            Ghost(9, 8, (255, 0, 0), "blinky", scatter_target=(17,1), exit_delay=0),
            Ghost(9, 9, (255, 184, 255), "pinky", scatter_target=(1,1), exit_delay=60),
            Ghost(8, 9, (0, 255, 255), "inky", scatter_target=(17,19), exit_delay=120),
            Ghost(10, 9, (255, 184, 82), "clyde", scatter_target=(1,19), exit_delay=180)
        ]

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
            pygame.draw.circle(screen, YELLOW, (self.x * CELL_SIZE + CELL_SIZE//2, self.y * CELL_SIZE + CELL_SIZE//2), CELL_SIZE//2 - 2)

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.pacman.direction = UP
                elif event.key == pygame.K_DOWN:
                    self.pacman.direction = DOWN
                elif event.key == pygame.K_LEFT:
                    self.pacman.direction = LEFT
                elif event.key == pygame.K_RIGHT:
                    self.pacman.direction = RIGHT

    def update(self):
        self.pacman.move(maze)
        result = self.pellets.eat(self.pacman)
        if result == "pellet":
            self.score += 10
        elif result == "energizer":
            self.score += 50
            for ghost in self.ghosts:
                ghost.frightened = True
                ghost.frightened_timer = FPS  * 10

        for ghost in self.ghosts:
            self.ghost_gate_open, self.ghost_gate_timer = ghost.move(self.pacman, self.scatter_mode, self.ghosts, maze, self.ghost_gate_open, self.ghost_gate_timer)

        for ghost in self.ghosts:
            if self.pacman.x == ghost.x and self.pacman.y == ghost.y and ghost.revive_timer == 0:
                if ghost.frightened:
                    ghost.respawn()
                    self.score += 200
                else:
                    self.lives -=1
                    if self.lives == 0:
                        self.reset()
                    else:
                        self.reset_positions()
        
        if not self.pellets.pellets and not self.pellets.energizers:
            self.reset()

    def draw(self):
        self.screen.fill((0, 0, 0))

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if maze[y][x] == '1':
                    pygame.draw.rect(self.screen, BLUE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif maze[y][x] == 'G':
                    color = (0, 255, 0) if self.ghost_gate_open else (0, 128, 255)
                    pygame.draw.rect(self.screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        self.pellets.draw(self.screen)
        self.pacman.draw(self.screen)
        for ghost in self.ghosts:
            ghost.draw(self.screen)

        # Draw lives
        for i in range(self.lives):  # Minus 1 because the current life is active
            x_pos = WIDTH - (i + 1) * 30  # Spacing the lives nicely
            y_pos = HEIGHT - 15
            pygame.draw.circle(self.screen, (255, 255, 0), (x_pos, y_pos), 10)

        # Draw score
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))  # Top-left corner


        pygame.display.flip()


def main():
    game = PacmanGameAI()
    game.run()

if __name__ == "__main__":
    main()