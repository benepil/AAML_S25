# pellet_manager.py

import pygame

CELL_SIZE = 24

class PelletManager:
    def __init__(self, maze):
        self.pellets = []
        self.energizers = []
        for y in range(len(maze)):
            for x in range(len(maze[0])):
                if maze[y][x] == '0':
                    if (x, y) in [(1, 1), (17, 1), (1, 19), (17, 19)]:
                        self.energizers.append((x, y))
                    else:
                        self.pellets.append((x, y))

    def draw(self, screen):
        for (x, y) in self.pellets:
            pygame.draw.circle(screen, (255, 255, 255), (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), 3)
        for (x, y) in self.energizers:
            pygame.draw.circle(screen, (0, 255, 255), (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), 6)

    def eat(self, pacman):
        if (pacman.x, pacman.y) in self.pellets:
            self.pellets.remove((pacman.x, pacman.y))
            return "pellet"
        if (pacman.x, pacman.y) in self.energizers:
            self.energizers.remove((pacman.x, pacman.y))
            return "energizer"
        return None
