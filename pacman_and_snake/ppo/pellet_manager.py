# pellet_manager.py
import pygame

CELL_SIZE = 24

class PelletManager:
    def __init__(self, maze):
        self.pellets = []
        self.energizers = []
        for y, row in enumerate(maze):
            for x, ch in enumerate(row):
                if ch == '0':
                    if (x, y) in [(1,1), (17,1), (1,19), (17,19)]:
                        self.energizers.append((x,y))
                    else:
                        self.pellets.append((x,y))

    def draw(self, screen):
        for x, y in self.pellets:
            pygame.draw.circle(
                screen, (255,255,255),
                (x*CELL_SIZE+CELL_SIZE//2, y*CELL_SIZE+CELL_SIZE//2), 3
            )
        for x, y in self.energizers:
            pygame.draw.circle(
                screen, (0,255,255),
                (x*CELL_SIZE+CELL_SIZE//2, y*CELL_SIZE+CELL_SIZE//2), 6
            )

    def eat(self, pacman):
        pos = (pacman.x, pacman.y)
        if pos in self.pellets:
            self.pellets.remove(pos)
            return "pellet"
        if pos in self.energizers:
            self.energizers.remove(pos)
            return "energizer"
        return None
