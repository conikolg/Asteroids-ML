import math
import random

import numpy as np
import pygame


class Asteroid:
    def __init__(self, core_radius: int = None, core_center: pygame.Vector2 = None, points: int = None,
                 corner_radius: int = None):
        self.points: list[pygame.Vector2] = []
        self.core_radius: int = random.randint(20, 70) if core_radius is None else core_radius
        self.center: pygame.Vector2 = pygame.Vector2(
            random.randint(100, 700), random.randint(100, 700)
        ) if core_center is None else core_center

        points_len: int = random.randint(8, 14) if points is None else points
        for i in range(points_len):
            corner_radius: int = random.randint(-self.core_radius // 4,
                                                self.core_radius // 4) if corner_radius is None else corner_radius
            angle: float = random.random() * math.pi * 2
            self.points.append(pygame.Vector2(
                int(self.core_radius * math.cos(2 * math.pi / points_len * i) + corner_radius * math.cos(angle)),
                int(self.core_radius * math.sin(2 * math.pi / points_len * i) + corner_radius * math.sin(angle))
            ))

        minx = min(p.x for p in self.points)
        miny = min(p.y for p in self.points)
        maxx = max(p.x for p in self.points)
        maxy = max(p.y for p in self.points)

        # Recenter around 0,0
        for p in self.points:
            p.x -= (maxx + minx) / 2
            p.y -= (maxy + miny) / 2

        minx = min(p.x for p in self.points)
        miny = min(p.y for p in self.points)
        maxx = max(p.x for p in self.points)
        maxy = max(p.y for p in self.points)

        self.img: pygame.Surface = pygame.Surface((maxx - minx + 2, maxy - miny + 2)).convert_alpha()
        self.img.fill((0, 0, 0, 0))
        for i in range(len(self.points)):
            start: pygame.Vector2 = self.points[i - 1] + pygame.Vector2(self.img.get_size()) / 2
            end: pygame.Vector2 = self.points[i] + pygame.Vector2(self.img.get_size()) / 2
            pygame.draw.line(self.img, (255, 255, 255), start, end)

    def draw(self) -> pygame.Surface:
        return self.img

    def __repr__(self):
        return str(self.points)

    def __str__(self):
        return self.__repr__()

    @property
    def bb(self) -> np.ndarray:
        return self.bounding_box

    @property
    def bounding_box(self) -> np.ndarray:
        """ Returns an array formatted as [x, y, w, h]. """

        minx = min(p.x for p in self.points)
        miny = min(p.y for p in self.points)
        return np.array([
            minx + self.center.x,
            miny + self.center.y,
            self.img.get_width(),
            self.img.get_height()
        ])
