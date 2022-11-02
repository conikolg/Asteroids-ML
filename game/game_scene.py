import random

import numpy as np
import pygame

from game.asteroid import Asteroid


class GameScene:
    def __init__(self):
        self.asteroids: list[Asteroid] = []

        self.generate_asteroids()

    def handle_events(self, events: list[pygame.event.Event]):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.generate_asteroids()

    def update(self):
        pass

    def render(self, screen: pygame.Surface) -> pygame.Surface:
        screen.fill((0, 0, 0))

        for asteroid in self.asteroids:
            img: pygame.Surface = asteroid.draw()
            screen.blit(img, asteroid.center - pygame.Vector2(img.get_size()))

        return screen

    def generate_asteroids(self, num_asteroids: tuple[int, int] = None):
        if num_asteroids is None:
            num_asteroids = (8, 14)
        self.asteroids = [Asteroid() for _ in range(random.randint(*num_asteroids))]

    @property
    def bounding_boxes(self) -> np.ndarray:
        # TODO: clip if bleeding outside the edges of the screen
        return np.array([a.bb for a in self.asteroids])
