import random

import pygame

from game.asteroid import Asteroid


class GameScene:
    def __init__(self):
        self.shapes: list[Asteroid] = []

        self.generate_asteroids()

    def handle_events(self, events: list[pygame.event.Event]):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.generate_asteroids()

    def update(self):
        pass

    def render(self, screen: pygame.Surface):
        screen.fill((0, 0, 0))

        for asteroid in self.shapes:
            img: pygame.Surface = asteroid.draw()
            screen.blit(img, asteroid.center - pygame.Vector2(img.get_size()))

    def generate_asteroids(self, num_asteroids: tuple[int, int] = None):
        if num_asteroids is None:
            num_asteroids = (8, 14)
        self.shapes = [Asteroid() for _ in range(random.randint(*num_asteroids))]
