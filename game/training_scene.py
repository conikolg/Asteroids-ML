import pygame


class TrainingScene:
    def __init__(self):
        self.scene_manager = None

    def handle_events(self, events: list[pygame.event.Event]):
        pass

    def update(self):
        pass

    def render(self, screen: pygame.Surface):
        screen.fill((0, 0, 0))
