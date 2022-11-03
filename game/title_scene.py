import sys

import pygame

from game.dataset_scene import DatasetScene
from game.training_scene import TrainingScene
from ui.button import Button


class TitleScene:
    def __init__(self):
        self.scene_manager = None

        f = pygame.font.Font(None, 60)
        self.title_render = f.render("Asteroids", True, (255, 255, 0))

        self.dataset_btn = Button("Datasets", (700, 200, 200, 75), text_font=pygame.font.Font(None, 50),
                                  on_click_fn=lambda: self.scene_manager.go_to(DatasetScene()))
        self.training_btn = Button("Training", (700, 300, 200, 75), text_font=pygame.font.Font(None, 50),
                                   on_click_fn=lambda: self.scene_manager.go_to(TrainingScene()))
        self.quit_btn = Button("Quit", (700, 400, 200, 75), text_font=pygame.font.Font(None, 50),
                               on_click_fn=lambda: sys.exit(0))

    def handle_events(self, events: list[pygame.event.Event]):
        self.dataset_btn.handle_events(events)
        self.training_btn.handle_events(events)
        self.quit_btn.handle_events(events)

    def update(self):
        pass

    def render(self, screen: pygame.Surface):
        screen.fill((0, 0, 0))

        screen.blit(self.title_render, (screen.get_width() / 2 - self.title_render.get_width() / 2, 50))

        screen.blit(self.dataset_btn.render(), self.dataset_btn.rect)
        screen.blit(self.training_btn.render(), self.training_btn.rect)
        screen.blit(self.quit_btn.render(), self.quit_btn.rect)
