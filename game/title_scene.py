import sys

import pygame

from game.dataset_scene import DatasetScene
from ui.button import Button


class TitleScene:
    def __init__(self):
        self.scene_manager = None

        def play_button_click():
            self.scene_manager.go_to(DatasetScene())

        self.play_button = Button("Datasets", (400, 720, 200, 75), text_font=pygame.font.Font(None, 50),
                                  on_click_fn=play_button_click)
        self.quit_button = Button("Quit", (400, 720, 150, 75), text_font=pygame.font.Font(None, 50),
                                  on_click_fn=lambda: sys.exit(0))

    def handle_events(self, events: list[pygame.event.Event]):
        self.play_button.handle_events(events)
        self.quit_button.handle_events(events)

    def update(self):
        pass

    def render(self, screen: pygame.Surface):
        screen.fill((0, 0, 0))

        f = pygame.font.Font(None, 60)
        title = f.render("Asteroids", True, (255, 255, 0))
        screen.blit(title, (screen.get_width() / 2 - title.get_width() / 2, 50))

        self.play_button.rect.centerx = screen.get_width() / 2
        self.play_button.rect.centery = screen.get_height() / 2
        self.quit_button.rect.centerx = screen.get_width() / 2
        self.quit_button.rect.centery = screen.get_height() / 2 + 100
        screen.blit(self.play_button.render(), self.play_button.rect)
        screen.blit(self.quit_button.render(), self.quit_button.rect)
