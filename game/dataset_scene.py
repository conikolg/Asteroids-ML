import pygame

from game.game_scene import GameScene
from ui.button import Button
from ui.text_box import TextBox


class DatasetScene:
    def __init__(self):
        self.game: GameScene = GameScene()

        self.normal_font: pygame.Font = pygame.font.Font(None, 48)

        # Num asteroid inputs
        self.num_asteroid_textbox_min = TextBox(
            rect=pygame.Rect(1000, 200, 100, 50),
            text_font=self.normal_font,
            validate_fn=lambda s: s.isnumeric(),
            value_fn=lambda s: int(s)
        )
        self.num_asteroid_textbox_max = TextBox(
            rect=pygame.Rect(1150, 200, 100, 50),
            text_font=self.normal_font,
            validate_fn=lambda s: s.isnumeric(),
            value_fn=lambda s: int(s)
        )

        # Regenerate button
        self.generate_btn = Button(
            text="Generate",
            rect=pygame.Rect(1000, 800, 200, 50),
            text_font=self.normal_font,
            on_click_fn=lambda: self.game.generate_asteroids()
        )

    def handle_events(self, events: list[pygame.event.Event]):
        self.game.handle_events(events)
        self.num_asteroid_textbox_min.handle_events(events, consume_events=False)
        self.num_asteroid_textbox_max.handle_events(events, consume_events=False)
        self.generate_btn.handle_events(events, consume_events=False)

    def update(self):
        self.game.update()

    def render(self, screen: pygame.Surface):
        screen.fill(pygame.Color("black"))
        game_img: pygame.Surface = pygame.Surface((800, 800))
        self.game.render(game_img)

        # Border around the actual game in green
        pygame.draw.rect(screen, pygame.Color("green"), (45, 45, 810, 810))
        screen.blit(game_img, (50, 50))

        # Draw ui components
        screen.blit(self.normal_font.render("Number of asteroids", True, pygame.Color("white")),
                    self.num_asteroid_textbox_min.rect.move(0, -35))
        screen.blit(self.normal_font.render("to", True, pygame.Color("white")),
                    self.num_asteroid_textbox_min.rect.move(self.num_asteroid_textbox_min.rect.w + 10, 10))
        screen.blit(self.num_asteroid_textbox_min.render(), self.num_asteroid_textbox_min.rect)
        screen.blit(self.num_asteroid_textbox_max.render(), self.num_asteroid_textbox_max.rect)

        screen.blit(self.generate_btn.render(), self.generate_btn.rect)
