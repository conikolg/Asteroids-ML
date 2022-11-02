from typing import Union, Callable

import pygame


class TextBox:
    def __init__(self,
                 rect: Union[pygame.rect.Rect, tuple],
                 placeholder_text: str = None,
                 inactive_color: tuple = None,
                 active_color: tuple = None,
                 outline_color: tuple = None,
                 outline_width: int = None,
                 text_font: pygame.font.Font = None,
                 text_color: tuple = None,
                 validate_fn: Callable = None,
                 value_fn: Callable = None):

        self.rect = rect
        self.placeholder_text = "" if placeholder_text is None else placeholder_text
        self.inactive_color = (255, 255, 255) if inactive_color is None else inactive_color
        self.active_color = (150, 150, 150) if active_color is None else active_color
        self.outline_color = (0, 0, 0) if outline_color is None else outline_color
        self.outline_width = 2 if outline_width is None else outline_width
        self.text_font = pygame.font.Font(None, 24) if text_font is None else text_font
        self.text_color = pygame.Color("black") if text_color is None else pygame.Color(text_color)
        self.validate_fn = (lambda _: True) if validate_fn is None else validate_fn
        self.value_fn = (lambda _: self._text) if value_fn is None else value_fn

        self._active: bool = False
        self._text: str = ""
        if isinstance(self.rect, tuple) and len(self.rect) == 4:
            self.rect = pygame.rect.Rect(self.rect)

    def handle_events(self, events: list[pygame.event.Event], consume_events=True):
        """
        Allow the text box to be responsive to pygame events.
        :param events: a list of pygame events
        :param consume_events: if True, events that this button "uses up" will be removed from the event list,
        preventing any future components from using those events. Enabled by default.
        :return: None
        """

        n = len(events)
        for i, event in enumerate(reversed(events)):
            # Determine if the button was clicked. Consume event if it did and invoke on_click callback.
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._active = self.rect.collidepoint(*pygame.mouse.get_pos())
                if self._active:
                    if consume_events:
                        events.pop(n - i - 1)
            if self._active and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self._text = self._text[:-1]
                else:
                    tmp_text: str = self._text + event.unicode
                    if self.validate_fn(tmp_text):
                        self._text = tmp_text

    def render(self) -> pygame.Surface:
        # Get the textbox overall shape + make outline
        img = pygame.Surface((self.rect.w, self.rect.h)).convert_alpha()
        img.fill(self.outline_color)

        # Fill correct background color
        img2 = pygame.Surface(
            (self.rect.w - self.outline_width * 2, self.rect.h - self.outline_width * 2)).convert_alpha()
        if self._active:
            img2.fill(self.active_color)
        else:
            img2.fill(self.inactive_color)

        # Draw text content
        if self._text:
            rendered_text: pygame.Surface = self.text_font.render(self._text, True, self.text_color)
        else:
            rendered_text: pygame.Surface = self.text_font.render(
                self.placeholder_text, True, self.text_color.lerp(
                    self.active_color if self.active_color else self.inactive_color, 0.75)
            )
        img2.blit(rendered_text, (
            img2.get_width() / 2 - rendered_text.get_width() / 2,
            img2.get_height() / 2 - rendered_text.get_height() / 2))

        # Put content on top of outline image
        img.blit(img2, (self.outline_width, self.outline_width))
        return img

    @property
    def value(self):
        return self.value_fn(self._text if self._text else self.placeholder_text)
