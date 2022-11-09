import multiprocessing
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import pygame

from game.game_scene import GameScene
from ui.button import Button
from ui.text_box import TextBox


class DatasetScene:
    def __init__(self):
        self.scene_manager = None
        self.game: GameScene = GameScene()

        self.normal_font: pygame.Font = pygame.font.Font(None, 48)

        # Num images input
        self.num_images = TextBox(
            rect=pygame.Rect(1000, 100, 150, 50),
            text_font=self.normal_font,
            placeholder_text="1000",
            validate_fn=lambda s: s.isnumeric(),
            value_fn=lambda s: int(s)
        )

        # Num asteroid inputs
        self.num_asteroid_textbox_min = TextBox(
            rect=pygame.Rect(1000, 200, 100, 50),
            text_font=self.normal_font,
            placeholder_text="1",
            validate_fn=lambda s: s.isnumeric(),
            value_fn=lambda s: int(s)
        )
        self.num_asteroid_textbox_max = TextBox(
            rect=pygame.Rect(1150, 200, 100, 50),
            text_font=self.normal_font,
            placeholder_text="1",
            validate_fn=lambda s: s.isnumeric(),
            value_fn=lambda s: int(s)
        )

        # Regenerate buttons
        self.test_generate_btn = Button(
            text="Test",
            rect=pygame.Rect(1000, 800, 100, 50),
            text_font=self.normal_font,
            on_click_fn=lambda: self.game.generate_asteroids(
                (self.num_asteroid_textbox_min.value, self.num_asteroid_textbox_max.value))
        )
        self.generate_btn = Button(
            text="Generate",
            rect=pygame.Rect(1120, 800, 200, 50),
            text_font=self.normal_font,
            on_click_fn=self.generate_datasets
        )

        self.game.generate_asteroids((self.num_asteroid_textbox_min.value, self.num_asteroid_textbox_max.value))

        self.mp_pool = multiprocessing.Pool()

    def handle_events(self, events: list[pygame.event.Event]):
        self.game.handle_events(events)
        self.num_images.handle_events(events, consume_events=False)
        self.num_asteroid_textbox_min.handle_events(events, consume_events=False)
        self.num_asteroid_textbox_max.handle_events(events, consume_events=False)
        self.test_generate_btn.handle_events(events, consume_events=False)
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

        # ======================
        #   Draw UI components
        # ======================

        # Number of images
        screen.blit(self.normal_font.render("Number of images", True, pygame.Color("white")),
                    self.num_images.rect.move(0, -35))
        screen.blit(self.num_images.render(), self.num_images.rect)

        # Number of asteroids
        screen.blit(self.normal_font.render("Number of asteroids", True, pygame.Color("white")),
                    self.num_asteroid_textbox_min.rect.move(0, -35))
        screen.blit(self.normal_font.render("to", True, pygame.Color("white")),
                    self.num_asteroid_textbox_min.rect.move(self.num_asteroid_textbox_min.rect.w + 10, 10))
        screen.blit(self.num_asteroid_textbox_min.render(), self.num_asteroid_textbox_min.rect)
        screen.blit(self.num_asteroid_textbox_max.render(), self.num_asteroid_textbox_max.rect)

        # Generate images
        screen.blit(self.test_generate_btn.render(), self.test_generate_btn.rect)
        screen.blit(self.generate_btn.render(), self.generate_btn.rect)

    def generate_datasets(self):
        now = dt.now()
        tasks: list = []
        print("Generating images... ", end="")
        tasks += self.generate_dataset(self.num_images.value, Path("./datasets/train/"))
        # print("Generating testing images...")
        tasks += self.generate_dataset(int(self.num_images.value * 0.3), Path("./datasets/test/"))
        # print("Generating validation images...")
        tasks += self.generate_dataset(int(self.num_images.value * 0.2), Path("./datasets/validation/"))
        for task in tasks:
            task.get()
        tasks = [
            self.mp_pool.apply_async(concat_labels, (Path(f"./datasets/{dataset}/"),))
            for dataset in "train test validation".split()
        ]
        for task in tasks:
            task.get()
        elapsed = (dt.now() - now).total_seconds()
        print(f"Done!\n"
              f"Image generation time: {elapsed} sec  (~{round(int(self.num_images.value * 1.5) / elapsed)} img/sec)")

    def generate_dataset(self, num_images: int, data_dir: Path) -> list:
        # Make the folders as necessary
        (data_dir / "images").mkdir(parents=True, exist_ok=True)
        # Segment into chunks of 100 images and queue to processing pool
        batch_size: int = 100
        tasks: list[multiprocessing.pool.AsyncResult] = [
            self.mp_pool.apply_async(
                func=generate_images,
                args=(idx,
                      batch_size if idx + batch_size <= num_images else num_images - idx,
                      self.num_asteroid_textbox_min.value,
                      self.num_asteroid_textbox_max.value,
                      data_dir)
            )
            for idx in range(0, num_images, batch_size)
        ]

        return tasks


def generate_images(start_idx: int, n: int, min_asteroids: int, max_asteroids: int, data_dir: Path):
    # New processes need to reinit pygame
    pygame.init()
    screen = pygame.display.set_mode((1, 1))
    # Holding place for bounding box labels
    labels = np.zeros(shape=(n, min_asteroids, 4))

    # New game object to ease offloading into new thread/process
    game: GameScene = GameScene()
    for i in range(start_idx, start_idx + n):
        # Generate/export image
        game.generate_asteroids((min_asteroids, max_asteroids))
        img: pygame.Surface = game.render(pygame.Surface((800, 800)))
        pygame.image.save(img, data_dir / f"images/img{i}.png")

        # Record label
        labels[i - start_idx]: np.ndarray = game.bounding_boxes

    # Save the labels
    np.save(str(data_dir / f"labels{start_idx}.npy"), labels)


def concat_labels(data_dir: Path):
    arrays: list = []
    files = sorted(data_dir.iterdir())
    for file in files:
        if file.name.endswith(".npy"):
            arrays.append(np.load(str(file)))
            Path.unlink(file)

    concat: np.ndarray = np.concatenate(arrays, axis=0)
    np.save(str(data_dir / "labels.npy"), concat)
