import random

import pygame
from matplotlib import pyplot as plt, patches
from torch.utils.data import DataLoader

from util.asteroid_dataset import AsteroidDataset


class TrainingScene:
    def __init__(self):
        self.scene_manager = None

        dataset = AsteroidDataset("./datasets/img", "./datasets/lbl.npy")
        self.dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        # Show a random sample from the dataset
        fig, ax = plt.subplots()
        img, bb = dataset[random.randint(0, len(dataset))]
        print(img, "\n", img.shape)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        rect = patches.Rectangle(bb[0, :2], bb[0, 2], bb[0, 3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()

        # Create a model
        # class Net(nn.Module):
        #     def __init__(self):
        #         super(Net, self).__init__()
        #         self.linear1 = nn.Linear(800, 800)
        #         self.linear2 = nn.Linear(50, 10)
        #
        #     def forward(self, x):
        #         x = F.relu(self.fc1(x))
        #         return self.fc2(x)

    def handle_events(self, events: list[pygame.event.Event]):
        pass

    def update(self):
        pass

    def render(self, screen: pygame.Surface):
        screen.fill((0, 0, 0))
