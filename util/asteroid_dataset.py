import numpy as np
import torch.utils.data
from PIL import Image


class AsteroidDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, labels_file: str, transform=None):
        self.image_dir = image_dir
        self.labels: np.ndarray = np.load(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = np.asarray(Image.open(f"{self.image_dir}/img{index}.png").convert("L"))
        if self.transform:
            sample = self.transform(image)

        # Offset by -1, -1... not entirely sure why. Pixel indexing difference between pygame and matplotlib?
        bb: np.ndarray = self.labels[index]
        bb[:][:2] -= 1

        return image, bb
