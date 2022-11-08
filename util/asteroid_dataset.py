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

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        # Get image and convert to greyscale
        image = Image.open(f"{self.image_dir}/img{index}.png").convert("L")
        # Transform to numpy array
        image = np.asarray(image).astype(np.float32)
        # Add the "channel" dimension
        image = np.expand_dims(image, axis=0)
        # Normalize it
        image /= 255
        if self.transform:
            sample = self.transform(image)

        # Get bounding boxes
        bb: np.ndarray = self.labels[index]
        # Currently, just get the one and only bounding box TODO: don't assume this
        bb: np.ndarray = bb[0].astype(np.float32)
        # Offset bb by -1, -1... not entirely sure why. Pixel indexing difference between pygame and matplotlib?
        bb[:2] -= 1

        return image, bb
