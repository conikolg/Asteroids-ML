import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt, patches
from torch.utils.data import DataLoader

from util.asteroid_dataset import AsteroidDataset


# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input is 1x800x800
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=11, stride=5)
        self.conv1_bn = nn.BatchNorm2d(num_features=self.conv1.out_channels)
        self.down_sample1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=64, kernel_size=5, stride=3)
        self.conv2_bn = nn.BatchNorm2d(num_features=self.conv2.out_channels)
        self.down_sample2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.linear1 = nn.Linear(self.conv2.out_channels * 23 * 23, 4)

    def forward(self, x: torch.Tensor):
        x = self.down_sample1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.down_sample2(F.relu(self.conv2_bn(self.conv2(x))))
        # print(x.shape)
        x = x.view(-1, self.conv2.out_channels * 23 * 23)
        x = self.linear1(x)
        return x


n_epochs = 6
batch_size_train = 32
batch_size_test = 8
learning_rate = 0.00001
momentum = 0.5
log_interval = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

network = Net()
network.to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate)


def train(epochs):
    train_losses = []
    train_dataset = AsteroidDataset("./datasets/train/images", "./datasets/train/labels.npy")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

    network.train()
    for epoch_idx in range(epochs):
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = network(inputs)
            loss = F.mse_loss(output, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch_idx + 1} "
                      f"[{batch_idx * len(inputs)}/{len(train_dataloader.dataset)} "
                      f"({100. * batch_idx / len(train_dataloader):.0f}%)]\t"
                      f"Loss: {loss.item():.6f}")
                train_losses.append(loss.item())

    fig, ax = plt.figure(), plt.axes()
    ax.plot(np.arange(len(train_losses)), train_losses)


def test():
    test_losses = []

    # Show random samples from the dataset
    network.eval()
    test_dataset = AsteroidDataset("./datasets/test/images", "./datasets/test/labels.npy")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, num_workers=0)
    img_test, bb1 = next(iter(test_dataloader))
    img_test, bb1 = img_test.to(device), bb1.to(device)
    with torch.no_grad():
        bb2 = network(img_test).cpu()

    img_test, bb1 = img_test.cpu(), bb1.cpu()
    rows, cols = 2, batch_size_test // 2
    fig, ax_arr = plt.subplots(rows, cols)
    for idx in range(batch_size_test):
        ax_arr[idx // cols, idx % cols].imshow(img_test[idx, 0], cmap='gray', vmin=0, vmax=1)
        rect1 = patches.Rectangle(bb1[idx, :2], bb1[idx, 2], bb1[idx, 3], linewidth=1, edgecolor='g', facecolor='none')
        rect2 = patches.Rectangle(bb2[idx, :2], bb2[idx, 2], bb2[idx, 3], linewidth=1, edgecolor='r', facecolor='none')
        ax_arr[idx // cols, idx % cols].add_patch(rect1)
        ax_arr[idx // cols, idx % cols].add_patch(rect2)
    plt.show()


def main():
    train(n_epochs)
    test()


if __name__ == '__main__':
    main()
