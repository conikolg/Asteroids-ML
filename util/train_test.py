import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt, patches
from torch.utils.data import DataLoader

from util.asteroid_dataset import AsteroidDataset


# Show a random sample from the dataset
# fig, ax = plt.subplots()
# img, bb = dataset[random.randint(0, len(dataset))]
# ax.imshow(img, cmap='gray', vmin=0, vmax=1)
# rect = patches.Rectangle(bb[:2], bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# plt.show()

# Create a model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input is 800x800
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=4)  # Outputs 4x200x200
        self.down_sample1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Outputs 4x100x100
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=2, stride=2)  # Outputs 16x50x50
        self.down_sample2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Outputs 16x25x25
        self.linear1 = nn.Linear(16 * 25 * 25, 4)

    def forward(self, x):
        x = F.relu(self.down_sample1(self.conv1(x)))
        x = F.relu(self.down_sample2(self.conv2(x)))
        x = x.view(-1, 16 * 25 * 25)
        x = self.linear1(x)
        return x


n_epochs = 3
batch_size_train = 16
batch_size_test = 8
learning_rate = 0.01
momentum = 0.5
log_interval = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = AsteroidDataset("./datasets/img", "./datasets/lbl.npy")
dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

network = Net()
network.to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
train_losses = []
test_losses = []
test_counter = [i * len(dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = network(inputs)
        loss = F.mse_loss(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} "
                  f"[{batch_idx * len(inputs)}/{len(dataloader.dataset)} "
                  f"({100. * batch_idx / len(dataloader):.0f}%)]\t"
                  f"Loss: {loss.item():.6f}")
            train_losses.append(loss.item())


def main():
    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch}...")
        train(epoch + 1)

    # Show a random sample from the dataset
    network.eval()
    dataloader_test = DataLoader(dataset, batch_size=batch_size_test, shuffle=True, num_workers=0)
    img_test, bb1 = next(iter(dataloader_test))
    img_test, bb1 = img_test.to(device), bb1.to(device)
    with torch.no_grad():
        bb2 = network(img_test).cpu()
        print(bb2)

    img_test, bb1 = img_test.cpu(), bb1.cpu()
    for idx in range(batch_size_test):
        fig, ax = plt.subplots()
        ax.imshow(img_test[idx, 0], cmap='gray', vmin=0, vmax=1)
        rect1 = patches.Rectangle(bb1[idx, :2], bb1[idx, 2], bb1[idx, 3], linewidth=1, edgecolor='g', facecolor='none')
        rect2 = patches.Rectangle(bb2[idx, :2], bb2[idx, 2], bb2[idx, 3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
    plt.show()


if __name__ == '__main__':
    main()
