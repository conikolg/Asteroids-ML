import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt, patches
from torch.utils.data import DataLoader

from util.asteroid_dataset import AsteroidDataset

dataset = AsteroidDataset("./datasets/img", "./datasets/lbl.npy")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


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
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        # print(batch_idx, data.shape, target.shape)
        optimizer.zero_grad()
        # print(data.shape, data.dtype)
        output = network(data)
        # print(output, output.shape, output.dtype)
        # print(target, target.shape, target.dtype)
        # mse_loss = nn.MSELoss()
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(dataloader.dataset)))


for epoch in range(n_epochs):
    train(epoch + 1)

# Show a random sample from the dataset
network.eval()
dataloader_test = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
for _ in range(10):
    fig, ax = plt.subplots()
    img_test, bb1 = next(iter(dataloader_test))
    # print(img_test.shape)
    with torch.no_grad():
        bb2 = network(img_test)
        print(bb2)
    ax.imshow(img_test[0], cmap='gray', vmin=0, vmax=1)
    rect1 = patches.Rectangle(bb1[0, :2], bb1[0, 2], bb1[0, 3], linewidth=1, edgecolor='g', facecolor='none')
    rect2 = patches.Rectangle(bb2[0, :2], bb2[0, 2], bb2[0, 3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
plt.show()
