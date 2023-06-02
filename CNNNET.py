import torch.nn as nn
import torch

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,stride=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(6*6*256, 2048)
        self.dr1=nn.Dropout(0.5)
        self.fc2=nn.Linear(2048,5)


    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)

        # print(x.shape)
        # exit(2)

        x = x.view(-1, 6*6*256)

        x = self.fc1(x)
        x=self.dr1(x)
        x=self.fc2(x)

        x=nn.functional.softmax(x)
        return x