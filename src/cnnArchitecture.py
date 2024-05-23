
import torch

import torch.nn as nn
import torch.nn.functional as FL


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,1)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*22*22, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # x: n, 3, 32, 32
        x = self.pool(FL.relu(self.conv1(x))) # x: n, 6, 14, 14 
        x = self.pool(FL.relu(self.conv2(x))) # x: n, 16, 5, 5
        x = x.view(-1, 16 * 22 * 22)            # x: n, 400 flatten 3D Tensor to 1D Tesnsor
        x = FL.relu(self.fc1(x))              # x: n, 120
        x = FL.relu(self.fc2(x))              # x: n, 84
        x = self.fc3(x)                       # x: n, 10
        return x