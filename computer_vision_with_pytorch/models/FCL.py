import torch.nn as nn
import torch

'''
This is the Fully-Connected_Layer model for differenciating number images for the MNIST dataset
'''
class mnist_FCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1 * 28 * 28, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out

'''
From here on, the models are for CIFAR10 dataset
'''
# Using nn.Sequential, designing the model become much more intuitive
class cifar10_FCL1(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3 * 32 * 32, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out

'''
From here on, the models are for CIFAR100 dataset
'''
class cifar100_FCL1(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3 * 32 * 32, 1024),
            nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            nn.ReLU(),
            torch.nn.Linear(1024, 100)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out