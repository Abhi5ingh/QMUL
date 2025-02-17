import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dims, 256), nn.LeakyReLU(0.2))
        self.fc2 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.2))
        self.fc3 = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(0.2))
        self.fc4 = nn.Sequential(nn.Linear(1024, output_dims), nn.Tanh())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dims):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dims, 1024), nn.LeakyReLU(0.2))
        self.fc2 = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(0.2))
        self.fc3 = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.2))
        self.fc4 = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
