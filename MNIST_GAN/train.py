import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm

from models import Generator, Discriminator
from data_loader import get_data_loader
from utils import create_noise, show_result

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 100
learning_rate = 0.0002
epochs = 100
image_size = 28
G_input_dim = 100
G_output_dim = image_size * image_size
D_input_dim = image_size * image_size

# Load dataset
train_loader = get_data_loader(batch_size)

# Initialize models
G_net = Generator(G_input_dim, G_output_dim).to(device)
D_net = Discriminator(D_input_dim).to(device)

# Loss function
criterion = nn.BCELoss().to(device)

# Optimizers
G_optimizer = optim.Adam(G_net.parameters(), lr=learning_rate)
D_optimizer = optim.Adam(D_net.parameters(), lr=learning_rate)

# Training loop
train_hist = {'D_losses': [], 'G_losses': []}
start_time = time.time()

for epoch in range(epochs):
    Loss_G, Loss_D = [], []
    for (image, _) in tqdm(train_loader):
        image = image.view(batch_size, D_input_dim).to(device)
        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        data_fake = G_net(create_noise(batch_size, G_input_dim).to(device))
        loss_d = criterion(D_net(image), real_label) + criterion(D_net(data_fake), fake_label)
        D_optimizer.zero_grad()
        loss_d.backward()
        D_optimizer.step()

        # Train Generator
        data_fake = G_net(create_noise(batch_size, G_input_dim).to(device))
        loss_g = criterion(D_net(data_fake), real_label)
        G_optimizer.zero_grad()
        loss_g.backward()
        G_optimizer.step()

        Loss_D.append(loss_d.item())
        Loss_G.append(loss_g.item())

    # Save training losses
    train_hist['D_losses'].append(np.mean(Loss_D))
    train_hist['G_losses'].append(np.mean(Loss_G))

    # Save generated images
    show_result(G_net, create_noise(25, 100).to(device), (epoch + 1), save=True, path=f'./MNIST_GAN_results/epoch_{epoch+1}.png')

# Save model
torch.save(G_net.state_dict(), "./MNIST_GAN_results/generator.pth")
torch.save(D_net.state_dict(), "./MNIST_GAN_results/discriminator.pth")
print("Training completed! Models saved.")
