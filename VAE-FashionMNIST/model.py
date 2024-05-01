import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


# --- defines the model --- #
class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        # self.encoder = nn.Linear(784, 500)
        self.encoder = nn.Sequential(nn.Conv2d(1, 128, kernel_size=3, stride=2),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 512, kernel_size=3, stride=2),
                                     nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=2),
                                     nn.ReLU(),
                                     nn.Flatten(),
                                     nn.Linear(2048, 500))
        self.mu = nn.Linear(500, z_dim)
        self.var = nn.Linear(500, z_dim)
        self.fc1 = nn.Linear(z_dim, 500)
        self.fc2 = nn.Linear(500, 784)

    def encode(self, x):
        h1 = F.relu(self.encoder(x))
        mu = self.mu(h1)
        logvar = self.var(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h3))

    def forward(self, x):
        # x: [batch size, 1, 28,28] -> x: [batch size, 784]
        # x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar