import torch
from torch import nn
from torch.nn import functional as F

class VAEInputModel(nn.Module):
    def __init__(self, input_dim=256, latent_dim=64, hidden_dim=128):
        super(VAEInputModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mean = self.mean(h)
        logvar = self.log_variance(h)
        return mean, logvar

    def reparameterize(self, mean, log_variance, train=True):
        if not train:
            return mean
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, train=True):
        mean, log_variance = self.encode(x)
        z = self.reparameterize(mean, log_variance, train)
        recovered_x = self.decode(z)
        return recovered_x, mean, log_variance
    
    def loss_function(self, recon_x, x, mean, log_variance):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
        return BCE + KLD, BCE, KLD