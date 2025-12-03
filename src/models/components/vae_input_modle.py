import torch
from torch import nn
from torch.nn import functional as F

class VAEInputModel(nn.Module):
    def __init__(self, input_dim=256, latent_global=16, latent_local=32, hidden_dim=128):
        super(VAEInputModel, self).__init__()
        self.encoder_local = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.GELU(),
            # nn.Linear(2*hidden_dim, 2*hidden_dim),
            # nn.LayerNorm(2*hidden_dim),
            # nn.GELU(),
            # nn.Linear(4*hidden_dim, 8*hidden_dim),
            # nn.LayerNorm(8*hidden_dim),
            # nn.GELU(),
        )
        self.mean_local = nn.Linear(hidden_dim, latent_local)
        self.log_variance_local = nn.Linear(hidden_dim, latent_local)

        self.encoder_global = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mean_global = nn.Linear(hidden_dim, latent_global)
        self.log_variance_global = nn.Linear(hidden_dim, latent_global)

        self.decoder = nn.Sequential(
            nn.Linear(latent_local+latent_global, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            # nn.Linear(8*hidden_dim, 4*hidden_dim),
            # nn.LayerNorm(4*hidden_dim),
            # nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.latent_local = latent_local
        self.latent_global = latent_global

    def encode_local(self, x):
        h = self.encoder_local(x)
        mean = self.mean_local(h)
        logvar = self.log_variance_local(h)
        return mean, logvar

    def encode_global(self, x):
        h = self.encoder_global(x)
        mean = self.mean_global(h)
        logvar = self.log_variance_global(h)
        return mean, logvar

    def reparameterize(self, mean, log_variance, train=True):
        if not train:
            return mean
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self,z):
        return self.decoder(z)

    def forward(self, x, pooled_x, train=True):
        mean_local, log_variance_local = self.encode_local(x)
        mean_global, log_variance_global = self.encode_global(pooled_x)
        z_local = self.reparameterize(mean_local, log_variance_local, train)
        z_global = self.reparameterize(mean_global, log_variance_global, train)
        
        B = pooled_x.size(0)
        T = x.size(0) // B
        z_global_exp = z_global.unsqueeze(1).expand(B, T, -1).reshape(B*T, -1)

        recovered_x = self.decode(z_local, z_global_exp)
        return recovered_x, mean, log_variance_local, mean_global, log_variance_global
