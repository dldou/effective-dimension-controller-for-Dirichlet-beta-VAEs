import torch
from torch import nn
from torch.nn import functional as F


class BetaDirVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        gamma = F.softplus(self.encoder(x)) + 1e-4
        return gamma.clamp(min=1e-4, max=1e3)

    def sample_latent(self, gamma: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Dirichlet(gamma).rsample()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = self.encode(x)
        z = self.sample_latent(gamma)
        x_hat = self.decode(z)
        return gamma, z, x_hat
