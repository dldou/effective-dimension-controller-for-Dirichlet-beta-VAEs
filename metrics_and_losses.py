import numpy as np
import torch


def spectral_angle(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v) + 1e-12
    cosine = np.clip(np.dot(u, v) / denom, -1.0, 1.0)
    return float(np.arccos(cosine))


def batch_d2(gamma: torch.Tensor) -> torch.Tensor:
    gamma0 = gamma.sum(dim=1)
    mean = gamma / gamma0.unsqueeze(1)
    return (gamma0 + 1.0) / (gamma0 * mean.square().sum(dim=1) + 1.0)


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return ((x - x_hat) ** 2).sum(dim=1).mean()


def kl_to_symmetric_dirichlet(gamma: torch.Tensor, alpha_prior: torch.Tensor) -> torch.Tensor:
    posterior = torch.distributions.Dirichlet(gamma)
    prior = torch.distributions.Dirichlet(alpha_prior)
    return torch.distributions.kl.kl_divergence(posterior, prior).mean()
