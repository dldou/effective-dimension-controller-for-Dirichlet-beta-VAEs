import torch
import numpy as np

from controller import D2BetaController
from metrics_and_losses import (
    batch_d2, 
    kl_to_symmetric_dirichlet, 
    reconstruction_loss
)

def train_beta_dirvae(
    model,
    data: np.ndarray,
    target_d2: float,
    *,
    alpha_prior: float = 1.0,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    beta_init: float = 1e-3,
    beta_min: float = 1e-8,
    beta_max: float = 1e1,
    controller_eta: float = 0.05,
    controller_ema_gamma: float = 0.1,
    controller_deadzone: float = 0.05,
    controller_warmup_epochs: int = 0,
    device: str = "cpu",
) -> dict[str, list[float]]:
    x = torch.as_tensor(data, dtype=torch.float32, device=device)
    model = model.to(device)

    alpha_prior_tensor = torch.full(
        (model.latent_dim,),
        fill_value=alpha_prior,
        dtype=torch.float32,
        device=device,
    )

    controller = D2BetaController(
        num_components=model.latent_dim,
        target_d2=target_d2,
        beta_init=beta_init,
        beta_min=beta_min,
        beta_max=beta_max,
        eta=controller_eta,
        ema_gamma=controller_ema_gamma,
        deadzone=controller_deadzone,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "beta": [],
        "beta_next": [],
        "d2_epoch": [],
        "d2_ema": [],
        "loss": [],
        "recon": [],
        "kl": [],
    }

    num_samples = x.shape[0]

    for epoch in range(epochs):
        beta = controller.get_beta()
        permutation = torch.randperm(num_samples, device=device)

        epoch_gammas = []
        epoch_losses = []
        epoch_recons = []
        epoch_kls = []

        for start in range(0, num_samples, batch_size):
            batch_indices = permutation[start:start + batch_size]
            batch = x[batch_indices]

            gamma, _, x_hat = model(batch)
            recon = reconstruction_loss(batch, x_hat)
            kl = kl_to_symmetric_dirichlet(gamma, alpha_prior_tensor)
            loss = recon + beta * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # positivity constraint on decoder weights 
            # (endmember spectra)
            with torch.no_grad():
                model.decoder.weight.clamp_(min=0.0)

            epoch_gammas.append(gamma.detach())
            epoch_losses.append(loss.item())
            epoch_recons.append(recon.item())
            epoch_kls.append(kl.item())

        gamma_epoch = torch.cat(epoch_gammas, dim=0)
        d2_epoch = batch_d2(gamma_epoch).mean().item()

        if epoch + 1 > controller_warmup_epochs:
            d2_ema, beta_next = controller.update(d2_epoch)
        else:
            d2_ema = float("nan")
            beta_next = beta

        history["beta"].append(beta)
        history["beta_next"].append(beta_next)
        history["d2_epoch"].append(d2_epoch)
        history["d2_ema"].append(d2_ema)
        history["loss"].append(float(np.mean(epoch_losses)))
        history["recon"].append(float(np.mean(epoch_recons)))
        history["kl"].append(float(np.mean(epoch_kls)))

        should_print = (epoch == 0) or ((epoch + 1) % max(1, epochs // 10) == 0) or (epoch + 1 == epochs)
        if should_print:
            warmup_tag = " warmup" if epoch + 1 <= controller_warmup_epochs else ""
            print(
                f"epoch {epoch + 1:03d}/{epochs} "
                f"beta={beta:.3e} "
                f"d2_epoch={d2_epoch:.3f} "
                f"d2_ema={d2_ema:.3f} "
                f"next_beta={beta_next:.3e} "
                f"loss={history['loss'][-1]:.4f}"
                f"{warmup_tag}"
            )

    return history
