import argparse
import torch

from data_utils import (
    deterministic_d2,
    make_usgs_synthetic_dataset
)
from models import BetaDirVAE
from plot_utils import save_training_plots
from training import train_beta_dirvae



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the beta-controller demo.")

    parser.add_argument("--output-dir", type=str, default="plots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--alpha-prior", type=float, default=1.0)
    parser.add_argument("--beta-init", type=float, default=1e-3)
    parser.add_argument("--beta-min", type=float, default=1e-8)
    parser.add_argument("--beta-max", type=float, default=1e1)
    parser.add_argument("--controller-eta", type=float, default=0.1)
    parser.add_argument("--controller-ema-gamma", type=float, default=0.2)
    parser.add_argument("--controller-warmup-epochs", type=int, default=2)
    parser.add_argument("--controller-deadzone", type=float, default=0.0)
    parser.add_argument("--target-d2", type=float, default=2.5)

    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--num-components", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.4)

    parser.add_argument("--synthetic-regime", choices=["separated", "correlated", "random"], default="correlated")
    parser.add_argument("--synthetic-noise-scale", type=float, default=0.01)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--usgs-mat", type=str, default=None)

    return parser.parse_args()



def build_dataset(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    data, abundances, endmembers, metadata = make_usgs_synthetic_dataset(
        num_samples=args.num_samples,
        num_components=args.num_components,
        alpha=args.alpha,
        noise_scale=args.synthetic_noise_scale,
        regime=args.synthetic_regime,
        seed=args.seed,
        data_root=args.data_root,
        usgs_mat=args.usgs_mat,
    )
    print("dataset=synthetic_demo")
    print(f"usgs_mat={metadata['usgs_mat_path']}")
    print(f"regime={metadata['regime']}")
    print(f"selected_indices={metadata['selected_indices']}")
    return data, abundances, endmembers, "synthetic_demo"



def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data, abundances, endmembers, prefix = build_dataset(args)

    true_mean_d2 = deterministic_d2(abundances).mean().item()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device={device}")
    print(f"reference deterministic d2 from abundances: {true_mean_d2:.3f}")
    print(f"controller target d2: {args.target_d2:.3f}")

    model = BetaDirVAE(input_dim=data.shape[1], latent_dim=abundances.shape[1], hidden_dim=args.hidden_dim)
    history = train_beta_dirvae(
        model,
        data,
        target_d2=args.target_d2,
        alpha_prior=args.alpha_prior,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_init=args.beta_init,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        controller_eta=args.controller_eta,
        controller_ema_gamma=args.controller_ema_gamma,
        controller_deadzone=args.controller_deadzone,
        controller_warmup_epochs=args.controller_warmup_epochs,
        device=device,
    )

    save_training_plots(history, model, endmembers, args.target_d2, output_dir=args.output_dir, prefix=prefix)

    print(
        f"final beta used={history['beta'][-1]:.3e} "
        f"final d2_epoch={history['d2_epoch'][-1]:.3f} "
        f"final d2_ema={history['d2_ema'][-1]:.3f}"
    )
    print(
        f"saved plots to {args.output_dir}/{prefix}_training.pdf, "
        f"{args.output_dir}/{prefix}_controller.pdf and {args.output_dir}/{prefix}_endmembers.pdf"
    )


if __name__ == "__main__":
    main()
