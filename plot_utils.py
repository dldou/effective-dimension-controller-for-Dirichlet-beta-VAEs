from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from metrics_and_losses import spectral_angle


def match_endmembers(predicted: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_endmembers = ground_truth.shape[0]
    cost = np.zeros((num_endmembers, num_endmembers), dtype=np.float64)

    for gt_index in range(num_endmembers):
        for pred_index in range(num_endmembers):
            cost[gt_index, pred_index] = spectral_angle(ground_truth[gt_index], predicted[pred_index])

    gt_indices, pred_indices = linear_sum_assignment(cost)
    order = pred_indices[np.argsort(gt_indices)]
    matched_predicted = predicted[order]
    sad_per_endmember = cost[np.arange(num_endmembers), order]
    return matched_predicted, order, sad_per_endmember



def save_training_summary_plot(history: dict[str, list[float]], output_path: str | Path) -> None:
    epochs = np.arange(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(epochs, history["loss"])
    axes[0, 0].set_title("Total loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("value")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["recon"])
    axes[0, 1].set_title("Reconstruction loss")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("value")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["kl"])
    axes[1, 0].set_title("KL divergence")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("value")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["beta"], label="beta")
    axes[1, 1].plot(epochs, history["beta_next"], linestyle="--", alpha=0.7, label="next beta")
    axes[1, 1].set_title("Beta")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("value")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)



def save_controller_plot(history: dict[str, list[float]], target_d2: float, output_path: str | Path) -> None:
    epochs = np.arange(1, len(history["beta"]) + 1)
    target_curve = np.full_like(epochs, fill_value=target_d2, dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(epochs, history["beta"], label="beta")
    axes[0].plot(epochs, history["beta_next"], linestyle="--", alpha=0.7, label="next beta")
    axes[0].set_title("Beta")
    axes[0].set_ylabel("value")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["d2_epoch"], label="d2 epoch")
    if not np.all(np.isnan(history["d2_ema"])):
        axes[1].plot(epochs, history["d2_ema"], label="d2 ema")
    axes[1].plot(epochs, target_curve, linestyle=":", color="black", linewidth=2.0, label="target d2")
    axes[1].set_title("Effective dimension d2")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)



def save_endmembers_plot(model, ground_truth_endmembers: np.ndarray, output_path: str | Path) -> None:
    predicted_endmembers = model.decoder.weight.detach().cpu().numpy().T
    matched_predicted, order, sad_per_endmember = match_endmembers(predicted_endmembers, ground_truth_endmembers)

    num_endmembers, num_bands = ground_truth_endmembers.shape
    num_cols = int(np.ceil(np.sqrt(num_endmembers)))
    num_rows = int(np.ceil(num_endmembers / num_cols))
    bands = np.arange(1, num_bands + 1)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows), squeeze=False)

    for endmember_index in range(num_endmembers):
        row = endmember_index // num_cols
        col = endmember_index % num_cols
        ax = axes[row, col]

        ax.plot(bands, ground_truth_endmembers[endmember_index], color="black", linewidth=2.0, label="ground truth")
        ax.plot(
            bands,
            matched_predicted[endmember_index],
            color="tab:blue",
            linewidth=2.0,
            linestyle="--",
            label="predicted",
        )
        ax.set_title(
            f"EM {endmember_index + 1} | pred {order[endmember_index] + 1} | SAD {np.degrees(sad_per_endmember[endmember_index]):.2f}°"
        )
        ax.set_xlabel("band")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        if endmember_index == 0:
            ax.legend()

    for axis_index in range(num_endmembers, num_rows * num_cols):
        row = axis_index // num_cols
        col = axis_index % num_cols
        axes[row, col].set_visible(False)

    mean_sad_deg = np.degrees(sad_per_endmember).mean()
    fig.suptitle(f"Final endmembers matched with Hungarian + SAD | mean SAD {mean_sad_deg:.2f}°")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, format="pdf")
    plt.close(fig)



def save_training_plots(
    history: dict[str, list[float]],
    model,
    ground_truth_endmembers: np.ndarray,
    target_d2: float,
    output_dir: str | Path,
    prefix: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_training_summary_plot(history, output_dir / f"{prefix}_training.pdf")
    save_controller_plot(history, target_d2, output_dir / f"{prefix}_controller.pdf")
    save_endmembers_plot(model, ground_truth_endmembers, output_dir / f"{prefix}_endmembers.pdf")
