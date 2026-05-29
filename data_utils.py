from pathlib import Path

import numpy as np
import scipy.io

from metrics_and_losses import spectral_angle


def deterministic_d2(abundances: np.ndarray) -> np.ndarray:
    """
        Compute the true effective dimension 
            d2 = 1 / sum_k a_k^2 
        for each abundance vector a.

        ( //!\\ Can be obtained here, only because when we have direct 
              access to the true abundances in the dataset.)
    """
    return 1.0 / np.sum(abundances ** 2, axis=1)


def _select_random(datalib: np.ndarray, num_components: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(datalib.shape[1], size=num_components, replace=False)


def _select_separated(datalib: np.ndarray, num_components: int, rng: np.random.Generator) -> np.ndarray:
    chosen = [int(rng.integers(datalib.shape[1]))]

    while len(chosen) < num_components:
        best_index = -1
        best_min_sad = -1.0
        for index in range(datalib.shape[1]):
            if index in chosen:
                continue
            min_sad = min(spectral_angle(datalib[:, index], datalib[:, selected]) for selected in chosen)
            if min_sad > best_min_sad:
                best_min_sad = min_sad
                best_index = index
        chosen.append(best_index)

    return np.array(chosen, dtype=int)


def _select_correlated(datalib: np.ndarray, num_components: int) -> np.ndarray:
    best_pair_sad = float("inf")
    first = 0
    second = 1

    for i in range(datalib.shape[1]):
        for j in range(i + 1, datalib.shape[1]):
            sad = spectral_angle(datalib[:, i], datalib[:, j])
            if sad < best_pair_sad:
                best_pair_sad = sad
                first = i
                second = j

    chosen = [first, second]

    while len(chosen) < num_components:
        best_index = -1
        best_mean_sad = float("inf")
        for index in range(datalib.shape[1]):
            if index in chosen:
                continue
            mean_sad = np.mean([spectral_angle(datalib[:, index], datalib[:, selected]) for selected in chosen])
            if mean_sad < best_mean_sad:
                best_mean_sad = mean_sad
                best_index = index
        chosen.append(best_index)

    return np.array(chosen, dtype=int)


def _ensure_unique_indices(
    indices: np.ndarray,
    library_size: int,
    num_components: int,
    rng: np.random.Generator,
) -> np.ndarray:
    unique_indices = list(dict.fromkeys(np.asarray(indices, dtype=int).tolist()))
    missing = num_components - len(unique_indices)

    if missing <= 0:
        return np.array(unique_indices[:num_components], dtype=int)

    available = np.setdiff1d(
        np.arange(library_size, dtype=int),
        np.array(unique_indices, dtype=int),
        assume_unique=False,
    )
    if available.size < missing:
        raise ValueError(
            f"Cannot select {num_components} distinct endmembers from a library of size {library_size}."
        )

    extra = rng.choice(available, size=missing, replace=False)
    unique_indices.extend(extra.tolist())
    return np.array(unique_indices, dtype=int)


def _resolve_usgs_mat_path(data_root: str | Path = "./data", usgs_mat: str | Path | None = None) -> Path:
    if usgs_mat is not None:
        path = Path(usgs_mat).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"USGS mat file not found: {path}")
        return path

    candidates = [
        Path(data_root).expanduser().resolve() / "raw" / "spectral_library" / "USGS" / "USGS_1995_Library.mat",
        Path("../beta_controller_vae/data/raw/spectral_library/USGS/USGS_1995_Library.mat").resolve(),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find USGS_1995_Library.mat. "
        "Pass --usgs-mat explicitly or provide a valid --data-root."
    )


def load_usgs_library(data_root: str | Path = "./data", usgs_mat: str | Path | None = None) -> tuple[np.ndarray, Path]:
    usgs_path = _resolve_usgs_mat_path(data_root=data_root, usgs_mat=usgs_mat)
    raw = scipy.io.loadmat(usgs_path)
    raw_library = np.array(raw["datalib"], dtype=np.float64)

    col_min = raw_library.min(axis=0, keepdims=True)
    col_max = raw_library.max(axis=0, keepdims=True)
    col_range = np.where(col_max > col_min, col_max - col_min, 1.0)
    normalized_library = ((raw_library - col_min) / col_range).astype(np.float32)

    return normalized_library, usgs_path


def make_usgs_synthetic_dataset(
    num_samples: int = 8192,
    num_components: int = 4,
    alpha: float = 0.4,
    noise_scale: float = 0.01,
    regime: str = "correlated",
    seed: int = 0,
    data_root: str | Path = "./data",
    usgs_mat: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    datalib, usgs_path = load_usgs_library(data_root=data_root, usgs_mat=usgs_mat)
    rng = np.random.default_rng(seed)

    if num_components < 2:
        raise ValueError("num_components must be at least 2 for the USGS synthetic demo.")
    if num_components > datalib.shape[1]:
        raise ValueError(f"num_components={num_components} exceeds library size {datalib.shape[1]}.")

    if regime == "separated":
        indices = _select_separated(datalib, num_components, rng)
    elif regime == "correlated":
        indices = _select_correlated(datalib, num_components)
    elif regime == "random":
        indices = _select_random(datalib, num_components, rng)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    indices = _ensure_unique_indices(indices, datalib.shape[1], num_components, rng)
    endmembers = datalib[:, indices].T.astype(np.float32)
    abundances = rng.dirichlet(alpha * np.ones(num_components), size=num_samples).astype(np.float32)

    noiseless_data = abundances @ endmembers
    sigma = float(noise_scale * np.median(np.linalg.norm(noiseless_data, axis=1)))
    noise = (rng.standard_normal(size=noiseless_data.shape) * sigma).astype(np.float32)
    data = noiseless_data + noise

    metadata = {
        "usgs_mat_path": str(usgs_path),
        "regime": regime,
        "selected_indices": indices.tolist(),
        "sigma_noise": sigma,
        "num_bands": int(datalib.shape[0]),
    }

    return data.astype(np.float32), abundances, endmembers, metadata
