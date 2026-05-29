# Effective-Dimension Control of $\beta$ in Dirichlet $\beta$-VAE for Blind Hyperspectral Unmixing Demo

This repository provides a minimal PyTorch implementation accompanying the paper **Effective-Dimension Control of $\beta$ in Dirichlet $\beta$-VAE for Blind Hyperspectral Unmixing**. The demo implements a Dirichlet $\beta$-VAE in which the KL regularization weight $\beta$ is adapted at the epoch level by a feedback controller to guide the inferred abundance vectors toward a prescribed effective dimension.   
The effective dimension is defined as the second-order Hill number, $d_2(a) = 1 / \sum_k a_k^2$, and measures the effective number of active endmembers represented in each mixture. Synthetic hyperspectral mixtures are generated from spectra drawn from the USGS spectral library.


## Repository Structure

- `models.py`: minimal Dirichlet VAE model
- `data_utils.py`: synthetic data generation utilities
- `controller.py`: `d2` beta-controller
- `training.py`: training loop and epoch-level controller update
- `plot_utils.py`: PDF plotting utilities
- `run_demo.py`: main entry point

## Requirements

The reference environment for this repository is the conda environment `bbhuenv` with:

- `python==3.12.13`
- `torch==2.11.0+cu126`
- `numpy==2.4.3`
- `scipy==1.17.1`
- `matplotlib==3.10.9`


## Main Idea

The training objective is:

```text
loss = reconstruction + beta * KL
```
where reconstruction is the Mean Square Error.

The controller monitors the posterior effective dimension `d2` and updates `beta` once per epoch.
Within a given epoch, `beta` is fixed for all optimization steps.
The updated `beta` value is only used at the next epoch.

The sign convention in `controller.py` is:
- if `d2` is above the target, `beta` is decreased
- if `d2` is below the target, `beta` is increased

## Demo

`run_demo.py` generates synthetic mixtures from the USGS spectral library.
It selects endmembers from the library, samples Dirichlet abundances, forms linear mixtures, and adds signal-dependent Gaussian noise.

```bash
python run_demo.py
```

If the USGS `.mat` file is not found automatically, pass it explicitly:

```bash
python run_demo.py \
  --usgs-mat /path/to/USGS_1995_Library.mat
```

Example:

```bash
python run_demo.py \
  --epochs 5000 \
  --controller-warmup-epochs 2 \
  --num-components 3 \
  --synthetic-regime correlated \
  --alpha 0.4 \
  --synthetic-noise-scale 0.01 \
  --target-d2 2.5
```

Supported synthetic regimes:

- `correlated`
- `separated`
- `random`

## Main Arguments

Key arguments in `run_demo.py`:

- `--epochs`: number of training epochs, default `5000`
- `--controller-warmup-epochs`: number of warmup epochs before controller updates, default `2`
- `--target-d2`: controller target
- `--beta-init`: initial beta
- `--beta-min`: lower beta bound
- `--beta-max`: upper beta bound
- `--alpha-prior`: Dirichlet prior concentration used in the KL term
- `--num-samples`: number of synthetic samples
- `--num-components`: latent dimension / number of endmembers
- `--alpha`: Dirichlet concentration used to generate synthetic abundances
- `--output-dir`: directory where plots are saved

Additional arguments:

- `--synthetic-regime`
- `--synthetic-noise-scale`
- `--data-root`
- `--usgs-mat`

## Generated Outputs

Each run produces three PDF files in `--output-dir`.
The filename prefix is `synthetic_demo`.

### 1. Training Summary

`<prefix>_training.pdf`

This figure contains four subplots:

- total loss
- reconstruction term
- raw KL term
- beta

### 2. Controller Dynamics

`<prefix>_controller.pdf`

This figure contains two subplots:

- beta over epochs
- `d2` over epochs, together with the `target_d2` line

### 3. Endmember Comparison

`<prefix>_endmembers.pdf`

This figure compares the final predicted endmembers with the ground-truth endmembers.
Predicted endmembers are matched to the ground truth using:

- spectral angle distance (SAD)
- Hungarian matching

Each subplot shows one matched pair.

## Example Commands

USGS-based synthetic demo:

```bash
python run_demo.py \
  --epochs 5000 \
  --num-components 3 \
  --synthetic-regime correlated \
  --target-d2 2.5
```

## Notes

- The KL subplot shows the raw KL term, not `beta * KL`.
- Very low or high `target_d2` values are not always reachable for a given dataset, prior, and beta range. See dicussion in the paper.
- If the controller reaches `beta_min` or `beta_max`, it has saturated and cannot push the system further in that direction.

## Citation

If you use this code, please cite 
```bibtex
@inproceedings{doutsas2026effective,
  title     = {Effective-Dimension Control of {$\beta$} in Dirichlet {$\beta$}-VAE for Blind Hyperspectral Unmixing},
  author    = {Doutsas, Delphine and Figliuzzi, Bruno},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2026},
  note      = {MORSE Workshop}
}
```
