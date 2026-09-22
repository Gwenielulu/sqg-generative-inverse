# SQG Generative Inverse

Diffusion- and latent-prior methods for reconstructing sparsely observed, noisy two-channel surface quasi-geostrophic (SQG) turbulence.

## Methods

### Pixel-space prior

- Unconditional reverse diffusion (`vanilla`)
- Observation-consistency projection (`projection`)
- Manifold-constrained gradient with projection (`mcg`)
- Diffusion posterior sampling (`dps`)
- Monte-Carlo posterior-sampling variant (`dps_plus`)
- Diffusion optimal control with matrix-free DDP (`doc`)

### Latent-space prior

- Periodic convolutional autoencoder
- IID latent prior based on rectified flow
- Variational posterior reconstruction in the compressed latent space

The physical-space experiments use 64 x 64 fields in the supplied examples. The latent-space examples use 256 x 256 fields compressed to a learned latent representation. These defaults reproduce the two experimental branches from which the code was consolidated; they can be changed in the YAML files.

## Repository layout

```text
configs/                 Model, inverse-method, and experiment configurations
src/sqg_inverse/
  data/                  Trajectory and IID-shard loaders
  models/                Pixel DDPM, autoencoder, and latent prior
  inverse/               DPS-family, optimal-control, and latent-VI solvers
  evaluation/            Field and spectral evaluation utilities
scripts/                 Data, training, reconstruction, and evaluation CLIs
tests/                   Lightweight unit tests
data/                    Local datasets (not tracked)
checkpoints/             Local model weights (not tracked)
outputs/                 Reconstruction outputs (not tracked)
```

## Installation

```bash
conda env create -f environment.yml
conda activate sqg-generative-inverse
pip install -e .
```

## Data preparation

Generate a small 64 x 64 example dataset:

```bash
python scripts/generate_sqg_data.py \
  --resolution 64 \
  --out_dir data/sqg64/netcdf \
  --ntraj 10 \
  --nsteps 100 \
  --chunk_size 2
```

Convert the trajectories into IID shards for pixel-DDPM training:

```bash
python scripts/make_iid_shards.py \
  --input_dir data/sqg64/netcdf \
  --output_dir data/sqg64/iid/train \
  --samples_per_shard 10000
```

For the latent branch, generate 256 x 256 trajectories and retain the temporal dimension:

```bash
python scripts/convert_nc_to_npy.py \
  --input_dir data/sqg256/netcdf \
  --output_dir data/sqg256/all \
  --expected_t 1000 \
  --expected_c 2 \
  --expected_h 256 \
  --expected_w 256

python scripts/split_dataset.py \
  --input_dir data/sqg256/all \
  --output_dir data/sqg256 \
  --expected_total 50
```

## Model training

Train the pixel-space DDPM:

```bash
python scripts/train_pixel_ddpm.py --config configs/train_pixel_ddpm.yaml
```

Train the autoencoder and latent prior:

```bash
python scripts/train_autoencoder.py --config configs/train_autoencoder.yaml

python scripts/cache_latents.py \
  --config configs/cache_latents.yaml \
  --ae_checkpoint runs/autoencoder/autoencoder_best.pt

python scripts/train_latent_prior.py \
  --config configs/train_latent_prior.yaml
```

## Configuration-driven reconstruction

The same entry point selects the inverse method from the experiment YAML:

```bash
python scripts/reconstruct.py \
  --config configs/experiments/physical/sqg64_dps.yaml

python scripts/reconstruct.py \
  --config configs/experiments/physical/sqg64_dps_plus.yaml

python scripts/reconstruct.py \
  --config configs/experiments/physical/sqg64_doc.yaml

python scripts/reconstruct.py \
  --config configs/experiments/latent/sqg256_latent_vi.yaml
```

For physical-space experiments, change only the referenced inverse configuration to select `vanilla`, `projection`, `mcg`, `dps`, `dps_plus`, or `diffusion_optimal_control`. The program writes the truth, observations, mask, reconstruction, metrics, and resolved configuration to the selected output directory.

## Acknowledgments

The SQG model and data-generation code are adapted from Jeffrey S. Whitaker's [`sqgturb`](https://github.com/jswhit/sqgturb) implementation. The dynamical formulation follows:

- Ross Tulloch and K. Shafer Smith, “A Note on the Numerical Representation of Surface Dynamics in Quasigeostrophic Turbulence: Application to the Nonlinear Eady Model,” *Journal of the Atmospheric Sciences*, 66(4), 1063–1068, 2009. https://doi.org/10.1175/2008JAS2921.1

This repository uses or adapts components from the following open-source projects:

- OpenAI’s **guided-diffusion** for the two-channel DDPM U-Net and Gaussian diffusion routines, associated with Prafulla Dhariwal and Alex Nichol, “Diffusion Models Beat GANs on Image Synthesis,” *Advances in Neural Information Processing Systems*, 2021.

- Henry Li and Marcus Pereira’s **diffusion_optimal_control** for the matrix-free DDP components, associated with “Solving Inverse Problems via Diffusion Optimal Control,” *Advances in Neural Information Processing Systems*, 2024.

- François Rozet and collaborators’ **LOLA** for parts of the autoencoder architecture, neural-network layers, data augmentation, and latent-model training code, associated with François Rozet, Ruben Ohana, Michael McCabe, Gilles Louppe, François Lanusse, and Shirley Ho, “Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics Emulation,” *Advances in Neural Information Processing Systems*, 2025.

- Phil Wang’s **rectified-flow-pytorch** for parts of the rectified-flow implementation. The underlying method follows Xingchao Liu, Chengyue Gong, and Qiang Liu, “Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow,” *International Conference on Learning Representations*, 2023.

The repository-local inverse-conditioning implementations follow the methods introduced in:

- Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, and Jong Chul Ye, “Diffusion Posterior Sampling for General Noisy Inverse Problems,” *International Conference on Learning Representations*, 2023.

- Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye, “Improving Diffusion Models for Inverse Problems Using Manifold Constraints,” *Advances in Neural Information Processing Systems*, 2022.
