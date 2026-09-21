# Data directory

Datasets are not distributed with this repository.

Use `scripts/generate_sqg_data.py` to generate two-channel SQG trajectories,
then convert and split them with `scripts/convert_nc_to_npy.py` and
`scripts/split_dataset.py`. Pixel-DDPM training expects IID shards with shape
`(N, 2, H, W)`. Latent-prior training expects trajectory files with shape
`(T, 2, H, W)`.

