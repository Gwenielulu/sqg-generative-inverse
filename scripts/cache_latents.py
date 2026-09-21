import torch
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
from sqg_inverse.models.autoencoder import SQGAutoencoder


def cache_latents(config, ae_checkpoint):
    data_cfg = config["data"]
    model_cfg = config["model"]
    output_cfg = config["output"]
    device = config["device"]
    frame_batch_size = config["batch_size"]
    if frame_batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {frame_batch_size}")
    output_dir = Path(output_cfg["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_cfg["data_dir"])
    files = sorted(data_dir.glob("*.npy"))
    if len(files) == 0:
        raise ValueError(f"No .npy files found in {data_dir}")
    print(f"Loading autoencoder from {ae_checkpoint}...")
    model = SQGAutoencoder(**model_cfg).to(device)
    checkpoint = torch.load(ae_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Encoding {len(files)} files to latent space...")
    print(f"Frame batch size: {frame_batch_size}")
    first_input_shape = None
    first_latent_shape = None
    with torch.no_grad():
        for file_path in tqdm(files):
            x_np = np.load(file_path)
            if x_np.ndim != 4:
                raise ValueError(
                    f"Expected (T, C, H, W), got {x_np.shape} in {file_path}"
                )
            x_all = torch.from_numpy(x_np).float()
            z_chunks = []
            T = x_all.shape[0]
            for start in range(0, T, frame_batch_size):
                end = min(start + frame_batch_size, T)
                x_chunk = x_all[start:end].to(device, non_blocking=True)
                z_chunk = model.encode(x_chunk)
                z_chunks.append(z_chunk.cpu())
            z_np = torch.cat(z_chunks, dim=0).numpy()
            save_path = output_dir / f"{file_path.stem}_latent.npy"
            np.save(save_path, z_np)
            if first_input_shape is None:
                first_input_shape = x_np.shape
                first_latent_shape = z_np.shape
    assert first_input_shape is not None and first_latent_shape is not None
    original_size = int(np.prod(first_input_shape)) * 4
    latent_size = int(np.prod(first_latent_shape)) * 4
    compression_ratio = original_size / latent_size
    print(f"\nCaching complete!")
    print(f"Example input shape: {first_input_shape}")
    print(f"Example latent shape: {first_latent_shape}")
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"Cached to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cache latents (config-driven)")
    parser.add_argument("--config", type=str, required=True, help="Cache config YAML")
    parser.add_argument(
        "--ae_checkpoint", type=str, required=True, help="Trained AE checkpoint"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print("=" * 60)
    print("Cache Configuration:")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 60)
    cache_latents(config, args.ae_checkpoint)
