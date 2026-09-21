from pathlib import Path
import torch
import yaml
from ..models.autoencoder import SQGAutoencoder

DEFAULT_SCALE_FACTOR = 0.0001 * 300.0 / 9.8


def load_ae_decoder(ae_config_path, ae_checkpoint_path, device):
    with open(Path(ae_config_path)) as f:
        config = yaml.safe_load(f)
    ae = SQGAutoencoder(**config["model"]).to(device)
    checkpoint = torch.load(
        Path(ae_checkpoint_path), map_location=device, weights_only=False
    )
    ae.load_state_dict(checkpoint["model_state_dict"])
    ae.eval()
    ae.requires_grad_(False)
    return ae


def decode_latents(ae, z, batch_size=2, amp=True):
    device = next(ae.parameters()).device
    z = z.to(device, non_blocking=True)
    decoded = []
    with torch.inference_mode():
        for start in range(0, z.shape[0], batch_size):
            chunk = z[start : start + batch_size]
            with torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
            ):
                if chunk.ndim == 4:
                    x = ae.decode(chunk, noisy=False)
                else:
                    x = ae.decode_trajectory(chunk, noisy=False)
            decoded.append(x.cpu())
    return torch.cat(decoded)


def restore_physical(x, sigma_train=2660.0, scale_factor=DEFAULT_SCALE_FACTOR):
    return x * sigma_train * scale_factor


def save_sample_field_vis(x, out_file, vmin=-20.0, vmax=20.0, cmap="jet"):
    import matplotlib.pyplot as plt

    x = x.detach().cpu().numpy()
    fig, axes = plt.subplots(1, x.shape[0], figsize=(3.2 * x.shape[0], 3.0))
    axes = [axes] if x.shape[0] == 1 else axes
    for channel, ax in enumerate(axes):
        ax.imshow(x[channel], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
        ax.set_axis_off()
        ax.set_title(f"ch{channel}")
    fig.tight_layout(pad=0.25)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_forecast_compare_vis(gt, pred, out_file, vmin=-20.0, vmax=20.0, cmap="jet"):
    import matplotlib.pyplot as plt

    gt = gt.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    fields = (gt, pred, pred - gt)
    cmaps = (cmap, cmap, "bwr")
    labels = ("GT", "Pred", "Diff")
    fig, axes = plt.subplots(3, gt.shape[0], figsize=(3.2 * gt.shape[0], 8.8))
    if gt.shape[0] == 1:
        axes = axes[:, None]
    for row, field in enumerate(fields):
        for channel in range(gt.shape[0]):
            axes[row, channel].imshow(
                field[channel],
                cmap=cmaps[row],
                vmin=vmin,
                vmax=vmax,
                interpolation="none",
            )
            axes[row, channel].set_axis_off()
        axes[row, 0].set_ylabel(labels[row])
    fig.tight_layout(pad=0.3)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
