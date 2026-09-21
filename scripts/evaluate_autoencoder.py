import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
import yaml
from einops import rearrange
from tqdm import tqdm

from sqg_inverse.data.sqg_dataset import SQGNpyDataset
from sqg_inverse.models.autoencoder import SQGAutoencoder
from sqg_inverse.utils.fourier import (
    isotropic_cross_correlation,
    isotropic_power_spectrum,
)
from sqg_inverse.utils.preprocess import denormalize, load_dataset_stats, normalize

DEFAULT_SCALE_FACTOR = 0.0001 * 300.0 / 9.8
VIS_VMIN = -20.0
VIS_VMAX = 20.0
VIS_CMAP = "jet"
VIS_DIFF_CMAP = "bwr"


def split_dir(data_cfg, split):
    return Path(data_cfg[f"{split}_dir"])


def splits(splits_text):
    return [s.strip() for s in splits_text.split(",") if s.strip()]


def losses(x, y, z, loss_name):
    if loss_name == "mse":
        return (x - y).square().mean()
    if loss_name == "mae":
        return (x - y).abs().mean()
    if loss_name == "vmse":
        x_flat = rearrange(x, "B C ... -> B C (...)")
        y_flat = rearrange(y, "B C ... -> B C (...)")
        term = (x_flat - y_flat).square().mean(dim=2) / (x_flat.var(dim=2) + 0.01)
        return term.mean()
    if loss_name == "vrmse":
        x_flat = rearrange(x, "B C ... -> B C (...)")
        y_flat = rearrange(y, "B C ... -> B C (...)")
        term = (x_flat - y_flat).square().mean(dim=2) / (x_flat.var(dim=2) + 0.01)
        return torch.sqrt(term).mean()
    if loss_name == "similarity":
        f = rearrange(z, "B ... -> B (...)")
        if f.shape[0] < 2:
            return torch.zeros((), device=f.device, dtype=f.dtype)
        sim = cosine_similarity(f[None, :], f[:, None], dim=-1)
        iu = torch.triu_indices(sim.shape[0], sim.shape[1], offset=1, device=sim.device)
        return sim[iu[0], iu[1]].mean()
    raise ValueError(f"Unknown loss type: {loss_name}")


def field_metrics(x, y):
    err = x - y
    se = err.square()
    mse = se.mean()
    mae = err.abs().mean()
    rmse = torch.sqrt(mse)
    nrmse = torch.sqrt(mse / (x.square().mean() + 1e-06))
    vrmse = torch.sqrt(mse / (torch.var(x) + 1e-06))
    reduce_dims = (0, 2, 3)
    mse_c = se.mean(dim=reduce_dims)
    mae_c = err.abs().mean(dim=reduce_dims)
    rmse_c = torch.sqrt(mse_c)
    nrmse_c = torch.sqrt(mse_c / (x.square().mean(dim=reduce_dims) + 1e-06))
    vrmse_c = torch.sqrt(mse_c / (x.var(dim=reduce_dims) + 1e-06))
    global_metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "vrmse": vrmse,
    }
    channel_metrics = {
        "mse": mse_c,
        "mae": mae_c,
        "rmse": rmse_c,
        "nrmse": nrmse_c,
        "vrmse": vrmse_c,
    }
    return (global_metrics, channel_metrics)


def physical_units(x, sigma_train, scale_factor):
    return x * float(sigma_train) * float(scale_factor)


def spectral_metrics(x, y, spectral_bands=4):
    p_x, k = isotropic_power_spectrum(x, spatial=2)
    p_y, _ = isotropic_power_spectrum(y, spatial=2)
    c_xy, _ = isotropic_cross_correlation(x, y, spatial=2)
    se_p = (1 - (p_y + 1e-06) / (p_x + 1e-06)).square()
    se_c = (1 - (c_xy + 1e-06) / torch.sqrt(p_x * p_y + 1e-12)).square()
    global_metrics = {
        "fourier_power_rmse": torch.sqrt(se_p.mean()),
        "fourier_corr_rmse": torch.sqrt(se_c.mean()),
    }
    channel_metrics = {
        "fourier_power_rmse": torch.sqrt(se_p.mean(dim=(0, 2))),
        "fourier_corr_rmse": torch.sqrt(se_c.mean(dim=(0, 2))),
    }
    k0 = torch.clamp(k[0], min=1e-08)
    bins = torch.logspace(
        k0.log2(), -1.0, steps=spectral_bands, base=2.0, device=k.device
    )
    band_power = []
    band_corr = []
    for i in range(spectral_bands):
        if i < spectral_bands - 1:
            mask = torch.logical_and(bins[i] <= k, k <= bins[i + 1])
        else:
            mask = bins[i] <= k
        if mask.any():
            band_power.append(float(torch.sqrt(se_p[..., mask].mean()).item()))
            band_corr.append(float(torch.sqrt(se_c[..., mask].mean()).item()))
        else:
            band_power.append(None)
            band_corr.append(None)
    band_metrics = {"band_power_rmse": band_power, "band_corr_rmse": band_corr}
    return (global_metrics, channel_metrics, band_metrics)


def plot_reconstruction(
    x_sample,
    y_sample,
    out_file,
    vmin=VIS_VMIN,
    vmax=VIS_VMAX,
    cmap=VIS_CMAP,
    diff_cmap=VIS_DIFF_CMAP,
):
    import matplotlib.pyplot as plt

    x_np = x_sample.detach().cpu().numpy()
    y_np = y_sample.detach().cpu().numpy()
    d_np = y_np - x_np
    channels = x_np.shape[0]
    fig, axs = plt.subplots(
        nrows=3, ncols=channels, figsize=(3.2 * channels, 9.6), squeeze=False
    )
    for c in range(channels):
        axs[0, c].imshow(x_np[c], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
        axs[1, c].imshow(y_np[c], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
        axs[2, c].imshow(
            d_np[c], cmap=diff_cmap, vmin=vmin, vmax=vmax, interpolation="none"
        )
        axs[0, c].set_xticks([])
        axs[0, c].set_yticks([])
        axs[1, c].set_xticks([])
        axs[1, c].set_yticks([])
        axs[2, c].set_xticks([])
        axs[2, c].set_yticks([])
        axs[0, c].set_title(f"channel {c}")
    axs[0, 0].set_ylabel("GT")
    axs[1, 0].set_ylabel("Recon")
    axs[2, 0].set_ylabel("Diff")
    fig.tight_layout(pad=0.33)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_psd(x_sample, y_sample, out_file, cmap=VIS_CMAP):
    import matplotlib.pyplot as plt

    x_cpu = x_sample.detach().cpu()
    y_cpu = y_sample.detach().cpu()
    channels = x_cpu.shape[0]
    fig, axs = plt.subplots(
        nrows=1, ncols=channels, figsize=(3.6 * channels, 3.2), squeeze=False
    )
    axs = axs[0]
    cm = plt.get_cmap(cmap)
    gt_color = cm(0.1)
    recon_color = cm(0.8)
    eps = 1e-12
    for c in range(channels):
        p_x, k = isotropic_power_spectrum(x_cpu[c], spatial=2)
        p_y, _ = isotropic_power_spectrum(y_cpu[c], spatial=2)
        k_np = np.clip(k.numpy(), eps, None)
        p_x_np = np.clip(p_x.numpy(), eps, None)
        p_y_np = np.clip(p_y.numpy(), eps, None)
        axs[c].loglog(1.0 / k_np, p_x_np, base=2, color=gt_color, label="GT")
        axs[c].loglog(1.0 / k_np, p_y_np, base=2, color=recon_color, label="Recon")
        axs[c].invert_xaxis()
        axs[c].grid(True, which="both", linestyle=":")
        axs[c].set_title(f"channel {c}")
        axs[c].set_xlabel("wavelength (px)")
    axs[0].set_ylabel("power spectrum density")
    axs[0].legend(loc="best")
    fig.tight_layout(pad=0.33)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate(
    model,
    split,
    split_dir,
    batch_size,
    num_workers,
    device,
    use_normalize,
    mean,
    std,
    loss_types,
    loss_weights,
    max_batches,
    include_fourier,
    spectral_bands,
    evaluate_in_physical,
    sigma_train,
    scale_factor,
    save_vis,
    save_psd_vis,
    vis_outdir,
):
    dataset = SQGNpyDataset(str(split_dir), mode="iid", transform=None)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    channels = model.in_channels
    total_samples = 0
    total_batches = 0
    scalar_sums = {
        "objective_loss": 0.0,
        "mse": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "nrmse": 0.0,
        "vrmse": 0.0,
    }
    channel_sums = {
        "mse": torch.zeros(channels, dtype=torch.float64),
        "mae": torch.zeros(channels, dtype=torch.float64),
        "rmse": torch.zeros(channels, dtype=torch.float64),
        "nrmse": torch.zeros(channels, dtype=torch.float64),
        "vrmse": torch.zeros(channels, dtype=torch.float64),
    }
    if include_fourier:
        scalar_sums["fourier_power_rmse"] = 0.0
        scalar_sums["fourier_corr_rmse"] = 0.0
        channel_sums["fourier_power_rmse"] = torch.zeros(channels, dtype=torch.float64)
        channel_sums["fourier_corr_rmse"] = torch.zeros(channels, dtype=torch.float64)
        band_sums_power = [0.0 for _ in range(spectral_bands)]
        band_sums_corr = [0.0 for _ in range(spectral_bands)]
        band_counts = [0 for _ in range(spectral_bands)]
    saved_vis = 0
    saved_psd_vis = 0
    split_vis_dir = None
    split_psd_vis_dir = None
    if save_vis > 0 and vis_outdir is not None:
        split_vis_dir = vis_outdir / split / "spatial"
        split_vis_dir.mkdir(parents=True, exist_ok=True)
    if save_psd_vis > 0 and vis_outdir is not None:
        split_psd_vis_dir = vis_outdir / split / "psd"
        split_psd_vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"AE eval [{split}]", ncols=100, ascii=True)
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = batch.to(device)
            x_in = normalize(x, mean, std) if use_normalize else x
            y_in, z = model(x_in)
            y = denormalize(y_in, mean, std) if use_normalize else y_in
            if evaluate_in_physical:
                x_eval = physical_units(
                    x, sigma_train=sigma_train, scale_factor=scale_factor
                )
                y_eval = physical_units(
                    y, sigma_train=sigma_train, scale_factor=scale_factor
                )
            else:
                x_eval = x
                y_eval = y
            objective = torch.zeros((), device=device)
            for loss_name, w in zip(loss_types, loss_weights):
                objective = objective + float(w) * losses(x_in, y_in, z, loss_name)
            core_global, core_channel = field_metrics(x_eval, y_eval)
            batch_n = x.shape[0]
            need_spatial = split_vis_dir is not None and saved_vis < save_vis
            need_psd = split_psd_vis_dir is not None and saved_psd_vis < save_psd_vis
            if need_spatial or need_psd:
                start_idx = total_samples
                for i in range(batch_n):
                    need_spatial = split_vis_dir is not None and saved_vis < save_vis
                    need_psd = (
                        split_psd_vis_dir is not None and saved_psd_vis < save_psd_vis
                    )
                    if not (need_spatial or need_psd):
                        break
                    sample_idx = start_idx + i
                    if need_spatial:
                        out_file = (
                            split_vis_dir / f"{split}_sample_{sample_idx:06d}.png"
                        )
                        plot_reconstruction(x_eval[i], y_eval[i], out_file=out_file)
                        saved_vis += 1
                    if need_psd:
                        out_file = (
                            split_psd_vis_dir
                            / f"{split}_sample_{sample_idx:06d}_psd.png"
                        )
                        plot_psd(x_eval[i], y_eval[i], out_file=out_file)
                        saved_psd_vis += 1
            total_samples += batch_n
            total_batches += 1
            scalar_sums["objective_loss"] += float(objective.item()) * batch_n
            for k, v in core_global.items():
                scalar_sums[k] += float(v.item()) * batch_n
            for k, v in core_channel.items():
                channel_sums[k] += v.detach().cpu().to(torch.float64) * batch_n
            if include_fourier:
                fourier_global, fourier_channel, band_metrics = spectral_metrics(
                    x_eval, y_eval, spectral_bands=spectral_bands
                )
                for k, v in fourier_global.items():
                    scalar_sums[k] += float(v.item()) * batch_n
                for k, v in fourier_channel.items():
                    channel_sums[k] += v.detach().cpu().to(torch.float64) * batch_n
                for i in range(spectral_bands):
                    p_val = band_metrics["band_power_rmse"][i]
                    c_val = band_metrics["band_corr_rmse"][i]
                    if p_val is not None:
                        band_sums_power[i] += p_val * batch_n
                        band_counts[i] += batch_n
                    if c_val is not None:
                        band_sums_corr[i] += c_val * batch_n
    if total_samples == 0:
        raise RuntimeError(
            f"No samples evaluated for split '{split}'. Check dataset and max_batches."
        )
    results = {
        "split": split,
        "dir": str(split_dir),
        "num_samples": int(total_samples),
        "num_batches": int(total_batches),
        "num_visualizations": int(saved_vis),
        "num_psd_visualizations": int(saved_psd_vis),
        "metrics": {},
        "metrics_per_channel": {},
    }
    for k, v in scalar_sums.items():
        results["metrics"][k] = v / total_samples
    for k, v in channel_sums.items():
        results["metrics_per_channel"][k] = (v / total_samples).tolist()
    if include_fourier:
        band_power = []
        band_corr = []
        for i in range(spectral_bands):
            if band_counts[i] > 0:
                band_power.append(band_sums_power[i] / band_counts[i])
                band_corr.append(band_sums_corr[i] / band_counts[i])
            else:
                band_power.append(None)
                band_corr.append(None)
        results["metrics"]["band_power_rmse"] = band_power
        results["metrics"]["band_corr_rmse"] = band_corr
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SQG AE reconstruction quality"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to AE training config YAML"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="AE checkpoint path"
    )
    parser.add_argument(
        "--stats_path", type=str, default=None, help="Path to normalization stats YAML"
    )
    parser.add_argument(
        "--splits", type=str, default="train,valid,test", help="Comma-separated splits"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Evaluation batch size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Dataloader workers"
    )
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu")
    parser.add_argument(
        "--max_batches", type=int, default=None, help="Optional cap for quick eval"
    )
    parser.add_argument(
        "--no_fourier", action="store_true", help="Disable Fourier metrics"
    )
    parser.add_argument(
        "--spectral_bands", type=int, default=4, help="Number of log-frequency bands"
    )
    parser.add_argument(
        "--save_vis",
        type=int,
        default=0,
        help="Number of reconstruction comparison images to save per split (0 disables)",
    )
    parser.add_argument(
        "--save_psd_vis",
        type=int,
        default=0,
        help="Number of PSD comparison images to save per split (0 disables)",
    )
    parser.add_argument(
        "--vis_outdir",
        type=str,
        default=None,
        help="Directory to save visualization images (default: <checkpoint_dir>/ae_eval_vis)",
    )
    parser.add_argument(
        "--evaluate_in_physical",
        type=lambda s: str(s).lower() in ("1", "true", "yes", "y"),
        default=True,
        help="Whether to evaluate metrics on restored physical fields (default: true)",
    )
    parser.add_argument(
        "--sigma_train",
        type=float,
        default=2660.0,
        help="Sigma divisor used in data generation (default: 2660.0)",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=DEFAULT_SCALE_FACTOR,
        help="Physical scale factor used for restoration (default: f*theta0/g)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    train_cfg = config["training"]
    device = torch.device(args.device or config["device"])
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SQGAutoencoder(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    model.eval()
    use_normalize = data_cfg["normalize"]
    if use_normalize:
        mean, std = load_dataset_stats(Path(args.stats_path))
        mean = mean.to(device)
        std = std.to(device)
    else:
        mean = torch.zeros(model_cfg["in_channels"], 1, 1, device=device)
        std = torch.ones(model_cfg["in_channels"], 1, 1, device=device)
    batch_size = args.batch_size or train_cfg["valid_batch_size"]
    num_workers = (
        args.num_workers if args.num_workers is not None else train_cfg["num_workers"]
    )
    splits = splits(args.splits)
    if len(splits) == 0:
        raise ValueError("No valid splits parsed from --splits")
    all_results = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "normalize": bool(use_normalize),
        "evaluate_in_physical": bool(args.evaluate_in_physical),
        "sigma_train": float(args.sigma_train),
        "scale_factor": float(args.scale_factor),
        "save_vis": int(args.save_vis),
        "save_psd_vis": int(args.save_psd_vis),
        "vis_vmin": float(VIS_VMIN),
        "vis_vmax": float(VIS_VMAX),
        "vis_cmap": VIS_CMAP,
        "splits": {},
    }
    vis_outdir = None
    if args.save_vis > 0 or args.save_psd_vis > 0:
        vis_outdir = (
            Path(args.vis_outdir)
            if args.vis_outdir
            else checkpoint_path.parent / "ae_eval_vis"
        )
        vis_outdir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        data_dir = split_dir(data_cfg, split)
        split_result = evaluate(
            model=model,
            split=split,
            split_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            use_normalize=use_normalize,
            mean=mean,
            std=std,
            loss_types=loss_cfg["types"],
            loss_weights=loss_cfg["weights"],
            max_batches=args.max_batches,
            include_fourier=not args.no_fourier,
            spectral_bands=args.spectral_bands,
            evaluate_in_physical=bool(args.evaluate_in_physical),
            sigma_train=float(args.sigma_train),
            scale_factor=float(args.scale_factor),
            save_vis=int(args.save_vis),
            save_psd_vis=int(args.save_psd_vis),
            vis_outdir=vis_outdir,
        )
        all_results["splits"][split] = split_result
    if len(all_results["splits"]) == 0:
        raise RuntimeError("No split was evaluated. Check data paths and --splits.")
    output_path = (
        Path(args.output)
        if args.output
        else checkpoint_path.parent / "ae_eval_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("=" * 100)
    print(f"AE evaluation saved to: {output_path}")
    if vis_outdir is not None:
        print(f"AE visualization directory: {vis_outdir}")
    for split, result in all_results["splits"].items():
        m = result["metrics"]
        print(
            f"[{split}] N={result['num_samples']} obj={m['objective_loss']:.6f} rmse={m['rmse']:.6f} nrmse={m['nrmse']:.6f} vrmse={m['vrmse']:.6f} vis={result['num_visualizations']} psd_vis={result['num_psd_visualizations']}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
