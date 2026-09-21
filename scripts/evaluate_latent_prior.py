import argparse
import json
from pathlib import Path
import random
import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F
import yaml

from sqg_inverse.data.sqg_dataset import LatentDataset
from sqg_inverse.models.iid_ldm import IIDLDM
from sqg_inverse.training.ldm_eval import (
    DEFAULT_SCALE_FACTOR,
    decode_latents,
    load_ae_decoder,
    restore_physical,
)
from sqg_inverse.utils.fourier import isotropic_power_spectrum
from sqg_inverse.utils.preprocess import load_dataset_stats, normalize


def pick_weights(checkpoint, model_variant):
    key = "ema_model_state_dict" if model_variant == "ema" else "model_state_dict"
    return checkpoint[key]


def plot_psd(out_file, k, psd_test, psd_gen):
    import matplotlib.pyplot as plt

    channels = psd_test.shape[0]
    fig, axs = plt.subplots(1, channels, figsize=(3.8 * channels, 3.2), squeeze=False)
    axs = axs[0]
    k_plot = np.clip(k, 1e-12, None)
    for c in range(channels):
        axs[c].loglog(
            1.0 / k_plot, np.clip(psd_test[c], 1e-12, None), base=2, label="test"
        )
        axs[c].loglog(
            1.0 / k_plot, np.clip(psd_gen[c], 1e-12, None), base=2, label="gen"
        )
        axs[c].invert_xaxis()
        axs[c].set_title(f"channel {c}")
        axs[c].set_xlabel("wavelength (px)")
        axs[c].grid(True, which="both", linestyle=":")
    axs[0].set_ylabel("power spectrum density")
    axs[0].legend(loc="best")
    fig.tight_layout(pad=0.33)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


class FieldMoments:

    def __init__(self, channels):
        self.channels = channels
        self.n = 0
        self.s1 = np.zeros(channels, dtype=np.float64)
        self.s2 = np.zeros(channels, dtype=np.float64)
        self.s3 = np.zeros(channels, dtype=np.float64)
        self.s4 = np.zeros(channels, dtype=np.float64)

    def update(self, x_phys):
        vals = (
            x_phys.detach()
            .cpu()
            .to(torch.float64)
            .permute(1, 0, 2, 3)
            .reshape(self.channels, -1)
            .numpy()
        )
        self.n += vals.shape[1]
        self.s1 += vals.sum(axis=1)
        self.s2 += np.square(vals).sum(axis=1)
        self.s3 += np.power(vals, 3).sum(axis=1)
        self.s4 += np.power(vals, 4).sum(axis=1)

    def finalize(self):
        if self.n <= 0:
            raise RuntimeError("No samples were accumulated for field moments.")
        m1 = self.s1 / self.n
        m2 = self.s2 / self.n
        m3 = self.s3 / self.n
        m4 = self.s4 / self.n
        c2 = np.clip(m2 - np.square(m1), 1e-18, None)
        c3 = m3 - 3.0 * m1 * m2 + 2.0 * np.power(m1, 3)
        c4 = m4 - 4.0 * m1 * m3 + 6.0 * np.square(m1) * m2 - 3.0 * np.power(m1, 4)
        std = np.sqrt(c2)
        skew = c3 / np.power(c2, 1.5)
        kurt = c4 / np.square(c2) - 3.0
        return {
            "mean": m1.tolist(),
            "std": std.tolist(),
            "skew": skew.tolist(),
            "kurtosis_excess": kurt.tolist(),
        }


class PsdAccumulator:

    def __init__(self):
        self.sum_p = None
        self.k = None
        self.count = 0

    def update(self, x_phys):
        p, k = isotropic_power_spectrum(x_phys, spatial=2)
        p = p.detach().cpu().to(torch.float64)
        if self.sum_p is None:
            self.sum_p = p.sum(dim=0)
            self.k = k.detach().cpu().to(torch.float64)
        else:
            self.sum_p += p.sum(dim=0)
        self.count += p.shape[0]

    def finalize(self):
        if self.sum_p is None or self.k is None or self.count <= 0:
            raise RuntimeError("No samples were accumulated for PSD.")
        mean_p = (self.sum_p / self.count).numpy()
        return (self.k.numpy(), mean_p)


def _ae_input(x_phys, *, phys_scale, ae_use_normalize, ae_mean, ae_std, device):
    x_model = x_phys / float(phys_scale)
    if ae_use_normalize:
        if ae_mean is None or ae_std is None:
            raise RuntimeError("AE normalization stats are required but missing.")
        x_model = normalize(x_model, ae_mean, ae_std)
    return x_model.to(device, non_blocking=True)


def ae_features(
    ae, x_phys, *, phys_scale, ae_use_normalize, ae_mean, ae_std, feature_type
):
    device = next(ae.parameters()).device
    x_in = _ae_input(
        x_phys,
        phys_scale=phys_scale,
        ae_use_normalize=ae_use_normalize,
        ae_mean=ae_mean,
        ae_std=ae_std,
        device=device,
    )
    with torch.no_grad():
        z = ae.encode(x_in)
        if feature_type == "gap":
            feat = F.adaptive_avg_pool2d(z, output_size=1).flatten(1)
        elif feature_type == "flatten":
            feat = z.flatten(1)
        else:
            raise ValueError(f"Unknown feature_type={feature_type}. Use gap/flatten.")
    return feat.detach().cpu().numpy().astype(np.float64, copy=False)


def fid(feat_test, feat_gen, eps=1e-06):
    if feat_test.ndim != 2 or feat_gen.ndim != 2:
        raise ValueError("FID expects 2D features [N, D].")
    if feat_test.shape[1] != feat_gen.shape[1]:
        raise ValueError(
            f"Feature dims mismatch: {feat_test.shape[1]} vs {feat_gen.shape[1]}"
        )
    mu1 = np.mean(feat_test, axis=0)
    mu2 = np.mean(feat_gen, axis=0)
    sigma1 = np.cov(feat_test, rowvar=False)
    sigma2 = np.cov(feat_gen, rowvar=False)
    if sigma1.ndim == 0:
        sigma1 = np.array([[float(sigma1)]], dtype=np.float64)
    if sigma2.ndim == 0:
        sigma2 = np.array([[float(sigma2)]], dtype=np.float64)
    cov_prod = sigma1 @ sigma2
    covmean = linalg.sqrtm(cov_prod)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0], dtype=np.float64) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return max(fid, 0.0)


def polynomial_mmd(x, y):
    m = x.shape[0]
    d = x.shape[1]
    k_xx = (x @ x.T / d + 1.0) ** 3
    k_yy = (y @ y.T / d + 1.0) ** 3
    k_xy = (x @ y.T / d + 1.0) ** 3
    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    sum_xy = k_xy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def kid(feat_test, feat_gen, *, subset_size=1000, subsets=100, seed=0):
    n_test, n_gen = (feat_test.shape[0], feat_gen.shape[0])
    m = min(subset_size, n_test, n_gen)
    if m < 2:
        raise ValueError("Not enough samples for KID.")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(subsets):
        idx_t = rng.choice(n_test, size=m, replace=False)
        idx_g = rng.choice(n_gen, size=m, replace=False)
        vals.append(polynomial_mmd(feat_test[idx_t], feat_gen[idx_g]))
    vals = np.array(vals, dtype=np.float64)
    return {
        "kid_mean": float(vals.mean()),
        "kid_std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        "kid_subset_size": int(m),
        "kid_subsets": int(subsets),
    }


def band_errors(psd_test, psd_gen, k, spectral_bands):
    se = np.square(1.0 - (psd_gen + 1e-06) / (psd_test + 1e-06))
    k0 = max(float(k[0]), 1e-08)
    bins = np.logspace(np.log2(k0), -1.0, num=spectral_bands, base=2.0)
    band_global = []
    band_per_channel = []
    for i in range(spectral_bands):
        if i < spectral_bands - 1:
            mask = np.logical_and(k >= bins[i], k <= bins[i + 1])
        else:
            mask = k >= bins[i]
        if not np.any(mask):
            band_global.append(None)
            band_per_channel.append([None] * psd_test.shape[0])
            continue
        se_band = se[:, mask]
        band_global.append(float(np.sqrt(np.mean(se_band))))
        band_per_channel.append(np.sqrt(np.mean(se_band, axis=1)).tolist())
    return {
        "band_power_rmse_global": band_global,
        "band_power_rmse_per_channel": band_per_channel,
    }


def slope_errors(psd_test, psd_gen, k, k_min, k_max):
    mask = np.logical_and(k >= k_min, k <= k_max)
    if np.sum(mask) < 2:
        raise ValueError(
            f"Insufficient k bins in [{k_min}, {k_max}] for slope fit. Available k range: [{float(k.min())}, {float(k.max())}]"
        )
    x = np.log(np.clip(k[mask], 1e-12, None))
    slopes_test = []
    slopes_gen = []
    slopes_diff = []
    for c in range(psd_test.shape[0]):
        y_test = np.log(np.clip(psd_test[c, mask], 1e-12, None))
        y_gen = np.log(np.clip(psd_gen[c, mask], 1e-12, None))
        s_test, _ = np.polyfit(x, y_test, deg=1)
        s_gen, _ = np.polyfit(x, y_gen, deg=1)
        slopes_test.append(float(s_test))
        slopes_gen.append(float(s_gen))
        slopes_diff.append(float(s_gen - s_test))
    return {
        "slope_k_range": [float(k_min), float(k_max)],
        "slope_test": slopes_test,
        "slope_gen": slopes_gen,
        "slope_diff": slopes_diff,
    }


def stat_errors(stats_test, stats_gen):
    out = {}
    for k in ["mean", "std", "skew", "kurtosis_excess"]:
        a = np.asarray(stats_test[k], dtype=np.float64)
        b = np.asarray(stats_gen[k], dtype=np.float64)
        out[f"{k}_delta_gen_minus_test"] = (b - a).tolist()
    return out


def test_samples(
    *,
    test_dir,
    num_trajectories,
    steps_per_trajectory,
    batch_size,
    test_is_physical,
    sigma_train,
    scale_factor,
    ae,
    ae_use_normalize,
    ae_mean,
    ae_std,
    feature_type,
):
    files = sorted(test_dir.glob("*.npy"))
    if len(files) < num_trajectories:
        raise ValueError(
            f"test_dir has only {len(files)} files, but num_trajectories={num_trajectories} is requested."
        )
    selected = files[:num_trajectories]
    features = []
    moments = None
    psd_acc = PsdAccumulator()
    phys_scale = float(sigma_train) * float(scale_factor)
    for fi, path in enumerate(selected):
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 4:
            raise ValueError(
                f"Expected test file shape (T,C,H,W), got {arr.shape} in {path}"
            )
        if arr.shape[0] < steps_per_trajectory:
            raise ValueError(
                f"{path} has T={arr.shape[0]} < steps_per_trajectory={steps_per_trajectory}"
            )
        if moments is None:
            moments = FieldMoments(channels=arr.shape[1])
        for start in range(0, steps_per_trajectory, batch_size):
            end = min(start + batch_size, steps_per_trajectory)
            x = torch.from_numpy(np.asarray(arr[start:end], dtype=np.float32))
            if test_is_physical:
                x_phys = x
            else:
                x_phys = restore_physical(
                    x, sigma_train=sigma_train, scale_factor=scale_factor
                )
            moments.update(x_phys)
            psd_acc.update(x_phys)
            f = ae_features(
                ae,
                x_phys,
                phys_scale=phys_scale,
                ae_use_normalize=ae_use_normalize,
                ae_mean=ae_mean,
                ae_std=ae_std,
                feature_type=feature_type,
            )
            features.append(f)
        print(f"[test] processed {fi + 1}/{num_trajectories}: {path.name}")
    assert moments is not None
    k, psd_mean = psd_acc.finalize()
    return {
        "files": [str(p) for p in selected],
        "features": np.concatenate(features, axis=0),
        "moments": moments.finalize(),
        "k": k,
        "psd_mean": psd_mean,
    }


def generated_samples(
    *,
    ldm_model,
    ae,
    out_gen_dir,
    num_trajectories,
    steps_per_trajectory,
    sample_steps,
    sample_time_warping,
    sample_warp_param,
    sample_batch_size,
    decode_batch_size,
    decode_amp,
    sigma_train,
    scale_factor,
    ae_use_normalize,
    ae_mean,
    ae_std,
    feature_type,
):
    out_gen_dir.mkdir(parents=True, exist_ok=True)
    features = []
    moments = None
    psd_acc = PsdAccumulator()
    phys_scale = float(sigma_train) * float(scale_factor)
    for traj_idx in range(num_trajectories):
        out_path = out_gen_dir / f"gen_traj{traj_idx:04d}.npy"
        chunk_arr = None
        written = 0
        while written < steps_per_trajectory:
            bsz = min(sample_batch_size, steps_per_trajectory - written)
            with torch.no_grad():
                z = ldm_model.sample(
                    batch_size=bsz,
                    steps=sample_steps,
                    time_warping=sample_time_warping,
                    warp_param=sample_warp_param,
                )
            x_model = decode_latents(
                ae=ae, z=z, batch_size=decode_batch_size, amp=decode_amp
            )
            x_phys = restore_physical(
                x_model, sigma_train=sigma_train, scale_factor=scale_factor
            )
            if chunk_arr is None:
                c, h, w = (x_phys.shape[1], x_phys.shape[2], x_phys.shape[3])
                chunk_arr = np.lib.format.open_memmap(
                    out_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(steps_per_trajectory, c, h, w),
                )
                moments = moments or FieldMoments(channels=c)
            x_np = x_phys.detach().cpu().numpy().astype(np.float32, copy=False)
            chunk_arr[written : written + bsz] = x_np
            written += bsz
            assert moments is not None
            moments.update(x_phys)
            psd_acc.update(x_phys)
            f = ae_features(
                ae,
                x_phys,
                phys_scale=phys_scale,
                ae_use_normalize=ae_use_normalize,
                ae_mean=ae_mean,
                ae_std=ae_std,
                feature_type=feature_type,
            )
            features.append(f)
        del chunk_arr
        print(f"[gen] saved {traj_idx + 1}/{num_trajectories}: {out_path.name}")
    assert moments is not None
    k, psd_mean = psd_acc.finalize()
    return {
        "files": [
            str(out_gen_dir / f"gen_traj{i:04d}.npy") for i in range(num_trajectories)
        ],
        "features": np.concatenate(features, axis=0),
        "moments": moments.finalize(),
        "k": k,
        "psd_mean": psd_mean,
    }


def ae_stats_path(ae_cfg, ae_ckpt, override):
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"AE stats file not found: {p}")
        return p
    c1 = ae_ckpt.parent / "data_stats.yaml"
    c2 = Path(ae_cfg["output"]["save_dir"]) / "data_stats.yaml"
    if c1.exists():
        return c1
    if c2.exists():
        return c2
    raise FileNotFoundError(
        "AE uses normalize=true but data_stats.yaml was not found near AE checkpoint/output. Pass --ae_stats_path explicitly."
    )


def main():
    parser = argparse.ArgumentParser(
        description="IID LDM distributional evaluation (AE-FID/KID + PSD + stats)"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="IID LDM training config YAML"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="IID LDM checkpoint path"
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="ema",
        choices=["raw", "ema"],
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory containing test trajectory npy files",
    )
    parser.add_argument(
        "--ae_config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ae_checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ae_stats_path",
        type=str,
        default=None,
        help="AE data_stats.yaml path (needed when AE normalize=true)",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=5,
        help="Number of trajectories to generate/evaluate",
    )
    parser.add_argument(
        "--steps_per_trajectory", type=int, default=1000, help="Frames per trajectory"
    )
    parser.add_argument(
        "--sample_steps", type=int, default=16, help="LDM ODE sample steps"
    )
    parser.add_argument(
        "--time_warping",
        type=str,
        default="linear",
        choices=["linear", "polynomial"],
        help="Sampling time grid warping.",
    )
    parser.add_argument(
        "--warp_param", type=float, default=0.5, help="Polynomial warping parameter s"
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=32,
        help="Batch size for latent sampling",
    )
    parser.add_argument(
        "--decode_batch_size", type=int, default=8, help="AE decode batch size"
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=64,
        help="Batch size when reading test npy files",
    )
    parser.add_argument(
        "--decode_amp",
        type=lambda s: str(s).lower() in ("1", "true", "yes", "y"),
        default=False,
    )
    parser.add_argument(
        "--test_is_physical",
        action="store_true",
        help="If set, test npy files are already physical fields",
    )
    parser.add_argument(
        "--sigma_train", type=float, default=None, help="Sigma for physical scaling"
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help="Scale factor for physical scaling",
    )
    parser.add_argument(
        "--feature_type", type=str, default="gap", choices=["gap", "flatten"]
    )
    parser.add_argument("--kid_subsets", type=int, default=100)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--spectral_bands", type=int, default=4)
    parser.add_argument("--slope_k_min", type=float, default=0.02)
    parser.add_argument("--slope_k_max", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--data_shape", type=int, nargs=3, default=None, metavar=("C", "H", "W")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    ae_config_path = args.ae_config
    ae_checkpoint_path = args.ae_checkpoint
    sigma_train = float(args.sigma_train if args.sigma_train is not None else 2660.0)
    scale_factor = float(
        args.scale_factor if args.scale_factor is not None else DEFAULT_SCALE_FACTOR
    )
    phys_scale = sigma_train * scale_factor
    ldm_device = torch.device(cfg["device"])
    ae_device = ldm_device
    ckpt_path = Path(args.checkpoint)
    sample_steps = args.sample_steps
    sample_time_warping = args.time_warping
    sample_warp_param = args.warp_param
    if sample_time_warping not in {"linear", "polynomial"}:
        raise ValueError(
            f"time_warping must be linear/polynomial, got {sample_time_warping}"
        )
    model = IIDLDM(cfg["model"], cfg["rectified_flow"]).to(ldm_device)
    ckpt_obj = torch.load(ckpt_path, map_location=ldm_device)
    state_dict = pick_weights(ckpt_obj, args.model_variant)
    state_tag = args.model_variant
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.requires_grad_(False)
    if args.data_shape is None:
        latent_data = LatentDataset(cfg["data"]["latent_valid_dir"], mode="iid")
        data_shape = tuple(latent_data[0].shape)
    else:
        data_shape = tuple(args.data_shape)
    model.rf.data_shape = data_shape
    ae = load_ae_decoder(
        ae_config_path=str(ae_config_path),
        ae_checkpoint_path=str(ae_checkpoint_path),
        device=ae_device,
    )
    with open(ae_config_path, "r") as f:
        ae_cfg = yaml.safe_load(f)
    ae_use_normalize = ae_cfg["data"]["normalize"]
    ae_mean = ae_std = None
    if ae_use_normalize:
        stats_path = ae_stats_path(ae_cfg, Path(ae_checkpoint_path), args.ae_stats_path)
        ae_mean, ae_std = load_dataset_stats(str(stats_path))
        ae_mean = ae_mean.to(ae_device)
        ae_std = ae_std.to(ae_device)
    else:
        stats_path = None
    out_root = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(cfg["output"]["save_dir"]) / "iid_metrics"
    )
    run_dir = (
        out_root
        / f"{ckpt_path.stem}_{state_tag}_traj{args.num_trajectories}_len{args.steps_per_trajectory}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = run_dir / "generated_physical_npy"
    print("=" * 90)
    print("IID LDM Distributional Evaluation")
    print("=" * 90)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Variant: {args.model_variant} -> loaded {state_tag}")
    print(f"LDM device: {ldm_device}")
    print(f"AE device: {ae_device}")
    print(f"Latent shape: {data_shape}")
    print(
        f"Trajectories x length: {args.num_trajectories} x {args.steps_per_trajectory}"
    )
    print(f"Sample steps: {sample_steps}")
    print(f"Time warping: {sample_time_warping} (s={sample_warp_param})")
    print(
        f"Physical scaling: sigma={sigma_train}, scale_factor={scale_factor}, sigma*scale={phys_scale}"
    )
    print(f"AE feature type: {args.feature_type}")
    print(f"AE normalize: {ae_use_normalize}")
    if stats_path is not None:
        print(f"AE stats: {stats_path}")
    print(f"Test dir: {args.test_dir}")
    print(f"Output: {run_dir}")
    print("=" * 90)
    gen = generated_samples(
        ldm_model=model,
        ae=ae,
        out_gen_dir=gen_dir,
        num_trajectories=args.num_trajectories,
        steps_per_trajectory=args.steps_per_trajectory,
        sample_steps=sample_steps,
        sample_time_warping=sample_time_warping,
        sample_warp_param=sample_warp_param,
        sample_batch_size=int(args.sample_batch_size),
        decode_batch_size=int(args.decode_batch_size),
        decode_amp=bool(args.decode_amp),
        sigma_train=sigma_train,
        scale_factor=scale_factor,
        ae_use_normalize=ae_use_normalize,
        ae_mean=ae_mean,
        ae_std=ae_std,
        feature_type=args.feature_type,
    )
    test = test_samples(
        test_dir=Path(args.test_dir),
        num_trajectories=args.num_trajectories,
        steps_per_trajectory=args.steps_per_trajectory,
        batch_size=int(args.test_batch_size),
        test_is_physical=bool(args.test_is_physical),
        sigma_train=sigma_train,
        scale_factor=scale_factor,
        ae=ae,
        ae_use_normalize=ae_use_normalize,
        ae_mean=ae_mean,
        ae_std=ae_std,
        feature_type=args.feature_type,
    )
    if not np.allclose(test["k"], gen["k"], atol=1e-12, rtol=1e-06):
        raise RuntimeError("PSD frequency bins mismatch between test and generated.")
    feat_test = test["features"]
    feat_gen = gen["features"]
    fid = fid(feat_test, feat_gen)
    kid = kid(
        feat_test,
        feat_gen,
        subset_size=int(args.kid_subset_size),
        subsets=int(args.kid_subsets),
        seed=int(args.seed),
    )
    k = test["k"]
    psd_test = test["psd_mean"]
    psd_gen = gen["psd_mean"]
    eps = 1e-12
    log_psd_rmse_ch = np.sqrt(
        np.mean(np.square(np.log(psd_gen + eps) - np.log(psd_test + eps)), axis=1)
    )
    band_errors = band_errors(
        psd_test, psd_gen, k, spectral_bands=int(args.spectral_bands)
    )
    slope_errors = slope_errors(
        psd_test,
        psd_gen,
        k,
        k_min=float(args.slope_k_min),
        k_max=float(args.slope_k_max),
    )
    stat_deltas = stat_errors(test["moments"], gen["moments"])
    plot_psd(run_dir / "psd_mean_compare.png", k, psd_test, psd_gen)
    np.savez(run_dir / "psd_mean_curves.npz", k=k, psd_test=psd_test, psd_gen=psd_gen)
    results = {
        "config": str(args.config),
        "checkpoint": str(ckpt_path),
        "state_tag": state_tag,
        "model_variant_requested": args.model_variant,
        "ldm_device": str(ldm_device),
        "ae_device": str(ae_device),
        "seed": int(args.seed),
        "num_trajectories": int(args.num_trajectories),
        "steps_per_trajectory": int(args.steps_per_trajectory),
        "num_samples": int(args.num_trajectories * args.steps_per_trajectory),
        "sample_steps": int(sample_steps),
        "time_warping": sample_time_warping,
        "warp_param": float(sample_warp_param),
        "sample_batch_size": int(args.sample_batch_size),
        "decode_batch_size": int(args.decode_batch_size),
        "decode_amp": bool(args.decode_amp),
        "test_batch_size": int(args.test_batch_size),
        "sigma_train": float(sigma_train),
        "scale_factor": float(scale_factor),
        "phys_scale": float(phys_scale),
        "ae_feature_type": args.feature_type,
        "ae_use_normalize": bool(ae_use_normalize),
        "test_is_physical": bool(args.test_is_physical),
        "generated_files": gen["files"],
        "test_files": test["files"],
        "fid": float(fid),
        **kid,
        "log_psd_rmse_per_channel": log_psd_rmse_ch.tolist(),
        "log_psd_rmse_global": float(np.mean(log_psd_rmse_ch)),
        "band_errors": band_errors,
        "slope_errors": slope_errors,
        "basic_stats_test": test["moments"],
        "basic_stats_gen": gen["moments"],
        "basic_stats_delta": stat_deltas,
    }
    out_json = run_dir / "iid_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 90)
    print(f"Saved metrics: {out_json}")
    print(f"FID: {results['fid']:.6f}")
    print(f"KID mean/std: {results['kid_mean']:.6e} / {results['kid_std']:.6e}")
    print(f"log-PSD RMSE (global): {results['log_psd_rmse_global']:.6f}")
    print(f"Generated npy dir: {gen_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
