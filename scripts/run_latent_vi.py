import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml

from sqg_inverse.inverse import Observation
from sqg_inverse.inverse.vi_solver import LatentModel, LatentVI
from sqg_inverse.models.iid_ldm import IIDLDM
from sqg_inverse.config import root
from sqg_inverse.training.ldm_eval import (
    load_ae_decoder,
    save_forecast_compare_vis,
    save_sample_field_vis,
)
from sqg_inverse.utils.fourier import (
    isotropic_cross_correlation,
    isotropic_power_spectrum,
)


def repo_path(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else root() / p


def pick_test(test_dir, test_index):
    files = sorted(test_dir.glob("*.npy"))
    return files[test_index]


def to_physical_units(x, *, test_space, sigma_train, scale_factor):
    space = str(test_space).strip().lower()
    if space == "physical":
        return x
    if space == "pv":
        return x * float(scale_factor)
    if space == "norm":
        return x * float(sigma_train) * float(scale_factor)
    raise ValueError(f"Unknown test_space: {test_space}")


def load_prior(train_config_path, checkpoint, device):
    with open(train_config_path, "r") as f:
        train_cfg = yaml.safe_load(f)
    model = IIDLDM(train_cfg["model"], train_cfg["rectified_flow"]).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["ema_model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model


def load_truth(
    test_file, frame_index, *, sigma_train, scale_factor, test_space, device
):
    arr = np.load(test_file, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(
            f"Expected test file shape (T,C,H,W), got {arr.shape} in {test_file}"
        )
    if frame_index < 0 or frame_index >= arr.shape[0]:
        raise IndexError(
            f"frame_index={frame_index} out of range [0, {arr.shape[0] - 1}]"
        )
    x = (
        torch.from_numpy(np.asarray(arr[frame_index], dtype=np.float32))
        .unsqueeze(0)
        .to(device)
    )
    return to_physical_units(
        x, test_space=test_space, sigma_train=sigma_train, scale_factor=scale_factor
    )


def metrics(truth_phys, recon_phys, y_obs, y_hat, mask, spectral_bands=4):
    err = recon_phys - truth_phys
    mse = float(err.square().mean().item())
    rmse = float(np.sqrt(mse))
    mask_f = mask.to(dtype=truth_phys.dtype)
    obs_field_err = ((err.square() * mask_f).sum() / mask_f.sum()).item()
    obs_rmse = float(np.sqrt(obs_field_err))
    if y_obs.ndim == 2 and y_hat.ndim == 2:
        obs_fit_mse = torch.mean((y_hat - y_obs).square()).item()
    else:
        obs_fit_mse = (((y_hat - y_obs).square() * mask_f).sum() / mask_f.sum()).item()
    obs_fit_rmse = float(np.sqrt(float(obs_fit_mse)))
    metrics = {
        "rmse_global": rmse,
        "mse_global": mse,
        "rmse_on_observed_field": obs_rmse,
        "rmse_in_observation_space": obs_fit_rmse,
        "observed_fraction_realized": float(mask_f.mean().item()),
    }
    if spectral_bands > 0:
        p_truth, k = isotropic_power_spectrum(truth_phys, spatial=2)
        p_recon, _ = isotropic_power_spectrum(recon_phys, spatial=2)
        c_tr, _ = isotropic_cross_correlation(truth_phys, recon_phys, spatial=2)
        se_p = (1.0 - (p_recon + 1e-06) / (p_truth + 1e-06)).square()
        se_c = (1.0 - (c_tr + 1e-06) / torch.sqrt(p_truth * p_recon + 1e-12)).square()
        log_psd_diff = (p_recon + 1e-12).log10() - (p_truth + 1e-12).log10()
        metrics.update(
            {
                "fourier_power_rmse": float(torch.sqrt(se_p.mean()).item()),
                "fourier_corr_rmse": float(torch.sqrt(se_c.mean()).item()),
                "fourier_power_rmse_per_channel": torch.sqrt(se_p.mean(dim=(0, 2)))
                .detach()
                .cpu()
                .tolist(),
                "fourier_corr_rmse_per_channel": torch.sqrt(se_c.mean(dim=(0, 2)))
                .detach()
                .cpu()
                .tolist(),
                "log_psd_rmse_global": float(
                    torch.sqrt(log_psd_diff.square().mean()).item()
                ),
                "log_psd_rmse_per_channel": torch.sqrt(
                    log_psd_diff.square().mean(dim=(0, 2))
                )
                .detach()
                .cpu()
                .tolist(),
                "spectral_bands": int(spectral_bands),
            }
        )
        channels = int(se_p.shape[1])
        k_min = torch.clamp(k[0], min=1e-08)
        k_max = torch.clamp(k[-1], min=float(k_min.item()))
        band_power_rmse_global = []
        band_corr_rmse_global = []
        band_power_rmse_per_channel = []
        band_corr_rmse_per_channel = []
        band_k_ranges = []
        if spectral_bands == 1:
            band_masks = [k >= k_min]
            band_edges = [(k_min, k_max)]
        else:
            bins = torch.logspace(
                k_min.log2(),
                k_max.log2(),
                steps=spectral_bands + 1,
                base=2.0,
                device=k.device,
            )
            band_masks = []
            band_edges = []
            for i in range(spectral_bands):
                left = bins[i]
                right = bins[i + 1]
                if i < spectral_bands - 1:
                    band_masks.append(torch.logical_and(left <= k, k < right))
                else:
                    band_masks.append(torch.logical_and(left <= k, k <= right))
                band_edges.append((left, right))
        for band_idx, band_mask in enumerate(band_masks):
            left, right = band_edges[band_idx]
            band_k_ranges.append([float(left.item()), float(right.item())])
            if band_mask.any():
                band_power_rmse_global.append(
                    float(torch.sqrt(se_p[..., band_mask].mean()).item())
                )
                band_corr_rmse_global.append(
                    float(torch.sqrt(se_c[..., band_mask].mean()).item())
                )
                band_power_c = []
                band_corr_c = []
                for c in range(channels):
                    band_power_c.append(
                        float(torch.sqrt(se_p[:, c, band_mask].mean()).item())
                    )
                    band_corr_c.append(
                        float(torch.sqrt(se_c[:, c, band_mask].mean()).item())
                    )
                band_power_rmse_per_channel.append(band_power_c)
                band_corr_rmse_per_channel.append(band_corr_c)
            else:
                band_power_rmse_global.append(None)
                band_corr_rmse_global.append(None)
                band_power_rmse_per_channel.append(None)
                band_corr_rmse_per_channel.append(None)
        metrics.update(
            {
                "band_k_ranges": band_k_ranges,
                "band_power_rmse_global": band_power_rmse_global,
                "band_corr_rmse_global": band_corr_rmse_global,
                "band_power_rmse_per_channel": band_power_rmse_per_channel,
                "band_corr_rmse_per_channel": band_corr_rmse_per_channel,
            }
        )
    return metrics


def _obs_count(y_obs, mask):
    if y_obs.ndim == 2:
        return float(y_obs.numel())
    return float(mask.to(dtype=torch.float32).sum().item())


def plot_losses(history, *, obs_count, out_file):
    import matplotlib.pyplot as plt

    xs = np.arange(len(history), dtype=np.int64)
    global_iter = np.array([h["global_iter"] for h in history])
    t = np.array([h["t"] for h in history])
    alpha_t = np.array([h["alpha_t"] for h in history])
    reg_loss = np.abs(np.array([h["reg_loss"] for h in history]))
    data_loss = np.array([h["data_loss"] for h in history])
    data_loss_per_obs = data_loss / obs_count
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    axs[0][0].plot(global_iter, data_loss_per_obs, lw=1.5, color="#1f77b4")
    axs[0][0].set_title("Data Loss per Observation")
    axs[0][0].set_xlabel("global_iter")
    axs[0][0].set_ylabel("loss")
    axs[0][0].grid(True, alpha=0.3)
    axs[0][1].plot(global_iter, reg_loss, lw=1.5, color="#d62728")
    axs[0][1].set_title("|Reg Loss|")
    axs[0][1].set_xlabel("global_iter")
    axs[0][1].set_ylabel("loss")
    axs[0][1].grid(True, alpha=0.3)
    axs[1][0].plot(global_iter, t, lw=1.5, color="#2ca02c")
    axs[1][0].set_title("t vs global_iter")
    axs[1][0].set_xlabel("global_iter")
    axs[1][0].set_ylabel("t")
    axs[1][0].grid(True, alpha=0.3)
    axs[1][1].plot(global_iter, alpha_t, lw=1.5, color="#9467bd")
    axs[1][1].set_title("alpha_t vs global_iter")
    axs[1][1].set_xlabel("global_iter")
    axs[1][1].set_ylabel("alpha_t")
    axs[1][1].grid(True, alpha=0.3)
    if len(history) > 1 and (not np.allclose(xs, global_iter)):
        axs[0][0].scatter(
            global_iter, data_loss_per_obs, s=8, alpha=0.4, color="#1f77b4"
        )
        axs[0][1].scatter(global_iter, reg_loss, s=8, alpha=0.4, color="#d62728")
        axs[1][0].scatter(global_iter, t, s=8, alpha=0.4, color="#2ca02c")
        axs[1][1].scatter(global_iter, alpha_t, s=8, alpha=0.4, color="#9467bd")
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def plot_inner_losses(history_inner, *, out_file):
    import matplotlib.pyplot as plt

    recs = history_inner
    seq_idx = np.arange(len(recs), dtype=np.int64)
    global_iter = np.array([r["global_iter"] for r in recs])
    inner_iter = np.array([r["inner_iter"] for r in recs])
    data_loss = np.array([r["data_loss"] for r in recs])
    obs_count = np.array([r["obs_count"] for r in recs])
    data_loss_per_obs = data_loss / obs_count
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    axs = axs[0]
    axs[0].plot(seq_idx, data_loss_per_obs, lw=1.3, color="#1f77b4")
    axs[0].set_title("Inner Data Loss per Observation")
    axs[0].set_xlabel("inner_global_iter")
    axs[0].set_ylabel("loss")
    axs[0].grid(True, alpha=0.3)
    valid = (
        np.isfinite(global_iter)
        & np.isfinite(inner_iter)
        & np.isfinite(data_loss_per_obs)
    )
    grouped = {}
    for gi, ii, li in zip(
        global_iter[valid], inner_iter[valid], data_loss_per_obs[valid]
    ):
        g = int(round(float(gi)))
        inner = int(round(float(ii)))
        grouped.setdefault(g, []).append((inner, float(li)))
    if len(grouped) == 0:
        axs[1].set_title("Inner Loss by Outer Iter")
        axs[1].text(0.5, 0.5, "no valid inner records", ha="center", va="center")
        axs[1].set_axis_off()
    else:
        max_show = 24
        keys = sorted(grouped.keys())
        show_keys = keys[:max_show]
        for g in show_keys:
            pts = sorted(grouped[g], key=lambda t: t[0])
            xs = np.array([p[0] for p in pts], dtype=np.float64)
            ys = np.array([p[1] for p in pts], dtype=np.float64)
            axs[1].plot(xs, ys, color="#7f7f7f", alpha=0.35, lw=1.0)
        max_inner = int(np.nanmax(inner_iter[valid]))
        mean_curve_x = []
        mean_curve_y = []
        for k in range(max_inner + 1):
            vals = [p[1] for g in show_keys for p in grouped[g] if p[0] == k]
            if len(vals) > 0:
                mean_curve_x.append(float(k))
                mean_curve_y.append(float(np.mean(vals)))
        if len(mean_curve_x) > 0:
            axs[1].plot(
                mean_curve_x,
                mean_curve_y,
                color="#d62728",
                lw=2.0,
                marker="o",
                ms=3,
                label="mean",
            )
            axs[1].legend(loc="best")
        axs[1].set_title("Inner Loss by Outer Iter")
        axs[1].set_xlabel("inner_iter")
        axs[1].set_ylabel("loss per observation")
        axs[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def plot_spectrum(truth_sample, recon_sample, out_file, *, cmap="jet"):
    import matplotlib.pyplot as plt

    x_cpu = truth_sample.detach().cpu()
    y_cpu = recon_sample.detach().cpu()
    channels = int(x_cpu.shape[0])
    eps = 1e-12
    fig, axs = plt.subplots(
        nrows=1, ncols=channels, figsize=(3.6 * channels, 3.2), squeeze=False
    )
    axs = axs[0]
    cm = plt.get_cmap(cmap)
    gt_color = cm(0.1)
    recon_color = cm(0.8)
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


def plot_observations(mask, out_file):
    import matplotlib.pyplot as plt

    m = mask[0].detach().cpu().numpy().astype(bool, copy=False)
    channels = int(m.shape[0])
    h = int(m.shape[1])
    w = int(m.shape[2])
    fig, axs = plt.subplots(
        1, channels, figsize=(3.2 * channels, 3.0), squeeze=False, facecolor="white"
    )
    axs = axs[0]
    for c in range(channels):
        canvas = np.ones((h, w), dtype=np.float32)
        canvas[m[c]] = 0.0
        axs[c].imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
        axs[c].set_xticks([])
        axs[c].set_yticks([])
        axs[c].set_title(f"obs points ch{c}")
    fig.tight_layout(pad=0.25)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"])
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    ldm_cfg = cfg["ldm"]
    ae_cfg = cfg["ae"]
    data_cfg = cfg["data"]
    obs_cfg = cfg["observation"]
    vi_cfg = cfg["vi"]
    out_cfg = cfg["output"]
    phys_cfg = cfg["physical"]
    sigma_train = phys_cfg["sigma_train"]
    scale_factor = phys_cfg["scale_factor"]
    train_config_path = repo_path(ldm_cfg["train_config"])
    ldm_checkpoint = repo_path(ldm_cfg["checkpoint"])
    ldm = load_prior(train_config_path, ldm_checkpoint, device)
    ae = load_ae_decoder(
        ae_config_path=repo_path(ae_cfg["config"]),
        ae_checkpoint_path=repo_path(ae_cfg["checkpoint"]),
        device=device,
    )
    wrapper = LatentModel(
        ldm=ldm,
        ae=ae,
        sigma_train=sigma_train,
        scale_factor=scale_factor,
    )
    test_file = pick_test(repo_path(data_cfg["test_dir"]), data_cfg["test_index"])
    frame_index = data_cfg["frame_index"]
    test_space = data_cfg["test_space"]
    truth_phys = load_truth(
        test_file=test_file,
        frame_index=frame_index,
        sigma_train=sigma_train,
        scale_factor=scale_factor,
        test_space=test_space,
        device=device,
    )
    operator_cfg = dict(obs_cfg["operator"])
    operator_cfg["fixed_mask"] = obs_cfg["fixed_mask"]
    operator_cfg["shared_channel_mask"] = obs_cfg["shared_channel_mask"]
    operator_cfg["mask_seed"] = obs_cfg["mask_seed"]
    operator = Observation.from_config(operator_cfg)
    y_obs, mask = operator.observe(
        truth_phys, noise=True, noise_seed=obs_cfg["noise_seed"]
    )
    solver = LatentVI(model=wrapper, forward_operator=operator, config=vi_cfg)
    result = solver.solve(y_obs, mask, seed + 2)
    scores = metrics(
        truth_phys=truth_phys,
        recon_phys=result.x_hat,
        y_obs=y_obs,
        y_hat=result.y_hat,
        mask=mask,
        spectral_bands=vi_cfg["spectral_bands"],
    )
    run_dir = repo_path(out_cfg["save_dir"]) / out_cfg["run_name"]
    run_dir.mkdir(parents=True)
    np.save(run_dir / "truth_physical.npy", truth_phys.detach().cpu().numpy())
    np.save(run_dir / "observation_y.npy", y_obs.detach().cpu().numpy())
    np.save(run_dir / "mask.npy", mask.detach().cpu().numpy().astype(np.uint8))
    np.save(run_dir / "recon_physical.npy", result.x_hat.detach().cpu().numpy())
    np.save(run_dir / "latent_mu.npy", result.latent_mu.detach().cpu().numpy())
    np.save(run_dir / "obs_fit_yhat.npy", result.y_hat.detach().cpu().numpy())
    obs_pkg = {
        "y": y_obs.detach().cpu().numpy(),
        "mask": mask.detach().cpu().numpy().astype(np.uint8),
        "truth_physical": truth_phys.detach().cpu().numpy(),
    }
    np.savez_compressed(run_dir / "observation_package.npz", **obs_pkg)
    with open(run_dir / "history_full.json", "w") as f:
        json.dump(result.history, f, indent=2)
    with open(run_dir / "history_inner_full.json", "w") as f:
        json.dump(result.history_inner, f, indent=2)
    plot_losses(
        history=result.history,
        obs_count=_obs_count(y_obs, mask),
        out_file=run_dir / "vi_optimization_curves.png",
    )
    plot_inner_losses(
        history_inner=result.history_inner,
        out_file=run_dir / "vi_inner_data_curves.png",
    )
    save_forecast_compare_vis(
        gt=truth_phys[0].detach().cpu(),
        pred=result.x_hat[0].detach().cpu(),
        out_file=run_dir / "truth_vs_recon.png",
        vmin=out_cfg["vis_vmin"],
        vmax=out_cfg["vis_vmax"],
        cmap=out_cfg["vis_cmap"],
    )
    if y_obs.ndim == 4:
        save_sample_field_vis(
            x=y_obs[0].detach().cpu(),
            out_file=run_dir / "observation_y.png",
            vmin=out_cfg["vis_vmin"],
            vmax=out_cfg["vis_vmax"],
            cmap=out_cfg["vis_cmap"],
        )
    plot_spectrum(
        truth_sample=truth_phys[0],
        recon_sample=result.x_hat[0],
        out_file=run_dir / "truth_vs_recon_psd.png",
        cmap=out_cfg["vis_cmap"],
    )
    plot_observations(mask=mask, out_file=run_dir / "observation_points.png")
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(scores, f, indent=2)
    with open(run_dir / "used_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(run_dir)
    print(scores)


if __name__ == "__main__":
    main()
