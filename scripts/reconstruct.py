import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from sqg_inverse.config import merge, read, root
from sqg_inverse.evaluation import score
from sqg_inverse.inverse import Observation, load_ddpm, sample_ddpm
from sqg_inverse.inverse.doc_solver import run_doc


def read_config(path):
    cfg = read(path)
    if "defaults" not in cfg:
        return cfg
    base = {}
    for key, filename in cfg.pop("defaults").items():
        filename = Path(filename)
        if not filename.is_absolute():
            filename = root() / filename
        base[key] = read(filename)
    return merge(base, cfg)


def full_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else root() / path


def read_truth(cfg, device):
    data = np.load(full_path(cfg["path"]), mmap_mode="r")
    data = data[cfg["sample_index"]]
    return torch.from_numpy(np.array(data, dtype=np.float32)).unsqueeze(0).to(device)


def run_pixel(cfg):
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg["device"])
    truth = read_truth(cfg["data"], device)
    obs_cfg = dict(cfg["observation"])
    noise_seed = obs_cfg.pop("noise_seed")
    obs = Observation.from_config(obs_cfg)
    y, mask = obs.observe(truth, noise=True, noise_seed=noise_seed)
    model, diffusion = load_ddpm(
        cfg["model"], full_path(cfg["checkpoint"]["path"]), device
    )
    inv = dict(cfg["inverse"])
    method = inv.pop("method").lower()
    if method in {"doc", "diffusion_optimal_control"}:
        pred, steps = run_doc(
            model,
            diffusion,
            y,
            obs,
            mask,
            tuple(truth.shape[1:]),
            device,
            {**inv, "seed": seed},
        )
        history = {"terminal_iterations": len(steps)}
    else:
        pred, loss = sample_ddpm(
            model, diffusion, method, y, obs, mask, tuple(truth.shape), device, inv
        )
        history = {"measurement_residual": loss}
    out = full_path(cfg["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "truth.npy", truth.cpu().numpy())
    np.save(out / "measurement.npy", y.cpu().numpy())
    np.save(out / "mask.npy", mask.cpu().numpy().astype(np.uint8))
    np.save(out / "reconstruction.npy", pred.cpu().numpy())
    result = score(truth, pred, mask)
    with (out / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2)
    with (out / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    with (out / "used_config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(json.dumps(result, indent=2))
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = read_config(args.config.resolve())
    if cfg["inverse"]["method"] == "latent_vi":
        script = Path(__file__).with_name("run_latent_vi.py")
        command = [sys.executable, str(script), "--config", str(args.config.resolve())]
        subprocess.call(command, cwd=root())
        return
    run_pixel(cfg)


if __name__ == "__main__":
    main()
