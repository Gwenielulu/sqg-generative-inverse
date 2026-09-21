import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from sqg_inverse.data.shards import infinite_loader
from sqg_inverse.models.pixel_ddpm.nn import update_ema
from sqg_inverse.models.pixel_ddpm.script_util import create_model_and_diffusion


def save(path, ema_model):
    torch.save(ema_model.state_dict(), path)


def train(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(config["device"])
    model_cfg = config["model"]
    train_cfg = config["training"]
    out = Path(config["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)

    model, diffusion = create_model_and_diffusion(model_cfg)
    model = model.to(device)
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    data = infinite_loader(
        config["data"]["train_dir"],
        train_cfg["batch_size"],
        model_cfg["in_channels"],
        model_cfg["image_size"],
        config["data"]["num_workers"],
    )

    for step in range(1, train_cfg["steps"] + 1):
        lr = train_cfg["learning_rate"] * min(1.0, step / train_cfg["warmup_steps"])
        optimizer.param_groups[0]["lr"] = lr

        batch, _ = next(data)
        batch = batch.to(device, non_blocking=True)
        timesteps = torch.randint(
            diffusion.num_timesteps, (batch.shape[0],), device=device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.training_losses(model, batch, timesteps, model_kwargs={})[
            "loss"
        ].mean()
        loss.backward()
        optimizer.step()
        update_ema(ema_model.parameters(), model.parameters(), train_cfg["ema_rate"])

        if step % train_cfg["log_every"] == 0:
            print(f"step {step:06d} loss {loss.item():.6f} lr {lr:.2e}")
        if step % train_cfg["save_every"] == 0:
            save(out / f"pixel_ddpm_{step:06d}.pt", ema_model)

    save(out / "pixel_ddpm_final.pt", ema_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
