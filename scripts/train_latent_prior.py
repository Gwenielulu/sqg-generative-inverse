import argparse
import copy
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from sqg_inverse.data.sqg_dataset import LatentDataset
from sqg_inverse.models.iid_ldm import IIDLDM


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.lerp_(param, 1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def lr_factor(step, warmup, total, start_factor, min_factor):
    if step < warmup:
        return start_factor + (1.0 - start_factor) * step / warmup
    progress = (step - warmup) / (total - warmup)
    return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))


def save(path, model, ema_model, optimizer, scheduler, step, config):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )


def train(config):
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["device"])
    data_cfg = config["data"]
    train_cfg = config["training"]
    optim_cfg = config["optim"]
    ema_cfg = config["ema"]
    out = Path(config["output"]["save_dir"])
    out.mkdir(parents=True, exist_ok=True)

    data = LatentDataset(data_cfg["latent_dir"], mode="iid")
    loader = DataLoader(
        data,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        persistent_workers=train_cfg["persistent_workers"],
        prefetch_factor=train_cfg["prefetch_factor"],
    )

    model = IIDLDM(config["model"], config["rectified_flow"]).to(device)
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=tuple(optim_cfg["betas"]),
        eps=optim_cfg["eps"],
        weight_decay=optim_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_factor(
            step,
            optim_cfg["warmup_steps"],
            train_cfg["steps"],
            optim_cfg["warmup_start_factor"],
            optim_cfg["min_lr"] / train_cfg["lr"],
        ),
    )

    amp_dtype = {
        "fp32": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[train_cfg["precision"]]
    step = 0
    epoch = 0
    ema_updates = 0
    while step < train_cfg["steps"]:
        epoch += 1
        model.train()
        progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for z in progress:
            z = z.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                loss = model(z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            step += 1

            if step >= ema_cfg["update_after_step"]:
                if (step - ema_cfg["update_after_step"]) % ema_cfg["update_every"] == 0:
                    if ema_updates == 0:
                        ema_model.load_state_dict(model.state_dict())
                    else:
                        update_ema(ema_model, model, ema_cfg["decay"])
                    ema_updates += 1

            progress.set_postfix(step=step, loss=f"{loss.item():.5f}")
            if step % train_cfg["save_every"] == 0:
                save(
                    out / f"checkpoint_{step}.pt",
                    model,
                    ema_model,
                    optimizer,
                    scheduler,
                    step,
                    config,
                )
            if step == train_cfg["steps"]:
                break

    save(
        out / "latent_prior_final.pt",
        model,
        ema_model,
        optimizer,
        scheduler,
        step,
        config,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
