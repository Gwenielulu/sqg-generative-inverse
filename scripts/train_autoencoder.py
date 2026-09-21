import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from sqg_inverse.data.augmentation import get_augmentation
from sqg_inverse.data.sqg_dataset import SQGNpyDataset
from sqg_inverse.models.autoencoder import SQGAutoencoder
from sqg_inverse.training.ae_loss import AutoEncoderLoss


def loader(path, config, transform, shuffle, batch_size):
    data = SQGNpyDataset(path, mode="iid", transform=transform)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config["persistent_workers"],
        prefetch_factor=config["prefetch_factor"],
    )


def save(path, model, optimizer, epoch, valid_loss, config):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "valid_loss": valid_loss,
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
    out = Path(config["output"]["save_dir"])
    out.mkdir(parents=True, exist_ok=True)

    transform = get_augmentation(data_cfg["augment"])
    train_loader = loader(
        data_cfg["train_dir"],
        train_cfg,
        transform,
        True,
        train_cfg["batch_size"],
    )
    valid_loader = loader(
        data_cfg["valid_dir"],
        train_cfg,
        None,
        False,
        train_cfg["valid_batch_size"],
    )

    model = SQGAutoencoder(**config["model"]).to(device)
    loss_fn = AutoEncoderLoss(**config["loss"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=tuple(optim_cfg["betas"]),
        weight_decay=optim_cfg["weight_decay"],
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=optim_cfg["warmup_start_factor"],
        total_iters=optim_cfg["warmup_epochs"],
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"] - optim_cfg["warmup_epochs"],
        eta_min=optim_cfg["min_lr"],
    )

    best = float("inf")
    no_improvement = 0
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for x in tqdm(train_loader, desc=f"train {epoch}", leave=False):
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x in valid_loader:
                x = x.to(device, non_blocking=True)
                valid_loss += loss_fn(model, x).item()
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        if epoch <= optim_cfg["warmup_epochs"]:
            warmup.step()
        else:
            cosine.step()

        print(
            f"epoch {epoch:04d} train {train_loss:.6f} valid {valid_loss:.6f} "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if valid_loss < best - train_cfg["early_stopping_min_delta"]:
            best = valid_loss
            no_improvement = 0
            save(
                out / "autoencoder_best.pt", model, optimizer, epoch, valid_loss, config
            )
        else:
            no_improvement += 1

        if epoch % config["output"]["save_every_epoch"] == 0:
            save(
                out / f"autoencoder_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                valid_loss,
                config,
            )

        patience = train_cfg["early_stopping_patience"]
        if patience > 0 and no_improvement >= patience:
            break

    save(out / "autoencoder_final.pt", model, optimizer, epoch, valid_loss, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
