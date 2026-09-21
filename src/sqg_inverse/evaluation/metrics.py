import torch


def score(truth, pred, mask=None):
    error = pred - truth
    out = {
        "rmse": torch.sqrt(error.square().mean()).item(),
        "mae": error.abs().mean().item(),
    }
    a = truth - truth.mean()
    b = pred - pred.mean()
    out["correlation"] = (
        (a * b).sum() / torch.sqrt(a.square().sum() * b.square().sum()).clamp_min(1e-12)
    ).item()
    if mask is not None:
        mask = mask.bool()
        if mask.any():
            out["rmse_observed"] = torch.sqrt(error[mask].square().mean()).item()
        if (~mask).any():
            out["rmse_unobserved"] = torch.sqrt(error[~mask].square().mean()).item()
    return out
