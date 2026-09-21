import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import cosine_similarity


class AutoEncoderLoss(nn.Module):

    def __init__(self, losses=["mse"], weights=[1.0]):
        super().__init__()
        assert len(losses) == len(weights), "losses and weights must have same length"
        self.losses = list(losses)
        self.register_buffer("weights", torch.as_tensor(weights))

    def forward(self, autoencoder, x, **kwargs):
        y, z = autoencoder(x, **kwargs)
        values = []
        for loss in self.losses:
            if loss == "mse":
                l = (x - y).square().mean()
            elif loss == "mae":
                l = (x - y).abs().mean()
            elif loss == "vmse":
                x_flat = rearrange(x, "B C ... -> B C (...)")
                y_flat = rearrange(y, "B C ... -> B C (...)")
                l = (x_flat - y_flat).square().mean(dim=2) / (x_flat.var(dim=2) + 0.01)
                l = l.mean()
            elif loss == "vrmse":
                x_flat = rearrange(x, "B C ... -> B C (...)")
                y_flat = rearrange(y, "B C ... -> B C (...)")
                l = (x_flat - y_flat).square().mean(dim=2) / (x_flat.var(dim=2) + 0.01)
                l = torch.sqrt(l).mean()
            elif loss == "similarity":
                f = rearrange(z, "B ... -> B (...)")
                l = cosine_similarity(f[None, :], f[:, None], dim=-1)
                tri = torch.triu_indices(*l.shape, offset=1, device=l.device)
                l = l[tuple(tri)]
                l = l.mean()
            else:
                raise ValueError(f"Unknown loss '{loss}'")
            values.append(l)
        values = torch.stack(values)
        return torch.vdot(self.weights, values)
