import math
import torch


class Observation:

    def __init__(
        self,
        name="identity",
        gain=1.0,
        quadratic_scale=7.0,
        noise_std=1.0,
        sparsity=0.25,
        fixed_mask=True,
        shared_channel_mask=True,
        mask_seed=42,
        compact_output=False,
    ):
        self.name = name.lower()
        self.gain = gain
        self.quadratic_scale = quadratic_scale
        self.noise_std = noise_std
        self.sparsity = sparsity
        self.fixed_mask = fixed_mask
        self.shared_channel_mask = shared_channel_mask
        self.mask_seed = mask_seed
        self.compact_output = compact_output
        self._mask = None
        self._shape = None
        self._count = 0

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)

    def make_mask(self, x):
        b, c, h, w = x.shape
        shape = (c, h, w)
        seed = self.mask_seed if self.fixed_mask else self.mask_seed + self._count
        if not self.fixed_mask or self._mask is None or self._shape != shape:
            g = torch.Generator(device="cpu").manual_seed(seed)
            if self.shared_channel_mask:
                mask = torch.rand((h, w), generator=g) < self.sparsity
                mask = mask[None, None].expand(1, c, h, w).clone()
            else:
                mask = torch.rand((1, c, h, w), generator=g) < self.sparsity
            if self.fixed_mask:
                self._mask = mask
                self._shape = shape
            else:
                self._count += 1
        else:
            mask = self._mask
        return mask.to(x.device).expand(b, c, h, w)

    def h(self, x):
        if self.name == "identity":
            return self.gain * x
        if self.name == "quadratic":
            return (x / self.quadratic_scale).square()
        return torch.atan(x)

    def h_inv(self, y, reference=None):
        if self.name == "identity":
            return y / self.gain
        if self.name == "quadratic":
            x = y.clamp_min(0).sqrt() * self.quadratic_scale
            if reference is not None:
                sign = torch.where(reference < 0, -1.0, 1.0)
                x = sign * x
            return x
        limit = math.pi / 2 - 0.0001
        return torch.tan(y.clamp(-limit, limit))

    def forward(self, x, mask=None, noise=False, noise_seed=None):
        mask = self.make_mask(x) if mask is None else mask.bool().to(x.device)
        y = self.h(x)
        if noise and self.noise_std > 0:
            if noise_seed is None:
                eps = torch.randn_like(y)
            else:
                g = torch.Generator(device="cpu").manual_seed(noise_seed)
                eps = torch.randn(y.shape, generator=g, dtype=y.dtype).to(y.device)
            y = y + self.noise_std * eps
        y = y * mask
        if self.compact_output:
            flat = y.reshape(y.shape[0], -1)
            flat_mask = mask.reshape(mask.shape[0], -1)
            y = torch.stack([flat[i, flat_mask[i]] for i in range(y.shape[0])])
        return y

    def observe(self, x, mask=None, noise=True, noise_seed=None):
        mask = self.make_mask(x) if mask is None else mask.bool().to(x.device)
        return (self.forward(x, mask=mask, noise=noise, noise_seed=noise_seed), mask)

    def pseudo_inv(self, y, mask=None, reference=None):
        if y.ndim == 2:
            b, c, h, w = mask.shape
            full = torch.zeros((b, c * h * w), dtype=y.dtype, device=y.device)
            flat_mask = mask.reshape(b, -1)
            for i in range(b):
                full[i, flat_mask[i]] = y[i]
            y = full.view(b, c, h, w)
        mask = self.make_mask(y) if mask is None else mask.bool().to(y.device)
        x = self.h_inv(y * mask, reference)
        return x * mask

    def masked_mse(self, pred, target, mask=None, reduction="mean"):
        if pred.ndim == 2:
            loss = (pred - target).square()
            return loss.sum() if reduction == "sum" else loss.mean()
        mask = self.make_mask(pred) if mask is None else mask.bool().to(pred.device)
        loss = (pred - target).square() * mask
        if reduction == "sum":
            return loss.sum()
        return loss.sum() / mask.sum().clamp_min(1)
