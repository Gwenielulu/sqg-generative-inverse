from collections.abc import Hashable

__all__ = ["ViTBlock", "ViT"]
import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint
from .attention import MultiheadSelfAttention
from .embedding import SineEncoding
from .layers import Patchify, Unpatchify


class ViTBlock(nn.Module):

    def __init__(
        self,
        channels,
        mod_features=0,
        ffn_factor=4,
        spatial=2,
        rope=True,
        dropout=None,
        checkpointing=False,
        **kwargs
    ):
        super().__init__()
        self.checkpointing = checkpointing
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        if mod_features > 0:
            self.ada_zero = nn.Sequential(
                nn.Linear(mod_features, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, 4 * channels),
                Rearrange("... (n C) -> n ... 1 C", n=4),
            )
            self.ada_zero[-2].weight.data.mul_(0.01)
        else:
            self.ada_zero = nn.Parameter(torch.randn(4, channels))
            self.ada_zero.data.mul_(0.01)
        self.msa = MultiheadSelfAttention(channels, **kwargs)
        if rope:
            amplitude = 100.0 ** (-torch.rand(channels // 2))
            direction = torch.nn.functional.normalize(
                torch.randn(spatial, channels // 2), dim=0
            )
            self.theta = nn.Parameter(amplitude * direction)
        else:
            self.theta = None
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_factor * channels),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            nn.Linear(ffn_factor * channels, channels),
        )

    def _forward(self, x, mod=None, coo=None, mask=None, skip=None):
        if self.theta is None:
            theta = None
        else:
            theta = torch.einsum("...ij,jk", coo, self.theta)
        if torch.is_tensor(self.ada_zero):
            a, b, c, d = self.ada_zero
        else:
            a, b, c, d = self.ada_zero(mod)
        y = (a + 1) * self.norm(x) + b
        y = y + self.msa(y, theta, mask)
        y = self.ffn(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)
        if skip is not None:
            y = (y + d * skip) * torch.rsqrt(1 + d * d)
        return y

    def forward(self, x, mod=None, coo=None, mask=None, skip=None):
        if self.checkpointing:
            return checkpoint(
                self._forward, x, mod, coo, mask, skip, use_reentrant=False
            )
        else:
            return self._forward(x, mod, coo, mask, skip)


class ViT(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        cond_channels=0,
        mod_features=0,
        hid_channels=1024,
        hid_blocks=3,
        spatial=2,
        patch_size=1,
        unpatch_size=None,
        window_size=None,
        **kwargs
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial
        if unpatch_size is None:
            unpatch_size = patch_size
        elif isinstance(unpatch_size, int):
            unpatch_size = [unpatch_size] * spatial
        self.patch = Patchify(patch_size, channel_last=True)
        self.unpatch = Unpatchify(unpatch_size, channel_last=True)
        self.in_proj = nn.Linear(
            math.prod(patch_size) * (in_channels + cond_channels), hid_channels
        )
        self.out_proj = nn.Linear(hid_channels, math.prod(patch_size) * out_channels)
        self.positional_embedding = nn.Sequential(
            SineEncoding(hid_channels),
            Rearrange("... N C -> ... (N C)"),
            nn.Linear(spatial * hid_channels, hid_channels),
        )
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    channels=hid_channels,
                    mod_features=mod_features,
                    spatial=spatial,
                    **kwargs
                )
                for _ in range(hid_blocks)
            ]
        )
        self.spatial = spatial
        if window_size is None:
            self.window_size = None
        elif isinstance(window_size, int):
            self.window_size = (window_size,) * spatial
        else:
            self.window_size = tuple(window_size)

    @staticmethod
    def coo_and_mask(shape, spatial, window_size, dtype, device):
        assert isinstance(shape, Hashable)
        assert isinstance(window_size, Hashable)
        coo = (torch.arange(size, device=device) for size in shape)
        coo = torch.cartesian_prod(*coo)
        coo = torch.reshape(coo, shape=(-1, spatial))
        if window_size is None:
            mask = None
        else:
            delta = torch.abs(coo[:, None] - coo[None, :])
            delta = torch.minimum(delta, delta.new_tensor(shape) - delta)
            mask = torch.all(delta <= coo.new_tensor(window_size) // 2, dim=-1)
        return (coo.to(dtype=dtype), mask)

    def forward(self, x, mod=None, cond=None):
        if cond is not None:
            x = torch.cat((x, cond), dim=1)
        x = self.patch(x)
        x = self.in_proj(x)
        shape = x.shape[-self.spatial - 1 : -1]
        coo, mask = self.coo_and_mask(
            shape,
            spatial=self.spatial,
            window_size=self.window_size,
            dtype=x.dtype,
            device=x.device,
        )
        x = skip = torch.flatten(x, -self.spatial - 1, -2)
        x = x + self.positional_embedding(coo)
        for block in self.blocks:
            x = block(x, mod, coo=coo, mask=mask, skip=skip)
        x = torch.unflatten(x, sizes=shape, dim=-2)
        x = self.out_proj(x)
        x = self.unpatch(x)
        return x
