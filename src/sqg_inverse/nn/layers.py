__all__ = [
    "ConvNd",
    "LayerNorm",
    "RMSNorm",
    "SelfAttentionNd",
    "Patchify",
    "Unpatchify",
    "Shrink",
    "Dilate",
]
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint


def ConvNd(in_channels, out_channels, spatial=2, identity_init=False, **kwargs):
    CONVS = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    if spatial in CONVS:
        Conv = CONVS[spatial]
    else:
        raise NotImplementedError()
    conv = Conv(in_channels, out_channels, **kwargs)
    if identity_init:
        kernel_size = conv.weight.shape[2:]
        kernel_center = [k // 2 for k in kernel_size]
        eye = torch.zeros_like(conv.weight.data)
        for i in range(out_channels):
            eye[(i, i % in_channels, *kernel_center)] = 1
        conv.weight.data.mul_(0.01)
        conv.weight.data.add_(eye)
    return conv


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.dim = dim if isinstance(dim, int) else tuple(dim)
        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self):
        return f"dim={self.dim}"

    def forward(self, x):
        mean = torch.mean(x, dim=self.dim, keepdim=True)
        variance = torch.mean((x - mean) ** 2, dim=self.dim, keepdim=True)
        dims = (self.dim,) if isinstance(self.dim, int) else tuple(self.dim)
        n = 1
        for d in dims:
            n *= x.shape[d]
        if n > 1:
            variance = variance * (n / (n - 1))
        return (x - mean) * torch.rsqrt(variance + self.eps)


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.dim = dim if isinstance(dim, int) else tuple(dim)
        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self):
        return f"dim={self.dim}"

    def forward(self, x):
        rms = torch.rsqrt(
            torch.mean(torch.square(x), dim=self.dim, keepdim=True) + self.eps
        )
        return x * rms


class SelfAttentionNd(nn.MultiheadAttention):

    def __init__(self, channels, heads=1, checkpointing=False, **kwargs):
        super().__init__(
            embed_dim=channels, num_heads=heads, batch_first=True, **kwargs
        )
        self.checkpointing = checkpointing

    def _forward(self, x):
        y = rearrange(x, "B C ...  -> B (...) C")
        qkv = torch.nn.functional.linear(y, self.in_proj_weight, self.in_proj_bias)
        q, k, v = rearrange(qkv, "B L (n H C) -> n B H L C", n=3, H=self.num_heads)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        y = rearrange(y, "B H L C -> B L (H C)")
        y = torch.nn.functional.linear(y, self.out_proj.weight, self.out_proj.bias)
        y = rearrange(y, "B L C -> B C L").reshape(x.shape)
        return y

    def forward(self, x):
        if self.checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


def Patchify(patch_size, channel_last=False):
    if len(patch_size) == 1:
        (l,) = patch_size
        if channel_last:
            return Rearrange("... C (L l) -> ... L (C l)", l=l)
        else:
            return Rearrange("... C (L l) -> ... (C l) L", l=l)
    elif len(patch_size) == 2:
        h, w = patch_size
        if channel_last:
            return Rearrange("... C (H h) (W w) -> ... H W (C h w)", h=h, w=w)
        else:
            return Rearrange("... C (H h) (W w) -> ... (C h w) H W", h=h, w=w)
    elif len(patch_size) == 3:
        l, h, w = patch_size
        if channel_last:
            return Rearrange(
                "... C (L l) (H h) (W w) -> ... L H W (C l h w)", l=l, h=h, w=w
            )
        else:
            return Rearrange(
                "... C (L l) (H h) (W w) -> ... (C l h w) L H W", l=l, h=h, w=w
            )
    elif len(patch_size) == 4:
        l, h, w, z = patch_size
        if channel_last:
            return Rearrange(
                "... C (L l) (H h) (W w) (Z z) -> ... L H W Z (C l h w z)",
                l=l,
                h=h,
                w=w,
                z=z,
            )
        else:
            return Rearrange(
                "... C (L l) (H h) (W w) (Z z) -> ... (C l h w z) L H W Z",
                l=l,
                h=h,
                w=w,
                z=z,
            )
    else:
        raise NotImplementedError()


def Unpatchify(patch_size, channel_last=False):
    if len(patch_size) == 1:
        (l,) = patch_size
        if channel_last:
            return Rearrange("... L (C l) -> ... C (L l)", l=l)
        else:
            return Rearrange("... (C l) L -> ... C (L l)", l=l)
    elif len(patch_size) == 2:
        h, w = patch_size
        if channel_last:
            return Rearrange("... H W (C h w) -> ... C (H h) (W w)", h=h, w=w)
        else:
            return Rearrange("... (C h w) H W -> ... C (H h) (W w)", h=h, w=w)
    elif len(patch_size) == 3:
        l, h, w = patch_size
        if channel_last:
            return Rearrange(
                "... L H W (C l h w) -> ... C (L l) (H h) (W w)", l=l, h=h, w=w
            )
        else:
            return Rearrange(
                "... (C l h w) L H W -> ... C (L l) (H h) (W w)", l=l, h=h, w=w
            )
    elif len(patch_size) == 4:
        l, h, w, z = patch_size
        if channel_last:
            return Rearrange(
                "... L H W Z (C l h w z) -> ... C (L l) (H h) (W w) (Z z)",
                l=l,
                h=h,
                w=w,
                z=z,
            )
        else:
            return Rearrange(
                "... (C l h w z) L H W Z -> ... C (L l) (H h) (W w) (Z z)",
                l=l,
                h=h,
                w=w,
                z=z,
            )
    else:
        raise NotImplementedError()


class Shrink(nn.Module):

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        slices = [slice(None, None, stride) for stride in self.stride]
        return x[(..., *slices)]


class Dilate(nn.Module):

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        kernel = torch.zeros(self.stride, dtype=x.dtype, device=x.device)
        kernel = kernel.flatten()
        kernel[0] = 1.0
        kernel = kernel.reshape(self.stride)
        return torch.kron(x.contiguous(), kernel)
