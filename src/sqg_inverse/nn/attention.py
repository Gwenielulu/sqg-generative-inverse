__all__ = ["MultiheadSelfAttention"]
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from .layers import RMSNorm


class MultiheadSelfAttention(nn.Module):

    def __init__(
        self,
        channels,
        attention_heads=1,
        qk_norm=True,
        dropout=None,
        checkpointing=False,
    ):
        super().__init__()
        assert channels % attention_heads == 0
        self.qkv_proj = nn.Linear(channels, 3 * channels, bias=False)
        self.y_proj = nn.Linear(channels, channels)
        if qk_norm:
            self.qk_norm = RMSNorm(dim=-1, eps=1e-05)
        else:
            self.qk_norm = nn.Identity()
        self.heads = attention_heads
        self.dropout = nn.Dropout(0.0 if dropout is None else dropout)
        self.checkpointing = checkpointing

    def _forward(self, x, theta=None, mask=None):
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "... L (n H C) -> n ... H L C", n=3, H=self.heads)
        q, k = (self.qk_norm(q), self.qk_norm(k))
        if theta is not None:
            theta = rearrange(theta, "... L (H C) -> ... H L C", H=self.heads)
            q, k = apply_rope(q, k, theta)
        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0,
        )
        y = rearrange(y, "... H L C -> ... L (H C)")
        y = self.y_proj(y)
        return y

    def forward(self, x, theta=None, mask=None):
        if self.checkpointing:
            return checkpoint(self._forward, x, theta, mask, use_reentrant=False)
        else:
            return self._forward(x, theta, mask)


def apply_rope(q, k, theta):
    q = q.unflatten(-1, (-1, 2))
    k = k.unflatten(-1, (-1, 2))
    q_real, q_imag = (q[..., 0], q[..., 1])
    k_real, k_imag = (k[..., 0], k[..., 1])
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    q = torch.stack(
        (
            q_real * cos_theta - q_imag * sin_theta,
            q_real * sin_theta + q_imag * cos_theta,
        ),
        dim=-1,
    ).flatten(-2)
    k = torch.stack(
        (
            k_real * cos_theta - k_imag * sin_theta,
            k_real * sin_theta + k_imag * cos_theta,
        ),
        dim=-1,
    ).flatten(-2)
    return (q, k)
