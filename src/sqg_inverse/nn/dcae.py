__all__ = ["DCEncoder", "DCDecoder"]
import math
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .layers import ConvNd, LayerNorm, Patchify, SelfAttentionNd, Unpatchify


class Residual(nn.Sequential):

    def forward(self, x):
        return x + super().forward(x)


class ResBlock(nn.Module):

    def __init__(
        self,
        channels,
        norm="layer",
        groups=16,
        attention_heads=None,
        ffn_factor=1,
        spatial=2,
        dropout=None,
        checkpointing=False,
        **kwargs
    ):
        super().__init__()
        self.checkpointing = checkpointing
        if norm == "layer":
            self.norm = LayerNorm(dim=-spatial - 1)
        elif norm == "group":
            self.norm = nn.GroupNorm(
                num_groups=min(groups, channels), num_channels=channels, affine=False
            )
        else:
            raise NotImplementedError()
        if attention_heads is None:
            self.attn = nn.Identity()
        else:
            self.attn = Residual(SelfAttentionNd(channels, heads=attention_heads))
            kwargs.update(kernel_size=1, padding=0)
        self.ffn = nn.Sequential(
            ConvNd(channels, ffn_factor * channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(ffn_factor * channels, channels, spatial=spatial, **kwargs),
        )
        self.ffn[-1].weight.data.mul_(0.01)

    def _forward(self, x):
        y = self.norm(x)
        y = self.attn(y)
        y = self.ffn(y)
        return x + y

    def forward(self, x):
        if self.checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class DCEncoder(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hid_channels=(64, 128, 256),
        hid_blocks=(3, 3, 3),
        kernel_size=3,
        stride=2,
        pixel_shuffle=True,
        norm="layer",
        attention_heads={},
        ffn_factor=1,
        spatial=2,
        patch_size=1,
        periodic=False,
        dropout=None,
        checkpointing=False,
        identity_init=True,
    ):
        super().__init__()
        assert len(hid_blocks) == len(hid_channels)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial
        if isinstance(stride, int):
            stride = [stride] * spatial
        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial
        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple((k // 2 for k in kernel_size)),
            padding_mode="circular" if periodic else "zeros",
        )
        self.patch = Patchify(patch_size=patch_size)
        self.descent = nn.ModuleList()
        for i, num_blocks in enumerate(hid_blocks):
            blocks = nn.ModuleList()
            if i > 0:
                if pixel_shuffle:
                    blocks.append(
                        nn.Sequential(
                            Patchify(patch_size=stride),
                            ConvNd(
                                hid_channels[i - 1] * math.prod(stride),
                                hid_channels[i],
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs
                            ),
                        )
                    )
                else:
                    blocks.append(
                        ConvNd(
                            hid_channels[i - 1],
                            hid_channels[i],
                            spatial=spatial,
                            stride=stride,
                            identity_init=identity_init,
                            **kwargs
                        )
                    )
            else:
                blocks.append(
                    ConvNd(
                        math.prod(patch_size) * in_channels,
                        hid_channels[i],
                        spatial=spatial,
                        **kwargs
                    )
                )
            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        norm=norm,
                        attention_heads=attention_heads.get(i, None),
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs
                    )
                )
            if i + 1 == len(hid_blocks):
                blocks.append(
                    ConvNd(
                        hid_channels[i],
                        out_channels,
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs
                    )
                )
            self.descent.append(blocks)

    def forward(self, x):
        x = self.patch(x)
        for blocks in self.descent:
            for block in blocks:
                x = block(x)
        return x


class DCDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hid_channels=(64, 128, 256),
        hid_blocks=(3, 3, 3),
        kernel_size=3,
        stride=2,
        pixel_shuffle=True,
        norm="layer",
        attention_heads={},
        ffn_factor=1,
        spatial=2,
        patch_size=1,
        periodic=False,
        dropout=None,
        checkpointing=False,
        identity_init=True,
    ):
        super().__init__()
        assert len(hid_blocks) == len(hid_channels)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial
        if isinstance(stride, int):
            stride = [stride] * spatial
        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial
        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple((k // 2 for k in kernel_size)),
            padding_mode="circular" if periodic else "zeros",
        )
        self.unpatch = Unpatchify(patch_size=patch_size)
        self.ascent = nn.ModuleList()
        for i, num_blocks in reversed(list(enumerate(hid_blocks))):
            blocks = nn.ModuleList()
            if i + 1 == len(hid_blocks):
                blocks.append(
                    ConvNd(
                        in_channels,
                        hid_channels[i],
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs
                    )
                )
            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        norm=norm,
                        attention_heads=attention_heads.get(i, None),
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs
                    )
                )
            if i > 0:
                if pixel_shuffle:
                    blocks.append(
                        nn.Sequential(
                            ConvNd(
                                hid_channels[i],
                                hid_channels[i - 1] * math.prod(stride),
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs
                            ),
                            Unpatchify(patch_size=stride),
                        )
                    )
                else:
                    blocks.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=tuple(stride), mode="nearest"),
                            ConvNd(
                                hid_channels[i],
                                hid_channels[i - 1],
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs
                            ),
                        )
                    )
            else:
                blocks.append(
                    ConvNd(
                        hid_channels[i],
                        math.prod(patch_size) * out_channels,
                        spatial=spatial,
                        **kwargs
                    )
                )
            self.ascent.append(blocks)

    def forward(self, x):
        for blocks in self.ascent:
            for block in blocks:
                x = block(x)
        x = self.unpatch(x)
        return x
