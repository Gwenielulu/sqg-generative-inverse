import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        t = t.reshape(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device) / half
        )
        embedding = torch.cat(
            [torch.cos(t[:, None] * freqs), torch.sin(t[:, None] * freqs)], dim=-1
        )
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding


class TimeConditionedModel(nn.Module):

    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, times):
        return self.backbone(x, mod=self.time_embed(times))
