import numpy as np
import torch
import torch.nn as nn


class SineEncoding(nn.Module):

    def __init__(self, features, omega=1000.0):
        super().__init__()
        assert features % 2 == 0
        freqs = np.linspace(0, 1, features // 2)
        freqs = omega ** (-freqs)
        self.register_buffer("freqs", torch.as_tensor(freqs, dtype=torch.float32))

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        return torch.cat((torch.sin(x * self.freqs), torch.cos(x * self.freqs)), dim=-1)
