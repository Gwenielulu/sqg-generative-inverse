import torch
import torch.nn as nn
from ..nn.dcae import DCEncoder, DCDecoder


class SQGAutoencoder(nn.Module):

    def __init__(
        self,
        in_channels=2,
        latent_channels=32,
        hid_channels=(64, 128, 256, 384, 512),
        hid_blocks=(3, 3, 3, 2, 2),
        saturation="softclip2",
        latent_noise=0.0,
        spatial=2,
        periodic=True,
        **kwargs,
    ):
        super().__init__()
        assert spatial == 2, "SQG data is 2D"
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.saturation_type = saturation
        self.latent_noise = latent_noise
        self.encoder = DCEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            spatial=spatial,
            periodic=periodic,
            **kwargs,
        )
        self.decoder = DCDecoder(
            in_channels=latent_channels,
            out_channels=in_channels,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            spatial=spatial,
            periodic=periodic,
            **kwargs,
        )

    def saturate(self, z):
        if self.saturation_type is None or self.saturation_type == "none":
            return z
        elif self.saturation_type == "softclip":
            return z / (1 + abs(z) / 5)
        elif self.saturation_type == "softclip2":
            return z * torch.rsqrt(1 + torch.square(z / 5))
        elif self.saturation_type == "tanh":
            return torch.tanh(z / 5) * 5
        elif self.saturation_type == "arcsinh":
            return torch.arcsinh(z)
        elif self.saturation_type == "rmsnorm":
            return z * torch.rsqrt(
                torch.mean(torch.square(z), dim=1, keepdim=True) + 1e-05
            )
        else:
            raise ValueError(f"Unknown saturation: {self.saturation_type}")

    def encode(self, x):
        z = self.encoder(x)
        z = self.saturate(z)
        return z

    def decode(self, z, noisy=True):
        if noisy and self.latent_noise > 0:
            z = z + self.latent_noise * torch.randn_like(z)
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z, noisy=True)
        return (x_recon, z)

    def encode_trajectory(self, x_traj):
        B, C, T, H, W = x_traj.shape
        x_flat = x_traj.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        z_flat = self.encode(x_flat)
        LC = z_flat.shape[1]
        H_prime, W_prime = z_flat.shape[2:]
        z_traj = z_flat.reshape(B, T, LC, H_prime, W_prime).permute(0, 2, 1, 3, 4)
        return z_traj

    def decode_trajectory(self, z_traj, noisy=False):
        B, LC, T, H_prime, W_prime = z_traj.shape
        z_flat = z_traj.permute(0, 2, 1, 3, 4).reshape(B * T, LC, H_prime, W_prime)
        x_flat = self.decode(z_flat, noisy=noisy)
        C = x_flat.shape[1]
        H, W = x_flat.shape[2:]
        x_traj = x_flat.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return x_traj
