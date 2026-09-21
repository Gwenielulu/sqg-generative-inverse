import torch
import torch.nn as nn
import torch.nn.functional as F


def polynomial_warp(t, s):
    return 4 * (1 - s) * t**3 + 6 * (s - 1) * t**2 + (3 - 2 * s) * t


def append_dims(t, ndims):
    return t.reshape(*t.shape, *(1,) * ndims)


class RectifiedFlow(nn.Module):

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std
        self.predict = "noise"
        self.eps = 0.005
        self.data_shape = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, clean):
        noise = torch.randn_like(clean)
        times = torch.sigmoid(
            torch.randn(clean.shape[0], device=self.device) * self.std + self.mean
        )
        padded_times = append_dims(times, clean.ndim - 1)
        noisy = noise.lerp(clean, padded_times)
        prediction = self.model(noisy, times=times)
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(self, batch_size, steps, time_warping, warp_param):
        noise = torch.randn(batch_size, *self.data_shape, device=self.device)

        def ode_fn(t, x):
            times = t.expand(x.shape[0])
            predicted_noise = self.model(x, times=times)
            padded_times = append_dims(times, x.ndim - 1)
            return (x - predicted_noise) / padded_times.clamp_min(self.eps)

        times = torch.linspace(0.0, 1.0, steps, device=self.device)
        if time_warping == "polynomial":
            times = polynomial_warp(times, warp_param)
        elif time_warping != "linear":
            raise ValueError(time_warping)
        x = noise
        for t0, t1 in zip(times[:-1], times[1:]):
            dt = t1 - t0
            midpoint = t0 + 0.5 * dt
            x_mid = x + 0.5 * dt * ode_fn(t0, x)
            x = x + dt * ode_fn(midpoint, x_mid)
        return x
