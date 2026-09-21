import numpy as np
import torch
from ..sde_lib import VPSDE


class Diffusion:

    def __init__(
        self,
        shape,
        weight_scale=1.0,
        num_steps=100,
        mode="ddim",
        control_type="input",
        u_nominal=None,
        device="cpu",
        eps=0.001,
        verbose=False,
        seed=None,
    ):
        self.shape = shape
        self.weight_scale = weight_scale
        self.no_running_cost = weight_scale == 0.0
        self.ndims = np.prod(shape)
        self.num_steps = num_steps
        self.timesteps = torch.linspace(1, eps, num_steps, device=device)
        self.inv_timesteps = lambda t: ((1 - t) * num_steps).long()
        self.device = device
        self.eps = eps
        self.verbose = verbose
        self.control_type = control_type
        self.mode = mode
        self.u_nominal = u_nominal
        self.dt = 1 / num_steps
        self.alphas = VPSDE().alphas
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(1000)
        if control_type == "input" or control_type == "output":
            self.control_dim = self.ndims
        elif control_type == "h":
            self.control_dim = np.prod(self.h_dim)
        else:
            raise NotImplementedError(f"control_type: {control_type} not supported")
        self.state_dim = self.ndims
        self.dt = (1 - self.eps) / self.num_steps

    def initialize_state(self, n, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        device = self.device
        t = torch.ones((n, 1), device=device)
        init_x = torch.randn((n, self.ndims), device=device)
        return init_x

    def u_diff(self, t, u):
        if self.u_nominal is not None:
            t_idx = self.inv_timesteps(t)
            u_nominal = self.u_nominal[t_idx].reshape(-1, self.ndims)
            assert u.shape[1:] == u_nominal.shape[1:]
            return u - u_nominal
        return u

    def alpha(self, t):
        t_int = (t * (len(self.alphas) - 1)).int()
        alpha = self.alphas[t_int.cpu()]
        return alpha

    def snr(self, t):
        alphat = self.alpha(t)
        return torch.sqrt(alphat) / torch.sqrt(1 - alphat + 1e-07)

    def weight(self, t):
        weight = self.snr(t).reshape(-1, 1).to(t.device)
        if self.weight_scale == 0.0:
            return torch.zeros_like(weight)
        weight = self.weight_scale * weight
        return weight / self.dt**2

    def running_state_cost(self, t, x, u):
        weight = self.weight(t).squeeze()
        x = x.reshape(-1, *self.shape)
        cost = self.state_cost(x, self.target)
        return weight * cost * 10.0

    def running_control_cost(self, t, x, u):
        weight = self.weight(t)
        return torch.squeeze(0.5 * (self.u_diff(t, u) ** 2 * weight).sum(axis=1))

    def terminal_cost(self, x, reduce_sum=True):
        x = x.reshape(-1, *self.shape)
        cost = self.state_cost(x, self.target)
        return cost

    def h_cost(self, t, x, u, low_mem_mode=False):
        weight = self.weight(t)
        if low_mem_mode:
            ones = torch.ones_like(weight, device=x.device)
            return ((ones * 0, ones * 0), (ones * 0, ones * weight))
        else:
            z = lambda *shape: torch.zeros(size=shape, device=x.device)
            n, d_s = x.shape
            _, d_c = u.shape
            luu = torch.diag_embed(torch.ones_like(u) * weight)
            lxu, lux, lxx = (z(n, d_s, d_c), z(n, d_c, d_s), z(n, d_s, d_s))
            return ((lxx, lxu), (lux, luu))

    def j_cost(self, t, x, u):
        weight = self.weight(t)
        z = lambda *shape: torch.zeros(size=shape, device=x.device)
        lx, lu = (
            z(len(x), self.state_dim, 1),
            (self.u_diff(t, u) * weight).reshape(len(x), self.control_dim, 1),
        )
        return (lx, lu)


class DDPMDiffusion(Diffusion):

    def __init__(self, state_cost, target, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target
        self.state_cost = state_cost
        self.operator = None
        self.measurement = None
        self.use_projection = False
        self.mask = None

    def set_projection_params(self, operator, measurement, use_projection=True):
        self.operator = operator
        self.measurement = measurement
        self.use_projection = use_projection

    def project_with_noise_consistency(self, x_next, t):
        if not self.use_projection or self.operator is None:
            return x_next
        t_next = t - self.dt
        if t_next < self.eps:
            return x_next
        t_int = self.discretize_time(t_next)
        alpha_bar_t = self.alphas[t_int.cpu()].to(x_next.device)
        noise = torch.randn_like(self.measurement)
        noisy_measurement = (
            torch.sqrt(alpha_bar_t) * self.measurement
            + torch.sqrt(1 - alpha_bar_t) * noise
        )
        x_next_reshaped = x_next.reshape(-1, *self.shape)
        if self.mask is not None:
            projected = (
                1 - self.mask
            ) * x_next_reshaped + self.mask * noisy_measurement
        else:
            projected = self.operator.project(
                data=x_next_reshaped, measurement=noisy_measurement
            )
        return projected.reshape(-1, self.ndims)

    def step(self, t, x, u):
        t_int = self.discretize_time(t - self.dt)
        if self.control_type == "input":
            assert x.shape == u.shape, f"x.shape: {x.shape}, u.shape: {u.shape}"
            x_next = self.step_fn((x + u).reshape(-1, *self.shape), t_int).reshape(
                -1, self.ndims
            )
        elif self.control_type == "output":
            assert x.shape == u.shape, f"x.shape: {x.shape}, u.shape: {u.shape}"
            x_next = (
                self.step_fn(x.reshape(-1, *self.shape), t_int).reshape(-1, self.ndims)
                + u
            )
        elif self.control_type == "h":
            x_next = self.step_fn(
                x.reshape(-1, *self.shape), t_int, u=u.reshape(-1, *self.h_dim)
            ).reshape(-1, self.ndims)
        x_next = self.project_with_noise_consistency(x_next, t)
        return x_next

    def std(self, t):
        std = VPSDE().marginal_prob(t.reshape(-1), t.reshape(-1))[1]
        return std.squeeze()

    def g(self, t):
        std = VPSDE().sde(t.reshape(-1), t.reshape(-1))[1]
        return std.squeeze()

    def discretize_time(self, t):
        t_int = (t * (self.get_model_steps() - 1)).int().reshape(-1)
        torch.manual_seed((self.seed + t_int).median().item())
        return t_int


class DPSDiffusion(DDPMDiffusion):

    def __init__(self, model, sampler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler
        self.model = model
        self.step_fn = step_fn = lambda x, t, u=None: sampler.p_sample(
            x=x, t=t, model=model
        )["sample"]

    def get_model_steps(self):
        return self.num_steps

    def denoise(self, t, x, u=None):
        shape = (-1,) + self.shape
        t_int = self.discretize_time(t) - 1
        t_int = t_int.clip(0, self.sampler.num_timesteps - 1)
        denoise_fn = lambda x, u=None: self.sampler.p_sample(
            x=x.reshape(shape), t=t_int, u=u, model=self.model
        )["pred_xstart"].reshape(-1, self.ndims)
        if self.control_type == "input" or self.control_type == "output":
            if u is None:
                u = torch.zeros_like(x)
            assert x.shape == u.shape
            return denoise_fn(x + u)
        elif self.control_type == "h":
            if u is None:
                return denoise_fn(x)
            return denoise_fn(x, u=u.reshape(-1, *self.h_dim))
