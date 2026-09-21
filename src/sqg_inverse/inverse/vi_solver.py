from types import SimpleNamespace
import math
import torch


def _append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *(1,) * ndims)


class LatentModel:

    def __init__(
        self,
        ldm,
        ae,
        *,
        sigma_train=2660.0,
        scale_factor=0.0001 * 300.0 / 9.8,
    ):
        self.ldm = ldm
        self.ae = ae
        self.phys_scale = float(sigma_train) * float(scale_factor)
        self.ldm.eval()
        self.ae.eval()
        self.ldm.requires_grad_(False)
        self.ae.requires_grad_(False)
        ldm_device = next(self.ldm.parameters()).device
        ae_device = next(self.ae.parameters()).device
        if ldm_device != ae_device:
            raise ValueError(
                f"IIDLDM and AE must be on the same device for VI optimization, got {ldm_device} vs {ae_device}"
            )
        self.device = ldm_device

    @property
    def latent_channels(self):
        return int(self.ldm.latent_channels)

    def get_timesteps(self, n_steps, device, ts_min):
        ts = torch.linspace(1.0, ts_min, n_steps + 2, device=device)
        return ts[1:-1]

    def _to_model_field(self, x_phys):
        return x_phys / self.phys_scale

    def _to_physical_field(self, x_model):
        return x_model * self.phys_scale

    @torch.no_grad()
    def encode(self, x_phys):
        x_phys = x_phys.to(self.device)
        x_model = self._to_model_field(x_phys)
        return self.ae.encode(x_model)

    def decode(self, latent):
        latent = latent.to(self.device)
        x_model = self.ae.decode(latent, noisy=False)
        return self._to_physical_field(x_model)

    def infer_latent_shape(self, x_phys):
        with torch.no_grad():
            z = self.encode(x_phys[:1])
        return tuple(z.shape)

    def single_step(self, latent_mu, t, noise):
        noise = noise.to(latent_mu.device, dtype=latent_mu.dtype)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t = t.to(device=latent_mu.device, dtype=latent_mu.dtype)
        if t.numel() == 1:
            t_batch = t.expand(latent_mu.shape[0])
        elif t.numel() == latent_mu.shape[0]:
            t_batch = t
        else:
            raise ValueError(
                f"t has {t.numel()} values but batch is {latent_mu.shape[0]}"
            )
        padded_t = _append_dims(t_batch, latent_mu.ndim - 1)
        a_t = 1.0 - padded_t
        sigma_t = padded_t
        x_t = a_t * latent_mu + sigma_t * noise
        model_times = (1.0 - t_batch).clamp(min=0.0, max=1.0)
        model_output = self.ldm.rf.model(x_t, times=model_times)
        eps_prediction = model_output
        v_pred = (x_t - eps_prediction) / a_t.clamp_min(self.ldm.rf.eps)
        return (eps_prediction, noise, a_t, sigma_t, v_pred)


class LatentVI:

    def __init__(self, model, forward_operator, config):
        self.model = model
        self.forward_operator = forward_operator
        self.config = config
        self.epochs = int(config["epochs"])
        self.n_steps = int(config["n_steps"])
        self.ts_min = float(config["ts_min"])
        self.likelihood_weight = float(config["likelihood_weight"])
        self.likelihood_steps = int(config["likelihood_steps"])
        self.early_stopping = float(config["early_stopping"])
        self.regularizer_weight = float(config["regularizer_weight"])
        self.optimizer_cfg = config["optimizer"]
        self.optimizer_dataterm_cfg = config["optimizer_dataterm"]
        self.history_stride = int(config["history_stride"])

    @staticmethod
    def _optimizer(optimizer_config, params):
        name = optimizer_config["name"].lower()
        kwargs = optimizer_config["kwargs"]
        if name == "adam":
            return torch.optim.Adam(params, **kwargs)
        if name == "adamw":
            return torch.optim.AdamW(params, **kwargs)
        if name == "sgd":
            return torch.optim.SGD(params, **kwargs)
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")

    def _optimizers(self, latent_mu):
        params = [latent_mu]
        optimizer = self._optimizer(self.optimizer_cfg, params)
        optimizer_dataterm = self._optimizer(self.optimizer_dataterm_cfg, params)
        return (optimizer, optimizer_dataterm)

    def _prior_loss(self, noise, a_t, sigma_t, t, latent_mu, v_pred):
        x_t = a_t * latent_mu + sigma_t * noise
        t_b = t.reshape(-1, 1).clamp_min(1e-06)
        one_minus_t = (1.0 - t.reshape(-1, 1)).clamp_min(1e-06)
        lambda_t_der = -2.0 * (1.0 / one_minus_t + 1.0 / t_b)
        lambda_t_der = _append_dims(lambda_t_der.squeeze(-1), x_t.ndim - 1)
        u_t = (
            -x_t / _append_dims((1.0 - t).clamp_min(1e-06), x_t.ndim - 1)
            - 0.5 * _append_dims(t, x_t.ndim - 1) * lambda_t_der * noise
        )
        return -self.regularizer_weight * (u_t - v_pred).reshape(x_t.shape[0], -1)

    def data_term(
        self,
        latent_mu,
        y,
        mask,
        optimizer_dataterm,
        likelihood_weight,
        trace_context,
    ):
        last = float("nan")
        inner_history = []
        if y.ndim == 2:
            obs_count = float(y.numel())
        else:
            obs_count = float(mask.to(dtype=latent_mu.dtype).sum().item())
        if obs_count <= 0.0:
            raise ValueError("No observed entries in mask (obs_count=0).")
        for inner_iter in range(self.likelihood_steps):
            optimizer_dataterm.zero_grad(set_to_none=True)
            x_hat = self.model.decode(latent_mu)
            y_hat = self.forward_operator.forward(x_hat, mask=mask, noise=False)
            if y_hat.ndim == 2 and y.ndim == 2:
                data_loss = (y_hat - y).square().sum()
            else:
                data_loss = ((y_hat - y).square() * mask.to(dtype=y_hat.dtype)).sum()
            last = float(data_loss.detach().item())
            weighted_data_loss = float(likelihood_weight) * data_loss
            early_stop = bool(
                self.early_stopping > 0.0 and last <= self.early_stopping * obs_count
            )
            trace = dict(trace_context)
            trace.update(
                {
                    "inner_iter": float(inner_iter),
                    "data_loss": float(last),
                    "likelihood_weighted_data_loss": float(
                        weighted_data_loss.detach().item()
                    ),
                    "obs_count": float(obs_count),
                    "early_stop_triggered": float(1.0 if early_stop else 0.0),
                }
            )
            inner_history.append(trace)
            if early_stop:
                break
            weighted_data_loss.backward()
            optimizer_dataterm.step()
        return (last, inner_history)

    def solve(self, y, mask, seed):
        device = self.model.device
        y = y.to(device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        mask = mask.to(device=device, dtype=torch.bool)
        x_inv = self.forward_operator.pseudo_inv(y, mask=mask)
        latent_mu = self.model.encode(x_inv).detach().clone()
        latent_mu.requires_grad_(True)
        optim_noise = torch.randn_like(latent_mu)
        timesteps = self.model.get_timesteps(self.n_steps, device, self.ts_min).to(
            dtype=latent_mu.dtype
        )
        history = []
        history_inner = []
        for epoch in range(self.epochs):
            optimizer, optimizer_dataterm = self._optimizers(latent_mu)
            for i, t_scalar in enumerate(timesteps):
                t = t_scalar.reshape(1)
                alpha_t = float((1.0 - t).clamp(min=0.0, max=1.0).item())
                rand_scale = math.sqrt(1.0 - alpha_t * alpha_t)
                noise_for_step = (
                    alpha_t * optim_noise.detach()
                    + rand_scale * torch.randn_like(optim_noise)
                )
                _, noise, a_t, sigma_t, v_pred = self.model.single_step(
                    latent_mu, t, noise_for_step
                )
                optim_noise = (
                    a_t * latent_mu + sigma_t * noise + a_t * v_pred
                ).detach()
                reg_term = self._prior_loss(
                    noise=noise,
                    a_t=a_t,
                    sigma_t=sigma_t,
                    t=t,
                    latent_mu=latent_mu,
                    v_pred=v_pred,
                )
                optimizer.zero_grad(set_to_none=True)
                reg_loss = (
                    reg_term.detach() * latent_mu.reshape(latent_mu.shape[0], -1)
                ).sum()
                reg_loss.backward()
                optimizer.step()
                likelihood_weight = self.regularizer_weight * self.likelihood_weight
                global_iter = epoch * timesteps.numel() + i
                data_loss, data_inner = self.data_term(
                    latent_mu=latent_mu,
                    y=y.detach(),
                    mask=mask,
                    optimizer_dataterm=optimizer_dataterm,
                    likelihood_weight=likelihood_weight,
                    trace_context={
                        "epoch": float(epoch),
                        "iter": float(i),
                        "global_iter": float(global_iter),
                        "t": float(t.item()),
                        "alpha_t": float(alpha_t),
                    },
                )
                history_inner.extend(data_inner)
                should_log = i % self.history_stride == 0 or i + 1 == timesteps.numel()
                if should_log:
                    history.append(
                        {
                            "epoch": float(epoch),
                            "iter": float(i),
                            "global_iter": float(global_iter),
                            "t": float(t.item()),
                            "alpha_t": float(alpha_t),
                            "reg_loss": float(reg_loss.detach().item()),
                            "data_loss": float(data_loss),
                        }
                    )
        x_hat = self.model.decode(latent_mu)
        y_hat = self.forward_operator.forward(x_hat, mask=mask, noise=False)
        return SimpleNamespace(
            x_hat=x_hat.detach(),
            latent_mu=latent_mu.detach(),
            y_hat=y_hat.detach(),
            history=history,
            history_inner=history_inner,
        )
