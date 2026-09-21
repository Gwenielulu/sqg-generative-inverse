import torch

GRADIENT_METHODS = {"dps", "ps", "dps_plus", "ps+", "mcg"}


def needs_grad(method):
    return method.lower() in GRADIENT_METHODS


def residual(x, y, obs, mask):
    return torch.linalg.vector_norm(obs.forward(x, mask=mask) - y)


def project(x, y, obs, mask, alpha=1.0):
    x_obs = obs.pseudo_inv(y, mask=mask, reference=x)
    return x + alpha * mask.to(x.dtype) * (x_obs - x)


def condition(method, x_t, sample, x0, y, obs, mask, cfg):
    method = method.lower()
    if method == "vanilla":
        return (sample, None)
    if method == "projection":
        sample = project(sample, y, obs, mask, cfg["alpha"])
        return (sample, float(residual(sample, y, obs, mask).detach().cpu()))
    if method in {"dps_plus", "ps+"}:
        loss = torch.zeros((), device=x_t.device, dtype=x_t.dtype)
        n = cfg["num_samples"]
        sigma = cfg["perturbation_std"]
        for _ in range(n):
            loss += residual(x0 + sigma * torch.randn_like(x0), y, obs, mask) / n
    else:
        loss = residual(x0, y, obs, mask)
    grad = torch.autograd.grad(loss, x_t)[0]
    sample = sample - cfg["scale"] * grad
    if method == "mcg":
        sample = project(sample, y, obs, mask, cfg["projection_alpha"])
    return (sample, float(loss.detach().cpu()))
