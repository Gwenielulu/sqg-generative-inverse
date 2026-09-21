from pathlib import Path
import torch
from tqdm.auto import tqdm
from ..models.pixel_ddpm.script_util import create_model_and_diffusion
from .conditioning import condition, needs_grad


def load_ddpm(cfg, checkpoint, device):
    model, diffusion = create_model_and_diffusion(cfg)
    weights = torch.load(Path(checkpoint), map_location="cpu")
    model.load_state_dict(weights)
    model.to(device).eval()
    return (model, diffusion)


def sample_ddpm(model, diffusion, method, y, obs, mask, shape, device, cfg):
    x = torch.randn(shape, device=device)
    history = []
    grad = needs_grad(method)
    steps = range(diffusion.num_timesteps - 1, -1, -1)
    for step in tqdm(steps, disable=not cfg["progress"]):
        x = x.detach().requires_grad_(grad)
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        with torch.set_grad_enabled(grad):
            out = diffusion.p_sample(
                model,
                x,
                t,
                clip_denoised=cfg["clip_denoised"],
                model_kwargs={},
            )
            x, loss = condition(
                method, x, out["sample"], out["pred_xstart"], y, obs, mask, cfg
            )
        x = x.detach()
        if loss is not None:
            history.append(loss)
    return (x, history)
