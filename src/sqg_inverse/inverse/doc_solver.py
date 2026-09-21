from functools import partial
import torch
from .optimal_control.envs.diffusion import DPSDiffusion
from .optimal_control.optimizers.mfddp import MFDDP


def run_doc(model, diffusion, y, obs, mask, shape, device, cfg):
    batch = y.shape[0]
    target = y.reshape(batch, -1)

    def cost(state, target):
        field = state.reshape(-1, *shape)
        pred = obs.forward(field, mask=mask).reshape(batch, -1)
        return torch.linalg.vector_norm(pred - target)

    env = DPSDiffusion(
        model=model,
        sampler=diffusion,
        state_cost=cost,
        target=target,
        shape=shape,
        weight_scale=cfg["weight_scale"],
        num_steps=cfg["num_steps"],
        device=device,
        seed=cfg["seed"],
    )
    env.mask = mask.to(y.dtype)
    if cfg["use_projection"]:
        env.set_projection_params(obs, y, use_projection=True)
    make_solver = partial(
        MFDDP,
        seed=cfg["seed"],
        lr=cfg["learning_rate"],
        k_jacobian=cfg["k_jacobian"],
        k_hessian=cfg["k_hessian"],
        k_mf=cfg["k_mf"],
        verbose=cfg["verbose"],
    )
    solver = make_solver(env)
    x0 = env.initialize_state(batch)
    _, states = solver.solve(x0, num_iterations=cfg["num_iterations"])
    pred = states[-1].to(device).reshape(batch, *shape)
    return (pred, states)
