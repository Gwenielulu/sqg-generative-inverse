import torch
from sqg_inverse.evaluation import score
from sqg_inverse.inverse.operators import Observation


def test_fixed_mask_and_noise_are_reproducible():
    field = torch.randn(2, 2, 8, 8)
    operator = Observation(
        sparsity=0.25,
        noise_std=0.1,
        fixed_mask=True,
        shared_channel_mask=True,
        mask_seed=7,
    )
    first, first_mask = operator.observe(field, noise=True, noise_seed=9)
    second, second_mask = operator.observe(field, noise=True, noise_seed=9)
    assert torch.equal(first_mask, second_mask)
    assert torch.equal(first, second)
    assert torch.equal(first_mask[:, 0], first_mask[:, 1])


def test_metrics_are_zero_for_exact_reconstruction():
    field = torch.randn(1, 2, 8, 8)
    mask = torch.ones_like(field, dtype=torch.bool)
    metrics = score(field, field, mask)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
