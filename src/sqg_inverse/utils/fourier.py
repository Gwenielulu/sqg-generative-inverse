import math
import torch
from functools import cache


@cache
def isotropic_binning(shape, bins=None, device=None):
    k = []
    for s in shape:
        k_i = torch.fft.fftfreq(s, device=device)
        k.append(k_i)
    k2 = map(torch.square, k)
    k2_iso = sum(torch.meshgrid(*k2, indexing="ij"))
    k_iso = torch.sqrt(k2_iso)
    if bins is None:
        bins = math.floor(math.sqrt(k_iso.ndim) * min(k_iso.shape) / 2)
    edges = torch.linspace(0, k_iso.max(), bins + 1, device=device)
    indices = torch.bucketize(k_iso.flatten(), edges)
    counts = torch.bincount(indices, minlength=bins + 1)
    return (edges, counts, indices)


def isotropic_power_spectrum(x, spatial=2):
    x = torch.as_tensor(x)
    batch, shape = (x.shape[:-spatial], x.shape[-spatial:])
    edges, counts, indices = isotropic_binning(shape, device=x.device)
    s = torch.fft.fftn(x, dim=tuple(range(-spatial, 0)), norm="ortho")
    p = torch.square(torch.abs(s))
    p = torch.flatten(p, start_dim=-spatial)
    p_iso = torch.zeros((*batch, *edges.shape), dtype=x.dtype, device=x.device)
    p_iso = p_iso.scatter_add(dim=-1, index=indices.expand_as(p), src=p)
    p_iso = p_iso / torch.clip(counts, min=1)
    return (p_iso[..., 1:], edges[1:])


def isotropic_cross_correlation(x, y, spatial=2):
    x, y = (torch.as_tensor(x), torch.as_tensor(y))
    x, y = torch.broadcast_tensors(x, y)
    batch, shape = (x.shape[:-spatial], x.shape[-spatial:])
    edges, counts, indices = isotropic_binning(shape, device=x.device)
    sx = torch.fft.fftn(x, dim=tuple(range(-spatial, 0)), norm="ortho")
    sy = torch.fft.fftn(y, dim=tuple(range(-spatial, 0)), norm="ortho")
    c = torch.abs(sx * torch.conj(sy))
    c = torch.flatten(c, start_dim=-spatial)
    c_iso = torch.zeros((*batch, *edges.shape), dtype=x.dtype, device=x.device)
    c_iso = c_iso.scatter_add(dim=-1, index=indices.expand_as(c), src=c)
    c_iso = c_iso / torch.clip(counts, min=1)
    return (c_iso[..., 1:], edges[1:])
