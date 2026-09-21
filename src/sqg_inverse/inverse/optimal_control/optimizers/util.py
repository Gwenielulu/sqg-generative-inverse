import torch


def inv(matrix, eps=1e-7):
    extra_dims = matrix.shape[:-2]
    size = min(*matrix.shape[-2:])
    eye = torch.diag_embed(torch.ones(size=(*extra_dims, size), device=matrix.device))
    return torch.inverse(matrix[..., :size, :size] + eps * eye)


def get_Q(dims, k, device):
    dims = tuple(dims)
    if dims[-1] > k:
        matrix = torch.randn((*dims, k), device=device)
        return torch.linalg.qr(matrix)[0]
    *extra_dims, rows = dims
    matrix = torch.eye(rows, k, device=device)
    return matrix.reshape([1] * len(extra_dims) + [rows, k]).tile(extra_dims + [1, 1])


def get_mjp_fn(fn, *xs, return_output=False, chunk_size=None, randomness="same"):
    output, vjp_fn = torch.func.vjp(fn, *xs)

    def mjp_fn(vectors):
        mapped_vjp = torch.vmap(
            vjp_fn,
            in_dims=-1,
            out_dims=1,
            randomness=randomness,
            chunk_size=chunk_size,
        )
        return mapped_vjp(vectors)

    if return_output:
        return output, mjp_fn
    return mjp_fn


def get_jmp_fn(fn, *xs, chunk_size=None, randomness="same"):
    def jmp_fn(vectors):
        jvp = lambda tangents: torch.func.jvp(fn, xs, tangents)[1]
        mapped_jvp = torch.vmap(
            jvp,
            in_dims=-1,
            out_dims=-1,
            randomness=randomness,
            chunk_size=chunk_size,
        )
        return mapped_jvp(vectors)

    return jmp_fn
