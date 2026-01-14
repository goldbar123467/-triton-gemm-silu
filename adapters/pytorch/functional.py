import torch
from core.kernels.gemm_silu import gemm_silu as _kernel


def gemm_silu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "needs GPU"
    return _kernel(a, b)
