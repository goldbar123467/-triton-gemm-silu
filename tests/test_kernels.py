import torch
import pytest
from core.kernels.gemm_silu import gemm_silu


def ref_gemm_silu(a, b):
    c = torch.matmul(a.float(), b.float())
    return (c * torch.sigmoid(c)).to(a.dtype)


@pytest.mark.parametrize("M,N,K", [(512, 512, 512), (1024, 1024, 1024), (128, 256, 64)])
def test_gemm_silu(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    out = gemm_silu(a, b)
    ref = ref_gemm_silu(a, b)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M,N,K", [(512, 512, 512)])
def test_benchmark(M, N, K, benchmark):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    benchmark(gemm_silu, a, b)
