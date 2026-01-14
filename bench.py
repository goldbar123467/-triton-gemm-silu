#!/usr/bin/env python3
"""Benchmark: Triton GEMM+SiLU vs PyTorch"""
import torch
import triton
from core.kernels.gemm_silu import gemm_silu

SIZES = [512, 1024, 2048, 4096]


def pytorch_fused(a, b):
    """PyTorch: matmul then SiLU (2 kernels, intermediate write)"""
    c = torch.matmul(a, b)
    return c * torch.sigmoid(c)


def pytorch_separate(a, b):
    """PyTorch: matmul + separate SiLU (shows memory overhead)"""
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    return torch.nn.functional.silu(c)


def bench(fn, a, b):
    for _ in range(10):
        fn(a, b)
    torch.cuda.synchronize()
    return triton.testing.do_bench(lambda: fn(a, b), warmup=50, rep=100)


def main():
    dev = torch.cuda.get_device_name(0)
    print(f"\n{'='*65}")
    print(f"  GEMM + SiLU Benchmark | {dev}")
    print(f"{'='*65}")
    print(f"{'N':>6} {'PyTorch':>12} {'Triton':>12} {'Ratio':>8} {'Mem Saved':>12}")
    print(f"{'-'*65}")

    for N in SIZES:
        a = torch.randn((N, N), device='cuda', dtype=torch.float16)
        b = torch.randn((N, N), device='cuda', dtype=torch.float16)

        pt_ms = bench(pytorch_fused, a, b)
        tr_ms = bench(gemm_silu, a, b)

        ratio = pt_ms / tr_ms
        # fusion saves one NxN fp16 intermediate tensor read+write
        mem_saved_mb = (N * N * 2 * 2) / 1e6  # read + write, 2 bytes each

        print(f"{N:>6} {pt_ms:>10.3f}ms {tr_ms:>10.3f}ms {ratio:>7.2f}x {mem_saved_mb:>10.1f}MB")

    print(f"{'-'*65}")
    print("Ratio >1 = Triton faster | Mem Saved = eliminated intermediate tensor")
    print()


if __name__ == "__main__":
    main()
