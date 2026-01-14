# gemm_silu

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Triton](https://img.shields.io/badge/triton-2.0+-green.svg)](https://github.com/triton-lang/triton)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

```
   ┌─────────────────────────────────────────────────────────────┐
   │                                                             │
   │   ██████╗ ███████╗███╗   ███╗███╗   ███╗                   │
   │  ██╔════╝ ██╔════╝████╗ ████║████╗ ████║                   │
   │  ██║  ███╗█████╗  ██╔████╔██║██╔████╔██║                   │
   │  ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚██╔╝██║    ┌──────────┐   │
   │  ╚██████╔╝███████╗██║ ╚═╝ ██║██║ ╚═╝ ██║ ×  │   SiLU   │   │
   │   ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝    └──────────┘   │
   │                                                             │
   │              Fused in one GPU kernel pass                   │
   └─────────────────────────────────────────────────────────────┘
```

Fused GEMM + SiLU activation in a single Triton kernel. No intermediate tensor, no extra memory traffic.

## Install

```bash
git clone https://github.com/youruser/gemm_silu.git
cd gemm_silu
python -m venv .venv && source .venv/bin/activate
pip install torch triton pytest
```

## Usage

```python
from core.kernels.gemm_silu import gemm_silu

a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
b = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)

# fused: C = SiLU(A @ B)
c = gemm_silu(a, b)
```

## Benchmark

```
=================================================================
  GEMM + SiLU Benchmark | NVIDIA GeForce RTX 2070
=================================================================
     N      PyTorch       Triton    Ratio    Mem Saved
-----------------------------------------------------------------
   512      0.033ms      0.106ms    0.32x        1.0MB
  1024      0.115ms      0.754ms    0.15x        4.2MB
  2048      0.817ms      5.624ms    0.15x       16.8MB
  4096      4.726ms     43.819ms    0.11x       67.1MB
-----------------------------------------------------------------
```

**Reality check**: cuBLAS (PyTorch backend) is *highly* optimized. Custom kernels rarely beat it for standard ops on consumer GPUs. The value is elsewhere.

```bash
python bench.py  # run yourself
```

## Why Triton?

### The Problem

```
PyTorch (2 kernel launches, intermediate tensor):
  A @ B → [write to VRAM] → read → SiLU → [write to VRAM]
              ↑ 67MB at 4096×4096

Triton (1 kernel, fused):
  A @ B → SiLU → [write to VRAM]
              ↑ no intermediate
```

### When Triton Wins

| Use Case | Why |
|----------|-----|
| **Operator fusion** | Skip intermediate tensors, reduce memory bandwidth |
| **Custom ops** | Ops that don't exist in PyTorch (flash attention started here) |
| **Research** | Iterate on GPU code in Python, not CUDA C++ |
| **Weird shapes** | cuBLAS is tuned for common sizes; custom kernels can specialize |
| **Memory bound** | When you're bottlenecked by bandwidth, not compute |

### When cuBLAS Wins

- Standard GEMM on standard sizes
- You're on consumer GPUs with good cuBLAS support
- You don't need fusion

## Q&A

**Q: Why is Triton slower than PyTorch here?**

cuBLAS has person-centuries of optimization. It's assembly-tuned per GPU architecture. Triton is a compiler—it generates good code but not hand-tuned code. On datacenter GPUs (A100, H100), the gap shrinks.

**Q: Then why use Triton?**

1. **Fusion**: The real win. GEMM+SiLU in one kernel = half the memory traffic
2. **Custom ops**: Try writing flash attention in CUDA. Then try Triton. Night and day.
3. **Portability**: Same code runs on NVIDIA and AMD
4. **Iteration speed**: Change kernel, run, see result. No compile step.

**Q: Triton vs CUDA?**

| | Triton | CUDA |
|---|--------|------|
| Language | Python-like | C++ |
| Learning curve | Days | Months |
| Peak perf | 80-95% of CUDA | 100% |
| Portability | NVIDIA + AMD | NVIDIA only |
| Iteration speed | Fast | Slow |

**Q: Triton vs other compilers (XLA, TVM, etc)?**

| | Triton | XLA | TVM |
|---|--------|-----|-----|
| Focus | Single kernels | Graph compilation | Graph compilation |
| Control | Fine-grained | Coarse | Medium |
| Ease | Write a function | Configure compiler | Write schedules |
| Best for | Custom ops | TPU workloads | Deployment |

Triton = "I want to write this one kernel myself"
XLA/TVM = "Compile my whole model"

**Q: Should I use this in production?**

For learning, yes. For production, probably use:
- `torch.compile()` with Triton backend (automatic fusion)
- Or vendor libraries when available

## Structure

```
gemm_silu/
├── core/kernels/     # @triton.jit kernels (no torch imports)
├── adapters/pytorch/ # torch wrappers
├── tests/
├── bench.py
└── README.md
```

Hexagonal-ish: core has no framework deps.

## Test

```bash
pytest tests/ -v
TRITON_INTERPRET=1 pytest tests/ -v  # CPU mode (no GPU)
```

## License

MIT
