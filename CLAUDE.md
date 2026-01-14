# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
source .venv/bin/activate
pytest tests/ -v                              # run tests
pytest tests/test_kernels.py::test_gemm_silu  # single test
TRITON_INTERPRET=1 pytest tests/ -v           # no GPU
```

## Structure

```
core/kernels/    # triton @jit kernels (no torch imports)
adapters/pytorch # torch wrappers
tests/
```

`core/` has zero torch imports. Adapters import from core.

## Triton

```python
# accumulate fp32, fuse silu before store
acc = acc * tl.sigmoid(acc)
tl.store(c_ptrs, acc.to(tl.float16), mask=mask)
```

Block sizes: multiples of 32. See `CONFIGS` in kernel file.
