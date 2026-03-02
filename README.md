# flashattention-triton-fwd-bwd

> **Era:** 2022 · technique from 2022, reimplemented in 2026 with Triton 3.x and PyTorch 2.x.

Scaffold for a tiled FlashAttention implementation with Triton forward and backward kernels,
causal masking, autograd support, CPU-interpreter correctness tests, and a GPU benchmark harness.

## Results

GPU TFLOPS: **pending**. No GPU measurements have been made.

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check .
pytest
```

## Limitations

The implementation and benchmark harness land on subsequent feature branches. GPU performance
has not been measured.

## License

MIT
