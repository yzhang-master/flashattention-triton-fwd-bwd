"""PyTorch references used only for correctness checks and benchmarks."""

import torch
import torch.nn.functional as F


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """Evaluate scaled dot-product attention in fp32."""
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=causal, scale=scale
    ).to(q.dtype)
