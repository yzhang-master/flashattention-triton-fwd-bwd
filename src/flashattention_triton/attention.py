"""Memory-efficient tiled attention implemented with Triton kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_forward_kernel(
    Q,
    K,
    V,
    OUT,
    LSE,
    N,
    D: tl.constexpr,
    scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < N
    base = bh * N * D
    q = tl.load(
        Q + base + offs_m[:, None] * D + offs_d[None, :],
        mask=mask_m[:, None] & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    hi = tl.minimum((pid_m + 1) * BLOCK_M, N) if CAUSAL else N

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        k = tl.load(
            K + base + offs_n[:, None] * D + offs_d[None, :],
            mask=mask_n[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        scores = tl.where(valid, scores, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(
            V + base + offs_n[:, None] * D + offs_d[None, :],
            mask=mask_n[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(
        OUT + base + offs_m[:, None] * D + offs_d[None, :],
        acc.to(OUT.dtype.element_ty),
        mask=mask_m[:, None] & (offs_d[None, :] < D),
    )
    tl.store(LSE + bh * N + offs_m, m_i + tl.log(l_i), mask=mask_m)


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k and v must have identical [batch, heads, tokens, dim] shapes")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k and v must have the same dtype")
    if q.shape[-1] < 16 or q.shape[-1] > 128:
        raise ValueError("head dimension must be between 16 and 128")
    if q.shape[-2] == 0:
        raise ValueError("sequence length must be positive")


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return attention output and fp32 log-sum-exp statistics."""
    _validate_inputs(q, k, v)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, heads, n, d = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((batch, heads, n), dtype=torch.float32, device=q.device)
    block_m = 16
    block_n = 16
    block_d = triton.next_power_of_2(d)
    sm_scale = d**-0.5 if scale is None else scale
    _attention_forward_kernel[(triton.cdiv(n, block_m), batch * heads)](
        q,
        k,
        v,
        out,
        lse,
        n,
        d,
        sm_scale,
        CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out, lse
