import pytest
import torch

from flashattention_triton import attention_forward
from flashattention_triton.reference import attention_reference


@pytest.mark.parametrize("shape", [(1, 1, 16, 16), (1, 2, 17, 32), (2, 2, 33, 16)])
@pytest.mark.parametrize("causal", [False, True])
def test_forward_matches_fp32_sdpa(shape: tuple[int, ...], causal: bool, device: str) -> None:
    torch.manual_seed(11)
    q, k, v = (torch.randn(shape, device=device) for _ in range(3))
    out, lse = attention_forward(q, k, v, causal=causal)
    expected = attention_reference(q, k, v, causal=causal)
    torch.testing.assert_close(out, expected, atol=3e-5, rtol=3e-5)

    scores = q @ k.transpose(-1, -2) * shape[-1] ** -0.5
    if causal:
        mask = torch.ones(shape[-2], shape[-2], dtype=torch.bool, device=device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    torch.testing.assert_close(lse, scores.logsumexp(-1), atol=2e-5, rtol=2e-5)


def test_causal_future_tokens_cannot_change_past(device: str) -> None:
    torch.manual_seed(12)
    q, k, v = (torch.randn(1, 1, 19, 16, device=device) for _ in range(3))
    baseline, _ = attention_forward(q, k, v, causal=True)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 8:] = 100.0
    v2[:, :, 8:] = -50.0
    changed, _ = attention_forward(q, k2, v2, causal=True)
    torch.testing.assert_close(changed[:, :, :8], baseline[:, :, :8], atol=0, rtol=0)


def test_noncontiguous_inputs_and_custom_scale(device: str) -> None:
    torch.manual_seed(13)
    q, k, v = (
        torch.randn(1, 2, 32, 20, device=device).transpose(-1, -2) for _ in range(3)
    )
    out, _ = attention_forward(q, k, v, causal=True, scale=0.2)
    expected = attention_reference(q, k, v, causal=True, scale=0.2)
    torch.testing.assert_close(out, expected, atol=3e-5, rtol=3e-5)


def test_rejects_mismatched_shapes(device: str) -> None:
    q = torch.randn(1, 1, 16, 16, device=device)
    with pytest.raises(ValueError, match="identical"):
        attention_forward(q, q, q[:, :, :8])
