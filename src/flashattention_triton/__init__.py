"""Tiled FlashAttention forward and backward in Triton."""

from .attention import attention_forward

__version__ = "0.1.0"

__all__ = ["__version__", "attention_forward"]
