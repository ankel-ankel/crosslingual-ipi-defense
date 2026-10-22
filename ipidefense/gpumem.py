"""Release the CUDA allocator cache after the prefill (memory placement only; tokens unchanged).
On Windows (WDDM) an allocation beyond VRAM silently spills to system memory and the cache keeps
reusing those blocks, so every later decode step would page over PCIe.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class ReleaseAfterPrefill(StoppingCriteria):
    """Never stops a sequence; its first call (right after the prefill) releases the cache."""

    def __init__(self) -> None:
        super().__init__()
        self.pending = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs: Any) -> torch.BoolTensor:
        if self.pending:
            self.pending = False
            release()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def lean_kwargs() -> dict:
    """Extra generate() kwargs: a fresh one-shot release after the prefill (CUDA only)."""
    if not torch.cuda.is_available():
        return {}
    return {"stopping_criteria": StoppingCriteriaList([ReleaseAfterPrefill()])}


def release() -> None:
    """Return unused cached blocks to the driver (no-op without CUDA)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
