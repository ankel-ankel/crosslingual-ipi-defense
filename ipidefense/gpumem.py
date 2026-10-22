from typing import Any

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class ReleaseAfterPrefill(StoppingCriteria):
    """Stopping criterion that only frees the cache once, after the prefill."""

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
    """Extra generate() kwargs that free the cache after the prefill."""
    if not torch.cuda.is_available():
        return {}
    return {"stopping_criteria": StoppingCriteriaList([ReleaseAfterPrefill()])}


def release() -> None:
    """Return unused cached blocks to the driver (no-op without CUDA)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
