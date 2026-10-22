"""The post-prefill criterion never stops a sequence and releases the cache exactly once; without
CUDA nothing is added to generate()."""
from __future__ import annotations

import torch

from ipidefense import gpumem as gm


def test_criterion_never_stops_and_releases_once(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(gm, "release", lambda: calls.append(1))
    crit = gm.ReleaseAfterPrefill()
    ids, scores = torch.zeros(3, 5, dtype=torch.long), torch.zeros(3, 10)
    for _ in range(4):
        out = crit(ids, scores)
        assert out.dtype == torch.bool and out.shape == (3,) and not out.any()
    assert calls == [1]


def test_no_extra_generate_kwargs_without_cuda() -> None:
    assert not torch.cuda.is_available()
    assert gm.lean_kwargs() == {}
    gm.release()  # no-op on CPU
