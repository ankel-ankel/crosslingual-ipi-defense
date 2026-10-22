import io
import re
from pathlib import Path
from typing import Any, TextIO

from ipidefense.training.sets import MODELS_DIR

RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def adapter_dir(run_name: str, overwrite: bool = False, models: Path = MODELS_DIR) -> Path:
    """MODELS/<run_name>, never over an existing adapter unless overwrite."""
    if not RUN_NAME.match(run_name):
        raise ValueError(f"invalid run name {run_name!r}")
    out = (models / run_name).resolve()
    if out.parent != models.resolve():
        raise ValueError(f"{out} is not directly under {models}")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    return out


def select(n: int, limit: int | None, longest: bool, lengths: list[int]) -> list[int]:
    """Record indices to train on: all, the first `limit`, or the `limit` longest."""
    if limit is None:
        return list(range(n))
    if longest:
        return sorted(sorted(range(n), key=lambda k: -lengths[k])[:limit])
    return list(range(min(limit, n)))


def length_stats(lens: list[tuple[int, int]], max_len: int) -> dict[str, Any]:
    """Length stats of (prompt, full) token pairs, raise if one has no supervision."""
    unsupervised = [k for k, (p, _) in enumerate(lens) if p >= max_len]
    if unsupervised:
        raise ValueError(f"{len(unsupervised)} sequences have no supervised token within "
                         f"max_len={max_len} (indices {unsupervised[:5]}); rebuild the data")
    fulls = [f for _, f in lens]
    return {
        "n_sequences": len(lens),
        "prompt_tokens_max": max(p for p, _ in lens),
        "seq_tokens_mean": round(sum(fulls) / len(fulls), 1),
        "seq_tokens_max": max(fulls),
        "n_truncated": sum(f > max_len for f in fulls),
        "supervised_tokens": sum(min(f, max_len) - p for p, f in lens),
    }


def lora_settings(model: Any) -> dict[str, Any]:
    cfg = model.peft_config["default"]
    return {"r": cfg.r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout,
            "targets": sorted(cfg.target_modules)}


class LossTee(io.TextIOBase):
    """Forwards writes and keeps the loss lines of the training loop."""

    PATTERN = re.compile(r"epoch (\d+) step (\d+) loss (\S+)")

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self._stream = stream
        self.losses: list[dict[str, float]] = []

    def write(self, text: str) -> int:
        self._stream.write(text)
        for m in self.PATTERN.finditer(text):
            self.losses.append({"epoch": int(m[1]), "step": int(m[2]), "loss": float(m[3])})
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
