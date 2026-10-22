"""Adapter registry: arm name -> backbone, adapter directory under MODELS, inline data format."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ipidefense.paths import MODELS

# name -> (backbone, adapter directory under MODELS or "" if undefended, data inline in user turn)
ADAPTERS: dict[str, tuple[str, str, bool]] = {
    "undef": ("qwen", "", False),
    "def-s0": ("qwen", "defense-qwen-seed0", False),
    "def-s1": ("qwen", "defense-qwen-seed1", False),
    "def-s2": ("qwen", "defense-qwen-seed2", False),
    "hard-s0": ("qwen", "defense-qwen-seed0-hardened", False),
    "hard-s1": ("qwen", "defense-qwen-seed1-hardened", False),
    "inline-s0": ("qwen", "defense-qwen-seed0-inline", True),
    "undef-inline": ("qwen", "", True),
    "refuse-s0": ("qwen", "defense-qwen-seed0-refuse", False),
}

# Training seed of each adapter. Host slices depend on the seed, so an adapter is only
# evaluated on slices of its own seed (another seed's slices overlap its training hosts).
ADAPTER_SEED: dict[str, int | None] = {
    "undef": None, "undef-inline": None,
    "def-s0": 0, "def-s1": 1, "def-s2": 2, "hard-s0": 0, "hard-s1": 1,
    "inline-s0": 0, "refuse-s0": 0,
}

# backbone -> base-model directory under MODELS
BASE_DIRS: dict[str, str] = {"qwen": "Qwen2.5-7B-Instruct",
                             "sealion": "Llama-SEA-LION-v3-8B-IT",
                             "mistral": "Mistral-7B-Instruct-v0.3"}

# Other backbones, seed 0: published adapter and inline flag. Mistral's chat template has no
# input role, so every Mistral arm has the data inline. Configurations A and B retrain them.
BACKBONES: dict[str, tuple[str, bool]] = {
    "sealion": ("defense-sealion-seed0", False),
    "mistral": ("defense-mistral-seed0-inline", True),
}
BACKBONE_CONFIGS = ("A", "B")


def backbone_adapter_dir(backbone: str, cfg: str) -> Path:
    return MODELS / f"{backbone}-cfg{cfg}-seed0"


for _bb, (_pub, _inline) in BACKBONES.items():
    ADAPTERS[f"{_bb}-undef"] = (_bb, "", _inline)
    ADAPTER_SEED[f"{_bb}-undef"] = None
    ADAPTERS[f"{_bb}-def-s0"] = (_bb, _pub, _inline)
    ADAPTER_SEED[f"{_bb}-def-s0"] = 0
    for _cfg in BACKBONE_CONFIGS:
        ADAPTERS[f"{_bb}-cfg{_cfg}-s0"] = (_bb, backbone_adapter_dir(_bb, _cfg).name, _inline)
        ADAPTER_SEED[f"{_bb}-cfg{_cfg}-s0"] = 0


def arms_base(names: Iterable[str], default: str = "qwen") -> str:
    """The one backbone of a set of arms (one model load serves them all)."""
    bases = {ADAPTERS[n][0] for n in names}
    if len(bases) > 1:
        raise ValueError(f"arms of different backbones cannot share one model load: "
                         f"{sorted(bases)}")
    return bases.pop() if bases else default


def check_adapter_base(name: str) -> None:
    """Refuse an adapter trained on another base model than its registry backbone.

    SEA-LION and Mistral LoRA tensors have identical shapes, so a mix-up would load silently.
    """
    base, adir, _ = ADAPTERS[name]
    if not adir:
        return
    path = MODELS / adir / "adapter_config.json"
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path} is missing (adapter not trained yet?)")
    trained_on = json.loads(path.read_text(encoding="utf-8")).get("base_model_name_or_path", "")
    # adapter configs may store Windows paths; compare the last path component only
    if trained_on.replace("\\", "/").rstrip("/").split("/")[-1] != BASE_DIRS[base]:
        raise ValueError(f"{name}: adapter trained on {trained_on!r}, registry backbone is "
                         f"{base} ({BASE_DIRS[base]})")
