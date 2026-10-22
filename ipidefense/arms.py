import json
from collections.abc import Iterable
from pathlib import Path

from ipidefense.paths import MODELS

# name -> (backbone, adapter dir or "", data inline)
ADAPTERS: dict[str, tuple[str, str, bool]] = {
    "undef": ("qwen", "", False),
    "def-s0": ("qwen", "defense-qwen-seed0", False),
    "def-s1": ("qwen", "defense-qwen-seed1", False),
    "def-s2": ("qwen", "defense-qwen-seed2", False),
    "hard-s0": ("qwen", "defense-qwen-seed0-hardened", False),
    "hard-s1": ("qwen", "defense-qwen-seed1-hardened", False),
    "inline-s0": ("qwen", "defense-qwen-seed0-inline", True),
    "undef-inline": ("qwen", "", True),
}

# adapters are only evaluated on slices of their own training seed
ADAPTER_SEED: dict[str, int | None] = {
    "undef": None, "undef-inline": None,
    "def-s0": 0, "def-s1": 1, "def-s2": 2, "hard-s0": 0, "hard-s1": 1,
    "inline-s0": 0,
}

# backbone -> base model dir under MODELS
BASE_DIRS: dict[str, str] = {"qwen": "Qwen2.5-7B-Instruct",
                             "sealion": "Llama-SEA-LION-v3-8B-IT",
                             "mistral": "Mistral-7B-Instruct-v0.3"}

# other backbones at seed 0, Mistral data is always inline
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
    """Single backbone shared by a set of arms."""
    bases = {ADAPTERS[n][0] for n in names}
    if len(bases) > 1:
        raise ValueError(f"arms of different backbones cannot share one model load: "
                         f"{sorted(bases)}")
    return bases.pop() if bases else default


def check_adapter_base(name: str) -> None:
    """Refuse an adapter trained on another base model."""
    base, adir, _ = ADAPTERS[name]
    if not adir:
        return
    path = MODELS / adir / "adapter_config.json"
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path} is missing (adapter not trained yet?)")
    trained_on = json.loads(path.read_text(encoding="utf-8")).get("base_model_name_or_path", "")
    # configs may store Windows paths
    if trained_on.replace("\\", "/").rstrip("/").split("/")[-1] != BASE_DIRS[base]:
        raise ValueError(f"{name}: adapter trained on {trained_on!r}, registry backbone is "
                         f"{base} ({BASE_DIRS[base]})")
