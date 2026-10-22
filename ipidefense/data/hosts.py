"""Alpaca host tasks (instruction + input) that carry the injections."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import cache

from ipidefense.paths import DATA

ALPACA = DATA / "alpaca_cleaned.json"


@dataclass(frozen=True)
class HostTask:
    instruction: str
    data: str


def _rows() -> list[dict]:
    if not ALPACA.exists():
        raise FileNotFoundError(f"{ALPACA} is missing: run scripts/prepare_data.py")
    return json.loads(ALPACA.read_text(encoding="utf-8"))


@cache
def load_en_hosts(n: int = 200, seed: int = 0) -> tuple[HostTask, ...]:
    usable = [
        r
        for r in _rows()
        if (r.get("input") or "").strip() and (r.get("instruction") or "").strip()
    ]
    random.Random(seed).shuffle(usable)
    return tuple(HostTask(r["instruction"].strip(), r["input"].strip()) for r in usable[:n])
