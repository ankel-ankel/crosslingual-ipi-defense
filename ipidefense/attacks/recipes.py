import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ipidefense.attacks.templates import TRAIN_TEMPLATES, TRAIN_WITNESSES
from ipidefense.attacks.witnesses import WitnessInjection
from ipidefense.data.hosts import HostTask
from ipidefense.data.splits import SLICES
from ipidefense.data.splits import hosts as split_hosts
from ipidefense.paths import DATA, RUNS
from ipidefense.rendering.literal import render_inj_literal

DATA_ROOT = DATA / "baselines"
RENDER_CACHE = DATA_ROOT / "render_cache.json"
RENDER_AUDIT = RUNS / "render_audit.json"

N_TRAIN = SLICES["train"][1]  # training hosts per seed
CLEAN_FRAC = 0.6  # share of clean training examples
LANGS = ("en", "es", "vi")  # training attack languages


# hosts
def train_hosts(seed: int) -> tuple[HostTask, ...]:
    return split_hosts(seed, "train")


# rendering
class CachedTranslator:
    """Read-only translator backed by the render cache."""

    def __init__(self, table: dict[str, dict[str, str]]) -> None:
        self._table = {lang: dict(m) for lang, m in table.items()}

    @classmethod
    def load(cls, path: Path = RENDER_CACHE) -> "CachedTranslator":
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: it ships with the repository "
                                    "(rebuilt by the render command of training/sets.py)")
        return cls(json.loads(path.read_text(encoding="utf-8"))["table"])

    def translate(self, text: str, target_lang: str) -> str:
        text = text.strip()
        if not text:
            return text
        try:
            return self._table[target_lang][text]
        except KeyError:
            raise KeyError(f"no cached {target_lang} translation for {text!r}") from None


def wrapper_segments(templates: Iterable[str]) -> list[str]:
    """Wrapper segments that render_inj_literal sends to NLLB."""
    segs: dict[str, None] = {}
    for template in templates:
        inj = WitnessInjection(template, "W")
        for seg in (inj.wrapper_prefix(), inj.wrapper_suffix()):
            if seg.strip():
                segs.setdefault(seg.strip(), None)
    return list(segs)


def audit_render(tr: Any, audit: Path = RENDER_AUDIT, langs: Iterable[str] = ("es", "vi")) -> dict:
    """Compare cached renderings with the render audit."""
    if not audit.exists():
        return {"available": False}
    langs = tuple(langs)
    rows = [
        r
        for r in json.loads(audit.read_text(encoding="utf-8"))["rows"]
        if r["pool"] in ("train", "eval") and r["lang"] in langs
    ]
    bad = [
        r
        for r in rows
        if render_inj_literal(tr, WitnessInjection(r["template"], r["witness"]), r["lang"])
        != r["old"]
    ]
    return {
        "available": True,
        "n_rows": len(rows),
        "n_mismatch": len(bad),
        "mismatches": [{k: r[k] for k in ("pool", "template", "lang", "old")} for r in bad[:10]],
    }


# recipes
def _witness(tr: Any, template: str, witness: str, lang: str) -> str:
    return render_inj_literal(tr, WitnessInjection(template, witness), lang)


def plan_a0(
    hosts: tuple[HostTask, ...], seed: int, tr: Any, skip: frozenset[int] = frozenset()
) -> list[dict]:
    """The defense's data mix. A skipped host consumes no random draw."""
    rng = random.Random(seed)
    out: list[dict] = []
    for i, h in enumerate(hosts):
        if i in skip:
            continue
        if rng.random() < CLEAN_FRAC:
            out.append({"host": i, "kind": "clean", "data": h.data})
            continue
        templ, wit = rng.choice(TRAIN_TEMPLATES), rng.choice(TRAIN_WITNESSES)
        lang = LANGS[i % len(LANGS)]
        inj = _witness(tr, templ, wit, lang)
        out.append({"host": i, "kind": "witness", "data": f"{h.data}\n{inj}", "template": templ,
                    "witness": wit, "lang": lang, "injection": inj})
    return out
