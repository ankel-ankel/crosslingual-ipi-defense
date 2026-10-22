"""Languages of the full suite (ml3) from the language-gate summary (ml2).

Usage: language_plan.py [summary.json] [--renderer llm] -> MAIN / EXTRA lines
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ipidefense.paths import RUNS

BASE = ("en", "es", "vi", "fr", "de", "zh")
SLOT_A, SLOT_B = ("ko", "ar", "th"), ("bn", "my", "km")
DEFAULT = RUNS / "suite" / "ml2" / "summary.json"


# First passer per slot (A: ko > ar > th, B: bn > my > km); with no slot-B passer the second
# slot-A passer is added instead; up to 2 other passers become seed-0-only rows.
def plan(summary: dict, renderer: str | None = None) -> tuple[list[str], list[str]]:
    """(ml3 languages, seed-0-only extra languages) from an ml2 summary."""
    if summary.get("exp") != "ml2":
        raise ValueError("not an ml2 summary")
    if renderer is not None and summary.get("renderer") != renderer:
        raise ValueError(f"ml2 summary renderer {summary.get('renderer')!r} != {renderer!r}")
    table = summary["decision"]["table"]
    missing = [x for x in ("en",) + SLOT_A + SLOT_B if x not in table]
    if missing:
        raise ValueError(f"ml2 table lacks {missing}")
    pass_a = [x for x in SLOT_A if table[x]["pass"]]
    pass_b = [x for x in SLOT_B if table[x]["pass"]]
    chosen = pass_a[:1] + (pass_b[:1] if pass_b else pass_a[1:2])
    extra = [x for x in pass_a + pass_b if x not in chosen][:2]
    return list(BASE) + chosen, extra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("summary", nargs="?", type=Path, default=DEFAULT)
    ap.add_argument("--renderer", default="llm")
    args = ap.parse_args()
    langs, extra = plan(json.loads(args.summary.read_text(encoding="utf-8")), args.renderer)
    print("MAIN " + " ".join(langs))
    print("EXTRA " + " ".join(extra))


if __name__ == "__main__":
    main()
