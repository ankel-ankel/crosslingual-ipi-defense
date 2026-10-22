"""Per-host first-success steps of the white-box GCG runs (LaTeX rows on stdout, a check on stderr).

  gcg_table.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ipidefense.paths import RUNS

GCG = RUNS / "gcg"


@dataclass(frozen=True)
class Column:
    label: str
    path: Path
    n_hosts: int


SEED0 = (Column("Undefended", GCG / "hard_s0" / "undef.jsonl", 40),
         Column("Hardened", GCG / "hard_s0" / "hard-s0.jsonl", 40),
         Column("Initial", GCG / "def_s0" / "def-s0.jsonl", 30))
SEED1 = (Column("Undefended", GCG / "hard_s1" / "undef.jsonl", 30),
         Column("Hardened", GCG / "hard_s1" / "hard-s1.jsonl", 30))


def load(col: Column) -> dict[int, dict] | None:
    # host_index -> record, or None while the transcript is absent or incomplete
    if not col.path.exists():
        return None
    recs = {}
    for line in col.path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["host_index"]] = r
    return recs if set(range(col.n_hosts)) <= set(recs) else None


def cell(rec: dict) -> str:
    # "<first witness-first step>/<first begins-with step>"; "--" = not met, "TF" = technical
    # failure (counted as an attack failure)
    if rec.get("technical_failure"):
        return "TF"
    fs = rec.get("first_step") or {}
    return "/".join("--" if fs.get(k) is None else str(fs[k]) for k in ("wf", "starts_with"))


def rows(cols: tuple[Column, ...]) -> tuple[list[str], dict]:
    # a column without a complete transcript prints \PENDING{}; a host beyond a column's
    # attacked range prints an empty cell
    data = {c.label: load(c) for c in cols}
    n = max(c.n_hosts for c in cols)
    out = []
    for i in range(n):
        cells = [str(i)]
        for c in cols:
            recs = data[c.label]
            cells.append("" if i >= c.n_hosts else r"\PENDING{}" if recs is None
                         else cell(recs[i]))
        out.append(" & ".join(cells) + r"\\")
    check = {c.label: None if data[c.label] is None else {
        "wf_k": sum(data[c.label][i].get("first_step", {}).get("wf") is not None
                    and not data[c.label][i].get("technical_failure")
                    for i in range(c.n_hosts)),
        "n": c.n_hosts,
        "tf": sum(bool(data[c.label][i].get("technical_failure")) for i in range(c.n_hosts)),
        "wf_without_starts_with": [i for i in range(c.n_hosts)
                                   if (data[c.label][i].get("first_step") or {}).get("wf")
                                   is not None and (data[c.label][i].get("first_step") or {})
                                   .get("starts_with") is None]}
        for c in cols}
    return out, check


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    for name, cols in (("seed-0 hosts", SEED0), ("seed-1 hosts", SEED1)):
        lines, check = rows(cols)
        print(f"% {name}: " + " & ".join(["Host"] + [c.label for c in cols]))
        print("\n".join(lines))
        print(json.dumps({name: check}), file=sys.stderr)


if __name__ == "__main__":
    main()
