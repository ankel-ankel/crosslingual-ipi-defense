"""White-box GCG: cache check, undefended sanity run, per-host attacks, universal suffix and its
evaluations, the paired McNemar test of def-s0 against hard-s0 and the per-host table. Every run
resumes from its transcript.

Run after main_results.py: it attacks its adapters def-s0, hard-s0 and hard-s1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # runs without installing

from ipidefense.runner import Run, Step, main, root_arg  # noqa: E402

GCG_RUN = "ipidefense.experiments.gcg_run"
RUNS = root_arg("IPI_RUNS", "runs")
# the prefix-cache loss misses the 0.02-nat check in bf16, so the attacks use the full forward
GCG = ["--steps", "500", "--width", "512", "--topk", "256", "--suffix-len", "20",
       "--check-every", "20", "--batch", "128", "--no-cache"]


def check(run: Run) -> None:
    run.py("ipidefense.experiments.gcg_cache_check")
    run.py(GCG_RUN, "--adapter", "undef", "--seed", "0", "--hosts", "10", "--steps", "20",
           "--out", "gcg/undef_sanity")


def per_host(run: Run) -> None:
    # the hardened runs also attack the undefended model on the same hosts
    run.py(GCG_RUN, "--adapter", "hard-s0", "--hosts", "40", *GCG, "--compare-undefended",
           "--out", "gcg/hard_s0")
    run.py(GCG_RUN, "--adapter", "def-s0", "--hosts", "30", *GCG, "--out", "gcg/def_s0")
    run.py(GCG_RUN, "--adapter", "hard-s1", "--hosts", "30", *GCG, "--compare-undefended",
           "--out", "gcg/hard_s1")


def universal(run: Run) -> None:
    suffix = f"{RUNS}/gcg/universal/universal.json"
    run.py(GCG_RUN, "--adapter", "hard-s0", "--universal", "--universal-hosts", "10",
           "--split", "fresh", *GCG, "--out", "gcg/universal")
    run.require(suffix)
    for arm in ("hard-s0", "def-s0", "hard-s1"):
        run.py(GCG_RUN, "--adapter", arm, "--hosts", "40", "--eval-suffix", suffix,
               "--out", f"gcg/universal_eval_{arm.replace('-', '_')}")
    run.py(GCG_RUN, "--adapter", "undef", "--seed", "0", "--hosts", "40", "--eval-suffix", suffix,
           "--out", "gcg/universal_eval_undef")


def report(run: Run) -> None:
    # paired exact McNemar of def-s0 against hardened seed 0 on hosts 0-29 (CPU)
    run.py("ipidefense.experiments.gcg_mcnemar", "gcg/hard_s0/hard-s0.jsonl",
           "gcg/def_s0/def-s0.jsonl", "--hosts", "30",
           "--out", "gcg/def_s0/mcnemar_vs_hard_s0.json")
    run.py("ipidefense.experiments.gcg_table")


STEPS = [
    Step("check", "prefix-cache check and a short undefended sanity run",
         "White-box GCG table and its checks", check),
    Step("per-host", "per-host attacks on hard-s0, def-s0 and hard-s1",
         "White-box GCG table", per_host),
    Step("universal", "universal suffix on 10 fresh hosts, evaluated unchanged on 40 hosts",
         "White-box GCG table", universal),
    Step("report", "paired McNemar test of def-s0 against hard-s0, then the per-host table",
         "White-box GCG table and its checks; per-host GCG table", report),
]


if __name__ == "__main__":
    main(__file__, __doc__, STEPS)
