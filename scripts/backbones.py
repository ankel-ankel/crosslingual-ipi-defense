"""SEA-LION and Mistral: IFEval cost, target census, cap-512 / cap-160 retraining, security
checks, IFEval breakdown and decision.

Run after main_results.py (adapters, IFEval reports) and languages.py (render audit).

The published adapters defense-sealion-seed0 and defense-mistral-seed0-inline and their unbatched
IFEval reports come from the steps sealion and mistral of main_results.py; step train checks
data/baselines/render_cache.json against the render audit of languages.py (step render-audit).
The Qwen census reads data/baselines/seed0/targets.jsonl. Batched IFEval logs go to
runs/ifeval_logged_b16.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # runs without installing

from ipidefense.runner import Run, Step, main  # noqa: E402

BACKBONES = "ipidefense.experiments.backbones"
IFEVAL_LOGGED = "ipidefense.evaluation.ifeval_logged"


def ifeval(run: Run) -> None:
    run.py(IFEVAL_LOGGED, "--batch", "16", "--arms", "sealion-undef", "sealion-def-s0")
    run.py(IFEVAL_LOGGED, "--batch", "16", "--arms", "mistral-undef", "mistral-def-s0")


def targets(run: Run) -> None:
    # the targets are shipped in data/backbones: these runs verify them and write the reports
    # (move the files aside to generate them again)
    for bb in ("sealion", "mistral"):
        run.py(BACKBONES, "targets", "--backbone", bb, "--cap", "512")
        run.py(BACKBONES, "targets", "--backbone", bb, "--cap", "160")
    run.py(BACKBONES, "census")


def train(run: Run) -> None:
    for bb in ("sealion", "mistral"):
        run.py(BACKBONES, "train", "--backbone", bb, "--cfg", "A")
        run.py(BACKBONES, "train", "--backbone", bb, "--cfg", "B")
    run.py(IFEVAL_LOGGED, "--batch", "16", "--arms", "sealion-cfgA-s0", "sealion-cfgB-s0")
    run.py(IFEVAL_LOGGED, "--batch", "16", "--arms", "mistral-cfgA-s0", "mistral-cfgB-s0")


def sec(run: Run) -> None:
    run.py(BACKBONES, "sec", "--backbone", "sealion")
    run.py(BACKBONES, "sec", "--backbone", "mistral")
    run.py(BACKBONES, "sec", "--backbone", "sealion", "--six")
    run.py(BACKBONES, "sec", "--backbone", "mistral", "--six")
    run.py(BACKBONES, "sec", "--backbone", "mistral", "--six",
           "--arms", "mistral-undef", "mistral-def-s0")
    run.py(BACKBONES, "sec", "--backbone", "sealion", "--six",
           "--arms", "sealion-undef", "sealion-def-s0")


def report(run: Run) -> None:
    run.py(BACKBONES, "breakdown")
    run.py(BACKBONES, "decide")


PAPER = "Cross-backbone table, configurations A / B and their IFEval breakdown"
STEPS = [
    Step("ifeval", "batched IFEval of the undefended backbones and the published adapters",
         PAPER, ifeval),
    Step("targets", "frozen-base targets with 512 and 160 new tokens, then the census", PAPER,
         targets),
    Step("train", "configuration A (cap 512, lr 1e-4) and B (cap 160, lr 5e-5), then their IFEval",
         PAPER, train),
    Step("sec", "security: en/es/vi for every arm, six languages for the best configuration",
         PAPER, sec),
    Step("report", "IFEval breakdown per arm and the decision", PAPER, report),
]


if __name__ == "__main__":
    main(__file__, __doc__, STEPS)
