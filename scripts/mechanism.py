import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # runs without installing

from ipidefense.runner import Run, Step, main, root_arg

SWAP = "ipidefense.experiments.channel_swap"
MASK = "ipidefense.experiments.lora_mask"

M = root_arg("IPI_MODELS", "models")
OUT = f"{root_arg('IPI_RUNS', 'runs')}/channel_swap_b32"


def swap(run: Run) -> None:
    # all families on seed-0 arms, the override family on seeds 1-2
    run.py(SWAP, "run", "--arms", "undef", "def-s0", "hard-s0", "inline-s0", "undef-inline",
           "--seeds", "0", "--batch", "32", "--out-dir", OUT)
    run.py(SWAP, "run", "--arms", "undef", "def-s1", "def-s2", "--seeds", "1", "2",
           "--families", "F-OVR", "--batch", "32", "--out-dir", OUT)
    run.py(SWAP, "summarize",
           "--arms", "undef", "def-s0", "hard-s0", "inline-s0", "undef-inline", "def-s1", "def-s2",
           "--seeds", "0", "1", "2", "--batch", "32", "--out-dir", OUT)


def mask(run: Run) -> None:
    # sanity gates first, run also writes the summary
    run.py(MASK, "run")
    run.py(MASK, "summarize")


def probe(run: Run) -> None:
    run.py("ipidefense.experiments.probe", "--lora-dir", f"{M}/defense-qwen-seed0", "--seed", "0")


STEPS = [
    Step("swap", "channel swap: the same instruction in the user turn or in the data channel",
         "Channel swap", swap),
    Step("mask", "the LoRA adapter switched off on token spans",
         "Causal localization (adapter switched off on token spans)", mask),
    Step("probe", "representation probe on the seed-0 adapter",
         "Representation probe and mechanism figure", probe),
]


if __name__ == "__main__":
    main(__file__, STEPS)
