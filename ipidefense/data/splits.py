"""Per-seed host slices of the shuffled Alpaca hosts, with leak-safe pairing for adapters."""
from __future__ import annotations

from ipidefense.arms import ADAPTER_SEED
from ipidefense.data.hosts import HostTask, load_en_hosts

# Slices of load_en_hosts(n, seed): it shuffles all usable rows and keeps the first n, so the
# boundaries are the same for every n.
SLICES: dict[str, tuple[int, int]] = {
    "train": (0, 1500),
    "security": (1500, 1650),
    "benign": (1650, 1750),
    "adaptive": (1750, 1850),
    "format": (1850, 2000),
    "pilot": (2000, 2030),
    "fresh": (2030, 2530),
    "lrsel": (2530, 2630),  # DPO learning-rate selection only, never in reported numbers
}


def hosts(seed: int, split: str) -> tuple[HostTask, ...]:
    lo, hi = SLICES[split]
    return load_en_hosts(hi, seed)[lo:hi]


def hosts_for(arm: str, split: str, seed: int | None = None) -> tuple[HostTask, ...]:
    """Hosts of a split on the arm's own training seed.

    Undefended arms take `seed` (required), so they share the hosts of the defended arm
    they are compared with.
    """
    own = ADAPTER_SEED[arm]
    if own is None:
        if seed is None:
            raise ValueError(f"{arm}: pass the seed of the arm it is paired with")
        return hosts(seed, split)
    if seed is not None and seed != own:
        raise ValueError(f"{arm} was trained on seed {own}; refusing seed-{seed} hosts (leakage)")
    return hosts(own, split)
