from ipidefense.arms import ADAPTER_SEED
from ipidefense.data.hosts import HostTask, load_en_hosts

# slices of load_en_hosts(n, seed), the same for every n
SLICES: dict[str, tuple[int, int]] = {
    "train": (0, 1500),
    "security": (1500, 1650),
    "benign": (1650, 1750),
    "adaptive": (1750, 1850),
    "format": (1850, 2000),
    "pilot": (2000, 2030),
    "fresh": (2030, 2530),
    "lrsel": (2530, 2630),  # only for DPO learning-rate selection
}


def hosts(seed: int, split: str) -> tuple[HostTask, ...]:
    lo, hi = SLICES[split]
    return load_en_hosts(hi, seed)[lo:hi]


def hosts_for(arm: str, split: str, seed: int | None = None) -> tuple[HostTask, ...]:
    """Hosts of a split on the arm's training seed (undefended arms need seed)."""
    own = ADAPTER_SEED[arm]
    if own is None:
        if seed is None:
            raise ValueError(f"{arm}: pass the seed of the arm it is paired with")
        return hosts(seed, split)
    if seed is not None and seed != own:
        raise ValueError(f"{arm} was trained on seed {own}; refusing seed-{seed} hosts (leakage)")
    return hosts(own, split)
