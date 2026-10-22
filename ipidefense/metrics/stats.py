"""Binomial bounds: one-sided Clopper-Pearson upper bound and Wilson interval."""
from __future__ import annotations

import math


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) exact upper bound for a binomial proportion."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(1 - alpha, k + 1, n - k))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))
