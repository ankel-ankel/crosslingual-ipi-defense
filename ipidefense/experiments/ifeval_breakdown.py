import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ipidefense.paths import DATA, RUNS
from ipidefense.utils import write_json

SUBSETS = ("CS", "IL", "SF", "OTHER")
LABELS = DATA / "ifeval_subsets.json"
N_PROMPTS = 541
EOS_BAND = (140, 180)
MID_SENTENCE_END = set(",;:-")
N_LONG = 15  # long-answer instructions, dropped in the sensitivity run
IFEVAL_DIR = RUNS / "ifeval_logged"
UNDEF = "undef"
ARMS = ("def-s0", "def-s1", "def-s2")
N_BOOT = 10_000


def load_arm(path: Path) -> dict[int, dict]:
    recs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["idx"]] = r
    if len(recs) != N_PROMPTS:
        raise SystemExit(f"{path}: {len(recs)} of {N_PROMPTS} prompts")
    return recs


def mid_sentence(resp: str) -> bool:
    s = resp.rstrip()
    return bool(s) and (s[-1].isalnum() or s[-1] in MID_SENTENCE_END)


def long_mask(labels: list[dict], rows: list[dict] | None) -> list[list[bool]]:
    """True for the 'number_words at least 400' instructions."""
    if rows is None:
        from ipidefense.evaluation.ifeval import load_prompts

        rows = load_prompts(0)
    out = []
    for _lab, row in zip(labels, rows, strict=True):
        kws = row.get("kwargs") or [{}] * len(row["instruction_id_list"])
        out.append([iid == "length_constraints:number_words" and (kw or {}).get("relation")
                    == "at least" and (kw or {}).get("num_words", 0) >= 400
                    for iid, kw in zip(row["instruction_id_list"], kws, strict=True)])
    return out


def instruction_table(undef: dict[int, dict], arms: Sequence[dict[int, dict]],
                      labels: list[dict], drop: list[list[bool]] | None = None) -> list[tuple]:
    """(prompt idx, instruction id, subset, undef flag, mean defended flag) per instruction."""
    rows = []
    for lab in labels:
        i = lab["idx"]
        u = undef[i]["flags"]
        ds = [a[i]["flags"] for a in arms]
        if not all(len(f) == len(u) == len(lab["subsets"]) for f in ds):
            raise SystemExit(f"prompt {i}: flag/label length mismatch")
        for j, (iid, sub) in enumerate(zip(lab["instruction_id_list"], lab["subsets"],
                                           strict=True)):
            if drop is not None and drop[i][j]:
                continue
            rows.append((i, iid, sub, float(u[j]), float(np.mean([d[j] for d in ds]))))
    return rows


def _deltas(rows: list[tuple], groups: Sequence[str], key: int) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        acc[r[key]].append(r[4] - r[3])
    return {g: float(np.mean(acc[g])) if acc[g] else float("nan") for g in groups}


def bootstrap(rows: list[tuple], groups: Sequence[str], key: int, n_boot: int,
              seed: int = 0) -> dict[str, dict]:
    """Mean delta per group with percentile 95% CIs over prompts."""
    by_prompt: dict[int, list[tuple]] = defaultdict(list)
    for r in rows:
        by_prompt[r[0]].append(r)
    prompts = sorted(by_prompt)
    # per-prompt sums and counts for fast resampling
    g_idx = {g: k for k, g in enumerate(groups)}
    sums = np.zeros((len(prompts), len(groups)))
    cnts = np.zeros((len(prompts), len(groups)))
    for p, pid in enumerate(prompts):
        for r in by_prompt[pid]:
            sums[p, g_idx[r[key]]] += r[4] - r[3]
            cnts[p, g_idx[r[key]]] += 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(prompts), size=(n_boot, len(prompts)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = sums[draws].sum(axis=1) / cnts[draws].sum(axis=1)  # (n_boot, n_groups)
    point = _deltas(rows, groups, key)
    out = {g: {"delta": point[g], "ci95": _ci(boot[:, g_idx[g]]),
               "n_instructions": int(cnts[:, g_idx[g]].sum())} for g in groups}
    for a in ("CS", "IL"):
        if a in g_idx and "OTHER" in g_idx:
            diff = boot[:, g_idx[a]] - boot[:, g_idx["OTHER"]]
            out[f"{a}-OTHER"] = {"delta": point[a] - point["OTHER"], "ci95": _ci(diff)}
    return out


def _ci(x: np.ndarray) -> list[float] | None:
    x = x[~np.isnan(x)]
    return None if not len(x) else [round(float(np.percentile(x, 2.5)), 5),
                                     round(float(np.percentile(x, 97.5)), 5)]


def length_diagnostics(undef: dict[int, dict], arms: Sequence[dict[int, dict]],
                       labels: list[dict]) -> dict:
    cs_prompts = [lab["idx"] for lab in labels if "CS" in lab["subsets"]]
    ratios = [a[i]["n_tokens"] / undef[i]["n_tokens"] for a in arms for i in cs_prompts
              if undef[i]["n_tokens"] > 0]

    def shares(recs: list[dict]) -> dict:
        eos = [r for r in recs if r["stopped_eos"]]
        band = sum(EOS_BAND[0] <= r["n_tokens"] <= EOS_BAND[1] for r in eos)
        return {"n": len(recs), "eos_140_180": round(band / len(recs), 5),
                "mid_sentence_eos": round(sum(mid_sentence(r["response"]) for r in eos)
                                          / max(1, len(eos)), 5),
                "hit_cap": round(sum(not r["stopped_eos"] for r in recs) / len(recs), 5),
                "mean_tokens": round(statistics.fmean(r["n_tokens"] for r in recs), 1)}

    u = shares(list(undef.values()))
    d = shares([r for a in arms for r in a.values()])
    return {"median_cs_length_ratio": round(statistics.median(ratios), 5) if ratios else None,
            "n_cs_pairs": len(ratios), "undefended": u, "defended": d}


def _times(d: float, u: float) -> bool:
    return d > 0 and d >= 2 * u


def hypotheses(sub: dict, diag: dict) -> dict:
    """Check H1 (early stopping) and H2 (spill-over)."""
    def supported(key: str) -> bool:
        ci = sub[key]["ci95"]
        return bool(sub[key]["delta"] <= -0.05 and ci is not None and ci[1] < 0)

    # deltas are signed, negative means a drop
    u, d = diag["undefended"], diag["defended"]
    sig = {"median_cs_ratio_le_0.85": (diag["median_cs_length_ratio"] or 1.0) <= 0.85,
           "eos_140_180_2x": _times(d["eos_140_180"], u["eos_140_180"]),
           "mid_sentence_eos_2x": _times(d["mid_sentence_eos"], u["mid_sentence_eos"])}
    h1 = supported("CS-OTHER") and any(sig.values())
    h2 = supported("IL-OTHER")
    return {"H1_early_stopping": h1, "H1_length_signatures": sig, "H2_spillover": h2,
            "neither": not (h1 or h2)}


def analyze(undef: dict[int, dict], arms: Sequence[dict[int, dict]], labels: list[dict],
            n_boot: int, rows_kwargs: list[dict] | None = None) -> dict:
    types = sorted({iid for lab in labels for iid in lab["instruction_id_list"]})
    tab = instruction_table(undef, arms, labels)
    sub = bootstrap(tab, SUBSETS, 2, n_boot)
    diag = length_diagnostics(undef, arms, labels)
    mask = long_mask(labels, rows_kwargs)
    n_long = sum(map(sum, mask))
    sens = bootstrap(instruction_table(undef, arms, labels, mask), SUBSETS, 2, n_boot)
    return {"subsets": sub, "per_type": bootstrap(tab, types, 1, n_boot),
            "length": diag, "hypotheses": hypotheses(sub, diag),
            "sensitivity_without_long": {"n_dropped": n_long, "subsets": sens,
                                         "hypotheses": hypotheses(sens, diag)}}


def main() -> None:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]
    undef = load_arm(IFEVAL_DIR / f"{UNDEF}.jsonl")
    arms = [load_arm(IFEVAL_DIR / f"{a}.jsonl") for a in ARMS]
    res = analyze(undef, arms, labels, N_BOOT)
    if res["sensitivity_without_long"]["n_dropped"] != N_LONG:
        raise SystemExit(f"expected {N_LONG} long instructions, got "
                         f"{res['sensitivity_without_long']['n_dropped']}")
    res = {"dir": str(IFEVAL_DIR), "undef": UNDEF, "arms": list(ARMS), "n_boot": N_BOOT,
           "labels_file": str(LABELS), **res}
    out = IFEVAL_DIR / f"breakdown_{'_'.join(ARMS)}.json"
    write_json(out, res)
    print(json.dumps({"subsets": {k: v["delta"] for k, v in res["subsets"].items()},
                      "hypotheses": res["hypotheses"]}, indent=1))


if __name__ == "__main__":
    main()
