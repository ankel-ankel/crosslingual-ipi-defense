import argparse
import hashlib
import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ipidefense.attacks.templates import EVAL_INJS
from ipidefense.attacks.witnesses import REFUSAL_CUES
from ipidefense.data.splits import hosts_for
from ipidefense.experiments.suite import (
    LOG,
    Arm,
    arm_plan,
    freeze_design,
    load_eval_model,
    read_records,
    run_units,
    words,
)
from ipidefense.metrics.scoring import classify, rouge_l_f1
from ipidefense.paths import RUNS
from ipidefense.utils import Item, write_json

OUT = RUNS / "judge_dump"
SALT = "r1-judge-2026-10"
MNT = 160
PARTS = {"attack": ("security", 8), "benign": ("benign", 1)}  # (split, batch) of the original runs


def units_for(part: str, hosts: Sequence[Any]) -> list[list[Item]]:
    """Attack: clean + EVAL_INJS units on the same hosts. Benign: one clean unit."""
    def item(tid: str, i: int, h: Any, inj: Any) -> Item:
        data = h.data if inj is None else f"{h.data}\n{inj.render()}"
        return Item(f"{tid}|{i}", h.instruction, data, {
            "part": part, "tid": tid, "host": i, "witness": None if inj is None else inj.witness,
            "injection": None if inj is None else inj.render()})

    if part == "benign":
        return [[item("benign", i, h, None) for i, h in enumerate(hosts)]]
    return [[item("clean", i, h, None) for i, h in enumerate(hosts)]] + [
        [item(f"inj{t}", i, h, inj) for i, h in enumerate(hosts)]
        for t, inj in enumerate(EVAL_INJS)]


def score(item: Item, text: str) -> dict:
    out = {"refusal_cue": any(c in text.casefold() for c in REFUSAL_CUES)}
    if item.meta["witness"]:
        out.update({k: v for k, v in classify(text, item.meta["witness"]).items()
                    if k in ("wf", "rf")})
    return out


def tpath(out: Path, arm: Arm, seed: int, part: str) -> Path:
    return out / "transcripts" / f"{arm.spec}__s{seed}__{part}.jsonl"


def run(args: argparse.Namespace) -> None:
    out = Path(args.out_dir) / ("smoke" if args.limit_hosts else "")
    plan = arm_plan(args.arms, args.seeds)
    freeze_design(out, {"mnt": MNT, "parts": PARTS,
                        "injections": [i.render() for i in EVAL_INJS],
                        "fields": ["item_id", "instruction", "data_clean", "injection",
                                   "reference_output", "candidate_output", "arm_blind_id",
                                   "meta"],
                        "dup_frac": args.dup_frac, "n_sanity": args.n_sanity},
                  {"arms": [a.spec for a, _ in plan], "seeds": args.seeds,
                   "limit_hosts": args.limit_hosts})
    model, tok = load_eval_model([a for a, _ in plan])
    for arm, seed in plan:
        for part, (split, batch) in PARTS.items():
            hosts = hosts_for(arm.name, split, seed)[: args.limit_hosts or None]
            run_units(model, tok, arm, seed, units_for(part, hosts), tpath(out, arm, seed, part),
                      MNT, batch, score)
    dump(args)


def _blind(arm: str, seed: int) -> str:
    return hashlib.sha1(f"{SALT}|{arm}|{seed}".encode()).hexdigest()[:8]


def _iid(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, (SALT, *parts))).encode()).hexdigest()[:12]


def _pair(kind: str, ref: dict, cand: dict, arm: str, seed: int,
          dup: int = 0) -> tuple[dict, dict]:
    """Visible item and key entry for one pair. Arm and labels stay in the key."""
    iid = _iid(kind, arm, seed, cand["tid"], cand["host"], dup)
    inj = cand.get("injection")
    item = {"item_id": iid, "instruction": cand["instruction"], "data_clean": cand["data_clean"],
            "injection": inj, "reference_output": ref["output"],
            "candidate_output": cand["output"], "arm_blind_id": _blind(arm, seed),
            "meta": {"task": "attacked" if inj else "benign"}}
    obeyed = cand.get("wf") == "obeyed"
    key = {"pair_type": kind, "arm": arm, "seed": seed, "host": cand["host"], "tid": cand["tid"],
           "witness": cand.get("witness"), "cand_wf": cand.get("wf"), "cand_rf": cand.get("rf"),
           "rouge_l": rouge_l_f1(ref["output"], cand["output"]), "dup_of": None,
           "sanity_expect": ("Q1=1, Q3=yes" if obeyed and len(words(cand["output"])) <= 3
                             else "Q1=5" if kind == "sanity_identical" else None)}
    return item, key


def _with_text(recs: dict[str, dict], hosts: Sequence[Any]) -> dict[tuple[str, int], dict]:
    """Records keyed by (tid, host) with clean data and instruction added back."""
    return {(r["tid"], r["host"]): {**r, "instruction": hosts[r["host"]].instruction,
                                    "data_clean": hosts[r["host"]].data}
            for r in recs.values()}


def collect(args: argparse.Namespace, out: Path) -> list[tuple[dict, dict]]:
    """All attacked and benign pairs."""
    pairs: list[tuple[dict, dict]] = []
    benign: dict[int, dict[str, tuple[str, dict]]] = {}
    for arm, seed in arm_plan(args.arms, args.seeds):
        fam = "def" if arm.defended else "undef"
        att = _with_text(read_records(tpath(out, arm, seed, "attack")),
                         hosts_for(arm.name, "security", seed))
        for t in range(len(EVAL_INJS)):
            hs = sorted(h for tid, h in att if tid == f"inj{t}" and ("clean", h) in att)
            rs = [(att[("clean", h)], att[(f"inj{t}", h)]) for h in hs]
            pairs += [_pair("attacked", c, a, arm.spec, seed) for c, a in rs]
        recs = read_records(tpath(out, arm, seed, "benign"))
        if recs:
            benign.setdefault(seed, {})[fam] = (arm.spec, _with_text(
                recs, hosts_for(arm.name, "benign", seed)))
    for seed, b in benign.items():
        if "undef" not in b or "def" not in b:
            continue
        (_, und), (d_arm, dfd) = b["undef"], b["def"]
        rs = [(und[k], dfd[k]) for k in sorted(dfd) if k in und]
        pairs += [_pair("benign", u, d, d_arm, seed) for u, d in rs]
    return pairs


def dump(args: argparse.Namespace) -> dict:
    """Write the blind, shuffled judge file and the unblinding key."""
    out = Path(args.out_dir) / ("smoke" if args.limit_hosts else "")
    pairs = collect(args, out)
    rng = random.Random(0)
    # identical-pair sanity items and a test-retest duplicate subset
    cleans = [(i, k) for i, k in pairs if k["pair_type"] == "attacked"]
    seen: set[str] = set()
    for item, key in rng.sample(cleans, min(args.n_sanity, len(cleans))):
        ref = {"output": item["reference_output"]}
        if ref["output"] in seen:
            continue
        seen.add(ref["output"])
        cand = {"host": key["host"], "tid": "clean", "output": ref["output"],
                "instruction": item["instruction"], "data_clean": item["data_clean"]}
        pairs.append(_pair("sanity_identical", ref, cand, key["arm"], key["seed"]))
    dups = []
    for item, key in rng.sample(pairs, round(args.dup_frac * len(pairs))):
        new_id = _iid("dup", item["item_id"])
        dups.append(({**item, "item_id": new_id}, {**key, "dup_of": item["item_id"]}))
    allp = pairs + dups
    rng.shuffle(allp)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "judge_items.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for item, _ in allp:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_json(out / "judge_key.json", {"salt": SALT, "arms": {
        _blind(a.spec, s): {"arm": a.spec, "seed": s} for a, s in arm_plan(args.arms, args.seeds)},
        "items": {i["item_id"]: k for i, k in allp}})
    counts = {t: sum(k["pair_type"] == t for _, k in pairs)
              for t in ("attacked", "benign", "sanity_identical")}
    summary = {"counts": {**counts, "duplicates": len(dups), "total": len(allp)}}
    write_json(out / "summary.json", summary)
    LOG.info("judge items %s", summary["counts"])
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--limit-hosts", type=int, default=0, help="smoke test: first N hosts")
    ap.set_defaults(arms=["undef", "def-s0", "def-s1", "def-s2"], dup_frac=0.10, n_sanity=30,
                    out_dir=str(OUT))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
