import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ipidefense.evaluation.runtime import mcnemar
from ipidefense.experiments.gcg_run import ROWS, host_hit
from ipidefense.paths import RUNS
from ipidefense.utils import Transcript, write_json

PAIR_KEYS = ("witness", "injection", "instruction_sha", "data_sha")


def _path(spec: str) -> Path:
    p = Path(spec)
    return p if p.is_absolute() else RUNS / p


def load_run(path: Path) -> tuple[dict[int, dict], dict]:
    # per-host records by host_index, plus design.json
    pre = path.parent / "design.json"
    if not path.exists() or not pre.exists():
        raise SystemExit(f"{path}: transcript or its design.json not found")
    recs: dict[int, dict] = {}
    for r in Transcript(path).done.values():
        if r["host_index"] in recs:
            raise SystemExit(f"{path}: several records for host {r['host_index']} "
                             "(per-host GCG transcripts only, not --eval-suffix)")
        recs[r["host_index"]] = r
    return recs, json.loads(pre.read_text(encoding="utf-8"))


def outcome(record: dict, row: str) -> bool:
    return bool(record["success"]) if row == "witness_first" else host_hit(record, row)


def paired_test(a: Path, b: Path, n_hosts: int) -> dict:
    # a technical failure counts as a failed attack
    (ra, pa), (rb, pb) = load_run(a), load_run(b)
    if pa["config"] != pb["config"]:
        diff = sorted(k for k in pa["config"].keys() | pb["config"].keys()
                      if pa["config"].get(k) != pb["config"].get(k))
        raise SystemExit(f"GCG config differs between the runs ({diff}): not the same protocol")
    asg = [{h["host_index"]: h for h in p["host_assignment"]} for p in (pa, pb)]
    hosts = list(range(n_hosts))
    for i in hosts:
        if i not in asg[0] or i not in asg[1]:
            raise SystemExit(f"host {i} is not in both host assignments")
        bad = [k for k in PAIR_KEYS if asg[0][i].get(k) != asg[1][i].get(k)]
        if bad:
            raise SystemExit(f"host {i} differs between the runs ({bad}): not a paired design")
    unpaired = {"a": [i for i in hosts if i not in ra], "b": [i for i in hosts if i not in rb]}
    if unpaired["a"] or unpaired["b"]:
        print(f"WARNING: hosts without a record are left out of the test: {unpaired}",
              file=sys.stderr, flush=True)
    paired = [i for i in hosts if i in ra and i in rb]
    if not paired:
        raise SystemExit(f"no host has a record in both runs ({unpaired}): nothing to test")
    rows = {}
    for row in ROWS:
        x, y = [outcome(ra[i], row) for i in paired], [outcome(rb[i], row) for i in paired]
        m = mcnemar(x, y)
        rows[row] = {"n": m["n"], "a_k": sum(x), "b_k": sum(y), "a_only": m["x_only"],
                     "b_only": m["y_only"], "p_exact": m["p_exact"]}

    def side(path: Path, pre: dict, recs: dict[int, dict]) -> dict:
        # arm is the transcript tag, e.g. <adapter>.jsonl or undef.jsonl
        return {"transcript": str(path), "arm": path.stem, "adapter": pre.get("adapter"),
                "config_hash": pre.get("config_hash"),
                "technical_failure_hosts": [i for i in paired if recs[i].get("technical_failure")]}

    return {"test": "exact two-sided McNemar on paired hosts (binomial on discordant pairs)",
            "primary": "witness_first", "a": side(a, pa, ra), "b": side(b, pb, rb),
            "hosts": paired, "unpaired_hosts": unpaired, "rows": rows}


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="per-host transcript, e.g. gcg/hard_s0/hard-s0.jsonl")
    ap.add_argument("b", help="the paired run's per-host transcript")
    ap.add_argument("--hosts", type=int, required=True, help="test hosts 0..N-1")
    ap.add_argument("--out", required=True, help="also write the JSON here")
    args = ap.parse_args(argv)
    res = paired_test(_path(args.a), _path(args.b), args.hosts)
    write_json(_path(args.out), res)
    print(json.dumps(res, indent=1), flush=True)
    return res


if __name__ == "__main__":
    main()
