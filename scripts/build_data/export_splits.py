"""Write data/splits/seed{0,1,2}.json: for every host slice, the Alpaca row indices in host order.

python scripts/build_data/export_splits.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from ipidefense.data.hosts import ALPACA, HostTask
from ipidefense.data.splits import SLICES, hosts
from ipidefense.paths import DATA


def row_order(rows: list[dict], seed: int) -> list[int]:
    # filter and shuffle of load_en_hosts on row indices (the shuffle depends on the length only)
    usable = [
        i
        for i, r in enumerate(rows)
        if (r.get("input") or "").strip() and (r.get("instruction") or "").strip()
    ]
    random.Random(seed).shuffle(usable)
    return usable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=str(DATA / "splits"))
    args = ap.parse_args()

    raw = ALPACA.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows = json.loads(raw.decode("utf-8"))
    for seed in args.seeds:
        order = row_order(rows, seed)
        slices = {name: order[lo:hi] for name, (lo, hi) in SLICES.items()}
        # the indices must give back exactly the hosts the experiments use
        for name, idx in slices.items():
            want = hosts(seed, name)
            got = tuple(HostTask(rows[i]["instruction"].strip(), rows[i]["input"].strip())
                        for i in idx)
            if got != want:
                raise SystemExit(f"seed {seed} slice {name}: indices do not reproduce the hosts")
        head = {
            "source": "alpaca_cleaned.json",
            "alpaca_sha256": digest,
            "seed": seed,
            "order": "rows with non-empty instruction and input, shuffled by random.Random(seed)",
            "bounds": {name: list(b) for name, b in SLICES.items()},
        }
        lines = [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
        lines += [' "slices": {']
        lines += [f"  {json.dumps(name)}: {json.dumps(idx)}," for name, idx in slices.items()]
        lines[-1] = lines[-1].rstrip(",")
        text = "{\n" + "\n".join(lines) + "\n }\n}\n"
        out = Path(args.out) / f"seed{seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.loads(text)  # well-formed
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
