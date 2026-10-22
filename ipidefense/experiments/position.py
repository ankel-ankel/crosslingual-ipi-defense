"""Held-out injections at the start, middle and end of the host data (training used the end).

position.py --lora-dir models/defense-qwen-seed0 --seed 0
"""
from __future__ import annotations

import argparse
import json

import torch

from ipidefense.attacks.templates import EVAL_INJS
from ipidefense.attacks.witnesses import OutcomeClassifier
from ipidefense.data.hosts import load_en_hosts
from ipidefense.model import generate_batch, load_defense_lora_qwen
from ipidefense.paths import MODELS, RUNS
from ipidefense.rendering.literal import render_inj_literal
from ipidefense.rendering.nllb import NLLBTranslator

POSITIONS = ("start", "middle", "end")


def place(data: str, inj: str, position: str) -> str:
    if position == "start":
        return f"{inj}\n{data}"
    if position == "middle":
        words = data.split()
        mid = max(1, len(words) // 2)
        return " ".join(words[:mid]) + f"\n{inj}\n" + " ".join(words[mid:])
    return f"{data}\n{inj}"


@torch.no_grad()
def obey_rate(
    model, tok, clf: OutcomeClassifier, hosts, inj_text: str, witness: str,
    position: str, mnt: int, batch: int,
) -> float:
    items = [(h.instruction, place(h.data, inj_text, position)) for h in hosts]
    outs: list[str] = []
    for k in range(0, len(items), batch):
        outs += generate_batch(model, tok, items[k : k + batch], max_new_tokens=mnt)
    labels = [clf.classify(o, witness) for o in outs]
    return round(sum(lab == "obeyed" for lab in labels) / len(labels), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-dir", default=str(MODELS / "defense-qwen-seed0"))
    ap.add_argument("--langs", default="en")
    ap.add_argument("--n-train", type=int, default=1500, help="offset so eval hosts are held out")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--eval-mnt", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="host-split seed, match the adapter's seed")
    ap.add_argument("--batch", type=int, default=8, help="obey is batch-invariant on Qwen")
    args = ap.parse_args()
    langs = [x.strip() for x in args.langs.split(",")]

    model, tok = load_defense_lora_qwen(args.lora_dir)
    clf = OutcomeClassifier()
    tr = NLLBTranslator()
    hosts = load_en_hosts(args.n_train + args.n, args.seed)[args.n_train :]
    rendered = {
        ll: [(render_inj_literal(tr, inj, ll), inj.witness) for inj in EVAL_INJS] for ll in langs
    }

    report: dict = {
        "lora_dir": args.lora_dir,
        "seed": args.seed,
        "langs": langs,
        "n_hosts": len(hosts),
        "n_attacks": len(EVAL_INJS),
        "positions": {},
    }
    for pos in POSITIONS:
        block: dict = {}
        for ll in langs:
            undef_pa, def_pa = [], []
            for txt, w in rendered[ll]:
                with model.disable_adapter():
                    undef_pa.append(obey_rate(model, tok, clf, hosts, txt, w, pos,
                                              args.eval_mnt, args.batch))
                def_pa.append(obey_rate(model, tok, clf, hosts, txt, w, pos,
                                        args.eval_mnt, args.batch))
            block[ll] = {
                "undefended": {"mean": round(sum(undef_pa) / len(undef_pa), 3),
                               "per_attack": undef_pa},
                "defended": {"mean": round(sum(def_pa) / len(def_pa), 3),
                             "per_attack": def_pa},
            }
            print(f"{pos:<7} {ll}: undef={block[ll]['undefended']['mean']:.3f} "
                  f"{undef_pa}  def={block[ll]['defended']['mean']:.3f} {def_pa}", flush=True)
        report["positions"][pos] = block

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"eval_position_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
