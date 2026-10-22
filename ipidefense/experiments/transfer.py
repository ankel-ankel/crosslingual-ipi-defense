"""Held-out injections in six languages (three unseen in training), undefended vs defended.

transfer.py --lora-dir models/defense-qwen-seed0 --seed 0
"""
from __future__ import annotations

import argparse
import json

from ipidefense.attacks.templates import EVAL_INJS
from ipidefense.attacks.witnesses import OutcomeClassifier
from ipidefense.data.hosts import load_en_hosts
from ipidefense.evaluation.obey import fmt_obey, obey_by_lang
from ipidefense.model import load_defended
from ipidefense.paths import MODELS, RUNS
from ipidefense.rendering.literal import render_inj_literal
from ipidefense.rendering.nllb import NLLBTranslator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-dir", default=str(MODELS / "defense-qwen-seed0"))
    ap.add_argument("--base", choices=["qwen", "llama", "sealion"], default="qwen")
    ap.add_argument("--trained-langs", default="en,es,vi", help="langs the LoRA was trained on")
    ap.add_argument("--langs", default="en,es,fr,de,zh,vi", help="eval langs (incl. unseen)")
    ap.add_argument("--n-train", type=int, default=1500, help="offset so eval hosts are held out")
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--eval-mnt", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="host-split seed, match the adapter's seed")
    ap.add_argument("--batch", type=int, default=1, help="generation batch size")
    args = ap.parse_args()
    langs = [x.strip() for x in args.langs.split(",")]
    trained = {x.strip() for x in args.trained_langs.split(",")}

    model, tok = load_defended(args.base, args.lora_dir)
    clf = OutcomeClassifier()
    tr = NLLBTranslator()
    eval_hosts = load_en_hosts(args.n_train + args.n_eval, args.seed)[args.n_train :]
    rendered = {
        ll: [(render_inj_literal(tr, inj, ll), inj.witness) for inj in EVAL_INJS] for ll in langs
    }

    with model.disable_adapter():
        undef = obey_by_lang(model, tok, clf, eval_hosts, rendered, args.eval_mnt, batch=args.batch)
    defended = obey_by_lang(model, tok, clf, eval_hosts, rendered, args.eval_mnt, batch=args.batch)

    report = {
        "lora_dir": args.lora_dir,
        "seed": args.seed,
        "trained_langs": sorted(trained),
        "eval_langs": langs,
        "n_eval": len(eval_hosts),
        "n_attacks": len(EVAL_INJS),
        "unseen_langs": [ll for ll in langs if ll not in trained],
        "undefended": undef,
        "defended": defended,
        "reduction": {ll: round(undef[ll]["mean"] - defended[ll]["mean"], 3) for ll in langs},
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"eval_transfer_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(fmt_obey({"undefended": undef, "defended": defended}), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
