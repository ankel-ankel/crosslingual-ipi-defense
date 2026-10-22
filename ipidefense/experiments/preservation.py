import argparse
import json

import torch

from ipidefense.attacks.templates import EVAL_INJS
from ipidefense.attacks.witnesses import OutcomeClassifier
from ipidefense.data.hosts import load_en_hosts
from ipidefense.metrics.scoring import rouge_l_f1
from ipidefense.model import generate_batch, load_defense_lora_qwen
from ipidefense.paths import MODELS, RUNS

N_TRAIN = 1500  # offset so eval hosts are held out
N = 150
MNT = 160
BATCH = 8


@torch.no_grad()
def answers(model, tok, items: list[tuple[str, str]], mnt: int, batch: int) -> list[str]:
    outs: list[str] = []
    for k in range(0, len(items), batch):
        outs += generate_batch(model, tok, items[k : k + batch], max_new_tokens=mnt)
    return outs


def arm_report(model, tok, clf: OutcomeClassifier, hosts, mnt: int, batch: int) -> dict:
    clean = answers(model, tok, [(h.instruction, h.data) for h in hosts], mnt, batch)
    per_attack = []
    for inj in EVAL_INJS:
        txt = inj.render()
        injected = answers(
            model, tok, [(h.instruction, f"{h.data}\n{txt}") for h in hosts], mnt, batch
        )
        rouges = [rouge_l_f1(c, i) for c, i in zip(clean, injected, strict=True)]
        obeys = sum(clf.classify(o, inj.witness) == "obeyed" for o in injected)
        per_attack.append({
            "witness": inj.witness,
            "mean_rouge_l_vs_own_clean": round(sum(rouges) / len(rouges), 4),
            "obeyed": obeys,
        })
        print(f"  {inj.witness:<9} rouge_l={per_attack[-1]['mean_rouge_l_vs_own_clean']:.4f} "
              f"obeyed={obeys}/{len(hosts)}", flush=True)
    overall = round(sum(a["mean_rouge_l_vs_own_clean"] for a in per_attack) / len(per_attack), 4)
    return {"mean_rouge_l": overall, "per_attack": per_attack}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-dir", default=str(MODELS / "defense-qwen-seed0"))
    ap.add_argument("--seed", type=int, default=0, help="host-split seed, match the adapter's seed")
    args = ap.parse_args()

    model, tok = load_defense_lora_qwen(args.lora_dir)
    clf = OutcomeClassifier()
    hosts = load_en_hosts(N_TRAIN + N, args.seed)[N_TRAIN:]

    print("undefended arm", flush=True)
    with model.disable_adapter():
        undef = arm_report(model, tok, clf, hosts, MNT, BATCH)
    print("defended arm", flush=True)
    defended = arm_report(model, tok, clf, hosts, MNT, BATCH)

    report = {
        "lora_dir": args.lora_dir,
        "seed": args.seed,
        "n_hosts": len(hosts),
        "n_attacks": len(EVAL_INJS),
        "mnt": MNT,
        "note": "ROUGE-L of each arm's injected-data answer vs the SAME arm's clean-data answer",
        "undefended": undef,
        "defended": defended,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"eval_attack_utility_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"undefended={undef['mean_rouge_l']:.4f}  defended={defended['mean_rouge_l']:.4f}",
          flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
