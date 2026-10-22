import argparse
import json

from ipidefense.attacks.witnesses import OutcomeClassifier
from ipidefense.data.hosts import load_en_hosts
from ipidefense.metrics.scoring import rouge_l_f1
from ipidefense.model import generate, load_defense_lora, load_defense_lora_qwen
from ipidefense.paths import MODELS, RUNS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["llama", "qwen"], default="qwen")
    ap.add_argument("--lora-dir", default=str(MODELS / "defense-qwen-seed0"))
    ap.add_argument("--skip", type=int, default=1650, help="# hosts used in training (held out)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0, help="host-split seed, match the adapter's seed")
    args = ap.parse_args()

    model, tok = (
        load_defense_lora_qwen(args.lora_dir)
        if args.base == "qwen"
        else load_defense_lora(args.lora_dir)
    )
    clf = OutcomeClassifier()
    hosts = load_en_hosts(args.skip + args.n, args.seed)[args.skip :]
    mnt = args.max_new_tokens

    rouges, len_ratios, refusals, rows = [], [], 0, []
    for h in hosts:
        with model.disable_adapter():
            ref = generate(model, tok, h.instruction, data=h.data, max_new_tokens=mnt)
        hyp = generate(model, tok, h.instruction, data=h.data, max_new_tokens=mnt)
        rl = rouge_l_f1(ref, hyp)
        # the witness argument is only a placeholder
        is_refusal = clf.classify(hyp, "PWNED") == "refused"
        rouges.append(rl)
        len_ratios.append(round(len(hyp.split()) / max(1, len(ref.split())), 3))
        refusals += int(is_refusal)
        rows.append(
            {
                "instruction": h.instruction[:80],
                "rouge_l": rl,
                "defended_refusal": is_refusal,
                "undef": ref[:200],
                "def": hyp[:200],
            }
        )

    n = len(hosts)
    report = {
        "base": args.base,
        "lora_dir": args.lora_dir,
        "seed": args.seed,
        "n": n,
        "mean_rouge_l_vs_undefended": round(sum(rouges) / n, 4),
        "clean_refusal_rate": round(refusals / n, 4),
        "mean_len_ratio_def_over_undef": round(sum(len_ratios) / n, 3),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"eval_data_channel_{args.base}.json"
    path.write_text(json.dumps({"summary": report, "rows": rows}, indent=2), encoding="utf-8")
    print(
        f"{args.base} n={n}  "
        f"rouge_l={report['mean_rouge_l_vs_undefended']:.4f}  "
        f"clean_refusal={report['clean_refusal_rate']:.4f}  "
        f"len_ratio={report['mean_len_ratio_def_over_undef']:.3f}",
        flush=True,
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
