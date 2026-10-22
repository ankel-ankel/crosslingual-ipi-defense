import argparse
import json
from pathlib import Path

from ipidefense.evaluation.ifeval import follow_flags, load_prompts
from ipidefense.model import generate, generate_batch, load_base_undefended, load_defended
from ipidefense.paths import MODELS, RUNS


def _load(model_kind: str, base: str, lora_dir: str):
    return load_defended(base, lora_dir) if model_kind == "lora" else load_base_undefended(base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["undefended", "lora"], default="undefended")
    ap.add_argument("--base", choices=["llama", "qwen", "sealion", "mistral"], default="qwen")
    ap.add_argument("--lora-dir", default=str(MODELS / "defense-qwen-seed0"))
    ap.add_argument("--n", type=int, default=541, help="0 = all IFEval prompts")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1, help="generation batch size")
    args = ap.parse_args()

    model, tok = _load(args.model, args.base, args.lora_dir)
    rows = load_prompts(args.n)
    if args.batch <= 1:
        resps = [
            generate(model, tok, row["prompt"], data=None, max_new_tokens=args.max_new_tokens)
            for row in rows
        ]
    else:
        resps = []
        for k in range(0, len(rows), args.batch):
            items = [(row["prompt"], None) for row in rows[k : k + args.batch]]
            resps += generate_batch(model, tok, items, max_new_tokens=args.max_new_tokens)
    prompt_ok = inst_ok = inst_tot = 0
    for row, resp in zip(rows, resps, strict=True):
        flags = follow_flags(row, resp)
        prompt_ok += int(all(flags))
        inst_ok += sum(flags)
        inst_tot += len(flags)
    report = {
        "model": args.model,
        "base": args.base,
        "lora_dir": args.lora_dir if args.model == "lora" else None,
        "n": len(rows),
        "prompt_strict_acc": round(prompt_ok / len(rows), 4),
        "instruction_strict_acc": round(inst_ok / inst_tot, 4) if inst_tot else 0.0,
    }
    sub = args.model if args.model != "lora" else f"lora_{Path(args.lora_dir).name}"
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"eval_utility_{args.base}_{sub}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{args.model} ({args.base}) n={report['n']}  "
        f"prompt_strict={report['prompt_strict_acc']:.4f}  "
        f"instruction_strict={report['instruction_strict_acc']:.4f}",
        flush=True,
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
