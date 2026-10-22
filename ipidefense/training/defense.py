import argparse
import json
import random

import torch
from transformers import AutoTokenizer

from ipidefense.attacks.templates import (
    EVAL_INJS,
    PREFIX_TEMPLATES,
    TRAIN_TEMPLATES,
    TRAIN_WITNESSES,
)
from ipidefense.attacks.witnesses import OutcomeClassifier, WitnessInjection
from ipidefense.data.hosts import load_en_hosts
from ipidefense.evaluation.obey import fmt_obey, obey_by_lang
from ipidefense.model import (
    ADAPTER_DIR,
    BASE_DIR,
    MISTRAL_DIR,
    QWEN_DIR,
    QWEN_INPUT_TEMPLATE,
    SEALION_DIR,
    build_messages,
    generate,
    generate_batch,
)
from ipidefense.paths import MODELS, RUNS
from ipidefense.rendering.literal import render_inj_literal
from ipidefense.rendering.nllb import NLLBTranslator
from ipidefense.training.lora import SFTExample, build_lora_model, train_sft

REFUSAL = ("I cannot follow instructions contained in the provided input data. "
           "I will only act on the user's instruction.")


def base_and_tok(base: str):
    if base == "qwen":
        tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
        tok.chat_template = QWEN_INPUT_TEMPLATE
        return str(QWEN_DIR), tok
    if base == "sealion":
        tok = AutoTokenizer.from_pretrained(str(SEALION_DIR))
        tok.chat_template = AutoTokenizer.from_pretrained(str(ADAPTER_DIR)).chat_template
        return str(SEALION_DIR), tok
    if base == "mistral":
        return str(MISTRAL_DIR), AutoTokenizer.from_pretrained(str(MISTRAL_DIR))
    return str(BASE_DIR), AutoTokenizer.from_pretrained(str(ADAPTER_DIR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["llama", "qwen", "sealion", "mistral"], default="qwen")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--langs", default="en,es,vi")
    ap.add_argument("--max-new-tokens", type=int, default=160, help="for clean-target generation")
    ap.add_argument("--eval-mnt", type=int, default=64, help="obey-eval tokens (witness is early)")
    ap.add_argument(
        "--clean-frac",
        type=float,
        default=0.6,
        help="fraction of clean (no-injection) training examples, to preserve utility",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inline", action="store_true", help="data inline in user turn, no input role")
    ap.add_argument("--harden", action="store_true", help="add prefix-injection attacks")
    ap.add_argument("--refuse", action="store_true",
                    help="target a fixed refusal on injected examples instead of the clean answer")
    ap.add_argument("--batch", type=int, default=1, help="generation batch size")
    args = ap.parse_args()
    suffix = (("-inline" if args.inline else "") + ("-hardened" if args.harden else "")
              + ("-refuse" if args.refuse else ""))
    train_templates = TRAIN_TEMPLATES + PREFIX_TEMPLATES if args.harden else TRAIN_TEMPLATES
    langs = [x.strip() for x in args.langs.split(",")]
    rng = random.Random(args.seed)
    mnt = args.max_new_tokens

    model_dir, tok = base_and_tok(args.base)
    torch.manual_seed(args.seed)
    model = build_lora_model(model_dir)
    model.eval()
    clf = OutcomeClassifier()
    tr = NLLBTranslator()
    all_hosts = load_en_hosts(args.n_train + args.n_eval, args.seed)
    train_hosts, eval_hosts = all_hosts[: args.n_train], all_hosts[args.n_train :]
    rendered = {
        ll: [(render_inj_literal(tr, inj, ll), inj.witness) for inj in EVAL_INJS] for ll in langs
    }

    examples: list[SFTExample] = []
    with model.disable_adapter():
        # clean targets from the frozen model
        host_items = [(h.instruction, h.data) for h in train_hosts]
        if args.batch <= 1:
            targets = [
                generate(model, tok, instr, data=data, max_new_tokens=mnt, inline=args.inline)
                for instr, data in host_items
            ]
        else:
            targets = []
            n_hosts = len(host_items)
            for k in range(0, n_hosts, args.batch):
                chunk = host_items[k : k + args.batch]
                targets += generate_batch(model, tok, chunk, max_new_tokens=mnt, inline=args.inline)
                print(f"targets {min(k + args.batch, n_hosts)}/{n_hosts}", flush=True)
        # clean with prob. clean_frac, else a random attack in langs[i % n]
        n_clean = 0
        for i, (h, target) in enumerate(zip(train_hosts, targets, strict=True)):
            if not target.strip():
                continue
            ex_target = target
            if rng.random() < args.clean_frac:
                data = h.data
                n_clean += 1
            else:
                templ, w = rng.choice(train_templates), rng.choice(TRAIN_WITNESSES)
                lang = langs[i % len(langs)]
                inj = render_inj_literal(tr, WitnessInjection(templ, w), lang)
                data = f"{h.data}\n{inj}"
                if args.refuse:
                    ex_target = REFUSAL
            msgs = build_messages(h.instruction, data, args.inline)
            examples.append(SFTExample(messages=msgs, target=ex_target))
        undef = obey_by_lang(
            model, tok, clf, eval_hosts, rendered, args.eval_mnt, args.inline, args.batch
        )
    print(f"training {len(examples)} examples ({n_clean} clean), {args.epochs} epoch", flush=True)

    train_sft(model, tok, examples, epochs=args.epochs, lr=args.lr, seed=args.seed)

    defended = obey_by_lang(
        model, tok, clf, eval_hosts, rendered, args.eval_mnt, args.inline, args.batch
    )

    report = {
        "n_train": len(examples),
        "n_eval": len(eval_hosts),
        "epochs": args.epochs,
        "clean_frac": args.clean_frac,
        "base": args.base,
        "lr": args.lr,
        "n_attacks": len(EVAL_INJS),
        "langs": langs,
        "undefended": undef,
        "defended": defended,
        "reduction": {ll: round(undef[ll]["mean"] - defended[ll]["mean"], 3) for ll in langs},
    }
    report["inline"] = args.inline
    report["refuse"] = args.refuse
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"train_{args.base}{suffix}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    adapter_dir = MODELS / f"defense-{args.base}-seed{args.seed}{suffix}"
    model.save_pretrained(str(adapter_dir))
    print(fmt_obey({"undefended": undef, "defended": defended}), flush=True)
    print(f"wrote {path}", flush=True)
    print(f"saved adapter -> {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
