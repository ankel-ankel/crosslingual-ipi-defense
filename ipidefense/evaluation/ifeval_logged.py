"""IFEval with per-prompt logs (response, generated tokens, EOS flag, strict flags).

Usage: ifeval_logged.py --arms undef def-s0 def-s1 def-s2 [--batch 16]
"""
from __future__ import annotations

import argparse
import contextlib
import random
import time
from pathlib import Path

import torch

from ipidefense.arms import ADAPTERS, arms_base, check_adapter_base
from ipidefense.evaluation.ifeval import follow_flags, load_prompts
from ipidefense.gpumem import lean_kwargs, release
from ipidefense.model import (
    build_inputs,
    build_messages,
    load_mistral_undefended,
    load_qwen_undefended,
    load_sealion_undefended,
)
from ipidefense.paths import MODELS, RUNS
from ipidefense.utils import Transcript, chunks, sha, write_json

# Published IFEval counts (instructions of 834, prompts of 541) of the unbatched utility runs
PUBLISHED = {"undef": (633, 366), "def-s0": (590, 331), "def-s1": (570, 313),
             "def-s2": (592, 327),
             "sealion-undef": (689, 412), "sealion-def-s0": (552, 303),
             "mistral-undef": (472, 248), "mistral-def-s0": (337, 158)}
GATE_TOL = 2  # instructions and prompts; the scorer is stochastic for 2 prompts
RANDOM_LETTER_IDX = (19, 22)  # keywords:letter_frequency with a non-letter -> random letter
N_LORA_MODULES = 196  # 28 layers x 7 projections
# LoRA modules per backbone (all 7 projections per layer): Qwen 28 layers, Llama-3.1 / Mistral 32
LORA_MODULES = {"qwen": N_LORA_MODULES, "sealion": 32 * 7, "mistral": 32 * 7}
OUT = RUNS / "ifeval_logged"


def default_out(batch: int) -> Path:
    """Unbatched and batched transcripts never share a directory."""
    return OUT if batch <= 1 else RUNS / f"ifeval_logged_b{batch}"


def seed_scorer() -> None:
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
    random.seed(0)


def check_adapters(model, arms: list[str]) -> None:
    """Fail fast if an adapter did not load (wrong keys would give a do-nothing adapter)."""
    norms = {}
    for arm in arms:
        if not ADAPTERS[arm][1]:
            continue
        mods = [m for m in model.modules() if hasattr(m, "lora_B") and arm in m.lora_B]
        assert len(mods) == LORA_MODULES[ADAPTERS[arm][0]], f"{arm}: {len(mods)} LoRA modules"
        norms[arm] = float(sum(m.lora_B[arm].weight.float().norm() for m in mods))
        assert norms[arm] > 0, f"{arm}: lora_B is all zero"
    assert len(set(round(v, 3) for v in norms.values())) == len(norms), (
        f"identical adapters {norms}")
    print(f"adapters OK: { {k: round(v, 2) for k, v in norms.items()} }", flush=True)


def load_base(base: str):
    """The undefended backbone with its tokenizer and chat template (as load_base_undefended)."""
    if base == "qwen":
        return load_qwen_undefended()
    if base == "sealion":
        return load_sealion_undefended()
    if base == "mistral":
        return load_mistral_undefended()
    raise ValueError(f"unknown backbone {base!r}")


def load_model(arms: list[str]):
    """One backbone with every requested adapter attached under its arm name."""
    from peft import PeftModel

    backbone = arms_base(arms)
    for arm in arms:
        base, _, inline = ADAPTERS[arm]
        if backbone == "qwen":
            assert base == "qwen" and not inline, f"{arm}: only Qwen role-format arms are supported"
        else:
            check_adapter_base(arm)
    base_model, tok = load_base(backbone)
    model = base_model
    for arm in arms:
        adir = ADAPTERS[arm][1]
        if not adir:
            continue
        path = str(MODELS / adir)
        if isinstance(model, PeftModel):
            model.load_adapter(path, adapter_name=arm)
        else:
            model = PeftModel.from_pretrained(base_model, path, adapter_name=arm,
                                              device_map={"": 0})
    model.eval()
    if isinstance(model, PeftModel):
        check_adapters(model, arms)
    return model, tok


@torch.no_grad()
def gen_logged(model, tok, prompt: str, mnt: int) -> dict:
    """Same call as ipidefense.model.generate, plus token-level metadata."""
    inputs = build_inputs(tok, prompt, None, False).to(model.device)
    out = model.generate(**inputs, max_new_tokens=mnt, do_sample=False,
                         pad_token_id=tok.eos_token_id, **lean_kwargs())
    ids = out[0, inputs["input_ids"].shape[1]:].tolist()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    n_gen = next((i for i, t in enumerate(ids) if t in eos_ids), len(ids))
    return {"response": tok.decode(ids, skip_special_tokens=True), "n_tokens": n_gen,
            "stopped_eos": n_gen < len(ids)}


@torch.no_grad()
def gen_logged_batch(model, tok, prompts: list[str], mnt: int) -> list[dict]:
    """Left-padded greedy batch (= ipidefense.model.generate_batch) with the same metadata;
    pads after an EOS are EOS ids, so n_tokens is unaffected."""
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    side, tok.padding_side = tok.padding_side, "left"
    try:
        texts = [tok.apply_chat_template(build_messages(p, None, False),
                                         add_generation_prompt=True, tokenize=False)
                 for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        out = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                             pad_token_id=tok.eos_token_id, **lean_kwargs())
    finally:
        tok.padding_side = side
    release()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    recs = []
    for ids in out[:, enc["input_ids"].shape[1]:].tolist():
        n_gen = next((i for i, t in enumerate(ids) if t in eos_ids), len(ids))
        recs.append({"response": tok.decode(ids, skip_special_tokens=True), "n_tokens": n_gen,
                     "stopped_eos": n_gen < len(ids)})
    return recs


def adapter_context(model, arm: str):
    from peft import PeftModel

    if not ADAPTERS[arm][1]:
        return model.disable_adapter() if isinstance(model, PeftModel) else contextlib.nullcontext()
    model.set_adapter(arm)
    return contextlib.nullcontext()


def _record(arm: str, i: int, row: dict, mnt: int, g: dict, batch: int) -> dict:
    seed_scorer()
    flags = follow_flags(row, g["response"])
    rec = {"key": f"{arm}:{i}", "arm": arm, "idx": i, "key_id": row.get("key"),
           "prompt_sha": sha(row["prompt"]), "mnt": mnt,
           "instruction_id_list": row["instruction_id_list"], "flags": flags, **g}
    return rec if batch <= 1 else {**rec, "batch": batch}


def _run_batched(model, tok, arm: str, rows: list[dict], mnt: int, tr: Transcript,
                 batch: int) -> None:
    # batches start at fixed offsets, so a resumed run keeps each prompt's neighbours
    t0, n_new = time.time(), 0
    for chunk in chunks(list(range(len(rows))), batch):
        if all(f"{arm}:{i}" in tr.done for i in chunk):
            continue
        gens = gen_logged_batch(model, tok, [rows[i]["prompt"] for i in chunk], mnt)
        tr.add([_record(arm, i, rows[i], mnt, g, batch)
                for i, g in zip(chunk, gens, strict=True) if f"{arm}:{i}" not in tr.done])
        n_new += len(chunk)
        print(f"[{arm}] {n_new} ({(time.time() - t0) / n_new:.2f}s/prompt, batch {batch})",
              flush=True)


def run_arm(model, tok, arm: str, rows: list[dict], mnt: int, out: Path,
            batch: int = 1) -> dict:
    tr = Transcript(out / f"{arm}.jsonl")
    for rec in tr.done.values():  # resume guard: never mix runs with different settings
        assert (rec["mnt"] == mnt and rec["prompt_sha"] == sha(rows[rec["idx"]]["prompt"])
                and rec.get("batch", 1) == batch), (
            f"{arm}: transcript {tr.path} was produced with different settings")
    todo = [(i, r) for i, r in enumerate(rows) if f"{arm}:{i}" not in tr.done]
    t0 = time.time()
    with adapter_context(model, arm):
        if batch > 1:
            _run_batched(model, tok, arm, rows, mnt, tr, batch)
        else:
            for n, (i, row) in enumerate(todo, 1):
                g = gen_logged(model, tok, row["prompt"], mnt)
                tr.add([_record(arm, i, row, mnt, g, batch)])
                if n % 25 == 0:
                    print(f"[{arm}] {n}/{len(todo)} ({(time.time() - t0) / n:.1f}s/prompt)",
                          flush=True)
    return summarize(arm, [tr.done[f"{arm}:{i}"] for i in range(len(rows))], out, batch)


def summarize(arm: str, recs: list[dict], out: Path, batch: int = 1) -> dict:
    inst = [f for r in recs for f in r["flags"]]
    n_inst, n_prompt = sum(inst), sum(all(r["flags"]) for r in recs)
    summary = {
        "arm": arm, "n_prompts": len(recs), "n_instructions": len(inst),
        "instruction_strict_count": n_inst, "prompt_strict_count": n_prompt,
        "instruction_strict_acc": round(n_inst / len(inst), 4),
        "prompt_strict_acc": round(n_prompt / len(recs), 4),
        "random_letter_prompts": {str(i): recs[i]["flags"] for i in RANDOM_LETTER_IDX
                                  if i < len(recs)},
        "mean_tokens": round(sum(r["n_tokens"] for r in recs) / len(recs), 1),
        "share_hit_cap": round(sum(not r["stopped_eos"] for r in recs) / len(recs), 4),
        "published_counts": PUBLISHED.get(arm),
        "reproduces_published": None,
    }
    if arm in PUBLISHED and len(recs) == 541:
        pi, pp = PUBLISHED[arm]
        if batch > 1:  # published counts are unbatched: a difference is batch drift
            summary["published_delta_informational"] = [n_inst - pi, n_prompt - pp]
        else:
            summary["reproduces_published"] = (abs(n_inst - pi) <= GATE_TOL
                                               and abs(n_prompt - pp) <= GATE_TOL)
    if batch > 1:
        summary["batch"] = batch
    write_json(out / f"{arm}_summary.json", summary)
    print(summary, flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", nargs="+", choices=list(ADAPTERS),
                    default=["undef", "def-s0", "def-s1", "def-s2"])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--n", type=int, default=0, help="0 = all 541 prompts (smoke test: 3)")
    ap.add_argument("--batch", type=int, default=1,
                    help="1 = unbatched (published path); N > 1 = left-padded batches of N")
    ap.add_argument("--out-dir", default=None,
                    help="default: runs/ifeval_logged (batch 1) or runs/ifeval_logged_b<N>")
    args = ap.parse_args()
    out = Path(args.out_dir) if args.out_dir else default_out(args.batch)
    if args.batch > 1 and out.resolve() == OUT.resolve():
        ap.error("batched runs must not write into the unbatched directory")
    rows = load_prompts(args.n)
    model, tok = load_model(args.arms)
    for arm in args.arms:
        run_arm(model, tok, arm, rows, args.max_new_tokens, out, args.batch)


if __name__ == "__main__":
    main()
