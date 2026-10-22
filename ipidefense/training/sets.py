import argparse
import functools
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from ipidefense.attacks.recipes import (
    DATA_ROOT,
    RENDER_CACHE,
    CachedTranslator,
    audit_render,
    train_hosts,
    wrapper_segments,
)
from ipidefense.attacks.templates import EVAL_INJS, TRAIN_TEMPLATES
from ipidefense.model import QWEN_DIR, QWEN_INPUT_TEMPLATE, build_inputs
from ipidefense.paths import MODELS, RUNS
from ipidefense.training.lora import SFTExample, encode_example
from ipidefense.utils import Item, Transcript, chunks, write_json

LOG = logging.getLogger("ipidefense.training.sets")

RUNS_DIR = RUNS / "baselines"
MODELS_DIR = MODELS

TARGET_MNT = 160  # max new tokens of the clean targets
MAX_LEN = 1024  # train_sft max_len

GenFn = Callable[[list[Item]], list[dict]]


# utilities
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Atomic write (temp file + replace), LF line ends on every platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def repair_tail(path: Path) -> None:
    """Drop a final record truncated by a crash."""
    if not path.exists():
        return
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        path.write_bytes(raw[: raw.rfind(b"\n") + 1])
        LOG.warning("dropped a truncated final record from %s", path)


def env_info() -> dict[str, Any]:
    import peft
    import transformers

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def seed_dir(seed: int, root: Path = DATA_ROOT) -> Path:
    return root / f"seed{seed}"


@functools.cache
def qwen_tokenizer() -> Any:
    """The Qwen tokenizer with the input-role chat template used for training."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    tok.chat_template = QWEN_INPUT_TEMPLATE
    return tok


# generation
def load_frozen(seed: int) -> tuple[Any, Any]:
    """The frozen model as the defense builds it, generate under disable_adapter()."""
    from ipidefense.training.lora import build_lora_model

    torch.manual_seed(seed)
    model = build_lora_model(str(QWEN_DIR))
    model.eval()
    return model, qwen_tokenizer()


def eos_ids(model: Any, tok: Any) -> frozenset[int]:
    eos = model.generation_config.eos_token_id
    ids = list(eos) if isinstance(eos, list | tuple) else [eos]
    return frozenset(int(t) for t in [*ids, tok.eos_token_id] if t is not None)


def gen_record(tok: Any, new_ids: list[int], mnt: int, eos: frozenset[int]) -> dict:
    """Decoded text, content-token count and cap-hit flag."""
    stop = next((k for k, t in enumerate(new_ids) if t in eos), None)
    return {
        "text": tok.decode(new_ids, skip_special_tokens=True),
        "n_tokens": len(new_ids) if stop is None else stop,
        "ended_eos": stop is not None,
        "cap_hit": stop is None and len(new_ids) >= mnt,
    }


def generate_one(
    model: Any, tok: Any, instruction: str, data: str | None, mnt: int, eos: frozenset[int]
) -> dict:
    """model.generate (inline=False), keeping the ids."""
    inputs = build_inputs(tok, instruction, data, False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=mnt, do_sample=False, pad_token_id=tok.eos_token_id
        )
    return gen_record(tok, out[0, inputs["input_ids"].shape[1] :].tolist(), mnt, eos)


def run_generation(
    path: Path,
    items: list[Item],
    make_gen: Callable[[], GenFn],
    batch: int,
    limit: int | None,
    label: str,
) -> dict:
    """Resumable generation into an append-only transcript."""
    repair_tail(path)
    transcript = Transcript(path)
    todo = transcript.pending(items)
    if limit is not None:
        todo = todo[:limit]
    t0, done, load_s = time.time(), 0, 0.0
    if todo:
        gen = make_gen()
        load_s = time.time() - t0
        LOG.info("%s: %d pending of %d (model load %.0fs)", label, len(todo), len(items), load_s)
        for chunk in chunks(todo, max(1, batch)):
            t = time.time()
            outs = gen(chunk)
            sec = round((time.time() - t) / len(chunk), 3)
            transcript.add([
                {"key": it.key, **it.meta, "instruction": it.instruction, "data": it.data,
                 **out, "sec": sec}
                for it, out in zip(chunk, outs, strict=True)
            ])
            done += len(chunk)
            if done % 50 < len(chunk) or done == len(todo):
                rate = (time.time() - t0 - load_s) / done
                LOG.info("%s: %d/%d (%.2fs/item, eta %.1f min)", label, done, len(todo), rate,
                         rate * (len(todo) - done) / 60)
    keys = {it.key for it in items}
    complete = keys <= transcript.done.keys()
    return {
        "n_items": len(items),
        "n_done": len(keys & transcript.done.keys()),
        "generated_now": done,
        "complete": complete,
        "model_load_s": round(load_s, 1),
        "generation_s": round(time.time() - t0 - load_s, 1),
        "file": str(path),
        "sha256": sha256_file(path) if complete else None,
    }


def gen_stats(recs: list[dict]) -> dict:
    n = len(recs)
    toks = [r["n_tokens"] for r in recs]
    n_cap = sum(bool(r["cap_hit"]) for r in recs)
    return {
        "n": n,
        "n_cap_hit": n_cap,
        "cap_hit_share": round(n_cap / n, 4) if n else None,
        "n_empty": sum(not r["text"].strip() for r in recs),
        "tokens_mean": round(sum(toks) / n, 1) if n else None,
        "tokens_max": max(toks, default=0),
    }


def _write_report(runs: Path, name: str, payload: dict) -> None:
    path = runs / f"{name}.json"
    write_json(path, payload)
    LOG.info("wrote %s", path)


# GPU phases
def phase_render(
    langs: tuple[str, ...] = ("es", "vi"),
    cache: Path = RENDER_CACHE,
    runs: Path = RUNS_DIR,
    allow_drift: bool = False,
) -> dict:
    """Translate every TRAIN_TEMPLATES / EVAL_INJS wrapper segment once with NLLB."""
    from ipidefense.rendering.nllb import NLLB_MODEL, NLLBTranslator, load_nllb

    t0 = time.time()
    templates = [*TRAIN_TEMPLATES, *(inj.wrapper_template for inj in EVAL_INJS)]
    segments = wrapper_segments(templates)
    nllb = NLLBTranslator()
    table = {ll: {s: nllb.translate(s, ll) for s in segments} for ll in langs}
    nllb_model, _ = load_nllb()
    write_json(cache, {
        "model": NLLB_MODEL,
        "revision": getattr(nllb_model.config, "_commit_hash", None),
        "langs": list(langs),
        "templates": templates,
        "table": table,
        "created": stamp(),
    })
    report = {
        "file": str(cache),
        "sha256": sha256_file(cache),
        "n_segments": len(segments),
        "audit": audit_render(CachedTranslator(table), langs=langs),
        "wall_s": round(time.time() - t0, 1),
        "env": env_info(),
    }
    _write_report(runs, "render", report)
    if report["audit"].get("n_mismatch") and not allow_drift:
        raise SystemExit("rendering differs from the render audit (RUNS/render_audit.json); "
                         "see the render report")
    return report


def phase_targets(
    seed: int, limit: int | None = None, root: Path = DATA_ROOT, runs: Path = RUNS_DIR
) -> dict:
    """Frozen-model clean answers for the 1,500 training hosts, unbatched."""
    t0 = time.time()
    hs = train_hosts(seed)
    items = [Item(f"h{i:04d}", h.instruction, h.data, {"host": i}) for i, h in enumerate(hs)]
    path = seed_dir(seed, root) / "targets.jsonl"

    def make_gen() -> GenFn:
        model, tok = load_frozen(seed)
        eos = eos_ids(model, tok)

        def gen(chunk: list[Item]) -> list[dict]:
            with model.disable_adapter():
                return [
                    generate_one(model, tok, it.instruction, it.data, TARGET_MNT, eos)
                    for it in chunk
                ]

        return gen

    stats = run_generation(path, items, make_gen, 1, limit, f"targets seed{seed}")
    report = {
        "seed": seed,
        "max_new_tokens": TARGET_MNT,
        "batched": False,
        **stats,
        "targets": gen_stats(list(Transcript(path).done.values())),
        "wall_s": round(time.time() - t0, 1),
        "finished": stamp(),
        "env": env_info(),
    }
    _write_report(runs, f"targets_seed{seed}", report)
    return report


def token_lengths(tok: Any, messages: list[dict], response: str) -> tuple[int, int]:
    """(prompt tokens, full tokens) under encode_example's masking."""
    ids, labels = encode_example(tok, SFTExample(messages=messages, target=response))
    return int((labels == -100).sum()), len(ids)


# CLI
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="Build the render cache and the frozen targets.")
    sub = ap.add_subparsers(dest="phase", required=True)
    p_render = sub.add_parser("render", help="GPU: NLLB wrapper translations (once)")
    p_render.add_argument("--allow-drift", action="store_true")
    p_tgt = sub.add_parser("targets", help="GPU: frozen clean answers, unbatched")
    p_tgt.add_argument("--seed", type=int, required=True)
    p_tgt.add_argument("--limit", type=int, help="generate at most N pending items (smoke)")
    args = ap.parse_args(argv)
    if args.phase == "render":
        phase_render(allow_drift=args.allow_drift)
    else:
        phase_targets(args.seed, args.limit)


if __name__ == "__main__":
    main()
