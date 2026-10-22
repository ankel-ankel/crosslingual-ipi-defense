"""Training sets of the comparison arms: one SFT example or DPO pair per training host.

Phases: render (NLLB wrapper translations), targets (frozen clean answers, unbatched), rejected
(frozen answers to the injection alone), build (per-arm JSONL + manifest), parity (optional).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ipidefense.attacks.recipes import (
    ALPACA_JSON,
    DATA_ROOT,
    LANGS,
    RENDER_CACHE,
    CachedTranslator,
    RefHost,
    audit_render,
    plan_a0,
    plan_a1,
    plan_b1,
    plan_b2,
    plan_b3,
    rejected_items,
    rejected_key,
    train_hosts,
    wrapper_segments,
)
from ipidefense.attacks.templates import EVAL_INJS, TRAIN_TEMPLATES
from ipidefense.data.splits import hosts as split_hosts
from ipidefense.gpumem import lean_kwargs, release
from ipidefense.model import QWEN_DIR, QWEN_INPUT_TEMPLATE, build_inputs, build_messages
from ipidefense.paths import MODELS, RUNS
from ipidefense.rendering.literal import render_inj_literal
from ipidefense.training.lora import SFTExample, encode_example
from ipidefense.utils import Item, Transcript, chunks, write_json

LOG = logging.getLogger("ipidefense.training.sets")

RUNS_DIR = RUNS / "baselines"
MODELS_DIR = MODELS
MAIN_RUNS = RUNS / "experiment"

TARGET_MNT = 160  # defense --max-new-tokens (secure targets)
EVAL_MNT = 64  # defense --eval-mnt
MAX_LEN = 1024  # train_sft max_len
SFT_ARMS = ("a0", "a1", "a2")
DPO_ARMS = ("b1", "b2", "b3")
ARMS = SFT_ARMS + DPO_ARMS
NEEDS_RENDER = frozenset({"a0", "a2", "b2"})
NEEDS_REJECTED = frozenset({"b1", "b2"})

GenFn = Callable[[list[Item]], list[dict]]


# ----------------------------------------------------------------------------- utilities
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """Records of a JSONL file, split on '\\n' only (as utils.Transcript does)."""
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: malformed JSON line") from exc
    return out


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Atomic write (temp file + replace), LF line ends on every platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def repair_tail(path: Path) -> None:
    """Drop a final record truncated by a crash, so that appends start on a fresh line."""
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


def seed_dir(seed: int, root: Path = DATA_ROOT, partial: bool = False) -> Path:
    return root / (f"seed{seed}-partial" if partial else f"seed{seed}")


@functools.cache
def qwen_tokenizer() -> Any:
    """The Qwen tokenizer with the input-role chat template used for training."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    tok.chat_template = QWEN_INPUT_TEMPLATE
    return tok


# ----------------------------------------------------------------------------- generation
def load_frozen(seed: int) -> tuple[Any, Any]:
    """The frozen model as the defense builds it (seeded LoRA wrapper, eval mode); generate
    inside model.disable_adapter()."""
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
    """Decoded text (as model.generate decodes it), content-token count, cap-hit flag."""
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
    """model.generate (stop_strings=None, inline=False), keeping the ids."""
    inputs = build_inputs(tok, instruction, data, False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=mnt, do_sample=False, pad_token_id=tok.eos_token_id
        )
    return gen_record(tok, out[0, inputs["input_ids"].shape[1] :].tolist(), mnt, eos)


def generate_many(
    model: Any, tok: Any, items: list[tuple[str, str | None]], mnt: int, eos: frozenset[int]
) -> list[dict]:
    """model.generate_batch (left padding, greedy), keeping the ids."""
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    side, tok.padding_side = tok.padding_side, "left"
    try:
        texts = [
            tok.apply_chat_template(
                build_messages(instr, data, False), add_generation_prompt=True, tokenize=False
            )
            for instr, data in items
        ]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=mnt, do_sample=False, pad_token_id=tok.eos_token_id,
                **lean_kwargs()
            )
    finally:
        tok.padding_side = side
    release()
    gen = out[:, enc["input_ids"].shape[1] :]
    return [gen_record(tok, g.tolist(), mnt, eos) for g in gen]


def run_generation(
    path: Path,
    items: list[Item],
    make_gen: Callable[[], GenFn],
    batch: int,
    limit: int | None,
    label: str,
) -> dict:
    """Resumable generation into an append-only transcript (one record per item key); the
    model is only loaded (make_gen) when something is pending."""
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


# ----------------------------------------------------------------------------- GPU phases
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
    """Frozen-model clean answers for the 1,500 training hosts, unbatched as in the defense."""
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


def phase_rejected(
    seed: int,
    batch: int = 16,
    limit: int | None = None,
    root: Path = DATA_ROOT,
    runs: Path = RUNS_DIR,
    cache: Path = RENDER_CACHE,
) -> dict:
    """Frozen-model greedy answers (<= 160 tokens) to each injection alone (b1, b2)."""
    t0 = time.time()
    items = rejected_items(seed, CachedTranslator.load(cache))
    path = seed_dir(seed, root) / "rejected.jsonl"

    def make_gen() -> GenFn:
        model, tok = load_frozen(seed)
        eos = eos_ids(model, tok)

        def gen(chunk: list[Item]) -> list[dict]:
            with model.disable_adapter():
                if batch <= 1:
                    return [
                        generate_one(model, tok, it.instruction, None, TARGET_MNT, eos)
                        for it in chunk
                    ]
                pairs = [(it.instruction, None) for it in chunk]
                return generate_many(model, tok, pairs, TARGET_MNT, eos)

        return gen

    stats = run_generation(path, items, make_gen, batch, limit, f"rejected seed{seed}")
    recs = list(Transcript(path).done.values())
    report = {
        "seed": seed,
        "max_new_tokens": TARGET_MNT,
        "batch": batch,
        **stats,
        "b1": gen_stats([r for r in recs if r["key"].startswith("b1:")]),
        "b2": gen_stats([r for r in recs if r["key"].startswith("b2:")]),
        "wall_s": round(time.time() - t0, 1),
        "finished": stamp(),
        "env": env_info(),
    }
    _write_report(runs, f"rejected_seed{seed}", report)
    return report


def phase_parity(seed: int = 0, runs: Path = RUNS_DIR, cache: Path = RENDER_CACHE) -> dict:
    """Undefended obey rates on the defense's eval hosts, unbatched, vs its training report."""
    from ipidefense.attacks.witnesses import OutcomeClassifier
    from ipidefense.evaluation.obey import obey_by_lang

    t0 = time.time()
    tr = CachedTranslator.load(cache)
    rendered = {
        ll: [(render_inj_literal(tr, inj, ll), inj.witness) for inj in EVAL_INJS] for ll in LANGS
    }
    model, tok = load_frozen(seed)
    with model.disable_adapter():
        now = obey_by_lang(
            model, tok, OutcomeClassifier(), split_hosts(seed, "security"), rendered, EVAL_MNT
        )
    main_path = MAIN_RUNS / f"seed{seed}" / "train.json"
    main_run = json.loads(main_path.read_text(encoding="utf-8"))["undefended"]
    diff = {
        ll: [round(a - b, 3) for a, b in zip(now[ll]["per_attack"], main_run[ll]["per_attack"],
                                             strict=True)]
        for ll in LANGS
    }
    report = {
        "seed": seed,
        "main_report": str(main_path),
        "per_attack_now": {ll: now[ll]["per_attack"] for ll in LANGS},
        "per_attack_main": {ll: main_run[ll]["per_attack"] for ll in LANGS},
        "diff": diff,
        "max_abs_diff": max(abs(x) for v in diff.values() for x in v),
        "rendered": rendered,
        "wall_s": round(time.time() - t0, 1),
        "env": env_info(),
    }
    _write_report(runs, f"parity_seed{seed}", report)
    return report


# ----------------------------------------------------------------------------- build (CPU)
@dataclass(frozen=True)
class BuildCtx:
    """skip: hosts with an empty frozen target, excluded from every arm (as in the defense);
    missing: hosts without a frozen target yet (partial smoke builds only)."""

    seed: int
    hosts: tuple[RefHost, ...]
    targets: dict[int, dict]
    skip: frozenset[int]
    tok: Any
    tr: Any = None
    rejected: dict[str, dict] | None = None
    missing: frozenset[int] = frozenset()

    @property
    def partial(self) -> bool:
        return bool(self.missing)

    @property
    def no_target(self) -> frozenset[int]:
        return self.skip | self.missing

    def rejected_text(self, key: str) -> str | None:
        rec = (self.rejected or {}).get(key)
        if rec is None and not self.partial:
            raise KeyError(f"no rejected response for {key}: finish the rejected phase")
        return None if rec is None else rec["text"]


def truncate_tokens(tok: Any, text: str, cap: int) -> tuple[str, int, bool]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= cap:
        return text, len(ids), False
    return tok.decode(ids[:cap]).rstrip("\ufffd"), cap, True


def _record(arm: str, ctx: BuildCtx, plan: dict, extra: dict, **responses: str) -> dict:
    i = plan["host"]
    return {
        "key": f"{arm}-s{ctx.seed}-h{i:04d}",
        "arm": arm,
        "seed": ctx.seed,
        "host": i,
        "messages": build_messages(ctx.hosts[i].instruction, plan["data"], False),
        **responses,
        "meta": {k: v for k, v in plan.items() if k not in ("host", "data")} | extra,
    }


def _frozen(t: dict) -> dict:
    return {"n_tokens": t["n_tokens"], "cap_hit": t["cap_hit"]}


def build_a0(ctx: BuildCtx) -> list[dict]:
    out = []
    for p in plan_a0(ctx.hosts, ctx.seed, ctx.tr, ctx.no_target):
        t = ctx.targets[p["host"]]
        out.append(_record("a0", ctx, p, {"target_src": "frozen", **_frozen(t)}, target=t["text"]))
    return out


def build_a2(ctx: BuildCtx) -> list[dict]:
    out = []
    for p in plan_a0(ctx.hosts, ctx.seed, ctx.tr, ctx.no_target):
        text, n_tok, cut = truncate_tokens(ctx.tok, ctx.hosts[p["host"]].output.strip(), TARGET_MNT)
        extra = {"target_src": "alpaca", "n_tokens": n_tok, "truncated": cut}
        out.append(_record("a2", ctx, p, extra, target=text))
    return out


def build_a1(ctx: BuildCtx) -> list[dict]:
    return [
        _record("a1", ctx, p, {"target_src": "alpaca"}, target=ctx.hosts[p["host"]].output.strip())
        for p in plan_a1(ctx.hosts, ctx.seed)
        if p["host"] not in ctx.skip
    ]


def _frozen_pairs(arm: str, ctx: BuildCtx, plan: list[dict]) -> list[dict]:
    out = []
    for p in plan:
        if p["host"] in ctx.no_target:
            continue
        rejected = ctx.rejected_text(rejected_key(arm, p))
        if rejected is None:
            continue
        t = ctx.targets[p["host"]]
        extra = {"chosen_src": "frozen", "rejected_src": "frozen", **_frozen(t)}
        out.append(_record(arm, ctx, p, extra, chosen=t["text"], rejected=rejected))
    return out


def build_b1(ctx: BuildCtx) -> list[dict]:
    return _frozen_pairs("b1", ctx, plan_b1(ctx.hosts, ctx.seed))


def build_b2(ctx: BuildCtx) -> list[dict]:
    return _frozen_pairs("b2", ctx, plan_b2(ctx.hosts, ctx.seed, ctx.tr))


def build_b3(ctx: BuildCtx) -> list[dict]:
    out = []
    for p in plan_b3(ctx.hosts, ctx.seed):
        if p["host"] in ctx.skip:
            continue
        chosen = ctx.hosts[p["host"]].output.strip()
        rejected = ctx.hosts[p["source"]].output.strip()
        extra = {"chosen_src": "alpaca", "rejected_src": "alpaca"}
        out.append(_record("b3", ctx, p, extra, chosen=chosen, rejected=rejected))
    return out


BUILDERS: dict[str, Callable[[BuildCtx], list[dict]]] = {
    "a0": build_a0, "a1": build_a1, "a2": build_a2,
    "b1": build_b1, "b2": build_b2, "b3": build_b3,
}


def token_lengths(tok: Any, messages: list[dict], response: str) -> tuple[int, int]:
    """(prompt tokens, full tokens) under encode_example's masking."""
    ids, labels = encode_example(tok, SFTExample(messages=messages, target=response))
    return int((labels == -100).sum()), len(ids)


def finalize(recs: list[dict], tok: Any, max_len: int = MAX_LEN) -> tuple[list[dict], dict]:
    """Drop records left without a supervised token after truncation; length statistics."""
    kept, dropped, prompts, seqs = [], [], [], []
    for rec in recs:
        responses = [rec["target"]] if "target" in rec else [rec["chosen"], rec["rejected"]]
        lens = [token_lengths(tok, rec["messages"], r) for r in responses]
        if any(p >= max_len for p, _ in lens):
            dropped.append(rec["key"])
            continue
        kept.append(rec)
        prompts.append(lens[0][0])
        seqs.extend(n for _, n in lens)
    stats: dict[str, Any] = {
        "kinds": dict(sorted(Counter(r["meta"]["kind"] for r in kept).items())),
        "prompt_tokens_max": max(prompts, default=0),
        "seq_tokens_mean": round(sum(seqs) / max(len(seqs), 1), 1),
        "seq_tokens_max": max(seqs, default=0),
        "n_seq_over_max_len": sum(n > max_len for n in seqs),
        "dropped_no_supervision": dropped,
    }
    if kept and "chosen" in kept[0]:
        stats["n_identical_pairs"] = sum(r["chosen"] == r["rejected"] for r in kept)
        stats["n_empty_rejected"] = sum(not r["rejected"].strip() for r in kept)
    if kept and kept[0]["arm"] == "a2":
        stats["n_target_truncated"] = sum(r["meta"]["truncated"] for r in kept)
    return kept, stats


def _host_records(hosts: tuple[RefHost, ...], targets: dict[int, dict]) -> list[dict]:
    """hosts.jsonl: host, Alpaca reference and frozen-model target side by side."""
    out = []
    for i, h in enumerate(hosts):
        t = targets.get(i, {})
        out.append({
            "host": i, "row": h.row, "instruction": h.instruction, "data": h.data,
            "reference": h.output, "target": t.get("text"), "n_tokens": t.get("n_tokens"),
            "cap_hit": t.get("cap_hit"), "ended_eos": t.get("ended_eos"),
        })
    return out


def _needed_rejected(
    arms: tuple[str, ...], hosts: tuple[RefHost, ...], seed: int, tr: Any
) -> set[str]:
    keys: set[str] = set()
    if "b1" in arms:
        keys |= {rejected_key("b1", p) for p in plan_b1(hosts, seed)}
    if "b2" in arms:
        keys |= {rejected_key("b2", p) for p in plan_b2(hosts, seed, tr)}
    return keys


def _load_targets(path: Path, n_hosts: int, allow_partial: bool) -> dict[int, dict]:
    if not path.exists():
        if allow_partial:
            return {}
        raise FileNotFoundError(f"{path} is missing: run the targets phase first")
    recs = {r["host"]: r for r in Transcript(path).done.values()}
    if len(recs) < n_hosts and not allow_partial:
        raise SystemExit(f"{path}: {len(recs)}/{n_hosts} targets; finish the targets phase "
                         "or pass --allow-partial (smoke tests only)")
    return recs


def phase_build(
    seed: int,
    arms: Iterable[str] = ARMS,
    root: Path = DATA_ROOT,
    allow_partial: bool = False,
    tr: Any = None,
    tok: Any = None,
    cache: Path = RENDER_CACHE,
) -> dict:
    """Write <root>/seed<k>/{arm}.jsonl, hosts.jsonl and manifest.json (SHA-256 of all)."""
    t0 = time.time()
    arms = tuple(arms)
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}")
    hs = train_hosts(seed)
    src = seed_dir(seed, root)
    targets = _load_targets(src / "targets.jsonl", len(hs), allow_partial)
    missing = frozenset(i for i in range(len(hs)) if i not in targets)
    skip = frozenset(i for i, t in targets.items() if not t["text"].strip())
    partial = bool(missing)
    out_dir = seed_dir(seed, root, partial=partial)
    if tr is None and NEEDS_RENDER & set(arms):
        tr = CachedTranslator.load(cache)
    tok = tok if tok is not None else qwen_tokenizer()
    rejected = None
    if NEEDS_REJECTED & set(arms):
        path = src / "rejected.jsonl"
        rejected = dict(Transcript(path).done) if path.exists() else {}
        lacking = _needed_rejected(arms, hs, seed, tr) - rejected.keys()
        if lacking and not partial:
            raise SystemExit(f"{path}: {len(lacking)} rejected responses missing; "
                             "finish the rejected phase first")
    ctx = BuildCtx(seed, hs, targets, skip, tok, tr, rejected, missing)
    inputs = {
        p.name: sha256_file(p)
        for p in (ALPACA_JSON, src / "targets.jsonl", src / "rejected.jsonl", cache)
        if p.exists()
    }
    manifest_path = out_dir / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    files = dict(old.get("files", {}))
    for arm in arms:
        recs, stats = finalize(BUILDERS[arm](ctx), tok)
        path = out_dir / f"{arm}.jsonl"
        write_jsonl(path, recs)
        files[arm] = {"file": path.name, "sha256": sha256_file(path), "n_records": len(recs),
                      **stats, "inputs": inputs, "built": stamp()}
        LOG.info("%s seed%d: %d records %s", arm, seed, len(recs), stats["kinds"])
    host_path = out_dir / "hosts.jsonl"
    write_jsonl(host_path, _host_records(hs, targets))
    files["hosts"] = {"file": host_path.name, "sha256": sha256_file(host_path),
                      "n_records": len(hs), "inputs": inputs, "built": stamp()}
    manifest = {
        "seed": seed,
        "partial": partial,
        "n_hosts": len(hs),
        "skipped_hosts": sorted(skip),
        "missing_targets": len(missing),
        "frozen_targets": gen_stats(list(targets.values())),
        "files": files,
        "updated": stamp(),
        "wall_s": round(time.time() - t0, 1),
    }
    write_json(manifest_path, manifest)
    LOG.info("wrote %s", manifest_path)
    return manifest


# ----------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="Build the comparison-arm training sets.")
    sub = ap.add_subparsers(dest="phase", required=True)
    p_render = sub.add_parser("render", help="GPU: NLLB wrapper translations (once)")
    p_render.add_argument("--allow-drift", action="store_true")
    p_tgt = sub.add_parser("targets", help="GPU: frozen clean answers, unbatched")
    p_rej = sub.add_parser("rejected", help="GPU: frozen answers to the injection alone")
    p_rej.add_argument("--batch", type=int, default=16)
    for p in (p_tgt, p_rej):
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--limit", type=int, help="generate at most N pending items (smoke)")
    p_build = sub.add_parser("build", help="CPU: per-arm JSONL + manifest")
    p_build.add_argument("--seed", type=int, required=True)
    p_build.add_argument("--arms", default=",".join(ARMS))
    p_build.add_argument("--allow-partial", action="store_true",
                         help="smoke only: build from incomplete GPU outputs into seed<k>-partial")
    p_par = sub.add_parser("parity", help="GPU: undefended obey rates vs the defense's report")
    p_par.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.phase == "render":
        phase_render(allow_drift=args.allow_drift)
    elif args.phase == "targets":
        phase_targets(args.seed, args.limit)
    elif args.phase == "rejected":
        phase_rejected(args.seed, args.batch, args.limit)
    elif args.phase == "build":
        arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
        phase_build(args.seed, arms, allow_partial=args.allow_partial)
    else:
        phase_parity(args.seed)


if __name__ == "__main__":
    main()
