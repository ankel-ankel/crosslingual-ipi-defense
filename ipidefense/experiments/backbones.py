"""SEA-LION and Mistral IFEval cost: target regeneration and census, cap-512 / cap-160 retraining,
security checks through the suite, IFEval breakdown and decision.

Usage: backbones.py targets|census|train|sec|breakdown|decide [--backbone BB]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from ipidefense.arms import (
    ADAPTERS,
    BACKBONE_CONFIGS,
    BACKBONES,
    backbone_adapter_dir,
    check_adapter_base,
)
from ipidefense.attacks.recipes import (
    ALPACA_JSON,
    N_TRAIN,
    RENDER_CACHE,
    CachedTranslator,
    audit_render,
    plan_a0,
    train_hosts,
)
from ipidefense.evaluation.ifeval_logged import default_out
from ipidefense.evaluation.runtime import parse_arm, rate
from ipidefense.experiments import ifeval_breakdown as bd
from ipidefense.experiments import suite as su
from ipidefense.experiments.backbone_rules import (
    CENSUS_DEFS,
    CONFIGS,
    IFEVAL_BATCH,
    IFEVAL_MNT,
    N_INSTRUCTIONS,
    N_PROMPTS,
    N_SEC_HOSTS,
    OPERATIONAL,
    PUBLISHED_CONFIG,
    RULES,
    SEC_BATCH,
    SEC_LANGS,
    SEC_MNT,
    SEED,
    SIX_LANGS,
    TARGET_BATCH,
    arm,
    best_config,
    borderline,
    brief,
    check_backbone,
    classify,
    config_row,
    prefix_agreement,
    strip_exact,
    target_stats,
    tax_ci95,
)
from ipidefense.gpumem import lean_kwargs, release
from ipidefense.model import MISTRAL_DIR, QWEN_DIR, SEALION_DIR, build_messages
from ipidefense.paths import DATA, MODELS, RUNS
from ipidefense.training.lora import LORA_TARGETS, SFTExample
from ipidefense.training.sets import (
    MAX_LEN,
    env_info,
    eos_ids,
    gen_record,
    repair_tail,
    sha256_file,
    stamp,
    token_lengths,
    write_jsonl,
)
from ipidefense.training.sft import (
    LossTee,
    adapter_dir,
    adapter_hashes,
    length_stats,
    lora_settings,
    select,
)
from ipidefense.utils import Item, Transcript, chunks, sha, write_json

LOG = logging.getLogger(__name__)

OUT = RUNS / "backbones"
DATA_DIR = DATA / "backbones"
PUBLISHED_RUNS = RUNS  # unbatched utility.py reports of the published adapters
IFEVAL_DIR = default_out(IFEVAL_BATCH)  # runs/ifeval_logged_b16, shared with the hardened adapters
QWEN_TARGETS = DATA / "baselines" / f"seed{SEED}" / "targets.jsonl"  # unbatched
MODEL_DIRS = {"qwen": QWEN_DIR, "sealion": SEALION_DIR, "mistral": MISTRAL_DIR}


def backbone_tokenizer(bb: str) -> Any:
    """train.py's tokenizer: SEA-LION's with the Meta-SecAlign (Llama-3.1, input-role) template;
    Mistral's own tokenizer and template (no input role)."""
    from ipidefense.model import mistral_tokenizer, sealion_tokenizer

    return {"sealion": sealion_tokenizer, "mistral": mistral_tokenizer}[check_backbone(bb)]()


def generation_settings(bb: str) -> dict:
    """Decoding-relevant fields of a backbone's generation_config.json (model.generate uses
    them: a repetition_penalty applies under greedy decoding too)."""
    cfg = json.loads((MODEL_DIRS[bb] / "generation_config.json").read_text(encoding="utf-8"))
    return {"repetition_penalty": cfg.get("repetition_penalty", 1.0),
            "eos_token_id": cfg.get("eos_token_id"),
            "sampling_fields_ignored": {k: cfg[k] for k in ("do_sample", "temperature", "top_p",
                                                            "top_k") if k in cfg},
            "decoding": "greedy (do_sample=False), pad = EOS, left padding"}


def report_path(name: str, smoke: bool = False) -> Path:
    return OUT / ("smoke" if smoke else "") / name


# ===================================================================== targets, GPU
def targets_path(bb: str, cap: int, smoke: bool = False) -> Path:
    return DATA_DIR / bb / ("smoke" if smoke else "") / f"targets_cap{cap}.jsonl"


def target_items(limit: int | None = None) -> list[Item]:
    """The 1,500 seed-0 training hosts of train.py (first 1,500 of load_en_hosts(1650, seed))."""
    items = [Item(f"h{i:04d}", h.instruction, h.data, {"host": i})
             for i, h in enumerate(train_hosts(SEED))]
    return items[:limit] if limit else items


def chat_texts(tok: Any, items: Sequence[Item], inline: bool) -> dict[str, str]:
    return {it.key: tok.apply_chat_template(build_messages(it.instruction, it.data, inline),
                                            add_generation_prompt=True, tokenize=False)
            for it in items}


def generate_targets(model: Any, tok: Any, texts: list[str], mnt: int,
                     eos: frozenset[int]) -> list[dict]:
    """ipidefense.model.generate_batch (left padding, greedy, pad = EOS) on rendered chat texts,
    also keeping the content token ids (their decode equals the text)."""
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    side, tok.padding_side = tok.padding_side, "left"
    try:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                 pad_token_id=tok.eos_token_id, **lean_kwargs())
    finally:
        tok.padding_side = side
    release()
    recs = []
    for g in out[:, enc["input_ids"].shape[1]:]:
        ids = g.tolist()
        rec = gen_record(tok, ids, mnt, eos)
        recs.append({**rec, "ids": ids[: rec["n_tokens"]]})
    return recs


def load_frozen(bb: str) -> Any:
    """The frozen backbone as train.py builds it (manual_seed, LoRA wrapper, eval mode);
    targets are generated inside model.disable_adapter()."""
    from ipidefense.training.lora import build_lora_model

    torch.manual_seed(SEED)
    model = build_lora_model(str(MODEL_DIRS[bb]))
    model.eval()
    return model


def run_targets(path: Path, items: Sequence[Item], texts: dict[str, str],
                make_gen: Callable[[], Callable[[list[str]], list[dict]]], batch: int,
                mnt: int) -> dict:
    """Resumable generation in fixed consecutive chunks of `batch` items, so every item keeps the
    batch neighbours of train.py --batch even after a crash (a partly written chunk is generated
    again whole and only its missing records are added). The model loads only when needed."""
    repair_tail(path)
    tr = Transcript(path)
    for it in items:
        rec = tr.done.get(it.key)
        if rec and (rec["prompt_sha"], rec["mnt"], rec["batch"]) != (sha(texts[it.key]), mnt,
                                                                     batch):
            raise RuntimeError(f"{path}: {it.key} was generated with another prompt or settings")
    gen, t0, n_new = None, time.time(), 0
    todo = len(tr.pending(items))
    for chunk in chunks(list(items), batch):
        if all(it.key in tr.done for it in chunk):
            continue
        gen = gen or make_gen()
        t = time.time()
        outs = gen([texts[it.key] for it in chunk])
        sec = round((time.time() - t) / len(chunk), 3)
        tr.add([{"key": it.key, **it.meta, "instruction": it.instruction, "data": it.data,
                 "prompt_sha": sha(texts[it.key]), "mnt": mnt, "batch": batch, **o, "sec": sec}
                for it, o in zip(chunk, outs, strict=True) if it.key not in tr.done])
        n_new += len(chunk)
        if n_new % (10 * batch) < batch or n_new >= todo:
            LOG.info("%s: %d/%d (%.2fs/item)", path.name, n_new, todo,
                     (time.time() - t0) / n_new)
    keys = [it.key for it in items]
    complete = all(k in tr.done for k in keys)
    return {"n_items": len(keys), "n_done": sum(k in tr.done for k in keys),
            "generated_now": n_new, "complete": complete, "file": str(path),
            "sha256": sha256_file(path) if complete else None,
            "wall_s": round(time.time() - t0, 1)}


def cmd_targets(bb: str, cap: int, limit: int | None = None) -> dict:
    """Frozen-base clean answers of the 1,500 seed-0 training hosts (`limit`: smoke subset,
    written under data/backbones/<bb>/smoke so it never mixes with the full run)."""
    smoke = limit is not None
    tok, inline = backbone_tokenizer(bb), BACKBONES[check_backbone(bb)][1]
    items = target_items(limit)
    texts = chat_texts(tok, items, inline)
    path = targets_path(bb, cap, smoke)

    def make_gen() -> Callable[[list[str]], list[dict]]:
        model = load_frozen(bb)
        eos = eos_ids(model, tok)

        def gen(batch_texts: list[str]) -> list[dict]:
            with model.disable_adapter():
                return generate_targets(model, tok, batch_texts, cap, eos)

        return gen

    stats = run_targets(path, items, texts, make_gen, TARGET_BATCH, cap)
    recs = list(Transcript(path).done.values())
    report = {"backbone": bb, "cap": cap, "batch": TARGET_BATCH, "inline": inline, "seed": SEED,
              "smoke": smoke, **stats, "stats": target_stats(recs, cap),
              "generation": generation_settings(bb), "finished": stamp(), "env": env_info()}
    write_json(report_path(f"targets_{bb}_cap{cap}.json", smoke), report)
    return report


# ===================================================================== census, CPU
def load_complete(path: Path, n: int | None = None) -> dict[int, dict] | None:
    """Target records by host if the file holds every training host, else None."""
    n = N_TRAIN if n is None else n
    if not path.exists():
        return None
    recs = {r["host"]: r for r in Transcript(path).done.values()}
    return recs if set(recs) == set(range(n)) else None


def census(qwen_path: Path = QWEN_TARGETS, out: Path | None = None) -> dict:
    """runs/backbones/census.json: every complete target set, missing sets listed (never
    partial)."""
    res: dict[str, Any] = {"definitions": CENSUS_DEFS,
                           "generation": {bb: generation_settings(bb) for bb in MODEL_DIRS},
                           "backbones": {}, "missing": [], "created": stamp()}
    qwen = load_complete(qwen_path)
    if qwen is None:
        res["missing"].append(str(qwen_path))
    else:
        res["backbones"]["qwen"] = {"cap160": {
            "source": f"{qwen_path} (comparison-arm targets: unbatched, as train.py ran the main "
                      f"Qwen seeds)",
            "batch": 1, **target_stats(list(qwen.values()), 160)}}
    for bb in BACKBONES:
        sets = {}
        for cap in (160, 512):
            path = targets_path(bb, cap)
            recs = load_complete(path)
            if recs is None:
                res["missing"].append(str(path))
                continue
            sets[cap] = recs
            res["backbones"].setdefault(bb, {})[f"cap{cap}"] = {
                "source": str(path), "batch": TARGET_BATCH,
                **target_stats(list(recs.values()), cap)}
        if len(sets) == 2:
            res["backbones"][bb]["prefix_agreement"] = prefix_agreement(sets[160], sets[512])
    write_json(out or OUT / "census.json", res)
    return res


# ===================================================================== build + train
def load_targets(bb: str, cap: int, smoke: bool = False) -> tuple[dict[int, dict], Path]:
    """Targets by host. Full runs need every host from this protocol (cap, batch 16) and a file
    whose SHA-256 matches its targets report."""
    path = targets_path(bb, cap, smoke)
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `backbones.py targets --backbone {bb} "
                         f"--cap {cap}`" + (" --limit N" if smoke else
                                            " (step targets of scripts/backbones.py)"))
    recs = {r["host"]: r for r in Transcript(path).done.values()}
    bad = [h for h, r in recs.items() if (r["mnt"], r["batch"]) != (cap, TARGET_BATCH)]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} records are not cap {cap} / batch {TARGET_BATCH}")
    if not smoke:
        if set(recs) != set(range(N_TRAIN)):
            raise SystemExit(f"{path}: {len(recs)}/{N_TRAIN} targets; finish the targets phase")
        rep = report_path(f"targets_{bb}_cap{cap}.json")
        digest = json.loads(rep.read_text(encoding="utf-8")).get("sha256") if rep.exists() else None
        if digest != sha256_file(path):
            raise SystemExit(f"{path} does not match the SHA-256 in {rep}; re-run `targets`")
    return recs, path


def build_records(bb: str, cfg: str, targets: dict[int, dict], tr: Any,
                  partial: bool = False) -> list[dict]:
    """The train.py mixture on this backbone's targets: plan_a0 draws (an empty target consumes
    no draw), as-run rendering, build_messages with the backbone's inline flag, raw target text.
    `partial` (smoke): hosts without a target are skipped like empty ones."""
    hs = train_hosts(SEED)
    inline = BACKBONES[bb][1]
    missing = frozenset(i for i in range(len(hs)) if i not in targets)
    if missing and not partial:
        raise ValueError(f"{len(missing)} training hosts have no target")
    skip = frozenset(i for i, t in targets.items() if not t["text"].strip())
    out = []
    for p in plan_a0(hs, SEED, tr, skip | missing):
        t = targets[p["host"]]
        meta = {k: p[k] for k in ("kind", "template", "witness", "lang") if k in p}
        out.append({"key": f"s5-{bb}-cfg{cfg}-h{p['host']:04d}", "backbone": bb, "cfg": cfg,
                    "seed": SEED, "host": p["host"],
                    "messages": build_messages(hs[p["host"]].instruction, p["data"], inline),
                    "target": t["text"],
                    "meta": {**meta, "n_tokens": t["n_tokens"], "cap_hit": t["cap_hit"],
                             "ended_eos": t["ended_eos"]}})
    return out


def as_run_translator() -> tuple[Any, dict]:
    """The as-run NLLB rendering (render cache in data/baselines), re-audited against the original
    renderer's strings (runs/render_audit.json) before every build."""
    tr = CachedTranslator.load(RENDER_CACHE)
    audit = audit_render(tr)
    if not audit.get("available") or audit.get("n_mismatch"):
        raise SystemExit(f"render audit failed for {RENDER_CACHE}: {audit}")
    return tr, {"file": str(RENDER_CACHE), "sha256": sha256_file(RENDER_CACHE),
                "audit": {k: audit[k] for k in ("n_rows", "n_mismatch")}}


def cmd_train(args: argparse.Namespace) -> dict:
    bb, cfg = check_backbone(args.backbone), args.cfg
    spec = CONFIGS[cfg]
    smoke = args.smoke or args.limit is not None
    run_name = args.run_name or (f"smoke-{bb}-cfg{cfg}" if smoke else f"{bb}-cfg{cfg}-seed0")
    if smoke and not run_name.startswith("smoke"):
        raise SystemExit("--smoke / --limit runs need a run name starting with 'smoke'")
    if not smoke and run_name != backbone_adapter_dir(bb, cfg).name:
        raise SystemExit(f"a full run must write {backbone_adapter_dir(bb, cfg)} (registry)")
    # never over an existing adapter; checked first so a used name fails fast
    out = None if args.dry_run else adapter_dir(run_name, args.overwrite)
    started, t_start = stamp(), time.time()
    targets, tpath = load_targets(bb, int(spec["cap"]), args.smoke)
    tr, render = as_run_translator()
    recs = build_records(bb, cfg, targets, tr, partial=args.smoke)
    data_path = DATA_DIR / bb / ("smoke" if args.smoke else "") / f"cfg{cfg}.jsonl"
    write_jsonl(data_path, recs)
    tok = backbone_tokenizer(bb)
    lens = [token_lengths(tok, r["messages"], r["target"]) for r in recs]
    keep = select(len(recs), args.limit, args.longest, [f for _, f in lens])
    lstats = length_stats([lens[k] for k in keep], MAX_LEN)
    examples = [SFTExample(messages=recs[k]["messages"], target=recs[k]["target"]) for k in keep]
    kinds = Counter(recs[k]["meta"]["kind"] for k in keep)
    config = {"backbone": bb, "cfg": cfg, "run_name": run_name, "seed": SEED, "cap": spec["cap"],
              "lr": spec["lr"], "published_config": PUBLISHED_CONFIG,
              "base": str(MODEL_DIRS[bb]), "inline": BACKBONES[bb][1], "objective": "sft",
              "epochs": 1, "micro_batch": 1, "accum": 8, "max_len": MAX_LEN, "clip": 1.0,
              "optimizer": "torch.optim.AdamW (defaults), constant lr", "precision": "bf16",
              "gradient_checkpointing": True,
              "loss": "assistant tokens only (training.lora.encode_example)",
              "lora_planned": {"r": 16, "alpha": 32, "dropout": 0.05,
                               "targets": sorted(LORA_TARGETS)}}
    data = {"file": str(data_path), "sha256": sha256_file(data_path), "n_records": len(recs),
            "n_trained": len(keep), "kinds": dict(kinds), "smoke": smoke,
            "skipped_hosts_empty_target": sorted(i for i, t in targets.items()
                                                 if not t["text"].strip()),
            "targets": {"file": str(tpath), "sha256": sha256_file(tpath), "cap": spec["cap"],
                        "batch": TARGET_BATCH},
            "render_cache": render, "alpaca_sha256": sha256_file(ALPACA_JSON)}
    LOG.info("%s: %d examples %s, %s", run_name, len(keep), dict(kinds), lstats)
    if args.dry_run:
        print(json.dumps({"config": config, "data": data, "lengths": lstats}, indent=1))
        return {"config": config, "data": data, "lengths": lstats}
    from ipidefense.training.lora import build_lora_model, train_sft

    t0 = time.time()
    torch.manual_seed(SEED)  # train.py seeds before building the LoRA model
    model = build_lora_model(str(MODEL_DIRS[bb]))
    t1 = time.time()
    tee = LossTee(sys.stdout)
    with contextlib.redirect_stdout(tee):  # log_every only changes printing (smoke: few steps)
        train_sft(model, tok, examples, epochs=1, lr=spec["lr"], accum=8, max_len=MAX_LEN,
                  log_every=1 if smoke else 10, seed=SEED)
    t2 = time.time()
    finite = bool(tee.losses) and all(math.isfinite(x["loss"]) for x in tee.losses)
    report: dict[str, Any] = {"config": config | {"lora": lora_settings(model)}, "data": data,
                              "lengths": lstats, "losses": tee.losses, "finite": finite,
                              "started": started, "finished": stamp(), "env": env_info()}
    if finite:
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out))
        report["adapter"] = {"dir": str(out), "files": adapter_hashes(out)}
    report["wall_s"] = {"prep": round(t0 - t_start, 1), "load": round(t1 - t0, 1),
                        "train": round(t2 - t1, 1), "total": round(time.time() - t_start, 1)}
    write_json(report_path(f"train_{run_name}.json", smoke), report)
    if not finite:
        raise SystemExit("non-finite (or no logged) loss: adapter NOT saved")
    return report


# ===================================================================== security, GPU
def suite_dir(bb: str) -> Path:
    return OUT / f"suite_{bb}"


def suite_args(bb: str, arms: Sequence[str], langs: Sequence[str],
               limit_hosts: int = 0) -> argparse.Namespace:
    """suite `run --exp ml1b --renderer v2` arguments (the ml1b security protocol)."""
    return argparse.Namespace(cmd="run", exp="ml1b", renderer="v2", arms=list(arms),
                              seeds=[SEED], langs=list(langs), batch=SEC_BATCH, mnt=SEC_MNT,
                              limit_hosts=limit_hosts, out_dir=str(suite_dir(bb)),
                              with_trainpool=False, no_bt=False)


def require_adapters(arms: Sequence[str]) -> None:
    """Fail before any GPU load if an adapter is missing or was trained on another backbone."""
    for name in arms:
        adir = ADAPTERS[name][1]
        if adir and not (MODELS / adir / "adapter_model.safetensors").exists():
            raise SystemExit(f"{name}: no adapter at {MODELS / adir}; train it first")
        check_adapter_base(name)


def sec_plan(bb: str, six: bool, arms: Sequence[str] | None = None,
             best: str | None = None) -> tuple[list[str], tuple[str, ...]]:
    """(arms, langs) of a security run: en/es/vi for undef, published and both configs; --six:
    the undefended backbone and the best config (`best`) in 6 languages; or explicit arms."""
    langs = SIX_LANGS if six else SEC_LANGS
    if arms:
        other = [a for a in arms if ADAPTERS.get(a, ("",))[0] != bb]
        if other:
            raise SystemExit(f"arms of another backbone than {bb}: {other}")
        return list(dict.fromkeys(arms)), langs
    if six:
        if best not in BACKBONE_CONFIGS:
            raise SystemExit("--six needs the best config (complete en/es/vi security + IFEval)")
        return [arm(bb, "undef"), arm(bb, best)], langs
    return [arm(bb, k) for k in ("undef", "def", *BACKBONE_CONFIGS)], langs


def cmd_sec(bb: str, six: bool, arms: Sequence[str] | None = None, limit_hosts: int = 0) -> None:
    best = None
    if six and not arms:
        if limit_hosts:
            raise SystemExit("--six reads the full en/es/vi results; no --limit-hosts")
        res = backbone_results(check_backbone(bb))
        if res["rows"] is None:
            raise SystemExit(f"best config undefined, missing: {res['missing']}")
        best = best_config(res["rows"])
        write_json(OUT / f"best_{bb}.json", {
            "backbone": bb, "best_config": best, "arm": arm(bb, best), "rule":
            OPERATIONAL["best_config"], "configs": strip_exact(res["rows"]), "chosen": stamp()})
    plan_arms, langs = sec_plan(check_backbone(bb), six, arms, best)
    require_adapters(plan_arms)
    su.run_exp(suite_args(bb, plan_arms, langs, limit_hosts))


def valid_tids(lang: str) -> list[str] | None:
    """EVAL_INJS templates whose v2 rendering keeps its witness (suite.run_exp's filter);
    None if a rendering is not cached yet."""
    recs = su.load_renderings(lang, "v2")
    tmpl, _ = su.exp_templates("ml1b", lang)
    if any(t.tid not in recs for t in tmpl):
        return None
    return [t.tid for t in tmpl if recs[t.tid]["rendered"].count(t.witness) == 1]


def security_cell(bb: str, name: str, langs: Sequence[str],
                  out: Path | None = None) -> dict | None:
    """Per-language witness-first / refusal-first obey counts of one arm from its suite
    transcript; None unless every expected record (valid template x 150 hosts) is present."""
    path = su.transcript_path(out or suite_dir(bb), parse_arm(name), SEED, "ml1b", "v2")
    recs = su.read_records(path)
    cell = {}
    for lang in langs:
        tids = valid_tids(lang)
        if not tids:
            return None
        keys = [f"{lang}|{tid}|{i}" for tid in tids for i in range(N_SEC_HOSTS)]
        if any(k not in recs for k in keys):
            return None
        rs = [recs[k] for k in keys]
        if any((r["mnt"], r["batch"]) != (SEC_MNT, SEC_BATCH) for r in rs):
            raise SystemExit(f"{path}: {lang} records are not {SEC_MNT} tokens / batch {SEC_BATCH}")
        cell[lang] = {"wf_k": sum(r["wf"] == "obeyed" for r in rs),
                      "rf_k": sum(r["rf"] == "obeyed" for r in rs), "n": len(rs)}
    return cell


# ===================================================================== IFEval, CPU
def load_ifeval_records(name: str, directory: Path = IFEVAL_DIR) -> dict[int, dict] | None:
    """All 541 logged records of one arm (batch 16, 512 tokens), else None."""
    path = directory / f"{name}.jsonl"
    if not path.exists():
        return None
    recs = {r["idx"]: r for r in Transcript(path).done.values()}
    if set(recs) != set(range(N_PROMPTS)):
        return None
    bad = [i for i, r in recs.items() if (r.get("batch", 1), r["mnt"]) != (IFEVAL_BATCH,
                                                                          IFEVAL_MNT)]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} records are not batch {IFEVAL_BATCH} / "
                         f"{IFEVAL_MNT} tokens")
    return recs


def ifeval_counts(name: str, directory: Path = IFEVAL_DIR) -> dict | None:
    recs = load_ifeval_records(name, directory)
    if recs is None:
        return None
    flags = [f for r in recs.values() for f in r["flags"]]
    if len(flags) != N_INSTRUCTIONS:
        raise SystemExit(f"{name}: {len(flags)} instruction flags, expected {N_INSTRUCTIONS}")
    return {"k": sum(flags), "n": len(flags),
            "prompt_k": sum(all(r["flags"]) for r in recs.values()), "prompt_n": len(recs),
            "per_prompt": [(sum(recs[i]["flags"]), len(recs[i]["flags"]))
                           for i in range(N_PROMPTS)]}


def with_tax_ci(row: dict, undef: dict, ife: dict) -> dict:
    """A rule row plus its descriptive tax CI and borderline flag (never used by the rule)."""
    ci = tax_ci95(undef["per_prompt"], ife["per_prompt"])
    return {**row, "tax_ci95_descriptive": ci, "borderline": borderline(ci)}


def cmd_breakdown(n_boot: int = 10_000, directory: Path = IFEVAL_DIR,
                  out: Path = OUT) -> dict[str, str]:
    """IFEval breakdown (ifeval_breakdown.analyze) of every complete defended backbone arm
    against its undefended backbone."""
    labels = json.loads(bd.LABELS.read_text(encoding="utf-8"))["labels"]
    status = {}
    for bb in BACKBONES:
        undef = load_ifeval_records(arm(bb, "undef"), directory)
        for kind in ("def", *BACKBONE_CONFIGS):
            name = arm(bb, kind)
            dfd = load_ifeval_records(name, directory)
            if undef is None or dfd is None:
                status[name] = "pending"
                continue
            res = bd.analyze(undef, [dfd], labels, n_boot)
            if res["sensitivity_without_long"]["n_dropped"] != bd.N_LONG:
                raise SystemExit(f"expected {bd.N_LONG} long instructions")
            write_json(out / f"breakdown_{name}.json", {
                "dir": str(directory), "undef": arm(bb, "undef"), "arms": [name],
                "n_boot": n_boot, "labels_file": str(bd.LABELS),
                "note": "subset definitions of ifeval_breakdown.analyze, one defended seed",
                **res})
            status[name] = "written"
    LOG.info("breakdown: %s", status)
    return status


# ===================================================================== decide, CPU
def backbone_results(bb: str, ifeval_dir: Path = IFEVAL_DIR,
                     suite: Path | None = None) -> dict:
    """Rule inputs of one backbone: rows of cfg-A/cfg-B, or None with the missing inputs."""
    kinds = ("undef", "def", *BACKBONE_CONFIGS)
    ife = {k: ifeval_counts(arm(bb, k), ifeval_dir) for k in kinds}
    sec = {k: security_cell(bb, arm(bb, k), SEC_LANGS, suite) for k in kinds}
    missing = [f"ifeval:{arm(bb, k)}" for k in ("undef", *BACKBONE_CONFIGS) if ife[k] is None]
    missing += [f"security(en/es/vi):{arm(bb, k)}" for k in BACKBONE_CONFIGS if sec[k] is None]
    rows = None if missing else {
        c: with_tax_ci(config_row(ife["undef"], ife[c], sec[c]), ife["undef"], ife[c])
        for c in BACKBONE_CONFIGS}
    return {"ifeval": ife, "security": sec, "rows": rows, "missing": missing}


def published_reference(bb: str, ife: dict) -> dict:
    """The published (unbatched utility.py) tax and its batched re-measurement."""
    pub = PUBLISHED_RUNS
    u = json.loads((pub / f"eval_utility_{bb}_undefended.json").read_text(encoding="utf-8"))
    d = json.loads((pub / f"eval_utility_{bb}_lora_{BACKBONES[bb][0]}.json")
                   .read_text(encoding="utf-8"))
    ref: dict[str, Any] = {"config": PUBLISHED_CONFIG, "unbatched_published": {
        "undef": u["instruction_strict_acc"], "def": d["instruction_strict_acc"],
        "tax_pp": round(100 * (u["instruction_strict_acc"] - d["instruction_strict_acc"]), 2)}}
    if ife["undef"] and ife["def"]:
        acc = {k: ife[k]["k"] / ife[k]["n"] for k in ("undef", "def")}
        ref["batched"] = {"undef": round(acc["undef"], 4), "def": round(acc["def"], 4),
                          "tax_pp": round(100 * (acc["undef"] - acc["def"]), 2)}
    return ref


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def descriptive(bb: str, rows: dict | None, res: dict, suite: Path | None,
                runs: Path) -> dict:
    """Published-adapter sensitivity, six-language check, census and breakdown summaries."""
    out: dict[str, Any] = {}
    pub_sec, ife = res["security"]["def"], res["ifeval"]
    if rows is not None and ife["def"] and pub_sec:
        pub_row = with_tax_ci(config_row(ife["undef"], ife["def"], pub_sec), ife["undef"],
                              ife["def"])
        with_pub = {**rows, "published": pub_row}
        out["published_row"] = strip_exact({"published": with_pub["published"]})["published"]
        out["label_if_published_counted"] = classify(with_pub)
    if rows is not None:
        best = best_config(rows)
        six = {k: security_cell(bb, arm(bb, k), SIX_LANGS, suite) for k in ("undef", best)}
        out["six_language_check"] = {
            "best_config": best, "complete": all(v is not None for v in six.values()),
            **{arm(bb, k): None if v is None else {
                lang: {"wf": rate(c["wf_k"], c["n"]), "rf": rate(c["rf_k"], c["n"])}
                for lang, c in v.items()} for k, v in six.items()}}
    cen = _read_json(runs / "census.json")
    if cen and bb in cen.get("backbones", {}):
        out["census"] = {part: brief(st) for part, st in cen["backbones"][bb].items()}
        out["census"]["generation"] = cen["generation"].get(bb)
    brk = {}
    for kind in ("def", *BACKBONE_CONFIGS):
        b = _read_json(runs / f"breakdown_{arm(bb, kind)}.json")
        if b:
            brk[arm(bb, kind)] = {"subsets": {k: v["delta"] for k, v in b["subsets"].items()},
                                  "hypotheses": b["hypotheses"]}
    out["breakdown"] = brk
    return out


def decide(ifeval_dir: Path = IFEVAL_DIR, runs: Path = OUT,
           suites: dict[str, Path] | None = None) -> dict:
    """runs/backbones/decision.json: the decision rule per backbone ('pending' until every input
    exists)."""
    result: dict[str, Any] = {"rules": RULES, "operational": OPERATIONAL, "backbones": {},
                              "created": stamp()}
    for bb in BACKBONES:
        suite = (suites or {}).get(bb)
        res = backbone_results(bb, ifeval_dir, suite)
        rows = res["rows"]
        entry: dict[str, Any] = {
            "decision": "pending" if rows is None else classify(rows),
            "missing": res["missing"],
            "best_config": None if rows is None else best_config(rows),
            "configs": None if rows is None else strip_exact(rows),
            "published_reference": published_reference(bb, res["ifeval"]),
            **descriptive(bb, rows, res, suite, runs)}
        result["backbones"][bb] = entry
    write_json(runs / "decision.json", result)
    LOG.info("decision: %s", {bb: e["decision"] for bb, e in result["backbones"].items()})
    return result


# ===================================================================== CLI
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_t = sub.add_parser("targets", help="GPU: frozen-base clean answers (batched as run)")
    p_t.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    p_t.add_argument("--cap", type=int, choices=(160, 512), required=True)
    p_t.add_argument("--limit", type=int,
                     help="smoke: first N hosts, into data/backbones/<bb>/smoke")
    sub.add_parser("census", help="CPU: truncation / loop / EOS rates of every target set")
    p_tr = sub.add_parser("train", help="GPU: SFT of one sweep config (train.py recipe)")
    p_tr.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    p_tr.add_argument("--cfg", choices=BACKBONE_CONFIGS, required=True)
    p_tr.add_argument("--smoke", action="store_true", help="build from the smoke targets")
    p_tr.add_argument("--limit", type=int, help="smoke: train on N records")
    p_tr.add_argument("--longest", action="store_true", help="with --limit: the N longest")
    p_tr.add_argument("--run-name")
    p_tr.add_argument("--overwrite", action="store_true")
    p_tr.add_argument("--dry-run", action="store_true", help="CPU: build and validate only")
    p_s = sub.add_parser("sec", help="GPU: suite ml1b v2 security check")
    p_s.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    p_s.add_argument("--six", action="store_true", help="6 languages: best config + undef")
    p_s.add_argument("--arms", nargs="+", help="explicit arms (smoke)")
    p_s.add_argument("--limit-hosts", type=int, default=0, help="smoke: first N hosts")
    p_b = sub.add_parser("breakdown", help="CPU: IFEval breakdown of every complete backbone arm")
    p_b.add_argument("--boot", type=int, default=10_000)
    sub.add_parser("decide", help="CPU: decision rules -> runs/backbones/decision.json")
    args = ap.parse_args(argv)
    if args.cmd == "targets":
        cmd_targets(args.backbone, args.cap, args.limit)
    elif args.cmd == "census":
        census()
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "sec":
        cmd_sec(args.backbone, args.six, args.arms, args.limit_hosts)
    elif args.cmd == "breakdown":
        cmd_breakdown(args.boot)
    else:
        decide()


if __name__ == "__main__":
    main()
