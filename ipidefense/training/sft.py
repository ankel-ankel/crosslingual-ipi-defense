"""Supervised fine-tuning of one comparison arm (a0, a1, a2) from a built JSONL.

--schedule constant (default) is training.lora.train_sft exactly; cosine adds linear warmup and
cosine decay. Example: sft.py --data data/baselines/seed0/a1.jsonl --seed 0
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

import torch

from ipidefense.model import QWEN_DIR
from ipidefense.training.lora import LORA_TARGETS, SFTExample, encode_example
from ipidefense.training.sets import (
    MAX_LEN,
    MODELS_DIR,
    RUNS_DIR,
    env_info,
    qwen_tokenizer,
    read_jsonl,
    sha256_file,
    stamp,
    token_lengths,
)
from ipidefense.utils import write_json

LOG = logging.getLogger("ipidefense.training.sft")
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SCHEDULES = ("constant", "cosine")
OPTIMIZER_TEXT = {
    "constant": "torch.optim.AdamW (defaults), constant lr",
    "cosine": "torch.optim.AdamW (defaults), transformers.get_cosine_schedule_with_warmup: "
              "linear warmup over ceil(warmup_ratio x total) optimizer steps, then cosine decay "
              "to 0 over the total steps; stepped once per optimizer step",
}
CARD_HEADING = "## Training metadata"


# ----------------------------------------------------------------------------- schedule
def schedule_plan(n_examples: int, accum: int, epochs: int, schedule: str = "constant",
                  warmup_ratio: float = 0.0) -> dict[str, Any]:
    """Optimizer steps of train_sft: ceil(n / accum) per epoch x epochs; cosine warms up over
    ceil(total x warmup_ratio) steps (the HF Trainer formula), constant has no warmup."""
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}: {SCHEDULES}")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(f"warmup_ratio {warmup_ratio} is not in [0, 1)")
    if schedule == "constant" and warmup_ratio:
        raise ValueError("warmup_ratio is used only with the cosine schedule")
    if n_examples < 1 or accum < 1 or epochs < 1:
        raise ValueError(f"need n_examples, accum, epochs >= 1 (got {n_examples}, {accum}, "
                         f"{epochs})")
    per_epoch = math.ceil(n_examples / accum)
    total = per_epoch * epochs
    warmup = math.ceil(total * warmup_ratio) if schedule == "cosine" else 0
    return {"schedule": schedule, "warmup_ratio": warmup_ratio, "steps_per_epoch": per_epoch,
            "total_steps": total, "warmup_steps": warmup}


def make_scheduler(opt: torch.optim.Optimizer, plan: dict[str, Any]) -> Any:
    """None for the constant schedule; cosine: HF get_cosine_schedule_with_warmup."""
    if plan["schedule"] == "constant":
        return None
    from transformers import get_cosine_schedule_with_warmup

    return get_cosine_schedule_with_warmup(opt, num_warmup_steps=plan["warmup_steps"],
                                           num_training_steps=plan["total_steps"])


def train_sft_sched(model: Any, tok: Any, examples: list[SFTExample], epochs: int = 1,
                    lr: float = 1e-4, accum: int = 8, max_len: int = 1024, log_every: int = 10,
                    seed: int = 0, schedule: str = "constant",
                    warmup_ratio: float = 0.0) -> dict[str, Any]:
    """train_sft with an optional scheduler stepped after each optimizer step; returns the plan,
    the optimizer step count and the lr used at every optimizer step."""
    plan = schedule_plan(len(examples), accum, epochs, schedule, warmup_ratio)
    torch.manual_seed(seed)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = make_scheduler(opt, plan)
    dev = next(model.parameters()).device
    step, lrs = 0, []
    for ep in range(epochs):
        order = torch.randperm(len(examples), generator=torch.Generator().manual_seed(seed + ep))
        opt.zero_grad()
        recent: list[float] = []
        for i, idx in enumerate(order.tolist()):
            ids, labels = encode_example(tok, examples[idx])
            ids = ids[:max_len].unsqueeze(0).to(dev)
            labels = labels[:max_len].unsqueeze(0).to(dev)
            loss = model(input_ids=ids, labels=labels).loss / accum
            loss.backward()
            recent.append(loss.item() * accum)
            last = (i + 1) == len(order)
            if (i + 1) % accum == 0 or last:
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                lrs.append(opt.param_groups[0]["lr"])
                opt.step()
                if sched is not None:
                    sched.step()
                opt.zero_grad()
                step += 1
                if step % log_every == 0:
                    avg = sum(recent) / len(recent)
                    print(f"epoch {ep} step {step} loss {avg:.4f}", flush=True)
                    recent = []
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    return {**plan, "optimizer_steps": step, "lr_peak": lr,
            "lr_after_last_step": opt.param_groups[0]["lr"], "lrs": lrs}


def write_card_metadata(out: Path, meta: dict[str, Any]) -> None:
    """Append the training metadata to the adapter's README.md (created if absent)."""
    with (out / "README.md").open("a", encoding="utf-8") as fh:
        fh.write(f"\n\n{CARD_HEADING}\n\n```json\n{json.dumps(meta, indent=1)}\n```\n")


def adapter_dir(run_name: str, overwrite: bool = False, models: Path = MODELS_DIR) -> Path:
    """MODELS/<run_name>, never over an existing adapter unless overwrite."""
    if not RUN_NAME.match(run_name):
        raise ValueError(f"invalid run name {run_name!r}")
    out = (models / run_name).resolve()
    if out.parent != models.resolve():
        raise ValueError(f"{out} is not directly under {models}")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    return out


def load_records(path: Path, seed: int, field: str) -> list[dict]:
    """Records of one arm and one seed that carry `field` ('target' or 'chosen')."""
    recs = read_jsonl(path)
    if not recs:
        raise ValueError(f"{path} has no records")
    bad = [r.get("key") for r in recs if r.get("seed") != seed or field not in r]
    if bad:
        raise ValueError(f"{path}: {len(bad)} records of another seed or schema, e.g. {bad[:3]}")
    arms = {r["arm"] for r in recs}
    if len(arms) != 1:
        raise ValueError(f"{path} mixes arms {sorted(arms)}")
    return recs


def check_manifest(path: Path, digest: str, allow_partial: bool = False) -> str:
    """'ok' if the build manifest lists this file with this SHA-256; raise on a mismatch or on
    a partial (smoke-only) build unless allow_partial."""
    manifest = path.parent / "manifest.json"
    if not manifest.exists():
        return "no manifest"
    content = json.loads(manifest.read_text(encoding="utf-8"))
    if content.get("partial") and not allow_partial:
        raise ValueError(f"{manifest} is a partial build: only --limit smoke runs may use it")
    entry = next((e for e in content.get("files", {}).values() if e.get("file") == path.name),
                 None)
    if entry is None:
        return "not in manifest"
    if entry["sha256"] != digest:
        raise ValueError(f"{path} does not match the SHA-256 in {manifest}")
    return "ok"


def select(n: int, limit: int | None, longest: bool, lengths: list[int]) -> list[int]:
    """Record indices to train on: all, the first `limit`, or the `limit` longest."""
    if limit is None:
        return list(range(n))
    if longest:
        return sorted(sorted(range(n), key=lambda k: -lengths[k])[:limit])
    return list(range(min(limit, n)))


def length_stats(lens: list[tuple[int, int]], max_len: int) -> dict[str, Any]:
    """lens = (prompt tokens, full tokens) per sequence; raise if one has no supervision."""
    unsupervised = [k for k, (p, _) in enumerate(lens) if p >= max_len]
    if unsupervised:
        raise ValueError(f"{len(unsupervised)} sequences have no supervised token within "
                         f"max_len={max_len} (indices {unsupervised[:5]}); rebuild the data")
    fulls = [f for _, f in lens]
    return {
        "n_sequences": len(lens),
        "prompt_tokens_max": max(p for p, _ in lens),
        "seq_tokens_mean": round(sum(fulls) / len(fulls), 1),
        "seq_tokens_max": max(fulls),
        "n_truncated": sum(f > max_len for f in fulls),
        "supervised_tokens": sum(min(f, max_len) - p for p, f in lens),
    }


def adapter_hashes(out: Path) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}


def lora_settings(model: Any) -> dict[str, Any]:
    cfg = model.peft_config["default"]
    return {"r": cfg.r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout,
            "targets": sorted(cfg.target_modules)}


class LossTee(io.TextIOBase):
    """Forwards writes and records the 'epoch E step S loss L' lines of the training loop."""

    PATTERN = re.compile(r"epoch (\d+) step (\d+) loss (\S+)")

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self._stream = stream
        self.losses: list[dict[str, float]] = []

    def write(self, text: str) -> int:
        self._stream.write(text)
        for m in self.PATTERN.finditer(text):
            self.losses.append({"epoch": int(m[1]), "step": int(m[2]), "loss": float(m[3])})
        return len(text)

    def flush(self) -> None:
        self._stream.flush()


def common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data", type=Path, required=True, help="built arm JSONL")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--run-name", help="adapter/report name (default <arm>-seed<k>)")
    ap.add_argument("--limit", type=int, help="smoke: N records (run name must start 'smoke')")
    ap.add_argument("--longest", action="store_true", help="with --limit: the N longest")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="CPU only: validate data and stop")


def resolve_run(args: argparse.Namespace, arm: str) -> tuple[str, Path]:
    run_name = args.run_name or f"{arm}-seed{args.seed}"
    if args.limit is not None and not run_name.startswith("smoke"):
        raise SystemExit("--limit is for smoke runs: pass --run-name smoke-...")
    return run_name, adapter_dir(run_name, args.overwrite)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="SFT of one comparison arm.")
    common_args(ap)
    ap.add_argument("--lr", type=float, default=1e-4, help="peak lr with --schedule cosine")
    ap.add_argument("--schedule", choices=SCHEDULES, default="constant",
                    help="constant (default, = training.lora.train_sft) or cosine with warmup")
    ap.add_argument("--warmup-ratio", type=float, default=0.0,
                    help="cosine only: linear warmup over ceil(ratio x total optimizer steps)")
    args = ap.parse_args(argv)
    if args.schedule == "constant" and args.warmup_ratio:
        ap.error("--warmup-ratio is used only with --schedule cosine")
    if not 0.0 <= args.warmup_ratio < 1.0:
        ap.error("--warmup-ratio must be in [0, 1)")

    started, t_start = stamp(), time.time()
    recs = load_records(args.data, args.seed, "target")
    arm = recs[0]["arm"]
    run_name, out = resolve_run(args, arm)
    digest = sha256_file(args.data)
    manifest = check_manifest(args.data, digest, args.limit is not None)
    tok = qwen_tokenizer()
    lens = [token_lengths(tok, r["messages"], r["target"]) for r in recs]
    keep = select(len(recs), args.limit, args.longest, [f for _, f in lens])
    recs = [recs[k] for k in keep]
    examples = [SFTExample(messages=r["messages"], target=r["target"]) for r in recs]
    stats = length_stats([lens[k] for k in keep], args.max_len)
    plan = schedule_plan(len(examples), args.accum, args.epochs, args.schedule,
                         args.warmup_ratio)
    config = {
        "arm": arm, "run_name": run_name, "seed": args.seed, "objective": "sft",
        "base": str(QWEN_DIR), "lr": args.lr, "epochs": args.epochs,
        "micro_batch": 1, "accum": args.accum, "max_len": args.max_len, "clip": 1.0,
        "optimizer": OPTIMIZER_TEXT[args.schedule], "precision": "bf16",
        "gradient_checkpointing": True,
        "loss": "assistant tokens only (training.lora.encode_example)",
        "lora_planned": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": sorted(LORA_TARGETS)},
        "schedule": args.schedule, "warmup_ratio": args.warmup_ratio,
        "steps_per_epoch": plan["steps_per_epoch"], "total_steps": plan["total_steps"],
        "warmup_steps": plan["warmup_steps"], "trainer": "training.sft.train_sft_sched",
    }
    data = {
        "path": str(args.data), "sha256": digest, "manifest": manifest,
        "n_records": len(recs), "kinds": dict(Counter(r["meta"]["kind"] for r in recs)),
        "smoke": args.limit is not None, "keys_first": [r["key"] for r in recs[:3]],
    }
    LOG.info("%s: %d examples %s, %s", run_name, len(recs), data["kinds"], stats)
    if args.dry_run:
        print(json.dumps({"config": config, "data": data, "lengths": stats}, indent=1))
        return

    from ipidefense.training.lora import build_lora_model

    t0 = time.time()
    torch.manual_seed(args.seed)  # seeded before building the LoRA model, as in the defense
    model = build_lora_model(str(QWEN_DIR))
    t1 = time.time()
    tee = LossTee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        sched = train_sft_sched(model, tok, examples, epochs=args.epochs, lr=args.lr,
                                accum=args.accum, max_len=args.max_len, seed=args.seed,
                                schedule=args.schedule, warmup_ratio=args.warmup_ratio)
    t2 = time.time()
    finite = all(math.isfinite(x["loss"]) for x in tee.losses)
    report: dict[str, Any] = {
        "config": config | {"lora": lora_settings(model)}, "data": data, "lengths": stats,
        "losses": tee.losses, "finite": finite, "schedule": sched, "started": started,
        "finished": stamp(), "env": env_info(),
    }
    if finite:
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out))
        write_card_metadata(out, {"config": report["config"], "data": data,
                                  "schedule": {k: v for k, v in sched.items() if k != "lrs"},
                                  "started": started})
        report["adapter"] = {"dir": str(out), "files": adapter_hashes(out)}
    t3 = time.time()
    report["wall_s"] = {"prep": round(t0 - t_start, 1), "load": round(t1 - t0, 1),
                        "train": round(t2 - t1, 1), "save": round(t3 - t2, 1),
                        "total": round(t3 - t_start, 1)}
    path = RUNS_DIR / f"train_{run_name}.json"
    write_json(path, report)
    LOG.info("wrote %s", path)
    if not finite:
        raise SystemExit("non-finite loss: adapter NOT saved")
    LOG.info("saved adapter -> %s", out)


if __name__ == "__main__":
    main()
