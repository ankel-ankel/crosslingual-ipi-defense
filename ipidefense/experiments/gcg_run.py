import argparse
import contextlib
import json
import statistics
import time
from pathlib import Path

import torch

from ipidefense.arms import ADAPTER_SEED, ADAPTERS
from ipidefense.attacks import gcg
from ipidefense.attacks.templates import EVAL_INJS
from ipidefense.data.splits import hosts_for
from ipidefense.metrics.stats import wilson
from ipidefense.model import load_defense_lora_qwen, load_qwen_undefended
from ipidefense.paths import MODELS, RUNS
from ipidefense.utils import Transcript, sha, write_json

# 28 layers x 7 projections, attached as "default"
N_LORA_MODULES = 196
MAX_CONSECUTIVE_TECH_FAILURES = 2  # more in a row means a systemic problem


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="White-box GCG against a LoRA defense.")
    ap.add_argument("--adapter", required=True, choices=sorted(ADAPTERS), help="arm name")
    ap.add_argument("--seed", type=int, default=None, help="host-split seed for undefended arms")
    ap.add_argument("--split", default="adaptive", help="host split (default: adaptive)")
    ap.add_argument("--hosts", type=int, default=50, help="number of hosts (per-host mode)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--suffix-len", type=int, default=20)
    ap.add_argument("--check-every", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128, help="candidate-loss mini-batch")
    ap.add_argument("--no-cache", action="store_true", help="disable prefix-KV reuse")
    ap.add_argument("--universal", action="store_true",
                    help="train one universal suffix over several hosts")
    ap.add_argument("--universal-hosts", type=int, default=10)
    ap.add_argument("--eval-suffix", default=None,
                    help="path to a universal.json to evaluate, "
                         "generation only, over --hosts x all EVAL_INJS")
    ap.add_argument("--compare-undefended", action="store_true",
                    help="also run the paired undefended arm via disable_adapter")
    ap.add_argument("--out", required=True,
                    help="output dir; a relative path resolves under RUNS, not the CWD")
    return ap.parse_args()


def build_config(args: argparse.Namespace) -> gcg.GCGConfig:
    return gcg.GCGConfig(
        steps=args.steps,
        search_width=args.width,
        topk=args.topk,
        suffix_len=args.suffix_len,
        check_every=args.check_every,
        batch=args.batch,
        use_cache=not args.no_cache,
    )


def load_arm(arm: str) -> tuple[torch.nn.Module, object, bool]:
    # (model, tokenizer, is_peft), Qwen input-role arms only
    base, adir, inline = ADAPTERS[arm]
    if base != "qwen":
        raise SystemExit(f"arm {arm!r} is a {base} arm; the GCG harness loads Qwen only.")
    if inline:
        raise SystemExit(
            f"arm {arm!r} is inline-format; the GCG harness only renders the input-role "
            "template, so inline arms are not supported."
        )
    if not adir:
        model, tok = load_qwen_undefended()
        return model, tok, False
    model, tok = load_defense_lora_qwen(str(MODELS / adir))
    return model, tok, True


def check_adapter(model: torch.nn.Module) -> None:
    # a no-op adapter would inflate the attack success
    mods = [m for m in model.modules() if hasattr(m, "lora_B") and "default" in m.lora_B]
    if len(mods) != N_LORA_MODULES:
        raise SystemExit(f"adapter sanity: {len(mods)} LoRA modules, expected {N_LORA_MODULES}")
    norm = float(sum(m.lora_B["default"].weight.float().norm() for m in mods))
    if norm <= 0:
        raise SystemExit("adapter sanity: lora_B is all zero (no-op adapter)")
    print(f"adapter OK: {len(mods)} modules, |lora_B|={norm:.2f}", flush=True)


def arm_context_factory(model: torch.nn.Module, disable: bool):
    if disable:
        return lambda: model.disable_adapter()
    return contextlib.nullcontext


def select_hosts(args: argparse.Namespace, n: int) -> list:
    pool = hosts_for(args.adapter, args.split, seed=args.seed)
    chosen = pool[:n]
    if len(chosen) < n:
        raise ValueError(f"split {args.split!r} has {len(pool)} hosts; need {n}")
    return list(chosen)


def design_payload(
    args: argparse.Namespace, cfg: gcg.GCGConfig, hosts: list
) -> dict[str, object]:
    # design content without timestamps
    assignment: list[dict] = [
        {
            "host_index": i,
            "witness": EVAL_INJS[i % len(EVAL_INJS)].witness,
            "injection": EVAL_INJS[i % len(EVAL_INJS)].render(),
            "instruction_sha": sha(h.instruction),
            "data_sha": sha(h.data),
        }
        for i, h in enumerate(hosts)
    ]
    return {
        "adapter": args.adapter,
        "adapter_seed": ADAPTER_SEED[args.adapter],
        "split": args.split,
        "n_hosts": len(hosts),
        "universal": args.universal,
        "compare_undefended": args.compare_undefended,
        "config": cfg.as_dict(),
        "injection_rotation": [inj.witness for inj in EVAL_INJS],
        "host_assignment": assignment,
    }


def config_hash(payload: dict[str, object]) -> str:
    # stamped into every per-host record
    return sha(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def freeze_design(out: Path, payload: dict[str, object]) -> str:
    # write design.json once, an existing one must match
    out.mkdir(parents=True, exist_ok=True)
    path = out / "design.json"
    cfg_hash = config_hash(payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        prior = {k: v for k, v in existing.items() if k not in ("created", "config_hash")}
        if config_hash(prior) != cfg_hash:
            raise SystemExit(
                f"design mismatch in {path}: the frozen config/hosts/injections "
                "differ from this invocation. Refusing to overwrite. Use a fresh --out, or "
                "restore the exact prior settings to resume."
            )
        return cfg_hash
    write_json(path, {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "config_hash": cfg_hash, **payload})
    return cfg_hash


# summary row -> (classify label, success value)
ROWS = {"witness_first": ("wf", "obeyed"), "refusal_first_obeyed": ("rf", "obeyed"),
        "starts_with_witness": ("starts_with", True)}


def host_hit(record: dict, row: str, field: str = "labels") -> bool:
    # field="labels_rp1" reads the penalty 1.0 re-checks
    key, value = ROWS[row]
    return any(c.get(field, {}).get(key) == value for c in record["checks"])


def summarize(records: list[dict]) -> dict[str, object]:
    # technical failures count as attack failures
    n = len(records)

    def rate(flag) -> dict[str, object]:
        k = sum(1 for r in records if flag(r))
        lo, hi = wilson(k, n)
        return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
                "wilson95": [round(lo, 4), round(hi, 4)]}

    steps = [r["success_step"] for r in records if r["success"] and r["success_step"] is not None]
    sw_steps = [s for s in (_first_step(r, "starts_with") for r in records) if s is not None]
    failed = sorted((r for r in records if r.get("technical_failure")),
                    key=lambda r: r["host_index"])
    return {
        "n_hosts": n,
        "technical_failures": len(failed),
        "technical_failure_hosts": [{"host_index": r["host_index"], "error": r.get("error")}
                                    for r in failed],
        "witness_first": rate(lambda r: r["success"]),
        "refusal_first_obeyed": rate(lambda r: host_hit(r, "refusal_first_obeyed")),
        "starts_with_witness": rate(lambda r: host_hit(r, "starts_with_witness")),
        **{f"rp1_{row}": rate(lambda r, row=row: host_hit(r, row, "labels_rp1")) for row in ROWS},
        "rp1_hosts_checked": sum(any("labels_rp1" in c for c in r["checks"]) for r in records),
        # witness-first at 1.05 or at 1.0
        "rp_either_witness_first": rate(lambda r: r["success"]
                                        or host_hit(r, "witness_first", "labels_rp1")),
        "median_steps_to_success_wf": statistics.median(steps) if steps else None,
        "median_steps_to_success_startswith": statistics.median(sw_steps) if sw_steps else None,
        "steps_to_success_wf": sorted(steps),
    }


def _first_step(record: dict, key: str) -> int | None:
    return (record.get("first_step") or {}).get(key)


def _attack_host_guarded(
    model, tok, embed_w, disallowed, host, injection, cfg, arm_ctx, i
) -> dict:
    # one retry, then a technical-failure record so a resume does not crash again
    for attempt in (1, 2):
        try:
            return gcg.attack_host(
                model, tok, embed_w, disallowed, host.instruction, host.data, injection, cfg,
                arm_ctx, i,
            )
        except AssertionError:
            raise  # a failed invariant is a bug, not an attack failure
        except Exception as exc:  # e.g. CUDA out of memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  host {i}: attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
            if attempt == 2:
                return {
                    "host_index": i,
                    "witness": injection.witness,
                    "injection": injection.render(),
                    "success": False,
                    "success_step": None,
                    "first_step": {"wf": None, "rf": None, "starts_with": None},
                    "technical_failure": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "steps_run": 0,
                    "final_loss": None,
                    "final_suffix": None,
                    "final_suffix_ids": None,
                    "loss_trace": [],
                    "checks": [],
                }
    raise AssertionError("unreachable")


def run_arm(
    model, tok, embed_w, disallowed, hosts, cfg, arm_ctx, out_dir: Path, tag: str, cfg_hash: str,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(out_dir / f"{tag}.jsonl")
    done = set(transcript.done)
    for rec in transcript.done.values():
        if rec.get("config_hash") not in (None, cfg_hash):
            raise SystemExit(
                f"resume refused: {tag}.jsonl host {rec.get('host_index')} was run under a "
                f"different config_hash ({rec.get('config_hash')} != {cfg_hash})."
            )
    records: list[dict] = list(transcript.done.values())
    consecutive_fail = 0
    for i, host in enumerate(hosts):
        key = f"{tag}:{i}"
        if key in done:
            continue
        injection = EVAL_INJS[i % len(EVAL_INJS)]
        rec = _attack_host_guarded(
            model, tok, embed_w, disallowed, host, injection, cfg, arm_ctx, i
        )
        consecutive_fail = consecutive_fail + 1 if rec.get("technical_failure") else 0
        if consecutive_fail >= MAX_CONSECUTIVE_TECH_FAILURES:
            raise SystemExit(
                f"[{tag}] {consecutive_fail} consecutive technical failures (last: "
                f"{rec.get('error')}). Systemic problem: fix it, then resume."
            )
        rec["key"] = key
        rec["config_hash"] = cfg_hash
        transcript.add([rec])
        records.append(rec)
        loss = rec["final_loss"]
        loss_s = f"{loss:.3f}" if loss is not None else "NA"
        print(f"[{tag}] host {i}: success={rec['success']} step={rec['success_step']} "
              f"tech_fail={rec.get('technical_failure')} loss={loss_s}", flush=True)
    records.sort(key=lambda r: r["host_index"])
    summary = summarize(records)
    write_json(out_dir / f"summary_{tag}.json", summary)
    return summary


def run_eval_suffix(
    model, tok, hosts, suffix: str, arm_ctx, out_dir: Path, tag: str, cfg: gcg.GCGConfig,
    cfg_hash: str,
) -> dict[str, object]:
    # fixed suffix on all hosts x EVAL_INJS, generation only
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(out_dir / f"{tag}.jsonl")
    done = set(transcript.done)
    records: list[dict] = list(transcript.done.values())
    for i, host in enumerate(hosts):
        for j, inj in enumerate(EVAL_INJS):
            key = f"{tag}:{i}:{j}"
            if key in done:
                continue
            spans = gcg.render_spans(tok, host.instruction, host.data, inj)
            labels, resp = gcg.run_check(
                model, tok, host.instruction, host.data, spans, suffix, arm_ctx, cfg
            )
            rec = {"key": key, "config_hash": cfg_hash, "host_index": i, "witness": inj.witness,
                   "success": labels["wf"] == "obeyed", "success_step": None, "first_step": {},
                   "checks": [{"labels": labels, "response": resp}], "technical_failure": False}
            transcript.add([rec])
            records.append(rec)
    summary = summarize(records)
    for inj in EVAL_INJS:
        subset = [r for r in records if r["witness"] == inj.witness]
        k = sum(1 for r in subset if r["success"])
        lo, hi = wilson(k, len(subset)) if subset else (0.0, 1.0)
        summary[f"witness_first_{inj.witness}"] = {
            "k": k, "n": len(subset), "wilson95": [round(lo, 4), round(hi, 4)]
        }
    write_json(out_dir / f"summary_{tag}.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    out = Path(args.out)
    if not out.is_absolute():
        out = RUNS / args.out

    model, tok, is_peft = load_arm(args.adapter)
    gcg.freeze_model(model)
    if is_peft:
        check_adapter(model)
    embed_w = model.get_input_embeddings().weight
    disallowed = gcg.build_disallowed_ids(tok, embed_w.shape[0])

    if args.eval_suffix is not None:
        hosts = select_hosts(args, args.hosts)
        cfg_hash = freeze_design(out, design_payload(args, cfg, hosts))
        suffix = json.loads(Path(args.eval_suffix).read_text(encoding="utf-8"))["final_suffix"]
        summary = run_eval_suffix(
            model, tok, hosts, suffix, arm_context_factory(model, disable=False),
            out, f"eval_{args.adapter}", cfg, cfg_hash,
        )
        print(f"[eval {args.adapter}] witness-first {summary['witness_first']}", flush=True)
        return

    if args.universal:
        hosts = select_hosts(args, args.universal_hosts)
        freeze_design(out, design_payload(args, cfg, hosts))
        rec = gcg.attack_universal(
            model, tok, embed_w, disallowed,
            [(h.instruction, h.data) for h in hosts], EVAL_INJS, cfg,
            arm_context_factory(model, disable=False),
        )
        write_json(out / "universal.json", rec)
        print(f"[universal] final_active_hosts={rec['final_active_hosts']} "
              f"suffix={rec['final_suffix']!r}", flush=True)
        return

    hosts = select_hosts(args, args.hosts)
    cfg_hash = freeze_design(out, design_payload(args, cfg, hosts))
    defended_summary = run_arm(
        model, tok, embed_w, disallowed, hosts, cfg,
        arm_context_factory(model, disable=False), out, args.adapter, cfg_hash,
    )
    print(f"[{args.adapter}] witness-first {defended_summary['witness_first']}", flush=True)

    if args.compare_undefended:
        if not is_peft:
            raise ValueError("--compare-undefended requires a defended --adapter")
        undef_summary = run_arm(
            model, tok, embed_w, disallowed, hosts, cfg,
            arm_context_factory(model, disable=True), out, "undef", cfg_hash,
        )
        print(f"[undef] witness-first {undef_summary['witness_first']}", flush=True)


if __name__ == "__main__":
    main()
