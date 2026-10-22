import json
import statistics
from collections.abc import Callable

from ipidefense.paths import MODELS, RUNS
from ipidefense.runner import Run, Step, main, root_arg, shown

TRAIN = "ipidefense.training.defense"
UTILITY = "ipidefense.experiments.utility"
TRANSFER = "ipidefense.experiments.transfer"
STYLES = "ipidefense.experiments.styles"
BENIGN = "ipidefense.experiments.benign"
ADAPTIVE = "ipidefense.experiments.adaptive"
CONTROLS = "ipidefense.experiments.controls"
NONEMISSION = "ipidefense.experiments.nonemission"

# main settings
SEEDS = [0, 1, 2]
BASE = "qwen"
N_TRAIN, N_EVAL = 1500, 150
IFEVAL_N = 541
DC_N = 100
EXP = RUNS / "experiment"

# model and run roots used in step arguments
M = root_arg("IPI_MODELS", "models")
R = root_arg("IPI_RUNS", "runs")


# three seeds
def stat(xs: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(xs), 4),
        "std": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
        "values": [round(x, 4) for x in xs],
    }


def save(summary: dict) -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def seed_result(s: int) -> dict:
    sd = EXP / f"seed{s}"
    return {
        "seed": s,
        "train": json.loads((sd / "train.json").read_text(encoding="utf-8")),
        "utility": json.loads((sd / "utility.json").read_text(encoding="utf-8")),
    }


def transfer_result(s: int) -> dict:
    out_name = f"eval_transfer_seed{s}.json"
    adapter_s = str(MODELS / f"defense-{BASE}-seed{s}")
    data = json.loads((EXP / out_name).read_text(encoding="utf-8"))
    if data.get("seed") != s or data.get("lora_dir") != adapter_s:
        raise SystemExit(
            f"{out_name}: stale cache (seed={data.get('seed')}, "
            f"lora_dir={data.get('lora_dir')}; expected seed {s}) -- delete it and re-run"
        )
    return data


def train(run: Run) -> None:
    if not run.dry_run:
        EXP.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict] = []
    for s in SEEDS:
        sd = EXP / f"seed{s}"
        if not run.dry_run:
            sd.mkdir(exist_ok=True)
        adapter = MODELS / f"defense-{BASE}-seed{s}"
        if (sd / "train.json").exists() and (sd / "utility.json").exists():
            print(f"seed {s}: cached, skipping", flush=True)
        else:
            run.py(TRAIN, "--base", BASE, "--seed", str(s),
                   "--n-train", str(N_TRAIN), "--n-eval", str(N_EVAL))
            run.copy(RUNS / f"train_{BASE}.json", sd / "train.json")
            run.py(UTILITY, "--base", BASE, "--model", "lora",
                   "--lora-dir", str(adapter), "--n", str(IFEVAL_N))
            run.copy(RUNS / f"eval_utility_{BASE}_lora_{adapter.name}.json", sd / "utility.json")
        if run.dry_run:
            print(f"write {shown(EXP / 'summary.json')} (seeds done so far)", flush=True)
            continue
        per_seed.append(seed_result(s))
        save({"status": f"{len(per_seed)}/{len(SEEDS)} seeds done", "per_seed": per_seed})


def ifeval(run: Run) -> None:
    undef_path = EXP / "undefended_utility.json"
    if undef_path.exists():
        print(f"{undef_path.name}: cached, skipping", flush=True)
        return
    if not run.dry_run:
        EXP.mkdir(parents=True, exist_ok=True)
    run.py(UTILITY, "--base", BASE, "--model", "undefended", "--n", str(IFEVAL_N))
    run.copy(RUNS / f"eval_utility_{BASE}_undefended.json", undef_path)


def transfer(run: Run) -> None:
    if not run.dry_run:
        EXP.mkdir(parents=True, exist_ok=True)
    nt, ne = str(N_TRAIN), str(N_EVAL)
    for s in SEEDS:
        out_name = f"eval_transfer_seed{s}.json"
        adapter_s = str(MODELS / f"defense-{BASE}-seed{s}")
        if not (EXP / out_name).exists():
            run.py(TRANSFER, "--lora-dir", adapter_s,
                   "--seed", str(s), "--n-train", nt, "--n-eval", ne)
            run.copy(RUNS / out_name, EXP / out_name)
        else:
            print(f"{out_name}: cached, skipping", flush=True)
        if not run.dry_run:
            transfer_result(s)


def seed0(run: Run) -> None:
    if not run.dry_run:
        EXP.mkdir(parents=True, exist_ok=True)
    nt, ne = str(N_TRAIN), str(N_EVAL)
    ref_adapter = str(MODELS / f"defense-{BASE}-seed{SEEDS[0]}")
    steps = [
        ("eval_robustness.json",
         [STYLES, "--lora-dir", ref_adapter, "--n-train", nt, "--n", ne]),
        (f"eval_data_channel_{BASE}.json",
         [BENIGN, "--base", BASE, "--lora-dir", ref_adapter,
          "--skip", str(N_TRAIN + N_EVAL), "--n", str(DC_N)]),
    ]
    for out_name, cmd in steps:
        if (EXP / out_name).exists():
            print(f"{out_name}: cached, skipping", flush=True)
            continue
        run.py(*cmd)
        run.copy(RUNS / out_name, EXP / out_name, if_exists=True)


def write_summary(run: Run) -> None:
    if run.dry_run:
        print(f"write {shown(EXP / 'summary.json')} (means and standard deviations over seeds)",
              flush=True)
        return
    seeds = SEEDS
    per_seed = [seed_result(s) for s in seeds]
    undef = json.loads((EXP / "undefended_utility.json").read_text(encoding="utf-8"))
    transfer_runs = [transfer_result(s) for s in seeds]
    ref_seed = seeds[0]

    langs = per_seed[0]["train"]["langs"]
    security = {
        ll: {
            "undefended": stat([p["train"]["undefended"][ll]["mean"] for p in per_seed]),
            "defended": stat([p["train"]["defended"][ll]["mean"] for p in per_seed]),
            "defended_outcomes": {
                k: round(
                    statistics.mean(
                        [p["train"]["defended"][ll]["outcomes"][k] for p in per_seed]
                    ),
                    3,
                )
                for k in ("obeyed", "refused", "ignored")
            },
        }
        for ll in langs
    }
    utility = {
        "undefended_instruction_strict": undef["instruction_strict_acc"],
        "undefended_prompt_strict": undef["prompt_strict_acc"],
        "defended_instruction_strict": stat(
            [p["utility"]["instruction_strict_acc"] for p in per_seed]
        ),
        "defended_prompt_strict": stat([p["utility"]["prompt_strict_acc"] for p in per_seed]),
    }
    tlangs = transfer_runs[0]["eval_langs"]
    unseen = set(transfer_runs[0]["unseen_langs"])
    if any(r["eval_langs"] != tlangs or set(r["unseen_langs"]) != unseen for r in transfer_runs):
        raise SystemExit("transfer runs disagree on eval_langs/unseen_langs across seeds")
    transfer = {
        ll: {
            "unseen": ll in unseen,
            "undefended": stat([r["undefended"][ll]["mean"] for r in transfer_runs]),
            "defended": stat([r["defended"][ll]["mean"] for r in transfer_runs]),
        }
        for ll in tlangs
    }
    summary = {
        "base": BASE, "seeds": seeds, "n_train": N_TRAIN, "n_eval": N_EVAL,
        "security": security, "utility": utility, "transfer": transfer, "per_seed": per_seed,
        "broader_evals": {
            "seed": ref_seed,
            "note": "transfer is multi-seed (mean+/-std over all seeds); robustness/"
                    f"data_channel are single-seed on the seed {ref_seed} adapter.",
        },
    }
    save(summary)

    bar = "=" * 64
    ui = utility["defended_instruction_strict"]
    up = utility["defended_prompt_strict"]
    print(f"\n{bar}", flush=True)
    print(f"  experiment summary   base={BASE}   seeds={seeds}", flush=True)
    print(bar, flush=True)

    print("\nsecurity   held-out injection obey-rate, mean +/- std over seeds", flush=True)
    print(f"  {'lang':<6}{'undefended':>12}{'defended':>20}", flush=True)
    for ll in langs:
        u, d = security[ll]["undefended"]["mean"], security[ll]["defended"]
        print(f"  {ll:<6}{u:>12.3f}{d['mean']:>13.3f} +/- {d['std']:.3f}", flush=True)

    print("\n           defended response to injection (mean over seeds)", flush=True)
    print(f"  {'lang':<6}{'ignored':>10}{'refused':>10}{'obeyed':>10}", flush=True)
    for ll in langs:
        o = security[ll]["defended_outcomes"]
        print(
            f"  {ll:<6}{o['ignored']:>10.3f}{o['refused']:>10.3f}{o['obeyed']:>10.3f}",
            flush=True,
        )

    print("\ntransfer   zero-shot cross-lingual obey-rate, mean +/- std over seeds", flush=True)
    print(f"  {'lang':<6}{'undefended':>12}{'defended':>20}", flush=True)
    for ll in tlangs:
        u, d = transfer[ll]["undefended"]["mean"], transfer[ll]["defended"]
        tag = "  unseen" if transfer[ll]["unseen"] else ""
        print(f"  {ll:<6}{u:>12.3f}{d['mean']:>13.3f} +/- {d['std']:.3f}{tag}", flush=True)

    print("\nutility    IFEval strict accuracy, mean +/- std over seeds", flush=True)
    print(f"  {'metric':<12}{'undefended':>9}{'defended':>20}", flush=True)
    print(
        f"  {'instruction':<12}{undef['instruction_strict_acc']:>9.4f}"
        f"{ui['mean']:>13.4f} +/- {ui['std']:.4f}",
        flush=True,
    )
    print(
        f"  {'prompt':<12}{undef['prompt_strict_acc']:>9.4f}"
        f"{up['mean']:>13.4f} +/- {up['std']:.4f}",
        flush=True,
    )

    print(
        f"\nbroader    single seed (seed {ref_seed}): robustness, data_channel",
        flush=True,
    )
    print(f"\nresults saved under {EXP}", flush=True)
    print(bar, flush=True)


# SEA-LION, role format
SEALION = str(MODELS / "defense-sealion-seed0")

# (marker output in RUNS, command, (file written, stored name) or None)
SEALION_PARTS = [
    ("train_sealion.json",
     [TRAIN, "--base", "sealion", "--seed", "0", "--batch", "16"], None),
    (
        "eval_transfer_sealion_seed0.json",
        [TRANSFER, "--base", "sealion", "--lora-dir", SEALION, "--seed", "0", "--batch", "32"],
        # rename so it does not clash with the Qwen seed-0 output
        ("eval_transfer_seed0.json", "eval_transfer_sealion_seed0.json"),
    ),
    (
        "eval_utility_sealion_lora_defense-sealion-seed0.json",
        [UTILITY, "--base", "sealion", "--model", "lora", "--lora-dir", SEALION],
        None,
    ),
    (
        "eval_utility_sealion_undefended.json",
        [UTILITY, "--base", "sealion", "--model", "undefended"],
        None,
    ),
]
INLINE = [
    ("train_qwen-inline.json",
     [TRAIN, "--base", "qwen", "--seed", "0", "--inline", "--batch", "16"], None),
    (
        "eval_utility_qwen_lora_defense-qwen-seed0-inline.json",
        [UTILITY, "--base", "qwen", "--model", "lora",
         "--lora-dir", str(MODELS / "defense-qwen-seed0-inline")],
        None,
    ),
]


def parts(items: list) -> Callable[[Run], None]:
    def step(run: Run) -> None:
        for out_name, cmd, rename in items:
            if (RUNS / out_name).exists():
                print(f"{out_name}: cached, skipping", flush=True)
                continue
            if rename is None:
                run.py(*cmd)
                continue
            with run.keep(RUNS / rename[0], RUNS / rename[1]):
                run.py(*cmd)
    return step


# defense evaluations, flags follow the arguments stored in the outputs
def hardened(run: Run) -> None:
    run.py(TRAIN, "--base", "qwen", "--seed", "0", "--harden")
    run.py(TRAIN, "--base", "qwen", "--seed", "1", "--harden")
    run.py(UTILITY, "--base", "qwen", "--model", "lora",
           "--lora-dir", f"{M}/defense-qwen-seed0-hardened")


def refuse(run: Run) -> None:
    run.py(TRAIN, "--base", "qwen", "--seed", "0", "--refuse")
    with run.keep(f"{R}/eval_transfer_seed0.json", f"{R}/eval_transfer_refuse.json"):
        run.py(TRANSFER, "--lora-dir", f"{M}/defense-qwen-seed0-refuse", "--seed", "0")
    run.py(UTILITY, "--base", "qwen", "--model", "lora",
           "--lora-dir", f"{M}/defense-qwen-seed0-refuse")
    with run.keep(f"{R}/eval_data_channel_qwen.json", f"{R}/eval_data_channel_qwen_refuse.json"):
        run.py(BENIGN, "--lora-dir", f"{M}/defense-qwen-seed0-refuse", "--seed", "0")


def mistral(run: Run) -> None:
    # batch 16 inferred from the run time, as for SEA-LION
    run.py(TRAIN, "--base", "mistral", "--inline", "--seed", "0", "--batch", "16")
    run.py(UTILITY, "--base", "mistral", "--model", "lora",
           "--lora-dir", f"{M}/defense-mistral-seed0-inline")
    run.py(UTILITY, "--base", "mistral", "--model", "undefended")


def styles(run: Run) -> None:
    # seed 0 comes from step seed0
    run.copy(f"{R}/experiment/eval_robustness.json", f"{R}/eval_robustness_seed0.json")
    for k in (1, 2):
        with run.keep(f"{R}/eval_robustness.json", f"{R}/eval_robustness_seed{k}.json"):
            run.py(STYLES, "--lora-dir", f"{M}/defense-qwen-seed{k}", "--seed", k)


def benign(run: Run) -> None:
    run.copy(f"{R}/experiment/eval_data_channel_qwen.json",
             f"{R}/eval_data_channel_qwen_seed0.json")
    for k in (1, 2):
        with run.keep(f"{R}/eval_data_channel_qwen.json",
                      f"{R}/eval_data_channel_qwen_seed{k}.json"):
            run.py(BENIGN, "--lora-dir", f"{M}/defense-qwen-seed{k}", "--seed", k)


def adaptive(run: Run) -> None:
    # the seed-1 run used the seed-1 hosts, judging by its undefended rates
    with run.keep(f"{R}/eval_adaptive.json", f"{R}/eval_adaptive_base.json"):
        run.py(ADAPTIVE, "--lora-dir", f"{M}/defense-qwen-seed0")
    with run.keep(f"{R}/eval_adaptive.json", f"{R}/eval_adaptive_hardened.json"):
        run.py(ADAPTIVE, "--lora-dir", f"{M}/defense-qwen-seed0-hardened")
    with run.keep(f"{R}/eval_adaptive.json", f"{R}/eval_adaptive_hardened_seed1.json"):
        run.py(ADAPTIVE, "--lora-dir", f"{M}/defense-qwen-seed1-hardened", "--seed", "1")


def controls(run: Run) -> None:
    # headline: unseen witnesses, prefix: unseen prefix wording
    run.py(CONTROLS, "--mode", "headline", "--adapter", f"{M}/defense-qwen-seed0",
           "--seed", "0", "--tag", "seed0")
    run.py(CONTROLS, "--mode", "headline", "--adapter", f"{M}/defense-qwen-seed1",
           "--seed", "1", "--tag", "seed1")
    run.py(CONTROLS, "--mode", "prefix", "--adapter", f"{M}/defense-qwen-seed0-hardened",
           "--seed", "0", "--n", "100", "--tag", "hardened-seed0")
    run.py(CONTROLS, "--mode", "prefix", "--adapter", f"{M}/defense-qwen-seed1-hardened",
           "--seed", "1", "--n", "100", "--tag", "hardened-seed1")


def nonemission(run: Run) -> None:
    # format hijacking, then answer substitution
    run.py(NONEMISSION, "--mode", "format", "--adapter", f"{M}/defense-qwen-seed0",
           "--seed", "0", "--tag", "seed0")
    run.py(NONEMISSION, "--mode", "format", "--adapter", f"{M}/defense-qwen-seed1",
           "--seed", "1", "--tag", "seed1")
    for k in (0, 1):
        run.py(NONEMISSION, "--mode", "content", "--adapter", f"{M}/defense-qwen-seed{k}",
               "--n", "200", "--tag", f"seed{k}")
        run.py(NONEMISSION, "--mode", "content", "--subtle",
               "--adapter", f"{M}/defense-qwen-seed{k}", "--n", "200", "--tag", f"subtle-seed{k}")


def position(run: Run) -> None:
    run.py("ipidefense.experiments.position", "--lora-dir", f"{M}/defense-qwen-seed0",
           "--seed", "0")


def preservation(run: Run) -> None:
    run.py("ipidefense.experiments.preservation", "--lora-dir", f"{M}/defense-qwen-seed0",
           "--seed", "0")


def case_study(run: Run) -> None:
    run.py("ipidefense.experiments.case_study")


STEPS = [
    Step("train", "defense adapter (Qwen2.5-7B-Instruct) for seeds 0, 1, 2, with its IFEval",
         "Training configuration; clean examples per seed; utility table (IFEval)",
         train, marker=False),
    Step("ifeval", "IFEval of the undefended model", "Utility table (IFEval)",
         ifeval, marker=False),
    Step("transfer", "held-out injections in six languages (three unseen), seeds 0, 1, 2",
         "Main security table and security figure (defended)", transfer, marker=False),
    Step("seed0", "unseen attack styles and benign data channel, seed 0",
         "Unseen attack styles; utility table (benign channel)", seed0, marker=False),
    Step("summary", "means and standard deviations over seeds, runs/experiment/summary.json",
         "Main security table; utility table", write_summary, marker=False),
    Step("sealion", "SEA-LION adapter (seed 0): training, transfer, IFEval",
         "IFEval of the backbones, unbatched", parts(SEALION_PARTS), marker=False),
    Step("inline", "defense with the data inline in the user turn (seed 0) and its IFEval",
         "Role-format ablation", parts(INLINE), marker=False),
    Step("hardened", "hardened defense (prefix templates in training), seeds 0 and 1",
         "Adaptive attack table; IFEval of the hardened adapters (unbatched, seed 0)", hardened),
    Step("refuse", "refusal as the target (seed 0): training, transfer, IFEval, benign channel",
         "Supervision-target ablation", refuse),
    Step("mistral", "Mistral with the data inline (seed 0); IFEval defended and undefended",
         "IFEval of the backbones, unbatched", mistral),
    Step("styles", "unseen attack styles, seeds 1 and 2", "Unseen attack styles", styles),
    Step("benign", "benign data channel, seeds 1 and 2", "Utility table (benign data channel)",
         benign),
    Step("adaptive", "defense-aware attack (24 strategies, 100 hosts)",
         "Adaptive attack table and figure", adaptive),
    Step("controls", "unseen witnesses (initial adapters) and unseen prefix wording (hardened "
         "adapters)", "Control evaluations", controls),
    Step("nonemission", "format hijacking and answer substitution on SQuAD (no witness)",
         "Non-emission table", nonemission),
    Step("position", "injection at the start, middle and end of the data (seed 0)",
         "Injection position", position),
    Step("preservation", "response preservation under attack (seed 0)",
         "Response preservation under attack", preservation),
    Step("case-study", "three hand-written messages, undefended and defended", "Case study",
         case_study),
]


if __name__ == "__main__":
    main(__file__, STEPS)
