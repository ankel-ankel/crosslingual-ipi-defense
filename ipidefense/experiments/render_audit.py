import argparse
import json

from ipidefense.attacks.templates import EVAL_INJS, PREFIX_TEMPLATES, TRAIN_TEMPLATES
from ipidefense.attacks.witnesses import WitnessInjection
from ipidefense.experiments.controls import CERT_INJS
from ipidefense.paths import RUNS
from ipidefense.rendering.literal import is_translatable, render_inj_literal, render_v2
from ipidefense.rendering.nllb import NLLBTranslator

LANGS = ("es", "vi", "fr", "de", "zh")
OUT = RUNS / "render_audit.json"


def inventory() -> dict[str, tuple[WitnessInjection, ...]]:
    """Template pools of the main experiments, with their original witnesses."""
    return {
        "train": tuple(WitnessInjection(t, "PWNED") for t in TRAIN_TEMPLATES),
        "prefix": tuple(WitnessInjection(t, "PWNED") for t in PREFIX_TEMPLATES),
        "eval": tuple(EVAL_INJS),
        "cert": CERT_INJS,
    }


def transcript_echo_counts(tails: dict[str, str]) -> dict[str, dict[str, int]]:
    """Count undefended control outputs that contain each language's old tail."""
    counts: dict[str, dict[str, int]] = {}
    for seed in (0, 1):
        path = RUNS / f"certify_transcripts_headline_seed{seed}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("arm") != "undefended" or rec.get("lang") not in tails:
                continue
            tail = tails[rec["lang"]]
            c = counts.setdefault(rec["lang"], {"n": 0, "with_tail": 0})
            c["n"] += 1
            c["with_tail"] += int(bool(tail) and tail in rec["output"])
    return counts


def main() -> None:
    argparse.ArgumentParser().parse_args()
    tr = NLLBTranslator()
    rows = []
    for pool, injs in inventory().items():
        for inj in injs:
            pre, suf = inj.wrapper_prefix(), inj.wrapper_suffix()
            for lang in LANGS:
                old = render_inj_literal(tr, inj, lang)
                new = render_v2(tr, inj, lang)
                rows.append({
                    "pool": pool, "template": inj.wrapper_template, "witness": inj.witness,
                    "lang": lang, "old": old, "new": new, "changed": old != new,
                    "prefix_translatable": is_translatable(pre),
                    "suffix_translatable": is_translatable(suf),
                    "suffix_raw": suf, "witness_once_new": new.count(inj.witness) == 1,
                })
    # old tail of the first control wrapper per language
    cert0 = [r for r in rows if r["pool"] == "cert" and r["witness"] == "XYZZY"]
    tails = {r["lang"]: r["old"].split("XYZZY", 1)[1].strip() for r in cert0}
    report = {
        "n_rows": len(rows),
        "n_changed": sum(r["changed"] for r in rows),
        "changed_by_pool": {p: sum(r["changed"] for r in rows if r["pool"] == p)
                            for p in inventory()},
        "witness_integrity_new_all": all(r["witness_once_new"] for r in rows),
        "cert_xyzzy_old_tails": tails,
        "transcript_echo_of_old_tail": transcript_echo_counts(tails),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False,
                     indent=1))


if __name__ == "__main__":
    main()
