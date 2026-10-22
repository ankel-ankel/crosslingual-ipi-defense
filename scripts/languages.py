"""Checks of the translations, original vs corrected renderer, language gate, expanded language
suite, human-written attacks (also machine-translated) and non-English hosts.

Run after main_results.py: it reads its adapters and control transcripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # runs without installing

from ipidefense.runner import Run, Step, main, root_arg  # noqa: E402

SUITE = "ipidefense.experiments.suite"
REVIEW = "ipidefense.experiments.review"
CANARY = "ipidefense.experiments.canary"
HUMAN_MT = "ipidefense.experiments.human_mt"

DATA = root_arg("IPI_DATA", "data")
RUNS = root_arg("IPI_RUNS", "runs")


def render_audit(run: Run) -> None:
    run.py("ipidefense.experiments.render_audit")


def reviews(run: Run) -> None:
    # to rebuild the translations: suite.py render, translations.py export / import,
    # review.py list / apply, canary.py build
    run.py(REVIEW, "check", "--renderer", "llm",
           "--langs", "ko", "ar", "th", "bn", "my", "km", "es", "vi", "fr", "de", "zh")
    run.py(REVIEW, "report", "--worklist", f"{DATA}/renderings/llm/review_worklist_all.json",
           f"{DATA}/renderings/llm/review_llm_A.json", f"{DATA}/renderings/llm/review_llm_B.json",
           "--out", f"{RUNS}/review_report_llm.json", "--force")
    run.py(CANARY, "score", f"{DATA}/renderings/llm/canary/review_canary_A.json",
           f"{DATA}/renderings/llm/canary/review_canary_B.json")


def renderer(run: Run) -> None:
    run.py(SUITE, "run", "--exp", "ml1b", "--renderer", "old", "--out-dir", f"{RUNS}/compare/ml1b")
    run.py(SUITE, "run", "--exp", "ml1b", "--renderer", "v2",
           "--langs", "en", "es", "vi", "fr", "de", "zh", "--out-dir", f"{RUNS}/compare/ml1b")
    run.py(SUITE, "summarize", "--exp", "ml1b", "--renderer", "v2",
           "--out-dir", f"{RUNS}/compare/ml1b")


def gate(run: Run) -> None:
    run.py(SUITE, "run", "--exp", "ml2", "--renderer", "llm")
    run.py(SUITE, "summarize", "--exp", "ml2", "--renderer", "llm")


def suite(run: Run) -> None:
    plan = run.output("ipidefense.experiments.language_plan", "--renderer", "llm",
                      dry="MAIN <main-languages>\nEXTRA <extra-languages>")
    if not run.dry_run:
        print(plan.rstrip("\n"), flush=True)
    lines = plan.splitlines()
    main_langs = [w for x in lines if x.startswith("MAIN ") for w in x[len("MAIN "):].split()]
    extra = [w for x in lines if x.startswith("EXTRA ") for w in x[len("EXTRA "):].split()]
    run.py(SUITE, "run", "--exp", "ml3", "--renderer", "llm", "--batch", "75",
           "--langs", *main_langs)
    if extra:
        if run.dry_run:
            print("  (the next two commands run only if the plan lists extra languages)")
        run.py(SUITE, "run", "--exp", "ml3", "--renderer", "llm", "--batch", "75",
               "--langs", *extra, "--arms", "undef", "def-s0", "hard-s0", "--seeds", "0")
        run.py(SUITE, "summarize", "--exp", "ml3", "--renderer", "llm",
               "--langs", *main_langs, *extra)


def human(run: Run) -> None:
    # needs the Tensor Trust texts (prepare_data.py); run also writes the summary
    run.py(SUITE, "run", "--exp", "human", "--langs", "en", "ru", "zh", "es", "--batch", "75")


def hosts(run: Run) -> None:
    # the hosts are shipped; translated_hosts.py prepare draws them again
    run.py("ipidefense.experiments.translated_hosts", "run")


def human_mt(run: Run) -> bool:
    # the translations of the Tensor Trust-derived templates are not shipped, so the check stops
    # until they are supplied (human_mt.py export, import, review-list, apply)
    if not run.ok(HUMAN_MT, "check"):
        return False
    run.py(SUITE, "run", "--exp", "human_mt", "--langs", "es", "vi", "fr", "de", "zh", "ko",
           "--batch", "75")
    run.py(SUITE, "run", "--exp", "human_mt", "--langs", "bn", "--batch", "48")
    run.py(SUITE, "summarize", "--exp", "human_mt", "--langs", "es", "vi", "fr", "de", "zh", "ko",
           "bn")
    run.py("ipidefense.experiments.human_mt_table")
    return True


STEPS = [
    Step("render-audit", "original vs corrected NLLB renderer: the templates it affects",
         "Original vs corrected renderer", render_audit),
    Step("reviews", "checks of the shipped translations: reviews, agreement, planted defects",
         "Translations and their checks", reviews),
    Step("renderer", "suite with the original and the corrected renderer (ml1b)",
         "Original vs corrected renderer; main security table and figure (undefended)",
         renderer),
    Step("gate", "language gate, undefended model (ml2)", "Language gate", gate),
    Step("suite", "full suite on the languages the gate selects (ml3)",
         "Expanded suite, Arabic / Thai", suite),
    Step("human", "human-written attacks in en, ru, zh, es", "Human-written attacks", human),
    Step("hosts", "non-English hosts (Bactrian-X es, vi, de, zh)", "Non-English hosts", hosts),
    Step("human-mt", "machine-translated human-written attacks (needs translations that are not "
         "shipped)", "Human-written attacks, machine-translated", human_mt),
]


if __name__ == "__main__":
    main(__file__, __doc__, STEPS)
