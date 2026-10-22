"""Whole-sentence translations of the suite templates (renderer llm): export and import.

Usage: translations.py export [--langs ...] | import FILE... [--allow-partial]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from ipidefense.experiments.suite import render_path, suite_templates
from ipidefense.paths import DATA
from ipidefense.rendering.literal import MARKUP
from ipidefense.utils import sha, write_json

LANGS = ("es", "vi", "fr", "de", "zh", "ko", "ar", "th", "bn", "my", "km")
REN = DATA / "renderings"
WORKLIST = REN / "llm_worklist.json"
REPORT = REN / "llm_import_report.json"
METHOD = "llm:claude-agent (whole-sentence translation)"


def _templates() -> dict[str, object]:
    return {t.tid: t for t in suite_templates()}


def export(langs: Sequence[str]) -> list[dict]:
    items = [{"lang": lang, "key": t.tid, "family": t.family, "english": t.english(),
              "witness": t.witness, "n_witness": t.english().count(t.witness),
              "markup": MARKUP.findall(t.english()), "n_newlines": t.english().count("\n")}
             for lang in langs for t in suite_templates()]
    WORKLIST.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} items ({len(langs)} languages) -> {WORKLIST}")
    return items


def integrity(english: str, witness: str, rendered: str) -> list[str]:
    """Problems that make a rendering unusable before any review (empty = OK)."""
    bad = []
    if not rendered.strip() or rendered.strip() == english.strip():
        bad.append("empty_or_untranslated")
    if rendered.count(witness) != english.count(witness):
        bad.append("witness_count")
    if Counter(MARKUP.findall(rendered)) != Counter(MARKUP.findall(english)):
        bad.append("markup")
    if rendered.count("\n") != english.count("\n"):
        bad.append("line_breaks")
    return bad


def import_files(paths: Sequence[str], allow_partial: bool = False) -> dict:
    """Input files: {"renderings": [{"lang", "key", "rendered"}, ...]}."""
    tmpl = _templates()
    got: dict[str, dict[str, str]] = {}
    for p in paths:
        for r in json.loads(Path(p).read_text(encoding="utf-8"))["renderings"]:
            got.setdefault(r["lang"], {})[r["key"]] = r["rendered"]
    report: dict[str, dict] = {}
    for lang, recs in sorted(got.items()):
        if lang not in LANGS:
            raise SystemExit(f"unknown language {lang!r}")
        missing = sorted(set(tmpl) - set(recs))
        unknown = sorted(set(recs) - set(tmpl))
        if unknown or (missing and not allow_partial):
            raise SystemExit(f"{lang}: missing {missing} unknown {unknown}")
        path = render_path(lang, "llm")
        cache = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
                 else {"lang": lang, "renderer": "llm", "method": METHOD, "records": {}})
        fails = {}
        for tid, text in sorted(recs.items()):
            t = tmpl[tid]
            bad = integrity(t.english(), t.witness, text)
            if bad:
                fails[tid] = bad
            old = cache["records"].get(tid)
            if old and old["rendered"] != text and old.get("review"):
                raise SystemExit(f"{lang}/{tid}: refusing to replace a reviewed rendering")
            # integrity failures are stored with their problems (never reviewed, never QA-ok),
            # so a run can proceed without them; a later import may replace them
            cache["records"][tid] = {
                "tid": tid, "family": t.family, "template": t.template, "witness": t.witness,
                "english": t.english(), "rendered": text, "sha1": sha(text), "method": METHOD,
                "backtranslation": None, "review": old.get("review") if old else None,
                "chrf_bt": None, "integrity": bad}
        write_json(path, cache)
        report[lang] = {"stored": len(recs), "passed_integrity": len(recs) - len(fails),
                        "integrity_fail": fails,
                        "missing": missing, "cache": str(path)}
    write_json(REPORT, report)
    print(json.dumps({k: {"stored": v["stored"], "fail": v["integrity_fail"]}
                      for k, v in report.items()}, ensure_ascii=False))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("export", "import"))
    ap.add_argument("files", nargs="*", help="import: translation files")
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--allow-partial", action="store_true", help="import: not every template")
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.langs)
    else:
        rep = (import_files(args.files, args.allow_partial) if args.files
               else ap.error("import needs translation files"))
        sys.exit(1 if any(v["integrity_fail"] for v in rep.values()) else 0)


if __name__ == "__main__":
    main()
