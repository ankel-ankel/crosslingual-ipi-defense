"""Planted-defect check of the whole-sentence translation review.

Usage: canary.py build | score A.json B.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from ipidefense.experiments.suite import render_path, suite_templates
from ipidefense.paths import DATA
from ipidefense.utils import write_json

LANGS = ("es", "vi", "fr", "de", "zh", "ko", "ar", "th", "bn", "my", "km")
DIR = DATA / "renderings" / "llm" / "canary"
TYPES = ("wrong_request", "no_request", "added_content", "untranslated")
N_REAL = 4
CTL = "ctl-invoice"


def build() -> dict:
    """4 planted defects (wrong_request, no_request, added_content, untranslated) and 4 real
    renderings per language, shuffled under opaque ids; key.json maps the ids to types."""
    tm = {t.tid: t for t in suite_templates()}
    fam = defaultdict(list)
    for t in tm.values():
        fam[t.family].append(t.tid)
    DIR.mkdir(parents=True, exist_ok=True)
    key: dict[str, dict] = {}
    for lang in LANGS:
        recs = json.loads(render_path(lang, "llm").read_text(encoding="utf-8"))["records"]
        rng = random.Random(f"canary-{lang}")
        ctl = recs[CTL]["rendered"]
        s1, p = rng.choice(sorted(fam["override"])), rng.choice(sorted(fam["prefix"]))
        s2 = rng.choice(sorted(fam["style"]))
        s3 = rng.choice(sorted(fam["defense"]))
        s4 = rng.choice(sorted(t for t in fam["override"] + fam["style"] if t not in (s1, s2)))
        w = {t: tm[t].witness for t in tm}
        canaries = [
            ("wrong_request", s1, recs[p]["rendered"].replace(w[p], w[s1]), p),
            ("no_request", s2, ctl.replace(w[CTL], w[s2]), CTL),
            ("added_content", s3, recs[s3]["rendered"] + " " + ctl, CTL),
            ("untranslated", s4, tm[s4].english(), None)]
        used = {s1, s2, s3, s4}
        pool = sorted(t for t in tm if t not in used and tm[t].family != "control")
        reals = [("real", t, recs[t]["rendered"], None) for t in rng.sample(pool, N_REAL)]
        items = canaries + reals
        rng.shuffle(items)
        work = []
        for i, (kind, src, rendered, donor) in enumerate(items, 1):
            qid = f"q{i:02d}"
            work.append({"lang": lang, "key": qid, "flagged": False,
                         "english": tm[src].english(), "rendered": rendered,
                         "witness": w[src], "qa_hard": [], "qa_review": []})
            key[f"{lang}|{qid}"] = {"type": kind, "source": src, "donor": donor}
        (DIR / f"review_worklist_{lang}.json").write_text(
            json.dumps(work, ensure_ascii=False, indent=1), encoding="utf-8")
    write_json(DIR / "key.json", key)
    print(f"{len(key)} items ({len(LANGS)} languages) -> {DIR}")
    return key


def score(a_path: str, b_path: str) -> dict:
    """Valid iff the merged verdict (pass only if both pass) fails >= 90% of all canaries and
    >= 75% of each type, and passes >= 90% of the real items."""
    key = json.loads((DIR / "key.json").read_text(encoding="utf-8"))
    votes = []
    for p in (a_path, b_path):
        vs = {f"{v['lang']}|{v['key']}": v["verdict"]
              for v in json.loads(Path(p).read_text(encoding="utf-8"))}
        if set(vs) != set(key):
            raise SystemExit(f"{p}: keys differ from the canary key")
        votes.append(vs)
    by: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "merged_fail": 0,
                                                          "a_fail": 0, "b_fail": 0})
    for k, info in key.items():
        c = by[info["type"]]
        c["n"] += 1
        c["a_fail"] += votes[0][k] == "fail"
        c["b_fail"] += votes[1][k] == "fail"
        c["merged_fail"] += votes[0][k] == "fail" or votes[1][k] == "fail"
    canary_n = sum(by[t]["n"] for t in TYPES)
    caught = sum(by[t]["merged_fail"] for t in TYPES)
    real = by["real"]
    res = {"by_type": dict(by),
           "canary_detection": round(caught / canary_n, 4),
           "per_type_detection": {t: round(by[t]["merged_fail"] / by[t]["n"], 4) for t in TYPES},
           "real_pass_rate": round(1 - real["merged_fail"] / real["n"], 4),
           "missed": sorted(k for k, i in key.items() if i["type"] != "real"
                            and votes[0][k] == "pass" and votes[1][k] == "pass"),
           "real_failed": sorted(k for k, i in key.items() if i["type"] == "real"
                                 and "fail" in (votes[0][k], votes[1][k]))}
    res["valid"] = bool(res["canary_detection"] >= 0.90
                        and min(res["per_type_detection"].values()) >= 0.75
                        and res["real_pass_rate"] >= 0.90)
    write_json(DIR / "canary_report.json", res)
    print(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sc = sub.add_parser("score")
    sc.add_argument("review_a")
    sc.add_argument("review_b")
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    else:
        score(args.review_a, args.review_b)


if __name__ == "__main__":
    main()
