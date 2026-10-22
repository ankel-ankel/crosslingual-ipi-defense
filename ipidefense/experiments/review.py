import argparse
import json
import os
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from ipidefense.experiments.suite import qa_language
from ipidefense.paths import DATA

NEW = ("ko", "ar", "th", "bn", "my", "km")  # languages added to the suite
LANGS = NEW
REN = DATA / "renderings"
WORK = REN / "review_worklist.json"
SUPP = REN / "review_worklist_supp.json"
VERDICTS = ("pass", "fail")


def _path(lang: str, renderer: str = "v2") -> Path:
    """Same naming as suite.render_path, relative to REN."""
    return REN / (f"{lang}.json" if renderer == "v2" else f"{lang}_{renderer}.json")


def _cache(lang: str, renderer: str = "v2") -> tuple[dict, dict]:
    """Cached renderings of one language with QA recomputed."""
    path = _path(lang, renderer)
    data = json.loads(path.read_text(encoding="utf-8"))
    recs = data.get("records", data)
    qa_language(recs, lang)
    return data, recs


def atomic_write(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def refuse_overwrite(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; pass --force to overwrite (review evidence)")


def _attack_recs(recs: dict) -> dict:
    """Attack and copy-control records that need review."""
    return {k: r for k, r in recs.items() if isinstance(r, dict)
            and r.get("family") != "trainpool" and not r.get("integrity")}


def _item(lang: str, k: str, r: dict, flagged: bool) -> dict:
    return {"lang": lang, "key": k, "flagged": flagged,
            "english": r.get("english") or r.get("source") or r.get("template"),
            "rendered": r.get("rendered"), "backtranslation": r.get("backtranslation"),
            "witness": r.get("witness"), "qa_hard": r.get("qa", {}).get("hard"),
            "qa_review": r.get("qa", {}).get("review")}


def _flagged(recs: dict) -> set[str]:
    return {k for k, r in _attack_recs(recs).items() if r.get("qa", {}).get("hard")}


def build_list(langs: Sequence[str] | None = None, out: Path | None = None,
               renderer: str = "v2", review_all: bool = False, force: bool = False) -> None:
    """Flagged items plus a fixed-seed 25% of the rest, or all with review_all."""
    out = out or WORK
    refuse_overwrite(out, force)
    items = []
    for lang in langs or LANGS:
        _, recs = _cache(lang, renderer)
        flagged = _flagged(recs)
        clean = sorted(k for k in _attack_recs(recs) if k not in flagged)
        rng = random.Random(f"review-{lang}")
        sample = (clean if review_all else
                  sorted(rng.sample(clean, max(1, round(0.25 * len(clean))))) if clean else [])
        items += [_item(lang, k, recs[k], k in flagged) for k in sorted(flagged) + sample]
    atomic_write(out, items)
    print(f"{len(items)} items -> {out}")


def build_supplement(langs: Sequence[str] | None = None, first_path: Path | None = None,
                     out: Path | None = None, force: bool = False) -> None:
    """Newly flagged unreviewed items plus a top-up of the random sample."""
    out = out or SUPP
    refuse_overwrite(out, force)
    first = json.loads((first_path or WORK).read_text(encoding="utf-8"))
    reviewed = {(w["lang"], w["key"]) for w in first}
    random_first = {(w["lang"], w["key"]) for w in first if not w["flagged"]}
    items = []
    for lang in langs or LANGS:
        _, recs = _cache(lang)
        flagged = _flagged(recs)
        unflagged = sorted(k for k in _attack_recs(recs) if k not in flagged)
        need = sorted(k for k in flagged if (lang, k) not in reviewed)
        have = [k for k in unflagged if (lang, k) in random_first]
        target = max(1, round(0.25 * len(unflagged))) if unflagged else 0
        pool = [k for k in unflagged if (lang, k) not in reviewed]
        rng = random.Random(f"review-supp-{lang}")
        extra = sorted(rng.sample(pool, max(0, min(len(pool), target - len(have)))))
        items += [_item(lang, k, recs[k], k in flagged) for k in need + extra]
        print(f"{lang}: flagged now {len(flagged)} (unreviewed {len(need)}); unflagged "
              f"{len(unflagged)}, random sample {len(have)} + {len(extra)} (target {target})")
    atomic_write(out, items)
    print(f"{len(items)} items -> {out}")


def check(langs: Sequence[str] | None = None, renderer: str = "v2") -> int:
    """Count renderings that still need a verdict."""
    missing = {}
    for lang in langs or LANGS:
        recs = _cache(lang, renderer)[1]
        need = set(_attack_recs(recs)) if renderer == "llm" else _flagged(recs)
        bad = sorted(k for k in need if recs[k].get("review") not in VERDICTS)
        if bad:
            missing[lang] = bad
    what = "renderings" if renderer == "llm" else "hard-flagged renderings"
    print(f"unreviewed {what} ({renderer}): {missing or 'none'}")
    return 1 if missing else 0


def load_votes(paths: Sequence[str], worklist: Path) -> tuple[list[dict], list[dict]]:
    """Load and validate the two reviewer files."""
    if len(paths) != 2:
        raise SystemExit(f"need exactly two reviewer files, got {len(paths)}")
    if Path(paths[0]).resolve() == Path(paths[1]).resolve():
        raise SystemExit("the two reviewer files must be different files")
    work = json.loads(Path(worklist).read_text(encoding="utf-8"))
    want = {(w["lang"], w["key"]) for w in work}
    votes = []
    for p in paths:
        vs = json.loads(Path(p).read_text(encoding="utf-8"))
        keys = [(v["lang"], v["key"]) for v in vs]
        if len(keys) != len(set(keys)):
            raise SystemExit(f"{p}: duplicate keys")
        if set(keys) != want:
            raise SystemExit(f"{p}: keys differ from {worklist}: missing "
                             f"{sorted(want - set(keys))[:5]} extra {sorted(set(keys) - want)[:5]}")
        bad = [v for v in vs if v.get("verdict") not in VERDICTS]
        if bad:
            raise SystemExit(f"{p}: invalid verdicts {[(v['lang'], v['key']) for v in bad][:5]}")
        votes.append({(v["lang"], v["key"]): v["verdict"] for v in vs})
    return work, votes


def apply(paths: Sequence[str], worklist: Path, renderer: str = "v2") -> dict:
    work, votes = load_votes(paths, worklist)
    merged = {(w["lang"], w["key"]): "pass" if all(d[(w["lang"], w["key"])] == "pass"
                                                   for d in votes) else "fail" for w in work}
    shown = {(w["lang"], w["key"]): w for w in work}
    caches = {}
    for lang in sorted({lg for lg, _ in merged}):  # validate everything before any write
        data, recs = _cache(lang, renderer)
        for (lg, key), verdict in merged.items():
            if lg != lang:
                continue
            if key not in recs:
                raise SystemExit(f"{lang}: verdict for unknown rendering {key}")
            w = shown[(lg, key)]
            if "rendered" in w and (w["rendered"], w.get("witness")) != (
                    recs[key]["rendered"], recs[key].get("witness")):
                raise SystemExit(f"{lang}/{key}: the cached text is not the text the reviewers "
                                 f"saw (renderer {renderer!r}); not applying")
            old = recs[key].get("review")
            if old not in (None, verdict):
                raise SystemExit(f"{lang}/{key}: already reviewed {old!r}; not changing it")
        caches[lang] = (data, recs)
    for lang, (data, recs) in caches.items():
        for (lg, key), verdict in merged.items():
            if lg == lang:
                recs[key]["review"] = verdict
        atomic_write(_path(lang, renderer), data)
    agree = sum(len({d[lk] for d in votes}) == 1 for lk in merged)
    print(f"applied {len(merged)} verdicts; agreement {agree}/{len(merged)}")
    return {"applied": len(merged), "agreement": agree}


def kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def report(paths: Sequence[str], worklist: Path) -> dict:
    """Per-language review statistics (no cache access)."""
    work, (va, vb) = load_votes(paths, worklist)
    per: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
    for w in work:
        lk = (w["lang"], w["key"])
        per[w["lang"]].append((va[lk] == "pass", vb[lk] == "pass", bool(w["flagged"])))
    out = {}
    for lang, rows in list(per.items()) + [("_all", [r for v in per.values() for r in v])]:
        a, b, f = zip(*rows, strict=True)
        both = [x and y for x, y in zip(a, b, strict=True)]
        unfl = [p for p, fl in zip(both, f, strict=True) if not fl]
        out[lang] = {"n": len(rows), "n_flagged": sum(f), "a_pass": sum(a), "b_pass": sum(b),
                     "both_pass": sum(both),
                     "both_pass_flagged": sum(p for p, fl in zip(both, f, strict=True) if fl),
                     "n_unflagged": len(unfl), "both_pass_unflagged": sum(unfl),
                     "unflagged_fail_rate": round(1 - sum(unfl) / len(unfl), 4) if unfl else None,
                     "agreement": sum(x == y for x, y in zip(a, b, strict=True)),
                     "kappa": kappa(a, b)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "supplement", "check", "apply", "report"))
    ap.add_argument("files", nargs="*", help="apply/report: the two reviewer verdict files")
    ap.add_argument("--langs", nargs="+")
    ap.add_argument("--first", type=Path, help="supplement: the first worklist")
    ap.add_argument("--worklist", type=Path, help="apply/report: the worklist reviewed")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    ap.add_argument("--renderer", choices=("v2", "llm"), default="v2")
    ap.add_argument("--all", action="store_true", help="list: every rendering (llm: 100%%)")
    args = ap.parse_args()
    if args.cmd == "list":
        build_list(args.langs, args.out, args.renderer, args.all, args.force)
    elif args.cmd == "supplement":
        build_supplement(args.langs, args.first, args.out, args.force)
    elif args.cmd == "check":
        sys.exit(check(args.langs, args.renderer))
    elif args.worklist is None:
        ap.error(f"{args.cmd} needs --worklist")
    elif args.cmd == "apply":
        apply(args.files, args.worklist, args.renderer)
    else:
        if args.out is None:
            ap.error("report needs --out")
        refuse_overwrite(args.out, args.force)
        res = report(args.files, args.worklist)
        atomic_write(args.out, res)
        print(json.dumps(res["_all"]))


if __name__ == "__main__":
    main()
