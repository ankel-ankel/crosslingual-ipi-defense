import argparse
import json
from collections import defaultdict

from ipidefense.experiments.taskacc import em_f1
from ipidefense.paths import DATA, RUNS
from ipidefense.utils import write_json

FILES = ("content_seed0", "content_seed1", "content_subtle-seed0", "content_subtle-seed1")
SQUAD = DATA / "squad_dev_v1.1.json"
OUT = RUNS / "squad_rescore.json"


def _key(q: str) -> str:
    return " ".join(q.split())


def gold_index() -> dict[str, list[list[str]]]:
    """Question text -> gold-answer lists, one per occurrence in SQuAD dev."""
    idx: dict[str, list[list[str]]] = defaultdict(list)
    for art in json.loads(SQUAD.read_text(encoding="utf-8"))["data"]:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                idx[_key(qa["question"])].append([a["text"] for a in qa["answers"]])
    return idx


def rescore_file(name: str, idx: dict[str, list[list[str]]]) -> dict:
    # transcripts keep only gold[:2], so re-join the full gold lists by question
    rows = [json.loads(x) for x in
            (RUNS / f"nonemission_transcripts_{name}.jsonl").read_text(
                encoding="utf-8").splitlines() if x.strip()]
    by_arm: dict[str, list[dict]] = defaultdict(list)
    issues = {"ambiguous_question": 0, "missing_question": 0, "gold_prefix_mismatch": 0}
    for r in rows:
        cands = idx.get(_key(r["q"]), [])
        if not cands and len(r["q"]) >= 100:  # transcripts cut questions at 100 characters
            pref = [k for k in idx if k.startswith(_key(r["q"]))]
            cands = [g for k in pref for g in idx[k]]
            issues["prefix_joined"] = issues.get("prefix_joined", 0) + 1
        if not cands:
            issues["missing_question"] += 1
            golds = list(r["gold"])
        else:
            match = [g for g in cands if g[:len(r["gold"])] == r["gold"]]
            if len(cands) > 1:
                issues["ambiguous_question"] += 1
            if not match:
                issues["gold_prefix_mismatch"] += 1
            golds = (match or cands)[0]
        em, f1 = em_f1(r["output"], golds)
        by_arm[r["arm"]].append({"em": em, "f1": f1,
                                 "published_correct": bool(r["correct"]),
                                 "corrupted": bool(r["corrupted"]),
                                 "truncated_300": len(r["output"]) >= 300})
    summary = {}
    for arm, rs in by_arm.items():
        n = len(rs)
        summary[arm] = {"n": n, "EM": round(sum(x["em"] for x in rs) / n, 4),
                        "F1": round(sum(x["f1"] for x in rs) / n, 4),
                        "published_inj_acc": round(sum(x["published_correct"] for x in rs) / n, 4),
                        "corruption_rate": round(sum(x["corrupted"] for x in rs) / n, 4),
                        "outputs_truncated_at_300": sum(x["truncated_300"] for x in rs)}
    return {"n_rows": len(rows), "join_issues": issues, "arms": summary}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    idx = gold_index()
    res = {"source": "nonemission_transcripts_<run>.jsonl (read-only)",
           "metric": "official SQuAD v1.1 normalize_answer / EM / token F1, max over golds",
           "note": "injected condition only (clean outputs were not stored); outputs stored "
                   "up to 300 characters; one article (Super_Bowl_50)",
           "runs": {name: rescore_file(name, idx) for name in FILES}}
    write_json(OUT, res)
    print(json.dumps({k: v["arms"] for k, v in res["runs"].items()}, indent=1))
    print(json.dumps({k: v["join_issues"] for k, v in res["runs"].items()}))


if __name__ == "__main__":
    main()
