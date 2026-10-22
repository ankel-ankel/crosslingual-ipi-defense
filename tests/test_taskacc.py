"""Task accuracy under attack: SQuAD metrics, units and scores, prepared items, the summary and
the shared-token sensitivity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from ipidefense.attacks.templates import EVAL_INJS
from ipidefense.evaluation.runtime import parse_arm
from ipidefense.experiments import taskacc as ta


def _write(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")


def test_squad_metrics_and_boolq_parser() -> None:
    assert ta.normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert ta.em_f1("the eiffel tower", ["Eiffel Tower", "Paris"]) == (1.0, 1.0)
    em, f1 = ta.em_f1("Eiffel Tower in Paris", ["Eiffel Tower"])
    assert em == 0.0 and f1 == pytest.approx(2 / 3)
    assert ta.parse_yes_no("Yes, it is.") == "yes" and ta.parse_yes_no("Nope") is None
    assert ta.parse_yes_no("The answer is no.") == "no"


def test_taskacc_units_and_scores() -> None:
    rows = [{"id": f"q{k}", "question": "Who?", "context": "Ctx.", "golds": ["Ada"],
             "decoy_far": "Tasmania", "decoy_far_type": "place", "decoy_near": "Bob"}
            for k in range(4)]
    u = ta.units_for("squad", rows)
    assert [x[0].meta["cond"] for x in u] == ["C0", "C1", "C2", "C3"]
    assert u[1][0].data.endswith(ta.note("Tasmania")) and u[2][0].data.endswith(ta.note("Bob"))
    assert [it.meta["witness"] for it in u[3]] == [i.witness for i in EVAL_INJS] + ["HIJACKED"]
    s = ta.score(u[1][0], "Tasmania")
    assert s["decoy_hit"] and not s["correct"] and s["f1"] == 0.0
    yn = dict(rows[0], golds=["yes"], decoy_far="no", decoy_near=None)
    ub = ta.units_for("boolq", [yn])
    assert [x[0].meta["cond"] for x in ub] == ["C0", "C1", "C3"]
    assert ta.score(ub[1][0], "No.")["decoy_hit"] and ta.score(ub[0][0], "yes")["correct"]
    assert ta.score(u[3][0], "HIJACKED")["wf"] == "obeyed"


@pytest.mark.skipif(not (ta.DATA / "squad500.json").exists(), reason="run taskacc.py prepare")
def test_prepared_datasets() -> None:
    for ds in ta.CONDS:
        rows, _ = ta.load_items(ds)
        assert len(rows) == ta.N_ITEMS[ds] and len({r["id"] for r in rows}) == len(rows)
        for r in rows:
            assert ta.disjoint(r["decoy_far"], r["golds"])
            if ds != "boolq":
                assert not ta.decoy_hit(r["context"], r["decoy_far"])
                assert r["decoy_near"] and ta.disjoint(r["decoy_near"], r["golds"])
    assert len(ta.POOL) >= 30


def test_taskacc_summary(tmp_path) -> None:
    args = argparse.Namespace(arms=["undef", "def-s0"], limit=0, out_dir=str(tmp_path))
    rows = [{"id": f"q{k}", "question": "Who?", "context": "Ctx.", "golds": ["Ada Byron"],
             "decoy_far": "Tasmania", "decoy_far_type": "place", "decoy_near": "Bob"}
            for k in range(6)]
    for arm in args.arms:
        recs = []
        for u in ta.units_for("squad", rows):
            for it in u:
                bad = arm == "undef" and it.meta["cond"] != "C0"
                text = (it.meta["decoy"] or it.meta["witness"] or "x") if bad else "Ada Byron"
                recs.append({"key": it.key, **it.meta, "output": text, **ta.score(it, text)})
        _write(tmp_path / "transcripts" / f"{arm}.jsonl", recs)
    s = ta.summarize(args)
    assert s["cells"]["def-s0|squad|C1"]["retention_f1"] == 1.0
    assert s["cells"]["undef|squad|C3"]["obey_wf"]["k"] == 6
    assert s["paired"]["def-s0-vs-undef|squad|C1"]["f1_diff"] == 1.0
    dec = s["decision"]["def|squad|C1"]
    assert dec["preserved"] and not dec["loss_prevented"]  # McNemar p = 0.03 with 6 items


# ----------------------------------------------------------------------------- shared tokens
def _rec(idx: int, cond: str, decoy: str | None, gold: str, f1: float) -> dict:
    return {"ds": "squad", "cond": cond, "idx": idx, "golds": [gold], "decoy": decoy,
            "decoy_type": "in-context" if cond == "C2" else None, "em": f1, "f1": f1,
            "contain": f1 > 0, "correct": f1 > 0, "decoy_hit": False}


def _recs() -> dict:
    golds = ["Mughal emperors", "the Battle of Hastings", "1066"]
    near = ["Timur", "Hastings Road", "AT&amp;T"]  # item 1 shares 'hastings'; items 0 and 2 do not
    recs = {(a, ds, c): [] for a in ("undef", "def-s0") for ds, cs in ta.CONDS.items()
            for c in cs}  # summarize() keys every condition, empty when not run
    for arm, f1 in (("undef", 1.0), ("def-s0", 1.0)):
        for c in ("C0", "C2"):
            recs[(arm, "squad", c)] = [_rec(i, c, near[i] if c == "C2" else None, golds[i],
                                            f1 if c == "C0" or i == 1 else 0.0)
                                       for i in range(3)]
    return recs


def test_dp3_excludes_only_shared_token_decoys() -> None:
    assert ta.dp3_excluded(_recs()) == {"hotpot": [], "squad": [1]}
    assert ta.f1_score("the", "the Battle") == 0.0  # articles removed by the official F1


def test_entity_decoys_counted_per_condition() -> None:
    assert ta.entity_decoys(_recs()) == {"squad|C2": 1}


def test_tables_on_kept_items_stay_paired() -> None:
    recs = _recs()
    drop = set(ta.dp3_excluded(recs)["squad"])
    kept = {k: [r for r in rs if r["idx"] not in drop] for k, rs in recs.items()}
    arms = [parse_arm("undef"), parse_arm("def-s0")]
    cells, pairs, _ = ta._tables(kept, arms)
    assert cells["def-s0|squad|C2"]["n"] == 2 and cells["def-s0|squad|C2"]["f1"] == 0.0
    assert cells["def-s0|squad|C0"]["f1"] == 1.0
    assert pairs["def-s0-vs-undef|squad|C2"]["f1_diff"] == 0.0
