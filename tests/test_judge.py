import argparse
import json
from pathlib import Path

import pytest

from ipidefense import paths
from ipidefense.attacks.templates import EVAL_INJS
from ipidefense.data.splits import hosts
from ipidefense.experiments import judge as rj
from ipidefense.experiments import judge_dump as jd
from ipidefense.experiments import judge_validate as v
from ipidefense.experiments import suite as su

needs_alpaca = pytest.mark.skipif(not (paths.DATA / "alpaca_cleaned.json").exists(),
                                  reason="run scripts/prepare_data.py")


# judge items
@needs_alpaca
def test_judge_dump_blind_items(tmp_path) -> None:
    args = argparse.Namespace(arms=["undef", "def-s0"], seeds=[0], dup_frac=0.1, n_sanity=3,
                              limit_hosts=0, out_dir=str(tmp_path))
    for arm in args.arms:
        a = su.parse_arm(arm)
        for part, (split, _) in jd.PARTS.items():
            units = jd.units_for(part, hosts(0, split)[:5])
            recs = [{"key": it.key, **it.meta, "output": f"{arm} answer {it.key}",
                     **jd.score(it, "HIJACKED" if arm == "undef" else "fine")}
                    for u in units for it in u]
            p = jd.tpath(tmp_path, a, 0, part)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    s = jd.dump(args)
    items = [json.loads(x) for x in (tmp_path / "judge_items.jsonl").read_text(
        encoding="utf-8").splitlines()]
    key = json.loads((tmp_path / "judge_key.json").read_text(encoding="utf-8"))
    assert s["counts"]["attacked"] == 2 * 3 * 5 and s["counts"]["benign"] == 5
    assert s["counts"]["duplicates"] == round(0.1 * (35 + s["counts"]["sanity_identical"]))
    for it in items:
        assert set(it) == {"item_id", "instruction", "data_clean", "injection",
                           "reference_output", "candidate_output", "arm_blind_id", "meta"}
        assert it["item_id"] in key["items"] and it["arm_blind_id"] in key["arms"]
        assert "def-s0" not in json.dumps(it["meta"]) and it["injection"] is None or (
            it["injection"] in [i.render() for i in EVAL_INJS])
    assert len({i["item_id"] for i in items}) == len(items)


# split, merge, score
def _dump(tmp: Path, n_att: int = 200, n_ben: int = 100) -> Path:
    items, key = [], {}
    for k in range(n_att + n_ben):
        kind = "attacked" if k < n_att else "benign"
        iid = f"i{k:04d}"
        items.append({"item_id": iid, "instruction": "t", "data_clean": "d",
                      "injection": "x" if kind == "attacked" else None,
                      "reference_output": "r", "candidate_output": "c",
                      "arm_blind_id": "z", "meta": {"task": kind}})
        key[iid] = {"pair_type": kind, "arm": "def-s0", "seed": 0, "dup_of": None,
                    "sanity_expect": None}
    (tmp / "judge_items.jsonl").write_text("\n".join(json.dumps(i) for i in items),
                                           encoding="utf-8")
    (tmp / "judge_key.json").write_text(json.dumps({"items": key}), encoding="utf-8")
    return tmp


def _judge(tmp: Path, name: str, rule) -> Path:
    items = rj.read_jsonl(tmp / "judge_items.jsonl")
    js = [{"item_id": it["item_id"], **rule(k, it)} for k, it in enumerate(items)]
    p = tmp / f"{name}.json"
    p.write_text(json.dumps({"judgments": js}), encoding="utf-8")
    return p


def _ok(q1: int = 5, q3: str = "no") -> dict:
    return {"q1": q1, "q2": 5, "q3": q3, "q4": "no"}


def test_split_hides_arm_and_covers_all(tmp_path: Path) -> None:
    d = _dump(tmp_path, 50, 20)
    paths_a = rj.split(d, "A", 30)
    rows = [json.loads(x) for p in paths_a for x in p.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 70 and all("arm_blind_id" not in r for r in rows)
    order_a = [r["item_id"] for r in rows]
    rows_b = [json.loads(x) for p in rj.split(d, "B", 30)
              for x in p.read_text(encoding="utf-8").splitlines()]
    assert sorted(order_a) == sorted(r["item_id"] for r in rows_b)
    assert order_a != [r["item_id"] for r in rows_b]  # independent shuffles


def test_outputs_go_to_the_output_directory(tmp_path: Path) -> None:
    d, out = tmp_path / "data", tmp_path / "out"
    d.mkdir()
    _dump(d, 20, 10)
    rj.split(d, "A", 10, out)
    assert len(list((out / "judge_chunks" / "A").glob("chunk_*.jsonl"))) == 3
    for name in ("A", "B"):
        assert rj.merge(d, name, [str(_judge(d, name, lambda k, it: _ok()))], out) == 0
    assert not (d / "judge_A.jsonl").exists() and not (d / "judge_chunks").exists()
    for name in ("judge_items.jsonl", "judge_key.json"):
        (out / name).write_bytes((d / name).read_bytes())
    assert rj.score(out, tmp_path / "summary")["n_scored"] == 30
    assert (tmp_path / "summary" / "judge_summary.json").exists()
    assert not (out / "judge_summary.json").exists()


def test_merge_flags_parse_failures(tmp_path: Path) -> None:
    d = _dump(tmp_path, 90, 10)
    good = _judge(d, "a", lambda k, it: _ok())
    assert rj.merge(d, "A", [str(good)]) == 0
    bad = _judge(d, "b", lambda k, it: _ok(q1=7) if k < 3 else _ok())  # 3% invalid
    assert rj.merge(d, "B", [str(bad)]) == 2
    meta = json.loads((d / "judge_B_merge.json").read_text(encoding="utf-8"))
    assert meta["rerun_required"] is True and meta["invalid_or_duplicate"] == 3


def test_kappa_and_newcombe() -> None:
    assert rj.kappa([True, False] * 10, [True, False] * 10) == 1.0
    assert rj.kappa([True] * 5 + [False] * 5, [False] * 5 + [True] * 5) == -1.0
    lo, hi = rj.newcombe(90, 100, 90, 100)
    assert lo < 0 < hi and abs(lo + hi) < 1e-9


def test_decision_preserved_when_attacked_matches_benign(tmp_path: Path) -> None:
    d = _dump(tmp_path, 1000, 300)
    for name in ("A", "B"):  # 90% faithful everywhere, 2% of attacked items influenced
        f = _judge(d, name, lambda k, it: _ok(q1=4 if k % 10 else 3,
                                              q3="yes" if k < 1000 and k % 50 == 0 else "no"))
        rj.merge(d, name, [str(f)])
    res = rj.score(d)
    dec = res["decision"]
    assert dec["fidelity_preserved"] is True and dec["influenced_rate"] == pytest.approx(0.02)
    assert res["agreement"]["acceptance"] == "primary"


def test_constant_labels_give_undefined_kappa_and_no_primary(tmp_path: Path) -> None:
    d = _dump(tmp_path, 100, 50)
    for name in ("A", "B"):
        rj.merge(d, name, [str(_judge(d, name, lambda k, it: _ok()))])
    agree = rj.score(d)["agreement"]
    assert agree["kappa_q3"] is None and agree["acceptance"] == "exploratory"


def test_decision_fails_on_influence_or_gap(tmp_path: Path) -> None:
    d = _dump(tmp_path, 1000, 300)
    att = lambda k, it: it["meta"]["task"] == "attacked"
    fa = _judge(d, "A", lambda k, it: _ok(q3="yes" if att(k, it) and k % 20 == 0 else "no"))
    fb = _judge(d, "B", lambda k, it: _ok())
    rj.merge(d, "A", [str(fa)])
    rj.merge(d, "B", [str(fb)])
    dec = rj.score(d)["decision"]
    assert dec["influenced_rate"] == pytest.approx(0.05) and dec["fidelity_preserved"] is False
    # a 10-pp fidelity gap under attack, no influence
    ga = _judge(d, "A", lambda k, it: _ok(q1=2 if att(k, it) and k % 10 == 0 else 5))
    rj.merge(d, "A", [str(ga)])
    dec2 = rj.score(d)["decision"]
    assert dec2["attacked_minus_benign_faithful"] == pytest.approx(-0.1)
    assert dec2["fidelity_preserved"] is False


def test_undef_arm_counted_once(tmp_path: Path) -> None:
    d = _dump(tmp_path, 20, 0)
    key = json.loads((d / "judge_key.json").read_text(encoding="utf-8"))
    for k, val in key["items"].items():
        val["arm"] = "undef" if int(k[1:]) % 2 else "def-s0"
    (d / "judge_key.json").write_text(json.dumps(key), encoding="utf-8")
    for name in ("A", "B"):
        rj.merge(d, name, [str(_judge(d, name, lambda k, it: _ok()))])
    t = rj.score(d)["table"]
    assert t["undef|attacked"]["n"] == 10 and t["def|attacked"]["n"] == 10
    assert t["def-s0|attacked"]["n"] == 10


def _sanity_dump(tmp: Path, n_ident: int, n_witness: int) -> Path:
    """Identical pairs (Q1=5), then witness-only items. i0029 duplicates i0000."""
    d = _dump(tmp, 30, 0)
    key = json.loads((d / "judge_key.json").read_text(encoding="utf-8"))
    for k in range(n_ident + n_witness):
        key["items"][f"i{k:04d}"]["sanity_expect"] = "Q1=5" if k < n_ident else "Q1=1, Q3=yes"
    key["items"]["i0029"] = {**key["items"]["i0000"], "dup_of": "i0000"}
    (d / "judge_key.json").write_text(json.dumps(key), encoding="utf-8")
    return d


def _witness_ok(fails: set[str]):
    def rule(k: int, it: dict) -> dict:
        if it["item_id"] in fails:
            return _ok(q1=4)
        return _ok(q1=1, q3="yes") if 10 <= k < 12 else _ok()
    return rule


@pytest.mark.parametrize("fails,review", [(set(), False), ({"i0001"}, False),
                                          ({"i0001", "i0002"}, True)])
def test_sanity_excludes_duplicates_and_uses_10pct_rule(tmp_path: Path, fails, review) -> None:
    d = _sanity_dump(tmp_path, 10, 2)
    for name in ("A", "B"):
        rj.merge(d, name, [str(_judge(d, name, _witness_ok(fails)))])
    res = rj.score(d)
    s = res["sanity"]
    assert s["Q1=5"]["n"] == 10 and s["Q1=5"]["ok"] == 10 - len(fails)  # dup i0029 not counted
    assert s["Q1=1, Q3=yes"] == {"n": 2, "ok": 2, "fail_rate": 0.0}
    assert res["sanity_check"]["prompt_review_required"] is review  # exactly 10% does not fire
    assert res["test_retest"]["A"]["n"] == 1


def test_conservative_labels() -> None:
    lab = rj._labels(_ok(q1=4, q3="yes"), _ok(q1=3, q3="no"))
    assert lab["faithful"] is False and lab["influenced"] is True
    assert lab["faithful_A"] is True and lab["influenced_B"] is False


# validation
IDS = [f"Q00{i}" for i in range(1, 7)]
STRATA = ["attacked|def-s0", "attacked|def-s0", "benign|def-s0", "benign|def-s0",
          "attacked|undef|0", "attacked|undef|0"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    rows = [{"item_id": q} for q in IDS]
    akey = {q: {"item_id": f"orig{q}", "stratum": s} for q, s in zip(IDS, STRATA, strict=True)}
    monkeypatch.setattr(v, "_audit", lambda src: (rows, akey))
    passes_ab = [{"item_id": f"orig{q}", "q1": q1, "q3": q3, "q4": "no"}
                 for q, q1, q3 in zip(IDS, (5, 4, 5, 3, 1, 2),
                                      ("no", "no", "no", "no", "yes", "yes"), strict=True)]
    monkeypatch.setattr(rj, "read_jsonl", lambda path: passes_ab)
    (tmp_path / "dump" / "validation").mkdir(parents=True)
    return tmp_path


def _reply(path: Path, q1s: tuple[int, ...], skip: str | None = None) -> str:
    js = [{"item_id": q, "q1": q1, "q2": 3, "q3": "yes" if q1 <= 2 else "no", "q4": "no"}
          for q, q1 in zip(IDS, q1s, strict=True) if q != skip]
    path.write_text("Here you go:\n```json\n" + json.dumps({"judgments": js}) + "\n```",
                    encoding="utf-8")
    return str(path)


def test_labelled_ingest_never_overwrites(env) -> None:
    dump, out = env / "dump", env / "out"
    v.ingest_web(_reply(env / "d.txt", (5, 4, 4, 3, 1, 1)), "D", "GPT-test", dump, out)
    rec = json.loads((out / "validation" / "pass_D.json").read_text(encoding="utf-8"))
    assert rec["model"] == "GPT-test" and set(rec["judgments"]) == set(IDS)
    with pytest.raises(SystemExit, match="never overwritten"):
        v.ingest_web(_reply(env / "d2.txt", (5, 5, 5, 5, 5, 5)), "D", None, dump, out)
    (dump / "validation" / "pass_C.json").write_text("{}", encoding="utf-8")  # shipped pass
    with pytest.raises(SystemExit, match="never overwritten"):
        v.ingest_web(_reply(env / "c.txt", (5, 5, 5, 5, 5, 5)), "C", None, dump, out)
    with pytest.raises(SystemExit, match="label must be one of"):
        v.ingest_web(_reply(env / "a.txt", (5, 4, 4, 3, 1, 1)), "A", None, dump, out)
    with pytest.raises(SystemExit, match="missing"):
        v.ingest_web(_reply(env / "e.txt", (5, 4, 4, 3, 1, 1), skip="Q003"), "E", None, dump, out)
    assert not (out / "validation" / "pass_E.json").exists()


def test_report_includes_every_web_pass(env) -> None:
    dump, out = env / "dump", env / "out"
    v.ingest_web(_reply(env / "c.txt", (5, 4, 5, 3, 1, 2)), "C", None, out, out)
    (out / "validation" / "pass_C.json").replace(dump / "validation" / "pass_C.json")  # shipped
    v.ingest_web(_reply(env / "e.txt", (5, 4, 4, 4, 1, 1)), "E", "Gemini-test", dump, out)
    v.report(dump, out)
    rep = json.loads((out / "validation" / "report.json").read_text(encoding="utf-8"))
    assert rep["raters"] == ["A", "B", "C", "E"]
    assert {"A~B", "A~C", "A~E", "B~C", "B~E", "C~E"} == set(rep["agreement"])
    assert rep["agreement"]["A~C"]["q1_binary_agree"] == 6
    assert rep["agreement"]["A~E"]["q1_binary_agree"] == 5  # Q004: A 3 (not faithful), E 4
    assert rep["rates_by_stratum"]["E"]["benign|def-s0"] == {"n": 2, "faithful": 2, "influenced": 0}
    assert (out / "validation" / "report.md").exists()
