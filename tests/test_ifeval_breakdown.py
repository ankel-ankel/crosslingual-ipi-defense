"""IFEval breakdown statistics on synthetic transcripts, and the subset labels."""
from __future__ import annotations

import json

import pytest

from ipidefense.evaluation.ifeval import PROMPTS
from ipidefense.experiments import ifeval_breakdown as bd
from ipidefense.experiments import ifeval_subsets as subs

needs_ifeval = pytest.mark.skipif(not PROMPTS.exists() or not bd.LABELS.exists(),
                                  reason="run scripts/prepare_data.py")

# 40 prompts: 10 with one CS instruction, 10 IL, 10 SF, 10 OTHER
SUBS = ["CS"] * 10 + ["IL"] * 10 + ["SF"] * 10 + ["OTHER"] * 10
IID = {"CS": "startend:end_checker", "IL": "change_case:english_lowercase",
       "SF": "length_constraints:number_words", "OTHER": "punctuation:no_comma"}


def _labels() -> list[dict]:
    return [{"idx": i, "subsets": [s], "instruction_id_list": [IID[s]]}
            for i, s in enumerate(SUBS)]


def _rows() -> list[dict]:
    return [{"instruction_id_list": [IID[s]],
             "kwargs": [{"relation": "less than", "num_words": 50} if s == "SF" else {}]}
            for s in SUBS]


def _arm(fail: dict[str, int], tokens: int = 300, eos: bool = True,
         resp: str = "Done.") -> dict[int, dict]:
    """fail[s] = number of prompts in subset s whose instruction fails."""
    out, seen = {}, {s: 0 for s in IID}
    for i, s in enumerate(SUBS):
        ok = seen[s] >= fail.get(s, 0)
        seen[s] += 1
        out[i] = {"idx": i, "flags": [ok], "n_tokens": tokens, "stopped_eos": eos,
                  "response": resp}
    return out


def test_cs_concentrated_drop_supports_h1_with_length_signature() -> None:
    undef = _arm({}, tokens=300)
    arms = [_arm({"CS": 6}, tokens=160) for _ in range(3)]  # CS drops 60 pp, others 0
    res = bd.analyze(undef, arms, _labels(), n_boot=2000, rows_kwargs=_rows())
    sub = res["subsets"]
    assert sub["CS"]["delta"] == pytest.approx(-0.6) and sub["OTHER"]["delta"] == 0.0
    assert sub["CS-OTHER"]["delta"] == pytest.approx(-0.6) and sub["CS-OTHER"]["ci95"][1] < 0
    h = res["hypotheses"]
    assert h["H1_early_stopping"] is True and h["H2_spillover"] is False
    assert h["H1_length_signatures"]["eos_140_180_2x"] is True  # 160 tokens, undef 300
    assert res["length"]["median_cs_length_ratio"] == pytest.approx(160 / 300, abs=1e-4)


def test_cs_drop_without_length_signature_is_not_h1() -> None:
    undef = _arm({}, tokens=300)
    arms = [_arm({"CS": 6}, tokens=300) for _ in range(3)]
    h = bd.analyze(undef, arms, _labels(), n_boot=2000, rows_kwargs=_rows())["hypotheses"]
    assert h["H1_early_stopping"] is False and h["neither"] is True


def test_il_spillover_supports_h2_and_diffuse_drop_supports_neither() -> None:
    undef = _arm({})
    il = bd.analyze(undef, [_arm({"IL": 5})] * 3, _labels(), 2000, _rows())["hypotheses"]
    assert il["H2_spillover"] is True and il["H1_early_stopping"] is False
    flat = {s: 3 for s in IID}
    diff = bd.analyze(undef, [_arm(flat)] * 3, _labels(), 2000, _rows())["hypotheses"]
    assert diff["neither"] is True


def test_seed_pooling_averages_defended_flags() -> None:
    undef = _arm({})
    arms = [_arm({"CS": 3}), _arm({"CS": 6}), _arm({"CS": 9})]
    res = bd.analyze(undef, arms, _labels(), 500, _rows())
    assert res["subsets"]["CS"]["delta"] == pytest.approx(-0.6)
    assert res["subsets"]["CS"]["n_instructions"] == 10


def test_mid_sentence_and_cap_diagnostics() -> None:
    assert bd.mid_sentence("and then the") and bd.mid_sentence("items:")
    assert not bd.mid_sentence("Done.") and not bd.mid_sentence("```") and not bd.mid_sentence("")
    undef = _arm({}, resp="Done.")
    arms = [_arm({}, resp="and then the", eos=True), _arm({}, eos=False, tokens=512),
            _arm({}, resp="Done.")]
    d = bd.analyze(undef, arms, _labels(), 200, _rows())["length"]
    assert d["undefended"]["mid_sentence_eos"] == 0
    assert d["defended"]["hit_cap"] == pytest.approx(1 / 3, abs=1e-4)
    assert d["defended"]["mid_sentence_eos"] == pytest.approx(0.5)  # 40 of 80 EOS responses


def test_long_instruction_mask() -> None:
    rows = [{"instruction_id_list": ["length_constraints:number_words", "punctuation:no_comma"],
             "kwargs": [{"relation": "at least", "num_words": 400}, {}]},
            {"instruction_id_list": ["length_constraints:number_words"],
             "kwargs": [{"relation": "at least", "num_words": 300}]}]
    labels = [{"idx": 0, "subsets": ["CS", "OTHER"],
               "instruction_id_list": rows[0]["instruction_id_list"]},
              {"idx": 1, "subsets": ["CS"], "instruction_id_list": rows[1]["instruction_id_list"]}]
    assert bd.long_mask(labels, rows) == [[True, False], [False]]


@needs_ifeval
def test_real_labels_have_15_long_instructions() -> None:
    labels = json.loads(bd.LABELS.read_text(encoding="utf-8"))["labels"]
    assert sum(map(sum, bd.long_mask(labels, None))) == bd.N_LONG


def test_subset_labels() -> None:
    assert subs.label("length_constraints:number_words", {"relation": "at least"}) == "CS"
    assert subs.label("length_constraints:number_sentences", {"relation": "less than"}) == "SF"
    assert subs.label("change_case:capital_word_frequency", {}) == "IL"
    assert subs.label("startend:quotation", None) == "CS"
    assert subs.label("punctuation:no_comma", {}) == "OTHER"
    with pytest.raises(ValueError):
        subs.label("length_constraints:number_words", {"relation": "about"})


@needs_ifeval
def test_shipped_subsets_match_build() -> None:
    assert subs.build() == json.loads(subs.OUT.read_text(encoding="utf-8"))
