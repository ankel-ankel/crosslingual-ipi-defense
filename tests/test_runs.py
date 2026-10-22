"""Dry runs of the run() entry points of suite, taskacc, judge_dump, legit and channel_swap with a
tiny PEFT model on CPU: design freeze before transcripts, generation, resume-safe files, summary."""
from __future__ import annotations

import argparse
import copy
import json

import pytest
import torch

from ipidefense import paths
from ipidefense.data import documents as ld
from ipidefense.experiments import channel_swap as cs
from ipidefense.experiments import judge_dump as jd
from ipidefense.experiments import legit as lg
from ipidefense.experiments import suite as su
from ipidefense.experiments import taskacc as ta

pytestmark = pytest.mark.skipif(
    not (paths.DATA / "alpaca_cleaned.json").exists() or not (ta.DATA / "squad500.json").exists(),
    reason="needs data/alpaca_cleaned.json (scripts/prepare_data.py) and data/taskacc")


@pytest.fixture(scope="module")
def tiny_peft(qwen_tok):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(qwen_tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=4096, tie_word_embeddings=False,
                      bos_token_id=151643, eos_token_id=151645, pad_token_id=151643)
    tiny = Qwen2ForCausalLM(cfg).eval()
    lora = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
    return get_peft_model(copy.deepcopy(tiny), lora, adapter_name="def-s0").eval()


def test_run_paths_end_to_end(qwen_tok, tiny_peft, tmp_path, monkeypatch) -> None:
    for mod in (su, ta, jd, lg, cs):
        monkeypatch.setattr(mod, "load_eval_model", lambda arms: (tiny_peft, qwen_tok))
    monkeypatch.setattr(su, "RENDER_DIR", tmp_path / "r")
    monkeypatch.setattr(ld, "OUT", tmp_path / "docs.json")
    monkeypatch.setattr(jd, "MNT", 4)
    ns, arms = argparse.Namespace, ["undef", "def-s0"]
    su.run_exp(ns(exp="ml3", langs=["en"], arms=arms, seeds=[0], renderer="v2", batch=8, mnt=4,
                  limit_hosts=2, out_dir=str(tmp_path / "su")))
    ta.run(ns(arms=arms, mnt=4, limit=2, spot_unbatched=2, out_dir=str(tmp_path / "ta")))
    jd.run(ns(arms=arms, seeds=[0], dup_frac=0.1, n_sanity=2, limit_hosts=2,
              out_dir=str(tmp_path / "jd")))
    for exp in ("directives", "docs"):
        lg.run(ns(exp=exp, arms=arms, seeds=[0], batch=8, mnt=4, unbatched=2, n_boot=10,
                  limit=2, audit=2, out_dir=str(tmp_path / "lg")))
    cs.run(ns(arms=arms, seeds=[0], families=list(cs.FAMILIES), batch=8, hidden_layers=[1, 2],
              hidden_arms=["def-s0@0"], limit_hosts=2, out_dir=str(tmp_path / "cs")))
    smoke = {"su": "su/smoke/ml3", "ta": "ta/smoke", "jd": "jd/smoke",
             "lg1": "lg/directives/smoke", "lg2": "lg/docs/smoke", "cs": "cs/smoke"}
    for d in smoke.values():
        assert (tmp_path / d / "design.json").exists()
    assert (tmp_path / "su/smoke/ml3/summary.json").exists()
    recs = su.read_records(su.transcript_path(tmp_path / "su/smoke", su.parse_arm("def-s0"), 0,
                                              "ml3", "v2"))
    n_tmpl = len(su.suite_templates())
    assert len(recs) == n_tmpl * 2 and all(r["batch"] == 8 and r["mnt"] == 4 for r in recs.values())
    assert json.loads((tmp_path / "ta/smoke/summary.json").read_text())["spot_unbatched"]
    assert (tmp_path / "jd/smoke/judge_items.jsonl").exists()
    for d in ("lg/directives/smoke", "lg/docs/smoke", "cs/smoke"):
        assert (tmp_path / d / "summary.json").exists()
    assert (tmp_path / "cs/smoke/hidden/def-s0__s0.npz").exists()
    assert not (tmp_path / "cs/smoke/hidden/undef__s0.npz").exists()
