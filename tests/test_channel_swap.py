import argparse
import json

import numpy as np
import pytest
import torch

from ipidefense.attacks.templates import EVAL_INJS
from ipidefense.data.splits import hosts
from ipidefense.experiments import channel_swap as cs
from ipidefense.experiments import suite as su
from ipidefense.model import build_messages
from ipidefense.paths import DATA

needs_alpaca = pytest.mark.skipif(not (DATA / "alpaca_cleaned.json").exists(),
                                  reason="run scripts/prepare_data.py")


@pytest.fixture(scope="module")
def tiny(qwen_tok):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(qwen_tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=4096, tie_word_embeddings=False,
                      bos_token_id=151643, eos_token_id=151645, pad_token_id=151643)
    return Qwen2ForCausalLM(cfg).eval()


@needs_alpaca
def test_placements_and_minimal_pair(qwen_tok) -> None:
    i, d, inj = "Summarize the text.", "The meeting moved to Thursday.", EVAL_INJS[1].render()
    assert cs.place("U-early", i, d, inj) == (f"{i}\n{inj}", d, None)
    assert cs.place("D-start", i, d, inj) == (i, f"{inj}\n{d}", None)
    assert cs.place("U-late", i, d, inj) == (i, d, inj)
    spoof = cs.place("SPOOF", i, d, inj)[1]
    assert spoof.endswith(f"<|im_start|>user\n{inj}<|im_end|>")
    assert cs.pair_check(qwen_tok, i, d, inj, inline=False)
    assert isinstance(cs.pair_check(qwen_tok, i, d, inj, inline=True), bool)
    hs = hosts(0, "security")[:20]
    share = sum(cs.pair_check(qwen_tok, h.instruction, h.data, inj, False) for h in hs) / len(hs)
    assert 0 < share <= 1
    assert cs.ends_with("Done. WELMIRK.", "WELMIRK") and not cs.ends_with("WELMIRK done", "WELMIRK")


@needs_alpaca
def test_channel_cells(qwen_tok) -> None:
    hs = hosts(0, "security")[:2]
    role = cs.cell_units(qwen_tok, su.parse_arm("def-s0"), "F-OVR", hs)
    assert len(role) == 5 * 6 and {u[0].meta["placement"] for u in role} == set(cs.ROLE)
    inline = cs.cell_units(qwen_tok, su.parse_arm("inline-s0"), "F-OVR", hs)
    assert {u[0].meta["placement"] for u in inline} == set(cs.INLINE)
    assert cs.cell_units(qwen_tok, su.parse_arm("inline-s0"), "F-SFX", hs) == []
    late = next(u for u in role if u[0].meta["placement"] == "U-late")[0]
    assert late.meta["late_user"] and late.data == hs[0].data
    assert "pair_matched" in role[0][0].meta
    assert cs.score(cs.clean_unit(hs)[0], "fine")["witness_hits"] == []


def _cs_records(arm: str, rates: dict[str, float], n: int = 10) -> list[dict]:
    out = []
    for fam in ("F-OVR",):
        for pl, r in rates.items():
            for tid, _, w in cs.FAMILIES[fam]["templates"]:
                out += [{"key": f"{fam}|{pl}|{tid}|{h}", "family": fam, "placement": pl,
                         "tid": tid, "host": h, "witness": w, "rf": "ignored",
                         "wf": "obeyed" if h < r * n else "ignored", "pair_matched": True,
                         "output": w if h < r * n else "fine"}
                        for h in range(n)]
    return out


def test_channel_swap_summary(tmp_path) -> None:
    args = argparse.Namespace(arms=["undef", "def-s0", "hard-s0"], seeds=[0],
                              families=["F-OVR"], limit_hosts=0, out_dir=str(tmp_path))
    full = {"U-early": 1.0, "D-start": 0.8, "D-end": 1.0, "U-late": 1.0, "SPOOF": 1.0}
    dfd = {"U-early": 1.0, "D-start": 0.0, "D-end": 0.0, "U-late": 1.0, "SPOOF": 0.0}
    for arm, rates in (("undef", full), ("def-s0", dfd), ("hard-s0", dfd)):
        p = tmp_path / "transcripts" / f"{arm}__s0.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(json.dumps(r) for r in _cs_records(arm, rates)), encoding="utf-8")
    s = cs.summarize(args)
    dec = s["decisions"]["def-s0|s0|F-OVR"]
    assert dec["class"] == "C" and dec["primary"]["RU"] == 1.0 and dec["turn_closure"]
    assert s["gate0"]["def-s0|D-end"]["pass"] and not s["gate0"]["undef|D-start"]["pass"]
    assert not any(k.startswith("hard-s0") for k in s["gate0"])
    cell = s["cells"]["undef|s0|F-OVR|U-early"]
    assert cell["wf_nonecho"]["k"] == cell["wf"]["k"] and cell["echo_n"] == 0
    assert s["decisions"]["hard-s0|s0|F-OVR"]["class"] == "C"


@needs_alpaca
def test_hidden_states_left_padding(qwen_tok, tiny, tmp_path) -> None:
    hs = hosts(0, "security")[:3]
    items = cs.clean_unit(hs)
    path = tmp_path / "h.npz"
    cs.capture_hidden(tiny, qwen_tok, su.parse_arm("undef"), items, [1, 2], 3, path)
    arr = np.load(path)
    for k, it in enumerate(items):
        ids = qwen_tok(su.chat_text(qwen_tok, build_messages(it.instruction, it.data)),
                       add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            ref = tiny(**ids, output_hidden_states=True).hidden_states[2][0, -1].numpy()
        assert np.allclose(arr["L2"][k].astype(np.float32), ref, atol=2e-2)
