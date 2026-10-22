import math
from pathlib import Path

import pytest
import torch

from ipidefense import paths
from ipidefense.attacks import recipes as rcp
from ipidefense.model import generate
from ipidefense.training import sets as ts
from ipidefense.training import sft
from ipidefense.training.lora import LORA_TARGETS

needs_alpaca = pytest.mark.skipif(not (paths.DATA / "alpaca_cleaned.json").exists(),
                                  reason="run scripts/prepare_data.py")


class FakeTranslator:
    """NLLB stand-in that tags the segment with the language."""

    def translate(self, text: str, target_lang: str) -> str:
        text = text.strip()
        return f"<{target_lang}> {text}" if text else text


@pytest.fixture(scope="module")
def tok(qwen_tok):
    return qwen_tok


@pytest.fixture(scope="module")
def fake_tr() -> FakeTranslator:
    return FakeTranslator()


def tiny_lora_model(vocab: int, dropout: float = 0.0):
    """2-layer random Qwen2 with our LoRA config, fp32 on CPU."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
        tie_word_embeddings=False, bos_token_id=151643, eos_token_id=151645,
        pad_token_id=151643,
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=dropout, bias="none",
                      task_type="CAUSAL_LM", target_modules=list(LORA_TARGETS))
    model = get_peft_model(Qwen2ForCausalLM(cfg), lora)
    model.enable_input_require_grads()
    return model


# data mix
@needs_alpaca
@pytest.mark.parametrize(("seed", "n_clean"), [(0, 902), (1, 909), (2, 882)])
def test_a0_clean_counts(seed: int, n_clean: int, fake_tr: FakeTranslator) -> None:
    hs = rcp.train_hosts(seed)
    plan = rcp.plan_a0(hs, seed, fake_tr)
    assert len(plan) == 1500
    assert sum(p["kind"] == "clean" for p in plan) == n_clean
    for p in plan:
        if p["kind"] == "witness":
            assert p["lang"] == rcp.LANGS[p["host"] % 3]
            assert p["data"] == f"{hs[p['host']].data}\n{p['injection']}"


# generation
@needs_alpaca
def test_generation_matches_model_generate(tok) -> None:
    model = tiny_lora_model(len(tok))
    model.eval()
    eos = ts.eos_ids(model, tok)
    hs = rcp.train_hosts(0)[:3]
    items = [(h.instruction, h.data) for h in hs]
    with model.disable_adapter():
        for instr, data in items:
            got = ts.generate_one(model, tok, instr, data, 12, eos)
            assert got["text"] == generate(model, tok, instr, data=data, max_new_tokens=12)
            assert got["n_tokens"] <= 12
    rec = ts.gen_record(tok, [40, 41, 151645, 151645], 160, frozenset({151645}))
    assert (rec["n_tokens"], rec["ended_eos"], rec["cap_hit"]) == (2, True, False)
    rec = ts.gen_record(tok, [40, 41, 42], 3, frozenset({151645}))
    assert (rec["n_tokens"], rec["ended_eos"], rec["cap_hit"]) == (3, False, True)


# guards
def test_adapter_dir_guards(tmp_path: Path) -> None:
    for bad in ("../evil", "a/b", ""):
        with pytest.raises(ValueError):
            sft.adapter_dir(bad, models=tmp_path)
    used = tmp_path / "a1-seed0"
    used.mkdir()
    (used / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        sft.adapter_dir("a1-seed0", models=tmp_path)
    assert sft.adapter_dir("a1-seed0", overwrite=True, models=tmp_path) == used.resolve()


def test_unsupervised_examples_are_rejected() -> None:
    with pytest.raises(ValueError):
        sft.length_stats([(10, 20), (1024, 1100)], 1024)


def test_loss_tee_records_training_log_lines() -> None:
    class Sink:
        def __init__(self) -> None:
            self.text = ""

        def write(self, s: str) -> int:
            self.text += s
            return len(s)

        def flush(self) -> None:
            pass

    sink = Sink()
    tee = sft.LossTee(sink)  # type: ignore[arg-type]
    tee.write("epoch 0 step 10 loss 1.2345")
    tee.write("\n")
    tee.write("epoch 0 step 20 loss nan\n")
    assert sink.text.count("\n") == 2
    assert tee.losses[0] == {"epoch": 0, "step": 10, "loss": 1.2345}
    assert math.isnan(tee.losses[1]["loss"])
