import argparse
import json
import random
from pathlib import Path

import pytest
import torch

from ipidefense import arms, paths
from ipidefense.attacks.recipes import RENDER_AUDIT, RENDER_CACHE, train_hosts
from ipidefense.attacks.templates import TRAIN_TEMPLATES, TRAIN_WITNESSES
from ipidefense.attacks.witnesses import WitnessInjection
from ipidefense.data.splits import hosts, hosts_for
from ipidefense.evaluation import ifeval_logged as ifl
from ipidefense.evaluation import runtime as rt
from ipidefense.experiments import backbone_rules as rules
from ipidefense.experiments import backbones as bb
from ipidefense.experiments import suite as su
from ipidefense.model import (
    ADAPTER_DIR,
    MISTRAL_DIR,
    QWEN_DIR,
    SEALION_DIR,
    build_messages,
    generate_batch,
    mistral_tokenizer,
    sealion_tokenizer,
)
from ipidefense.rendering.literal import render_inj_literal
from ipidefense.training.sets import eos_ids, sha256_file
from ipidefense.utils import Item, Transcript, sha, write_json

INLINE = {b: inline for b, (_, inline) in arms.BACKBONES.items()}
ALPACA = paths.DATA / "alpaca_cleaned.json"
needs_alpaca = pytest.mark.skipif(not ALPACA.exists(), reason="run scripts/prepare_data.py")
needs_models = pytest.mark.skipif(
    not all((d / f).exists() for d in (QWEN_DIR, SEALION_DIR, MISTRAL_DIR)
            for f in ("config.json", "generation_config.json"))
    or not (ADAPTER_DIR / "tokenizer_config.json").exists(),
    reason="base models not under models/")


def _has_tokenizer(d: Path) -> bool:
    return (d / "tokenizer_config.json").exists()


# fixtures
@pytest.fixture(scope="module")
def stok():
    if not (_has_tokenizer(SEALION_DIR) and _has_tokenizer(ADAPTER_DIR)):
        pytest.skip("SEA-LION / Meta-SecAlign tokenizers not under models/")
    return sealion_tokenizer()


@pytest.fixture(scope="module")
def mtok():
    if not _has_tokenizer(MISTRAL_DIR):
        pytest.skip("Mistral tokenizer not under models/")
    return mistral_tokenizer()


def _tiny_llama(vocab: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=False, bos_token_id=128000,
                      eos_token_id=[128001, 128008, 128009])
    return LlamaForCausalLM(cfg).eval()


def _tiny_mistral(vocab: int):
    from transformers import MistralConfig, MistralForCausalLM

    torch.manual_seed(0)
    cfg = MistralConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=4096, tie_word_embeddings=False,
                        bos_token_id=1, eos_token_id=2)
    return MistralForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def backbones(stok, mtok) -> dict:
    """backbone -> (tiny model, real tokenizer)."""
    return {"sealion": (_tiny_llama(len(stok)), stok), "mistral": (_tiny_mistral(len(mtok)), mtok)}


def _with_lora(model, name: str = "default"):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                     "up_proj", "down_proj"])
    return get_peft_model(model, cfg, adapter_name=name)


def _fake_adapter(tmp_path: Path, base_dir_name: str) -> Path:
    d = tmp_path / f"adapter-{base_dir_name}"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": f"\\models\\{base_dir_name}"}), encoding="utf-8")
    return d


class Dummy:
    """Stand-in for a loaded backbone."""

    def eval(self):
        return self


# registry
def test_backbone_registry_entries() -> None:
    for b, (pub, inline) in arms.BACKBONES.items():
        assert arms.ADAPTERS[f"{b}-undef"] == (b, "", inline)
        assert arms.ADAPTERS[f"{b}-def-s0"] == (b, pub, inline)
        assert arms.ADAPTER_SEED[f"{b}-undef"] is None and arms.ADAPTER_SEED[f"{b}-def-s0"] == 0
        for cfg in arms.BACKBONE_CONFIGS:
            base, adir, inl = arms.ADAPTERS[f"{b}-cfg{cfg}-s0"]
            assert (base, inl) == (b, inline) and arms.ADAPTER_SEED[f"{b}-cfg{cfg}-s0"] == 0
            assert paths.MODELS / adir == arms.backbone_adapter_dir(b, cfg)
            assert adir == f"{b}-cfg{cfg}-seed0"
    assert INLINE == {"sealion": False, "mistral": True}
    assert {b: arms.BASE_DIRS[b] for b in ("qwen", "sealion", "mistral")} == {
        "qwen": QWEN_DIR.name, "sealion": SEALION_DIR.name, "mistral": MISTRAL_DIR.name}


@needs_alpaca
def test_backbone_arms_are_leak_safe() -> None:
    with pytest.raises(ValueError):  # seed-0 adapters only on seed-0 slices
        hosts_for("sealion-cfgA-s0", "security", 1)
    assert hosts_for("mistral-undef", "security", 0) == hosts(0, "security")


def test_parse_arm_and_backbone_mixing() -> None:
    a = rt.parse_arm("mistral-cfgA-s0")
    assert (a.inline, a.own_seed, a.defended, a.family) == (True, 0, True, "mistral-cfgA")
    s = rt.parse_arm("sealion-undef")
    assert (s.inline, s.own_seed, s.defended) == (False, None, False)
    with pytest.raises(ValueError):  # StruQ filter strips Qwen ChatML, so Qwen arms only
        rt.parse_arm("sealion-def-s0+filter")
    assert rt.parse_arm("def-s0+filter").filtered
    with pytest.raises(ValueError):
        rt.parse_arm("nope")
    assert arms.arms_base(["undef", "def-s0"]) == "qwen" and arms.arms_base([]) == "qwen"
    assert arms.arms_base(["mistral-undef", "mistral-cfgB-s0"]) == "mistral"
    for mixed in (["sealion-undef", "mistral-undef"], ["undef", "sealion-undef"]):
        with pytest.raises(ValueError):
            arms.arms_base(mixed)
    plan = rt.arm_plan(["sealion-undef", "sealion-cfgA-s0"], [0])
    assert [(x.spec, sd) for x, sd in plan] == [("sealion-undef", 0), ("sealion-cfgA-s0", 0)]


def test_check_adapter_base(tmp_path: Path, monkeypatch) -> None:
    arms.check_adapter_base("sealion-undef")  # undefended arms pass
    right = _fake_adapter(tmp_path, "Llama-SEA-LION-v3-8B-IT")
    monkeypatch.setitem(arms.ADAPTERS, "sealion-def-s0", ("sealion", str(right), False))
    arms.check_adapter_base("sealion-def-s0")
    wrong = _fake_adapter(tmp_path, "Mistral-7B-Instruct-v0.3")
    monkeypatch.setitem(arms.ADAPTERS, "sealion-cfgA-s0", ("sealion", str(wrong), False))
    with pytest.raises(ValueError):  # identical LoRA shapes would load silently
        arms.check_adapter_base("sealion-cfgA-s0")
    monkeypatch.setitem(arms.ADAPTERS, "mistral-cfgA-s0", ("mistral", str(wrong), True))
    arms.check_adapter_base("mistral-cfgA-s0")
    monkeypatch.setitem(arms.ADAPTERS, "mistral-cfgB-s0", ("mistral", str(tmp_path / "x"), True))
    with pytest.raises(FileNotFoundError):
        arms.check_adapter_base("mistral-cfgB-s0")


# templates
def test_template_routing_per_backbone(stok, mtok) -> None:
    it = Item("k", "Summarize.", "Some data.", {})
    sl = su.messages_for(it, rt.parse_arm("sealion-def-s0"))
    assert sl == build_messages("Summarize.", "Some data.", False)
    assert su.chat_text(stok, sl) == (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSummarize.<|eot_id|>"
        "<|start_header_id|>input<|end_header_id|>\n\nSome data.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n")
    ms = su.messages_for(it, rt.parse_arm("mistral-def-s0"))
    assert ms == [{"role": "user", "content": "Summarize.\n\nSome data."}]
    assert su.chat_text(mtok, ms) == "<s>[INST] Summarize.\n\nSome data.[/INST]"
    from jinja2.exceptions import TemplateError

    with pytest.raises(TemplateError):  # Mistral's template has no input role
        mtok.apply_chat_template(build_messages("I", "D", False), tokenize=False)
    for name in ("sealion-def-s0", "mistral-cfgA-s0"):  # IFEval: no data channel, one user turn
        assert build_messages("P", None, arms.ADAPTERS[name][2]) == [{"role": "user",
                                                                       "content": "P"}]


@needs_models
def test_backbone_tokenizers_are_the_training_tokenizers() -> None:
    from ipidefense.training.defense import base_and_tok

    for b in ("sealion", "mistral"):
        model_dir, tok = base_and_tok(b)
        assert model_dir == str(bb.MODEL_DIRS[b])
        assert tok.chat_template == bb.backbone_tokenizer(b).chat_template
    assert bb.generation_settings("qwen")["repetition_penalty"] == 1.05
    assert {b: bb.generation_settings(b)["repetition_penalty"]
            for b in ("sealion", "mistral")} == {"sealion": 1.0, "mistral": 1.0}


# loaders
def _recording_loaders(calls: list[str]):
    def make(name: str):
        def load():
            calls.append(name)
            return Dummy(), f"tok-{name}"
        return load
    return make


def test_ifeval_load_model_dispatch(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    make = _recording_loaders(calls)
    for b in ("qwen", "sealion", "mistral"):
        monkeypatch.setattr(ifl, f"load_{b}_undefended", make(b))
    for names, expect in ((["sealion-undef"], "sealion"), (["mistral-undef"], "mistral"),
                          (["undef"], "qwen")):
        calls.clear()
        _, tok = ifl.load_model(names)
        assert calls == [expect] and tok == f"tok-{expect}"
    with pytest.raises(ValueError):
        ifl.load_model(["undef", "sealion-undef"])
    for qwen_inline in (["inline-s0"], ["undef-inline"]):  # Qwen: role-format arms only
        with pytest.raises(AssertionError):
            ifl.load_model(qwen_inline)
    wrong = _fake_adapter(tmp_path, "Mistral-7B-Instruct-v0.3")
    monkeypatch.setitem(arms.ADAPTERS, "sealion-cfgA-s0", ("sealion", str(wrong), False))
    calls.clear()
    with pytest.raises(ValueError):  # refused before any model is loaded
        ifl.load_model(["sealion-undef", "sealion-cfgA-s0"])
    assert calls == []


def test_eval_load_dispatch(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    make = _recording_loaders(calls)
    monkeypatch.setattr(rt, "load_qwen_undefended", make("qwen"))
    monkeypatch.setattr(rt, "load_base", lambda b: make(b)())
    assert rt.load_eval_model([rt.parse_arm("sealion-undef")])[1] == "tok-sealion"
    assert rt.load_eval_model([rt.parse_arm("undef")])[1] == "tok-qwen"
    assert rt.load_eval_model([])[1] == "tok-qwen"  # an empty plan loads Qwen
    assert calls == ["sealion", "qwen", "qwen"]
    with pytest.raises(ValueError):
        rt.load_eval_model([rt.parse_arm("undef"), rt.parse_arm("mistral-undef")])
    wrong = _fake_adapter(tmp_path, "Llama-SEA-LION-v3-8B-IT")
    monkeypatch.setitem(arms.ADAPTERS, "mistral-cfgB-s0", ("mistral", str(wrong), True))
    with pytest.raises(ValueError):
        rt.load_eval_model([rt.parse_arm("mistral-undef"), rt.parse_arm("mistral-cfgB-s0")])
    assert calls == ["sealion", "qwen", "qwen"]


@needs_models
def test_lora_module_counts(stok, monkeypatch) -> None:
    assert ifl.N_LORA_MODULES == 196 and ifl.LORA_MODULES["qwen"] == 196
    for b, d in (("qwen", QWEN_DIR), ("sealion", SEALION_DIR), ("mistral", MISTRAL_DIR)):
        layers = json.loads((d / "config.json").read_text(encoding="utf-8"))["num_hidden_layers"]
        assert ifl.LORA_MODULES[b] == 7 * layers
    model = _with_lora(_tiny_llama(len(stok)), "sealion-cfgA-s0")
    with torch.no_grad():  # loaded adapters are frozen, this tiny one is trainable
        with pytest.raises(AssertionError):  # 2 tiny layers x 7 != 224
            ifl.check_adapters(model, ["sealion-undef", "sealion-cfgA-s0"])
        monkeypatch.setitem(ifl.LORA_MODULES, "sealion", 14)
        with pytest.raises(AssertionError):  # LoRA B is zero at init, so it does nothing
            ifl.check_adapters(model, ["sealion-cfgA-s0"])
        for m in model.modules():
            if hasattr(m, "lora_B") and "sealion-cfgA-s0" in m.lora_B:
                torch.nn.init.normal_(m.lora_B["sealion-cfgA-s0"].weight)
        ifl.check_adapters(model, ["sealion-undef", "sealion-cfgA-s0"])


# generation
@needs_alpaca
@pytest.mark.parametrize("b", ["sealion", "mistral"])
def test_generation_parity_per_backbone(b: str, backbones) -> None:
    model, tok = backbones[b]
    prompts = ["Write a haiku about rain.", "List three colors.", "Say hi."]
    got = ifl.gen_logged_batch(model, tok, prompts, 5)
    assert [g["response"] for g in got] == generate_batch(model, tok, [(p, None) for p in prompts],
                                                          5)
    assert ifl.gen_logged_batch(model, tok, prompts[:1], 5) == [ifl.gen_logged(model, tok,
                                                                               prompts[0], 5)]
    hs = train_hosts(0)[:3]
    items = [Item(f"h{i:04d}", h.instruction, h.data, {"host": i}) for i, h in enumerate(hs)]
    texts = bb.chat_texts(tok, items, INLINE[b])
    recs = bb.generate_targets(model, tok, [texts[it.key] for it in items], 6,
                               eos_ids(model, tok))
    assert [r["text"] for r in recs] == generate_batch(
        model, tok, [(h.instruction, h.data) for h in hs], max_new_tokens=6, inline=INLINE[b])
    for r in recs:
        assert len(r["ids"]) == r["n_tokens"] <= 6
        assert tok.decode(r["ids"], skip_special_tokens=True) == r["text"]


@needs_alpaca
@pytest.mark.parametrize("b", ["sealion", "mistral"])
def test_cmd_targets_end_to_end(b: str, backbones, tmp_path: Path, monkeypatch) -> None:
    tok = backbones[b][1]
    fresh = (_tiny_llama if b == "sealion" else _tiny_mistral)(len(tok))
    hs = train_hosts(0)[:3]
    expected = generate_batch(fresh, tok, [(h.instruction, h.data) for h in hs],
                              max_new_tokens=8, inline=INLINE[b])  # the train.py --batch path
    frozen = _with_lora(fresh)  # training LoRA wrapper, targets come with it disabled
    monkeypatch.setattr(bb, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(bb, "OUT", tmp_path / "runs")
    monkeypatch.setattr(bb, "backbone_tokenizer", lambda x: tok)
    monkeypatch.setattr(bb, "load_frozen", lambda x: frozen)
    monkeypatch.setattr(bb, "env_info", lambda: {"cpu_test": True})
    monkeypatch.setattr(bb, "generation_settings", lambda x: {})
    rep = bb.cmd_targets(b, 8, limit=3)
    path = bb.targets_path(b, 8, smoke=True)
    assert rep["smoke"] and rep["complete"] and rep["batch"] == 16 and path.parent.name == "smoke"
    recs = sorted(Transcript(path).done.values(), key=lambda r: r["host"])
    assert [r["text"] for r in recs] == expected
    assert all((r["mnt"], r["batch"]) == (8, 16) for r in recs)
    assert (tmp_path / "runs" / "smoke" / f"targets_{b}_cap8.json").exists()
    monkeypatch.setattr(bb, "load_frozen", lambda x: pytest.fail("model loaded with nothing to do"))
    assert bb.cmd_targets(b, 8, limit=3)["generated_now"] == 0


def test_run_targets_fixed_chunks_resume_and_guard(tmp_path: Path) -> None:
    items = [Item(f"h{i:04d}", f"instr {i}", f"data {i}", {"host": i}) for i in range(10)]
    texts = {it.key: f"T{it.key}" for it in items}
    batches: list[list[str]] = []

    def make_gen():
        def gen(chunk_texts):
            batches.append(list(chunk_texts))
            return [{"text": t.lower(), "n_tokens": 1, "ended_eos": True, "cap_hit": False,
                     "ids": [7]} for t in chunk_texts]
        return gen

    path = tmp_path / "t.jsonl"
    assert bb.run_targets(path, items, texts, make_gen, 4, 160)["complete"]
    assert batches == [[texts[f"h{i:04d}"] for i in range(k, min(k + 4, 10))] for k in (0, 4, 8)]
    lines = path.read_text(encoding="utf-8").split("\n")
    kept = [x for x in lines if x and json.loads(x)["key"] not in ("h0005", "h0006")]
    path.write_text("\n".join(kept) + '\n{"key": "h0005", "te', encoding="utf-8")  # crash
    batches.clear()
    rep = bb.run_targets(path, items, texts, make_gen, 4, 160)
    assert rep["complete"] and rep["generated_now"] == 4
    assert batches == [[texts[f"h{i:04d}"] for i in range(4, 8)]]  # the whole chunk, same mates
    assert sorted(Transcript(path).done) == [it.key for it in items]
    assert len([x for x in path.read_text(encoding="utf-8").split("\n") if x]) == 10
    with pytest.raises(RuntimeError):  # never mix settings in one transcript
        bb.run_targets(path, items, texts, make_gen, 4, 512)


# census
def test_loop_period() -> None:
    lp = rules.loop_period
    assert lp("The answer is 42. " * 4, False) == 4
    assert lp("very " * 20, False) == 1
    assert lp("Thank you! Thank you! Thank you!", False) == 0  # 6 words, too short
    assert lp("A normal answer that ends without any repetition at all, as most do.", False) == 0
    assert lp(" ".join(f"{i}. item" for i in range(1, 30)), False) == 0  # numbered, not a repeat
    cut = "go to the store and buy milk " * 4 + "mi"  # cut mid-word by the token cap
    assert lp(cut, True) == 7 and lp(cut, False) == 0
    para = " ".join(f"w{i}" for i in range(120))  # paragraph-length unit
    longer = " ".join(f"w{i}" for i in range(121))
    assert lp(" ".join([para] * 3), False) == 120 and lp(" ".join([longer] * 3), False) == 0
    assert lp(" ".join([para] * 2), False) == 0  # two repeats are not a loop


def _rec(key: str, text: str, n: int, eos: bool, cap: int) -> dict:
    return {"key": key, "text": text, "n_tokens": n, "ended_eos": eos,
            "cap_hit": (not eos) and n >= cap}


def test_target_stats_and_prefix_agreement() -> None:
    recs = [_rec("h0", "Short answer.", 3, True, 160),
            _rec("h1", "word " * 200, 160, False, 160),          # capped loop
            _rec("h2", "x " * 100, 159, True, 160),              # EOS as the 160th token
            _rec("h3", "", 0, True, 160)]
    st = rules.target_stats(recs, 160)
    assert (st["trunc_at_160"]["k"], st["eos_before_cap"]["k"], st["cap_hit"]["k"]) == (1, 3, 1)
    assert (st["loop"]["k"], st["loop_and_cap_hit"]["k"], st["n_empty"]) == (2, 1, 1)
    assert st["loop_periods"] == {1: 2} and st["loop_keys"] == ["h1", "h2"]
    assert "trunc_at_160_eos_before_cap" not in st and st["tokens"]["max"] == 160
    long = [_rec("h0", "a b c.", 300, True, 512), _rec("h1", "y " * 600, 512, False, 512),
            _rec("h2", "ok", 20, True, 512)]
    st5 = rules.target_stats(long, 512)
    assert (st5["trunc_at_160"]["k"], st5["trunc_at_160_eos_before_cap"]["k"]) == (2, 1)
    assert st5["eos_before_cap"]["k"] == 2 and rules.target_stats([], 160) == {"n": 0, "cap": 160}
    r160 = {0: {"ids": [1, 2], "cap_hit": False, "ended_eos": True},
            1: {"ids": list(range(160)), "cap_hit": True, "ended_eos": False},
            2: {"ids": [5], "cap_hit": False, "ended_eos": True}}
    r512 = {0: {"ids": [1, 2], "cap_hit": False, "ended_eos": True},
            1: {"ids": list(range(300)), "cap_hit": False, "ended_eos": True},
            2: {"ids": [5, 6], "cap_hit": False, "ended_eos": True}}
    pa = rules.prefix_agreement(r160, r512)
    assert (pa["n"], pa["identical"], pa["differ_hosts"]) == (3, 2, [2])


def _write_targets(path: Path, n: int, cap: int, batch: int = 16, ids: bool = True) -> None:
    tr = Transcript(path)
    tr.add([{"key": f"h{i:04d}", "host": i, "text": f"Answer {i}.", "n_tokens": 3,
             "ended_eos": True, "cap_hit": False, "mnt": cap, "batch": batch,
             **({"ids": [1, 2, 3]} if ids else {})} for i in range(n)])


def test_census_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bb, "N_TRAIN", 4)
    monkeypatch.setattr(bb, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(bb, "generation_settings", lambda b: {"repetition_penalty": b})
    qwen = tmp_path / "qwen_targets.jsonl"
    _write_targets(qwen, 4, 160, ids=False)
    for cap in (160, 512):
        _write_targets(bb.targets_path("sealion", cap), 4, cap)
    _write_targets(bb.targets_path("mistral", 160), 3, 160)  # incomplete, never summarized
    res = bb.census(qwen_path=qwen, out=tmp_path / "census.json")
    assert set(res["backbones"]) == {"qwen", "sealion"}
    assert res["backbones"]["sealion"]["prefix_agreement"]["identical"] == 4
    assert res["backbones"]["qwen"]["cap160"]["batch"] == 1
    assert str(bb.targets_path("mistral", 160)) in res["missing"]
    assert str(bb.targets_path("mistral", 512)) in res["missing"]
    assert set(res["generation"]) == {"qwen", "sealion", "mistral"}
    assert json.loads((tmp_path / "census.json").read_text(encoding="utf-8"))["definitions"]


# build and train
class FakeTr:
    def translate(self, text: str, lang: str) -> str:
        text = text.strip()
        return f"<{lang}> {text}" if text else text


def _training_examples(targets: list[str], inline: bool, tr) -> list[tuple]:
    """Defense training mixture for seed 0, clean_frac 0.6, langs en/es/vi."""
    rng, langs, out = random.Random(0), ["en", "es", "vi"], []
    for i, (h, target) in enumerate(zip(train_hosts(0), targets, strict=True)):
        if not target.strip():
            continue
        if rng.random() < 0.6:
            data = h.data
        else:
            templ, w = rng.choice(TRAIN_TEMPLATES), rng.choice(TRAIN_WITNESSES)
            inj = render_inj_literal(tr, WitnessInjection(templ, w), langs[i % 3])
            data = f"{h.data}\n{inj}"
        out.append((build_messages(h.instruction, data, inline), target))
    return out


@needs_alpaca
@pytest.mark.parametrize("b", ["sealion", "mistral"])
def test_build_records_match_training_mixture(b: str) -> None:
    texts = [" " if i in (5, 900) else f"Frozen answer {i}." for i in range(1500)]
    targets = {i: {"text": t, "n_tokens": 4, "cap_hit": False, "ended_eos": True}
               for i, t in enumerate(texts)}
    recs = bb.build_records(b, "A", targets, FakeTr())
    assert [(r["messages"], r["target"]) for r in recs] == _training_examples(texts, INLINE[b],
                                                                               FakeTr())
    assert len(recs) == 1498 and all(r["key"].startswith(f"s5-{b}-cfgA-h") for r in recs)
    assert all(len(r["messages"]) == (1 if INLINE[b] else 2) for r in recs)
    with pytest.raises(ValueError):  # a full build needs every host
        bb.build_records(b, "A", {i: targets[i] for i in range(10)}, FakeTr())
    assert len(bb.build_records(b, "A", {i: targets[i] for i in range(10)}, FakeTr(),
                                partial=True)) == 9


def _train_args(**kw) -> argparse.Namespace:
    base = {"backbone": "mistral", "cfg": "B", "smoke": False, "limit": None, "longest": False,
            "run_name": None, "overwrite": False, "dry_run": True}
    return argparse.Namespace(**{**base, **kw})


@needs_alpaca
@pytest.mark.skipif(not RENDER_CACHE.exists() or not RENDER_AUDIT.exists(),
                    reason="needs data/baselines/render_cache.json and runs/render_audit.json")
def test_train_dry_run_and_guards(mtok, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bb, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(bb, "OUT", tmp_path / "runs")
    path = bb.targets_path("mistral", 160)
    _write_targets(path, 1500, 160)
    with pytest.raises(SystemExit):  # no targets report with this file's SHA-256 yet
        bb.cmd_train(_train_args())
    write_json(bb.report_path("targets_mistral_cap160.json"), {"sha256": sha256_file(path)})
    res = bb.cmd_train(_train_args())
    cfg, data = res["config"], res["data"]
    assert (cfg["cap"], cfg["lr"], cfg["inline"], cfg["seed"]) == (160, 5e-5, True, 0)
    assert cfg["base"] == str(MISTRAL_DIR) and cfg["run_name"] == "mistral-cfgB-seed0"
    assert (cfg["epochs"], cfg["accum"], cfg["max_len"]) == (1, 8, 1024)
    assert data["n_records"] == 1500 and data["kinds"] == {"clean": 902, "witness": 598}
    assert data["render_cache"]["audit"]["n_mismatch"] == 0
    built = [json.loads(x) for x in Path(data["file"]).read_text(encoding="utf-8").split("\n") if x]
    assert len(built) == 1500 and all(len(r["messages"]) == 1 for r in built)
    assert arms.backbone_adapter_dir("mistral", "B").name == cfg["run_name"]
    with pytest.raises(SystemExit):  # a full run writes only the registry adapter
        bb.cmd_train(_train_args(run_name="other-name"))
    with pytest.raises(SystemExit):
        bb.cmd_train(_train_args(limit=4, run_name="not-smoke"))
    _write_targets(bb.targets_path("mistral", 512), 1500, 512, batch=1)
    with pytest.raises(SystemExit):  # targets not from the batch-16 protocol
        bb.cmd_train(_train_args(cfg="A"))


# security
def test_sec_plan_and_cmd_sec(tmp_path: Path, monkeypatch) -> None:
    untrained = ("sealion", str(tmp_path / "untrained"), False)
    monkeypatch.setitem(arms.ADAPTERS, "sealion-cfgA-s0", untrained)
    with pytest.raises(SystemExit):  # adapters must exist before any GPU load
        bb.require_adapters(["sealion-undef", "sealion-cfgA-s0"])
    trained = _fake_adapter(tmp_path, "Llama-SEA-LION-v3-8B-IT")
    (trained / "adapter_model.safetensors").write_bytes(b"")
    monkeypatch.setitem(arms.ADAPTERS, "sealion-cfgA-s0", ("sealion", str(trained), False))
    bb.require_adapters(["sealion-undef", "sealion-cfgA-s0"])
    monkeypatch.undo()
    plan_arms, langs = bb.sec_plan("sealion", False)
    assert plan_arms == ["sealion-undef", "sealion-def-s0", "sealion-cfgA-s0", "sealion-cfgB-s0"]
    assert langs == ("en", "es", "vi")
    assert bb.sec_plan("mistral", True, best="B") == (["mistral-undef", "mistral-cfgB-s0"],
                                                      ("en", "es", "vi", "fr", "de", "zh"))
    with pytest.raises(SystemExit):
        bb.sec_plan("mistral", True)
    with pytest.raises(SystemExit):
        bb.sec_plan("sealion", False, ["mistral-undef"])
    ns = bb.suite_args("sealion", plan_arms, langs)
    assert (ns.exp, ns.renderer, ns.seeds, ns.batch, ns.mnt, ns.limit_hosts) == (
        "ml1b", "v2", [0], 32, 64, 0)
    assert Path(ns.out_dir) == bb.suite_dir("sealion")
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(su, "run_exp", seen.append)
    monkeypatch.setattr(bb, "require_adapters", lambda a: None)
    monkeypatch.setattr(bb, "OUT", tmp_path)
    bb.cmd_sec("sealion", False)
    assert seen[-1].arms == plan_arms and seen[-1].langs == ["en", "es", "vi"]
    rows = {"A": _row((83, 100), (70, 100)), "B": _row((83, 100), (78, 100))}
    monkeypatch.setattr(bb, "backbone_results", lambda b: {"rows": rows, "missing": []})
    bb.cmd_sec("sealion", True)
    assert seen[-1].arms == ["sealion-undef", "sealion-cfgB-s0"] and len(seen[-1].langs) == 6
    best = json.loads((tmp_path / "best_sealion.json").read_text(encoding="utf-8"))
    assert best["best_config"] == "B" and best["arm"] == "sealion-cfgB-s0"
    monkeypatch.setattr(bb, "backbone_results", lambda b: {"rows": None, "missing": ["x"]})
    with pytest.raises(SystemExit):
        bb.cmd_sec("sealion", True)


@needs_alpaca
@pytest.mark.parametrize("b", ["sealion", "mistral"])
def test_suite_runs_backbone_arm_end_to_end(b: str, backbones, tmp_path: Path,
                                            monkeypatch) -> None:
    model, tok = backbones[b]
    monkeypatch.setattr(bb, "OUT", tmp_path)
    monkeypatch.setattr(su, "RENDER_DIR", tmp_path / "renderings")
    monkeypatch.setattr(su, "load_eval_model", lambda names: (model, tok))
    su.run_exp(bb.suite_args(b, [f"{b}-undef"], ["en"], limit_hosts=2))
    smoke = tmp_path / f"suite_{b}" / "smoke"
    path = su.transcript_path(smoke, rt.parse_arm(f"{b}-undef"), 0, "ml1b", "v2")
    recs = list(su.read_records(path).values())
    assert len(recs) == 6 and all((r["mnt"], r["batch"]) == (64, 32) for r in recs)
    h = hosts(0, "security")[0]
    r = next(x for x in recs if x["key"] == "en|ovr-hijacked|0")
    data = f"{h.data}\n{su.load_renderings('en', 'v2')['ovr-hijacked']['rendered']}"
    msgs = build_messages(h.instruction, data, INLINE[b])  # the backbone's own format
    assert r["prompt_sha"] == sha(su.chat_text(tok, msgs))
    monkeypatch.setattr(bb, "N_SEC_HOSTS", 2)
    cell = bb.security_cell(b, f"{b}-undef", ["en"], out=smoke)
    assert cell["en"]["n"] == 6 and cell["en"]["wf_k"] >= cell["en"]["rf_k"]
    assert bb.security_cell(b, f"{b}-undef", ["en", "es"], out=smoke) is None  # incomplete


# rules
def _sec(obey: dict[str, tuple[int, int]]) -> dict:
    return {lang: {"wf_k": k, "rf_k": 0, "n": n} for lang, (k, n) in obey.items()}


SECURE = _sec({lang: (0, 450) for lang in rules.SEC_LANGS})


def _row(u: tuple[int, int], d: tuple[int, int], sec: dict = SECURE) -> dict:
    return rules.config_row({"k": u[0], "n": u[1]}, {"k": d[0], "n": d[1]}, sec)


def test_rules_exact_boundaries() -> None:
    at_001 = _sec({lang: (1, 100) for lang in rules.SEC_LANGS})  # obey exactly 0.01
    rows = {"A": _row((83, 100), (75, 100), at_001), "B": _row((83, 100), (60, 100))}
    assert rows["A"]["tax_pp_exact"] == "8" and rows["A"]["tax_le_8pp"]  # exactly 8 pp
    assert rows["A"]["security_ok"] and rules.classify(rows) == rules.CONFIG_DRIVEN
    over = _sec({"en": (4, 450), "es": (5, 450), "vi": (0, 450)})  # es 0.0111 > 0.01
    rows = {"A": _row((83, 100), (75, 100), over), "B": _row((83, 100), (71, 100))}
    assert not rows["A"]["security_ok"] and rows["B"]["tax_pp_exact"] == "12"
    assert rules.classify(rows) == rules.PARTLY  # the 8-pp config fails security, A < 12
    # float traps avoided by exact arithmetic
    assert (29 / 100 - 17 / 100) * 100 < 12 and (14 / 100 - 6 / 100) * 100 > 8
    rows = {"A": _row((29, 100), (17, 100)), "B": _row((82, 100), (70, 100))}
    assert rows["A"]["tax_pp_exact"] == "12" and rows["A"]["tax_ge_12pp"]
    assert rules.classify(rows) == rules.OBJECTIVE_DRIVEN  # exactly 12 pp in every config
    rows = {"A": _row((14, 100), (6, 100), at_001), "B": _row((14, 100), (0, 100))}
    assert rows["A"]["tax_pp_exact"] == "8" and rules.classify(rows) == rules.CONFIG_DRIVEN
    u = (8300, 10000)
    rows = {"A": _row(u, (7499, 10000)), "B": _row(u, (7101, 10000))}  # 8.01 and 11.99 pp
    assert not rows["A"]["tax_le_8pp"] and not rows["B"]["tax_ge_12pp"]
    assert rules.classify(rows) == rules.PARTLY
    rows = {"A": _row((83, 100), (71, 100)), "B": _row((83, 100), (75, 100))}  # 12 and 8 pp
    assert rules.classify(rows) == rules.CONFIG_DRIVEN
    rows = {"A": _row((689, 834), (622, 834)), "B": _row((689, 834), (623, 834))}  # 834 instr.
    assert (rows["A"]["tax_le_8pp"], rows["B"]["tax_le_8pp"]) == (False, True)
    missing_lang = _row((83, 100), (80, 100), _sec({"en": (0, 450), "es": (0, 450)}))
    assert not missing_lang["security_ok"]


def test_tax_ci95_and_borderline() -> None:
    flat = rules.tax_ci95([(2, 2)] * 50, [(1, 2)] * 50, n_boot=500)
    assert flat == [50.0, 50.0]  # every prompt loses half its instructions, no spread
    undef = [(1, 1)] * 100
    wide = rules.tax_ci95(undef, [(0, 1)] * 10 + [(1, 1)] * 90)  # 10 pp on 100 prompts
    assert wide[0] < 8 and wide[1] > 12 and rules.borderline(wide)
    tight = rules.tax_ci95([(1, 1)] * 2000, [(0, 1)] * 400 + [(1, 1)] * 1600)  # 20 pp
    assert 12 < tight[0] < 20 < tight[1] and not rules.borderline(tight)
    assert rules.borderline([7.5, 8.0]) and rules.borderline([12.0, 12.4])  # inclusive ends
    assert not rules.borderline([8.01, 11.99])
    with pytest.raises(ValueError):  # prompts must pair up
        rules.tax_ci95([(1, 2)], [(1, 1)])


def test_gcg_harness_refuses_backbone_arms() -> None:
    from ipidefense.experiments import gcg_run

    for name in ("sealion-undef", "mistral-undef", "sealion-def-s0", "mistral-cfgA-s0"):
        with pytest.raises(SystemExit, match="Qwen only"):  # before any model load
            gcg_run.load_arm(name)


def test_best_config_rule() -> None:
    insecure = _sec({"en": (9, 450), "es": (0, 450), "vi": (0, 450)})
    worse = _sec({"en": (12, 450), "es": (0, 450), "vi": (0, 450)})
    assert rules.best_config({"A": _row((90, 100), (80, 100)),
                              "B": _row((90, 100), (81, 100))}) == "B"  # lower tax
    assert rules.best_config({"A": _row((90, 100), (85, 100), insecure),
                              "B": _row((90, 100), (77, 100))}) == "B"  # only B secure
    assert rules.best_config({"A": _row((90, 100), (85, 100), worse),
                              "B": _row((90, 100), (70, 100), insecure)}) == "B"  # lower obey
    assert rules.best_config({"A": _row((90, 100), (80, 100)),
                              "B": _row((90, 100), (80, 100))}) == "A"  # full tie


def _write_ifeval(directory: Path, name: str, flags: list[list[bool]], batch: int = 16) -> None:
    Transcript(directory / f"{name}.jsonl").add([
        {"key": f"{name}:{i}", "arm": name, "idx": i, "flags": f, "mnt": 512, "batch": batch,
         "n_tokens": 10, "stopped_eos": True, "response": "x"} for i, f in enumerate(flags)])


def _write_security(suite: Path, name: str, langs, obeyed: int = 0) -> None:
    path = su.transcript_path(suite, rt.parse_arm(name), 0, "ml1b", "v2")
    Transcript(path).add([
        {"key": f"{lang}|t1|{i}", "lang": lang, "wf": "obeyed" if i < obeyed else "ignored",
         "rf": "ignored", "mnt": 64, "batch": 32} for lang in langs for i in range(2)])


def _write_utility(runs: Path, b: str, undef: float, dfd: float) -> None:
    write_json(runs / f"eval_utility_{b}_undefended.json", {"instruction_strict_acc": undef})
    write_json(runs / f"eval_utility_{b}_lora_{arms.BACKBONES[b][0]}.json",
               {"instruction_strict_acc": dfd})


def test_decide_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bb, "N_PROMPTS", 4)
    monkeypatch.setattr(bb, "N_INSTRUCTIONS", 8)
    monkeypatch.setattr(bb, "N_SEC_HOSTS", 2)
    monkeypatch.setattr(bb, "valid_tids", lambda lang: ["t1"])
    monkeypatch.setattr(bb, "PUBLISHED_RUNS", tmp_path / "published")
    for b in arms.BACKBONES:
        _write_utility(tmp_path / "published", b, 0.8, 0.7)
    ife, suite, runs = tmp_path / "ifeval", tmp_path / "suite", tmp_path / "runs"
    ok, one_off = [[True, True]] * 4, [[True, False]] + [[True, True]] * 3
    _write_ifeval(ife, "sealion-undef", ok)
    _write_ifeval(ife, "sealion-def-s0", [[False, False]] * 2 + [[True, True]] * 2)
    _write_ifeval(ife, "sealion-cfgA-s0", one_off)  # tax 12.5 pp
    _write_ifeval(ife, "sealion-cfgB-s0", ok)       # tax 0 pp
    for kind in ("undef", "def", "A", "B"):
        _write_security(suite, rules.arm("sealion", kind), rules.SEC_LANGS,
                        obeyed=1 if kind == "undef" else 0)
    _write_security(suite, "sealion-cfgB-s0", ("fr", "de", "zh"))
    _write_ifeval(ife, "mistral-undef", ok)
    res = bb.decide(ifeval_dir=ife, runs=runs, suites={"sealion": suite})
    sl, mi = res["backbones"]["sealion"], res["backbones"]["mistral"]
    assert sl["decision"] == rules.CONFIG_DRIVEN and sl["best_config"] == "B"
    assert sl["configs"]["A"]["tax_pp"] == 12.5 and sl["configs"]["B"]["security_ok"]
    ci = sl["configs"]["A"]["tax_ci95_descriptive"]  # descriptive only
    assert ci[0] <= 12.5 <= ci[1] and isinstance(sl["configs"]["A"]["borderline"], bool)
    assert sl["configs"]["B"]["tax_ci95_descriptive"] == [0.0, 0.0]
    assert sl["published_row"]["tax_pp"] == 50.0 and "tax_ci95_descriptive" in sl["published_row"]
    assert sl["label_if_published_counted"] == rules.CONFIG_DRIVEN
    assert sl["published_reference"]["unbatched_published"]["tax_pp"] == 10.0
    assert sl["published_reference"]["batched"]["tax_pp"] == 50.0
    six = sl["six_language_check"]
    assert six["best_config"] == "B" and not six["complete"]  # undef lacks fr/de/zh
    assert six["sealion-cfgB-s0"]["zh"]["wf"]["n"] == 2
    assert mi["decision"] == "pending" and "ifeval:mistral-cfgA-s0" in mi["missing"]
    assert json.loads((runs / "decision.json").read_text(encoding="utf-8"))["rules"] == rules.RULES
    _write_ifeval(tmp_path / "bad", "sealion-undef", ok, batch=1)
    with pytest.raises(SystemExit):  # unbatched transcripts never enter the batched rule
        bb.ifeval_counts("sealion-undef", tmp_path / "bad")


# Qwen IFEval path
def test_qwen_ifeval_constants_and_single_load(monkeypatch) -> None:
    assert {k: ifl.PUBLISHED[k] for k in ("undef", "def-s0", "def-s1", "def-s2")} == {
        "undef": (633, 366), "def-s0": (590, 331), "def-s1": (570, 313), "def-s2": (592, 327)}
    assert (ifl.GATE_TOL, ifl.RANDOM_LETTER_IDX, ifl.OUT) == (2, (19, 22),
                                                               paths.RUNS / "ifeval_logged")
    assert ifl.default_out(1) == ifl.OUT
    assert ifl.default_out(16) == paths.RUNS / "ifeval_logged_b16" == bb.IFEVAL_DIR
    calls: list[str] = []
    make = _recording_loaders(calls)
    for b in ("qwen", "sealion", "mistral"):
        monkeypatch.setattr(ifl, f"load_{b}_undefended", make(b))
    ifl.load_model(["undef"])
    assert calls == ["qwen"]  # one Qwen load, no adapter for the undefended arm
