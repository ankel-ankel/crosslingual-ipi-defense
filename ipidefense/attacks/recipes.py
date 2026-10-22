"""Injection recipes and per-seed training plans of the comparison arms (CPU only).

a0 ours, a1 StruQ, b1 SecAlign++ (Meta-SecAlign), b2 DPO on our witness injections, b3 SecAlign;
a2 reuses a0's plan. Injected tasks come only from the same seed's 1,500 training hosts.
"""
from __future__ import annotations

import functools
import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ipidefense.attacks.templates import TRAIN_TEMPLATES, TRAIN_WITNESSES
from ipidefense.attacks.witnesses import WitnessInjection
from ipidefense.data.splits import SLICES
from ipidefense.paths import DATA, RUNS
from ipidefense.rendering.literal import render_inj_literal
from ipidefense.utils import Item

ALPACA_JSON = DATA / "alpaca_cleaned.json"
DATA_ROOT = DATA / "baselines"
RENDER_CACHE = DATA_ROOT / "render_cache.json"
RENDER_AUDIT = RUNS / "render_audit.json"

N_TRAIN = SLICES["train"][1]  # 1500 training hosts per seed
N_LOADED = SLICES["security"][1]  # the defense loads n_train + n_eval = 1650 hosts
CLEAN_FRAC = 0.6  # defense --clean-frac
LANGS = ("en", "es", "vi")  # defense --langs

# "Other" delimiters of SecAlign config.py:11-17 (identical in Meta_SecAlign config.py:12-19)
OTHER_DELM: dict[str, tuple[str, ...]] = {
    "mark": (
        "{s}", "|{s}|", "<{s}>", "[{s}]", "<|{s}|>", "[|{s}|]", "<[{s}]>", "'''{s}'''", "***{s}***",
    ),
    "inst": ("Command", "Rule", "Prompt", "Task"),
    "inpt": ("Data", "Context", "Text"),
    "resp": ("Output", "Answer", "Reply"),
    "user": ("", "Prompter ", "User ", "Human "),
    "asst": ("", "Assistant ", "Chatbot ", "Bot ", "GPT ", "AI "),
}
N_TEST_DELM = 2  # StruQ/SecAlign training never uses the last two options (held out for testing)
STRUQ_SLOTS = ("naive", "clean", "completion", "clean")  # struq.py:139-147, one copy per host
# StruQ's test-time filter (test.py:81-89) with the special-token strings of our template
STRUQ_FILTERED = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "##")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- hosts
@dataclass(frozen=True)
class RefHost:
    """A host task with its Alpaca reference response and source row."""

    instruction: str
    data: str
    output: str
    row: int


@functools.cache
def _alpaca_rows(path: str) -> tuple[dict, ...]:
    return tuple(json.loads(Path(path).read_text(encoding="utf-8")))


def load_ref_hosts(n: int, seed: int, path: Path = ALPACA_JSON) -> tuple[RefHost, ...]:
    """load_en_hosts(n, seed) plus the reference output: same rows, filter and shuffle."""
    usable = [
        (k, r)
        for k, r in enumerate(_alpaca_rows(str(path)))
        if (r.get("input") or "").strip() and (r.get("instruction") or "").strip()
    ]
    random.Random(seed).shuffle(usable)
    return tuple(
        RefHost(r["instruction"].strip(), r["input"].strip(), r.get("output") or "", k)
        for k, r in usable[:n]
    )


def train_hosts(seed: int) -> tuple[RefHost, ...]:
    """The defense's training hosts: load_en_hosts(1650, seed)[:1500]."""
    return load_ref_hosts(N_LOADED, seed)[:N_TRAIN]


# ----------------------------------------------------------------------------- rendering
class CachedTranslator:
    """Read-only stand-in for NLLBTranslator, built from the render cache."""

    def __init__(self, table: dict[str, dict[str, str]]) -> None:
        self._table = {lang: dict(m) for lang, m in table.items()}

    @classmethod
    def load(cls, path: Path = RENDER_CACHE) -> CachedTranslator:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: it ships with the repository "
                                    "(rebuilt by the render command of training/sets.py)")
        return cls(json.loads(path.read_text(encoding="utf-8"))["table"])

    def translate(self, text: str, target_lang: str) -> str:
        text = text.strip()
        if not text:
            return text
        try:
            return self._table[target_lang][text]
        except KeyError:
            raise KeyError(f"no cached {target_lang} translation for {text!r}") from None


def wrapper_segments(templates: Iterable[str]) -> list[str]:
    """Every non-blank wrapper segment that render_inj_literal sends to NLLB."""
    segs: dict[str, None] = {}
    for template in templates:
        inj = WitnessInjection(template, "W")
        for seg in (inj.wrapper_prefix(), inj.wrapper_suffix()):
            if seg.strip():
                segs.setdefault(seg.strip(), None)
    return list(segs)


def audit_render(tr: Any, audit: Path = RENDER_AUDIT, langs: Iterable[str] = ("es", "vi")) -> dict:
    """Compare the cached renderings with the original-renderer strings of the render audit."""
    if not audit.exists():
        return {"available": False}
    langs = tuple(langs)
    rows = [
        r
        for r in json.loads(audit.read_text(encoding="utf-8"))["rows"]
        if r["pool"] in ("train", "eval") and r["lang"] in langs
    ]
    bad = [
        r
        for r in rows
        if render_inj_literal(tr, WitnessInjection(r["template"], r["witness"]), r["lang"])
        != r["old"]
    ]
    return {
        "available": True,
        "n_rows": len(rows),
        "n_mismatch": len(bad),
        "mismatches": [{k: r[k] for k in ("pool", "template", "lang", "old")} for r in bad[:10]],
    }


# ----------------------------------------------------------------------------- recipes
def other_delimiters(rng: random.Random, holdout: bool) -> dict[str, str]:
    """One random delimiter per role sharing one mark. holdout=True never uses the last
    N_TEST_DELM options (StruQ/SecAlign training); holdout=False samples all (Meta-SecAlign)."""

    def pool(name: str) -> tuple[str, ...]:
        return OTHER_DELM[name][:-N_TEST_DELM] if holdout else OTHER_DELM[name]

    mark = rng.choice(pool("mark")) + ":"

    def word(name: str) -> str:
        text = rng.choice(pool("user" if name in ("inst", "inpt") else "asst"))
        text += rng.choice(pool(name))
        p = rng.random()
        return text.upper() if p < 1 / 3 else text.lower() if p < 2 / 3 else text

    return {name: mark.format(s=word(name)) for name in ("inst", "inpt", "resp")}


def struq_injected_prompt(inj: RefHost) -> str:
    """StruQ/SecAlign injected prompt (SecAlign struq.py:57-60, align.py:43)."""
    if inj.instruction.endswith("?"):
        return f"answer the following question. {inj.instruction} {inj.data}"
    return f"{inj.instruction[0].lower()}{inj.instruction[1:]} {inj.data}"


def _with_period(data: str) -> str:
    return data if data.endswith(".") else data + "."


def struq_naive(data: str, prompt: str) -> str:
    """Naive attack (SecAlign struq.py:62-65)."""
    return f"{_with_period(data)} {prompt[0].upper()}{prompt[1:]}"


def struq_completion(
    data: str, fake_response: str, inj: RefHost, prompt: str, rng: random.Random
) -> str:
    """Completion-Other (SecAlign struq.py:69-74): a fake response, then the injected task with
    random training delimiters; str.capitalize() lower-cases the rest as the original does."""
    d = other_delimiters(rng, holdout=True)
    text = (
        f"{_with_period(data)}\n\n{d['resp']}\n{fake_response}\n\n"
        f"{d['inst']}\n{prompt.capitalize()}"
    )
    if inj.data:
        text += f"\n\n{d['inpt']}\n{inj.data}"
    return text


def meta_completion(fake_response: str, inj: RefHost, rng: random.Random) -> str:
    """Meta-SecAlign create_injection_for_completion (utils.py:167-182)."""
    d = other_delimiters(rng, holdout=False)
    text = f"{d['resp']}\n\n{fake_response}\n\n{d['inst']}\n\n{inj.instruction}"
    if inj.data:
        text += f"\n\n{d['inpt']}\n\n{inj.data}"
    return text


def struq_filter(data: str) -> str:
    """StruQ's recursive test-time filter (test.py:81-89) for the Qwen ChatML template."""
    while any(t in data for t in STRUQ_FILTERED):
        for t in STRUQ_FILTERED:
            data = data.replace(t, "")
    return data


def _other_host(rng: random.Random, i: int, n: int) -> int:
    """Uniform over the other n - 1 training hosts (never i, never outside the slice)."""
    j = rng.randrange(n - 1)
    return j + (j >= i)


def _witness(tr: Any, template: str, witness: str, lang: str) -> str:
    return render_inj_literal(tr, WitnessInjection(template, witness), lang)


def plan_a0(
    hosts: tuple[RefHost, ...], seed: int, tr: Any, skip: frozenset[int] = frozenset()
) -> list[dict]:
    """The defense's data mix; a skipped host (empty target) consumes no draw, as there."""
    rng = random.Random(seed)
    out: list[dict] = []
    for i, h in enumerate(hosts):
        if i in skip:
            continue
        if rng.random() < CLEAN_FRAC:
            out.append({"host": i, "kind": "clean", "data": h.data})
            continue
        templ, wit = rng.choice(TRAIN_TEMPLATES), rng.choice(TRAIN_WITNESSES)
        lang = LANGS[i % len(LANGS)]
        inj = _witness(tr, templ, wit, lang)
        out.append({"host": i, "kind": "witness", "data": f"{h.data}\n{inj}", "template": templ,
                    "witness": wit, "lang": lang, "injection": inj})
    return out


def plan_a1(hosts: tuple[RefHost, ...], seed: int) -> list[dict]:
    """StruQ: per host one of the copies [Naive, clean, Completion, clean]."""
    rng = random.Random(f"a1-struq-{seed}")
    out: list[dict] = []
    for i, h in enumerate(hosts):
        kind = STRUQ_SLOTS[rng.randrange(len(STRUQ_SLOTS))]
        if kind == "clean":
            out.append({"host": i, "kind": kind, "data": h.data})
            continue
        j = _other_host(rng, i, len(hosts))
        prompt = struq_injected_prompt(hosts[j])
        if kind == "naive":
            data = struq_naive(h.data, prompt)
        else:
            data = struq_completion(h.data, h.output, hosts[j], prompt, rng)
        out.append({"host": i, "kind": kind, "data": data, "source": j})
    return out


def plan_b1(hosts: tuple[RefHost, ...], seed: int) -> list[dict]:
    """Meta-SecAlign utils.py:77-88 with randomized_injection_position=True."""
    rng = random.Random(f"b1-secalignpp-{seed}")
    out: list[dict] = []
    for i, h in enumerate(hosts):
        j = _other_host(rng, i, len(hosts))
        inj = hosts[j]
        alone = f"{inj.instruction} {inj.data}"
        if rng.random() < 0.9:
            start = rng.random() < 0.5
            kind = "straight-start" if start else "straight-end"
            data = f"{alone} {h.data}" if start else f"{h.data} {alone}"
        else:
            kind, data = "completion", f"{h.data}\n\n{meta_completion(h.output, inj, rng)}"
        out.append({"host": i, "kind": kind, "data": data, "source": j, "alone": alone})
    return out


def plan_b2(hosts: tuple[RefHost, ...], seed: int, tr: Any) -> list[dict]:
    """Every host injected with a fresh witness draw, appended as in the defense."""
    rng = random.Random(f"b2-witness-{seed}")
    out: list[dict] = []
    for i, h in enumerate(hosts):
        templ, wit = rng.choice(TRAIN_TEMPLATES), rng.choice(TRAIN_WITNESSES)
        lang = LANGS[i % len(LANGS)]
        inj = _witness(tr, templ, wit, lang)
        out.append({"host": i, "kind": "witness", "data": f"{h.data}\n{inj}", "template": templ,
                    "witness": wit, "lang": lang, "alone": inj})
    return out


def plan_b3(hosts: tuple[RefHost, ...], seed: int) -> list[dict]:
    """SecAlign align.py:38-56 (attack NaiveCompletion, naive_proportion 0.9)."""
    rng = random.Random(f"b3-secalign-{seed}")
    out: list[dict] = []
    for i, h in enumerate(hosts):
        j = _other_host(rng, i, len(hosts))
        prompt = struq_injected_prompt(hosts[j])
        if rng.random() < 0.9:
            kind, data = "naive", struq_naive(h.data, prompt)
        else:
            kind, data = "completion", struq_completion(h.data, h.output, hosts[j], prompt, rng)
        out.append({"host": i, "kind": kind, "data": data, "source": j})
    return out


def rejected_key(arm: str, plan: dict) -> str:
    """Rejected responses are shared: b1 by injected host, b2 by rendered injection."""
    if arm == "b1":
        return f"b1:h{plan['source']:04d}"
    return f"b2:{sha256_text(plan['alone'])[:16]}"


def rejected_items(seed: int, tr: Any) -> list[Item]:
    """Unique 'injection alone' prompts of b1 and b2 (one user turn each)."""
    hs = train_hosts(seed)
    prompts: dict[str, str] = {}
    for arm, plan in (("b1", plan_b1(hs, seed)), ("b2", plan_b2(hs, seed, tr))):
        for p in plan:
            prompts.setdefault(rejected_key(arm, p), p["alone"])
    return [Item(key, text, None, {}) for key, text in prompts.items()]
