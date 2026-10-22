"""Shared helpers of the human-written attack pools: adaptation, text screens and language ID."""
from __future__ import annotations

import re
import unicodedata

from lid import identify  # seeded langdetect (DetectorFactory.seed = 0)

PLACEHOLDER = "{witness}"
MIN_LEN, MAX_LEN = 20, 600

CHAT_TOKENS = re.compile(
    r"<\|[^|<>\n]{1,40}\|>|\[/?INST\]|<</?SYS>>|</?s>|<start_of_turn>|<end_of_turn>"
    r"|<bos>|<eos>|<pad>|\bim_start\b|\bim_end\b|endoftext",
    re.I,
)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
GCG_SUFFIX = re.compile(r"similarlyNow|oppositeley", re.I)  # Zou et al. (2023) optimised suffix
ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060-\u2064\ufeff\u00ad]")
TRAIN_WITNESS = re.compile(r"\b(?:pwned|banana|hacked|clicked)\b", re.I)  # training witnesses
WORD_COUNT = re.compile(r"\b(?:two|three|four|2|3|4)[- ]words?\b", re.I)
EN_FUNC = frozenset(
    "the a an and or of to in on for with is are be been was were this "  # noqa: SIM905
    "that it you your i me my say print output write ignore above previous instructions "
    "instead just only following do not now please what will would should can from as at "
    "by all any".split()
)


def esc(s: str) -> str:
    """Escape literal braces so the template is safe for str.format(witness=...)."""
    return s.replace("{", "{{").replace("}", "}}")


def adapt(text: str, pattern: re.Pattern) -> dict:
    """The single target-phrase match replaced by {witness}, round trip checked:
    dict(ok, reason, adapted, matched, n_match, braces_escaped)."""
    ms = list(pattern.finditer(text))
    out = {"ok": False, "reason": "", "adapted": None, "matched": None, "n_match": len(ms),
           "braces_escaped": False}
    if len(ms) != 1:
        out["reason"] = f"n_target={len(ms)}"
        return out
    m = ms[0]
    adapted = esc(text[: m.start()]) + PLACEHOLDER + esc(text[m.end():])
    if adapted.count(PLACEHOLDER) != 1:
        out["reason"] = "placeholder_count"
        return out
    try:
        if adapted.format(witness=m.group(0)) != text:
            out["reason"] = "roundtrip_mismatch"
            return out
    except (KeyError, IndexError, ValueError):
        out["reason"] = "format_error"
        return out
    out.update(ok=True, adapted=adapted, matched=m.group(0),
               braces_escaped=("{" in text or "}" in text))
    return out


def residual(text: str, pattern: re.Pattern) -> str:
    return pattern.sub(" ", text)


def quality_reason(text: str) -> str:
    """Generic text filters; returns '' when the text passes."""
    if not MIN_LEN <= len(text) <= MAX_LEN:
        return "length"
    if CHAT_TOKENS.search(text):
        return "chat_template_token"
    if CONTROL.search(text):
        return "control_char"
    if ZERO_WIDTH.search(text):
        return "zero_width_char"
    if GCG_SUFFIX.search(text):
        return "machine_generated_suffix"
    return ""


def en_ratio(text: str) -> float:
    toks = re.findall(r"[A-Za-z']+", text.casefold())
    return sum(t in EN_FUNC for t in toks) / len(toks) if toks else 0.0


def lang_of(resid: str) -> dict:
    """Seeded LID on the residual (target phrase removed) + English-function-word ratio."""
    r = identify(resid)
    r["en_ratio"] = round(en_ratio(resid), 3)
    return r


def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"[\W_]+", "", s)
