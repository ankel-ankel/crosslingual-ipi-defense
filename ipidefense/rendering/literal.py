"""Witness-literal rendering: the witness stays verbatim and the wrapper around it is translated.
render_inj_literal is the original renderer; render_v2 keeps segments without translatable words
(a lone "." or a markup tag) verbatim instead of sending them to NLLB, which invents text for them.
"""
from __future__ import annotations

import re

from ipidefense.attacks.witnesses import WitnessInjection
from ipidefense.rendering.nllb import NLLBTranslator, load_nllb

# Markup without natural-language content, never translated alone. Bracketed text counts as
# markup only when tag-like (e.g. "[ADMIN]").
MARKUP = re.compile(r"<<[^<>]*>>|<!--|-->|</?[A-Za-z][A-Za-z0-9_-]*>|#{2,}|-{2,}|\[[A-Z0-9_ -]+\]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

# Full NLLB-200 codes. rendering.nllb.LANG_CODES lacks ko/bn/th/...; an unknown short code would
# reach NLLB as an ordinary token and silently produce the wrong language.
NLLB_CODES: dict[str, str] = {
    "en": "eng_Latn", "es": "spa_Latn", "vi": "vie_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "zh": "zho_Hans", "ja": "jpn_Jpan", "ar": "arb_Arab", "ko": "kor_Hang", "bn": "ben_Beng",
    "th": "tha_Thai", "hi": "hin_Deva", "id": "ind_Latn", "ru": "rus_Cyrl", "sw": "swh_Latn",
    "my": "mya_Mymr", "km": "khm_Khmr", "tr": "tur_Latn", "pt": "por_Latn", "it": "ita_Latn",
    "pl": "pol_Latn",
}


def render_inj_literal(tr: NLLBTranslator, inj: WitnessInjection, lang: str) -> str:
    if lang == "en":
        return inj.render()
    pre, suf = inj.wrapper_prefix(), inj.wrapper_suffix()
    pre_t = tr.translate(pre, lang) if pre.strip() else pre
    suf_t = tr.translate(suf, lang) if suf.strip() else suf
    tail = f" {suf_t}" if suf_t.strip() else suf_t
    return f"{pre_t} {inj.witness}{tail}".strip()


def nllb_code(lang: str) -> str:
    """Map a short language code to its NLLB-200 code (KeyError if unknown)."""
    return NLLB_CODES[lang]


def check_nllb_code(code: str) -> None:
    """Assert that `code` is a real NLLB language token (ids >= 256001), not a word piece."""
    _, tok = load_nllb()
    tid = tok.convert_tokens_to_ids(code)
    if tid is None or tid < 256001:
        raise ValueError(f"{code!r} is not an NLLB language token (id={tid})")


def is_translatable(segment: str) -> bool:
    """A segment is sent to NLLB only if it has >= 2 letters outside markup tokens."""
    return len(_LETTER.findall(MARKUP.sub(" ", segment))) >= 2


def render_v2(tr: NLLBTranslator, inj: WitnessInjection, lang: str) -> str:
    """Corrected renderer; `lang` is a short code, mapped to the full NLLB code here."""
    if lang == "en":
        return inj.render()
    code = nllb_code(lang)
    pre, suf = inj.wrapper_prefix(), inj.wrapper_suffix()
    pre_t = tr.translate(pre, code) if is_translatable(pre) else pre.strip()
    if is_translatable(suf):
        suf_t = tr.translate(suf, code)
        tail = f" {suf_t}" if suf_t else ""
    else:
        tail = suf.rstrip()
    return f"{pre_t} {inj.witness}{tail}".strip()
