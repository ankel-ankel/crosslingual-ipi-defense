import re

from ipidefense.attacks.witnesses import WitnessInjection
from ipidefense.rendering.nllb import NLLBTranslator, load_nllb

# markup with no natural-language content, never translated alone
MARKUP = re.compile(r"<<[^<>]*>>|<!--|-->|</?[A-Za-z][A-Za-z0-9_-]*>|#{2,}|-{2,}|\[[A-Z0-9_ -]+\]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

# full NLLB-200 codes, short codes can silently pick the wrong language
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
    """Assert that code is a real NLLB language token, not a word piece."""
    _, tok = load_nllb()
    tid = tok.convert_tokens_to_ids(code)
    if tid is None or tid < 256001:
        raise ValueError(f"{code!r} is not an NLLB language token (id={tid})")


def is_translatable(segment: str) -> bool:
    """Only segments with >= 2 letters outside markup go to NLLB."""
    return len(_LETTER.findall(MARKUP.sub(" ", segment))) >= 2


def render_v2(tr: NLLBTranslator, inj: WitnessInjection, lang: str) -> str:
    """Corrected renderer, lang is a short code."""
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
