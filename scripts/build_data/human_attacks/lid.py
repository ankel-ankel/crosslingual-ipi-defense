"""Seeded language identification helpers (langdetect + Unicode-script rules)."""
from __future__ import annotations

import re
import unicodedata

from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0

_SCRIPTS = (
    ("HANGUL", "ko"), ("HIRAGANA", "ja"), ("KATAKANA", "ja"), ("CJK", "zh"),
    ("CYRILLIC", "cyr"), ("ARABIC", "ar"), ("THAI", "th"), ("DEVANAGARI", "hi"),
    ("BENGALI", "bn"), ("HEBREW", "he"), ("GREEK", "el"), ("LATIN", "latin"),
)


def script_profile(text: str) -> dict[str, int]:
    prof: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for key, lab in _SCRIPTS:
            if key in name:
                prof[lab] = prof.get(lab, 0) + 1
                break
        else:
            prof["other"] = prof.get("other", 0) + 1
    return prof


def langdetect_top(text: str) -> tuple[str, float]:
    try:
        r = detect_langs(text)
        return r[0].lang, float(r[0].prob)
    except Exception:
        return "und", 0.0


def identify(text: str) -> dict:
    """Return dict(lang, prob, script, n_letters, method)."""
    t = re.sub(r"\s+", " ", text).strip()
    prof = script_profile(t)
    n = sum(prof.values())
    if n < 3:
        return {"lang": "und", "prob": 0.0, "script": "none", "n_letters": n, "method": "too_short"}
    dom, cnt = max(prof.items(), key=lambda kv: kv[1])
    ld, p = langdetect_top(t)
    if dom != "latin" and cnt / n >= 0.3 and dom not in ("other",):
        if dom == "zh" and prof.get("ja", 0) > 0:
            dom = "ja"
        if dom == "cyr":
            lang = ld if ld in ("ru", "uk", "bg", "mk", "sr", "be", "kk") else "ru"
        elif dom == "ar":
            lang = ld if ld in ("ar", "fa", "ur") else "ar"
        elif dom == "hi":
            lang = ld if ld in ("hi", "mr", "ne") else "hi"
        else:
            lang = dom
        return {"lang": lang, "prob": round(p, 3), "script": dom, "n_letters": n,
                "method": "script"}
    ld = "zh" if ld.startswith("zh") else ld
    return {"lang": ld, "prob": round(p, 3), "script": "latin" if dom == "latin" else dom,
            "n_letters": n, "method": "langdetect"}
