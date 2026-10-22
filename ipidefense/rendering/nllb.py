"""NLLB-200 (distilled 600M) translator with an in-memory cache; greedy decoding."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLLB_MODEL = "facebook/nllb-200-distilled-600M"

LANG_CODES = {
    "en": "eng_Latn",
    "vi": "vie_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
}


@lru_cache(maxsize=1)
def load_nllb() -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    model.generation_config.max_length = None
    return model, tok


@dataclass(frozen=True)
class NLLBTranslator:
    src: str = "eng_Latn"
    _cache: dict = field(default_factory=dict, compare=False)

    @torch.no_grad()
    def translate(self, text: str, target_lang: str) -> str:
        text = text.strip()
        if not text:
            return text
        tgt = LANG_CODES.get(target_lang, target_lang)
        key = (text, tgt)
        if key in self._cache:
            return self._cache[key]
        model, tok = load_nllb()
        tok.src_lang = self.src
        enc = tok(text, return_tensors="pt").to(model.device)
        bos = tok.convert_tokens_to_ids(tgt)
        out = model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=128)
        result = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        self._cache[key] = result
        return result
