"""Hide the GPU before any test module is imported; shared Qwen tokenizer fixture."""
from __future__ import annotations

import os

import pytest

# "-1" rather than an empty value: an empty CUDA_VISIBLE_DEVICES does not hide the GPU everywhere
HIDE_GPU = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = HIDE_GPU


def pytest_collection_finish(session: pytest.Session) -> None:
    # keep the GPU hidden even if a test module set another value on import
    os.environ["CUDA_VISIBLE_DEVICES"] = HIDE_GPU


@pytest.fixture(scope="session")
def qwen_tok():
    """Qwen2.5-7B-Instruct tokenizer with the input-role chat template (skipped when absent)."""
    from ipidefense.model import QWEN_DIR, QWEN_INPUT_TEMPLATE

    if not (QWEN_DIR / "tokenizer_config.json").exists():
        pytest.skip(f"Qwen tokenizer not found under {QWEN_DIR}")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    tok.chat_template = QWEN_INPUT_TEMPLATE
    return tok
