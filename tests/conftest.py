import os

import pytest

# an empty value does not hide the GPU everywhere
HIDE_GPU = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = HIDE_GPU


def pytest_collection_finish(session: pytest.Session) -> None:
    # a test module may have changed it on import
    os.environ["CUDA_VISIBLE_DEVICES"] = HIDE_GPU


@pytest.fixture(scope="session")
def qwen_tok():
    """Qwen tokenizer with the input-role template, skipped when absent."""
    from ipidefense.model import QWEN_DIR, QWEN_INPUT_TEMPLATE

    if not (QWEN_DIR / "tokenizer_config.json").exists():
        pytest.skip(f"Qwen tokenizer not found under {QWEN_DIR}")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    tok.chat_template = QWEN_INPUT_TEMPLATE
    return tok
