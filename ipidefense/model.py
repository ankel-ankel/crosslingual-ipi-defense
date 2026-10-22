import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ipidefense.paths import MODELS

ADAPTER_DIR = MODELS / "Meta-SecAlign-8B"
QWEN_DIR = MODELS / "Qwen2.5-7B-Instruct"
SEALION_DIR = MODELS / "Llama-SEA-LION-v3-8B-IT"
MISTRAL_DIR = MODELS / "Mistral-7B-Instruct-v0.3"

# ChatML without a default system prompt, renders any role incl. "input"
QWEN_INPUT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def load_defense_lora_qwen(adapter_dir: str) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    tok.chat_template = QWEN_INPUT_TEMPLATE
    base = AutoModelForCausalLM.from_pretrained(
        str(QWEN_DIR), dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, adapter_dir, device_map={"": 0})
    model.eval()
    return model, tok


def load_qwen_undefended() -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    tok.chat_template = QWEN_INPUT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        str(QWEN_DIR), dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    return model, tok


def sealion_tokenizer() -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(str(SEALION_DIR))
    tok.chat_template = AutoTokenizer.from_pretrained(str(ADAPTER_DIR)).chat_template
    return tok


def load_defense_lora_sealion(adapter_dir: str) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    tok = sealion_tokenizer()
    base = AutoModelForCausalLM.from_pretrained(
        str(SEALION_DIR), dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, adapter_dir, device_map={"": 0})
    model.eval()
    return model, tok


def load_sealion_undefended() -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    model = AutoModelForCausalLM.from_pretrained(
        str(SEALION_DIR), dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    return model, sealion_tokenizer()


def mistral_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(str(MISTRAL_DIR))


def load_defense_lora_mistral(adapter_dir: str) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    base = AutoModelForCausalLM.from_pretrained(
        str(MISTRAL_DIR), dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, adapter_dir, device_map={"": 0})
    model.eval()
    return model, mistral_tokenizer()


def load_mistral_undefended() -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    model = AutoModelForCausalLM.from_pretrained(
        str(MISTRAL_DIR), dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    return model, mistral_tokenizer()


def load_defended(base: str, adapter_dir: str) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    if base == "qwen":
        return load_defense_lora_qwen(adapter_dir)
    if base == "sealion":
        return load_defense_lora_sealion(adapter_dir)
    return load_defense_lora_mistral(adapter_dir)


def load_base_undefended(base: str) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    if base == "qwen":
        return load_qwen_undefended()
    if base == "sealion":
        return load_sealion_undefended()
    return load_mistral_undefended()


def build_messages(instruction: str, data: str | None = None, inline: bool = False) -> list[dict]:
    if data is None:
        return [{"role": "user", "content": instruction}]
    if inline:
        return [{"role": "user", "content": f"{instruction}\n\n{data}"}]
    return [{"role": "user", "content": instruction}, {"role": "input", "content": data}]


def build_inputs(
    tok: PreTrainedTokenizerBase, instruction: str, data: str | None = None, inline: bool = False
) -> dict[str, torch.Tensor]:
    return tok.apply_chat_template(
        build_messages(instruction, data, inline),
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    instruction: str,
    data: str | None = None,
    max_new_tokens: int = 256,
    inline: bool = False,
) -> str:
    inputs = build_inputs(tok, instruction, data, inline).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


@torch.no_grad()
def generate_batch(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    items: list[tuple[str, str | None]],
    max_new_tokens: int = 64,
    inline: bool = False,
) -> list[str]:
    # left padding so prompts end right before the new tokens
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    side = tok.padding_side
    tok.padding_side = "left"
    texts = [
        tok.apply_chat_template(
            build_messages(instr, data, inline), add_generation_prompt=True, tokenize=False
        )
        for instr, data in items
    ]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id
    )
    tok.padding_side = side
    gen = out[:, enc["input_ids"].shape[1] :]
    return [tok.decode(g, skip_special_tokens=True) for g in gen]
