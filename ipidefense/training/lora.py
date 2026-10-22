from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ipidefense.model import BASE_DIR

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class SFTExample:
    messages: list[dict]
    target: str


def build_lora_model(
    model_dir: str = str(BASE_DIR), r: int = 16, alpha: int = 32, dropout: float = 0.05
) -> torch.nn.Module:
    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": 0})
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(base, cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def encode_example(
    tok: PreTrainedTokenizerBase, ex: SFTExample
) -> tuple[torch.Tensor, torch.Tensor]:
    full = tok.apply_chat_template(
        [*ex.messages, {"role": "assistant", "content": ex.target}],
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"][0]
    prompt = tok.apply_chat_template(
        ex.messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"][0]
    # loss covers the assistant turn only
    labels = full.clone()
    labels[: len(prompt)] = -100
    return full, labels


def train_sft(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    examples: list[SFTExample],
    epochs: int = 1,
    lr: float = 1e-4,
    accum: int = 8,
    max_len: int = 1024,
    log_every: int = 10,
    seed: int = 0,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    dev = next(model.parameters()).device
    step = 0
    for ep in range(epochs):
        order = torch.randperm(len(examples), generator=torch.Generator().manual_seed(seed + ep))
        opt.zero_grad()
        recent: list[float] = []
        # batch size 1 with grad accumulation, a final partial group still steps
        for i, idx in enumerate(order.tolist()):
            ids, labels = encode_example(tok, examples[idx])
            ids = ids[:max_len].unsqueeze(0).to(dev)
            labels = labels[:max_len].unsqueeze(0).to(dev)
            loss = model(input_ids=ids, labels=labels).loss / accum
            loss.backward()
            recent.append(loss.item() * accum)
            last = (i + 1) == len(order)
            if (i + 1) % accum == 0 or last:
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % log_every == 0:
                    avg = sum(recent) / len(recent)
                    print(f"epoch {ep} step {step} loss {avg:.4f}", flush=True)
                    recent = []
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    return model
