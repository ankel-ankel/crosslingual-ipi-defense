import contextlib
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ipidefense.attacks.witnesses import WitnessInjection
from ipidefense.metrics.scoring import classify
from ipidefense.model import build_messages, generate

# Qwen2.5 special tokens and pad rows start at this id
QWEN_SPECIAL_FLOOR = 151643
# token id of " !", used for the initial suffix
INIT_TOKEN_ID = 753
# marks the suffix slot in the rendered prompt
_SENTINEL = "\u0000\u0000GCGSUFFIX\u0000\u0000"

ArmCtx = Callable[[], contextlib.AbstractContextManager]


# defaults follow the SecAlign / StruQ GCG setup
@dataclass(frozen=True)
class GCGConfig:
    steps: int = 500
    search_width: int = 512
    topk: int = 256
    suffix_len: int = 20
    n_replace: int = 1
    check_every: int = 20
    batch: int = 128  # candidate-loss mini-batch
    max_new_tokens: int = 64  # greedy generation at each check
    seed: int = 0
    init_token_id: int = INIT_TOKEN_ID
    use_cache: bool = True  # False = full forward, no prefix cache
    early_stop: bool = True
    allow_non_ascii: bool = False
    universal_add_every: int = 50  # not read by the loops

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# tokens and spans
def build_disallowed_ids(
    tok: PreTrainedTokenizerBase, embed_rows: int, allow_non_ascii: bool = False
) -> Tensor:
    # specials, pad rows and non-ASCII or non-printable tokens
    blocked = set(range(QWEN_SPECIAL_FLOOR, embed_rows))
    if not allow_non_ascii:
        for tid in range(QWEN_SPECIAL_FLOOR):
            piece = tok.decode([tid])
            if not (piece.isascii() and piece.isprintable()):
                blocked.add(tid)
    return torch.tensor(sorted(blocked), dtype=torch.long)


@dataclass(frozen=True)
class PromptSpans:
    """Token spans around the suffix for one host and injection."""

    before_ids: tuple[int, ...]
    after_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    prefix_str: str
    post_str: str
    witness: str
    inj_text: str


def render_spans(
    tok: PreTrainedTokenizerBase,
    instruction: str,
    data: str,
    injection: WitnessInjection,
) -> PromptSpans:
    # split the rendered prompt at the sentinel
    inj_text = injection.render()
    data_full = f"{data}\n{inj_text}{_SENTINEL}"
    rendered = tok.apply_chat_template(
        build_messages(instruction, data_full), add_generation_prompt=True, tokenize=False
    )
    prefix_str, post_str = rendered.split(_SENTINEL)
    return PromptSpans(
        before_ids=tuple(tok.encode(prefix_str, add_special_tokens=False)),
        after_ids=tuple(tok.encode(post_str, add_special_tokens=False)),
        target_ids=tuple(tok.encode(injection.witness, add_special_tokens=False)),
        prefix_str=prefix_str,
        post_str=post_str,
        witness=injection.witness,
        inj_text=inj_text,
    )


def filter_candidates(
    tok: PreTrainedTokenizerBase, cand: Tensor, spans: PromptSpans
) -> Tensor | None:
    # keep candidates that re-tokenise to the same ids
    before, after = list(spans.before_ids), list(spans.after_ids)
    kept: list[Tensor] = []
    for row in cand:
        ids = row.tolist()
        suffix_str = tok.decode(ids)
        recon = tok.encode(spans.prefix_str + suffix_str + spans.post_str, add_special_tokens=False)
        if recon == before + ids + after:
            kept.append(row)
    if not kept:
        return None
    return torch.stack(kept)


# ported from nanoGCG 0.3.0 by Gray Swan AI (MIT, see THIRD_PARTY.md)
def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor | None = None,
) -> Tensor:
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(
        torch.rand((search_width, n_optim_tokens), device=grad.device)
    )[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    return original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)


def sample_valid_candidates(
    tok: PreTrainedTokenizerBase,
    optim: Tensor,
    grad: Tensor,
    spans: PromptSpans,
    cfg: GCGConfig,
    disallowed: Tensor,
    max_rounds: int = 16,
) -> tuple[Tensor, int, int]:
    # oversample until search_width candidates pass, else keep the current suffix
    collected: list[Tensor] = []
    seen: set[tuple[int, ...]] = set()
    rounds = 0
    while len(collected) < cfg.search_width and rounds < max_rounds:
        cand = sample_ids_from_grad(
            optim, grad, cfg.search_width, cfg.topk, cfg.n_replace, disallowed
        )
        kept = filter_candidates(tok, cand, spans)
        rounds += 1
        if kept is None:
            continue
        for row in kept:
            key = tuple(row.tolist())
            if key not in seen:
                seen.add(key)
                collected.append(row)
    if not collected:
        return optim.unsqueeze(0), 0, rounds
    stacked = torch.stack(collected[: cfg.search_width])
    return stacked, stacked.shape[0], rounds


# forward passes
def _ce_from_kept(logits: Tensor, target: Tensor) -> Tensor:
    # mean target CE per row, in fp32 since bf16 is too coarse here
    pred = logits[:, :-1, :].float()
    b, t, v = pred.shape
    ce = F.cross_entropy(pred.reshape(b * t, v), target.reshape(b * t), reduction="none")
    return ce.view(b, t).mean(dim=1)


def token_gradient(
    model: torch.nn.Module,
    embed_w: Tensor,
    spans: PromptSpans,
    optim_ids: Sequence[int],
    arm_ctx: ArmCtx,
    device: torch.device,
) -> Tensor:
    # gradient w.r.t. the one-hot suffix, shape [n, V]
    vocab = embed_w.shape[0]
    target = torch.tensor(spans.target_ids, device=device)
    onehot = torch.zeros(len(optim_ids), vocab, device=device, dtype=embed_w.dtype)
    onehot.scatter_(1, torch.tensor(optim_ids, device=device).unsqueeze(1), 1.0)
    onehot.requires_grad_(True)
    optim_emb = onehot @ embed_w
    with arm_ctx():
        before = embed_w[torch.tensor(spans.before_ids, device=device)].detach()
        after = embed_w[torch.tensor(spans.after_ids, device=device)].detach()
        tgt_emb = embed_w[target].detach()
        seq = torch.cat([before, optim_emb, after, tgt_emb], dim=0).unsqueeze(0)
        attn = torch.ones(1, seq.shape[1], device=device, dtype=torch.long)
        out = model(
            inputs_embeds=seq, attention_mask=attn, use_cache=False, logits_to_keep=len(target) + 1
        )
        loss = _ce_from_kept(out.logits, target).mean()
        # autograd.grad so that no grads land on the LoRA weights
        (grad,) = torch.autograd.grad(loss, [onehot])
    return grad.detach().clone()


@torch.no_grad()
def compute_prefix_kv(
    model: torch.nn.Module, before_ids: Sequence[int], arm_ctx: ArmCtx, device: torch.device
) -> list[tuple[Tensor, Tensor]]:
    # KV cache of the fixed prefix, once per host
    before = torch.tensor(before_ids, device=device).unsqueeze(0)
    attn = torch.ones_like(before)
    with arm_ctx():
        out = model(input_ids=before, attention_mask=attn, use_cache=True, logits_to_keep=1)
    return [(k.detach(), v.detach()) for k, v, *_ in out.past_key_values]


def _batch_cache(prefix_kv: list[tuple[Tensor, Tensor]], batch: int) -> DynamicCache:
    # fresh cache per mini-batch, DynamicCache.update works in place
    data = [
        (k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)) for k, v in prefix_kv
    ]
    return DynamicCache(ddp_cache_data=data)


@torch.no_grad()
def candidate_loss(
    model: torch.nn.Module,
    spans: PromptSpans,
    cand: Tensor,
    cfg: GCGConfig,
    prefix_kv: list[tuple[Tensor, Tensor]] | None,
    arm_ctx: ArmCtx,
    device: torch.device,
) -> Tensor:
    # mean target CE per candidate, shape [C]
    n_cand = cand.shape[0]
    target = torch.tensor(spans.target_ids, device=device)
    after = torch.tensor(spans.after_ids, device=device)
    before = torch.tensor(spans.before_ids, device=device)
    keep = len(target) + 1
    losses = torch.empty(n_cand, device=device)
    with arm_ctx():
        for start in range(0, n_cand, cfg.batch):
            sub = cand[start : start + cfg.batch]
            bsz = sub.shape[0]
            tail = torch.cat([sub, after.repeat(bsz, 1), target.repeat(bsz, 1)], dim=1)
            if prefix_kv is not None:
                cache = _batch_cache(prefix_kv, bsz)
                prefix_len = prefix_kv[0][0].shape[-2]
                attn = torch.ones(bsz, prefix_len + tail.shape[1], device=device, dtype=torch.long)
                out = model(
                    input_ids=tail,
                    attention_mask=attn,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=keep,
                )
            else:
                full = torch.cat([before.repeat(bsz, 1), tail], dim=1)
                attn = torch.ones_like(full)
                out = model(
                    input_ids=full, attention_mask=attn, use_cache=False, logits_to_keep=keep
                )
            losses[start : start + bsz] = _ce_from_kept(out.logits, target.repeat(bsz, 1))
    return losses


def run_check(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    instruction: str,
    data: str,
    spans: PromptSpans,
    suffix_str: str,
    arm_ctx: ArmCtx,
    cfg: GCGConfig,
    rep_penalty: float | None = None,
) -> tuple[dict[str, object], str]:
    # rep_penalty overrides repetition_penalty for this call only
    planted = f"{data}\n{spans.inj_text}{suffix_str}"
    gen_cfg = model.generation_config
    original = gen_cfg.repetition_penalty
    try:
        if rep_penalty is not None:
            gen_cfg.repetition_penalty = rep_penalty
        with arm_ctx():
            resp = generate(
                model, tok, instruction, data=planted, max_new_tokens=cfg.max_new_tokens
            )
    finally:
        gen_cfg.repetition_penalty = original
    return classify(resp, spans.witness), resp


def check_steps(cfg: GCGConfig) -> set[int]:
    steps = {1}
    steps.update(range(cfg.check_every, cfg.steps + 1, cfg.check_every))
    return steps


# attack loops
def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()


def attack_host(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    embed_w: Tensor,
    disallowed: Tensor,
    host_instruction: str,
    host_data: str,
    injection: WitnessInjection,
    cfg: GCGConfig,
    arm_ctx: ArmCtx,
    host_index: int,
    on_event: Callable[[dict], None] | None = None,
) -> dict[str, object]:
    # stop at the first witness-first check, misses are re-checked at penalty 1.0
    device = embed_w.device
    torch.manual_seed(cfg.seed + host_index)
    spans = render_spans(tok, host_instruction, host_data, injection)
    optim = torch.tensor([cfg.init_token_id] * cfg.suffix_len, device=device)
    prefix_kv = None
    if cfg.use_cache:
        prefix_kv = compute_prefix_kv(model, spans.before_ids, arm_ctx, device)
    allowed_steps = check_steps(cfg)
    loss_trace: list[float] = []
    n_valid_trace: list[int] = []
    sec_per_step: list[float] = []
    checks: list[dict] = []
    first_step: dict[str, int | None] = {"wf": None, "rf": None, "starts_with": None}
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        step_t0 = time.time()
        grad = token_gradient(model, embed_w, spans, optim.tolist(), arm_ctx, device)
        filtered, n_valid, _ = sample_valid_candidates(tok, optim, grad, spans, cfg, disallowed)
        losses = candidate_loss(model, spans, filtered, cfg, prefix_kv, arm_ctx, device)
        best = int(losses.argmin())
        optim = filtered[best]
        loss_trace.append(float(losses[best]))
        n_valid_trace.append(n_valid)
        sec_per_step.append(round(time.time() - step_t0, 3))
        if step in allowed_steps:
            suffix_str = tok.decode(optim.tolist())
            labels, resp = run_check(
                model, tok, host_instruction, host_data, spans, suffix_str, arm_ctx, cfg
            )
            # every selected suffix must round-trip exactly
            assert filter_candidates(tok, optim.unsqueeze(0), spans) is not None
            entry = {
                "step": step,
                "loss": loss_trace[-1],
                "labels": labels,
                "response": resp,
                "suffix": suffix_str,
                "suffix_ids": optim.tolist(),
                "n_valid": n_valid,
                "sec_per_step": sec_per_step[-1],
            }
            if labels["wf"] != "obeyed":
                labels_rp1, resp_rp1 = run_check(
                    model, tok, host_instruction, host_data, spans, suffix_str, arm_ctx, cfg,
                    rep_penalty=1.0,
                )
                entry["labels_rp1"] = labels_rp1
                entry["response_rp1"] = resp_rp1
            checks.append(entry)
            if first_step["wf"] is None and labels["wf"] == "obeyed":
                first_step["wf"] = step
            if first_step["rf"] is None and labels["rf"] == "obeyed":
                first_step["rf"] = step
            if first_step["starts_with"] is None and labels["starts_with"] is True:
                first_step["starts_with"] = step
            if cfg.early_stop and labels["starts_with"] is True:
                break
    record = {
        "host_index": host_index,
        "witness": spans.witness,
        "injection": spans.inj_text,
        "n_target_tokens": len(spans.target_ids),
        "success": first_step["wf"] is not None,
        "success_step": first_step["wf"],
        "first_step": first_step,
        "technical_failure": False,
        "steps_run": len(loss_trace),
        "final_loss": loss_trace[-1] if loss_trace else None,
        "final_suffix": tok.decode(optim.tolist()),
        "final_suffix_ids": optim.tolist(),
        "loss_trace": loss_trace,
        "n_valid_trace": n_valid_trace,
        "sec_per_step": sec_per_step,
        "checks": checks,
        "wall_seconds": round(time.time() - t0, 2),
    }
    if on_event is not None:
        on_event(record)
    return record


def attack_universal(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    embed_w: Tensor,
    disallowed: Tensor,
    hosts: Sequence[tuple[str, str]],
    injections: Sequence[WitnessInjection],
    cfg: GCGConfig,
    arm_ctx: ArmCtx,
    on_event: Callable[[dict], None] | None = None,
) -> dict[str, object]:
    # one suffix for several hosts (Zou et al. 2023), hosts join one at a time
    device = embed_w.device
    torch.manual_seed(cfg.seed)
    spans = [render_spans(tok, instr, data, injections[i % len(injections)])
             for i, (instr, data) in enumerate(hosts)]
    prefix_kv = [
        compute_prefix_kv(model, sp.before_ids, arm_ctx, device) if cfg.use_cache else None
        for sp in spans
    ]
    optim = torch.tensor([cfg.init_token_id] * cfg.suffix_len, device=device)
    allowed_steps = check_steps(cfg)
    active = 1
    loss_trace: list[float] = []
    inclusion: list[dict] = []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        grad = sum(
            token_gradient(model, embed_w, spans[i], optim.tolist(), arm_ctx, device)
            for i in range(active)
        )
        cand = sample_ids_from_grad(
            optim, grad, cfg.search_width, cfg.topk, cfg.n_replace, disallowed
        )
        # a candidate must re-tokenise cleanly for every active host
        for i in range(active):
            cand = filter_candidates(tok, cand, spans[i])
            if cand is None:
                break
        if cand is None:
            cand = optim.unsqueeze(0)
        total = torch.zeros(cand.shape[0], device=device)
        for i in range(active):
            total = total + candidate_loss(
                model, spans[i], cand, cfg, prefix_kv[i], arm_ctx, device
            )
        best = int(total.argmin())
        optim = cand[best]
        loss_trace.append(float(total[best]) / active)
        if step in allowed_steps and active < len(hosts):
            suffix_str = tok.decode(optim.tolist())
            passed = all(
                run_check(
                    model, tok, hosts[i][0], hosts[i][1], spans[i], suffix_str, arm_ctx, cfg
                )[0]["wf"] == "obeyed"
                for i in range(active)
            )
            if passed:
                active += 1
                inclusion.append({"step": step, "active_hosts": active})
    record = {
        "n_train_hosts": len(hosts),
        "final_active_hosts": active,
        "final_suffix": tok.decode(optim.tolist()),
        "final_suffix_ids": optim.tolist(),
        "loss_trace": loss_trace,
        "inclusion": inclusion,
        "wall_seconds": round(time.time() - t0, 2),
    }
    if on_event is not None:
        on_event(record)
    return record
