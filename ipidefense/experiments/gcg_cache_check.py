import argparse
import contextlib

import torch

from ipidefense.attacks import gcg as g
from ipidefense.attacks.witnesses import WitnessInjection
from ipidefense.data.splits import hosts
from ipidefense.model import load_defense_lora_qwen
from ipidefense.paths import MODELS, RUNS
from ipidefense.utils import write_json

GATE = 0.02


def main() -> None:
    argparse.ArgumentParser().parse_args()
    # hardened seed-0 adapter, 8 candidates per pilot host
    model, tok = load_defense_lora_qwen(str(MODELS / "defense-qwen-seed0-hardened"))
    g.freeze_model(model)
    emb = model.get_input_embeddings().weight
    disallowed = g.build_disallowed_ids(tok, emb.shape[0])
    dev = emb.device

    def ctx() -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()

    rows = []
    for i, h in enumerate(hosts(0, "pilot")[:5]):
        spans = g.render_spans(tok, h.instruction, h.data,
                               WitnessInjection("Reply {witness}.", "HIJACKED"))
        opt = torch.tensor([g.INIT_TOKEN_ID] * 20, device=dev)
        grad = g.token_gradient(model, emb, spans, opt.tolist(), ctx, dev)
        cand, _, _ = g.sample_valid_candidates(tok, opt, grad, spans,
                                               g.GCGConfig(search_width=8), disallowed)
        kv = g.compute_prefix_kv(model, spans.before_ids, ctx, dev)
        a = g.candidate_loss(model, spans, cand, g.GCGConfig(batch=8, use_cache=True), kv, ctx,
                             dev)
        b = g.candidate_loss(model, spans, cand, g.GCGConfig(batch=8, use_cache=False), None,
                             ctx, dev)
        delta = float((a - b).abs().max())
        rows.append({"host": i, "max_abs_delta": delta, "loss_cache": a.tolist(),
                     "loss_nocache": b.tolist()})
        print(f"host {i}: max|delta| = {delta:.5f}", flush=True)
    worst = max(r["max_abs_delta"] for r in rows)
    out = {"gate": GATE, "max_abs_delta": worst, "passes": worst <= GATE, "rows": rows}
    write_json(RUNS / "gcg" / "cache_check.json", out)
    verdict = "PASS" if out["passes"] else "FAIL"
    print(f"cache check {verdict}: worst max|delta| = {worst:.5f}", flush=True)


if __name__ == "__main__":
    main()
