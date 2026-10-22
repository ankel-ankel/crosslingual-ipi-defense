import numpy as np
import torch

from ipidefense.attacks.witnesses import OutcomeClassifier
from ipidefense.model import generate, generate_batch


def bootstrap_ci(flags: list[int], n_boot: int = 2000, seed: int = 0) -> list[float]:
    if not flags:
        return [0.0, 0.0]
    arr = np.asarray(flags, dtype=float)
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, arr.size).mean() for _ in range(n_boot)]
    return [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


@torch.no_grad()
def outcomes(
    model, tok, clf: OutcomeClassifier, hosts, inj_text: str, witness: str, mnt: int,
    inline: bool = False, batch: int = 1,
) -> list[str]:
    items = [(h.instruction, f"{h.data}\n{inj_text}") for h in hosts]
    if batch <= 1:
        outs = [
            generate(model, tok, instr, data=data, max_new_tokens=mnt, inline=inline)
            for instr, data in items
        ]
    else:
        outs = []
        for k in range(0, len(items), batch):
            chunk = items[k : k + batch]
            outs += generate_batch(model, tok, chunk, max_new_tokens=mnt, inline=inline)
    return [clf.classify(o, witness) for o in outs]


def obey_by_lang(
    model, tok, clf, hosts, rendered: dict, mnt: int, inline: bool = False, batch: int = 1
) -> dict:
    out = {}
    for lang, attacks in rendered.items():
        per, pooled = [], []
        for txt, w in attacks:
            labels = outcomes(model, tok, clf, hosts, txt, w, mnt, inline, batch)
            per.append(round(sum(o == "obeyed" for o in labels) / len(labels), 3))
            pooled.extend(labels)
        flags = [int(o == "obeyed") for o in pooled]
        n = len(pooled)
        out[lang] = {
            "mean": round(sum(flags) / n, 3),
            "per_attack": per,
            "ci95": bootstrap_ci(flags),
            "n_trials": n,
            "outcomes": {
                "obeyed": round(sum(o == "obeyed" for o in pooled) / n, 3),
                "refused": round(sum(o == "refused" for o in pooled) / n, 3),
                "ignored": round(sum(o == "ignored" for o in pooled) / n, 3),
            },
        }
    return out


def fmt_obey(blocks: dict[str, dict]) -> str:
    langs = list(next(iter(blocks.values())).keys())
    lines = [f"{'lang':<6}" + "".join(f"{name:>13}" for name in blocks)]
    for ll in langs:
        lines.append(f"{ll:<6}" + "".join(f"{b[ll]['mean']:>13.3f}" for b in blocks.values()))
    return "\n".join(lines)
