import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # runs without installing

from ipidefense.paths import RUNS
from ipidefense.runner import Run, Step, main, shown, user_path

OUT = str(RUNS / "figures")

# single-column serif style
_names = {f.name for f in fm.fontManager.ttflist}
SERIF = "Times New Roman" if "Times New Roman" in _names else "DejaVu Serif"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [SERIF],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

RED = "#A32D2D"     # undefended / broken
TEAL = "#0F6E56"    # defended

COLW = 3.45         # column width (inches)
HT = 2.05


def _frame(ax):
    """No top/right spine, faint y grid behind the marks."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3, width=0.8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", stem + ".pdf/.png")


def fig_security():
    # dumbbell plot, trained vs unseen languages
    langs = ["en", "es", "vi", "fr", "de", "zh"]
    # undefended values with the corrected renderer
    undef = [0.963, 0.519, 0.783, 0.733, 0.724, 0.659]
    defd = [0.001, 0.001, 0.000, 0.001, 0.001, 0.000]
    x = np.array([0, 1, 2, 3.7, 4.7, 5.7])  # gap between the two groups
    fig, ax = plt.subplots(figsize=(COLW, HT))
    for xi, u, d in zip(x, undef, defd, strict=True):
        ax.plot([xi, xi], [d, u], color="0.78", lw=1.1, zorder=2)
    ax.scatter(x, undef, s=26, color=RED, zorder=3, label="undefended")
    ax.scatter(x, defd, s=26, color=TEAL, zorder=3, label="defended",
               edgecolors="white", linewidths=0.5)
    for xi, u in zip(x, undef, strict=True):
        ax.text(xi, u + 0.04, f"{u:.3f}", ha="center", va="bottom",
                fontsize=6.2, color="0.15")
    for xi, d in zip(x, defd, strict=True):
        ax.text(xi + 0.14, d + 0.012, f"{d:.3f}", ha="left", va="bottom",
                fontsize=6.0, color=TEAL)
    ax.set_ylim(-0.03, 1.14)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("injection obey rate")
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_xlim(-0.55, 6.25)
    ax.text(1.0, -0.30, "trained", ha="center", va="top", fontsize=7.5,
            style="italic", color="0.35", transform=ax.get_xaxis_transform())
    ax.text(4.7, -0.30, "unseen", ha="center", va="top", fontsize=7.5,
            style="italic", color="0.35", transform=ax.get_xaxis_transform())
    _frame(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2,
              frameon=False, handlelength=1.0, columnspacing=1.6,
              scatterpoints=1)
    save(fig, "fig_security")


def fig_mechanism():
    # dumbbell by channel, undefended vs defended
    x = np.array([0.0, 1.0])
    base = [44.96, 30.35]        # data channel, user channel
    defended = [-0.05, 1.61]
    fig, ax = plt.subplots(figsize=(COLW, HT))
    for xi, b, d in zip(x, base, defended, strict=True):
        ax.plot([xi, xi], [d, b], color="0.78", lw=1.1, zorder=2)
    ax.scatter(x, base, s=26, color=RED, zorder=3, label="base model")
    ax.scatter(x, defended, s=26, color=TEAL, zorder=3, label="defended model",
               edgecolors="white", linewidths=0.5)
    ax.axhline(0, color="0.4", lw=0.7, zorder=1)
    for xi, b in zip(x, base, strict=True):
        ax.text(xi, b + 1.6, f"{b:.2f}", ha="center", va="bottom",
                fontsize=6.6, color="0.15")
    ax.text(x[0] + 0.07, defended[0] - 2.6, f"{defended[0]:.2f}", ha="left",
            va="top", fontsize=6.6, color=TEAL)
    ax.text(x[1] + 0.07, defended[1] + 2.0, f"{defended[1]:.2f}", ha="left",
            va="bottom", fontsize=6.6, color=TEAL)
    ax.set_ylim(-7, 55)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(r"compliance push $P_c$")
    ax.set_xticks(x)
    ax.set_xticklabels(["data channel", "user channel"])
    _frame(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2,
              frameon=False, handlelength=1.0, columnspacing=1.6)
    save(fig, "fig_mechanism")


def fig_adaptive():
    # lollipop so near-zero values stay visible
    labels = ["initial defense\n(seed 0)", "hardened\n(seed 0)", "hardened\n(seed 1)"]
    asr = [0.890, 0.010, 0.040]
    colors = [RED, TEAL, TEAL]
    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(COLW, HT))
    for xi, v in zip(x, asr, strict=True):
        ax.plot([xi, xi], [0, v], color="0.78", lw=1.1, zorder=2)
    ax.scatter(x, asr, s=26, color=colors, zorder=3,
               edgecolors="white", linewidths=0.5)
    for xi, v in zip(x, asr, strict=True):
        ax.text(xi, v + 0.035, f"{v:.3f}", ha="center", va="bottom",
                fontsize=7.0, color="0.15")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("best-of-$N$ attack success")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _frame(ax)
    save(fig, "fig_adaptive")


FIGURES = {"security": fig_security, "mechanism": fig_mechanism, "adaptive": fig_adaptive}


def draw(name: str) -> Callable[[Run], None]:
    def step(run: Run) -> None:
        global OUT
        OUT = str(run.args.out)
        if run.dry_run:
            print(f"draw {shown(Path(OUT) / f'fig_{name}')}.pdf and .png", flush=True)
            return
        os.makedirs(OUT, exist_ok=True)
        FIGURES[name]()
    return step


def options(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", type=user_path, default=OUT, help="output folder")


STEPS = [
    Step("security", "obey rates of the undefended and the defended model per language",
         "Security figure", draw("security")),
    Step("mechanism", "compliance push in the data and the user channel", "Mechanism figure",
         draw("mechanism")),
    Step("adaptive", "best-of-N attack success on the initial and the hardened defense",
         "Adaptive attack figure", draw("adaptive")),
]


if __name__ == "__main__":
    main(__file__, STEPS, options)
