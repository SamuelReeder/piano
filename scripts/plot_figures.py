"""Draw the README figures: validation curves and a piano roll of a generated sample.

    uv run python scripts/plot_figures.py samples/sample-15.mid \\
        --run "MAESTRO only, 39M" checkpoints/maestro-only/metrics.jsonl \\
        --run "Aria-MIDI pretraining, 85M" checkpoints/pretrain/metrics.jsonl
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from pianist import tokenizer as tok

SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RUN_COLORS = ("#eb6834", "#2a78d6")  # categorical palette slots 2 and 1

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED, "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def validation_curve(runs: list[tuple[str, Path]], out: Path) -> None:
    """MAESTRO validation loss of each run; all runs score the same validation windows."""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    top = 0.0
    for (label, metrics), color in zip(runs, RUN_COLORS, strict=False):
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        x, y = np.array([(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]).T
        ax.plot(x, y, color=color, linewidth=2, label=label)
        best = int(np.argmin(y))
        ax.plot(x[best], y[best], "o", color=color, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2)
        ax.annotate(f"{y[best]:.3f}", (x[best], y[best]), xytext=(0, -16),
                    textcoords="offset points", ha="center", color=INK, fontsize=10)
        top = max(top, y[min(2, len(y) - 1)])
    ax.set_ylim(None, top + 0.05)
    ax.set_xlim(0, None)
    ax.set_xlabel("optimizer step (64k tokens each)")
    ax.set_ylabel("validation cross-entropy (nats / token)")
    ax.set_title("MAESTRO validation loss")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "validation_curve.png")


def piano_roll(midi: Path, out: Path, seconds: float = 60.0) -> None:
    notes = tok.read_midi(midi)
    notes = notes[notes.onset < seconds]
    blues = LinearSegmentedColormap.from_list("velocity", ["#b7d3f3", "#2a78d6", "#0d3a73"])
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)
    order = np.argsort(notes.velocity)  # loud notes on top
    ax.barh(notes.pitch[order], notes.duration[order], left=notes.onset[order], height=0.9,
            color=blues(notes.velocity[order] / 127), linewidth=0)
    c4 = [p for p in range(24, 109, 12) if notes.pitch.min() - 6 <= p <= notes.pitch.max() + 6]
    ax.set_yticks(c4, [f"C{p // 12 - 1}" for p in c4])
    ax.set_ylim(notes.pitch.min() - 3, notes.pitch.max() + 3)
    ax.set_xlim(0, seconds)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("time (s)")
    ax.set_title(f"Generated sample, first {seconds:.0f} s (darker = louder)")
    fig.tight_layout()
    fig.savefig(out / "piano_roll.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path, help="generated MIDI file for the piano roll")
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "METRICS"),
                        default=[], help="a run to plot: label and metrics.jsonl path")
    parser.add_argument("--out", type=Path, default=Path("docs"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    if args.run:
        validation_curve([(label, Path(path)) for label, path in args.run], args.out)
    piano_roll(args.sample, args.out)
