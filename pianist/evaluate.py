"""Evaluate a checkpoint on the held-out MAESTRO test split.

    pianist-evaluate checkpoints/finetune/best.pt --samples outputs/candidates \\
        --out docs/results/final.json

Reports cross-entropy per token and per token class, bits per note (tokenization-agnostic
within this encoding: every note costs a pitch, velocity and duration token plus its
timing), and a count-based bigram baseline trained on the same training split. With
``--samples``, it also compares simple musical statistics of generated MIDI files with
real test performances.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from pianist import tokenizer as tok
from pianist.data import Corpus, eval_windows
from pianist.model import load_checkpoint


@torch.no_grad()
def model_nll(model, windows: torch.Tensor, batch_size: int = 16):
    """Per-target negative log-likelihoods (nats) and target ids, padding removed."""
    device = model.embed.weight.device
    nll, targets = [], []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model.loss(batch[:, :-1], batch[:, 1:], reduction="none")
        y = batch[:, 1:].flatten()
        keep = y != tok.PAD
        nll.append(loss[keep].cpu())
        targets.append(y[keep].cpu())
    return torch.cat(nll).numpy(), torch.cat(targets).numpy()


def bigram_nll(train: Corpus, windows: torch.Tensor, alpha: float = 0.1):
    """Add-``alpha`` smoothed token bigram model restricted to the grammar."""
    counts = np.zeros((tok.VOCAB_SIZE, tok.VOCAB_SIZE))
    for i in range(len(train)):
        t = tok.encode(train.piece(i))
        np.add.at(counts, (t[:-1], t[1:]), 1)
    grammar = tok.grammar_table()
    probs = (counts + alpha) * grammar
    probs /= probs.sum(1, keepdims=True)
    w = windows.numpy()
    x, y = w[:, :-1].flatten(), w[:, 1:].flatten()
    keep = y != tok.PAD
    return -np.log(probs[x[keep], y[keep]]), y[keep]


def summarize(nll: np.ndarray, targets: np.ndarray) -> dict:
    classes = tok.token_class(targets)
    n_notes = int((classes == tok.CLASS_PITCH).sum())
    out = {"nll_per_token": float(nll.mean()), "perplexity": float(math.exp(nll.mean())),
           "bits_per_note": float(nll.sum() / math.log(2) / n_notes), "tokens": int(len(nll))}
    for c, name in enumerate(tok.CLASS_NAMES):
        if (classes == c).any():
            out[f"nll_{name}"] = float(nll[classes == c].mean())
    return out


def music_stats(pieces: list[tok.Notes]) -> dict:
    """Coarse descriptive statistics; useful for spotting degenerate output, not quality."""
    def per_piece(n: tok.Notes) -> dict:
        length = n.onset[-1] + n.duration[-1] - n.onset[0]
        onsets = np.unique(np.round(n.onset / tok.STEP_SECONDS))
        pcs = np.bincount(n.pitch % 12, minlength=12) / len(n)
        return {"notes_per_second": len(n) / length,
                "notes_per_onset": len(n) / len(onsets),
                "pitch_mean": n.pitch.mean(), "pitch_std": n.pitch.std(),
                "pitch_range": int(n.pitch.max() - n.pitch.min()),
                "velocity_mean": n.velocity.mean(), "velocity_std": n.velocity.std(),
                "duration_median_s": float(np.median(n.duration)),
                "pitch_class_entropy_bits": float(-(pcs[pcs > 0] * np.log2(pcs[pcs > 0])).sum())}
    rows = [per_piece(p) for p in pieces if len(p) > 10]
    return {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}


def excerpts(corpus: Corpus, seconds: float, seed: int = 0) -> list[tok.Notes]:
    """One random excerpt of ``seconds`` per piece, so statistics such as pitch range
    compare like with like (whole pieces span more of the keyboard than 90 s does)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(len(corpus)):
        p = corpus.piece(i)
        start = p.onset[0] + rng.uniform(0, max(0.0, p.onset[-1] - p.onset[0] - seconds))
        w = p[(p.onset >= start) & (p.onset < start + seconds)]
        out.append(tok.Notes(w.onset - start, w.duration, w.pitch, w.velocity))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--samples", type=Path, help="directory of generated .mid files")
    parser.add_argument("--out", type=Path, help="write results as JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.checkpoint, device)
    corpus = Corpus.load(args.data_dir / f"{args.split}.npz")
    windows = eval_windows(corpus, model.cfg.seq_len)

    results = {"checkpoint": str(args.checkpoint), "step": ckpt["step"], "split": args.split,
               "pieces": len(corpus), "n_params": model.num_params(),
               "model": summarize(*model_nll(model, windows)),
               "bigram_baseline": summarize(*bigram_nll(Corpus.load(args.data_dir / "train.npz"),
                                                        windows))}
    print(f"{args.split} split: {len(corpus)} performances, {results['model']['tokens']:,} tokens")
    print(f"{'':>18} {'model':>10} {'bigram':>10}")
    model_res, bigram_res = results["model"], results["bigram_baseline"]
    for key in model_res:
        if key != "tokens":
            print(f"{key:>18} {model_res[key]:10.3f} {bigram_res[key]:10.3f}")

    if args.samples:
        generated = [tok.read_midi(p) for p in sorted(args.samples.glob("*.mid"))]
        seconds = float(np.median([n.onset[-1] + n.duration[-1] for n in generated]))
        reference = excerpts(corpus, seconds)
        key = f"{args.split}_excerpts"
        results["music_stats"] = {"generated": music_stats(generated),
                                  key: music_stats(reference), "n_generated": len(generated),
                                  "excerpt_seconds": round(seconds, 1)}
        print(f"\n{len(generated)} samples vs one random {seconds:.0f} s excerpt per "
              f"{args.split} performance")
        print(f"{'':>26} {'generated':>10} {args.split:>10}")
        stats = results["music_stats"]
        for name, value in stats["generated"].items():
            print(f"{name:>26} {value:10.3f} {stats[key][name]:10.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
