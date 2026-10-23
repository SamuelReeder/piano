"""Compose new pieces with a trained checkpoint.

    pianist-generate checkpoints/finetune/best.pt --num 4 --render mp3
    pianist-generate checkpoints/finetune/best.pt --prompt chopin.mid --prompt-seconds 10

Each sample is written as ``<out-dir>/<name>-<seed>.mid`` (and ``.mp3`` / ``.wav`` with
``--render``). Seeds make samples reproducible on the same hardware and software.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pianist import tokenizer as tok
from pianist.model import load_checkpoint
from pianist.sampling import generate


def cut_at_pause(notes: tok.Notes, max_seconds: float, window: float = 0.15) -> tok.Notes:
    """Drop everything from the onset that follows the longest gap in
    ``[(1 - window) * max_seconds, max_seconds]``, so the piece ends between phrases rather
    than mid-gesture. Notes before the cut keep their full length and ring out."""
    if len(notes) == 0 or notes.onset[-1] < max_seconds:
        return notes
    onsets = np.unique(notes.onset)
    gaps = np.diff(onsets)
    candidates = np.flatnonzero((onsets[1:] >= (1 - window) * max_seconds)
                                & (onsets[1:] <= max_seconds))
    if len(candidates) == 0:
        return notes[notes.onset < max_seconds]
    cut = onsets[1:][candidates[np.argmax(gaps[candidates])]]
    return notes[notes.onset < cut]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--name", default="sample")
    parser.add_argument("--num", type=int, default=1, help="number of samples")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first sample")
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="end the piece at its longest pause in the last 15%% before this")
    parser.add_argument("--min-notes", type=int, default=200,
                        help="do not allow the piece to end before this many notes")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--prompt", type=Path, help="MIDI file whose opening is continued")
    parser.add_argument("--prompt-seconds", type=float, default=10.0)
    parser.add_argument("--render", choices=("mp3", "wav"), help="also render audio")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"loaded {args.checkpoint} (step {ckpt['step']}, val loss {ckpt['val_loss']:.4f})")

    prompt = None
    if args.prompt:
        notes = tok.read_midi(args.prompt)
        notes = notes[notes.onset < notes.onset[0] + args.prompt_seconds]
        prompt = tok.encode(notes, eos=False).tolist()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seed, args.seed + args.num):
        generator = torch.Generator(device).manual_seed(seed)
        tokens = generate(model, prompt, args.max_tokens, args.temperature, args.top_p,
                          args.min_notes, generator)
        notes = tok.decode(tokens)
        status = "ended" if tokens[-1] == tok.EOS else "hit max tokens"
        if args.max_seconds and len(notes) and notes.onset[-1] >= args.max_seconds:
            notes = cut_at_pause(notes, args.max_seconds)
            status = "cut at a pause"
        path = args.out_dir / f"{args.name}-{seed}.mid"
        tok.write_midi(notes, path)
        length = notes.onset[-1] + notes.duration[-1] if len(notes) else 0.0
        print(f"{path}: {len(notes)} notes, {length:.0f} s, {status}")
        meta = {"checkpoint": str(args.checkpoint), "step": ckpt["step"], "seed": seed,
                "temperature": args.temperature, "top_p": args.top_p,
                "max_tokens": args.max_tokens, "max_seconds": args.max_seconds,
                "min_notes": args.min_notes,
                "prompt": str(args.prompt) if args.prompt else None,
                "notes": len(notes), "seconds": round(float(length), 2)}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        if args.render:
            from pianist.render import render

            print(render(path, path.with_suffix("." + args.render)))


if __name__ == "__main__":
    main()
