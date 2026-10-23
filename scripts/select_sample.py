"""Screen generated samples for degenerate output and rank the rest by how typical they are.

    uv run python scripts/select_sample.py outputs/candidates

This is how the showcase sample was chosen, without listening: a sample must be 80-100 s
long, have no 200-note stretch with pitch-class entropy below 2.3 bits (a loop or trill),
use at least 40 distinct keys, and contain no silence (no note sounding) over 3 s. Among the
samples that pass, the one whose texture statistics are closest (mean absolute z-score) to
the real MAESTRO test performances is listed first.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pianist import tokenizer as tok
from pianist.data import Corpus

FEATURES = ("notes_per_second", "pitch_mean", "pitch_std", "velocity_mean", "velocity_std")


def features(n: tok.Notes) -> dict:
    length = n.onset[-1] + n.duration[-1] - n.onset[0]
    return {"notes_per_second": len(n) / length, "pitch_mean": n.pitch.mean(),
            "pitch_std": n.pitch.std(), "velocity_mean": n.velocity.mean(),
            "velocity_std": n.velocity.std()}


def screen(n: tok.Notes) -> list[str]:
    problems = []
    length = n.onset[-1] + n.duration[-1]
    if not 80 <= length <= 100:
        problems.append(f"length {length:.0f} s")
    entropies = []
    for s in range(0, max(1, len(n) - 100), 200):
        pcs = np.bincount(n.pitch[s:s + 200] % 12, minlength=12) / len(n.pitch[s:s + 200])
        entropies.append(-(pcs[pcs > 0] * np.log2(pcs[pcs > 0])).sum())
    if min(entropies) < 2.3:
        problems.append(f"repetitive stretch ({min(entropies):.2f} bits)")
    if len(np.unique(n.pitch)) < 40:
        problems.append(f"{len(np.unique(n.pitch))} distinct keys")
    sounding_until = np.maximum.accumulate(n.onset + n.duration)
    gap = np.max(n.onset[1:] - sounding_until[:-1], initial=0)  # nothing sounding at all
    if gap > 3:
        problems.append(f"{gap:.1f} s silence")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    test = Corpus.load(args.data_dir / "test.npz")
    reference = np.array([[features(test.piece(i))[k] for k in FEATURES]
                          for i in range(len(test))])
    mean, std = reference.mean(0), reference.std(0)

    rows = []
    for path in sorted(args.samples.glob("*.mid")):
        notes = tok.read_midi(path)
        f = np.array([features(notes)[k] for k in FEATURES])
        seed = json.loads(path.with_suffix(".json").read_text())["seed"]
        rows.append((seed, path.name, screen(notes), float(np.abs((f - mean) / std).mean())))
    rows.sort(key=lambda r: (bool(r[2]), r[3]))
    for seed, name, problems, distance in rows:
        verdict = "pass" if not problems else "fail: " + ", ".join(problems)
        print(f"{name:<32} seed {seed:3d}  distance {distance:4.2f}  {verdict}")


if __name__ == "__main__":
    main()
