"""Dataset download, preprocessing and training/evaluation datasets.

``pianist-prepare`` converts the MAESTRO v3 MIDI files into one compact ``.npz`` per
official split (train / validation / test), so the splits used for training and
evaluation never mix and preprocessing runs only once. With ``--aria``, it also builds a
pretraining corpus from the Aria-MIDI transcriptions, excluding anything that overlaps
the MAESTRO validation or test compositions.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from pianist import tokenizer as tok

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
SPLITS = ("train", "validation", "test")


class Corpus:
    """Many pieces stored as concatenated compact note arrays plus piece boundaries.

    Arrays are kept as float32 / int8 (10 bytes per note) so that a pretraining corpus of
    100M+ notes fits in memory and is shared by data-loader workers; slices are widened to
    float64 / int64 on access.
    """

    def __init__(self, notes: tok.Notes, starts: np.ndarray, titles: list[str] | None = None):
        self.notes = tok.Notes(notes.onset.astype(np.float32, copy=False),
                               notes.duration.astype(np.float32, copy=False),
                               notes.pitch.astype(np.int8, copy=False),
                               notes.velocity.astype(np.int8, copy=False))
        self.starts = np.asarray(starts, dtype=np.int64)  # piece i: notes[starts[i]:starts[i+1]]
        self.titles = titles or [""] * (len(starts) - 1)

    def __len__(self) -> int:
        return len(self.starts) - 1

    @property
    def num_notes(self) -> int:
        return int(self.starts[-1])

    def slice(self, a: int, b: int) -> tok.Notes:
        n = self.notes
        return tok.Notes(n.onset[a:b].astype(np.float64), n.duration[a:b].astype(np.float64),
                         n.pitch[a:b].astype(np.int64), n.velocity[a:b].astype(np.int64))

    def piece(self, i: int) -> tok.Notes:
        return self.slice(self.starts[i], self.starts[i + 1])

    def piece_of(self, note_index: int) -> int:
        return int(np.searchsorted(self.starts, note_index, side="right") - 1)

    @classmethod
    def from_pieces(cls, pieces: list[tok.Notes], titles: list[str] | None = None) -> Corpus:
        starts = np.concatenate([[0], np.cumsum([len(p) for p in pieces])])
        cat = tok.Notes(*(np.concatenate([getattr(p, f) for p in pieces])
                          for f in ("onset", "duration", "pitch", "velocity")))
        return cls(cat, starts, titles)

    @classmethod
    def concat(cls, corpora: list[Corpus]) -> Corpus:
        offsets = np.cumsum([0] + [c.num_notes for c in corpora[:-1]])
        starts = np.concatenate([[0]] + [c.starts[1:] + o for c, o in zip(corpora, offsets,
                                                                            strict=True)])
        cat = tok.Notes(*(np.concatenate([getattr(c.notes, f) for c in corpora])
                          for f in ("onset", "duration", "pitch", "velocity")))
        return cls(cat, starts, [t for c in corpora for t in c.titles])

    def save(self, path: Path) -> None:
        n = self.notes
        np.savez(path, onset=n.onset, duration=n.duration, pitch=n.pitch, velocity=n.velocity,
                 starts=self.starts, titles=np.array(self.titles))

    @classmethod
    def load(cls, path: Path) -> Corpus:
        z = np.load(path)
        notes = tok.Notes(z["onset"], z["duration"], z["pitch"], z["velocity"])
        return cls(notes, z["starts"], list(z["titles"]))


# --------------------------------------------------------------------------------------
# Datasets


def _augment(notes: tok.Notes, rng: np.random.Generator, max_transpose: int,
             max_stretch: float) -> tok.Notes:
    """Random key transposition (kept on the keyboard) and tempo stretch."""
    lo = max(-max_transpose, tok.PITCH_MIN - int(notes.pitch.min()))
    hi = min(max_transpose, tok.PITCH_MAX - int(notes.pitch.max()))
    shift = rng.integers(lo, hi + 1) if lo <= hi else 0
    stretch = np.exp(rng.uniform(-np.log1p(max_stretch), np.log1p(max_stretch)))
    return tok.Notes(notes.onset * stretch, notes.duration * stretch, notes.pitch + shift,
                     notes.velocity)


class RandomWindows(IterableDataset):
    """Endless stream of random, augmented ``seq_len + 1`` token windows for training.

    Windows start on a note boundary chosen uniformly over all notes in the split, so
    long pieces are sampled in proportion to their length. A window that starts at a
    piece's first note begins with BOS; one that runs past its last note ends with EOS
    and is padded.
    """

    def __init__(self, corpus: Corpus, seq_len: int, seed: int = 0, max_transpose: int = 5,
                 max_stretch: float = 0.05):
        self.corpus, self.seq_len, self.seed = corpus, seq_len, seed
        self.max_transpose, self.max_stretch = max_transpose, max_stretch

    def window(self, rng: np.random.Generator) -> np.ndarray:
        c = self.corpus
        start = int(rng.integers(c.num_notes))
        piece = c.piece_of(start)
        end = int(c.starts[piece + 1])
        is_first = bool(start == c.starts[piece])
        stop = min(end, start + self.seq_len // 2 + 1)  # >= 3 tokens per note, so this fills L+1
        notes = _augment(c.slice(start, stop), rng, self.max_transpose, self.max_stretch)
        tokens = tok.encode(notes, bos=is_first, eos=bool(stop == end))[: self.seq_len + 1]
        return np.pad(tokens, (0, self.seq_len + 1 - len(tokens)), constant_values=tok.PAD)

    def __iter__(self):
        info = get_worker_info()
        worker = info.id if info else 0
        rng = np.random.default_rng([self.seed, worker])
        while True:
            yield torch.from_numpy(self.window(rng))


def eval_windows(corpus: Corpus, seq_len: int, max_windows: int | None = None) -> torch.Tensor:
    """Deterministic evaluation set: each piece is tokenized whole (no augmentation) and cut
    into consecutive ``seq_len + 1`` windows that overlap by one token, so every target
    token is scored exactly once. The last window of each piece is padded."""
    windows = []
    for i in range(len(corpus)):
        tokens = tok.encode(corpus.piece(i))
        for s in range(0, len(tokens) - 1, seq_len):
            w = tokens[s:s + seq_len + 1]
            windows.append(np.pad(w, (0, seq_len + 1 - len(w)), constant_values=tok.PAD))
    windows = np.stack(windows)
    if max_windows is not None and len(windows) > max_windows:
        keep = np.random.default_rng(0).choice(len(windows), max_windows, replace=False)
        windows = windows[np.sort(keep)]
    return torch.from_numpy(windows)


# --------------------------------------------------------------------------------------
# Preparation CLI


def download_maestro(raw_dir: Path) -> Path:
    root = raw_dir / "maestro-v3.0.0"
    if not (root / "maestro-v3.0.0.csv").exists():
        print(f"Downloading MAESTRO v3 MIDI (58 MB) to {raw_dir} ...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MAESTRO_URL) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extractall(raw_dir)
    return root


def _stats(name: str, pieces: list[tok.Notes]) -> str:
    n_tokens = sum(len(tok.encode(p)) for p in pieces)
    hours = sum(p.onset[-1] + p.duration[-1] for p in pieces if len(p)) / 3600
    n_notes = sum(len(p) for p in pieces)
    return (f"{name:>10}: {len(pieces):6,d} performances, {hours:8,.1f} h, {n_notes:11,d} notes, "
            f"{n_tokens:13,d} tokens")


def prepare_maestro(raw_dir: Path, out_dir: Path, workers: int = 8) -> None:
    root = download_maestro(raw_dir)
    with open(root / "maestro-v3.0.0.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        for split in SPLITS:
            subset = [r for r in rows if r["split"] == split]
            pieces = list(pool.map(tok.read_midi, [root / r["midi_filename"] for r in subset],
                                   chunksize=8))
            titles = [f"{r['canonical_composer']} - {r['canonical_title']}" for r in subset]
            Corpus.from_pieces(pieces, titles).save(out_dir / f"{split}.npz")
            print(_stats(split, pieces))


# Aria-MIDI (Bradshaw & Colton, ICLR 2025): ~371k deduplicated solo-piano transcriptions.

ARIA_URL = ("https://huggingface.co/datasets/loubb/aria-midi/resolve/main/"
            "aria-midi-v1-deduped-ext.tar.gz")
NGRAM = 8
# Fraction of a file's pitch 8-grams found in the MAESTRO validation/test set above which it
# is dropped. Unrelated MAESTRO pieces never exceed 0.03; other performances of the same
# composition are usually well above 0.1 (see README).
MAX_OVERLAP = 0.04


def pitch_ngrams(notes: tok.Notes, n: int = NGRAM) -> np.ndarray:
    """Hashes of every run of ``n`` consecutive pitches. Onsets are grouped to 50 ms so that
    chord order is the same for a performance and its (slightly jittered) transcription."""
    order = np.lexsort((notes.pitch, np.round(notes.onset / 0.05)))
    p = notes.pitch[order].astype(np.int64)
    if len(p) < n:
        return np.zeros(0, np.int64)
    h = np.zeros(len(p) - n + 1, np.int64)
    for i in range(n):
        h = h * 131 + p[i:len(p) - n + 1 + i]
    return np.unique(h)


_reference: np.ndarray | None = None


def _init_reference(reference: np.ndarray) -> None:
    global _reference
    _reference = reference


def _read_and_check(path: Path) -> tuple[tok.Notes | None, float]:
    try:
        notes = tok.read_midi(path)
    except Exception:  # a handful of transcriptions are unreadable
        return None, 0.0
    grams = pitch_ngrams(notes)
    overlap = float(np.isin(grams, _reference, assume_unique=True).mean()) if len(grams) else 0.0
    # Return compact dtypes to keep inter-process transfer and parent memory small.
    return tok.Notes(notes.onset.astype(np.float32), notes.duration.astype(np.float32),
                     notes.pitch.astype(np.int8), notes.velocity.astype(np.int8)), overlap


def download_aria(raw_dir: Path) -> Path:
    root = raw_dir / "aria-midi" / "aria-midi-v1-deduped-ext"
    if not (root / "metadata.json").exists():
        import tarfile
        import tempfile

        print(f"Downloading Aria-MIDI deduped (2 GB) to {root.parent} ...")
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile() as archive:
            with urllib.request.urlopen(ARIA_URL) as response:
                while chunk := response.read(1 << 24):
                    archive.write(chunk)
            archive.seek(0)
            with tarfile.open(fileobj=archive, mode="r:gz") as tar:
                tar.extractall(root.parent, filter="data")
    return root


def prepare_aria(raw_dir: Path, out_dir: Path, workers: int = 8,
                 genres: tuple[str, ...] = ("classical",), min_audio_score: float = 0.95,
                 min_notes: int = 100) -> None:
    """Build ``aria.npz``: filtered by genre and transcription audio-quality score, with any
    file overlapping a MAESTRO validation/test composition removed."""
    root = download_aria(raw_dir)
    metadata = json.loads((root / "metadata.json").read_text())
    paths = {p.stem: p for p in (root / "data").glob("*/*.mid")}
    selected = []
    for file_id, entry in metadata.items():
        if entry["metadata"].get("genre") not in genres:
            continue
        for segment, score in entry["audio_scores"].items():
            path = paths.get(f"{file_id}_{segment}")
            if path is not None and score >= min_audio_score:
                selected.append(path)
    print(f"Aria-MIDI: {len(selected):,} of {len(paths):,} files match genre {genres} and "
          f"audio score >= {min_audio_score}")

    held_out = [Corpus.load(out_dir / f"{s}.npz") for s in ("validation", "test")]
    reference = np.unique(np.concatenate([pitch_ngrams(c.piece(i)) for c in held_out
                                          for i in range(len(c))]))
    pieces, titles, dropped_overlap, dropped_short = [], [], 0, 0
    with ProcessPoolExecutor(workers, initializer=_init_reference,
                             initargs=(reference,)) as pool:
        for path, (notes, overlap) in zip(selected, pool.map(_read_and_check, selected,
                                                             chunksize=64), strict=True):
            if notes is None or len(notes) < min_notes:
                dropped_short += 1
            elif overlap > MAX_OVERLAP:
                dropped_overlap += 1
            else:
                pieces.append(notes)
                titles.append(path.stem)
    print(f"dropped {dropped_overlap:,} overlapping MAESTRO validation/test compositions and "
          f"{dropped_short:,} short or unreadable files")
    Corpus.from_pieces(pieces, titles).save(out_dir / "aria.npz")
    print(_stats("aria", pieces))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--aria", action="store_true",
                        help="also build the Aria-MIDI pretraining corpus (2 GB download)")
    parser.add_argument("--aria-genres", default="classical",
                        help="comma-separated Aria-MIDI genres to keep")
    parser.add_argument("--aria-min-score", type=float, default=0.95)
    args = parser.parse_args()
    if not all((args.out_dir / f"{s}.npz").exists() for s in SPLITS):
        prepare_maestro(args.raw_dir, args.out_dir, args.workers)
    if args.aria:
        prepare_aria(args.raw_dir, args.out_dir, args.workers,
                     tuple(args.aria_genres.split(",")), args.aria_min_score)


if __name__ == "__main__":
    main()
