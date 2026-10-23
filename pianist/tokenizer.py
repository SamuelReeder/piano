"""Note-level event tokenizer for expressive solo piano MIDI.

Each note becomes a short group of tokens::

    [SHIFT ...] PITCH VELOCITY DURATION

``SHIFT`` tokens advance the onset clock in 10 ms steps (one token covers up to
one second, longer gaps repeat it). Simultaneous notes share an onset, so
chords cost no shift tokens. Durations include the sustain pedal and are
quantized on a log scale, which keeps precision where the ear needs it (short
notes) without spending vocabulary on long ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np

STEP_SECONDS = 0.01
MAX_SHIFT_STEPS = 100
PITCH_MIN, PITCH_MAX = 21, 108  # the 88 keys of a piano, A0 to C8
NUM_VELOCITY_BINS = 32
NUM_DURATION_BINS = 64
MIN_DURATION_STEPS, MAX_DURATION_STEPS = 1, 1000  # 10 ms to 10 s

PAD, BOS, EOS = 0, 1, 2
SHIFT_OFFSET = 3
PITCH_OFFSET = SHIFT_OFFSET + MAX_SHIFT_STEPS
VELOCITY_OFFSET = PITCH_OFFSET + (PITCH_MAX - PITCH_MIN + 1)
DURATION_OFFSET = VELOCITY_OFFSET + NUM_VELOCITY_BINS
VOCAB_SIZE = DURATION_OFFSET + NUM_DURATION_BINS

# Geometric bin edges, in 10 ms steps; a duration maps to the bin whose log-centre is nearest.
_DURATION_CENTRES = np.geomspace(MIN_DURATION_STEPS, MAX_DURATION_STEPS, NUM_DURATION_BINS)
_DURATION_EDGES = np.sqrt(_DURATION_CENTRES[:-1] * _DURATION_CENTRES[1:])

# Token classes, used by the sampling grammar and per-class loss reporting.
CLASS_SPECIAL, CLASS_SHIFT, CLASS_PITCH, CLASS_VELOCITY, CLASS_DURATION = range(5)
CLASS_NAMES = ("special", "shift", "pitch", "velocity", "duration")


def token_class(tokens: np.ndarray) -> np.ndarray:
    """Map token ids to their class (see ``CLASS_*``)."""
    edges = np.array([SHIFT_OFFSET, PITCH_OFFSET, VELOCITY_OFFSET, DURATION_OFFSET])
    return np.searchsorted(edges, tokens, side="right")


@dataclass
class Notes:
    """A piece as parallel arrays, sorted by onset then pitch. Times are in seconds."""

    onset: np.ndarray
    duration: np.ndarray
    pitch: np.ndarray
    velocity: np.ndarray

    def __len__(self) -> int:
        return len(self.onset)

    def __getitem__(self, index) -> Notes:
        return Notes(self.onset[index], self.duration[index], self.pitch[index],
                     self.velocity[index])

    @classmethod
    def empty(cls) -> Notes:
        return cls(np.zeros(0), np.zeros(0), np.zeros(0, np.int64), np.zeros(0, np.int64))

    def sorted(self) -> Notes:
        return self[np.lexsort((self.pitch, self.onset))]


# --------------------------------------------------------------------------------------
# MIDI I/O


def read_midi(path: str | Path, sustain: bool = True) -> Notes:
    """Read every note from a MIDI file, optionally lengthening notes held by the sustain pedal.

    With ``sustain``, a note released while the pedal is down keeps sounding until the
    pedal lifts or the same key is struck again, which is what a listener actually hears.
    """
    now = 0.0
    pedal_down = False
    active: dict[int, list[tuple[float, int]]] = {}  # pitch -> stack of (onset, velocity)
    held: dict[int, list[int]] = {}  # pitch -> indices of key-released notes held by the pedal
    onset, offset, pitch, velocity = [], [], [], []

    def finish(index: int, time: float) -> None:
        offset[index] = max(time, onset[index] + STEP_SECONDS)

    for msg in mido.MidiFile(str(path)):  # iterating a MidiFile yields delta times in seconds
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            for index in held.pop(msg.note, []):  # re-striking a key damps its previous note
                finish(index, now)
            active.setdefault(msg.note, []).append((now, msg.velocity))
        elif msg.type in ("note_off", "note_on"):
            if not active.get(msg.note):
                continue
            start, vel = active[msg.note].pop(0)
            onset.append(start)
            offset.append(now)
            pitch.append(msg.note)
            velocity.append(vel)
            if sustain and pedal_down:
                held.setdefault(msg.note, []).append(len(onset) - 1)
        elif msg.type == "control_change" and msg.control == 64 and sustain:
            down = msg.value >= 64
            if pedal_down and not down:
                for indices in held.values():
                    for index in indices:
                        finish(index, now)
                held.clear()
            pedal_down = down

    for p, stack in active.items():  # notes never released end with the file
        for start, vel in stack:
            onset.append(start)
            offset.append(now)
            pitch.append(p)
            velocity.append(vel)
    for indices in held.values():
        for index in indices:
            finish(index, now)

    onset_arr = np.asarray(onset, dtype=np.float64)
    notes = Notes(
        onset=onset_arr,
        duration=np.maximum(np.asarray(offset, dtype=np.float64) - onset_arr, STEP_SECONDS),
        pitch=np.asarray(pitch, dtype=np.int64),
        velocity=np.asarray(velocity, dtype=np.int64),
    )
    keep = (notes.pitch >= PITCH_MIN) & (notes.pitch <= PITCH_MAX)
    return notes[keep].sorted()


def write_midi(notes: Notes, path: str | Path, tail_seconds: float = 3.0) -> None:
    """Write notes to a single-track piano MIDI file.

    A no-op controller event is placed ``tail_seconds`` after the last note ends so that
    renderers keep running long enough for the final notes to ring out.
    """
    ticks_per_beat, tempo = 480, 500_000  # 120 bpm, so one tick is 1/960 s
    to_ticks = ticks_per_beat * 1_000_000 / tempo
    # One MIDI channel cannot sound the same key twice, so a re-struck key damps the last note.
    duration = notes.duration.copy()
    order = np.lexsort((notes.onset, notes.pitch))
    same_key = notes.pitch[order[1:]] == notes.pitch[order[:-1]]
    gap = notes.onset[order[1:]] - notes.onset[order[:-1]]
    earlier = order[:-1][same_key]
    duration[earlier] = np.minimum(duration[earlier], np.maximum(gap[same_key], STEP_SECONDS))

    events = []
    for start, dur, p, v in zip(notes.onset, duration, notes.pitch, notes.velocity, strict=True):
        events.append((round(start * to_ticks), 1, int(p), int(v)))
        events.append((round((start + dur) * to_ticks), 0, int(p), 0))
    events.sort(key=lambda e: (e[0], e[1]))  # releases before attacks at the same tick
    end = max((e[0] for e in events), default=0) + round(tail_seconds * to_ticks)

    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(mido.Message("program_change", program=0, time=0))
    now = 0
    for tick, is_on, p, v in events:
        kind = "note_on" if is_on else "note_off"
        track.append(mido.Message(kind, note=p, velocity=v, time=tick - now))
        now = tick
    track.append(mido.Message("control_change", control=7, value=100, time=end - now))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    midi.tracks.append(track)
    midi.save(str(path))


# --------------------------------------------------------------------------------------
# Tokens


def encode(notes: Notes, bos: bool = True, eos: bool = True) -> np.ndarray:
    """Convert notes to a token sequence. Onsets are relative to the first note."""
    if len(notes) == 0:
        return np.array([BOS] * bos + [EOS] * eos, dtype=np.int64)
    notes = Notes(notes.onset.astype(np.float64), notes.duration.astype(np.float64),
                  notes.pitch.astype(np.int64), notes.velocity.astype(np.int64))
    steps = np.round(notes.onset / STEP_SECONDS).astype(np.int64)
    deltas = np.diff(steps, prepend=steps[0])
    pitch = PITCH_OFFSET + np.clip(notes.pitch, PITCH_MIN, PITCH_MAX) - PITCH_MIN
    velocity = VELOCITY_OFFSET + np.clip(notes.velocity, 1, 127) * NUM_VELOCITY_BINS // 128
    duration_bin = np.searchsorted(_DURATION_EDGES, notes.duration / STEP_SECONDS)
    duration = DURATION_OFFSET + duration_bin

    # Number of shift tokens per note, then the tokens themselves via one flat fill.
    n_shift = -(-deltas // MAX_SHIFT_STEPS)  # ceil division
    group = n_shift + 3
    starts = np.concatenate([[0], np.cumsum(group)[:-1]])
    out = np.empty(group.sum(), dtype=np.int64)
    out[:] = SHIFT_OFFSET + MAX_SHIFT_STEPS - 1  # full one-second shifts by default
    last_shift = starts + n_shift - 1
    has_shift = n_shift > 0
    remainder = deltas - (n_shift - 1) * MAX_SHIFT_STEPS
    out[last_shift[has_shift]] = SHIFT_OFFSET + remainder[has_shift] - 1
    out[starts + n_shift] = pitch
    out[starts + n_shift + 1] = velocity
    out[starts + n_shift + 2] = duration

    parts = [[BOS]] * bos + [out] + [[EOS]] * eos
    return np.concatenate(parts).astype(np.int64)


def decode(tokens: np.ndarray | list[int]) -> Notes:
    """Convert tokens back to notes, skipping any malformed note groups."""
    onset, duration, pitch, velocity = [], [], [], []
    now = 0
    tokens = list(tokens)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == EOS:
            break
        if SHIFT_OFFSET <= t < PITCH_OFFSET:
            now += t - SHIFT_OFFSET + 1
        elif PITCH_OFFSET <= t < VELOCITY_OFFSET and i + 2 < len(tokens):
            v, d = tokens[i + 1], tokens[i + 2]
            if VELOCITY_OFFSET <= v < DURATION_OFFSET and DURATION_OFFSET <= d < VOCAB_SIZE:
                onset.append(now * STEP_SECONDS)
                pitch.append(t - PITCH_OFFSET + PITCH_MIN)
                velocity.append(((v - VELOCITY_OFFSET) * 128 + 64) // NUM_VELOCITY_BINS)
                duration.append(_DURATION_CENTRES[d - DURATION_OFFSET] * STEP_SECONDS)
                i += 2
        i += 1
    if not onset:
        return Notes.empty()
    return Notes(np.array(onset), np.array(duration), np.array(pitch, dtype=np.int64),
                 np.array(velocity, dtype=np.int64))


def allowed_next(prev: int) -> np.ndarray:
    """Boolean mask over the vocabulary of tokens that may follow ``prev``.

    The grammar is ``BOS (SHIFT* PITCH VELOCITY DURATION)* EOS``, so sampling with this
    mask can never produce a malformed note.
    """
    mask = np.zeros(VOCAB_SIZE, dtype=bool)
    if PITCH_OFFSET <= prev < VELOCITY_OFFSET:
        mask[VELOCITY_OFFSET:DURATION_OFFSET] = True
    elif VELOCITY_OFFSET <= prev < DURATION_OFFSET:
        mask[DURATION_OFFSET:] = True
    elif prev == EOS:
        mask[EOS] = True
    else:  # BOS, SHIFT or DURATION: start (or keep timing) the next note, or end the piece
        mask[SHIFT_OFFSET:VELOCITY_OFFSET] = True
        mask[EOS] = prev != BOS
    return mask


def grammar_table() -> np.ndarray:
    """``table[prev]`` is ``allowed_next(prev)``; a (vocab, vocab) lookup for batched sampling."""
    return np.stack([allowed_next(t) for t in range(VOCAB_SIZE)])
