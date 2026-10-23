import mido
import numpy as np
import pytest

from pianist import tokenizer as tok


def make_notes(n=200, seed=0):
    rng = np.random.default_rng(seed)
    gaps = rng.choice([0.0, 0.0, 0.12, 0.31, 2.57], size=n)  # chords, runs and a long rest
    onset = np.cumsum(gaps)
    return tok.Notes(onset=onset, duration=rng.uniform(0.02, 6.0, n),
                     pitch=rng.integers(tok.PITCH_MIN, tok.PITCH_MAX + 1, n),
                     velocity=rng.integers(1, 128, n)).sorted()


def test_round_trip_is_close():
    notes = make_notes()
    decoded = tok.decode(tok.encode(notes))
    assert len(decoded) == len(notes)
    np.testing.assert_array_equal(decoded.pitch, notes.pitch)
    np.testing.assert_allclose(decoded.onset, notes.onset - notes.onset[0], atol=0.0101)
    assert np.abs(decoded.velocity - notes.velocity).max() <= 2
    assert np.abs(np.log(decoded.duration / notes.duration)).max() < 0.06  # log-scale bins


def test_encoding_is_stable_after_one_round_trip():
    once = tok.encode(tok.decode(tok.encode(make_notes())))
    twice = tok.encode(tok.decode(once))
    np.testing.assert_array_equal(once, twice)


def test_encoded_sequences_follow_the_grammar():
    tokens = tok.encode(make_notes())
    grammar = tok.grammar_table()
    assert tokens[0] == tok.BOS and tokens[-1] == tok.EOS
    assert all(grammar[a, b] for a, b in zip(tokens[:-1], tokens[1:], strict=True))
    assert tokens.max() < tok.VOCAB_SIZE


def test_long_gaps_use_repeated_shifts():
    notes = tok.Notes(np.array([0.0, 2.5]), np.array([0.5, 0.5]), np.array([60, 64]),
                      np.array([80, 80]))
    tokens = tok.encode(notes, bos=False, eos=False)
    shifts = tokens[(tokens >= tok.SHIFT_OFFSET) & (tokens < tok.PITCH_OFFSET)]
    assert list(shifts - tok.SHIFT_OFFSET + 1) == [100, 100, 50]
    assert tok.decode(tokens).onset[1] == pytest.approx(2.5)


def test_decode_skips_truncated_notes():
    tokens = tok.encode(make_notes(5), eos=False)
    assert len(tok.decode(tokens[:-1])) == 4  # last note lost its duration token


def test_midi_file_round_trip(tmp_path):
    notes = tok.decode(tok.encode(make_notes()))
    tok.write_midi(notes, tmp_path / "x.mid")
    back = tok.read_midi(tmp_path / "x.mid", sustain=False)
    assert len(back) == len(notes)
    np.testing.assert_allclose(back.onset, notes.onset, atol=2e-3)
    np.testing.assert_array_equal(back.pitch, notes.pitch)


def write_track(path, events):
    """events: (time_in_ticks, mido.Message); 480 ticks/beat at 120 bpm = 1 s per 960 ticks."""
    track, now = mido.MidiTrack(), 0
    for tick, msg in sorted(events, key=lambda e: e[0]):
        track.append(msg.copy(time=tick - now))
        now = tick
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(track)
    midi.save(path)


def test_sustain_pedal_extends_and_restrike_damps(tmp_path):
    on = lambda p: mido.Message("note_on", note=p, velocity=64)  # noqa: E731
    off = lambda p: mido.Message("note_off", note=p)  # noqa: E731
    pedal = lambda v: mido.Message("control_change", control=64, value=v)  # noqa: E731
    write_track(tmp_path / "p.mid", [
        (0, pedal(127)), (0, on(60)), (480, off(60)),  # held by the pedal...
        (960, on(64)), (1440, off(64)),                # ...as is this one...
        (1920, on(60)), (2400, off(60)),               # ...until 60 is struck again
        (2880, pedal(0)),
    ])
    notes = tok.read_midi(tmp_path / "p.mid")
    durations = dict(zip(zip(notes.pitch.tolist(), notes.onset.round(2).tolist(), strict=True),
                         notes.duration.round(2).tolist(), strict=True))
    assert durations[(60, 0.0)] == 2.0   # damped by the re-strike at 2 s
    assert durations[(64, 1.0)] == 2.0   # released with the pedal at 3 s
    assert durations[(60, 2.0)] == 1.0
    plain = tok.read_midi(tmp_path / "p.mid", sustain=False)
    assert plain.duration.round(2).tolist() == [0.5, 0.5, 0.5]


def test_cut_at_pause_ends_between_phrases():
    from pianist.generate import cut_at_pause

    onset = np.concatenate([np.arange(0, 86, 0.25), np.arange(88, 100, 0.25)])  # pause at 86-88
    n = len(onset)
    notes = tok.Notes(onset, np.full(n, 0.2), np.full(n, 60), np.full(n, 64))
    cut = cut_at_pause(notes, 95)
    assert cut.onset[-1] == 85.75
