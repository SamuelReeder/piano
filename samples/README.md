# Samples

All samples come from the final checkpoint (85M parameters, pretrained on Aria-MIDI and
fine-tuned on MAESTRO). Each has a MIDI file, an MP3 rendered with
[`pianist/render.py`](../pianist/render.py), and a `.json` file with every generation setting.
Settings: temperature 1.0, top-p 0.95, at most 5,000 tokens, at least 300 notes, ending at the
longest pause in the final 15% before 100 s.

| File | What it is |
| --- | --- |
| [`sample-15.mp3`](sample-15.mp3) | **Showcase.** Unconditional, seed 15, 88 s |
| [`sample-12.mp3`](sample-12.mp3) | Unconditional, seed 12 |
| [`sample-5.mp3`](sample-5.mp3) | Unconditional, seed 5 |
| [`nocturne-continuation-2.mp3`](nocturne-continuation-2.mp3) | The first 12 s are a real performance of Chopin's Nocturne Op. 15 No. 1 from the held-out MAESTRO test split. Everything after that is the model (seed 2 of 4). |

## How these were chosen

The samples were chosen by an automatic screen, not by listening.
[`candidates/`](candidates/) holds all 16 unconditional samples (seeds 0–15) as MIDI.
[`scripts/select_sample.py`](../scripts/select_sample.py) rejects a sample if any of these hold:

- it is not 80–100 s long
- a 200-note stretch falls below 2.3 bits of pitch-class entropy (a loop or trill)
- it uses fewer than 40 distinct keys
- it contains more than 3 s with no note sounding

It ranks the samples that pass by how close their texture statistics are to real
MAESTRO test performances. Three of the 16 passed:

```
sample-15.mid  seed  15  distance 1.00  pass
sample-12.mid  seed  12  distance 1.06  pass
sample-5.mid   seed   5  distance 1.13  pass
sample-6.mid   seed   6  distance 0.66  fail: 3.4 s silence
sample-8.mid   seed   8  distance 0.69  fail: repetitive stretch (1.53 bits)
sample-7.mid   seed   7  distance 0.98  fail: 3.1 s silence
sample-14.mid  seed  14  distance 1.02  fail: 4.0 s silence
sample-10.mid  seed  10  distance 1.06  fail: 27 distinct keys
sample-11.mid  seed  11  distance 1.33  fail: 25 distinct keys
sample-9.mid   seed   9  distance 1.53  fail: 3.1 s silence
sample-3.mid   seed   3  distance 1.56  fail: length 26 s, 37 distinct keys
sample-1.mid   seed   1  distance 1.60  fail: 12.3 s silence
sample-2.mid   seed   2  distance 1.78  fail: 29 distinct keys
sample-4.mid   seed   4  distance 1.81  fail: repetitive stretch (1.41 bits), 27 distinct keys
sample-0.mid   seed   0  distance 1.84  fail: repetitive stretch (1.58 bits), 10 distinct keys
sample-13.mid  seed  13  distance 2.16  fail: 31 distinct keys
```

Failing a rule does not always mean a sample sounds bad. A quiet piece in a narrow register
fails the distinct-keys rule, for example. Even so, the table is an honest picture of how
often the model goes wrong. For comparison, more than 3 s of silence occurs in only 4% of
random 90 s excerpts of real test performances.

To regenerate (identical on the same GPU and software versions):

```bash
uv run pianist-generate checkpoints/finetune/best.pt --num 16 --max-tokens 5000 \
    --max-seconds 100 --min-notes 300 --render mp3 --out-dir outputs/candidates
uv run python scripts/select_sample.py outputs/candidates
```
