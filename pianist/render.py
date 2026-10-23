"""Render MIDI to audio with FluidSynth and a sampled grand piano.

Unlike splicing one recording per note, a synthesizer lets every note ring for its full
release (and the room reverb tail) while later notes start, so sustained chords and
pedalled passages sound the way they were played.

Requires the ``fluidsynth`` binary (``apt install fluidsynth``, ``brew install fluid-synth``
or ``conda install -c conda-forge fluidsynth``) and, for MP3 output, ``ffmpeg``. Set
``PIANIST_FLUIDSYNTH`` / ``PIANIST_SOUNDFONT`` to override where they are found.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

SOUNDFONT_URL = ("https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/"
                 "SalamanderGrandPiano-SF2-V3+20200602.tar.xz")
DEFAULT_SOUNDFONT = Path("data/soundfonts/SalamanderGrandPiano-SF2-V3+20200602/"
                         "SalamanderGrandPiano-V3+20200602.sf2")

# A medium concert-hall reverb; FluidSynth's defaults are a small, dry room.
FLUIDSYNTH_SETTINGS = {
    "synth.sample-rate": "44100",
    "synth.gain": "0.5",
    "synth.polyphony": "512",
    "synth.reverb.active": "1",
    "synth.reverb.room-size": "0.75",
    "synth.reverb.damp": "0.35",
    "synth.reverb.width": "0.9",
    "synth.reverb.level": "0.45",
    "synth.chorus.active": "0",
}


def download_soundfont(dest: Path = DEFAULT_SOUNDFONT) -> Path:
    """Fetch the Salamander Grand Piano SoundFont (CC BY 3.0, 296 MB download)."""
    if not dest.exists():
        print(f"Downloading Salamander Grand Piano to {dest.parent.parent} ...")
        dest.parent.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile() as archive:
            with urllib.request.urlopen(SOUNDFONT_URL) as response:
                shutil.copyfileobj(response, archive)
            archive.seek(0)
            with tarfile.open(fileobj=archive, mode="r:xz") as tar:
                tar.extractall(dest.parent.parent, filter="data")
    return dest


def _fluidsynth() -> str:
    path = os.environ.get("PIANIST_FLUIDSYNTH") or shutil.which("fluidsynth")
    if not path:
        raise RuntimeError("fluidsynth not found; install it or set PIANIST_FLUIDSYNTH")
    return path


def render(midi: str | Path, out: str | Path, soundfont: str | Path | None = None) -> Path:
    """Render ``midi`` to ``out`` (``.wav`` or ``.mp3``), loudness-normalized for MP3."""
    soundfont = Path(soundfont or os.environ.get("PIANIST_SOUNDFONT") or DEFAULT_SOUNDFONT)
    if not soundfont.exists():
        raise FileNotFoundError(f"{soundfont} missing; run `python -m pianist.render --download`")
    out = Path(out)
    wav = out if out.suffix == ".wav" else out.with_suffix(".tmp.wav")
    settings = [arg for k, v in FLUIDSYNTH_SETTINGS.items() for arg in ("-o", f"{k}={v}")]
    subprocess.run([_fluidsynth(), "-ni", "-q", *settings, "-F", str(wav), str(soundfont),
                    str(midi)], check=True, capture_output=True)
    if out.suffix == ".mp3":
        # Trim the silent tail after the last reverb has decayed (internal rests are kept),
        # then EBU R128 loudness normalization to -16 LUFS, a common streaming target.
        filters = ("areverse,silenceremove=start_periods=1:start_threshold=-70dB,areverse,"
                   "apad=pad_dur=0.5,loudnorm=I=-16:TP=-1.5:LRA=11")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-af", filters,
                        "-ar", "44100", "-codec:a", "libmp3lame", "-q:a", "2", str(out)],
                       check=True)
        wav.unlink()
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render MIDI files with FluidSynth.")
    parser.add_argument("midi", nargs="*", type=Path)
    parser.add_argument("--download", action="store_true", help="fetch the default SoundFont")
    parser.add_argument("--format", choices=("mp3", "wav"), default="mp3")
    args = parser.parse_args()
    if args.download:
        download_soundfont()
    for path in args.midi:
        print(render(path, path.with_suffix("." + args.format)))
