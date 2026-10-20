# song → sheet music

Isolate one part of a song (vocals, bass, guitar, piano, other), transcribe it, and view it
as sheet music with transpose buttons (Alto Sax E♭ by default).

## Setup (once)
    brew install cairo ffmpeg yt-dlp
    python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

## Use
    .venv/bin/python viewer.py      # opens http://localhost:8765

Click **+ New song**, paste a YouTube link (or choose an audio file), pick the part, and hit
**Transcribe**. Progress shows on the page and the sheet music opens when it's done (about a minute).

Command line works too:

    .venv/bin/python song.py "https://www.youtube.com/watch?v=..."   # or a local .mp3/.wav
    .venv/bin/python song.py song.mp3 --part bass --bpm 96
    .venv/bin/python song.py "https://..." --audio-only              # just download the audio

Options: `--part vocals|bass|guitar|piano|other`, `--bpm N` (override tempo detection),
`--time 3/4`, `--poly` (allow chords for vocals/bass), `--no-open`.

Viewer: instrument presets, ±½ step and ±octave buttons (or arrow keys), alto-sax range
warnings, and **Download PDF** for the key you're looking at.
Outputs land in `output/<song>/`: separated stem `.wav`, `.mid`, `.musicxml` (open in MuseScore to edit), `.pdf`.

## How vocals are transcribed
Three tools are cross-referenced (see `song.transcribe_vocals_ensemble`):
- **[GAME](https://github.com/openvpi/GAME)** (OpenVPI, MIT) - a model trained on singing - decides note boundaries and pitches.
  Runs from the official ONNX release via `game.py`; the model downloads to `models/` on first use and is only
  accepted if its sha256 matches the pinned GitHub release digest. `game_slicer.py` is copied from GAME (MIT).
- **[Whisper](https://github.com/openai/whisper)** (`small.en`) - word timings decide which notes share a word (slurs, and
  how many notes a word keeps in Simple mode). Only timings and syllable counts are cached, not the text.
- **CREPE** - frame-level pitch breaks ties when GAME lands between two semitones.
If GAME can't be loaded, vocals fall back to CREPE-only transcription.
