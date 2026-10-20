# song → sheet music

Isolate one part of a song (vocals, bass, guitar, piano, other), transcribe it, and read it
as sheet music in your browser: transposed for your instrument (Alto Sax E♭ by default),
with lyrics under the notes and playback alongside the original recording.

Runs locally only. It needs Python and a few hundred MB of models, so it can't be hosted
on a static site like GitHub Pages.

## Setup (once)
    brew install cairo ffmpeg yt-dlp
    python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

The GAME vocal model (~190 MB) and the instrument sounds download automatically the first
time they're needed.

## Use
    .venv/bin/python viewer.py      # opens http://localhost:8765

Click **+ New song**, paste a YouTube link (or choose an audio file), pick the part, and hit
**Transcribe**. Progress shows on the page and the sheet music opens when it's done (about a
minute). Playlist / radio parameters in YouTube links are stripped, so only the one video is
downloaded.

### In the viewer
- **Instrument** presets (Concert C, Alto Sax E♭, Tenor Sax B♭, Trumpet / Clarinet B♭),
  ±½ step and ±octave transpose buttons, alto-sax range warnings.
- **Detail**: *Simple* (the tune: one in-key note per syllable, 8th-note grid),
  *Balanced* (keeps clearly held note changes and runs, 16th grid), *Detailed* (everything detected).
- **Lyrics** on/off: every word starts its own note; notes within one word are slurred.
- **Clef**: treble (default) or bass.
- **Playback**: alto sax samples or a plain synth, with the current note and measure
  highlighted. Click any measure or note to jump there; speed 50–100%.
- **Hear** the original audio in sync: the isolated part, the backing track, or the full mix.
- **Download PDF** of exactly what's on screen.

### Command line
    .venv/bin/python song.py "https://www.youtube.com/watch?v=..."   # or a local .mp3/.wav
    .venv/bin/python song.py song.mp3 --part bass --bpm 96
    .venv/bin/python song.py "https://..." --audio-only              # just download the audio

Options: `--part vocals|bass|guitar|piano|other`, `--bpm N` (override tempo detection),
`--time 3/4`, `--poly` (allow chords for vocals/bass), `--audio-only`, `--no-open`.

Everything is written to `output/` (ignored by git): the downloaded audio, separated stems,
and per song `<part>.mid`, `.musicxml` (open in MuseScore to edit), `.pdf`, and a `.json`
the viewer rebuilds the score from.

Audio is decoded at full precision and lowered just enough to avoid clipping (YouTube's audio
often peaks above full scale once decoded).

## How it works
- **Separation**: [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs`, or
  `htdemucs_6s` for guitar/piano) splits out the part and its backing track.
- **Vocals** cross-reference several tools (see `song.transcribe_vocals_ensemble`):
  - **[GAME](https://github.com/openvpi/GAME)** (OpenVPI, MIT), a model trained on singing, decides
    note boundaries and pitches. It runs from the official ONNX release via `game.py`; the model is only
    accepted if its sha256 matches the pinned release digest. `game_slicer.py` is copied from GAME (MIT).
  - **CREPE** pitch tracking breaks ties when GAME lands between two semitones, and fills in
    words GAME skipped.
  - **Word timings** from the video's YouTube captions (when the song came from YouTube), with
    **[Whisper](https://github.com/openai/whisper)** (`small.en`) filling in words the captions
    missed. Words decide where notes start, which notes are slurred, and the lyrics. In songs
    with lyrics, wordless phrases much quieter than the singing (backing vocals or instrument
    bleed) are dropped.
  - If GAME can't be loaded, vocals fall back to CREPE-only transcription.
- **Other parts**: [Basic Pitch](https://github.com/spotify/basic-pitch).
- **Notation**: music21 (key detection, spelling, quantising) and
  [Verovio](https://www.verovio.org) (engraving); cairosvg + pypdf for the PDF.
- **Instrument sounds**: FluidR3 GM samples from
  [midi-js-soundfonts](https://github.com/gleitz/midi-js-soundfonts).

## Note on copyright
Transcriptions and lyrics of commercial songs are copyrighted. Keep what you make for your
own practice; `output/` is git-ignored so none of it gets committed.
