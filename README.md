# song → sheet music

Isolate one part of a song (vocals, bass, guitar, piano, other), transcribe it, and read it
as sheet music in your browser: transposed for your instrument (Alto Sax E♭ by default),
with lyrics under the notes and playback alongside the original recording.

Or just **split** a song into its parts (vocals, drums, bass, guitar, piano, the rest) and
mix any combination: karaoke tracks, drumless play-alongs, isolated vocals.

Runs locally only. It needs Python and a few hundred MB of models, so it can't be hosted
on a static site like GitHub Pages.

## Setup (once)
    brew install cairo ffmpeg yt-dlp
    python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

The GAME vocal model (~190 MB), the karaoke separation model (~50 MB, only if you add
harmonies) and the instrument sounds download automatically the first time they're needed.

## Use
    .venv/bin/python viewer.py      # opens http://localhost:8765

Click **New song**, paste a link (or choose a file), pick the part, and hit **Transcribe**.
Progress shows on the page and the sheet music opens when it's done (about a minute).

**Links**: YouTube / YouTube Music, SoundCloud, Bandcamp, Dailymotion, Internet Archive and
Mixcloud download directly (via [yt-dlp](https://github.com/yt-dlp/yt-dlp)). Spotify, Apple
Music, Deezer and TIDAL don't allow downloads, so their song links are looked up by title and
artist and the same song (matched by length) is fetched from YouTube. Playlist, album, radio
and "playing from" parts of links are stripped or refused, so only one song is ever downloaded.

**Files**: any audio or video format ffmpeg reads: MP3, AAC/M4A, ALAC, FLAC, WAV (any bit
depth/rate), AIFF, Ogg, Opus, WMA, AC3, CAF, WavPack and more, or the soundtrack of a video
(MP4, MOV, MKV, WebM, AVI...).

### Split
Switch to **Split** at the top left. Pick a song, choose **4 parts** (vocals, drums, bass,
other - best quality) or **6 parts** (adds guitar and piano), optionally split the vocals
into lead and backing, and hit **Split this song**. Then turn parts on/off, set their volumes,
solo one, or use a preset (No vocals, Vocals only, No drums, Drums and bass, Piano and
drums...), listen, and **Download this mix** as MP3, M4A, FLAC or WAV.

### In the viewer
- **Instrument** presets (Concert C, Alto Sax E♭, Tenor Sax B♭, Trumpet / Clarinet B♭),
  ±½ step and ±octave transpose buttons, alto-sax range warnings.
- **Detail**, based on how sure the transcription is of each note (vocals):
  - *Balanced* (the default): every note it's sure of, exactly as sung, with runs, 16ths,
    triplets and grace notes. Unsure notes are left out, unless one carries a word.
  - *Simple*: just the tune, one in-key note per syllable on an 8th-note grid.
  - *Detailed*: everything it heard, including backing-vocal bleed and notes whose pitch it's
    unsure of, which are drawn in grey.
  A note's confidence comes from CREPE: whether the voice is clearly heard there, whether it
  agrees with GAME on the pitch and holds it steadily, and the note's length and loudness.
  Beats sung in triplets (8th-note or quarter-note triplets) are written as triplets; in Simple
  and Balanced only when at least 3 notes fill the triplet. Swung 8ths are written straight,
  and no tuplet bigger than a triplet is ever used.
  In Balanced and Detailed, clear scoops (the voice sliding 2-4 semitones into a held note)
  are written as small slashed grace notes, so they don't read as melody notes or runs.
- **Lyrics** on/off (View menu): every word starts its own note; notes within one word are slurred.
- **Harmony** (vocals): click **Add** to separate the backing vocals from the lead and write
  their harmony as a second voice, stems down, in the bars where it sings (about a minute).
  Then toggle it on and off.
- **Clef**: treble (default) or bass.
- **Look** (in the **View** menu): *Classic* (engraved) or *Jazz*, hand-copied in the spirit of
  a jazz fake book: notes in Steinberg's Petaluma music font (bundled with Verovio) and words in
  its matching handwriting, Petaluma Script, drawn as if by pen. Staff lines, stems, barlines and
  beams become ink strokes with ragged edges and changing pressure; stems lean and overshoot
  their beams; noteheads are slanted, stretched blobs and no two glyphs are quite the same
  shape; on screen a photocopy-grain texture speckles every edge. The variation is generated
  (seeded by position, so a page always looks the same), and the PDF gets all of it except
  the grain.
- **Bars per line**: *Auto* fits as many bars as the music needs; pick a number (− / +) for
  that many bars on every line, all the same width so barlines line up in columns down the page.
  If the busiest bar can't fit, the whole page is drawn smaller rather than squeezed.
- **Zoom**: ⌘+ / ⌘− / ⌘0 (Ctrl on Windows), pinch or Ctrl+scroll, or the Zoom buttons, 50–300%.
  Only the music zooms; zoomed in, the page scrolls sideways.
- **Playback**: alto sax samples or a plain synth, with the current note and measure
  highlighted. Click any measure or note to jump there; speed 50–100%.
- **Hear** the original audio in sync: the isolated part, the backing track, the full mix, or
  (after adding harmony) the lead vocal or backing vocals alone.
- **Download** exactly what's on screen as a **PDF**, as **MusicXML** (opens in MuseScore,
  Dorico, Finale or Sibelius, marked as a transposing part so it still plays at concert pitch
  there), or as **MIDI** (concert pitch, at the song's tempo).
- **Key spelling**: where a key can be written two ways (5 sharps or 7 flats, F♯ or G♭...), the
  status line offers the other one; the choice is remembered per instrument.
- **Lyrics** held over several notes get an extender line, as in printed vocal music.
- **Edit**: fix anything by hand. Click a note to select it (⇧-click for more), or click empty
  staff to add one there. Change pitch (↑ ↓, ⇧ for an octave, or type A–G), length (1–5, `.` dot,
  `T` triplet), the word under it (`L`, Tab to the next note), slurs (`S`), grace notes (`/`), or
  move it (⌥ ← →); delete notes, insert or delete bars, and change the key or time signature.
  Add a note right after the selected one with `+` (it fills a following rest, or splits the
  note). The key menu takes the key as written for your instrument. Undo and redo everything.
  Each detail level is edited separately. **Save** makes your version the one that's shown,
  played and downloaded; unsaved changes are kept until you save or discard them. **History**
  keeps every save: open any of them (or the original) and Save to restore it. **Revert to
  original** goes back to the transcription, which is always kept.
  The editor works from the keyboard alone, and a screen reader hears each selected note
  ("F sharp 4, dotted 8th, bar 5 beat 2, word: city").

### Command line
    .venv/bin/python song.py "https://www.youtube.com/watch?v=..."   # or a local .mp3/.wav
    .venv/bin/python song.py song.mp3 --part bass --bpm 96
    .venv/bin/python song.py "https://..." --audio-only              # just download the audio

Options: `--part vocals|bass|guitar|piano|other`, `--bpm N` (override tempo detection),
`--time 3/4`, `--poly` (allow chords for vocals/bass), `--harmony` (also transcribe the backing
vocals), `--audio-only`, `--split 4|6` (split into parts instead of transcribing; add
`--vocal-layers` for lead/backing), `--no-open`.

Everything is written to `output/` (ignored by git): the downloaded audio, separated parts and mixes,
and per song `<part>.mid`, `.musicxml` (open in MuseScore to edit), `.pdf`, and a `.json`
the viewer rebuilds the score from. `originals/` keeps the transcription of each detail level as
a note list, and `edits/` holds your saved edits (with every save in `edits/history/`).

Audio is decoded at full precision to 44.1 kHz stereo and lowered just enough to avoid
clipping (YouTube's audio often peaks above full scale once decoded).

## How it works
- **Separation**: [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs`, or
  `htdemucs_6s` for guitar/piano) splits out the part and its backing track.
- **Vocals** cross-reference several tools (see `song.transcribe_vocals_ensemble`):
  - **[GAME](https://github.com/openvpi/GAME)** (OpenVPI, MIT), a model trained on singing, decides
    note boundaries and pitches. It runs from the official ONNX release via `game.py`; the model is only
    accepted if its sha256 matches the pinned release digest. `game_slicer.py` is copied from GAME (MIT).
  - **CREPE** pitch tracking breaks ties when GAME lands between two semitones, fills in
    words GAME skipped, and spots scoops into notes for grace notes.
  - **Word timings** from the video's YouTube captions (when the song came from YouTube), with
    **[Whisper](https://github.com/openai/whisper)** (`small.en`) filling in words the captions
    missed. Words decide where notes start, which notes are slurred, and the lyrics. In songs
    with lyrics, wordless phrases much quieter than the singing (backing vocals or instrument
    bleed) are dropped.
  - If GAME can't be loaded, vocals fall back to CREPE-only transcription.
- **Harmony**: UVR's MDX-Net karaoke model (`UVR_MDXNET_KARA_2`, from the
  [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) model release)
  splits the vocals into lead and backing, run from its ONNX file by `karaoke.py` (numpy STFT +
  onnxruntime, following python-audio-separator's MDX-Net code, MIT). The model is only accepted
  if its sha256 matches the pinned digest. GAME then transcribes the backing track; notes that
  are too quiet (lead leaking through) or just double the lead are left out. The model weights'
  repository doesn't state a license.
- **Other parts**: [Basic Pitch](https://github.com/spotify/basic-pitch).
- **Notation**: music21 (key detection, spelling, quantising) and
  [Verovio](https://www.verovio.org) (engraving); cairosvg + pypdf for the PDF.
- **Jazz look**: [Petaluma](https://github.com/steinbergmedia/petaluma) by Steinberg (SIL Open
  Font License 1.1). `fonts/PetalumaScript.otf` is the unmodified file from that repository
  (commit 95d295f, sha256 `3b4be972…c32c8`), with its `OFL.txt` and `FONTLOG.txt`.
- **Instrument sounds**: FluidR3 GM samples from
  [midi-js-soundfonts](https://github.com/gleitz/midi-js-soundfonts).

## Note on copyright
Transcriptions and lyrics of commercial songs are copyrighted. Keep what you make for your
own practice; `output/` is git-ignored so none of it gets committed.
