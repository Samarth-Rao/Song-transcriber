#!/usr/bin/env python3
"""Turn a song (file or YouTube URL) into sheet music for one isolated part.

Pipeline: yt-dlp (optional) -> Demucs (stem separation) -> Basic Pitch (audio -> notes)
          -> quantize to a beat grid -> music21 (MusicXML) -> Verovio + cairosvg (PDF)

When done it opens a local viewer (viewer.py) with transpose buttons, e.g. for alto sax.
"""
import argparse
import contextlib
import io
import json
import logging
import re
import subprocess
import sys
import warnings
from pathlib import Path

import viewer

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent
PARTS = ["vocals", "bass", "other", "guitar", "piano"]
MONOPHONIC = {"vocals", "bass"}
# Rough pitch ranges (Hz) per part, to cut down on junk notes.
FREQ_RANGE = {"vocals": (80, 1100), "bass": (30, 400)}


class PipelineError(Exception):
    """A user-facing failure (bad link, no notes found...)."""


log_sink = None  # set by the viewer to collect progress messages for the web page


def log(msg):
    print(f"==> {msg}", flush=True)
    if log_sink is not None:
        log_sink.append(msg)


# ---------------------------------------------------------------- input / separation

YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def clean_url(url: str) -> str:
    """Reduce a YouTube link to just its video, dropping &list=, &start_radio=, &t=, etc.

    Pasted links from a mix or radio otherwise risk pulling the whole playlist.
    Non-YouTube links are returned unchanged (yt-dlp still gets --no-playlist).
    """
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in ("youtube.com", "music.youtube.com", "youtube-nocookie.com"):
        parts = parsed.path.strip("/").split("/")
        if parts[0] == "watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parts[0] in ("shorts", "embed", "live", "v") and len(parts) > 1:
            video_id = parts[1]
        else:
            video_id = ""
    else:
        return url.strip()
    if not YOUTUBE_ID.match(video_id):
        raise PipelineError("That link doesn't point to a single video (is it a playlist?). "
                            "Open the song itself on YouTube and copy that link.")
    return f"https://www.youtube.com/watch?v={video_id}"


def decode_clean(src: Path, dest: Path, headroom_db=1.0) -> Path:
    """Decode any audio file to a 16-bit WAV without clipping.

    Lossy formats (YouTube's Opus, MP3, AAC) often decode to peaks above full scale; a plain
    16-bit conversion chops those off, which sounds like clicks and crackle. Decode at full
    precision first and, if needed, turn the whole track down so its peak sits `headroom_db`
    below full scale.
    """
    import numpy as np
    import soundfile as sf

    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-f", "f32le", "-ac", "2", "-"],
                         capture_output=True)
    if out.returncode != 0:
        raise PipelineError(f"Couldn't decode {src.name}: {out.stderr.decode().strip()[-300:]}")
    rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                           "stream=sample_rate", "-of", "csv=p=0", str(src)],
                          capture_output=True, text=True).stdout.strip().splitlines()[0]
    x = np.frombuffer(out.stdout, dtype=np.float32).reshape(-1, 2)
    peak = float(np.abs(x).max()) or 1.0
    gain = min(1.0, 10 ** (-headroom_db / 20) / peak)
    if gain < 1.0:
        log(f"Lowered the level {-20 * np.log10(gain):.1f} dB so the decoded audio doesn't clip")
    sf.write(str(dest), x * gain, int(rate), subtype="PCM_16")
    return dest


def fetch_audio(source: str, out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    if re.match(r"https?://", source.strip()):
        source = clean_url(source)
        log(f"Downloading audio from {source}")
        tmpl = str(out_root / "%(title)s.%(ext)s")
        # Original audio stream (no conversion by yt-dlp - see decode_clean).
        result = subprocess.run(
            ["yt-dlp", "-f", "bestaudio", "--no-playlist",
             "--print", "after_move:filepath", "-o", tmpl, source],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise PipelineError(f"yt-dlp failed:\n{result.stderr.strip()}\n\n"
                                "If YouTube changed something, updating usually fixes it: brew upgrade yt-dlp")
        original = Path(result.stdout.strip().splitlines()[-1])
        audio = decode_clean(original, original.with_suffix(".wav"))
        original.unlink()
        audio.with_suffix(".url").write_text(source)   # lets us fetch its captions later
        return audio
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise PipelineError(f"File not found: {path}")
    # Work from a clean WAV copy; the original file is left untouched.
    dest = out_root / f"{path.stem}.wav"
    if not dest.exists() or dest.stat().st_mtime < path.stat().st_mtime:
        decode_clean(path, dest)
    return dest


def separate(audio: Path, part: str, out_dir: Path) -> Path:
    model = "htdemucs_6s" if part in ("guitar", "piano") else "htdemucs"
    stem = out_dir / "separated" / model / audio.stem / f"{part}.wav"
    if stem.exists():
        log(f"Using cached {part} stem")
        return stem
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"Separating '{part}' with Demucs ({model} on {device}) - this takes a minute")
    result = subprocess.run(
        [sys.executable, "-m", "demucs", "-n", model, "--two-stems", part,
         "-d", device, "-o", str(out_dir / "separated"), str(audio)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not stem.exists():
        raise PipelineError("Demucs failed:\n" + "\n".join(result.stderr.strip().splitlines()[-15:]))
    return stem


# ---------------------------------------------------------------- transcription

def transcribe(stem: Path, part: str, mono: bool, words=()):
    """Returns notes as (start_s, end_s, midi_pitch, strength, connected), sorted by start.

    `connected` means the note follows the previous one without a break in the sound
    (sung on one breath / a bend), which is notated as a slur and played legato.
    """
    if part == "vocals" and mono:
        try:
            import importlib.util
            import game
            if importlib.util.find_spec("onnxruntime") is None:
                raise ImportError("onnxruntime isn't installed")
            game.ensure_model()
        except Exception as e:   # can't install/load the model: fall back to CREPE only
            log(f"GAME unavailable ({e}); using CREPE pitch tracking instead")
            notes = transcribe_vocals(stem)
        else:
            notes = transcribe_vocals_ensemble(stem, words)
    else:
        notes = transcribe_basic_pitch(stem, part, mono)
    before = len(notes)
    notes = loudness_gate(stem, notes)
    if before - len(notes):
        log(f"Dropped {before - len(notes)} notes where the {part} track is near-silent")
    return notes


def transcribe_basic_pitch(stem: Path, part: str, mono: bool):
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    log("Transcribing notes with Basic Pitch")
    fmin, fmax = FREQ_RANGE.get(part, (None, None))
    with contextlib.redirect_stdout(io.StringIO()):  # basic-pitch prints debug tensor info
        _, _, events = predict(
            str(stem), ICASSP_2022_MODEL_PATH,
            onset_threshold=0.6 if mono else 0.5,
            frame_threshold=0.3,
            minimum_note_length=100 if mono else 80,
            minimum_frequency=fmin, maximum_frequency=fmax,
        )
    notes = sorted((float(s), float(e), int(p), float(a)) for s, e, p, a, *_ in events)
    if mono:
        notes = to_monophonic(notes)
    return [(s, e, p, a, False) for s, e, p, a in notes]


def crepe_pitch(stem: Path, hop=0.01):
    """CREPE f0 (Hz) and periodicity (voicing confidence) every `hop` seconds."""
    import librosa
    import torch
    import torchcrepe

    log("Tracking vocal pitch with CREPE")
    y, _ = librosa.load(str(stem), sr=16000, mono=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    f0, per = torchcrepe.predict(torch.tensor(y)[None], 16000, hop_length=int(16000 * hop),
                                 fmin=65, fmax=1000, model="full", batch_size=1024,
                                 device=device, return_periodicity=True)
    per = torchcrepe.filter.median(per, 3)[0].cpu().numpy()
    f0 = torchcrepe.filter.mean(f0, 3)[0].cpu().numpy()
    return y, f0, per


def whisper_words(stem: Path):
    """Words from Whisper as (start_s, end_s, syllables, text), cached next to the stem."""
    import re

    cache = stem.with_name(stem.stem + ".words.json")
    if cache.exists():
        cached = [tuple(w) for w in json.loads(cache.read_text())]
        if all(len(w) == 4 for w in cached):    # older caches kept timings only
            return cached
    import whisper

    log("Finding word timings with Whisper")
    model = whisper.load_model("small.en", device="cpu")   # whisper verifies the download's sha256
    result = model.transcribe(str(stem), word_timestamps=True, fp16=False, language="en",
                              condition_on_previous_text=False)
    words = []
    for seg in result["segments"]:
        for w in seg["words"]:
            text = w["word"].strip().strip('.,!?;:"')
            if not text:
                continue
            syllables = max(1, len(re.findall(r"[aeiouy]+", text.lower().rstrip("e") or text.lower())))
            words.append((float(w["start"]), float(w["end"]), syllables, text))
    cache.write_text(json.dumps(words))
    return words


def align_words(notes, words, dips=(), split=True, match_tol=0.3, edge=0.15, snap=0.15):
    """Pair Whisper's words with note starts; returns (notes, aligned words, splits made).

    Whisper's word times are typically ~0.1 s (up to ~0.3 s) off, so each word claims the
    nearest unclaimed note start within `match_tol` and its lyric goes exactly there. A word
    with no note start of its own that lands inside a held note (away from its edges) splits
    that note - at the nearest re-attack found in the audio (`dips`), if one is close.
    Aligned words are (start_s, end_s, text).
    """
    notes = [list(n) for n in notes]
    claimed, aligned, cuts = set(), [], {}
    for i, (ws, we, _, text) in enumerate(words):
        # Words stay in the order they're sung: a word's note starts after the previous
        # word's and before the next word is heard.
        after = aligned[-1][0] if aligned else -1.0
        before = words[i + 1][0] if i + 1 < len(words) else float("inf")
        free = [j for j, n in enumerate(notes) if j not in claimed
                and abs(n[0] - ws) <= match_tol and after < n[0] < before]
        if free:
            j = min(free, key=lambda j: abs(notes[j][0] - ws))
            claimed.add(j)
            aligned.append((notes[j][0], we, text))
            continue
        inside = next((j for j, n in enumerate(notes) if n[0] + edge < ws < n[1] - edge), None)
        if split and inside is not None:
            near = [d for d in dips if abs(d - ws) <= snap and notes[inside][0] + edge < d < notes[inside][1] - edge]
            cut = min(near, key=lambda d: abs(d - ws)) if near else ws
            cuts.setdefault(inside, []).append(cut)
            aligned.append((cut, we, text))
        else:
            aligned.append((max(ws, after + 0.01), we, text))
    out = []
    for j, (s, e, p, a, c) in enumerate(notes):
        for cut in sorted(cuts.get(j, [])):
            out.append([s, cut, p, a, c])
            s, c = cut, False   # the new word is a new syllable, not slurred
        out.append([s, e, p, a, c])
    return out, aligned, sum(len(v) for v in cuts.values())


def clean_word(text):
    """A caption word without speaker markers (">>"), tags ("[singing]") or end punctuation."""
    import re
    return re.sub(r"\[[^\]]*\]|>>", "", text).strip().strip('.,!?;:"')


def youtube_words(stem: Path, audio: Path):
    """Words from YouTube's captions for the video the audio came from (if any), as
    (start_s, end_s, syllables, text). Uses uploaded subtitles' timing when they carry
    per-word times, otherwise the auto-generated captions. Cached next to the stem."""
    import re
    import tempfile

    url_file = audio.with_suffix(".url")
    if not url_file.exists():
        return []
    cache = stem.with_name(stem.stem + ".youtube.json")
    if cache.exists():
        return [tuple(w) for w in json.loads(cache.read_text())]
    log("Fetching YouTube captions")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["yt-dlp", "-q", "--skip-download", "--write-auto-subs", "--write-subs",
                        "--sub-langs", "en-orig,en", "--sub-format", "json3",
                        "-o", f"{tmp}/caps", url_file.read_text().strip()],
                       capture_output=True, text=True)
        files = sorted(Path(tmp).glob("caps*.json3"), key=lambda f: "en-orig" not in f.name)
        if not files:
            cache.write_text("[]")
            return []
        caps = json.loads(files[0].read_text())
    starts = []
    for ev in caps.get("events", []):
        for seg in ev.get("segs") or []:
            text = clean_word(seg.get("utf8", ""))
            if text:   # [Music] and the like clean to nothing
                starts.append(((ev["tStartMs"] + seg.get("tOffsetMs", 0)) / 1000, text))
    words = []
    for i, (t, text) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else t + 0.5
        syllables = max(1, len(re.findall(r"[aeiouy]+", text.lower().rstrip("e") or text.lower())))
        words.append((t, min(end, t + 2.0), syllables, text))
    cache.write_text(json.dumps(words))
    return words


def song_words(stem: Path, audio: Path):
    """Word timings cross-checked from two recognisers.

    YouTube's captions (when the song came from YouTube) land closer to the sung note starts
    than Whisper's (measured on a test song: 63% vs 50% within 0.1 s), so they lead; Whisper
    fills in words YouTube missed (nothing of YouTube's within 0.3 s).
    """
    try:
        wh = whisper_words(stem)
    except Exception as e:
        log(f"Whisper unavailable ({e})")
        wh = []
    try:
        yt = youtube_words(stem, audio)
    except Exception as e:
        log(f"YouTube captions unavailable ({e})")
        yt = []
    yt = [(*w[:3], clean_word(w[3])) for w in yt]   # older caches kept tags like ">>"
    yt = [w for w in yt if w[3]]
    if not yt:
        return wh
    same = lambda a, b: a.lower().strip("'") == b.lower().strip("'")
    # Whisper's extra words: nothing of YouTube's within 0.3 s, and not the same word
    # YouTube already has a little further off.
    extra = [w for w in wh if all(abs(w[0] - y[0]) > 0.3 for y in yt)
             and not any(abs(w[0] - y[0]) < 1.0 and same(w[3], y[3]) for y in yt)]
    if extra:
        log(f"Whisper added {len(extra)} words the YouTube captions missed")
    return sorted(list(yt) + extra)


def transcribe_vocals_ensemble(stem: Path, words):
    """Vocal notes by cross-referencing three tools:

    - GAME (a model trained on singing) decides where notes start/end and their pitch;
    - CREPE's frame-level pitch breaks ties when GAME lands between two semitones;
    - word timings (YouTube captions / Whisper, see song_words) decide which notes belong to
      one word, so "connected" (a slur,
      and one syllable for the Simple/Balanced levels) means "same syllable of the same word",
      and a held note that a new word starts inside is split there.
    """
    import numpy as np
    import game

    log("Transcribing notes with GAME")
    raw = game.Game().transcribe(stem)
    y, f0, per = crepe_pitch(stem)

    notes = []
    for s, e, score in raw:
        pitch = round(score)
        frac = score - pitch
        if abs(frac) > 0.3:   # GAME is unsure between two semitones: ask CREPE
            a, b = int((s + 0.2 * (e - s)) / 0.01), int((e - 0.2 * (e - s)) / 0.01) + 1
            seg = f0[a:b][(per[a:b] > 0.45) & (f0[a:b] > 0)]
            if len(seg) >= 3:
                crepe_midi = float(np.median(69 + 12 * np.log2(seg / 440)))
                if abs(crepe_midi - score) < 1:
                    pitch = round(crepe_midi)
        notes.append([s, e, int(pitch), 1.0, False])

    # A held note with another word starting partway through is really two notes on the same
    # pitch: split it where that word starts (snapped to a re-attack in the audio).
    dips = [i * 0.01 for i in sorted(syllable_dips(y, len(f0), 160))]
    notes, _, cuts_made = align_words(notes, words, dips)
    if cuts_made:
        log(f"Split {cuts_made} held notes where a new word starts")

    # Which word does each note belong to (largest overlap, with Whisper's ~0.1 s slack)?
    def word_of(s, e):
        best, best_overlap = None, 0.0
        for i, (ws, we, *_) in enumerate(words):
            overlap = min(e, we + 0.1) - max(s, ws - 0.1)
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        return best
    owner = [word_of(n[0], n[1]) for n in notes]
    # Within a word, notes run together (slurred) - except that a word with several syllables
    # gets new syllables at its widest gaps / longest following notes.
    i = 0
    while i < len(notes):
        j = i
        while j + 1 < len(notes) and owner[j + 1] is not None and owner[j + 1] == owner[i] \
                and notes[j + 1][0] - notes[j][1] < 0.06:
            j += 1
        group = list(range(i, j + 1))
        if len(group) > 1:
            syllables = words[owner[i]][2]
            joins = group[1:]
            ranked = sorted(joins, key=lambda k: (notes[k][0] - notes[k - 1][1], notes[k][1] - notes[k][0]), reverse=True)
            splits = set(ranked[:syllables - 1])
            for k in joins:
                notes[k][4] = k not in splits
        i = j + 1
    return [tuple(n) for n in notes]


def transcribe_vocals(stem: Path, hop=0.01, confidence=0.45, smooth_frames=9, min_run=6,
                      min_note=0.09, gap_merge=0.05):
    """Melody from CREPE pitch tracking, which handles vibrato and slides better than Basic Pitch.

    Frames are smoothed with a ~90 ms median (removes vibrato flicker but keeps 16th notes),
    snapped to semitones, and split into notes wherever the pitch or voicing changes.
    """
    import numpy as np

    y, f0, per = crepe_pitch(stem, hop)

    voiced = (per > confidence) & (f0 > 0)
    midi = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0, 1) / 440), np.nan)
    k = smooth_frames // 2
    smoothed = np.array([np.nanmedian(midi[max(0, i - k): i + k + 1]) if v else np.nan
                         for i, v in enumerate(voiced)])
    q = np.where(voiced, np.round(smoothed), -1).astype(int)
    attacks = syllable_dips(y, len(q), int(16000 * hop))

    # Runs of equal semitone, split at new syllables; very short runs are glitches, so fold
    # them into the previous note (unless they start a syllable).
    runs = []
    for i, v in enumerate(q):
        if runs and runs[-1][0] == v and i not in attacks:
            runs[-1][2] = i + 1
        else:
            runs.append([v, i, i + 1, i in attacks])
    for i, r in enumerate(runs):
        prev = runs[i - 1] if i else None
        if r[0] >= 0 and r[2] - r[1] < min_run and prev and prev[0] >= 0 and not r[3]:
            r[0] = prev[0]
    notes = []
    for v, s, e, attack in runs:
        if v < 0:
            continue
        start, end = s * hop, e * hop
        connected = bool(notes) and start - notes[-1][1] < gap_merge and not attack
        if connected and notes[-1][2] == v:
            notes[-1] = (notes[-1][0], end, v, notes[-1][3], notes[-1][4])
        else:
            notes.append((start, end, int(v), float(per[s:e].mean()), connected))
    notes = absorb_short(notes, min_note)
    return fix_octave_jumps([n for n in notes if n[1] - n[0] >= min_note])


def syllable_dips(y, n_frames, hop, depth_before=5.0, depth_after=4.0, window=15,
                  rise=3.5, rise_window=10, onset_ratio=3.0):
    """Frame indices where a new syllable starts without the voice stopping.

    Two signatures, both from the vocal stem's loudness (frames of `hop` samples, ~10 ms):
    - a dip: voiced consonants (m, n, l, w, r...) and vowel-to-vowel word changes keep the
      pitch tracker voiced, but the level drops several dB and comes back;
    - a re-attack: the level jumps up sharply together with a strong onset, which is how a
      new word on the same pitch sounds even when the dip before it is shallow.
    """
    import librosa
    import numpy as np

    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=4 * hop, hop_length=hop)[0] + 1e-9)
    db = np.pad(db, (0, max(0, n_frames - len(db))), constant_values=db.min())[:n_frames]
    onset = librosa.onset.onset_strength(y=y, sr=16000, hop_length=hop)
    onset = np.pad(onset, (0, max(0, n_frames - len(onset))))[:n_frames]
    strong = onset_ratio * np.median(onset[db > np.percentile(db, 50)])
    dips = set()
    for i in range(window, n_frames - window):
        if db[i] > db[i - 4: i + 5].min():
            continue
        before, after = db[i - window: i].max() - db[i], db[i + 1: i + 1 + window].max() - db[i]
        if before >= depth_before and after >= depth_after:
            dips.add(i)
        elif db[i + 1: i + 1 + rise_window].max() - db[i] >= rise and onset[i: i + 7].max() >= strong:
            dips.add(i)
    # One attack can trip both tests a few frames apart; keep the first of any close pair.
    kept = []
    for i in sorted(dips):
        if not kept or i - kept[-1] >= window:
            kept.append(i)
    return set(kept)


def fix_octave_jumps(notes, window=4.0, max_dist=12):
    """Move notes more than an octave from the surrounding melody back by whole octaves.

    Pitch trackers sometimes lock onto a harmonic or a bass note under the voice and report
    the right note in the wrong octave, often for several notes in a row - so compare against
    the melody's median pitch over the surrounding seconds, not just the adjacent notes.
    """
    import numpy as np

    starts = np.array([n[0] for n in notes])
    pitches = np.array([n[2] for n in notes])
    durs = np.array([n[1] - n[0] for n in notes])
    fixed = []
    for s, e, p, *rest in notes:
        near = np.abs(starts - s) < window
        order = np.argsort(pitches[near])
        cum = np.cumsum(durs[near][order])
        ref = pitches[near][order][np.searchsorted(cum, cum[-1] / 2)]   # duration-weighted median
        if abs(p - ref) > max_dist:
            p = int(p + 12 * round((ref - p) / 12))
        fixed.append((s, e, p, *rest))
    return fixed


# ---------------------------------------------------------------- smoothing (detail levels)

DETAIL_LEVELS = ("simple", "balanced", "detailed")


def absorb_short(notes, min_len):
    """Fold notes shorter than `min_len` that are part of a connected phrase into a neighbour.

    These are mostly pitches passed through while bending: a scoop into a note joins the
    note it's heading to, a fall-off or a quick wobble joins the note before it. Notes that
    stand alone (separate syllables) are left as they are.
    """
    notes = [list(n) for n in notes]
    changed = True
    while changed:
        changed = False
        for i in sorted(range(len(notes)), key=lambda j: notes[j][1] - notes[j][0]):
            n = notes[i]
            if n[1] - n[0] >= min_len:
                break   # sorted by length: nothing shorter left
            prev = notes[i - 1] if i > 0 and n[4] else None
            nxt = notes[i + 1] if i + 1 < len(notes) and notes[i + 1][4] else None
            if not prev and not nxt:
                continue
            passing = prev and nxt and min(prev[2], nxt[2]) < n[2] < max(prev[2], nxt[2])
            if nxt and (not prev or passing or nxt[2] == n[2]):
                nxt[0], nxt[4] = n[0], n[4]      # scoop: the target note starts earlier
            else:
                prev[1] = n[1]                   # fall-off / wobble: previous note lasts longer
            del notes[i]
            changed = True
            break
    # Neighbouring connected notes that ended up on the same pitch become one note.
    merged = []
    for n in notes:
        if merged and n[4] and merged[-1][2] == n[2]:
            merged[-1][1] = n[1]
        else:
            merged.append(n)
    return [tuple(n) for n in merged]


def snap_to_key(notes, k, max_len):
    """Move out-of-key notes shorter than `max_len` to the neighbouring in-key semitone
    that's closest to the surrounding melody (bends pass through these chromatic pitches)."""
    scale = {p.pitchClass for p in k.getScale().getPitches()}
    out = []
    for i, n in enumerate(notes):
        s, e, p, *rest = n
        if p % 12 not in scale and e - s < max_len:
            around = [notes[j][2] for j in (i - 1, i + 1) if 0 <= j < len(notes)] or [p]
            target = sum(around) / len(around)
            options = [q for q in (p - 1, p + 1, p - 2, p + 2) if q % 12 in scale]
            p = min(options, key=lambda q: (abs(q - target), abs(q - p)))
        out.append((s, e, p, *rest))
    return out


def collapse_syllables(notes, k, bpm, hold_beats, snap_all, run_steps=False):
    """One note per sung syllable, the way a person would write the tune down.

    A syllable is a run of connected notes (no break in the voice). Pitches the singer
    passes through while scooping, wobbling or falling are folded into the syllable's main
    note - the pitch it spends the most time on. A pitch held for `hold_beats` or longer
    inside a syllable stays its own (slurred) note.

    A syllable that ends by moving 3+ semitones and landing on a pitch held for 100 ms is a
    run, not a fall-off: the landing note is kept, and with `run_steps` so are the in-key
    pitches it passes through on the way.
    """
    beat = 60.0 / bpm
    hold = hold_beats * beat
    syllables = []
    for n in notes:
        if syllables and n[4]:
            syllables[-1].append(n)
        else:
            syllables.append([n])

    out = []
    for syl in syllables:
        subs = snap_to_key(syl, k, max_len=float("inf") if snap_all else hold)
        runs = []                                   # consecutive same pitch -> one run
        for n in subs:
            if runs and runs[-1][2] == n[2]:
                runs[-1][1] = n[1]
            else:
                runs.append(list(n[:4]))
        kept = [i for i, r in enumerate(runs) if r[1] - r[0] >= hold]
        if not kept:
            totals = {}
            for r in runs:
                totals[r[2]] = totals.get(r[2], 0) + r[1] - r[0]
            main = max(totals, key=lambda p: (totals[p], p))
            kept = [max((i for i, r in enumerate(runs) if r[2] == main),
                        key=lambda i: runs[i][1] - runs[i][0])]
        last = kept[-1]
        if last + 1 < len(runs):
            land = runs[-1]
            if land[1] - land[0] >= 0.1 and abs(land[2] - runs[last][2]) >= 3:
                lo, hi = sorted((land[2], runs[last][2]))
                steps = [i for i in range(last + 1, len(runs) - 1)
                         if run_steps and lo < runs[i][2] < hi and runs[i][1] - runs[i][0] >= 0.05]
                kept = kept + steps + [len(runs) - 1]
        # Two kept notes on the same pitch with only a wobble between them are one note.
        merged = []
        for i in kept:
            if merged and runs[merged[-1]][2] == runs[i][2]:
                runs[merged[-1]][1] = runs[i][1]
            else:
                merged.append(i)
        kept = merged
        # Lead-ins belong to the first kept note, notes between two kept notes lead into the
        # next one, and anything after the last kept note is its fall-off.
        for j, ki in enumerate(kept):
            start = syl[0][0] if j == 0 else runs[kept[j - 1]][1]
            end = syl[-1][1] if j == len(kept) - 1 else runs[ki][1]
            out.append((start, end, runs[ki][2], runs[ki][3], syl[0][4] if j == 0 else True))
    return out


def simplify(notes, detail, k, bpm):
    """Apply a detail level to a monophonic line (see DETAIL_LEVELS)."""
    if detail == "simple":      # the tune: one in-key note per syllable
        notes = collapse_syllables(notes, k, bpm, hold_beats=2, snap_all=True)
    elif detail == "balanced":  # one note per syllable, but keep clearly held note changes
        notes = collapse_syllables(notes, k, bpm, hold_beats=0.75, snap_all=False, run_steps=True)
    return notes


def loudness_gate(stem: Path, notes, floor_db=35):
    """Drop notes where the stem is far quieter than its loud parts.

    Separation leaves faint bleed in a stem (e.g. the instrumental intro in vocals.wav), and
    the transcribers normalise level, so without this they turn that hiss into notes.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(str(stem), sr=22050, mono=True)
    hop = 220
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0] + 1e-9)
    loud = db > np.percentile(db, 95) - floor_db
    def audible(s, e):
        a, b = int(s * sr / hop), max(int(e * sr / hop), int(s * sr / hop) + 1)
        return loud[a:b].any()
    return [n for n in notes if audible(n[0], n[1])]


def notes_to_midi(notes, path: Path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for s, e, p, a, *_ in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(40 + 80 * min(max(a, 0), 1)), pitch=p, start=s, end=e))
    pm.instruments.append(inst)
    pm.write(str(path))


def to_monophonic(notes):
    """Keep one note at a time: quieter overlapping notes lose, outliers dropped, then trim and merge."""
    out = []
    for n in notes:
        if out and n[0] < out[-1][1] - 0.03:  # starts while the previous note still sounds
            # Quieter, or an octave+ leap mid-note (bleed from another part): ignore it.
            if n[3] <= out[-1][3] or abs(n[2] - out[-1][2]) > 12:
                continue
            if n[0] - out[-1][0] < 0.05:  # near-simultaneous onset: replace
                out[-1] = n
                continue
        out.append(n)
    # Drop isolated notes more than an octave away from both neighbours (bleed from other parts).
    def outlier(i):
        neighbours = [out[j][2] for j in (i - 1, i + 1) if 0 <= j < len(out)]
        return bool(neighbours) and all(abs(out[i][2] - q) > 12 for q in neighbours)
    out = [n for i, n in enumerate(out) if not outlier(i)]
    trimmed = []
    for i, (s, e, p, a) in enumerate(out):
        if i + 1 < len(out):
            e = min(e, out[i + 1][0])
        # Merge with previous note of the same pitch if the gap is tiny (vibrato split).
        if trimmed and trimmed[-1][2] == p and s - trimmed[-1][1] < 0.08:
            ps, _, _, pa = trimmed[-1]
            trimmed[-1] = (ps, e, p, max(pa, a))
        elif e - s >= 0.08:
            trimmed.append((s, e, p, a))
    return trimmed


def analyze_rhythm(audio: Path):
    """Rough tempo plus drum-hit times from the full mix.

    The grid is fitted to the drums rather than the transcribed notes: drums sit on the beat,
    while a singer's note starts wander around it.
    """
    import librosa
    import numpy as np

    log("Finding the beat")
    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    while bpm < 70:
        bpm *= 2
    while bpm > 160:
        bpm /= 2
    hits = librosa.onset.onset_detect(y=librosa.effects.percussive(y), sr=sr, units="time")
    # Where the music actually starts (skips silence/hiss at the top of a YouTube rip).
    db = 20 * np.log10(librosa.feature.rms(y=y, hop_length=512)[0] + 1e-9)
    start = float(librosa.frames_to_time(np.argmax(db > np.percentile(db, 95) - 30), sr=sr, hop_length=512))
    return bpm, hits, start


def fit_grid(onsets, bpm, search=True):
    """Fine-tune tempo and beat phase so note onsets land on 8th-note lines.

    Treats each onset's position within an 8th note as an angle; the tempo whose angles
    cluster tightest (largest mean resultant length) wins, and the mean angle gives the offset.
    """
    import numpy as np

    onsets = np.asarray(onsets, dtype=float)
    candidates = np.arange(bpm * 0.95, bpm * 1.05, 0.05) if search else [bpm]
    best = (-1.0, bpm, 0.0)
    for b in candidates:
        eighth = 30.0 / b
        z = np.exp(2j * np.pi * onsets / eighth).mean()
        if abs(z) > best[0]:
            best = (abs(z), b, (np.angle(z) / (2 * np.pi)) % 1 * eighth)
    _, bpm, offset = best
    return float(bpm), float(offset)


def quantize(notes, bpm, grid=0.25, mono=False):
    """Seconds -> quarter-note offsets snapped to `grid`.

    Notes may carry a lyric as a 6th field. In a single line, a worded note that would land
    on the same grid slot as the note before it moves to the next half-grid slot instead of
    merging into it, so each word keeps its own note.

    Returns {start: (dur, [(pitch, amp)], connected, lyric)}."""
    beat = 60.0 / bpm
    events = {}
    prev = None
    for s, e, p, a, *rest in sorted(notes, key=lambda n: n[0]):
        connected = bool(rest and rest[0])
        lyric = rest[1] if len(rest) > 1 else None
        qs = round(s / beat / grid) * grid
        qe = max(round(e / beat / grid) * grid, qs + grid)
        if mono and prev is not None and qs <= prev:
            if lyric and events[prev][3]:
                qs = prev + grid / 2
                qe = max(qe, qs + grid / 2)
            else:
                qs = prev
        dur, pitches, conn, lyr = events.get(qs, (0, [], connected, None))
        if p not in [x[0] for x in pitches]:
            pitches.append((p, a))
        events[qs] = (max(dur, qe - qs), pitches, conn, lyr or lyric)
        prev = qs
    # One voice per staff: cut each event off at the next onset, and fill gaps of a 16th
    # or less so the part reads legato instead of littered with tiny rests.
    starts = sorted(events)
    for i, st in enumerate(starts[:-1]):
        dur, pitches, conn, lyr = events[st]
        gap = starts[i + 1] - st
        events[st] = (gap if gap - dur <= grid else dur, pitches, conn, lyr)
    return events


def words_to_notes(notes, words, tol=0.25, min_len=0.08, reach=0.5):
    """Give every word its own note: each word claims the nearest unclaimed note onset within
    `tol` s, otherwise splits the note it starts inside (same pitch, new attack), otherwise
    takes the next note within `reach` s. A note starting a word is never slurred to the one
    before it. Returns notes with the word as a 6th field (None for a word's later notes)."""
    notes = sorted([list(n[:5]) + [None] for n in notes], key=lambda n: n[0])
    for ws, _, text in words:     # in sung order
        text = clean_word(text)
        if not text:
            continue
        last = max((i for i, n in enumerate(notes) if n[5] is not None), default=-1)
        free = [i for i in range(last + 1, len(notes)) if abs(notes[i][0] - ws) <= tol]
        inside = next((i for i in range(max(last, 0), len(notes))
                       if notes[i][0] + min_len <= ws <= notes[i][1] - min_len), None)
        if free:
            i = min(free, key=lambda i: abs(notes[i][0] - ws))
        elif inside is not None:
            n = notes[inside]
            notes.insert(inside + 1, [ws, n[1], n[2], n[3], False, None])
            n[1] = ws
            i = inside + 1
        else:
            i = next((i for i in range(last + 1, len(notes)) if 0 <= notes[i][0] - ws <= reach), None)
            if i is None:
                if last >= 0 and ws - notes[last][1] < reach:   # too close to split: share
                    notes[last][5] += " " + text
                continue
        notes[i][4] = False
        notes[i][5] = text
    return [tuple(n) for n in notes]


# ---------------------------------------------------------------- notation / rendering

def build_score(notes, part, mono, bpm, title, time_sig, detail="simple", words=None, lyrics=True):
    """Notes (seconds from bar 1) -> music21 Score at the given detail level.

    `words` - optional [(start_s, end_s, text)] - each start a new note (see words_to_notes),
    and with `lyrics` are written under them."""
    from music21 import chord, clef, key, layout, metadata, meter, note, spanner, stream

    # Key estimate from all notes, weighted by duration.
    probe = stream.Stream()
    for s, e, p, *_ in notes:
        n = note.Note(p)
        n.quarterLength = max(round((e - s) * bpm / 60 * 4) / 4, 0.25)
        probe.append(n)
    k = probe.analyze("key") if notes else key.Key("C")

    if mono:
        if words:
            notes = words_to_notes(notes, words)
        texts = {n[0]: n[5] for n in notes if len(n) > 5 and n[5]}
        # Simplifying keeps each word's first note (a word always starts a new syllable).
        notes = [(*n[:5], texts.get(n[0])) for n in simplify([n[:5] for n in notes], detail, k, bpm)]
    staves = [notes] if mono else [
        [n for n in notes if n[2] >= 60],  # treble
        [n for n in notes if n[2] < 60],   # bass
    ]
    grid = 0.5 if detail == "simple" else 0.25

    scale_names = {p.pitchClass: p.name for p in k.getScale().getPitches()}

    def spell(midi_num):
        """Spell from the key: scale notes as the scale names them, others sharp in sharp
        keys and flat in flat keys. Marked as deliberate so transposing keeps the spelling
        (music21 re-guesses spellings it considers inferred, e.g. F# up a 6th -> Eb)."""
        from music21 import pitch as m21pitch
        name = scale_names.get(midi_num % 12)
        if name is None:
            name = m21pitch.Pitch(midi=midi_num).name
            p = m21pitch.Pitch(name)
            if k.sharps < 0 and p.accidental and p.accidental.name == "sharp":
                name = p.getEnharmonic().name
            elif k.sharps >= 0 and p.accidental and p.accidental.name == "flat":
                name = p.getEnharmonic().name
        p = m21pitch.Pitch(name)
        p.octave = None
        p.midi = midi_num   # keeps the name's letter, fixes the octave (handles Cb/B# edge cases)
        p.spellingIsInferred = False
        if p.accidental and p.accidental.name == "natural":
            p.accidental = None
        return p

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.movementName = title
    score.metadata.composer = f"{part} · {k.tonic.name} {k.mode} · {round(bpm)} BPM"
    parts = []
    for i, staff_notes in enumerate(staves):
        p = stream.Part()
        # Treble by default (the viewer can switch single-staff parts to bass clef).
        p.insert(0, clef.TrebleClef() if mono or i == 0 else clef.BassClef())
        p.insert(0, key.KeySignature(k.sharps))
        p.insert(0, meter.TimeSignature(time_sig))
        slurs, group, prev_end = [], [], None
        for start, (dur, pitches, connected, lyric) in sorted(quantize(staff_notes, bpm, grid, mono).items()):
            if mono:
                el = note.Note(spell(max(pitches, key=lambda x: x[1])[0]))
            else:
                el = chord.Chord([spell(x[0]) for x in sorted(pitches)])
            el.quarterLength = dur
            if lyric and lyrics:
                el.addLyric(lyric)
            p.insert(start, el)
            # Connected notes with no rest between them form a slur.
            if mono and connected and group and prev_end == start:
                group.append(el)
            else:
                if len(group) > 1:
                    slurs.append(group)
                group = [el]
            prev_end = start + dur
        if len(group) > 1:
            slurs.append(group)
        for g in slurs:
            p.insert(0, spanner.Slur(g))
        p.makeRests(fillGaps=True, inPlace=True)
        p.makeNotation(inPlace=True)
        parts.append(p)
        score.insert(0, p)
    if len(parts) > 1:
        score.insert(0, layout.StaffGroup(parts, symbol="brace"))
    return score


# ---------------------------------------------------------------- main

def process(source: str, part="vocals", bpm=None, time_sig="4/4", poly=False) -> Path:
    """Full pipeline. Returns the MusicXML path (a PDF at concert pitch is written next to it)."""
    audio = fetch_audio(source, ROOT / "output" / "_downloads")
    out_dir = ROOT / "output" / audio.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    mono = part in MONOPHONIC and not poly

    stem = separate(audio, part, out_dir)
    words = song_words(stem, audio) if part == "vocals" else []
    notes = transcribe(stem, part, mono, words)
    if not notes:
        raise PipelineError(f"No notes detected in the {part} part.")
    notes_to_midi(notes, out_dir / f"{part}.mid")

    rough_bpm, drum_hits, music_start = analyze_rhythm(audio)
    # Too few drum hits (e.g. a ballad with no drums): fall back to the notes themselves.
    onsets = drum_hits if len(drum_hits) >= 50 else [n[0] for n in notes]
    bpm, offset = fit_grid(onsets, bpm or rough_bpm, search=not bpm)
    # Bar 1 starts on the grid beat nearest to where the music starts. (Assumes the song
    # opens on a downbeat - true for most songs, not for ones that open with a pickup.)
    beat = 60.0 / bpm
    offset += round((music_start - offset) / beat) * beat
    raw_notes = notes   # still in audio time, for lining the lyrics up with them
    notes = [(max(s - offset, 0), max(e - offset, 0), *rest) for s, e, *rest in notes]

    # Raw notes + audio sync info, so the viewer can rebuild the score at any detail level
    # and play the original audio in time with it.
    info = {
        "part": part, "mono": mono, "bpm": bpm, "time_sig": time_sig, "title": audio.stem,
        "audio_offset": offset,   # seconds into the audio where bar 1 starts
        "audio": str(audio), "stem": str(stem), "backing": str(stem.with_name(f"no_{part}.wav")),
        "notes": [[float(n[0]), float(n[1]), int(n[2]), float(n[3]), bool(n[4])] for n in notes],
        "words": [[max(s - offset, 0), max(e - offset, 0), text]
                  for s, e, text in align_words(raw_notes, words, split=False)[1]] if words else [],
    }
    (out_dir / f"{part}.json").write_text(json.dumps(info))

    log("Building score")
    score = build_score(notes, part, mono, bpm, audio.stem, time_sig, words=info["words"], lyrics=False)
    k = score.recurse().getElementsByClass("KeySignature").first()
    log(f"Tempo {bpm:.0f} BPM, key signature: {viewer.key_name(k.sharps) if k else '?'}")
    xml = out_dir / f"{part}.musicxml"
    score.write("musicxml", fp=str(xml))
    log("Rendering PDF")
    (out_dir / f"{part}.pdf").write_bytes(viewer.render_pdf(xml.read_bytes()))
    log("Done")
    return xml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="audio file path or YouTube URL")
    ap.add_argument("-p", "--part", choices=PARTS, default="vocals", help="which part to transcribe (default: vocals)")
    ap.add_argument("--bpm", type=float, help="tempo override (auto-detected if omitted)")
    ap.add_argument("--time", default="4/4", help="time signature (default: 4/4)")
    ap.add_argument("--poly", action="store_true", help="allow chords even for vocals/bass")
    ap.add_argument("--audio-only", action="store_true", help="just download/extract the audio and stop")
    ap.add_argument("--no-open", action="store_true", help="don't open the viewer when done")
    args = ap.parse_args()

    try:
        if args.audio_only:
            print(f"\nAudio saved to {fetch_audio(args.source, ROOT / 'output' / '_downloads')}")
            return
        xml = process(args.source, args.part, args.bpm, args.time, args.poly)
    except PipelineError as e:
        sys.exit(str(e))
    print(f"\nDone. Files in {xml.parent}")
    if not args.no_open:
        viewer.serve(xml)


if __name__ == "__main__":
    main()
