#!/usr/bin/env python3
"""Turn a song (file or link) into sheet music for one isolated part, or split it into parts.

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

    has_audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                "stream=codec_type", "-of", "csv=p=0", str(src)],
                               capture_output=True, text=True).stdout.strip()
    if not has_audio:
        raise PipelineError(f"{src.name} has no audio track (or isn't an audio/video file).")
    # Any format ffmpeg reads (it reads nearly all of them), as 44.1 kHz stereo float.
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-f", "f32le",
                          "-ac", "2", "-ar", "44100", "-"], capture_output=True)
    if out.returncode != 0 or not out.stdout:
        raise PipelineError(f"Couldn't decode {src.name}: {out.stderr.decode().strip()[-300:]}")
    rate = 44100
    x = np.frombuffer(out.stdout, dtype=np.float32).reshape(-1, 2)
    peak = float(np.abs(x).max()) or 1.0
    gain = min(1.0, 10 ** (-headroom_db / 20) / peak)
    if gain < 1.0:
        log(f"Lowered the level {-20 * np.log10(gain):.1f} dB so the decoded audio doesn't clip")
    sf.write(str(dest), x * gain, rate, subtype="PCM_16")
    return dest


# Sites yt-dlp (the open-source downloader used here) downloads from reliably, by host -
# each checked working. Only links to these are accepted; anything else gets a clear message.
DOWNLOAD_SITES = {
    "youtube.com": "YouTube", "youtu.be": "YouTube", "music.youtube.com": "YouTube Music",
    "youtube-nocookie.com": "YouTube", "soundcloud.com": "SoundCloud", "on.soundcloud.com": "SoundCloud",
    "bandcamp.com": "Bandcamp", "dailymotion.com": "Dailymotion", "dai.ly": "Dailymotion",
    "archive.org": "Internet Archive", "mixcloud.com": "Mixcloud",
}
# Streaming services that don't allow downloads: their links are looked up by title and artist
# and the same song is fetched from YouTube instead.
LOOKUP_SITES = {"open.spotify.com": "Spotify", "music.apple.com": "Apple Music",
                "deezer.com": "Deezer", "tidal.com": "TIDAL", "listen.tidal.com": "TIDAL"}


def single_song_path(site: str, path: str, query: str) -> bool:
    """Whether a link's path is one song (not a playlist, album or profile) on that site."""
    seg = [p for p in path.split("/") if p]
    if site == "SoundCloud":     # /<user>/<track>; /<user>/sets/<set> is a playlist
        return len(seg) >= 2 and seg[1] not in ("sets", "albums", "tracks", "likes", "reposts")
    if site == "Bandcamp":
        return "track" in seg
    if site in ("Spotify", "Deezer", "TIDAL"):
        return "track" in seg
    if site == "Apple Music":
        return "song" in seg or ("album" in seg and "i=" in query)
    if site == "Mixcloud":       # /<user>/<mix>/
        return len(seg) >= 2 and seg[0] not in ("discover", "categories")
    return True


def site_of(host: str):
    """('download' | 'lookup', site name) for a link's host, or None."""
    host = host.lower().removeprefix("www.").removeprefix("m.")
    for table, kind in ((DOWNLOAD_SITES, "download"), (LOOKUP_SITES, "lookup")):
        for domain, name in table.items():
            if host == domain or host.endswith("." + domain):
                return kind, name
    return None


def supported_sites() -> str:
    names = list(dict.fromkeys([*DOWNLOAD_SITES.values(), *LOOKUP_SITES.values()]))
    return ", ".join(names)


def song_link(url: str):
    """Check a pasted link and reduce it to one song. Returns (kind, site, url)."""
    from urllib.parse import urlparse, urlunparse

    url = url.strip()
    parsed = urlparse(url)
    site = site_of(parsed.hostname or "")
    if not site:
        raise PipelineError(f"Links from {parsed.hostname or 'that site'} aren't supported. "
                            f"Try one from: {supported_sites()}. Or download the audio and choose the file.")
    kind, name = site
    if name.startswith("YouTube"):
        return kind, name, clean_url(url)
    if not single_song_path(name, parsed.path, parsed.query):
        raise PipelineError(f"That {name} link isn't a single song (is it a playlist, album or profile?). "
                            "Open the song itself and copy that link.")
    # Drop tracking / "playing from this playlist" parameters (SoundCloud's ?in=, ?si=, ...);
    # Apple Music's ?i= is the song within an album page, so it stays.
    query = parsed.query if name == "Apple Music" else ""
    return kind, name, urlunparse(parsed._replace(query=query, fragment=""))


def lookup_song(site: str, url: str):
    """(artist, title, seconds or None) for a streaming-service song link, read from its
    public page / API (no account needed)."""
    import html
    import urllib.request

    ua = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
    get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers=ua), timeout=20).read().decode("utf-8", "ignore")
    try:
        if site == "Spotify":
            m = re.search(r"/track/([A-Za-z0-9]+)", url)
            if m:
                page = get(f"https://open.spotify.com/embed/track/{m.group(1)}")
                data = json.loads(re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', page, re.S).group(1))
                ent = data["props"]["pageProps"]["state"]["data"]["entity"]
                return ", ".join(a["name"] for a in ent.get("artists", [])), ent["name"], (ent.get("duration") or 0) / 1000 or None
        elif site == "Deezer":
            m = re.search(r"/track/(\d+)", url)
            if m:
                t = json.loads(get(f"https://api.deezer.com/track/{m.group(1)}"))
                if "title" in t:
                    return t["artist"]["name"], t["title"], t.get("duration")
        else:   # Apple Music, TIDAL: "<title> by <artist> on ..." in the page's og:title / title
            page = get(url)
            meta = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]*)"', page)
            text = html.unescape(meta.group(1) if meta else re.search(r"<title[^>]*>(.*?)</title>", page, re.S).group(1))
            text = text.replace("\xa0", " ").strip("\u200e ")
            m = re.match(r"(.+?) by (.+?) on (Apple Music|TIDAL)", text) or re.match(r"(.+?) - Song.*? by (.+?) - Apple Music", text)
            if m:
                dur = re.search(r"Duration:? (\d+):(\d\d)", page)
                return m.group(2), m.group(1), int(dur.group(1)) * 60 + int(dur.group(2)) if dur else None
            m = re.match(r"(.+?) - (.+)", text)     # TIDAL's og:title is "Artist - Title"
            if m:
                return m.group(1), m.group(2), None
    except Exception as e:
        raise PipelineError(f"Couldn't read the song's details from {site} ({e}). "
                            "Paste a YouTube link for it instead.")
    raise PipelineError(f"Couldn't find a song on that {site} page. Copy the link of a single song.")


def youtube_match(artist: str, title: str, seconds=None) -> str:
    """URL of the YouTube video that best matches a song: of the top search results, the
    first whose length is within 10 s of the song's (studio recordings, not live versions)."""
    result = subprocess.run(["yt-dlp", "--flat-playlist", "-J", f"ytsearch5:{artist} - {title} audio"],
                            capture_output=True, text=True)
    try:
        entries = [e for e in json.loads(result.stdout).get("entries", []) if e.get("id")]
    except json.JSONDecodeError:
        entries = []
    if not entries:
        raise PipelineError(f"Couldn't find \"{title}\" by {artist} on YouTube. Paste a YouTube link instead.")
    if seconds:
        close = [e for e in entries if e.get("duration") and abs(e["duration"] - seconds) <= 10]
        entries = close or entries
    best = entries[0]
    log(f"Matched on YouTube: {best.get('title', best['id'])}")
    return f"https://www.youtube.com/watch?v={best['id']}"


def fetch_audio(source: str, out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    if re.match(r"https?://", source.strip()):
        kind, site, url = song_link(source)
        if kind == "lookup":
            artist, title, seconds = lookup_song(site, url)
            log(f"{site} doesn't allow downloads, so looking up \"{title}\" by {artist} on YouTube")
            url = youtube_match(artist, title, seconds)
        log(f"Downloading audio from {url}")
        tmpl = str(out_root / "%(title).150B.%(ext)s")
        # Original audio stream (no conversion by yt-dlp - see decode_clean). One item only,
        # whatever the link turns out to be.
        result = subprocess.run(
            ["yt-dlp", "-f", "bestaudio/best", "--no-playlist", "--playlist-items", "1", "--max-downloads", "1",
             "--print", "after_move:filepath", "-o", tmpl, url],
            capture_output=True, text=True,
        )
        # 101 = stopped at --max-downloads, i.e. success
        if result.returncode not in (0, 101) or not result.stdout.strip():
            raise PipelineError(f"yt-dlp couldn't download that link:\n{result.stderr.strip()[-600:]}\n\n"
                                "If the site changed something, updating usually fixes it: brew upgrade yt-dlp")
        original = Path(result.stdout.strip().splitlines()[-1])
        audio = decode_clean(original, original.with_suffix(".wav"))
        if original != audio:
            original.unlink()
        audio.with_suffix(".url").write_text(url)   # lets us fetch its captions later
        return audio
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise PipelineError(f"File not found: {path}")
    # Work from a clean WAV copy; the original file is left untouched.
    dest = out_root / f"{path.stem}.wav"
    if path == dest:
        return dest
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


# ---------------------------------------------------------------- splitting into parts

SPLIT_MODELS = {   # Demucs models: parts each one separates
    "htdemucs": ["vocals", "drums", "bass", "other"],
    "htdemucs_6s": ["vocals", "drums", "bass", "guitar", "piano", "other"],
}


def split_dir(audio: Path, model: str) -> Path:
    return ROOT / "output" / audio.stem / "separated" / model / audio.stem


def split_parts(audio: Path, model: str) -> list[str]:
    """Parts already separated for this song with this model (empty if not split yet)."""
    d = split_dir(audio, model)
    parts = [p for p in SPLIT_MODELS[model] if (d / f"{p}.wav").exists()]
    if len(parts) < len(SPLIT_MODELS[model]):
        return []
    return parts + [p for p in ("lead_vocals", "backing_vocals") if (d / f"{p}.wav").exists()]


def split_song(source: str, model="htdemucs", vocal_layers=False) -> Path:
    """Separate a song into all of a model's parts (and optionally the vocals into lead and
    backing). Returns the song's audio path."""
    import numpy as np
    import soundfile as sf

    audio = fetch_audio(source, ROOT / "output" / "_downloads")
    d = split_dir(audio, model)
    if not split_parts(audio, model):
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        n = len(SPLIT_MODELS[model])
        log(f"Splitting into {n} parts with Demucs ({model} on {device}) - this takes a minute or two")
        result = subprocess.run(
            [sys.executable, "-m", "demucs", "-n", model, "-d", device,
             "-o", str(d.parent.parent), str(audio)], capture_output=True, text=True)
        if result.returncode != 0 or not split_parts(audio, model):
            raise PipelineError("Demucs failed:\n" + "\n".join(result.stderr.strip().splitlines()[-15:]))
    if vocal_layers and not (d / "backing_vocals.wav").exists():
        import karaoke
        log("Splitting the vocals into lead and backing")
        lead, back = karaoke.split(d / "vocals.wav")
        for name, y in (("lead_vocals", lead), ("backing_vocals", back)):
            sf.write(d / f"{name}.wav", np.clip(y.T, -1, 1), karaoke.SR, subtype="PCM_16")
    log("Done")
    return audio


def mix_parts(audio: Path, model: str, gains: dict, dest: Path) -> Path:
    """Sum the chosen parts at the given gains (1 = as in the song) into a 16-bit WAV, turned
    down as a whole only if the sum would clip. Lead/backing vocals replace "vocals"."""
    import numpy as np
    import soundfile as sf

    d = split_dir(audio, model)
    mix = None
    for part, gain in gains.items():
        path = d / f"{part}.wav"
        if gain <= 0 or not path.exists():
            continue
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mix = y * gain if mix is None else mix[:len(y)] + y[:len(mix)] * gain
    if mix is None:
        y, sr = sf.read(str(d / f"{SPLIT_MODELS[model][0]}.wav"), dtype="float32", always_2d=True)
        mix = np.zeros_like(y)
    peak = float(np.abs(mix).max())
    if peak > 0.99:
        mix *= 0.99 / peak
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), mix, sr, subtype="PCM_16")
    return dest


# ---------------------------------------------------------------- transcription

def transcribe(stem: Path, part: str, mono: bool, words=()):
    """Returns (notes, scored): notes as (start_s, end_s, midi_pitch, strength, connected),
    sorted by start. `scored` means strength is a confidence (see note_confidence), not loudness.

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
            notes, scored = transcribe_vocals(stem), False
        else:
            notes, scored = transcribe_vocals_ensemble(stem, words), True
    else:
        notes, scored = transcribe_basic_pitch(stem, part, mono), False
    before = len(notes)
    notes = loudness_gate(stem, notes)
    if before - len(notes):
        log(f"Dropped {before - len(notes)} notes where the {part} track is near-silent")
    return notes, scored


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

    import numpy as np

    y, _ = librosa.load(str(stem), sr=16000, mono=True)
    # Cached next to the stem, so finding grace notes later doesn't need another pass.
    cache = stem.with_name(f"{stem.stem}.crepe{int(hop * 1000)}ms.npy")
    if cache.exists() and cache.stat().st_mtime >= stem.stat().st_mtime:
        f0, per = np.load(cache)
        return y, f0, per
    log("Tracking vocal pitch with CREPE")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    f0, per = torchcrepe.predict(torch.tensor(y)[None], 16000, hop_length=int(16000 * hop),
                                 fmin=65, fmax=1000, model="full", batch_size=1024,
                                 device=device, return_periodicity=True)
    per = torchcrepe.filter.median(per, 3)[0].cpu().numpy()
    f0 = torchcrepe.filter.mean(f0, 3)[0].cpu().numpy()
    np.save(cache, np.stack([f0, per]))
    return y, f0, per


def find_graces(notes, f0, per, hop=0.01, min_slide=0.06, max_slide=0.25, min_dev=1.5, max_dev=4.5):
    """Scoops into notes, as {note start: pitch the voice slid in from}.

    A scoop is a note whose pitch starts `min_dev`-`max_dev` semitones off (above or below) and
    takes `min_slide`-`max_slide` s to arrive, then is held at least that long. Shorter
    wobbles are just the attack. A slide that starts where the previous note left off (sung
    straight on from it) is a legato connection, not a grace note."""
    import numpy as np

    midi = np.where((per > 0.45) & (f0 > 0), 69 + 12 * np.log2(np.maximum(f0, 1) / 440), np.nan)
    graces = {}
    for i, (s, e, p, *_) in enumerate(notes):
        a = int(round(s / hop))
        look = min(int(max_slide / hop), int((e - s) / hop) // 2)
        dev = midi[a:a + look] - p
        arrived = np.where(np.abs(dev) < 0.5)[0]
        if not len(arrived):
            continue
        k = arrived[0]
        pre = dev[:k][~np.isnan(dev[:k])]
        if not min_slide / hop <= k or len(pre) < 4 or (e - s) - k * hop < k * hop:
            continue
        lead = float(np.median(pre[:max(3, len(pre) // 2)]))
        if not min_dev <= abs(lead) <= max_dev:
            continue
        from_pitch = p + round(lead)
        if i and s - notes[i - 1][1] < 0.08 and abs(notes[i - 1][2] - from_pitch) <= 1:
            continue
        graces[s] = from_pitch
    return graces


def add_graces(sidecar_path: Path):
    """Find scoops for an already-transcribed vocal part (see find_graces) and store them."""
    info = json.loads(sidecar_path.read_text())
    _, f0, per = crepe_pitch(Path(info["stem"]))
    off = info["audio_offset"]
    notes = [(s + off, e + off, p) for s, e, p, *_ in info["notes"]]
    info["graces"] = [[round(s - off, 4), p] for s, p in find_graces(notes, f0, per).items()]
    sidecar_path.write_text(json.dumps(info))
    return len(info["graces"])


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

    import librosa
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=1024, hop_length=160)[0] + 1e-9)
    notes = fill_missed_words(notes, words, f0, per)
    # Notes that are probably not the lead singer sit out the word alignment below and come
    # back at the end marked unsure, so only the Detailed level shows them.
    junk = wordless_junk(notes, words, db)
    unsure = [n for i, n in enumerate(notes) if i in junk]
    notes = [n for i, n in enumerate(notes) if i not in junk]
    if unsure:
        log(f"Marked {len(unsure)} notes with no words that are quieter than the singing as unsure")

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
    for n in unsure:
        n[4] = False
    junk = {id(n) for n in unsure}
    notes = sorted(notes + unsure)
    for n, c in zip(notes, note_confidence(notes, f0, per, db)):
        n[3] = min(c, 0.2) if id(n) in junk else c   # strength field = confidence
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

# Simple: the bare tune, one in-key note per syllable on 8ths.
# Balanced: every note we're sure of, as sung (runs, melismas, 16ths, grace notes).
# Detailed: everything transcribed, notes we're unsure of included (drawn grey).
DETAIL_LEVELS = ("simple", "balanced", "detailed")
SURE = 0.5   # note confidence (see note_confidence) below which a note only shows in Detailed
UNSURE_COLOR = "#8c95a6"


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


def fill_missed_words(notes, words, f0, per, hop=0.01, min_gap=0.3, voiced=0.5):
    """GAME sometimes marks a sung stretch as silence (seen: 2.7 s of a chorus line, which it
    did find when given a shorter clip). Where words fall in a gap between notes and CREPE
    hears a steady pitch, add one note per word at CREPE's median pitch."""
    import numpy as np

    if not words:
        return notes
    notes = sorted(notes)
    edges = [(-1.0, notes[0][0] if notes else float("inf"))]
    edges += [(a[1], b[0]) for a, b in zip(notes, notes[1:])]
    edges += [(notes[-1][1], float("inf"))] if notes else []
    added = []
    for gs, ge in edges:
        if ge - gs < min_gap:
            continue
        inside = [w for w in words if gs <= w[0] < ge]
        for i, (ws, we, *_) in enumerate(inside):
            end = min(inside[i + 1][0] if i + 1 < len(inside) else we, ge)
            a, b = int(max(ws, gs) / hop), int(end / hop)
            if b - a < 10:
                continue
            ok = (per[a:b] > 0.45) & (f0[a:b] > 0)
            if ok.mean() < voiced:
                continue
            first = a + int(np.argmax(ok))
            pitch = round(float(np.median(69 + 12 * np.log2(f0[a:b][ok] / 440))))
            added.append([first * hop, end, pitch, 1.0, False])
    if added:
        log(f"Added {len(added)} notes for words the note model missed")
    return sorted(notes + added)


def wordless_junk(notes, words, db, hop=0.01, phrase_gap=0.6, quieter=5.0):
    """Indices of notes that are probably not the lead singer: in a song with lyrics, phrases
    with no words that are clearly quieter than the singing (backing vocals or another
    instrument bleeding into the vocal track), and single notes over an octave from both
    neighbours (separation artefacts)."""
    import numpy as np

    if not words or not notes:
        return set()
    level = lambda s, e: float(np.median(db[int(s / hop):max(int(e / hop), int(s / hop) + 1)]))
    sung = np.median([level(n[0], n[1]) for n in notes])
    phrases, cur = [], [0]
    for i in range(1, len(notes)):
        if notes[i][0] - notes[cur[-1]][1] >= phrase_gap:
            phrases.append(cur)
            cur = [i]
        else:
            cur.append(i)
    phrases.append(cur)
    junk = set()
    for ph in phrases:
        s, e = notes[ph[0]][0], notes[ph[-1]][1]
        if not any(s - 0.3 <= w[0] <= e for w in words) and level(s, e) < sung - quieter:
            junk.update(ph)
            continue
        for j, i in enumerate(ph):
            near = [notes[ph[x]][2] for x in (j - 1, j + 1) if 0 <= x < len(ph)]
            if near and all(abs(notes[i][2] - q) > 12 for q in near):
                junk.add(i)
    return junk


def note_confidence(notes, f0, per, db, hop=0.01):
    """How sure we are of each note, 0-1, from what CREPE hears in its middle (edges are
    attacks and slides): is it sung (voiced), on this pitch (median within a third of a
    semitone), and held there (steady)? Very short notes and notes much quieter than the
    singing count for less."""
    import numpy as np

    midi = np.where((per > 0.45) & (f0 > 0), 69 + 12 * np.log2(np.maximum(f0, 1) / 440), np.nan)
    levels = [float(np.median(db[int(s / hop):max(int(e / hop), int(s / hop) + 1)])) for s, e, *_ in notes]
    sung = float(np.median(levels)) if levels else 0.0
    clip = lambda x: min(max(x, 0.0), 1.0)
    out = []
    for (s, e, p, *_), lvl in zip(notes, levels):
        a, b = int(s / hop), int(e / hop)
        trim = (b - a) // 6
        seg = midi[a + trim:max(b - trim, a + trim + 3)]
        heard = seg[~np.isnan(seg)]
        if len(heard) < 2:
            out.append(0.0)
            continue
        voiced = clip(len(heard) / len(seg) / 0.6)
        on_pitch = clip(1 - (abs(float(np.median(heard)) - p) - 0.3) / 0.5)
        steady = float(np.mean(np.abs(heard - p) < 0.75))
        length = clip((e - s - 0.06) / 0.09)
        loud = clip(1 - (sung - lvl - 3) / 9)
        out.append(round(min(voiced * on_pitch * steady, length, loud) ** 0.5, 3))
    return out


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


def sure_notes(notes, min_conf=SURE):
    """Drop the notes we're unsure of (strength field below `min_conf`) from a monophonic line
    of (start, end, pitch, confidence, connected, lyric, ...). An unsure note in the middle of
    a syllable is part of the note next to it, so that note takes over its time; one that
    carries a word stays, since the word needs a note."""
    out, pending = [], None
    for n in map(list, notes):
        if n[3] >= min_conf or n[5]:
            if pending and n[4] and n[0] - pending[1] < 0.06:   # it led into this note
                n[0] = pending[0]
            out.append(n)
            pending = None
        elif n[4] and out and n[0] - out[-1][1] < 0.06:       # it trailed the note before
            out[-1][1] = n[1]
        else:
            pending = n
    return [tuple(n) for n in out]


def simplify(notes, detail, k, bpm, scored=True):
    """Apply a detail level (see DETAIL_LEVELS) to a monophonic line of
    (start, end, pitch, confidence, connected, lyric, grace). Without confidence scores
    (`scored` False: the strength field is loudness), Balanced falls back to one note per
    syllable plus clearly held note changes and runs."""
    if detail == "detailed":
        return notes
    if scored:
        notes = sure_notes(notes)
    if detail == "balanced" and scored:
        # Short out-of-key notes are almost always a slide passing through, not a melody note.
        return snap_to_key(notes, k, max_len=0.75 * 60 / bpm)
    if detail == "balanced":
        texts = {n[0]: n[5] for n in notes if n[5]}
        return [(*n, texts.get(n[0]), None) for n in collapse_syllables(
            [n[:5] for n in notes], k, bpm, hold_beats=0.75, snap_all=False, run_steps=True)]
    texts = {n[0]: n[5] for n in notes if n[5]}
    # A word always starts a new syllable, so each word keeps its first note's start.
    return [(*n, texts.get(n[0]), None)
            for n in collapse_syllables([n[:5] for n in notes], k, bpm, hold_beats=2, snap_all=True)]


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


def triplet_beats(onsets, grid, min_notes=2, tol=0.07, margin=0.03):
    """Beats (whole quarter notes from bar 1) whose onsets sit on triplet thirds rather than
    on the straight `grid`. `onsets` are in quarter notes.

    A beat fits triplets when every onset inside it is within `tol` of a third and together they
    are at least `margin` closer to the thirds than to the straight grid. Written in triplets:
    beats that fit with onsets on both thirds; a beat on the last third followed by a beat on
    the first (a quarter-note triplet across both); and a beat with only a first-third onset if
    it fits with twice the margin (otherwise it's more likely a late 16th). A lone last-third
    onset is a swung 8th and stays straight. A triplet group also needs at least `min_notes`
    notes starting in it (counting one on its downbeat), so a lone triplet doesn't turn an
    easy rhythm into a hard one."""
    import math

    inside, on_beat = {}, set()
    for t in onsets:
        b = math.floor(t + 0.12)
        f = t - b
        if 0.15 < f < 0.85:
            inside.setdefault(b, []).append(f)
        elif f <= 0.15:
            on_beat.add(b)

    def fits(fs, m):
        es = sum(abs(f - round(f / grid) * grid) for f in fs)
        et = sum(abs(f - round(f * 3) / 3) for f in fs)
        return all(abs(f - round(f * 3) / 3) <= tol for f in fs) and et + m <= es

    def enough(*beats):
        return sum(len(inside.get(b, ())) for b in beats) + (beats[0] in on_beat) >= min_notes

    slots = {b: {round(f * 3) for f in fs} for b, fs in inside.items() if fits(fs, margin)}
    trip = set()
    for b, s in slots.items():
        if s == {1, 2} and enough(b):
            trip.add(b)
        elif s == {2} and slots.get(b + 1, set()) & {1} and enough(b, b + 1):
            trip |= {b, b + 1}
        elif s == {1} and fits(inside[b], 2 * margin) and enough(b):
            trip.add(b)
    return trip


def rhythm_pieces(start, dur, trip):
    """Cut a note wherever the beat changes between straight and triplet, so each piece is
    written in one kind of rhythm and tied to the next, instead of one note with an odd length
    that can only be written as a 12:7-style tuplet."""
    import math

    cuts = [b for b in range(math.floor(start) + 1, math.ceil(start + dur))
            if (b - 1 in trip) != (b in trip)]
    edges = [start, *cuts, start + dur]
    return [(x, y - x) for x, y in zip(edges, edges[1:])]


def quantize(notes, bpm, grid=0.25, mono=False, min_triplet=2):
    """Seconds -> quarter-note offsets (Fractions) snapped to `grid`, or to triplet thirds in
    beats that are sung in triplets (see triplet_beats; `min_triplet` is its min_notes).

    Notes may carry a lyric as a 6th field. In a single line, a worded note that would land
    on the same slot as the note before it moves to the next half-grid slot (next third in a
    triplet beat) instead of merging into it, so each word keeps its own note.

    A 7th field is a grace-note pitch, kept with the note it leads into.

    Returns ({start: (dur, [(pitch, amp)], connected, lyric, grace)}, triplet beats); split each
    note with rhythm_pieces before writing it."""
    import math
    from fractions import Fraction

    beat = 60.0 / bpm
    g = Fraction(grid).limit_denominator(64)
    third = Fraction(1, 3)
    # Triplet or straight is judged against 16ths (what's sung), even when writing on 8ths.
    trip = triplet_beats([n[0] / beat for n in notes], min(grid, 0.25), min_triplet)

    def step(t):
        return third if math.floor(t) in trip else g

    def snap(x):
        b = math.floor(x)
        if b in trip:
            return b + Fraction(round((x - b) * 3), 3)
        return b + round((x - b) / grid) * g

    events = {}
    prev = None
    for s, e, p, a, *rest in sorted(notes, key=lambda n: n[0]):
        connected = bool(rest and rest[0])
        lyric = rest[1] if len(rest) > 1 else None
        grace = rest[2] if len(rest) > 2 else None
        qs = snap(s / beat)
        qe = max(snap(e / beat), qs + step(qs))
        if mono and prev is not None and qs <= prev:
            if lyric and events[prev][3]:
                bump = third if math.floor(prev) in trip else g / 2
                qs = prev + bump
                qe = max(qe, qs + bump)
            else:
                qs = prev
        dur, pitches, conn, lyr, gr = events.get(qs, (0, [], connected, None, grace))
        if p not in [x[0] for x in pitches]:
            pitches.append((p, a))
        events[qs] = (max(dur, qe - qs), pitches, conn, lyr or lyric, gr)
        prev = qs
    # One voice per staff: cut each event off at the next onset, and fill gaps of a grid step
    # or less so the part reads legato instead of littered with tiny rests.
    starts = sorted(events)
    for i, st in enumerate(starts[:-1]):
        dur, pitches, conn, lyr, gr = events[st]
        gap = starts[i + 1] - st
        events[st] = (gap if gap - dur <= g else dur, pitches, conn, lyr, gr)
    return events, trip


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


def transcribe_harmony(stem: Path, lead_notes, min_len=0.1, level_db=-16.0):
    """Backing-vocal notes (harmonies under/over the lead), in audio time.

    The vocals stem is split into lead and backing vocals (karaoke.py), and GAME transcribes the
    backing track. Kept: notes no more than `level_db` below the whole vocals track (quieter
    is lead leaking through the split), that don't just double the lead in unison or octaves.
    """
    import librosa
    import numpy as np
    import soundfile as sf
    import game
    import karaoke

    lead_path, back_path = stem.with_name(f"lead_{stem.name}"), stem.with_name(f"backing_{stem.name}")
    if not back_path.exists():
        log("Separating the lead vocal from the backing vocals")
        lead, back = karaoke.split(stem)
        for path, y in ((lead_path, lead), (back_path, back)):
            sf.write(path, np.clip(y.T, -1, 1), karaoke.SR, subtype="PCM_16")
    log("Transcribing the backing vocals with GAME")
    raw = game.Game().transcribe(back_path)

    hop, sr = 220, 22050
    rms = lambda path: 20 * np.log10(librosa.feature.rms(
        y=librosa.load(str(path), sr=sr, mono=True)[0], frame_length=2048, hop_length=hop)[0] + 1e-9)
    back_db, all_db = rms(back_path), rms(stem)
    notes = []
    for s, e, score in raw:
        a, b = int(s * sr / hop), max(int(e * sr / hop), int(s * sr / hop) + 1)
        pitch = int(round(score))
        if e - s < min_len or np.median(back_db[a:b] - all_db[a:b]) < level_db:
            continue
        under = [n[2] for n in lead_notes if min(e, n[1]) - max(s, n[0]) > 0.5 * (e - s)]
        if under and all((pitch - q) % 12 == 0 for q in under):
            continue   # the lead, doubled
        notes.append((s, e, pitch, 1.0, False))
    log(f"Found {len(notes)} harmony notes")
    return notes, lead_path, back_path


def add_harmony(sidecar_path: Path):
    """Add backing-vocal harmony to an already transcribed vocals score (its .json sidecar)."""
    info = json.loads(sidecar_path.read_text())
    if info["part"] != "vocals":
        raise PipelineError("Harmonies can only be added to a vocals score.")
    offset = info["audio_offset"]
    lead = [(n[0] + offset, n[1] + offset, n[2]) for n in info["notes"]]
    notes, lead_path, back_path = transcribe_harmony(Path(info["stem"]), lead)
    info["harmony"] = [[max(s - offset, 0), max(e - offset, 0), p, a, c] for s, e, p, a, c in notes]
    info["lead_vocals"], info["backing_vocals"] = str(lead_path), str(back_path)
    sidecar_path.write_text(json.dumps(info))
    save_originals(sidecar_path)
    return sidecar_path


# ---------------------------------------------------------------- notation / rendering

def tied_pieces(make, start, dur, trip):
    """Elements for one note, cut by rhythm_pieces and tied together. `make()` builds each piece."""
    from music21 import tie

    pieces = rhythm_pieces(start, dur, trip)
    els = []
    for i, (s, d) in enumerate(pieces):
        el = make()
        el.quarterLength = d
        if len(pieces) > 1:
            el.tie = tie.Tie("start" if i == 0 else "stop" if i == len(pieces) - 1 else "continue")
        els.append((s, el))
    return els


def add_divisi(lead, harmony):
    """Where the harmony sings, split the lead's bars into two voices: lead stems up, harmony
    stems down (its rests hidden, so a part that only joins in briefly stays uncluttered)."""
    from music21 import note, stream

    for m1, m2 in zip(lead.getElementsByClass(stream.Measure), harmony.getElementsByClass(stream.Measure)):
        if not any(isinstance(n, note.Note) for n in m2.notesAndRests):
            continue
        v1, v2 = stream.Voice(id="1"), stream.Voice(id="2")
        for v, m, stem in ((v1, m1, "up"), (v2, m2, "down")):
            for el in list(m.notesAndRests):
                off = el.offset
                m.remove(el)
                if el.isRest:
                    if v is v2:
                        el.style.hideObjectOnPrint = True
                else:
                    el.stemDirection = stem
                for t in el.duration.tuplets:   # triplet brackets on the stem side
                    t.placement = "above" if stem == "up" else "below"
                v.insert(off, el)
        m1.insert(0, v1)
        m1.insert(0, v2)


# ---------------------------------------------------------------- scores as editable note lists
#
# A score is worked out in two steps: score_events turns the transcription into a "note list"
# (plain JSON: concert pitch, positions in quarter notes from bar 1), and events_score draws a
# note list with music21. The viewer edits note lists and saves them, so an edited score is
# drawn exactly like a transcribed one. Note list format:
#
#   {"version": 1, "part", "mono", "bpm", "title", "time_sig": "4/4", "key": ["D", "major"],
#    "length": "336" (quarter notes, rounded up to whole bars when drawn),
#    "staves": [[note, ...], ...] (one staff, or treble + bass for chords),
#    "harmony": [note, ...] (optional second voice for the first staff)}
#   note: {"q": "17/3" (start), "d": "1/3" (length), "p": [54] (MIDI pitches; several = a chord),
#          "lyric": "word", "slur": true (slurred from the note before), "grace": 52 (a grace
#          note's pitch), "unsure": true (drawn grey)} - all but q, d and p optional.

EVENTS_VERSION = 1


def _frac(x) -> str:
    from fractions import Fraction
    return str(Fraction(x).limit_denominator(960))


def detect_key(notes):
    """Key estimate from all notes, weighted by duration (seconds)."""
    from music21 import key, note, stream

    probe = stream.Stream()
    for s, e, p, *_ in notes:
        n = note.Note(p)
        n.quarterLength = max(round((e - s) * 4) / 4, 0.25)
        probe.append(n)
    return probe.analyze("key") if notes else key.Key("C")


def score_events(notes, part, mono, bpm, title, time_sig, detail="simple", words=None,
                 harmony=None, graces=None, scored=False) -> dict:
    """Notes (seconds from bar 1) -> a note list (see above) at the given detail level.

    `words` - optional [(start_s, end_s, text)] - each start a new note (see words_to_notes)
    and become lyrics. `harmony` - optional backing-vocal notes - becomes the second voice.
    `graces` - optional [(note start_s, pitch)] scoops (see find_graces) - become grace notes,
    except at the Simple level (just the tune). `scored`: the notes' strength field is a
    confidence (see note_confidence), which decides what Balanced leaves out and Detailed
    marks unsure."""
    import math
    from music21 import meter

    k = detect_key([(s * bpm / 60, e * bpm / 60, p) for s, e, p, *_ in notes])
    if mono:
        if words:
            notes = words_to_notes(notes, words)
        grace_of = {round(s, 4): p for s, p in graces or ()}
        notes = simplify([(*n[:5], n[5] if len(n) > 5 else None, grace_of.get(round(n[0], 4)))
                          for n in notes], detail, k, bpm, scored)
    staves = [notes] if mono else [
        [n for n in notes if n[2] >= 60],  # treble
        [n for n in notes if n[2] < 60],   # bass
    ]
    grid = 0.5 if detail == "simple" else 0.25
    # Outside Detailed, only write triplets where 3+ notes make the rhythm worth reading.
    min_triplet = 2 if detail == "detailed" else 3
    scale = {p.pitchClass for p in k.getScale().getPitches()}

    def in_key_grace(g, main):
        """The scale note nearest where the scoop starts, on the same side of the main note."""
        side = 1 if g > main else -1
        options = [x for x in (g, g - side, g + side) if x % 12 in scale and (x - main) * side > 0]
        return options[0] if options else g

    out_staves, end = [], 0
    for staff_notes in staves:
        evs = []
        quantized, _ = quantize(staff_notes, bpm, grid, mono, min_triplet)
        for start, (dur, pitches, connected, lyric, grace) in sorted(quantized.items()):
            p = [max(pitches, key=lambda x: x[1])[0]] if mono else sorted({x[0] for x in pitches})
            ev = {"q": _frac(start), "d": _frac(dur), "p": p}
            if lyric:
                ev["lyric"] = lyric
            if mono and connected:
                ev["slur"] = True
            if mono and grace is not None:
                ev["grace"] = in_key_grace(grace, p[0])
            if scored and mono and detail == "detailed" and max(a for _, a in pitches) < SURE:
                ev["unsure"] = True
            evs.append(ev)
            end = max(end, start + dur)
        out_staves.append(evs)
    bar = meter.TimeSignature(time_sig).barDuration.quarterLength
    length = max(math.ceil(end / bar), 1) * bar

    result = {"version": EVENTS_VERSION, "part": part, "mono": mono, "bpm": bpm, "title": title,
              "time_sig": time_sig, "key": [k.tonic.name, k.mode], "length": _frac(length),
              "staves": out_staves}
    if mono and harmony:
        quantized, _ = quantize(snap_to_key(sorted(harmony), k, max_len=float("inf")), bpm, grid,
                                mono=True, min_triplet=min_triplet)
        result["harmony"] = [{"q": _frac(s), "d": _frac(min(d, length - s)),
                              "p": [max(pitches, key=lambda x: x[1])[0]]}
                             for s, (d, pitches, *_) in sorted(quantized.items()) if s < length]
    return result


def events_triplets(events):
    """Beats that hold triplet rhythm: some note starts or ends on a third of a beat."""
    import math
    from fractions import Fraction

    trip = set()
    for ev in events:
        q, d = Fraction(ev["q"]), Fraction(ev["d"])
        for t in (q, q + d):
            if t.denominator % 3 == 0:
                trip.add(math.floor(t))
    return trip


def events_score(ev: dict, lyrics=True, harmony=True):
    """Draw a note list (see above) as a music21 Score. Every note gets the id "n<staff>-<i>"
    ("h-<i>" in the harmony voice; "-<k>" on later tied pieces, "-g" on its grace note), so the
    viewer can tell which listed note a drawn one is."""
    import math
    from fractions import Fraction
    from music21 import chord, clef, key, layout, metadata, meter, note, spanner, stream
    from music21 import pitch as m21pitch

    k = key.Key(*ev["key"])
    mono, time_sig = ev["mono"], ev["time_sig"]
    scale_names = {p.pitchClass: p.name for p in k.getScale().getPitches()}

    def spell(midi_num):
        """Spell from the key: scale notes as the scale names them, others sharp in sharp
        keys and flat in flat keys. Marked as deliberate so transposing keeps the spelling
        (music21 re-guesses spellings it considers inferred, e.g. F# up a 6th -> Eb)."""
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

    def grace_note(midi_num):
        n = note.Note(spell(midi_num)).getGrace()
        n.duration.slash = True   # a quick scoop (acciaccatura), not a melody note
        return n

    all_events = [e for staff in ev["staves"] for e in staff] + list(ev.get("harmony") or [])
    end = max([Fraction(e["q"]) + Fraction(e["d"]) for e in all_events] + [Fraction(ev["length"])])
    bar = Fraction(meter.TimeSignature(time_sig).barDuration.quarterLength)
    length = max(math.ceil(end / bar), 1) * bar

    def build(events, prefix, words=True, staff_clef=None):
        """One voice's notes as a Part, with ties where rhythm needs them, and slurs."""
        p = stream.Part()
        # Clef, key and time go in before the bars are made, so accidentals are worked out
        # against the key signature (otherwise every sharp in the key gets written out).
        p.insert(0, staff_clef or clef.TrebleClef())
        p.insert(0, key.KeySignature(k.sharps))
        p.insert(0, meter.TimeSignature(time_sig))
        trip = events_triplets(events)
        slurs, group, prev_end, drawn = [], [], None, []
        for i, e in enumerate(events):
            start, dur = Fraction(e["q"]), Fraction(e["d"])
            if dur <= 0 or not e["p"]:
                continue
            pitches = [spell(m) for m in sorted(e["p"])]
            make = (lambda: note.Note(pitches[0])) if len(pitches) == 1 else (lambda: chord.Chord(pitches))
            pieces = tied_pieces(make, start, min(dur, length - start), trip)
            if e.get("grace") is not None:
                g = grace_note(e["grace"])
                g.id = f"{prefix}-{i}-g"
                p.insert(start, g)
            for j, (s, el) in enumerate(pieces):
                el.id = f"{prefix}-{i}" + (f"-{j}" if j else "")
                if e.get("unsure"):
                    el.style.color = UNSURE_COLOR
                p.insert(s, el)
            if words and lyrics and e.get("lyric"):
                pieces[0][1].addLyric(e["lyric"])
                if e.get("unsure"):   # the word itself isn't in doubt
                    pieces[0][1].lyrics[0].style.color = "#000000"
            els = [el for _, el in pieces]
            drawn.append((e, els, start, start + dur))
            # Slurred notes with no rest between them form a slur.
            if e.get("slur") and group and prev_end == start:
                group.extend(els)
            else:
                if len(group) > 1:
                    slurs.append(group)
                group = els
            prev_end = start + dur
        if len(group) > 1:
            slurs.append(group)
        for g in slurs:
            p.insert(0, spanner.Slur(g))
        # A word sung over several slurred notes gets an extender line to its last note
        # (marked here; the viewer writes it into the MusicXML, which music21 can't).
        if words and lyrics:
            for first, (e, els, _, _) in enumerate(drawn):
                last = first
                while e.get("lyric") and last + 1 < len(drawn) and drawn[last + 1][0].get("slur") \
                        and not drawn[last + 1][0].get("lyric") and drawn[last + 1][2] == drawn[last][3]:
                    last += 1
                if last > first:
                    els[0].editorial.extend = "start"
                    drawn[last][1][-1].editorial.extend = "stop"
        if p.highestTime < length:
            p.insert(p.highestTime, note.Rest(quarterLength=length - p.highestTime))
        p.makeRests(fillGaps=True, inPlace=True)
        p.makeNotation(inPlace=True)
        # Lengths no single note can show (e.g. 5 16ths) become tied notes here rather than on
        # export, so every drawn note gets its own id. Halves keep their note's id (split
        # here or at a barline); number the later ones.
        p.splitAtDurations(recurse=True)
        seen = {}
        for n in p.recurse().notes:
            if isinstance(n.id, str) and n.id in seen:
                seen[n.id] += 1
                n.id = f"{n.id}-b{seen[n.id]}"
            else:
                seen[n.id] = 0
        return p

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.movementName = ev["title"]
    score.metadata.composer = f"{ev['part']} · {k.tonic.name} {k.mode} · {round(ev['bpm'])} BPM"
    parts = []
    for i, events in enumerate(ev["staves"]):
        # Treble by default (the viewer can switch single-staff parts to bass clef).
        p = build(events, f"n{i}", staff_clef=clef.TrebleClef() if mono or i == 0 else clef.BassClef())
        if i == 0 and harmony and ev.get("harmony"):
            add_divisi(p, build(ev["harmony"], "h", words=False))
        parts.append(p)
        score.insert(0, p)
    if len(parts) > 1:
        score.insert(0, layout.StaffGroup(parts, symbol="brace"))
    return score


def build_score(notes, part, mono, bpm, title, time_sig, detail="simple", words=None, lyrics=True,
                harmony=None, graces=None, scored=False):
    """Notes (seconds from bar 1) -> music21 Score at the given detail level (see score_events;
    `lyrics` decides whether the words are shown)."""
    return events_score(score_events(notes, part, mono, bpm, title, time_sig, detail, words,
                                     harmony, graces, scored), lyrics, harmony=bool(harmony))


def save_originals(sidecar_path: Path):
    """Keep the transcription's note lists for all three detail levels, untouched by edits, in
    originals/ next to the score (the viewer's edits go in edits/)."""
    info = json.loads(sidecar_path.read_text())
    folder = sidecar_path.parent / "originals"
    folder.mkdir(exist_ok=True)
    part = sidecar_path.stem
    for detail in DETAIL_LEVELS:
        ev = score_events([tuple(n) for n in info["notes"]], info["part"], info["mono"], info["bpm"],
                          info["title"], info["time_sig"], detail, words=info.get("words"),
                          harmony=info.get("harmony"), graces=info.get("graces"),
                          scored=info.get("scored", False))
        (folder / f"{part}.{detail}.json").write_text(json.dumps(ev))


# ---------------------------------------------------------------- main

def process(source: str, part="vocals", bpm=None, time_sig="4/4", poly=False, harmony=False) -> Path:
    """Full pipeline. Returns the MusicXML path (a PDF at concert pitch is written next to it)."""
    audio = fetch_audio(source, ROOT / "output" / "_downloads")
    out_dir = ROOT / "output" / audio.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    mono = part in MONOPHONIC and not poly

    stem = separate(audio, part, out_dir)
    words = song_words(stem, audio) if part == "vocals" else []
    notes, scored = transcribe(stem, part, mono, words)
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
        "scored": scored,         # note strengths are confidences (see note_confidence)
        "audio": str(audio), "stem": str(stem), "backing": str(stem.with_name(f"no_{part}.wav")),
        "notes": [[float(n[0]), float(n[1]), int(n[2]), float(n[3]), bool(n[4])] for n in notes],
        "words": [[max(s - offset, 0), max(e - offset, 0), text]
                  for s, e, text in align_words(raw_notes, words, split=False)[1]] if words else [],
    }
    (out_dir / f"{part}.json").write_text(json.dumps(info))
    if part == "vocals" and mono:
        log(f"Found {add_graces(out_dir / f'{part}.json')} scoops to write as grace notes")
    if harmony and part == "vocals" and mono:
        add_harmony(out_dir / f"{part}.json")
    else:
        save_originals(out_dir / f"{part}.json")

    log("Building score")
    score = build_score(notes, part, mono, bpm, audio.stem, time_sig, words=info["words"], lyrics=False,
                        scored=scored)
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
    ap.add_argument("source", help="audio/video file, or a link (YouTube, SoundCloud, Spotify, ...)")
    ap.add_argument("-p", "--part", choices=PARTS, default="vocals", help="which part to transcribe (default: vocals)")
    ap.add_argument("--bpm", type=float, help="tempo override (auto-detected if omitted)")
    ap.add_argument("--time", default="4/4", help="time signature (default: 4/4)")
    ap.add_argument("--poly", action="store_true", help="allow chords even for vocals/bass")
    ap.add_argument("--harmony", action="store_true",
                    help="vocals: also transcribe the backing vocals' harmony (shown as a second voice)")
    ap.add_argument("--audio-only", action="store_true", help="just download/extract the audio and stop")
    ap.add_argument("--split", type=int, choices=[4, 6],
                    help="don't transcribe: split the song into 4 or 6 parts (WAVs, mix them in the viewer)")
    ap.add_argument("--vocal-layers", action="store_true", help="with --split: also split vocals into lead and backing")
    ap.add_argument("--no-open", action="store_true", help="don't open the viewer when done")
    args = ap.parse_args()

    try:
        if args.audio_only:
            print(f"\nAudio saved to {fetch_audio(args.source, ROOT / 'output' / '_downloads')}")
            return
        if args.split:
            audio = split_song(args.source, "htdemucs_6s" if args.split == 6 else "htdemucs", args.vocal_layers)
            model = "htdemucs_6s" if args.split == 6 else "htdemucs"
            print(f"\nParts saved in {split_dir(audio, model)}")
            return
        xml = process(args.source, args.part, args.bpm, args.time, args.poly, args.harmony)
    except PipelineError as e:
        sys.exit(str(e))
    print(f"\nDone. Files in {xml.parent}")
    if not args.no_open:
        viewer.serve(xml)


if __name__ == "__main__":
    main()
