#!/usr/bin/env python3
"""Local sheet-music viewer with transpose buttons.

    python viewer.py [path/to/part.musicxml]

Serves every transcription under output/ at http://localhost:8765. Transposition is done
with music21 (correct spelling + key signature) and rendered with Verovio.
"""
import io
import json
import sys
import threading
import warnings
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
PORT = 8765

INTERVALS = ["P1", "m2", "M2", "m3", "M3", "P4", "A4", "P5", "m6", "M6", "m7", "M7"]
# Written range of the alto sax, as MIDI numbers (Bb3 - F#6).
ALTO_RANGE = (58, 90)
_render_lock = threading.Lock()  # Verovio toolkits aren't thread-safe


def list_scores():
    return sorted(str(p.relative_to(OUTPUT)) for p in OUTPUT.glob("*/*.musicxml"))


def resolve(rel: str) -> Path:
    path = (OUTPUT / rel).resolve()
    if OUTPUT not in path.parents or path.suffix != ".musicxml" or not path.exists():
        raise FileNotFoundError(rel)
    return path


def interval_for(semitones: int):
    """Semitones -> music21 Interval with a sensible name (e.g. +9 -> M6, -14 -> -M9)."""
    from music21 import interval

    n = abs(semitones)
    octaves, rem = divmod(n, 12)
    name = INTERVALS[rem]
    number = int(name[1:]) + 7 * octaves
    if number == 1 and octaves:  # whole octaves: P8, P15...
        number = 1 + 7 * octaves
    return interval.Interval(("-" if semitones < 0 else "") + name[0] + str(number))


def sidecar(rel: str) -> dict | None:
    """The notes + audio sync info song.py saves next to each score (older runs lack it)."""
    path = resolve(rel).with_suffix(".json")
    return json.loads(path.read_text()) if path.exists() else None


@lru_cache(maxsize=16)
def base_score(rel: str, detail: str, mtime: float, lyrics: bool = False):
    import song
    from music21 import converter

    info = sidecar(rel)
    if not info:
        return converter.parse(str(resolve(rel)))
    # Words always shape the notes (one note per word); `lyrics` only decides if they show.
    return song.build_score([tuple(n) for n in info["notes"]], info["part"], info["mono"],
                            info["bpm"], info["title"], info["time_sig"], detail,
                            words=info.get("words"), lyrics=lyrics)


@lru_cache(maxsize=64)
def transposed(rel: str, semitones: int, clef_name: str = "treble", detail: str = "simple",
               mtime: float = 0, lyrics: bool = False):
    """Returns (musicxml bytes, info dict) for the score transposed by `semitones`.

    `mtime` is only part of the cache key, so a re-transcribed score isn't served stale."""
    import copy
    from music21 import clef, interval, key
    from music21.musicxml.m21ToXml import GeneralObjectExporter

    score = copy.deepcopy(base_score(rel, detail, mtime, lyrics))
    ks = next(iter(score.recurse().getElementsByClass(key.KeySignature)), key.KeySignature(0))

    iv = interval_for(semitones)
    if semitones % 12:
        # Pick the enharmonic spelling with the fewest accidentals (Db major, not C# major).
        d2 = interval.Interval("d2")
        options = [iv, interval.add([iv, d2]), interval.add([iv, d2.reverse()])]
        iv = min(options, key=lambda i: abs(ks.transpose(i).sharps))
    if semitones:
        score = score.transpose(iv)
        # Accidental visibility was decided for the old key; redo it for the new one.
        for part in score.parts:
            for n in part.recurse().notes:
                for p in n.pitches:
                    if p.accidental:
                        p.accidental.displayStatus = None
            part.makeAccidentals(inPlace=True, overrideStatus=True)

    # Single-staff parts get the chosen clef; piano-style grand staves keep treble + bass.
    if len(score.parts) == 1:
        new_clef = clef.BassClef if clef_name == "bass" else clef.TrebleClef
        for c in list(score.recurse().getElementsByClass(clef.Clef)):
            c.activeSite.replace(c, new_clef())

    pitches = [p.midi for p in score.pitches]
    new_ks = next(iter(score.recurse().getElementsByClass(key.KeySignature)), key.KeySignature(0))
    info = {
        "semitones": semitones,
        "interval": f"{'up' if semitones > 0 else 'down'} {iv.name}" if semitones else "none",
        "keySig": key_name(new_ks.sharps),
        "low": midi_name(min(pitches)) if pitches else "",
        "high": midi_name(max(pitches)) if pitches else "",
        "belowAlto": sum(p < ALTO_RANGE[0] for p in pitches),
        "aboveAlto": sum(p > ALTO_RANGE[1] for p in pitches),
        "bpm": score_bpm(score),
    }
    return GeneralObjectExporter(score).parse(), info


def score_bpm(score) -> float:
    """song.py stores the detected tempo in the composer line, e.g. "vocals · C major · 120 BPM"."""
    import re
    m = re.search(r"(\d+(?:\.\d+)?) BPM", (score.metadata and score.metadata.composer) or "")
    return float(m.group(1)) if m else 120.0


def key_name(sharps: int) -> str:
    if sharps == 0:
        return "no sharps/flats"
    return f"{abs(sharps)} {'sharp' if sharps > 0 else 'flat'}{'s' if abs(sharps) > 1 else ''}"


def midi_name(m: int) -> str:
    from music21 import pitch
    return pitch.Pitch(midi=m).nameWithOctave.replace("-", "b")


def render_svgs(xml: bytes, screen: bool, timing: bool = False):
    """Render pages to SVG. With timing=True also returns note/measure timing for playback,
    taken from the same Verovio load so the element ids match the SVG."""
    import verovio

    with _render_lock:
        # Explicit resource path: Verovio can't find its fonts from non-main threads otherwise.
        tk = verovio.toolkit(False)
        tk.setResourcePath(str(Path(verovio.__file__).parent / "data"))
        tk.setOptions({
            "pageWidth": 2100, "pageHeight": 2970, "scale": 45 if screen else 40,
            "pageMarginTop": 80, "pageMarginBottom": 80,
            "pageMarginLeft": 100, "pageMarginRight": 100,
            "adjustPageHeight": screen, "svgViewBox": screen,
            "footer": "none", "breaks": "auto", **LYRIC_LAYOUT,
        })
        tk.loadData(xml.decode())
        svgs = [align_lyrics(tk.renderToSVG(i)) for i in range(1, tk.getPageCount() + 1)]
        if not timing:
            return svgs
        return svgs, playback_timing(tk)


# Slightly smaller lyrics, and more room between words, so notes spread out enough for each
# word to fit between its note and the next word (align_lyrics shrinks the few that don't).
LYRIC_LAYOUT = {"lyricSize": 4.0, "lyricWordSpace": 5}


@lru_cache(maxsize=1)
def _lyric_font():
    """The serif font the lyrics are drawn in (Verovio asks for "Times, serif"), for measuring
    words; None if it isn't installed (then widths are estimated)."""
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/Times.ttc", "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, 1000)
    return None


def align_lyrics(svg: str, min_scale=0.7) -> str:
    """Start every word exactly at its note's head, and shrink a word that would run into the
    next one until it fits (down to `min_scale`, then squeeze it).

    Verovio places words by its own width estimates, which differ from the real font: words
    began up to a notehead's width before their note and sometimes ran together."""
    import re
    import xml.etree.ElementTree as ET

    if 'class="syl"' not in svg:
        return svg
    SVG, XL = "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"
    ET.register_namespace("", SVG)
    ET.register_namespace("xlink", XL)
    q = lambda tag: f"{{{SVG}}}{tag}"
    root = ET.fromstring(svg)
    font = _lyric_font()
    width = lambda word, size: (font.getlength(word) / 1000 if font else 0.5 * len(word)) * size

    for system in root.iter(q("g")):
        if system.get("class") != "system":
            continue
        right = max((float(m) for path in system.iter(q("path")) if path.get("d", "").startswith("M")
                     for m in re.findall(r"L(-?[\d.]+)", path.get("d"))), default=float("inf"))
        words = []
        for note in system.iter(q("g")):
            if note.get("class") != "note":
                continue
            use = next((u for g in note.iter(q("g")) if g.get("class") == "notehead" for u in g.iter(q("use"))), None)
            text = next((t for g in note.iter(q("g")) if g.get("class") == "syl" for t in g.iter(q("text"))), None)
            if use is None or text is None:
                continue
            sized = [t for t in text.iter(q("tspan")) if t.get("font-size", "").endswith("px")]
            word = "".join(text.itertext()).strip()
            m = re.search(r"translate\((-?[\d.]+)", use.get("transform", ""))
            if not sized or not word or not m:
                continue
            words.append([float(m.group(1)), text, sized[0], word])
        words.sort(key=lambda w: w[0])
        for i, (x, text, tspan, word) in enumerate(words):
            size = float(tspan.get("font-size")[:-2])
            text.set("x", str(round(x)))
            # The last word on a line may run a little past the staff into the margin.
            limit = (words[i + 1][0] - 0.3 * size) if i + 1 < len(words) else right + 2 * size
            w = width(word, size)
            if w > limit - x and w > 0:
                scale = max((limit - x) / w, min_scale)
                tspan.set("font-size", f"{round(size * scale)}px")
                if scale == min_scale and limit > x:   # still too long: squeeze it into the gap
                    text.set("textLength", str(round(limit - x)))
                    text.set("lengthAdjust", "spacingAndGlyphs")
    return ET.tostring(root, encoding="unicode")


def slurs_and_ties(tk):
    """From Verovio's MEI (same ids as the SVG): the set of note ids that continue a slur
    (every slurred note except the first), and a {tie end id: tie start id} map."""
    import xml.etree.ElementTree as ET

    ns = {"m": "http://www.music-encoding.org/ns/mei"}
    xid = "{http://www.w3.org/XML/1998/namespace}id"
    root = ET.fromstring(tk.getMEI())
    order = [n.get(xid) for n in root.iter("{http://www.music-encoding.org/ns/mei}note")]
    index = {nid: i for i, nid in enumerate(order)}
    ends = lambda el: ((el.get("startid") or "").lstrip("#"), (el.get("endid") or "").lstrip("#"))
    slurred = set()
    for slur in root.iterfind(".//m:slur", ns):
        a, b = ends(slur)
        if a in index and b in index:
            slurred.update(order[index[a] + 1: index[b] + 1])
    ties = {b: a for a, b in map(ends, root.iterfind(".//m:tie", ns))}
    return slurred, ties


def playback_timing(tk):
    """Notes as {id, q (start, quarter notes), dur, pitch (written MIDI), legato} plus
    measure starts. `legato` = slurred from the previous note (played without a new attack)."""
    timemap = tk.renderToTimemap({"includeMeasures": True, "includeRests": False})
    slurred, ties = slurs_and_ties(tk)
    starts, notes, measures = {}, [], []
    for entry in timemap:
        q = entry["qstamp"]
        if "measureOn" in entry:
            measures.append({"id": entry["measureOn"], "q": q})
        for note_id in entry.get("on", []):
            starts[note_id] = q
        for note_id in entry.get("off", []):
            if note_id in starts:
                start = starts.pop(note_id)
                pitch = tk.getMIDIValuesForElement(note_id).get("pitch")
                notes.append({"id": note_id, "ids": [note_id], "q": start, "dur": q - start,
                              "pitch": pitch, "legato": note_id in slurred})
    notes.sort(key=lambda n: n["q"])
    # The timemap lists both halves of a tied note; fold each continuation into the note it's
    # tied from so it sounds as one held note (and both halves light up).
    by_id = {n["id"]: n for n in notes}
    root_of = lambda nid: root_of(ties[nid]) if nid in ties and ties[nid] in by_id else nid
    for n in notes:
        if n["id"] in ties and ties[n["id"]] in by_id:
            head = by_id[root_of(n["id"])]
            head["dur"] = max(head["dur"], n["q"] + n["dur"] - head["q"])
            head["ids"].append(n["id"])
            n["tied"] = True
    notes = [n for n in notes if not n.get("tied")]
    end = max((n["q"] + n["dur"] for n in notes), default=0)
    return {"notes": notes, "measures": measures, "totalQ": end}


def audio_path(side: dict, which: str) -> Path | None:
    path = Path(side.get({"stem": "stem", "backing": "backing", "mix": "audio"}.get(which, ""), ""))
    return path if path.name and path.exists() else None


# ---------------------------------------------------------------- instrument sounds
# FluidR3 General MIDI samples (one mp3 per key, A0-C8), downloaded once into soundfonts/.

SOUNDFONT_URL = "https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM/{}-mp3.js"
SOUNDS = {"alto_sax", "tenor_sax", "trumpet", "clarinet", "flute", "acoustic_grand_piano"}
_sound_lock = threading.Lock()


def soundfont(name: str) -> bytes:
    """JSON {"C4": "data:audio/mp3;base64,...", ...} for a General MIDI instrument."""
    import re
    import urllib.request

    if name not in SOUNDS:
        raise FileNotFoundError(name)
    cache = ROOT / "soundfonts" / f"{name}.json"
    with _sound_lock:
        if not cache.exists():
            with urllib.request.urlopen(SOUNDFONT_URL.format(name), timeout=60) as r:
                text = r.read().decode()
            samples = dict(re.findall(r'"([A-G]b?\d)"\s*:\s*"(data:audio/[^"]+)"', text))
            if not samples:
                raise ValueError(f"couldn't parse soundfont {name}")
            cache.parent.mkdir(exist_ok=True)
            cache.write_text(json.dumps(samples))
        return cache.read_bytes()


def render_pdf(xml: bytes) -> bytes:
    import cairosvg
    from pypdf import PdfWriter

    writer = PdfWriter()
    for svg in render_svgs(xml, screen=False):
        buf = io.BytesIO()
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=buf)
        buf.seek(0)
        writer.append(buf)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# ---------------------------------------------------------------- transcription jobs
# One job at a time: Demucs already uses the whole GPU, and song.log_sink is global.

job = {"state": "idle"}
_job_lock = threading.Lock()


def start_job(source: str, part: str, bpm: float | None) -> bool:
    with _job_lock:
        if job.get("state") == "running":
            return False
        job.clear()
        job.update(state="running", log=[], source=source, part=part, result=None, error=None)
    threading.Thread(target=_run_job, args=(source, part, bpm), daemon=True).start()
    return True


def _run_job(source, part, bpm):
    import song  # imported lazily: song imports this module

    song.log_sink = job["log"]
    try:
        xml = song.process(source, part, bpm)
        job.update(state="done", result=str(xml.relative_to(OUTPUT)))
        transposed.cache_clear()  # a re-run may have overwritten an existing score
    except song.PipelineError as e:
        job.update(state="error", error=str(e))
    except Exception as e:
        job.update(state="error", error=f"{type(e).__name__}: {e}")
    finally:
        song.log_sink = None


class Handler(BaseHTTPRequestHandler):
    default_score = ""

    def log_message(self, *args):
        pass

    def send(self, body: bytes, ctype: str, status=200, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                html = PAGE.replace("__DEFAULT__", json.dumps(self.default_score))
                return self.send(html.encode(), "text/html; charset=utf-8")
            if url.path == "/favicon.ico":
                return self.send(b"", "image/x-icon", 204)
            if url.path == "/api/list":
                return self.send(json.dumps(list_scores()).encode(), "application/json")
            if url.path == "/api/job":
                return self.send(json.dumps(job).encode(), "application/json")
            if url.path == "/api/sound":
                return self.send(soundfont(q.get("name", "")), "application/json",
                                 extra={"Cache-Control": "max-age=31536000"})
            rel, semis = q.get("f", ""), int(q.get("t", 0))
            if url.path == "/api/audio":
                return self.send_audio(rel, q.get("which", "stem"))
            detail = q.get("detail") if q.get("detail") in ("simple", "balanced", "detailed") else "simple"
            xml, info = transposed(rel, semis, "bass" if q.get("clef") == "bass" else "treble",
                                   detail, resolve(rel).with_suffix(".json").stat().st_mtime
                                   if resolve(rel).with_suffix(".json").exists() else resolve(rel).stat().st_mtime,
                                   q.get("lyrics") == "1")
            if url.path == "/api/render":
                svgs, timing = render_svgs(xml, screen=True, timing=True)
                side = sidecar(rel)
                audio = None
                if side:
                    audio = {"offset": side["audio_offset"],
                             "available": [w for w in ("stem", "backing", "mix") if audio_path(side, w)]}
                if side:
                    info = {**info, "bpm": side["bpm"]}   # exact tempo - the subtitle's is rounded
                info["hasLyrics"] = bool(side and side.get("words"))
                body = {"svgs": svgs, **timing, **info, "audio": audio,
                        "part": side["part"] if side else Path(rel).stem}
                return self.send(json.dumps(body).encode(), "application/json")
            if url.path == "/api/pdf":
                label = q.get("label", "")
                name = Path(rel).parent.name + f" - {Path(rel).stem}" + (f" ({label})" if label else "")
                ascii_name = name.replace("♭", "b").encode("ascii", "ignore").decode()
                disposition = f'attachment; filename="{ascii_name}.pdf"; filename*=UTF-8\'\'{quote(name)}.pdf'
                return self.send(render_pdf(xml), "application/pdf",
                                 extra={"Content-Disposition": disposition})
            self.send(b"not found", "text/plain", 404)
        except FileNotFoundError:
            self.send(b"score not found", "text/plain", 404)
        except Exception as e:  # surface errors in the page rather than hanging
            self.send(f"error: {e}".encode(), "text/plain", 500)


    def send_audio(self, rel: str, which: str):
        side = sidecar(rel)
        path = side and audio_path(side, which)
        if not path:
            return self.send(b"no audio for this score - re-run it to enable", "text/plain", 404)
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if a and b else size - 1
        end = min(end, size - 1)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass   # the browser cancels range requests all the time while seeking

    def do_POST(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if url.path == "/api/upload":
            # Raw file body; keep only the base name so uploads can't escape the folder.
            name = Path(q.get("name", "upload")).name or "upload"
            dest = OUTPUT / "_uploads" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            return self.send(json.dumps({"path": str(dest)}).encode(), "application/json")
        if url.path == "/api/job":
            req = json.loads(body or b"{}")
            source = (req.get("source") or "").strip()
            if not source:
                return self.send(b"Paste a link or choose a file first.", "text/plain", 400)
            bpm = float(req["bpm"]) if req.get("bpm") else None
            if not start_job(source, req.get("part", "vocals"), bpm):
                return self.send(b"Another song is still processing.", "text/plain", 409)
            return self.send(b"{}", "application/json")
        self.send(b"not found", "text/plain", 404)


def serve(default: Path | None = None, open_browser=True):
    if default:
        Handler.default_score = str(default.resolve().relative_to(OUTPUT))
    elif scores := list_scores():
        Handler.default_score = scores[-1]
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}/"
    print(f"\nSheet music viewer running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sheet Music</title>
<style>
  :root { --bg:#f4f3ef; --panel:#fff; --ink:#1d1d1b; --muted:#6b6a65; --line:#dedcd5;
          --accent:#b5651d; --accent-ink:#fff; --warn:#a33a2a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--line);
           padding:10px 20px; display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; }
  select { font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:#fff;
           max-width:340px; }
  .group { display:flex; align-items:center; gap:4px; }
  .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; margin-right:4px; }
  button { font:inherit; border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:6px;
           padding:6px 10px; cursor:pointer; min-width:38px; }
  button:hover { border-color:#bbb; }
  button.on { background:var(--accent); border-color:var(--accent); color:var(--accent-ink); }
  button.primary { background:var(--ink); color:#fff; border-color:var(--ink); }
  #status { display:flex; flex-wrap:wrap; gap:6px 16px; padding:10px 20px; color:var(--muted); }
  #status b { color:var(--ink); font-weight:600; }
  #status .warn { color:var(--warn); }
  main { padding:0 20px 40px; display:flex; flex-direction:column; align-items:center; gap:16px; }
  .page { background:#fff; width:100%; max-width:1000px; box-shadow:0 1px 3px rgba(0,0,0,.12); border-radius:2px; }
  .page svg { display:block; width:100%; height:auto; }
  #msg { padding:40px; color:var(--muted); }
  main.loading .page { opacity:.45; transition:opacity .15s; }
  #new { border-bottom:1px solid var(--line); background:var(--panel); padding:14px 20px; }
  #new form { display:flex; flex-wrap:wrap; gap:8px; align-items:center; max-width:1000px; }
  #new input[type=url] { flex:1 1 320px; min-width:0; font:inherit; padding:7px 10px;
                         border:1px solid var(--line); border-radius:6px; }
  #new input[type=number] { width:90px; font:inherit; padding:7px 8px; border:1px solid var(--line); border-radius:6px; }
  #new .or { color:var(--muted); }
  #filename { color:var(--muted); max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #progress { margin-top:10px; font-size:13px; color:var(--muted); white-space:pre-wrap; max-width:1000px; }
  #progress b { color:var(--ink); }
  #progress .err { color:var(--warn); }
  main { padding-bottom:90px; }
  .page g.note, .page g.chord, .page g.measure { cursor:pointer; }
  .page g.playing { fill:var(--accent); color:var(--accent); }
  .page rect.measure-hl { fill:var(--accent); fill-opacity:.12; stroke:none; pointer-events:none; }
  .page rect.measure-hit { fill:var(--accent); fill-opacity:0; stroke:none; pointer-events:all; cursor:pointer; }
  .page rect.measure-hit:hover { fill-opacity:.06; }
  #transport { position:fixed; left:0; right:0; bottom:0; z-index:3; background:var(--panel);
               border-top:1px solid var(--line); padding:10px 20px calc(10px + env(safe-area-inset-bottom, 0px));
               display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; }
  #play { min-width:92px; }
  #scrubWrap { flex:1 1 280px; display:flex; align-items:center; gap:10px; min-width:0; }
  #scrub { flex:1; min-width:0; accent-color:var(--accent); }
  #clock { font-variant-numeric:tabular-nums; color:var(--muted); white-space:nowrap; }
  #clock b { color:var(--ink); }
  #soundMsg { color:var(--muted); font-size:12px; }
  kbd { font-size:11px; border:1px solid var(--line); border-radius:3px; padding:0 4px; background:#fafafa; }
</style></head>
<body>
<header>
  <button id="newBtn" class="primary">+ New song</button>
  <select id="score" aria-label="Score"></select>
  <div class="group" role="group" aria-label="Instrument">
    <span class="label">Instrument</span>
    <button data-preset="0">Concert C</button>
    <button data-preset="9">Alto Sax E♭</button>
    <button data-preset="14">Tenor Sax B♭</button>
    <button data-preset="2">Trumpet / Clarinet B♭</button>
  </div>
  <div class="group" role="group" aria-label="Transpose">
    <span class="label">Transpose</span>
    <button data-step="-12" title="Down an octave (↓)">−8va</button>
    <button data-step="-1" title="Down a half step (←)">−½</button>
    <button data-step="1" title="Up a half step (→)">+½</button>
    <button data-step="12" title="Up an octave (↑)">+8va</button>
  </div>
  <div class="group" role="group" aria-label="Detail">
    <span class="label">Detail</span>
    <button data-detail="simple" title="Snap to the key, fold bends into the main notes, 8th-note rhythm">Simple</button>
    <button data-detail="balanced" title="Fold quick bend notes, keep held chromatic notes and 16ths">Balanced</button>
    <button data-detail="detailed" title="Everything as transcribed">Detailed</button>
  </div>
  <div class="group">
    <span class="label">Lyrics</span>
    <button id="lyricsBtn" title="Show the words under the notes">Show</button>
  </div>
  <div class="group" role="group" aria-label="Clef">
    <span class="label">Clef</span>
    <button data-clef="treble">Treble</button>
    <button data-clef="bass">Bass</button>
  </div>
  <button class="primary" id="pdf">Download PDF</button>
</header>
<section id="new" hidden>
  <form id="newForm">
    <input type="url" id="src" placeholder="Paste a YouTube link…" aria-label="YouTube link">
    <span class="or">or</span>
    <button type="button" id="pick">Choose audio file</button>
    <input type="file" id="file" accept="audio/*,video/*" hidden>
    <span id="filename"></span>
    <select id="part" aria-label="Part">
      <option value="vocals">Vocals</option>
      <option value="bass">Bass</option>
      <option value="guitar">Guitar</option>
      <option value="piano">Piano</option>
      <option value="other">Other (keys, synths, horns)</option>
    </select>
    <input type="number" id="bpm" min="30" max="300" placeholder="BPM (auto)" aria-label="Tempo (optional)">
    <button type="submit" class="primary" id="go">Transcribe</button>
  </form>
  <div id="progress"></div>
</section>
<div id="status"></div>
<main id="pages"><div id="msg">Loading…</div></main>
<footer id="transport">
  <button id="play" class="primary" title="Play / pause (Space)">▶ Play</button>
  <div id="scrubWrap">
    <input type="range" id="scrub" min="0" max="1" step="0.01" value="0" aria-label="Playback position">
    <span id="clock"><b>0:00</b> / 0:00 · m. 1</span>
  </div>
  <div class="group" role="group" aria-label="Speed">
    <span class="label">Speed</span>
    <button data-speed="0.5">50%</button>
    <button data-speed="0.75">75%</button>
    <button data-speed="0.9">90%</button>
    <button data-speed="1">100%</button>
  </div>
  <div class="group">
    <span class="label">Hear</span>
    <button id="notesBtn" title="Play the transcribed notes">Notes</button>
    <select id="orig" aria-label="Original audio">
      <option value="off">No original</option>
      <option value="stem">Original vocals</option>
      <option value="backing">Backing track</option>
      <option value="mix">Full song</option>
    </select>
  </div>
  <div class="group">
    <span class="label">Sound</span>
    <select id="sound" aria-label="Playback sound">
      <option value="alto_sax">Alto sax</option>
      <option value="tenor_sax">Tenor sax</option>
      <option value="trumpet">Trumpet</option>
      <option value="clarinet">Clarinet</option>
      <option value="flute">Flute</option>
      <option value="acoustic_grand_piano">Piano</option>
      <option value="synth">Basic synth</option>
    </select>
    <span id="soundMsg"></span>
  </div>
</footer>
<script>
const DEFAULT = __DEFAULT__;
const PRESETS = {0:"Concert C", 9:"Alto Sax E♭", 14:"Tenor Sax B♭", 2:"Trumpet/Clarinet B♭"};
const $ = s => document.querySelector(s);
const store = { get(k){ try { return localStorage.getItem(k); } catch { return null; } },
                set(k,v){ try { localStorage.setItem(k,v); } catch {} } };
let file = DEFAULT, t = 9, inst = 9, detail = "simple", lyrics = store.get("lyrics") === "1", req = 0, clef = store.get("clef") === "bass" ? "bass" : "treble";   // default: alto sax

function label() {
  if (PRESETS[t]) return PRESETS[t];
  for (const base of [9, 14, 2, 0]) {          // e.g. "Alto Sax E♭ +8va"
    const d = t - base;
    if (d && d % 12 === 0) return `${PRESETS[base]} ${d > 0 ? "+" : "−"}${Math.abs(d / 12)}8va`;
  }
  return `${t > 0 ? "+" : ""}${t} semitones`;
}

async function render() {
  const id = ++req;
  store.set("t:" + file, t);
  store.set("clef", clef);
  store.set("detail:" + file, detail);
  store.set("lyrics", lyrics ? "1" : "0");
  $("#lyricsBtn").classList.toggle("on", lyrics);
  $("#lyricsBtn").textContent = lyrics ? "On" : "Off";
  document.querySelectorAll("[data-clef]").forEach(b => b.classList.toggle("on", b.dataset.clef === clef));
  document.querySelectorAll("[data-detail]").forEach(b => b.classList.toggle("on", b.dataset.detail === detail));
  document.querySelectorAll("[data-preset]").forEach(b => b.classList.toggle("on", +b.dataset.preset === t));
  $("#pages").classList.add("loading");
  const r = await fetch(`/api/render?f=${encodeURIComponent(file)}&t=${t}&clef=${clef}&detail=${detail}&lyrics=${lyrics ? 1 : 0}`);
  if (id !== req) return;                       // a newer click superseded this one
  $("#pages").classList.remove("loading");
  if (!r.ok) { $("#pages").innerHTML = `<div id="msg">${await r.text()}</div>`; return; }
  const d = await r.json();
  $("#pages").innerHTML = d.svgs.map(s => `<div class="page">${s}</div>`).join("");
  player.load(d);
  $("#lyricsBtn").disabled = !d.hasLyrics;
  $("#lyricsBtn").title = d.hasLyrics ? "Show the words under the notes" : "No lyrics for this song (re-run it to add them)";
  const warn = [];
  if (d.belowAlto) warn.push(`${d.belowAlto} notes below alto sax range`);
  if (d.aboveAlto) warn.push(`${d.aboveAlto} notes above alto sax range`);
  $("#status").innerHTML =
    `<span>Showing <b>${label()}</b></span>` +
    `<span>Transposed <b>${d.interval}</b> (${t > 0 ? "+" : ""}${t})</span>` +
    `<span>Key signature <b>${d.keySig}</b></span>` +
    `<span>Range <b>${d.low} – ${d.high}</b></span>` +
    (warn.length ? `<span class="warn">${warn.join(" · ")} (try ±8va)</span>` : "") +
    `<span><kbd>←</kbd><kbd>→</kbd> half step <kbd>↑</kbd><kbd>↓</kbd> octave</span>`;
}

document.querySelectorAll("[data-clef]").forEach(b => b.onclick = () => { clef = b.dataset.clef; render(); });
document.querySelectorAll("[data-detail]").forEach(b => b.onclick = () => { detail = b.dataset.detail; render(); });
$("#lyricsBtn").onclick = () => { lyrics = !lyrics; render(); };
function setT(v) { t = Math.max(-36, Math.min(36, v)); render(); }

document.querySelectorAll("[data-preset]").forEach(b => b.onclick = () => {
  inst = +b.dataset.preset;          // what the player's instrument is (playback sounds at concert pitch)
  store.set("inst:" + file, inst);
  setT(inst);
});
document.querySelectorAll("[data-step]").forEach(b => b.onclick = () => setT(t + +b.dataset.step));
$("#pdf").onclick = () => {
  location.href = `/api/pdf?f=${encodeURIComponent(file)}&t=${t}&clef=${clef}&detail=${detail}&lyrics=${lyrics ? 1 : 0}&label=${encodeURIComponent(label())}`;
};
$("#score").onchange = e => loadList(e.target.value);
document.addEventListener("keydown", e => {
  const inField = ["SELECT", "INPUT", "TEXTAREA", "BUTTON"].includes(e.target.tagName);
  if (e.key === " " && (!inField || e.target.id === "scrub" || e.target.tagName === "BUTTON")) {
    e.preventDefault(); player.toggle(); return;
  }
  if (inField && e.target.tagName !== "BUTTON") return;   // let fields (and the scrub bar) use arrow keys
  const step = {ArrowLeft:-1, ArrowRight:1, ArrowDown:-12, ArrowUp:12}[e.key];
  if (step) { e.preventDefault(); setT(t + step); }
});

// ---- playback
// Notes come from Verovio's timemap (positions in quarter notes, written pitch). They sound at
// concert pitch: written pitch minus the instrument's transposition (alto sax sounds a M6 lower).
const NAMES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"];
const midiName = m => NAMES[m % 12] + (Math.floor(m / 12) - 1);

// A box around one measure, sized from its five staff lines only - not the notes, which
// would make measures with high/low notes taller than the rest.
function measureRect(measureEl, cls) {
  const staff = measureEl.querySelector("g.staff");
  const lines = staff ? [...staff.children].filter(c => c.tagName === "path") : [];
  const boxes = (lines.length ? lines : [staff || measureEl]).map(l => l.getBBox());
  const x = Math.min(...boxes.map(b => b.x)), right = Math.max(...boxes.map(b => b.x + b.width));
  const y = Math.min(...boxes.map(b => b.y)), bottom = Math.max(...boxes.map(b => b.y + b.height));
  const pad = (bottom - y) * 0.5;
  const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  r.setAttribute("class", cls);
  r.setAttribute("x", x); r.setAttribute("y", y - pad);
  r.setAttribute("width", right - x); r.setAttribute("height", bottom - y + 2 * pad);
  return r;
}

const player = {
  file: null, notes: [], measures: [], noteById: {}, totalQ: 0, bpm: 120,
  speed: +(store.get("speed") || 1), sound: store.get("sound") || "alto_sax",
  ctx: null, master: null, playing: false, pending: 0, posQ: 0, startQ: 0, startTime: 0, voices: [],
  fonts: {}, buffers: {}, lit: [], litMeasure: null, hlRect: null, dragging: false, wasPlaying: false,
  notesOn: store.get("notesOn") !== "0", orig: store.get("orig") || "off", audioInfo: null,
  audio: Object.assign(new Audio(), {preload: "auto", preservesPitch: true, volume: 0.9}),

  // Original audio is the reference clock's follower: bar 1 (q = 0) is at audioInfo.offset seconds.
  origOn() { return this.orig !== "off" && !!this.audioInfo && this.audioInfo.available.includes(this.orig); },
  audioTimeFor(q) { return this.audioInfo.offset + q * 60 / this.bpm; },
  syncAudioSource() {
    if (!this.origOn()) return;
    const src = `/api/audio?f=${encodeURIComponent(this.file)}&which=${this.orig}`;
    if (this.audio.dataset.src !== src) { this.audio.src = src; this.audio.dataset.src = src; }
  },

  secPerQ() { return 60 / this.bpm / this.speed; },
  // Where playback is *as heard*: the AudioContext clock runs ahead of the speakers by the
  // output latency (~0.17 s in Chrome), so subtract it - otherwise the highlight leads the sound.
  latency() { return (this.ctx && (this.ctx.outputLatency || this.ctx.baseLatency)) || 0; },
  now() {
    if (!this.playing) return this.posQ;
    return this.startQ + Math.max(0, this.ctx.currentTime - this.latency() - this.startTime) / this.secPerQ();
  },
  sounding(n) { return Math.max(21, Math.min(108, n.pitch - inst)); },

  reset(f) { this.stop(); this.file = f; this.posQ = 0; },

  load(d) {   // called after every render (new transposition/clef re-creates the SVG ids)
    const q = this.now(), resume = this.playing || !!this.pending;
    this.stop();
    Object.assign(this, {notes: d.notes, measures: d.measures, totalQ: d.totalQ, bpm: d.bpm,
                         audioInfo: d.audio, lit: [], litMeasure: null, hlRect: null});
    this.noteById = {};
    this.notes.forEach(n => (n.ids || [n.id]).forEach(id => this.noteById[id] = n));
    // Invisible click target over each measure's staff (blank space in an SVG isn't clickable).
    for (const m of this.measures) {
      const el = document.getElementById(m.id);
      if (el) el.insertBefore(measureRect(el, "measure-hit"), el.firstChild);
    }
    const avail = (d.audio && d.audio.available) || [];
    [...$("#orig").options].forEach(o => {
      if (o.value !== "off") o.disabled = !avail.includes(o.value);
      if (o.value === "stem") o.textContent = `Original ${d.part || "part"}`;
    });
    $("#orig").title = d.audio ? "" : "Re-run this song to enable original audio";
    this.syncAudioSource();
    this.posQ = Math.min(q, this.totalQ);
    $("#scrub").max = this.totalQ || 1;
    if (resume) this.play(); else this.draw(this.posQ);
  },

  async ensureSound() {
    if (this.sound === "synth" || !this.notesOn) return true;
    try {
      if (!this.fonts[this.sound]) {
        $("#soundMsg").textContent = "loading sound…";
        const r = await fetch(`/api/sound?name=${this.sound}`);
        if (!r.ok) throw new Error(await r.text());
        this.fonts[this.sound] = await r.json();
      }
      const font = this.fonts[this.sound];
      const needed = [...new Set(this.notes.map(n => this.sounding(n)))];
      await Promise.all(needed.map(async m => {
        const k = this.sound + ":" + m;
        if (this.buffers[k] || !font[midiName(m)]) return;
        const b64 = font[midiName(m)].split(",")[1];
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        this.buffers[k] = await this.ctx.decodeAudioData(bytes.buffer);
      }));
      $("#soundMsg").textContent = "";
      return true;
    } catch (err) {
      $("#soundMsg").textContent = "couldn't load that sound - using basic synth";
      return false;
    }
  },

  async play() {
    if (!this.notes.length) return;
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.8;
      this.master.connect(this.ctx.destination);
    }
    await this.ctx.resume();
    if (this.posQ >= this.totalQ - 0.01) this.posQ = 0;
    const token = this.pending = (this.pending || 0) + 1;   // stop() bumps this to cancel
    this.setButton("Loading…");
    const sampled = await this.ensureSound();
    if (token !== this.pending) return;   // paused/seeked while the sound was loading
    let lead = 0.08;
    if (this.origOn()) {
      // Start the original first, then line the notes up with wherever it actually started.
      this.syncAudioSource();
      const a = this.audio;
      a.playbackRate = this.speed;
      a.currentTime = Math.max(0, this.audioTimeFor(this.posQ));
      try { await a.play(); } catch (err) { $("#soundMsg").textContent = "couldn't play the original audio"; }
      if (token !== this.pending) { a.pause(); return; }
      if (!a.paused) { this.posQ = (a.currentTime - this.audioInfo.offset) * this.bpm / 60; lead = 0; }
    }
    this.pending = 0;
    this.playing = true;
    this.setButton();
    this.startQ = this.posQ;
    this.startTime = this.ctx.currentTime + lead;
    this.gapBase = null;
    this.lastResync = 0;
    const spq = this.secPerQ();
    if (this.notesOn) this.notes.forEach((n, i) => {
      const endQ = n.q + n.dur;
      if (endQ <= this.startQ) return;
      const t0 = this.startTime + Math.max(0, n.q - this.startQ) * spq;
      const t1 = this.startTime + (endQ - this.startQ) * spq;
      const m = this.sounding(n), prev = this.notes[i - 1];
      const slurred = n.legato && prev && Math.abs(prev.q + prev.dur - n.q) < 1e-6 && n.q > this.startQ;
      const buf = sampled && this.buffers[this.sound + ":" + m];
      this.voices.push(buf ? this.sample(buf, t0, t1, slurred) : this.synth(m, t0, t1, slurred));
    });
    requestAnimationFrame(() => this.tick());
  },

  envelope(t0, t1, level, attack = 0.01) {
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(level, t0 + attack);
    g.gain.setValueAtTime(level, Math.max(t0 + attack, t1 - 0.02));
    g.gain.linearRampToValueAtTime(0, t1 + 0.06);
    g.connect(this.master);
    return g;
  },
  sample(buf, t0, t1, slurred) {
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    if (slurred) {
      // Slurred (like a sax slur: new note, no new tongue): skip the sample's attack and
      // crossfade in under the previous note. No pitch slide.
      src.connect(this.envelope(t0, t1, 0.9, 0.03));
      src.start(t0, Math.min(0.08, buf.duration / 4));
    } else {
      src.connect(this.envelope(t0, t1, 0.9));
      src.start(t0);
    }
    src.stop(t1 + 0.1);
    return src;
  },
  synth(m, t0, t1, slurred) {
    const osc = this.ctx.createOscillator();
    osc.type = "triangle";
    osc.frequency.value = 440 * Math.pow(2, (m - 69) / 12);
    osc.connect(this.envelope(t0, t1, 0.35, slurred ? 0.03 : 0.01));
    osc.start(t0); osc.stop(t1 + 0.1);
    return osc;
  },

  stop() {
    if (this.playing) this.posQ = this.now();
    this.playing = false;
    if (this.pending) this.pending++;
    this.voices.forEach(v => { try { v.stop(); } catch {} });
    this.voices = [];
    this.audio.pause();
    this.setButton();
  },
  toggle() { this.playing || this.pending ? this.stop() : this.play(); },
  seek(q) {
    const resume = this.playing || !!this.pending;
    this.stop();
    this.posQ = Math.max(0, Math.min(q, this.totalQ));
    if (resume) this.play(); else this.draw(this.posQ);
  },

  tick() {
    if (!this.playing) return;
    const q = this.now();
    if (q >= this.totalQ) { this.stop(); this.posQ = this.totalQ; this.draw(q); return; }
    // Keep the original audio locked to the score. The two clocks differ by a constant
    // (the media element's own pipeline delay), so measure that once as a baseline and only
    // re-sync when the gap really changes - every re-sync is an audible skip.
    if (this.origOn() && !this.audio.paused) {
      const gap = this.audio.currentTime - this.audioTimeFor(q), t = this.ctx.currentTime;
      if (this.gapBase === null) {
        if (t - this.startTime > 0.5) this.gapBase = gap;   // let playback settle first
      } else if (Math.abs(gap - this.gapBase) > 0.3 && t - this.lastResync > 2) {
        this.audio.currentTime = this.audioTimeFor(q) + this.gapBase;
        this.lastResync = t;
      }
    }
    this.draw(q);
    requestAnimationFrame(() => this.tick());
  },

  draw(q) {
    // Light up sounding notes.
    const on = this.notes.filter(n => q >= n.q && q < n.q + n.dur).flatMap(n => n.ids || [n.id]);
    if (on.join() !== this.lit.join()) {
      this.lit.forEach(id => document.getElementById(id)?.classList.remove("playing"));
      on.forEach(id => document.getElementById(id)?.classList.add("playing"));
      this.lit = on;
    }
    // Shade the current measure and keep it on screen.
    let mi = 0;
    while (mi + 1 < this.measures.length && this.measures[mi + 1].q <= q + 1e-6) mi++;
    const m = this.measures[mi];
    if (m && m.id !== this.litMeasure) {
      this.hlRect?.remove();
      const el = document.getElementById(m.id);
      if (el) {
        const r = measureRect(el, "measure-hl");
        el.insertBefore(r, el.firstChild);
        this.hlRect = r;
        if (this.playing) {
          const box = r.getBoundingClientRect(), top = $("header").getBoundingClientRect().bottom;
          const bottom = $("#transport").getBoundingClientRect().top;
          if (box.top < top || box.bottom > bottom) r.scrollIntoView({block: "center", behavior: "smooth"});
        }
      }
      this.litMeasure = m.id;
    }
    // Scrub bar + clock (shown in real song time, independent of practice speed).
    if (!this.dragging) $("#scrub").value = q;
    const fmt = qq => { const s = Math.max(0, Math.round(qq * 60 / this.bpm)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
    $("#clock").innerHTML = `<b>${fmt(q)}</b> / ${fmt(this.totalQ)} · m. ${mi + 1}`;
  },

  setButton(text) { $("#play").textContent = text || (this.playing ? "❚❚ Pause" : "▶ Play"); },
};

$("#play").onclick = () => player.toggle();
$("#scrub").addEventListener("pointerdown", () => {
  player.dragging = true; player.wasPlaying = player.playing || !!player.pending; player.stop();
});
$("#scrub").addEventListener("input", e => { player.posQ = +e.target.value; player.draw(player.posQ); });
$("#scrub").addEventListener("change", e => {
  player.dragging = false;
  player.posQ = +e.target.value;
  if (player.wasPlaying) { player.wasPlaying = false; player.play(); }
});
// Click anywhere in a measure to jump there: on a note -> that note; elsewhere -> the last
// note at or left of the click (or the start of the measure if you click before its first note).
$("#pages").addEventListener("click", e => {
  const measure = e.target.closest("g.measure");
  if (!measure) return;
  const noteOf = el => player.noteById[el.id];
  const clicked = e.target.closest("g.note, g.chord");
  let hit = clicked && (noteOf(clicked) || [...clicked.querySelectorAll("g.note")].map(noteOf).find(Boolean));
  if (!hit) {
    for (const el of measure.querySelectorAll("g.note")) {
      const n = noteOf(el);
      if (n && el.getBoundingClientRect().left <= e.clientX + 2 && (!hit || n.q > hit.q)) hit = n;
    }
  }
  const target = hit ? hit.q : player.measures.find(m => m.id === measure.id)?.q;
  if (target !== undefined) player.seek(target);
});
function setSpeed(v) {
  const resume = player.playing;
  player.stop();
  player.speed = v; store.set("speed", v);
  document.querySelectorAll("[data-speed]").forEach(b => b.classList.toggle("on", +b.dataset.speed === v));
  if (resume) player.play();
}
document.querySelectorAll("[data-speed]").forEach(b => b.onclick = () => setSpeed(+b.dataset.speed));
setSpeed(player.speed);
function restartWith(change) {
  const resume = player.playing || !!player.pending;
  player.stop();
  change();
  if (resume) player.play();
}
$("#notesBtn").classList.toggle("on", player.notesOn);
$("#notesBtn").onclick = () => restartWith(() => {
  player.notesOn = !player.notesOn; store.set("notesOn", player.notesOn ? "1" : "0");
  $("#notesBtn").classList.toggle("on", player.notesOn);
});
$("#orig").value = player.orig;
$("#orig").onchange = e => restartWith(() => {
  player.orig = e.target.value; store.set("orig", player.orig); player.syncAudioSource();
});
$("#sound").value = player.sound;
$("#sound").onchange = e => {
  const resume = player.playing;
  player.stop();
  player.sound = e.target.value; store.set("sound", player.sound);
  if (resume) player.play();
};

// ---- new song panel
let chosenFile = null, startedAt = 0;
$("#newBtn").onclick = () => { $("#new").hidden = !$("#new").hidden; if (!$("#new").hidden) $("#src").focus(); };
$("#pick").onclick = () => $("#file").click();
$("#file").onchange = e => {
  chosenFile = e.target.files[0] || null;
  $("#filename").textContent = chosenFile ? chosenFile.name : "";
  if (chosenFile) $("#src").value = "";
};
$("#src").oninput = () => { if ($("#src").value) { chosenFile = null; $("#file").value = ""; $("#filename").textContent = ""; } };

$("#newForm").onsubmit = async e => {
  e.preventDefault();
  let source = $("#src").value.trim();
  if (!source && !chosenFile) return showProgress("Paste a YouTube link or choose an audio file.", true);
  $("#go").disabled = true;
  if (chosenFile) {
    showProgress(`Uploading ${chosenFile.name}…`);
    const up = await fetch(`/api/upload?name=${encodeURIComponent(chosenFile.name)}`, {method: "POST", body: chosenFile});
    if (!up.ok) { $("#go").disabled = false; return showProgress(await up.text(), true); }
    source = (await up.json()).path;
  }
  const r = await fetch("/api/job", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({source, part: $("#part").value, bpm: $("#bpm").value || null})});
  if (!r.ok) { $("#go").disabled = false; return showProgress(await r.text(), true); }
  startedAt = Date.now();
  poll();
};

function showProgress(html, isError) {
  $("#progress").innerHTML = isError ? `<span class="err">${esc(html)}</span>` : html;
}
const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

async function poll() {
  const j = await (await fetch("/api/job")).json();
  if (j.state === "running") {
    $("#new").hidden = false; $("#go").disabled = true;
    if (!startedAt) startedAt = Date.now();
    const secs = Math.round((Date.now() - startedAt) / 1000);
    const steps = j.log.map(esc);
    if (steps.length) steps[steps.length - 1] = `<b>${steps[steps.length - 1]}…</b>`;
    showProgress((steps.length ? steps.join("\n") : "<b>Starting…</b>") + `\n${secs}s elapsed`);
    return setTimeout(poll, 1000);
  }
  $("#go").disabled = false; startedAt = 0;
  if (j.state === "error") return showProgress(j.error, true);
  if (j.state === "done" && j.result) {
    showProgress(`<b>Done</b> - showing ${esc(j.result)}`);
    $("#src").value = ""; chosenFile = null; $("#file").value = ""; $("#filename").textContent = "";
    await loadList(j.result);
  }
}

async function loadList(select) {
  const list = await (await fetch("/api/list")).json();
  if (!list.length) {
    $("#new").hidden = false;
    $("#pages").innerHTML = `<div id="msg">No songs yet - paste a YouTube link above to get started.</div>`;
    return;
  }
  file = select && list.includes(select) ? select : (list.includes(file) ? file : list[list.length - 1]);
  $("#score").innerHTML = list.map(f => `<option${f === file ? " selected" : ""}>${esc(f)}</option>`).join("");
  const switched = file !== player.file;
  t = +(store.get("t:" + file) ?? 9);
  inst = +(store.get("inst:" + file) ?? 9);
  detail = store.get("detail:" + file) || "simple";
  if (switched) player.reset(file);
  render();
}

(async () => {
  await loadList();
  poll();   // pick up a job that was already running (e.g. after a page refresh)
})();

</script>
</body></html>
"""


if __name__ == "__main__":
    serve(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
