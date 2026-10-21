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

# How the music looks. Music fonts are SMuFL fonts that ship with Verovio; Jazz uses Petaluma,
# Steinberg's Real Book-style hand-copied font, with its matching handwriting for the words
# (Petaluma Script, in fonts/ with its SIL Open Font License; verified against Steinberg's repo).
FONTS = ROOT / "fonts"
STYLES = {"classic": {"music": "Leipzig", "text": None, "lines": {}},
          "jazz": {"music": "Petaluma", "text": ("Petaluma Script", FONTS / "PetalumaScript.otf"),
                   # Felt-pen weight for everything Verovio draws as lines rather than glyphs.
                   "lines": {"stemWidth": 0.3, "barLineWidth": 0.45, "staffLineWidth": 0.18,
                             "ledgerLineThickness": 0.38, "thickBarlineThickness": 1.4,
                             "slurMidpointThickness": 0.95, "slurEndpointThickness": 0.2,
                             "tieMidpointThickness": 0.85, "tieEndpointThickness": 0.2,
                             "lyricLineThickness": 0.35, "tupletBracketThickness": 0.32},
                   "hand": True}}


def register_fonts():
    """Let cairo (the PDF renderer) find the text fonts in fonts/, for this process only -
    nothing is installed on the system. Must run before cairo first draws text."""
    import ctypes
    import ctypes.util

    for style in STYLES.values():
        if not style["text"] or not style["text"][1].exists():
            continue
        path = str(style["text"][1]).encode()
        try:
            if sys.platform == "darwin":   # cairo finds fonts through CoreText on macOS
                cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
                ct = ctypes.CDLL("/System/Library/Frameworks/CoreText.framework/CoreText")
                cf.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
                cf.CFURLCreateFromFileSystemRepresentation.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                                       ctypes.c_long, ctypes.c_bool]
                ct.CTFontManagerRegisterFontsForURL.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
                url = cf.CFURLCreateFromFileSystemRepresentation(None, path, len(path), False)
                ct.CTFontManagerRegisterFontsForURL(url, 1, None)   # 1 = this process only
            else:                          # and through fontconfig elsewhere
                fc = ctypes.CDLL(ctypes.util.find_library("fontconfig"))
                fc.FcConfigAppFontAddFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
                fc.FcConfigAppFontAddFile(None, path)
        except Exception:
            pass   # the PDF falls back to the default text font


register_fonts()


def list_scores():
    return sorted(str(p.relative_to(OUTPUT)) for p in OUTPUT.glob("*/*.musicxml"))


def resolve(rel: str) -> Path:
    path = (OUTPUT / rel).resolve()
    if OUTPUT not in path.parents or path.suffix != ".musicxml" or not path.exists():
        raise FileNotFoundError(rel)
    return path


def song_audio(name: str) -> Path:
    """A downloaded/imported song's clean WAV by its name (never a path from the request)."""
    for path in (OUTPUT / "_downloads").glob("*.wav"):
        if path.stem == name:
            return path
    raise FileNotFoundError(name)


def list_songs():
    """Songs available to split: [{name, splits: {model: [parts]}}], newest first."""
    import song
    paths = sorted((OUTPUT / "_downloads").glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.stem, "splits": {m: song.split_parts(p, m) for m in song.SPLIT_MODELS}} for p in paths]


MIX_FORMATS = {"wav": ("audio/wav", []), "flac": ("audio/flac", ["-c:a", "flac"]),
               "mp3": ("audio/mpeg", ["-c:a", "libmp3lame", "-b:a", "320k"]),
               "m4a": ("audio/mp4", ["-c:a", "aac", "-b:a", "256k"])}


def mix_file(name: str, model: str, gains: dict) -> Path:
    """The mix of a song's parts at these gains, built once and cached (the 12 newest kept)."""
    import hashlib
    import song
    audio = song_audio(name)
    if model not in song.SPLIT_MODELS or not song.split_parts(audio, model):
        raise FileNotFoundError(name)
    key = hashlib.sha1(json.dumps([model, sorted(gains.items())]).encode()).hexdigest()[:16]
    dest = OUTPUT / audio.stem / "mixes" / f"{model}-{key}.wav"
    with _mix_lock:
        if not dest.exists():
            song.mix_parts(audio, model, gains, dest)
            old = sorted(dest.parent.glob("*.wav"), key=lambda p: p.stat().st_mtime)[:-12]
            for p in old:
                p.unlink()
    return dest


_mix_lock = threading.Lock()


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


def level_file(rel: str, detail: str, folder: str = "edits") -> Path:
    """Where a detail level's saved edits (or, with folder="originals", its untouched
    transcription) live: next to the score, e.g. <song>/edits/vocals.balanced.json."""
    score = resolve(rel)
    return score.parent / folder / f"{score.stem}.{detail}.json"


def score_mtime(rel: str, detail: str) -> float:
    """Changes whenever the transcription or this level's edits change (a cache key)."""
    side, edits = resolve(rel).with_suffix(".json"), level_file(rel, detail)
    base = side.stat().st_mtime if side.exists() else resolve(rel).stat().st_mtime
    return max(base, edits.stat().st_mtime if edits.exists() else 0)


@lru_cache(maxsize=16)
def transcribed_events(rel: str, detail: str, mtime: float) -> dict:
    """The note list song.py makes from the transcription (see song.score_events)."""
    import song

    info = sidecar(rel)
    return song.score_events([tuple(n) for n in info["notes"]], info["part"], info["mono"],
                             info["bpm"], info["title"], info["time_sig"], detail,
                             words=info.get("words"), harmony=info.get("harmony"),
                             graces=info.get("graces"), scored=info.get("scored", False))


def level_events(rel: str, detail: str) -> tuple[dict, bool]:
    """(note list, edited?): the saved edits for this level if there are any, else the
    transcription's. Harmony added after the edits were saved still shows up."""
    path = level_file(rel, detail)
    if not path.exists():
        return json.loads(json.dumps(transcribed_events(rel, detail, score_mtime(rel, detail)))), False
    ev = json.loads(path.read_text())
    if "harmony" not in ev:
        fresh = transcribed_events(rel, detail, resolve(rel).with_suffix(".json").stat().st_mtime)
        if fresh.get("harmony"):
            ev["harmony"] = fresh["harmony"]
    return ev, True


def save_edits(rel: str, detail: str, ev: dict):
    """Save a level's edited note list. The untouched transcription of all three levels is kept
    in originals/ (written now if this song predates that)."""
    import song

    song.events_score(ev)   # refuse anything that can't be drawn
    if not all(level_file(rel, d, "originals").exists() for d in song.DETAIL_LEVELS):
        song.save_originals(resolve(rel).with_suffix(".json"))
    path = level_file(rel, detail)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(ev))
    # Every save is also kept in the level's history, so any of them can be restored.
    import time
    folder = history_dir(rel, detail)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{int(time.time() * 1000)}.json").write_text(json.dumps(ev))
    for old in sorted(folder.glob("*.json"))[:-HISTORY_KEEP]:
        old.unlink()


HISTORY_KEEP = 200   # saves kept per detail level


def history_dir(rel: str, detail: str) -> Path:
    score = resolve(rel)
    return score.parent / "edits" / "history" / f"{score.stem}.{detail}"


def history(rel: str, detail: str) -> list[dict]:
    """A level's saved versions, newest first: [{id, time (ms), notes}]."""
    out = []
    for path in sorted(history_dir(rel, detail).glob("*.json"), reverse=True):
        try:
            ev = json.loads(path.read_text())
            out.append({"id": path.stem, "time": int(path.stem),
                        "notes": sum(len(s) for s in ev["staves"]) + len(ev.get("harmony") or [])})
        except (ValueError, KeyError):
            continue
    return out


def history_version(rel: str, detail: str, version: str) -> dict:
    """One saved version's note list ("original": the transcription's)."""
    if version == "original":
        kept = level_file(rel, detail, "originals")
        return json.loads(kept.read_text()) if kept.exists() else \
            json.loads(json.dumps(transcribed_events(rel, detail, resolve(rel).with_suffix(".json").stat().st_mtime)))
    if not version.isdigit():
        raise FileNotFoundError(version)
    return json.loads((history_dir(rel, detail) / f"{version}.json").read_text())


@lru_cache(maxsize=16)
def base_score(rel: str, detail: str, mtime: float, lyrics: bool = False, harmony: bool = False):
    import song
    from music21 import converter

    if not sidecar(rel):
        return converter.parse(str(resolve(rel)))
    # Words always shape the notes (one note per word); `lyrics` only decides if they show.
    return song.events_score(level_events(rel, detail)[0], lyrics, harmony)


@lru_cache(maxsize=64)
def transposed(rel: str, semitones: int, clef_name: str = "treble", detail: str = "balanced",
               mtime: float = 0, lyrics: bool = False, harmony: bool = False, bars: int = 0,
               enharmonic: bool = False):
    """Returns (musicxml bytes, info dict) for the score transposed by `semitones`.

    `mtime` is only part of the cache key, so a re-transcribed or edited score isn't served stale."""
    import copy

    return present(copy.deepcopy(base_score(rel, detail, mtime, lyrics, harmony)), semitones, clef_name, bars,
                   enharmonic)


def present(score, semitones: int, clef_name: str = "treble", bars: int = 0, enharmonic: bool = False):
    """A score as shown: transposed by `semitones`, in the chosen clef, `bars` bars per line.
    The key is spelled with the fewest accidentals (Db major, not C# major), or with
    `enharmonic` its other spelling where there is one (Gb for F#, Cb for B...).
    Returns (musicxml bytes, info dict). Changes `score`."""
    from music21 import clef, interval, key, layout, stream
    from music21.musicxml.m21ToXml import GeneralObjectExporter

    ks = next(iter(score.recurse().getElementsByClass(key.KeySignature)), key.KeySignature(0))

    base = interval_for(semitones)
    d2 = interval.Interval("d2")
    options = [base, interval.add([base, d2]), interval.add([base, d2.reverse()])]
    sharps = {i.name: ks.transpose(i).sharps for i in options}
    options = [i for i in options if abs(sharps[i.name]) <= 7]
    iv = min(options, key=lambda i: (abs(sharps[i.name]), i is not base))
    alt = next((i for i in options if abs(sharps[i.name] - sharps[iv.name]) == 12), None)
    if enharmonic and alt:
        iv, alt = alt, iv
    if iv.name != "P1":
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

    # A fixed number of bars per line: a line break before every `bars`-th bar.
    if bars:
        for part in score.parts:
            for i, m in enumerate(part.getElementsByClass(stream.Measure)):
                for old in list(m.getElementsByClass(layout.SystemLayout)):
                    m.remove(old)
                if i and i % bars == 0:
                    m.insert(0, layout.SystemLayout(isNew=True))

    pitches = [p.midi for p in score.pitches]
    new_ks = next(iter(score.recurse().getElementsByClass(key.KeySignature)), key.KeySignature(0))
    info = {
        "semitones": semitones,
        "interval": f"{'up' if semitones > 0 else 'down'} {base.name}" if semitones else "none",
        "keySig": key_name(new_ks.sharps),
        "keySharps": new_ks.sharps,
        "altKeySig": key_name(sharps[alt.name]) if alt else None,   # the other spelling, if any
        "altSharps": sharps[alt.name] if alt else None,
        "low": midi_name(min(pitches)) if pitches else "",
        "high": midi_name(max(pitches)) if pitches else "",
        "belowAlto": sum(p < ALTO_RANGE[0] for p in pitches),
        "aboveAlto": sum(p > ALTO_RANGE[1] for p in pitches),
        "bpm": score_bpm(score),
        # MusicXML <transpose>: from the written notes to how they sound.
        "transpose": transpose_element(iv) if iv.semitones else None,
    }
    return add_extenders(GeneralObjectExporter(score).parse(), score), info


def transpose_element(iv) -> dict:
    """The MusicXML <transpose> values for a part written `iv` above concert pitch
    (alto sax: written a M6 up, so diatonic -5, chromatic -9)."""
    semis = -iv.semitones
    steps = -(iv.generic.undirected - 1) * (1 if iv.semitones >= 0 else -1)
    octave = int(semis / 12)
    return {"diatonic": steps - 7 * octave, "chromatic": semis - 12 * octave, "octave": octave}


def musicxml_export(xml: bytes, info: dict) -> bytes:
    """The score as shown, marked as a transposing part (so notation programs play it at
    concert pitch and can switch it back)."""
    tr = info.get("transpose")
    if not tr:
        return xml
    element = (f"<transpose><diatonic>{tr['diatonic']}</diatonic><chromatic>{tr['chromatic']}</chromatic>"
               + (f"<octave-change>{tr['octave']}</octave-change>" if tr["octave"] else "") + "</transpose>")
    out, parts = [], xml.decode().split("<part id=")
    for i, part in enumerate(parts):    # into each part's first <attributes>
        out.append(part.replace("</attributes>", element + "</attributes>", 1) if i else part)
    return "<part id=".join(out).encode()


def midi_export(rel: str, detail: str, harmony: bool) -> bytes:
    """The score at concert pitch, at the song's tempo, as a MIDI file."""
    import copy
    from music21 import midi, tempo

    score = copy.deepcopy(base_score(rel, detail, score_mtime(rel, detail), False, harmony))
    bpm = (sidecar(rel) or {}).get("bpm") or score_bpm(score)
    for part in score.parts:
        part.insert(0, tempo.MetronomeMark(number=round(bpm, 2)))
    return midi.translate.music21ObjectToMidiFile(score).writestr()


def add_extenders(xml: bytes, score) -> bytes:
    """Write the lyric extender lines song.events_score marks (a held word's underline) into
    the MusicXML, which music21 doesn't export: <extend> on the word, and on its last note."""
    import re

    marks = {n.id: n.editorial.get("extend") for n in score.recurse().notes if n.editorial.get("extend")}
    if not marks:
        return xml
    # A last note split at a barline has both halves marked; the line ends at the later one.
    for nid in [i for i, k in marks.items() if k == "stop" and any(j.startswith(i + "-b") for j in marks)]:
        del marks[nid]

    def mark(m):
        note_xml, kind = m.group(0), marks.get(m.group(1))
        if kind == "start" and "</text>" in note_xml:
            return note_xml.replace("</text>", '</text>\n<extend type="start" />', 1)
        if kind == "stop":
            return note_xml.replace("</note>", '<lyric number="1"><extend type="stop" /></lyric>\n</note>')
        return note_xml
    return re.sub(rb'<note id="([^"]+)">.*?</note>'.decode(), mark, xml.decode(), flags=re.S).encode()


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


PAGE_W, PAGE_H, MARGIN_X = 2100, 2970, 100


def render_svgs(xml: bytes, screen: bool, timing: bool = False, bars: int = 0, style: str = "classic"):
    """Render pages to SVG. With timing=True also returns note/measure timing for playback,
    taken from the same Verovio load so the element ids match the SVG.

    `bars` > 0 sets that many bars per line, all the same width so barlines line up in columns
    down the page (see equal_bars). When the busiest bar wouldn't fit, the music is laid out
    on a wider page shown at the same size - everything (notes and words) draws smaller
    instead of squeezing together. `style` is a key of STYLES."""
    import verovio

    scale = 45 if screen else 40
    look = STYLES.get(style, STYLES["classic"])

    def toolkit(width, breaks, justify=True):
        # Explicit resource path: Verovio can't find its fonts from non-main threads otherwise.
        tk = verovio.toolkit(False)
        tk.setResourcePath(str(Path(verovio.__file__).parent / "data"))
        zoom = width / PAGE_W        # a wider layout page, drawn at the usual page size
        tk.setOptions({
            "pageWidth": width, "pageHeight": round(PAGE_H * zoom), "scale": scale / zoom,
            "pageMarginTop": round(80 * zoom), "pageMarginBottom": round(80 * zoom),
            "pageMarginLeft": MARGIN_X, "pageMarginRight": MARGIN_X,
            "adjustPageHeight": screen, "svgViewBox": screen, "footer": "none",
            "breaks": breaks, "noJustification": not justify, "font": look["music"], **LYRIC_LAYOUT,
            **look["lines"],
        })
        tk.loadData(xml.decode())
        return tk

    with _render_lock:
        width = PAGE_W
        if bars:
            # One endless unjustified line gives every bar's minimum comfortable width.
            natural = toolkit(PAGE_W, "none", justify=False).renderToSVG(1)
            need = bars_width(natural, bars)
            width = max(PAGE_W, int(need / 10) + 2 * MARGIN_X + 1)
        # "line": keep the encoded line breaks (every `bars` bars), paginate automatically.
        tk = toolkit(width, "line" if bars else "auto")
        svgs = [tk.renderToSVG(i) for i in range(1, tk.getPageCount() + 1)]
        if bars:
            svgs = equal_bars(svgs, bars, (width - 2 * MARGIN_X) * 10)
        if look["text"]:   # Verovio lays text out in Times; draw it in the style's handwriting
            svgs = [s.replace('font-family="Times, serif"', f'font-family="\'{look["text"][0]}\', Times, serif"')
                    for s in svgs]
        svgs = [align_lyrics(svg, look["text"] and look["text"][1]) for svg in svgs]
        if look.get("hand"):
            svgs = [pennify(svg, screen) for svg in svgs]
        if not timing:
            return svgs
        return svgs, playback_timing(tk)


_SVG_NS = "http://www.w3.org/2000/svg"
_MOVABLE = {"note", "chord", "rest", "mRest", "clef", "keySig", "meterSig", "mSpace", "space"}
_HEADER = {"clef", "keySig", "meterSig"}
_NUM = r"-?\d+(?:\.\d+)?"


def _cls(el):
    return (el.get("class") or "").split(" ")[0]


def _first_x(el):
    """x of the first positioned thing in an SVG group (a glyph's translate or a path's M)."""
    import re
    for d in el.iter():
        m = re.search(rf"translate\(({_NUM})", d.get("transform", ""))
        if m:
            return float(m.group(1))
        m = re.match(rf"M\s*({_NUM})", d.get("d", ""))
        if m:
            return float(m.group(1))
    return None


def _measures(root):
    """[(system, [(measure, x0, x1, content_start)])]: each bar's staff-line span, and where its
    music starts after a line's opening clef / key / time signature."""
    import re
    g = f"{{{_SVG_NS}}}g"
    systems = []
    for system in root.iter(g):
        if _cls(system) != "system":
            continue
        ms = []
        for i, m in enumerate(el for el in system if _cls(el) == "measure"):
            xs = [float(v) for st in m if _cls(st) == "staff" for path in st if path.tag.endswith("path")
                  for v in re.findall(rf"[ML]\s*({_NUM})", path.get("d", ""))]
            if not xs:
                continue
            x0, x1 = min(xs), max(xs)
            c0 = x0
            if i == 0:
                ends = [_first_x(u) for h in m.iter(g) if _cls(h) in _HEADER for u in h.iter() if u.get("transform")]
                ends = [x for x in ends if x is not None]
                if ends:
                    c0 = max(ends) + 300     # past the last header glyph
            ms.append((m, x0, x1, c0))
        if ms:
            systems.append((system, ms))
    return systems


def bars_width(natural_svg: str, bars: int) -> float:
    """Line width (SVG units) needed for `bars` bars each as wide as the busiest bar, measured
    from Verovio's single unjustified line, plus the clef / key / time signature at the start."""
    import xml.etree.ElementTree as ET
    systems = _measures(ET.fromstring(natural_svg))
    if not systems:
        return 0
    ms = [m for _, sys_ms in systems for m in sys_ms]
    header = ms[0][3] - ms[0][1]
    return header + bars * max(x1 - c0 for _, _, x1, c0 in ms)


def equal_bars(svgs, bars, avail):
    """Make every bar the same width, so barlines line up in columns down every page.

    Each note, rest or clef moves as one piece (with its accidentals, dots and lyric), staff
    lines and barlines move to the new bar edges, and beams, ties and slurs are reshaped to
    follow the notes they join. `avail` is the line width in SVG units, already wide enough
    (bars_width) that no bar ends up narrower than its natural width."""
    import re
    import xml.etree.ElementTree as ET

    import numpy as np

    SVG = _SVG_NS
    ET.register_namespace("", SVG)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    g = f"{{{SVG}}}g"
    cls, first_x, num = _cls, _first_x, _NUM
    MOVABLE = _MOVABLE
    roots = [ET.fromstring(svg) for svg in svgs]
    systems = [sm for root in roots for sm in _measures(root)]
    if not systems:
        return svgs
    start = min(ms[0][1] for _, ms in systems)
    header = max(ms[0][3] - start for _, ms in systems)
    unit = (avail - header) / bars

    def edit(el, fx):
        """Map every x coordinate in `el` itself (not its children) through fx."""
        if el.get("transform"):
            el.set("transform", re.sub(rf"translate\(({num})",
                                       lambda m: f"translate({fx(float(m.group(1))):.0f}", el.get("transform"), 1))
        if el.get("d") and el.tag.endswith("path"):
            out, cmd, k = [], "M", 0
            for tok in re.findall(rf"[A-Za-z]|{num}", el.get("d")):
                if tok.isalpha():
                    cmd, k = tok, 0
                    out.append(tok)
                    continue
                v = float(tok)
                if cmd in "MLCQST":        # absolute x/y pairs (all Verovio emits)
                    v = fx(v) if k % 2 == 0 else v
                elif cmd == "H":
                    v = fx(v)
                k += 1
                out.append(f"{v:.0f}")
            el.set("d", " ".join(out))
        if el.get("points"):
            pts = [float(v) for v in re.findall(num, el.get("points"))]
            el.set("points", " ".join(f"{fx(pts[i]):.0f},{pts[i + 1]:.0f}" for i in range(0, len(pts) - 1, 2)))
        for attr in ("x", "cx", "x1", "x2"):
            if el.get(attr) and re.fullmatch(num, el.get(attr)):
                el.set(attr, f"{fx(float(el.get(attr))):.0f}")

    def shift_all(el, d):
        for sub in el.iter():
            edit(sub, lambda x: x + d)

    for system, ms in systems:
        ctrl, moves = [], []      # control points (x, shift) for reshaping; groups to move
        for j, (m, x0, x1, c0) in enumerate(ms):
            X0 = start if j == 0 else start + header + j * unit
            C0 = start + header if j == 0 else X0
            X1 = start + header + (j + 1) * unit
            k = (X1 - C0) / max(x1 - c0, 1)
            place = lambda x: x if x < c0 else C0 + (x - c0) * k
            ctrl += [(x0, X0 - x0), (x1, X1 - x1)]
            if j == 0 and c0 > x0:
                ctrl.append((c0, C0 - c0))
            inside = set()          # descendants of a group already moving (a chord's notes)
            for el in m.iter(g):
                if cls(el) in MOVABLE and id(el) not in inside:
                    inside.update(id(sub) for sub in el.iter())
                    a = first_x(el)
                    if a is None:
                        continue
                    d = place(a) - a
                    moves.append((el, d))
                    ctrl.append((a, d))
                    for stem in el.iter(g):        # beams attach at stems: pin them too
                        if cls(stem) == "stem":
                            sx = first_x(stem)
                            if sx is not None:
                                ctrl.append((sx, d))
        ctrl.sort()
        xs, ds = np.array([c[0] for c in ctrl]), np.array([c[1] for c in ctrl])
        fx = lambda x: x + float(np.interp(x, xs, ds))
        moved = set()
        for el, d in moves:
            shift_all(el, d)
            moved.update(id(sub) for sub in el.iter())
        for el in system.iter():
            if id(el) in moved:
                continue
            if cls(el).startswith("ledgerLines"):
                for path in el:      # a ledger line belongs to one note: move it whole
                    m = re.findall(rf"[ML]\s*({num})", path.get("d", ""))
                    if m:
                        mid = (float(m[0]) + float(m[-1])) / 2
                        shift_all(path, fx(mid) - mid)
                        moved.add(id(path))
                moved.add(id(el))
                continue
            edit(el, fx)
    return [ET.tostring(root, encoding="unicode") for root in roots]


# Slightly smaller lyrics, and more room between words, so notes spread out enough for each
# word to fit between its note and the next word (align_lyrics shrinks the few that don't).
LYRIC_LAYOUT = {"lyricSize": 4.0, "lyricWordSpace": 5}


def _absolute_path(d: str):
    """An SVG path as absolute segments [(cmd, [x, y, ...])] using only M, L, C, Q and Z
    (glyph outlines use relative curves, smooth curves and h/v lines)."""
    import re

    tokens = re.findall(r"[MmLlHhVvCcSsQqTtZz]|-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?", d)
    out, i, cmd = [], 0, None
    x = y = sx = sy = 0.0
    last_ctrl = None   # previous control point, for reflecting into S/T curves

    def nums(n):
        nonlocal i
        vals = [float(t) for t in tokens[i:i + n]]
        i += n
        return vals

    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                out.append(("Z", []))
                x, y, last_ctrl = sx, sy, None
                continue
        rel = cmd.islower()
        c = cmd.upper()
        ox, oy = (x, y) if rel else (0.0, 0.0)
        if c == "M":
            px, py = nums(2)
            x, y = ox + px, oy + py
            sx, sy = x, y
            out.append(("M", [x, y]))
            cmd, last_ctrl = ("l" if rel else "L"), None
        elif c == "L":
            px, py = nums(2)
            x, y, last_ctrl = ox + px, oy + py, None
            out.append(("L", [x, y]))
        elif c == "H":
            (px,) = nums(1)
            x, last_ctrl = ox + px, None
            out.append(("L", [x, y]))
        elif c == "V":
            (py,) = nums(1)
            y, last_ctrl = (y if rel else 0) + py, None
            out.append(("L", [x, y]))
        elif c in "CS":
            if c == "C":
                x1, y1, x2, y2, px, py = nums(6)
                c1 = (ox + x1, oy + y1)
            else:
                x2, y2, px, py = nums(4)
                c1 = (2 * x - last_ctrl[0], 2 * y - last_ctrl[1]) if last_ctrl else (x, y)
            p = [c1[0], c1[1], ox + x2, oy + y2, ox + px, oy + py]
            out.append(("C", p))
            last_ctrl, x, y = (p[2], p[3]), p[4], p[5]
        elif c in "QT":
            if c == "Q":
                x1, y1, px, py = nums(4)
                c1 = (ox + x1, oy + y1)
            else:
                px, py = nums(2)
                c1 = (2 * x - last_ctrl[0], 2 * y - last_ctrl[1]) if last_ctrl else (x, y)
            p = [c1[0], c1[1], ox + px, oy + py]
            out.append(("Q", p))
            last_ctrl, x, y = c1, p[2], p[3]
        else:
            i += 1
    return out


# Photocopy grain on screen, in the SVG's units (~20 per screen pixel): noise frequency, how far
# it pushes an edge, and the softening before the hard cut.
GRAIN = {"soften": 1.5, "grain": 0.32, "amount": 15}


def pennify(svg: str, screen: bool = True, variants: int = 4) -> str:
    """Draw the page as if by a copyist's pen, for the Jazz look:

    - every ruled line (staff lines, stems, barlines, ledger lines) becomes a filled ink stroke
      with ragged edges whose width swells and thins along it; staff lines drift gently
      (smoothly along each line, so the separately drawn bar segments still join up);
    - beams get rough outlines;
    - glyphs (noteheads, flags, rests, clefs, accidentals...) come in `variants` hand-warped
      copies - wobbly outlines with a little grit, so no two notes are quite the same shape -
      and each one drawn gets a copy, picked by its position, plus a slight tilt;
    - on screen, an ink-bleed texture roughens every edge (PDFs can't show SVG filters).

    All of it is seeded from positions, so a page looks the same every time it's drawn. Runs
    after layout, so nothing the lyric alignment or equal-width bars measured moves."""
    import math
    import random
    import re
    import xml.etree.ElementTree as ET

    SVG, XL = "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"
    ET.register_namespace("", SVG)
    ET.register_namespace("xlink", XL)
    root = ET.fromstring(svg)
    q = lambda tag: f"{{{SVG}}}{tag}"
    # (equal-width bars rewrites these as "M 0 1207 L 9705 1207")
    line_re = re.compile(r"^M\s*(-?[\d.]+)[ ,]+(-?[\d.]+)\s*L\s*(-?[\d.]+)[ ,]+(-?[\d.]+)\s*$")
    rng = lambda *key: random.Random(":".join(k if isinstance(k, str) else str(round(k)) for k in key))

    def wave(r, scale):
        """A smooth random function of one variable: a few sine waves with random phases."""
        ph = [r.uniform(0, 6.3) for _ in range(3)]
        return lambda t: (math.sin(t / scale + ph[0]) + 0.5 * math.sin(t / (scale * 0.37) + ph[1])
                          + 0.25 * math.sin(t / (scale * 0.13) + ph[2])) / 1.75

    def ink_stroke(x1, y1, x2, y2, width, r, drift=None, step=45, pressure_amt=0.38, rough=0.55, bow=0.0):
        """A pen line as a filled outline: ragged edges (`rough`), width swelling and thinning
        (`pressure_amt`), bowed sideways by `bow` at its middle. `drift(x)` moves it sideways
        as a function of absolute x (staff lines, so a line's bar segments join up)."""
        length = math.hypot(x2 - x1, y2 - y1) or 1
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        nx, ny = -uy, ux
        n = max(3, int(length / step))
        pressure = wave(r, max(length / 2.5, 40))

        def grit():   # edge roughness: random, then smoothed so it reads as ink, not hair
            g = [r.uniform(-1, 1) for _ in range(n + 1)]
            for _ in range(2):
                g = [(g[max(i - 1, 0)] + 2 * g[i] + g[min(i + 1, n)]) / 4 for i in range(n + 1)]
            return g
        gl, gr = grit(), grit()
        left, right = [], []
        for i in range(n + 1):
            s = length * i / n
            px, py = x1 + ux * s, y1 + uy * s
            off = (drift(px) if drift else 0) + bow * math.sin(math.pi * s / length)
            half = width / 2 * (1 + pressure_amt * pressure(s))
            el, er = half * (1 + rough * gl[i]), half * (1 + rough * gr[i])
            cx, cy = px + nx * off, py + ny * off
            left.append((cx + nx * el, cy + ny * el))
            right.append((cx - nx * er, cy - ny * er))
        return "M" + " L".join(f"{x:.0f} {y:.0f}" for x, y in left + right[::-1]) + "Z"

    def drift_for(y):
        r = rng("staffline", y)
        w, tilt = wave(r, 2400), r.uniform(-1, 1) * 0.0003
        return lambda x: 3 * w(x) + tilt * x

    parent = {c: p for p in root.iter() for c in p}
    cls = lambda el: (el.get("class") or "").split(" ")[0]

    # ---- uneven spacing, like a hand: each note (with its stem, accidentals, dots, words)
    # moves sideways a little - never more than a fraction of the gap to its neighbours, and
    # less under lyrics - and beam ends and ledger lines follow their notes.
    use_x = lambda el: next((float(m.group(1)) for u in el.iter(q("use"))
                             for m in [re.search(r"translate\((-?[\d.]+),", u.get("transform", ""))] if m), None)
    has_words = any(cls(g) == "syl" for g in root.iter(q("g")))
    stem_dx, head_dx = {}, []
    for layer in (g for g in root.iter(q("g")) if cls(g) == "layer"):
        items = []
        for el in layer.iter(q("g")):
            if cls(el) not in ("note", "chord", "rest"):
                continue
            up, nested = parent.get(el), False
            while up is not None and up is not layer:   # a note inside a chord moves with it
                nested |= cls(up) == "chord"
                up = parent.get(up)
            x = use_x(el)
            if not nested and x is not None:
                items.append((x, el))
        items.sort(key=lambda t: t[0])
        for i, (x, el) in enumerate(items):
            gaps = [abs(x - items[j][0]) for j in (i - 1, i + 1) if 0 <= j < len(items)] or [400]
            room = min(min(gaps) * (0.12 if has_words else 0.22), 26 if has_words else 46)
            dx = rng("space", x, use_x(el) or 0).uniform(-1, 1) * room
            if abs(dx) < 1:
                continue
            el.set("transform", f"translate({dx:.0f}, 0)")
            head_dx.append((x, dx))
            for st in el.iter(q("g")):
                if cls(st) == "stem":
                    for p in st.iter(q("path")):
                        m = re.match(r"M\s*(-?[\d.]+)", p.get("d", ""))
                        if m:
                            stem_dx[float(m.group(1))] = dx
    for g in root.iter(q("g")):
        if cls(g) == "beam":   # beam corners sit on stem x positions: move them with the stems
            for poly in g.findall(q("polygon")):
                pts = [tuple(map(float, p.split(","))) for p in poly.get("points", "").split()]
                moved = []
                for px, py in pts:
                    near = min(stem_dx, key=lambda sx: abs(sx - px), default=None)
                    moved.append((px + (stem_dx[near] if near is not None and abs(near - px) < 20 else 0), py))
                poly.set("points", " ".join(f"{px:.0f},{py:.0f}" for px, py in moved))
        elif cls(g) == "ledgerLines":   # a ledger line follows the note it sits under
            for p in g.iter(q("path")):
                m = re.match(r"^M\s*(-?[\d.]+)[ ,]+(-?[\d.]+)\s*L\s*(-?[\d.]+)[ ,]+(-?[\d.]+)\s*$", p.get("d", ""))
                if not m:
                    continue
                x1, y1, x2, y2 = map(float, m.groups())
                inside = [dx for hx, dx in head_dx if x1 - 30 <= hx <= x2]
                if inside:
                    p.set("d", f"M{x1 + inside[0]:.0f} {y1:.0f} L{x2 + inside[0]:.0f} {y2:.0f}")

    def in_beam(el):
        while el is not None:
            if (el.get("class") or "").split(" ")[0] == "beam":
                return True
            el = parent.get(el)
        return False

    # ---- ruled lines -> ink strokes
    for g in root.iter(q("g")):
        kind = (g.get("class") or "").split(" ")[0]
        for el in list(g):
            tag = el.tag.split("}")[1]
            if tag == "polyline" and el.get("stroke-width"):   # tuplet brackets and the like
                pts = [tuple(map(float, p.split(","))) for p in el.get("points", "").split()]
                w = float(el.get("stroke-width"))
                r = rng(kind, *pts[0])
                d = " ".join(ink_stroke(*a, *b, w * r.uniform(0.95, 1.2), r, step=30, pressure_amt=0.15, rough=0.35)
                             for a, b in zip(pts, pts[1:]))
                g.remove(el)
                ET.SubElement(g, q("path"), {"d": d, "fill": "currentColor", "style": "stroke:none"})
                continue
            if kind in ("slur", "tie") and tag == "path":   # filled curves: a slight hand wobble
                r = rng(kind, *map(float, re.findall(r"-?[\d.]+", el.get("d", ""))[:2] or [0]))
                wx, wy = wave(r, 700), wave(r, 700)
                el.set("d", " ".join(cmd + " ".join(f"{c:.0f}" for i in range(0, len(p), 2)
                                                    for c in (p[i] + 6 * wx(p[i + 1]), p[i + 1] + 9 * wy(p[i])))
                                     for cmd, p in _absolute_path(el.get("d", ""))))
                continue
            if kind == "beam" and tag == "polygon":
                pts = [tuple(map(float, p.split(","))) for p in el.get("points", "").split()]
                if len(pts) == 4:
                    # A beam stays a parallelogram whose ends sit on the stems; the hand shows
                    # only in a slight bow and an uneven ink edge (both edges move together, so
                    # the thickness holds).
                    r = rng("beam", *pts[0])
                    (ax, ay), (bx, by), (cx, cy), (dx, dy) = pts
                    n = max(6, int(abs(bx - ax) / 50))
                    bow, sl, sr = r.uniform(-7, 7), r.uniform(-4, 4), r.uniform(-4, 4)
                    et, eb = wave(r, 180), wave(r, 180)
                    def shift(t):   # same for top and bottom edge at a point t (0..1) along the beam
                        return sl * (1 - t) + sr * t + bow * math.sin(math.pi * t)
                    top = [(ax + (bx - ax) * i / n, ay + (by - ay) * i / n + shift(i / n) + 3 * et(i * 50))
                           for i in range(n + 1)]
                    bot = [(dx + (cx - dx) * i / n, dy + (cy - dy) * i / n + shift(i / n) + 3 * eb(i * 50 + 700))
                           for i in range(n + 1)]
                    el.set("points", " ".join(f"{x:.0f},{y:.0f}" for x, y in top + bot[::-1]))
                    el.set("fill-opacity", f"{r.uniform(0.9, 1):.2f}")
                continue
            m = tag == "path" and line_re.match(el.get("d", ""))
            if not m:
                continue
            x1, y1, x2, y2 = map(float, m.groups())
            w = float(el.get("stroke-width", "18"))
            r = rng(kind, x1, y1, x2, y2)
            if kind == "staff" and abs(y2 - y1) < 1:
                el.set("data-line", f"{x1:.0f} {y1:.0f} {x2:.0f}")   # where it was ruled (for bar boxes)
                d = ink_stroke(x1, y1, x2, y2, w * r.uniform(0.9, 1.1), r, drift=drift_for(y1), step=60,
                               pressure_amt=0.06, rough=0.2)
            elif kind == "stem":
                length, beamed = abs(y2 - y1), in_beam(g)
                # a little off-plumb; a beamed stem only barely, so its tip still meets the beam
                lean = r.uniform(-4, 4) if beamed else r.choice((-1, 1)) * length * r.uniform(0.015, 0.045)
                over = r.uniform(5, 30) if beamed and r.random() < 0.6 else 0   # poking past the beam
                y2 += math.copysign(over, y2 - y1)
                d = ink_stroke(x1, y1, x2 + lean, y2, w * r.uniform(1.0, 1.3), r, step=30,
                               pressure_amt=0.2, rough=0.3, bow=r.uniform(-1, 1) * length * 0.015)
            elif kind == "barLine":   # a steady width; it's the lean and the pen edge that vary
                length = abs(y2 - y1)
                lean = r.uniform(-1, 1) * length * 0.025
                d = ink_stroke(x1, y1, x2 + lean, y2, w * r.uniform(0.95, 1.08), r, step=30,
                               pressure_amt=0.08, rough=0.3, bow=r.uniform(-1, 1) * length * 0.02)
            elif kind == "ledgerLines":
                # a short, heavy marker dash, nearly level; a stack is only roughly even
                tilt, nudge = r.uniform(-4, 4), r.uniform(-9, 9)
                x1, x2 = x1 - r.uniform(-12, 14), x2 + r.uniform(-12, 14)
                d = ink_stroke(x1, y1 + nudge - tilt, x2, y2 + nudge + tilt, w * r.uniform(1.8, 2.3), r,
                               step=25, pressure_amt=0.12, rough=0.25, bow=r.uniform(-3, 3))
            else:                       # any other ruled line: brackets, extender lines...
                d = ink_stroke(x1, y1, x2, y2, w * r.uniform(0.9, 1.3), r)
            el.set("d", d)
            el.set("fill", "currentColor")
            el.set("style", "stroke:none")   # Verovio's stylesheet strokes every path
            el.set("fill-opacity", f"{r.uniform(0.8, 1):.2f}")
            el.attrib.pop("stroke-width", None)

    # ---- glyphs -> several hand-warped copies each
    defs = root.find(q("defs"))
    centre = {}   # glyph id -> the middle of its outline, in glyph units
    if defs is not None:
        for glyph in list(defs):
            gid, path = glyph.get("id"), glyph.find(q("path"))
            if not gid or path is None:
                continue
            segs = _absolute_path(path.get("d", ""))
            xs = [p[i] for _, p in segs for i in range(0, len(p), 2)]
            ys = [p[i] for _, p in segs for i in range(1, len(p), 2)]
            if xs:
                centre[gid] = ((min(xs) + max(xs)) / 2, -(min(ys) + max(ys)) / 2)   # paths are drawn flipped
            for k in range(1, variants):
                r = rng("glyph", gid.split("-")[0], k)
                wx, wy = wave(r, 260), wave(r, 260)
                d = []
                for cmd, pts in segs:
                    moved = []
                    for i in range(0, len(pts), 2):
                        x, y = pts[i], pts[i + 1]   # a smooth wobble of the outline, plus grit
                        moved += [x + 12 * wx(y + 400) + r.uniform(-4, 4), y + 12 * wy(x + 900) + r.uniform(-4, 4)]
                    d.append(cmd + " ".join(f"{c:.0f}" for c in moved))
                copy = ET.SubElement(defs, glyph.tag, dict(glyph.attrib, id=f"{gid}-v{k}"))
                ET.SubElement(copy, path.tag, dict(path.attrib, d=" ".join(d)))
        for use in root.iter(q("use")):
            href = use.get(f"{{{XL}}}href", "")
            m = re.search(r"translate\((-?[\d.]+), (-?[\d.]+)\)", use.get("transform", ""))
            if not href.startswith("#") or not m:
                continue
            r = rng("use", *map(float, m.groups()))
            k = r.randrange(variants)
            if k:
                use.set(f"{{{XL}}}href", f"{href}-v{k}")
            sm = re.search(r"scale\(([\d.]+), ([\d.]+)\)", use.get("transform", ""))
            if not sm:
                continue
            sx, sy = float(sm.group(1)), float(sm.group(2))
            base_id = href[1:].split("-v")[0]
            if (parent.get(use) is not None and (parent[use].get("class") or "") == "notehead"
                    and base_id in centre):
                # a copyist's notehead: a long, steeply slanted blob, stretched and turned
                # about its own middle so it stays where the stem meets it
                gx, gy = centre[base_id]
                ox, oy = gx * sx, gy * sy
                tilt = r.uniform(-20, -6)
                sx2, sy2 = sx * r.uniform(1.04, 1.16), sy * r.uniform(0.86, 0.96)
                use.set("transform", f"translate({float(m.group(1)) + ox:.0f}, {float(m.group(2)) + oy:.0f}) "
                                     f"rotate({tilt:.1f}) scale({sx2:.3f}, {sy2:.3f}) translate({-gx:.0f}, {-gy:.0f})")
                continue
            else:   # a rest is a short pen stroke: it can't be off by much; other glyphs a little
                rest = parent.get(use) is not None and (parent[use].get("class") or "").split(" ")[0] in ("rest", "mRest", "multiRest")
                size = r.uniform(0.98, 1.02) if rest else r.uniform(0.95, 1.06)
                sx, sy, turn = sx * size, sy * size, f"rotate({r.uniform(-1.5, 1.5) if rest else r.uniform(-3, 3):.1f})"
            use.set("transform", f"translate({m.group(1)}, {m.group(2)}) {turn} scale({sx:.3f}, {sy:.3f})")

    # ---- on screen: photocopy grain. Fine noise pushes every edge pixel about, then a hard
    # alpha cut turns the result into speckled, broken edges like a copied hand-written chart.
    # It only moves ink that's there, so colours (highlighted, selected, unsure notes) stay.
    if screen and defs is not None:
        fid = f"ink-{root.get('id', 'page')}"
        f = ET.SubElement(defs, q("filter"), {"id": fid, "x": "-1%", "y": "-1%", "width": "102%", "height": "102%",
                                             "color-interpolation-filters": "sRGB"})
        ET.SubElement(f, q("feTurbulence"), {"type": "fractalNoise", "baseFrequency": str(GRAIN["grain"]),
                                             "numOctaves": "2", "seed": "5", "result": "noise"})
        ET.SubElement(f, q("feDisplacementMap"), {"in": "SourceGraphic", "in2": "noise", "scale": str(GRAIN["amount"]),
                                                  "xChannelSelector": "R", "yChannelSelector": "G", "result": "moved"})
        ET.SubElement(f, q("feGaussianBlur"), {"in": "moved", "stdDeviation": str(GRAIN["soften"]), "result": "soft"})
        ct = ET.SubElement(f, q("feComponentTransfer"), {"in": "soft"})
        ET.SubElement(ct, q("feFuncA"), {"type": "table", "tableValues": "0 0.2 0.9 1 1 1"})   # firm, keeps thin lines
        for page in root.iter(q("g")):
            if (page.get("class") or "") == "page-margin":
                page.set("filter", f"url(#{fid})")
    return ET.tostring(root, encoding="unicode")


@lru_cache(maxsize=4)
def _lyric_font(path: Path | None = None):
    """The font the lyrics are drawn in, for measuring words: `path`, or the serif Verovio asks
    for ("Times, serif"); None if it isn't installed (then widths are estimated)."""
    from PIL import ImageFont
    for path in ([path] if path else []) + ["/System/Library/Fonts/Times.ttc",
                 "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(str(path), 1000)
    return None


def align_lyrics(svg: str, font_path: Path | None = None, min_scale=0.7) -> str:
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
    font = _lyric_font(font_path)
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
    key = {"stem": "stem", "lead": "lead_vocals", "harmony": "backing_vocals", "backing": "backing",
           "mix": "audio"}.get(which, "")
    path = Path(side.get(key, ""))
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


def render_pdf(xml: bytes, bars: int = 0, style: str = "classic") -> bytes:
    import cairosvg
    from pypdf import PdfWriter

    writer = PdfWriter()
    for svg in render_svgs(xml, screen=False, bars=bars, style=style):
        buf = io.BytesIO()
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=buf)
        buf.seek(0)
        writer.append(buf)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def encode_audio(wav: Path, fmt: str, codec: list) -> bytes:
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"mix.{fmt}"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), *codec, str(out)], check=True)
        return out.read_bytes()


def song_sites():
    import song
    return {"download": sorted(set(song.DOWNLOAD_SITES.values())), "lookup": sorted(set(song.LOOKUP_SITES.values()))}


# ---------------------------------------------------------------- transcription jobs
# One job at a time: Demucs already uses the whole GPU, and song.log_sink is global.

job = {"state": "idle"}
_job_lock = threading.Lock()


def start_job(source: str, part: str, bpm: float | None, harmony_for: str | None = None,
              split: dict | None = None) -> bool:
    """Transcribe a new song, add backing-vocal harmony to a score (`harmony_for`), or split a
    song into parts (`split` = {"model", "vocal_layers"})."""
    with _job_lock:
        if job.get("state") == "running":
            return False
        job.clear()
        kind = "harmony" if harmony_for else "split" if split else "song"
        job.update(state="running", kind=kind, log=[], source=source, part=part, result=None, error=None,
                   model=(split or {}).get("model"))
    threading.Thread(target=_run_job, args=(source, part, bpm, harmony_for, split), daemon=True).start()
    return True


def _run_job(source, part, bpm, harmony_for=None, split=None):
    import song  # imported lazily: song imports this module

    song.log_sink = job["log"]
    try:
        if split:
            audio = song.split_song(source, split.get("model", "htdemucs"), bool(split.get("vocal_layers")))
            job.update(state="done", result=audio.stem)
        elif harmony_for:
            song.add_harmony(resolve(harmony_for).with_suffix(".json"))
            job.update(state="done", result=harmony_for)
        else:
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
            if url.path == "/fonts/PetalumaScript.otf":   # the only file served from fonts/
                return self.send((FONTS / "PetalumaScript.otf").read_bytes(), "font/otf",
                                 extra={"Cache-Control": "max-age=31536000"})
            if url.path == "/favicon.ico":
                return self.send(b"", "image/x-icon", 204)
            if url.path == "/api/list":
                return self.send(json.dumps(list_scores()).encode(), "application/json")
            if url.path == "/api/job":
                return self.send(json.dumps(job).encode(), "application/json")
            if url.path == "/api/songs":
                return self.send(json.dumps({"songs": list_songs(), "sites": song_sites()}).encode(), "application/json")
            if url.path in ("/api/mix", "/api/mixdownload"):
                gains = {k: max(0.0, min(2.0, float(v))) for k, _, v in
                         (g.partition(":") for g in q.get("g", "").split(",") if ":" in g)}
                path = mix_file(q.get("song", ""), q.get("model", "htdemucs"), gains)
                if url.path == "/api/mix":
                    return self.send_file(path, "audio/wav")
                fmt = q.get("fmt") if q.get("fmt") in MIX_FORMATS else "wav"
                ctype, codec = MIX_FORMATS[fmt]
                body = path.read_bytes() if fmt == "wav" else encode_audio(path, fmt, codec)
                name = f"{q.get('song', 'song')} ({q.get('label') or 'mix'}).{fmt}"
                ascii_name = name.encode("ascii", "ignore").decode().replace('"', "'")
                return self.send(body, ctype, extra={"Content-Disposition":
                                 f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(name)}'})
            if url.path == "/api/sound":
                return self.send(soundfont(q.get("name", "")), "application/json",
                                 extra={"Cache-Control": "max-age=31536000"})
            rel, semis = q.get("f", ""), int(q.get("t", 0))
            bars = max(0, min(int(q.get("bars") or 0), 12))
            if url.path == "/api/audio":
                return self.send_audio(rel, q.get("which", "stem"))
            detail = q.get("detail") if q.get("detail") in ("simple", "balanced", "detailed") else "balanced"
            if url.path == "/api/events":
                if not sidecar(rel):
                    return self.send(b"This score can't be edited: re-run the song first.", "text/plain", 400)
                ev, edited = level_events(rel, detail)
                return self.send(json.dumps({"events": ev, "edited": edited}).encode(), "application/json")
            if url.path == "/api/history":
                if q.get("version"):
                    return self.send(json.dumps(history_version(rel, detail, q["version"])).encode(), "application/json")
                return self.send(json.dumps(history(rel, detail)).encode(), "application/json")
            xml, info = transposed(rel, semis, "bass" if q.get("clef") == "bass" else "treble",
                                   detail, score_mtime(rel, detail), q.get("lyrics") == "1",
                                   q.get("harmony") == "1", bars, q.get("enh") == "1")
            if url.path == "/api/render":
                return self.send_render(rel, detail, xml, info, bars, q.get("style", "classic"))
            if url.path in ("/api/pdf", "/api/export"):
                label = q.get("label", "")
                name = Path(rel).parent.name + f" - {Path(rel).stem}" + (f" ({label})" if label else "")
                fmt = "pdf" if url.path == "/api/pdf" else q.get("fmt", "musicxml")
                if fmt == "pdf":
                    body, ctype, ext = render_pdf(xml, bars, q.get("style", "classic")), "application/pdf", "pdf"
                elif fmt == "mid":   # concert pitch: what the score sounds like
                    name = Path(rel).parent.name + f" - {Path(rel).stem} (concert pitch)"
                    body, ctype, ext = midi_export(rel, detail, q.get("harmony") == "1"), "audio/midi", "mid"
                else:
                    body, ctype, ext = musicxml_export(xml, info), "application/vnd.recordare.musicxml+xml", "musicxml"
                ascii_name = name.replace("♭", "b").encode("ascii", "ignore").decode()
                disposition = f'attachment; filename="{ascii_name}.{ext}"; filename*=UTF-8\'\'{quote(name)}.{ext}'
                return self.send(body, ctype, extra={"Content-Disposition": disposition})
            self.send(b"not found", "text/plain", 404)
        except FileNotFoundError:
            self.send(b"score not found", "text/plain", 404)
        except Exception as e:  # surface errors in the page rather than hanging
            self.send(f"error: {e}".encode(), "text/plain", 500)


    def send_render(self, rel: str, detail: str, xml: bytes, info: dict, bars: int, style: str = "classic"):
        svgs, timing = render_svgs(xml, screen=True, timing=True, bars=bars, style=style)
        side = sidecar(rel)
        audio = None
        if side:
            audio = {"offset": side["audio_offset"],
                     "available": [w for w in ("stem", "lead", "harmony", "backing", "mix")
                                   if audio_path(side, w)]}
            info = {**info, "bpm": side["bpm"]}   # exact tempo - the subtitle's is rounded
        info["hasLyrics"] = bool(side and side.get("words"))
        info["hasHarmony"] = bool(side and "harmony" in side)
        info["canHarmony"] = bool(side and side["part"] == "vocals" and side["mono"])
        info["editable"] = bool(side)
        info["edited"] = bool(side) and level_file(rel, detail).exists()
        body = {"svgs": svgs, **timing, **info, "audio": audio,
                "part": side["part"] if side else Path(rel).stem}
        return self.send(json.dumps(body).encode(), "application/json")

    def send_audio(self, rel: str, which: str):
        side = sidecar(rel)
        path = side and audio_path(side, which)
        if not path:
            return self.send(b"no audio for this score - re-run it to enable", "text/plain", 404)
        return self.send_file(path, "audio/wav")

    def send_file(self, path: Path, ctype: str):
        """A file with HTTP Range support (audio elements seek with range requests)."""
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if a and b else size - 1
        end = min(end, size - 1)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
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
        if url.path in ("/api/render", "/api/edits"):
            try:
                return self.post_score(url.path, json.loads(body or b"{}"))
            except FileNotFoundError:
                return self.send(b"score not found", "text/plain", 404)
            except Exception as e:
                return self.send(f"error: {e}".encode(), "text/plain", 400)
        if url.path == "/api/job":
            req = json.loads(body or b"{}")
            if req.get("split"):
                split = req["split"]
                source = (req.get("source") or "").strip()
                if not source and split.get("song"):
                    try:
                        source = str(song_audio(split["song"]))
                    except FileNotFoundError:
                        return self.send(b"That song isn't downloaded any more.", "text/plain", 404)
                if not source:
                    return self.send(b"Paste a link or choose a file first.", "text/plain", 400)
                if not start_job(source, "", None, split=split):
                    return self.send(b"Another song is still processing.", "text/plain", 409)
                return self.send(b"{}", "application/json")
            if req.get("harmony"):
                if not sidecar(req["harmony"]):
                    return self.send(b"Re-run this song first: it was made before harmonies were supported.",
                                     "text/plain", 400)
                if not start_job("", "vocals", None, harmony_for=req["harmony"]):
                    return self.send(b"Another song is still processing.", "text/plain", 409)
                return self.send(b"{}", "application/json")
            source = (req.get("source") or "").strip()
            if not source:
                return self.send(b"Paste a link or choose a file first.", "text/plain", 400)
            bpm = float(req["bpm"]) if req.get("bpm") else None
            if not start_job(source, req.get("part", "vocals"), bpm):
                return self.send(b"Another song is still processing.", "text/plain", 409)
            return self.send(b"{}", "application/json")
        self.send(b"not found", "text/plain", 404)


    def post_score(self, path: str, req: dict):
        """POST /api/render draws an unsaved note list (the editor's draft) like GET /api/render.
        POST /api/edits saves a level's note list ({events}) or deletes its edits ({revert})."""
        import song

        rel = req.get("f", "")
        detail = req.get("detail") if req.get("detail") in song.DETAIL_LEVELS else "balanced"
        if not sidecar(rel):
            return self.send(b"This score can't be edited: re-run the song first.", "text/plain", 400)
        if path == "/api/edits":
            if req.get("revert"):
                level_file(rel, detail).unlink(missing_ok=True)
            else:
                save_edits(rel, detail, req["events"])
            return self.send(json.dumps({"edited": level_file(rel, detail).exists()}).encode(), "application/json")
        bars = max(0, min(int(req.get("bars") or 0), 12))
        score = song.events_score(req["events"], bool(req.get("lyrics")), bool(req.get("harmony")))
        xml, info = present(score, int(req.get("t", 0)), "bass" if req.get("clef") == "bass" else "treble", bars,
                            bool(req.get("enh")))
        return self.send_render(rel, detail, xml, info, bars, req.get("style", "classic"))


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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
  @font-face { font-family:"Petaluma Script"; src:url("/fonts/PetalumaScript.otf") format("opentype"); font-display:swap; }
  /* Practice room: cool grey desk, white score pages, engraving ink, sax-brass accent. */
  :root { --desk:#e5e8ed; --paper:#ffffff; --ink:#1f2633; --muted:#5b6474; --line:#d0d5dd;
          --brass:#94650f; --brass-wash:#f1dfae; --brass-light:#e8c66e; --warn:#a1301f;
          --serif:"Source Serif 4", "Iowan Old Style", Georgia, serif;
          --sans:-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--desk); color:var(--ink); font:14px/1.45 var(--sans); }
  :focus-visible { outline:2px solid var(--brass); outline-offset:2px; }
  button, select, input { font:inherit; color:inherit; }

  /* ---- toolbar */
  header { position:sticky; top:0; z-index:2; background:var(--paper); border-bottom:1px solid var(--line);
           padding:calc(14px + env(safe-area-inset-top, 0px)) 28px 14px; }
  .bar { display:flex; flex-wrap:wrap; align-items:center; gap:12px 16px; max-width:1320px; margin:0 auto; }
  .bar + .bar { margin-top:12px; }
  .title { display:flex; align-items:center; gap:14px; flex:1 1 360px; min-width:0; }
  #score { font:600 19px/1.2 var(--serif); border:0; background:transparent; padding:4px 26px 4px 2px;
           flex:1 1 auto; min-width:12ch; width:auto; max-width:640px; text-overflow:ellipsis; cursor:pointer; appearance:none;
           background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%235b6474' fill='none' stroke-width='1.5'/%3E%3C/svg%3E") right 8px center no-repeat; }
  #score:hover { color:var(--brass); }
  .actions { display:flex; gap:8px; flex-wrap:wrap; }
  .download { display:inline-flex; }
  button.quiet.round { min-width:34px; padding:0; border-radius:50%; font-weight:600; border-color:var(--line); color:var(--muted); }
  button.quiet.round:hover { color:var(--ink); border-color:#a9b0bc; }

  /* ---- "View" menu: settings you set once, kept out of the toolbar */
  #practiceBar { justify-content:flex-start; }
  .menu { position:relative; margin-left:auto; }
  .menu summary { list-style:none; cursor:pointer; border:1px solid var(--line); border-radius:8px; padding:6px 30px 6px 12px;
                  min-height:34px; display:flex; align-items:center; background:var(--paper) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%235b6474' fill='none' stroke-width='1.5'/%3E%3C/svg%3E") right 11px center no-repeat; }
  .menu summary::-webkit-details-marker { display:none; }
  .menu summary:hover { border-color:#a9b0bc; }
  .menu[open] summary { border-color:var(--brass-light); color:var(--brass); }
  .menu-panel { position:absolute; right:0; top:calc(100% + 8px); z-index:6; background:var(--paper);
                border:1px solid var(--line); border-radius:12px; padding:8px 16px;
                box-shadow:0 12px 32px rgba(31,38,51,.16); display:grid; min-width:300px; }
  .menu-panel .row { display:flex; align-items:center; justify-content:space-between; gap:24px; padding:10px 0; }
  .menu-panel .row + .row { border-top:1px solid var(--line); }
  .menu-panel .row[hidden] { display:none; }
  .download select { border-radius:8px 0 0 8px; border-right:0; }
  .download button { border-radius:0 8px 8px 0; }
  .group { display:flex; align-items:center; gap:7px; }
  .label { color:var(--muted); font-size:13px; }
  button { border:1px solid var(--line); background:var(--paper); border-radius:8px; padding:6px 12px;
           cursor:pointer; min-height:34px; }
  button:hover { border-color:#a9b0bc; }
  button:disabled { opacity:.5; cursor:default; }
  button.primary { background:var(--ink); border-color:var(--ink); color:#fff; }
  button.primary:hover { background:#323b4b; }
  header button.on { background:var(--brass-wash); border-color:var(--brass-light); color:var(--brass); font-weight:600; }
  button.quiet { border-color:transparent; background:transparent; color:var(--muted); }
  button.quiet:hover { color:var(--ink); border-color:var(--line); }
  /* Joined buttons for one-of-several choices */
  .seg { display:inline-flex; background:var(--desk); border-radius:9px; padding:2px; gap:2px; max-width:100%; overflow-x:auto; }
  .seg button { border:0; background:transparent; border-radius:7px; min-height:30px; padding:4px 10px; white-space:nowrap; }
  .seg button:hover { background:rgba(255,255,255,.7); }
  .seg button.on { background:var(--paper); color:var(--brass); font-weight:600;
                   box-shadow:0 0 0 1px var(--line), 0 1px 2px rgba(31,38,51,.08); }
  #barsN { min-width:34px; font-variant-numeric:tabular-nums; }
  select { border:1px solid var(--line); border-radius:8px; background:var(--paper); padding:6px 8px; min-height:34px; }

  /* ---- status line under the toolbar */
  #status { display:flex; flex-wrap:wrap; gap:6px 22px; max-width:1320px; margin:0 auto;
            padding:18px 28px 0; color:var(--muted); font-size:13px; }
  #status b { color:var(--ink); font-weight:600; }
  #status .warn { color:var(--warn); }
  #status button.inline { min-height:0; padding:0 6px; margin-left:4px; font-size:13px; color:var(--brass); text-decoration:underline; text-underline-offset:2px; }
  kbd { font:11px var(--sans); border:1px solid var(--line); border-bottom-width:2px; border-radius:4px;
        padding:0 5px; background:var(--paper); }

  /* ---- new song panel */
  #new { max-width:1180px; margin:20px auto 0; padding:0 28px; }
  #new .panel { background:var(--paper); border:1px solid var(--line); border-radius:12px; padding:20px 22px; }
  #new h2 { font:600 18px/1.3 var(--serif); margin:0 0 12px; }
  #new form { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  #new input[type=url] { flex:1 1 320px; min-width:0; padding:8px 12px; border:1px solid var(--line); border-radius:8px; }
  #new input[type=number] { width:110px; padding:8px; border:1px solid var(--line); border-radius:8px; }
  #new .or { color:var(--muted); }
  #new .sites { margin:10px 0 0; color:var(--muted); font-size:13px; max-width:90ch; }
  #new label { display:flex; gap:6px; align-items:center; }
  #filename { color:var(--muted); max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #progress { margin-top:12px; font-size:13px; color:var(--muted); white-space:pre-wrap; }
  #progress:empty { display:none; }
  #progress b { color:var(--ink); }
  #progress .err { color:var(--warn); }

  /* ---- first-visit introduction */
  #welcome { max-width:1180px; margin:28px auto 0; padding:0 28px; }
  #welcome .intro { background:var(--paper); border-radius:14px; border:1px solid var(--line);
                    padding:32px 36px; display:grid; grid-template-columns:minmax(0,1.25fr) minmax(0,1fr); gap:32px 48px; }
  #welcome h2 { font:600 clamp(26px, 3.2vw, 34px)/1.15 var(--serif); margin:0 0 12px; letter-spacing:-.01em; max-width:24ch; text-wrap:balance; }
  #welcome p { margin:0 0 14px; max-width:60ch; color:var(--muted); font-size:15px; }
  #welcome ol { list-style:none; counter-reset:step; margin:22px 0 24px; padding:0; display:grid; gap:16px; }
  #welcome ol li { counter-increment:step; display:grid; grid-template-columns:34px 1fr; gap:2px 12px; max-width:62ch; }
  #welcome ol li::before { content:counter(step); grid-row:span 2; width:28px; height:28px; border-radius:50%;
                           background:var(--brass-wash); color:var(--brass); font:600 15px/28px var(--serif); text-align:center; }
  #welcome ol b { font-weight:600; }
  #welcome ol span { color:var(--muted); }
  #welcome aside { border-left:1px solid var(--line); padding-left:36px; }
  #welcome h3 { font:600 17px/1.3 var(--serif); margin:4px 0 12px; }
  #welcome ul { margin:0; padding:0 0 0 18px; display:grid; gap:10px; color:var(--muted); max-width:46ch; }
  #welcome ul b { color:var(--ink); font-weight:600; }
  #welcome .actions { margin-top:4px; }

  /* ---- score */
  main { padding:28px 24px calc(140px + env(safe-area-inset-bottom, 0px)); display:flex; flex-direction:column;
         align-items:safe center; gap:24px; overflow-x:auto; }
  /* --page-w: the fitted width (up to 1040px) times the score zoom; zoomed in, pages scroll sideways */
  .page { background:var(--paper); width:var(--page-w, 100%); max-width:none; flex:none; border-radius:6px; padding:12px 8px;
          box-shadow:0 1px 2px rgba(31,38,51,.08), 0 6px 24px rgba(31,38,51,.06); }
  .page svg { display:block; width:100%; height:auto; }
  #msg { padding:56px 24px; color:var(--muted); font-size:15px; text-align:center; }
  main.loading .page { opacity:.45; transition:opacity .15s; }
  .page g.note, .page g.chord, .page g.measure { cursor:pointer; }
  .page g.playing { fill:var(--brass); color:var(--brass); }
  .page rect.measure-hl { fill:var(--brass-light); fill-opacity:.22; stroke:none; pointer-events:none; display:none; }
  body.playing .page rect.measure-hl { display:inline; }   /* the current bar, only while playing */
  /* No fill until hovered (an invisible fill still shows through the Jazz look's ink filter). */
  .page rect.measure-hit { fill:none; stroke:none; pointer-events:all; cursor:pointer; }
  .page rect.measure-hit:hover { fill:var(--brass-light); fill-opacity:.12; }

  /* ---- playback bar: ink, floating clear of the window edge */
  #transport { position:fixed; z-index:3; left:16px; right:16px; bottom:calc(16px + env(safe-area-inset-bottom, 0px));
               max-width:1180px; margin:0 auto; background:var(--ink); color:#e9ecf1; border-radius:14px;
               padding:12px 18px; display:flex; flex-wrap:wrap; gap:10px 22px; align-items:center;
               box-shadow:0 10px 30px rgba(31,38,51,.28); }
  #transport .label { color:#a3abb9; }
  #transport button { background:transparent; border-color:#3d4658; color:inherit; }
  #transport button:hover { border-color:#6b7487; }
  #transport button.on { background:var(--brass-light); border-color:var(--brass-light); color:var(--ink); font-weight:600; }
  #transport .seg { background:#2b3343; }
  #transport .seg button { border:0; }
  #transport .seg button.on { box-shadow:none; }
  #transport select { background:#2b3343; border-color:#3d4658; color:inherit; }
  #play { min-width:96px; background:var(--brass-light) !important; border-color:var(--brass-light) !important;
          color:var(--ink) !important; font-weight:600; }
  #scrubWrap { flex:1 1 280px; display:flex; align-items:center; gap:12px; min-width:0; }
  #scrub { flex:1; min-width:0; accent-color:var(--brass-light); }
  #clock { font-variant-numeric:tabular-nums; color:#a3abb9; white-space:nowrap; }
  #clock b { color:#fff; font-weight:600; }
  #soundMsg { color:#a3abb9; font-size:12px; }

  /* ---- mode switch and the split view */
  #mode { flex:none; overflow:visible; }   /* never squeezed by a long song title */
  #mode button { font-weight:600; }
  body.split header .bar + .bar, body.split #viewMenu, body.split .download, body.split .title #score,
  body.split #status, body.split #pages, body.split #transport { display:none; }
  body.split #new .song-only, body:not(.split) #new .split-only { display:none; }
  #splitView { max-width:980px; margin:28px auto 140px; padding:0 28px; }
  .split-card { background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:28px 32px; }
  .split-head { display:flex; flex-wrap:wrap; align-items:center; gap:12px 20px; margin-bottom:20px; }
  #splitSong { font:600 21px/1.2 var(--serif); border:0; padding:4px 26px 4px 2px; flex:1 1 280px; min-width:0;
               max-width:100%; appearance:none; cursor:pointer;
               background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%235b6474' fill='none' stroke-width='1.5'/%3E%3C/svg%3E") right 8px center no-repeat; }
  #splitEmpty p { color:var(--muted); max-width:60ch; margin:0 0 14px; }
  .opts { display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; margin:12px 0 18px; }
  .opts label { display:flex; gap:8px; align-items:center; }
  .presets { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:18px; }
  .presets button { border-radius:999px; padding:4px 14px; min-height:30px; }
  .part-row { display:grid; grid-template-columns:150px minmax(120px, 1fr) 52px auto; align-items:center;
          gap:8px 16px; padding:10px 0; border-top:1px solid var(--line); }
  .part-row:last-child { border-bottom:1px solid var(--line); }
  .part-row .name { display:flex; align-items:center; gap:10px; font-weight:600; }
  .part-row .name button { min-width:44px; }
  .part-row.off .name span, .part-row.off .pct { color:var(--muted); }
  .part-row input[type=range] { width:100%; accent-color:var(--brass); }
  .part-row .pct { font-variant-numeric:tabular-nums; color:var(--muted); text-align:right; }
  .mixplay { display:flex; flex-wrap:wrap; align-items:center; gap:12px 16px; margin-top:22px; }
  #mixPlay { min-width:96px; background:var(--brass-light); border-color:var(--brass-light); font-weight:600; }
  #mixScrub { flex:1 1 260px; accent-color:var(--brass); }
  #mixClock { font-variant-numeric:tabular-nums; color:var(--muted); }
  #mixState { color:var(--muted); font-size:13px; min-width:70px; }
  .export { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:18px; padding-top:18px;
            border-top:1px solid var(--line); }
  .export .hint { color:var(--muted); font-size:13px; }

  /* ---- editing: a third toolbar row, selected notes in blue */
  :root { --select:#2458c6; }
  #editBar { display:none; padding-top:12px; border-top:1px dashed var(--line); }
  body.editing #editBar { display:flex; }
  body.split #editBar { display:none; }
  #editBtn.on { background:var(--select); border-color:var(--select); color:#fff; font-weight:600; }
  .page g.sel, .page g.sel * { fill:var(--select); color:var(--select); }
  body.editing .page rect.measure-hit { cursor:crosshair; }
  body.editing .page rect.measure-hit:hover { fill:var(--select); fill-opacity:.05; }
  #editBar .actions { align-items:center; }
  .sr-only { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }
  #editState { color:var(--muted); font-size:13px; }
  #editState.dirty { color:var(--brass); font-weight:600; }
  #revertBtn.confirm { color:var(--warn); border-color:var(--warn); }
  .lyric-input { position:absolute; z-index:5; font:17px/1.2 var(--serif); width:10ch; padding:3px 7px;
                 border:2px solid var(--select); border-radius:6px; background:var(--paper); color:var(--ink); }

  @media (max-width: 760px) {
    #splitView { padding:0 12px; }
    .split-card { padding:20px 16px; }
    .part-row { grid-template-columns:1fr 52px; }
    .part-row .name { grid-column:1 / -1; }
    header { position:static; }
    header, #status, #new, #welcome { padding-left:16px; padding-right:16px; }
    .group { max-width:100%; }
    #transport .label { display:none; }
    #transport { gap:8px 10px; }
    #scrubWrap { flex-basis:calc(100% - 110px); }
    #welcome .intro { grid-template-columns:1fr; padding:24px 20px; }
    #welcome aside { border-left:0; border-top:1px solid var(--line); padding:20px 0 0; }
    main { padding-left:8px; padding-right:8px; padding-bottom:170px; }
    #transport { left:8px; right:8px; bottom:8px; padding:10px 12px; }
  }
  /* ⌘P / File > Print: just the music, each score page on its own sheet, whatever the zoom */
  @page { margin:12mm; }
  @media print {
    header, #status, #transport, #welcome, #new, #splitView, #sr, .lyric-input { display:none !important; }
    body, main { background:#fff !important; padding:0 !important; margin:0 !important; overflow:visible !important; display:block !important; }
    .page { width:100% !important; box-shadow:none !important; border-radius:0; padding:0; margin:0;
            break-inside:avoid; break-after:page; }
    .page:last-child { break-after:auto; }
    .page svg { width:100% !important; height:auto; max-height:calc(100vh - 2px); }
    .page rect.measure-hl, .page rect.measure-hit, .page g.sel { display:none; }
  }
  @media (prefers-reduced-motion: reduce) { * { transition:none !important; scroll-behavior:auto !important; } }
</style></head>
<body>
<header>
  <div class="bar">
    <div class="title">
      <div class="seg" id="mode" role="tablist" aria-label="What to do">
        <button data-mode="sheet" role="tab">Sheet music</button>
        <button data-mode="split" role="tab">Split</button>
      </div>
      <select id="score" aria-label="Song"></select>
    </div>
    <div class="actions">
      <button id="newBtn">New song</button>
      <button id="editBtn" hidden title="Change, add and delete notes, words and bars">Edit</button>
      <div class="download">
        <select id="dlFmt" aria-label="Download format" title="PDF to print; MusicXML opens in MuseScore, Dorico, Finale and Sibelius; MIDI for a DAW">
          <option value="pdf">PDF</option>
          <option value="musicxml">MusicXML</option>
          <option value="mid">MIDI</option>
        </select>
        <button class="primary" id="pdf" title="Download in the chosen format (⌘P / Ctrl+P prints the music, with a preview)">Download</button>
      </div>
      <button id="helpBtn" class="quiet round" title="How it works" aria-label="How it works">?</button>
    </div>
  </div>
  <div class="bar" id="practiceBar">
    <div class="group" role="group" aria-label="Detail">
      <span class="label">Detail</span>
      <div class="seg">
        <button data-detail="simple" title="Just the tune: one in-key note per word or syllable, 8th-note rhythm">Simple</button>
        <button data-detail="balanced" title="Every note it's sure of, as sung: runs, 16ths, triplets and grace notes">Balanced</button>
        <button data-detail="detailed" title="Everything it heard, with notes it's unsure of in grey">Detailed</button>
      </div>
    </div>
    <div class="group">
      <label class="label" for="instSel">Instrument</label>
      <select id="instSel" title="Shows the music in your instrument's key">
        <option value="0">Concert C</option>
        <option value="9">Alto sax E♭</option>
        <option value="14">Tenor sax B♭</option>
        <option value="2">Trumpet, clarinet B♭</option>
        <option value="custom" hidden></option>
      </select>
    </div>
    <div class="group" role="group" aria-label="Transpose">
      <span class="label">Transpose</span>
      <div class="seg">
        <button data-step="-12" title="Down an octave (↓)">−8va</button>
        <button data-step="-1" title="Down a half step (←)">−½</button>
        <button data-step="1" title="Up a half step (→)">+½</button>
        <button data-step="12" title="Up an octave (↑)">+8va</button>
      </div>
    </div>
    <div class="group" role="group" aria-label="Zoom">
      <span class="label">Zoom</span>
      <div class="seg">
        <button data-zoom="-1" title="Zoom out (⌘− / Ctrl −)" aria-label="Zoom out">−</button>
        <button id="zoomPct" title="Back to 100% (⌘0 / Ctrl 0)">100%</button>
        <button data-zoom="1" title="Zoom in (⌘+ / Ctrl +, or pinch)" aria-label="Zoom in">+</button>
      </div>
    </div>
    <details class="menu" id="viewMenu">
      <summary>View</summary>
      <div class="menu-panel">
        <div class="row">
          <span class="label">Lyrics</span>
          <button id="lyricsBtn" title="Show the words under the notes">Off</button>
        </div>
        <div class="row" id="harmonyGroup" hidden>
          <span class="label">Harmony</span>
          <button id="harmonyBtn">Add</button>
        </div>
        <div class="row" role="group" aria-label="Clef">
          <span class="label">Clef</span>
          <div class="seg">
            <button data-clef="treble">Treble</button>
            <button data-clef="bass">Bass</button>
          </div>
        </div>
        <div class="row" role="group" aria-label="Look">
          <span class="label">Look</span>
          <div class="seg">
            <button data-style="classic" title="Engraved, like printed sheet music">Classic</button>
            <button data-style="jazz" title="Hand-copied, like a jazz fake book">Jazz</button>
          </div>
        </div>
        <div class="row" role="group" aria-label="Bars per line">
          <span class="label">Bars per line</span>
          <div class="seg">
            <button id="barsAuto" title="Fit as many bars on each line as the music needs">Auto</button>
            <button data-bars="-1" title="Fewer bars per line" aria-label="Fewer bars per line">−</button>
            <button id="barsN" title="The same number of bars on every line, all the same width">4</button>
            <button data-bars="1" title="More bars per line" aria-label="More bars per line">+</button>
          </div>
        </div>
      </div>
    </details>
  </div>
  <div class="bar" id="editBar" aria-label="Edit the notes">
    <div class="group" id="voiceGroup" role="group" aria-label="Write into">
      <span class="label">Write</span>
      <div class="seg">
        <button data-voice="n0" title="New notes go into the melody">Melody</button>
        <button data-voice="h" title="New notes go into the harmony (stems down)">Harmony</button>
      </div>
    </div>
    <div class="group" role="group" aria-label="Note length">
      <span class="label">Length</span>
      <div class="seg">
        <button data-base="24" title="16th note (1)">1/16</button>
        <button data-base="48" title="8th note (2)">1/8</button>
        <button data-base="96" title="Quarter note (3)">1/4</button>
        <button data-base="192" title="Half note (4)">1/2</button>
        <button data-base="384" title="Whole note (5)">1</button>
      </div>
      <div class="seg">
        <button id="dotBtn" title="Dotted: half as long again (.)">Dot</button>
        <button id="tripBtn" title="Triplet: three in the time of two (T)">Triplet</button>
      </div>
    </div>
    <div class="group" role="group" aria-label="Change the selected notes">
      <button data-act="add" title="Add a note right after the selected one, at the chosen length (+)">Add after</button>
      <button data-act="delete" title="Delete the selected notes (Delete)">Delete</button>
      <button data-act="lyric" title="Type the word under the selected note (L or Enter; Tab for the next note)">Word</button>
      <button data-act="slur" title="Slur from the note before, or remove the slur (S)">Slur</button>
      <button data-act="grace" title="Add a grace note a step below, or remove it (/ ; ? for a step above)">Grace</button>
    </div>
    <div class="group" role="group" aria-label="Bars">
      <span class="label">Bar</span>
      <button data-act="insertBar" title="Insert an empty bar before the selected note's bar (at the end if nothing is selected)">Insert</button>
      <button data-act="deleteBar" title="Delete the selected note's bar and everything in it (the last bar if nothing is selected)">Delete</button>
    </div>
    <div class="group">
      <span class="label">Key</span>
      <select id="keySel" aria-label="Key, as written for your instrument" title="The key as written for your instrument (what's on the page)"></select>
      <span class="label">Time</span>
      <select id="timeSel" aria-label="Time signature">
        <option>2/4</option><option>3/4</option><option>4/4</option><option>5/4</option>
        <option>6/8</option><option>9/8</option><option>12/8</option><option>2/2</option>
      </select>
    </div>
    <div class="actions">
      <select id="histSel" aria-label="Version history" title="Open an earlier save (or the original) to look at; Save to keep it"></select>
      <button id="undoBtn" title="Undo (⌘Z / Ctrl+Z)">Undo</button>
      <button id="redoBtn" title="Redo (⇧⌘Z / Ctrl+Y)">Redo</button>
      <span id="editState" aria-live="polite"></span>
      <button id="revertBtn" class="quiet" title="Go back to the transcription for this detail level (your saves stay in History)">Revert to original</button>
      <button id="discardBtn" title="Throw away changes since the last save">Discard changes</button>
      <button class="primary" id="saveBtn" title="Save (⌘S / Ctrl+S)">Save</button>
    </div>
  </div>
</header>
<section id="welcome" hidden>
  <div class="intro">
    <div>
      <h2>Turn a recording into sheet music you can play</h2>
      <p>Give it a song and it pulls out one part, usually the singer, writes it down as notes, and shows
        it in your instrument's key so you can read along.</p>
      <ol>
        <li><b>Add a song</b><span>Paste a link (YouTube, SoundCloud, Spotify and more) or choose any audio or video file, then pick the part. It takes about a minute.</span></li>
        <li><b>Read it in your key</b><span>Alto sax is the default. Switch instrument, transpose, or turn the lyrics on.</span></li>
        <li><b>Play along</b><span>Press <kbd>Space</kbd> to play, click any bar to jump there, slow it down, and hear the original underneath.</span></li>
      </ol>
      <div class="actions">
        <button class="primary" id="welcomeNew">Add a song</button>
        <button id="welcomeClose">Got it</button>
      </div>
    </div>
    <aside>
      <h3>What to expect</h3>
      <ul>
        <li><b>A strong first draft.</b> Most notes come out right; some won't. Where the page and the recording disagree, trust your ear.</li>
        <li><b>Balanced</b> shows every note it's sure of, as sung. <b>Simple</b> boils it down to the tune; <b>Detailed</b> adds everything else it heard, with unsure notes in grey.</li>
        <li><b>Rhythm is rounded</b> to the nearest 8th or 16th note, so swung or pushed phrases look straighter than they sound.</li>
        <li><b>Split</b> pulls a song apart into vocals, drums, bass and more. Mute or solo any part,
          then download the mix.</li>
        <li><b>Everything stays on this computer.</b> Songs you add are saved in the <code>output</code> folder.</li>
      </ul>
    </aside>
  </div>
</section>
<section id="new" hidden>
  <div class="panel">
    <h2>Add a song</h2>
    <form id="newForm">
      <input type="url" id="src" placeholder="Paste a link to a song" aria-label="Link to a song">
      <span class="or">or</span>
      <button type="button" id="pick">Choose audio file</button>
      <input type="file" id="file" hidden
        accept="audio/*,video/*,.mp3,.m4a,.aac,.wav,.flac,.ogg,.oga,.opus,.aif,.aiff,.aifc,.wma,.alac,.caf,.ac3,.eac3,.dts,.mp2,.mpga,.amr,.ape,.wv,.tta,.au,.snd,.w64,.mka,.mkv,.mp4,.m4v,.mov,.webm,.avi,.wmv,.flv,.3gp,.ts,.mts,.m2ts,.mpg,.mpeg,.vob">
      <span id="filename"></span>
      <select id="part" aria-label="Part" class="song-only">
        <option value="vocals">Vocals</option>
        <option value="bass">Bass</option>
        <option value="guitar">Guitar</option>
        <option value="piano">Piano</option>
        <option value="other">Other (keys, synths, horns)</option>
      </select>
      <input type="number" id="bpm" class="song-only" min="30" max="300" placeholder="Tempo (auto)" aria-label="Tempo in BPM (optional)">
      <select id="newModel" class="split-only" aria-label="How many parts">
        <option value="htdemucs">4 parts: vocals, drums, bass, other</option>
        <option value="htdemucs_6s">6 parts: adds guitar and piano</option>
      </select>
      <label class="split-only"><input type="checkbox" id="newLayers"> Split vocals into lead and backing</label>
      <button type="submit" class="primary" id="go">Transcribe</button>
    </form>
    <p class="sites" id="sites"></p>
    <div id="progress"></div>
  </div>
</section>
<section id="splitView" hidden>
  <div class="split-card">
    <div class="split-head">
      <select id="splitSong" aria-label="Song"></select>
      <div class="seg" id="modelSeg" role="group" aria-label="How many parts">
        <button data-model="htdemucs" title="Best quality">4 parts</button>
        <button data-model="htdemucs_6s" title="Adds guitar and piano; those two are less clean">6 parts</button>
      </div>
    </div>
    <div id="splitEmpty" hidden>
      <p id="splitEmptyMsg"></p>
      <div class="opts">
        <label><input type="checkbox" id="splitLayers"> Also split the vocals into lead and backing</label>
      </div>
      <button class="primary" id="splitGo">Split this song</button>
    </div>
    <div id="mixer" hidden>
      <div class="presets" id="presets"></div>
      <div id="stems"></div>
      <div class="mixplay">
        <button id="mixPlay">▶ Play</button>
        <input type="range" id="mixScrub" min="0" max="1" step="0.1" value="0" aria-label="Position">
        <span id="mixClock">0:00 / 0:00</span>
        <span id="mixState" aria-live="polite"></span>
      </div>
      <div class="export">
        <select id="mixFmt" aria-label="File format">
          <option value="mp3">MP3</option>
          <option value="m4a">M4A (AAC)</option>
          <option value="flac">FLAC (lossless)</option>
          <option value="wav">WAV (lossless)</option>
        </select>
        <button class="primary" id="mixDownload">Download this mix</button>
        <span class="hint">Exactly what you hear: the parts that are on, at their volumes.</span>
      </div>
    </div>
    <div id="noSongs" hidden><p>No songs yet. Click <b>New song</b> to add one.</p></div>
  </div>
</section>
<div id="status"></div>
<div id="sr" class="sr-only" aria-live="polite"></div>
<main id="pages"><div id="msg">Loading…</div></main>
<footer id="transport">
  <button id="play" title="Play / pause (Space)">▶ Play</button>
  <div id="scrubWrap">
    <input type="range" id="scrub" min="0" max="1" step="0.01" value="0" aria-label="Playback position">
    <span id="clock"><b>0:00</b> / 0:00, bar 1</span>
  </div>
  <div class="group" role="group" aria-label="Speed">
    <span class="label">Speed</span>
    <div class="seg">
      <button data-speed="0.5">50%</button>
      <button data-speed="0.75">75%</button>
      <button data-speed="0.9">90%</button>
      <button data-speed="1">100%</button>
    </div>
  </div>
  <div class="group">
    <span class="label">Hear</span>
    <button id="notesBtn" title="Play the transcribed notes">Notes</button>
    <select id="orig" aria-label="Original audio">
      <option value="off">No original</option>
      <option value="stem">Original part</option>
      <option value="lead">Lead vocal</option>
      <option value="harmony">Backing vocals</option>
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
const PRESETS = {0:"Concert C", 9:"Alto sax E♭", 14:"Tenor sax B♭", 2:"Trumpet / clarinet B♭"};
const $ = s => document.querySelector(s);
const store = { get(k){ try { return localStorage.getItem(k); } catch { return null; } },
                set(k,v){ try { localStorage.setItem(k,v); } catch {} },
                del(k){ try { localStorage.removeItem(k); } catch {} } };
let harmony = store.get("harmony") !== "0";
let barsEven = store.get("barsEven") === "1", barsN = +(store.get("barsN") || 4);   // bars per line
const barsParam = () => barsEven ? barsN : 0;   // shown whenever a song has it, unless turned off
let look = store.get("style") === "jazz" ? "jazz" : "classic";   // how the music looks
const enh = () => store.get("enh:" + t) === "1";   // spell this transposition's key the other way
let file = DEFAULT, t = 9, inst = 9, detail = "balanced", lyrics = store.get("lyrics") === "1", req = 0, clef = store.get("clef") === "bass" ? "bass" : "treble";   // default: alto sax

function label() {
  if (PRESETS[t]) return PRESETS[t];
  for (const base of [9, 14, 2, 0]) {          // e.g. "Alto Sax E♭ +8va"
    const d = t - base;
    if (d && d % 12 === 0) {
      const n = Math.abs(d / 12), sign = d > 0 ? "+" : "−";
      return `${PRESETS[base]} ${n === 1 ? sign + "8va" : n === 2 ? sign + "15ma" : `${sign}${n} octaves`}`;
    }
  }
  return `${t > 0 ? "+" : ""}${t} semitones`;
}

async function render() {
  const id = ++req;
  store.set("t:" + file, t);
  store.set("clef", clef);
  store.set("level:" + file, detail);
  store.set("lyrics", lyrics ? "1" : "0");
  $("#lyricsBtn").classList.toggle("on", lyrics);
  $("#lyricsBtn").textContent = lyrics ? "On" : "Off";
  document.querySelectorAll("[data-clef]").forEach(b => b.classList.toggle("on", b.dataset.clef === clef));
  store.set("barsEven", barsEven ? "1" : "0"); store.set("barsN", barsN);
  $("#barsAuto").classList.toggle("on", !barsEven); $("#barsN").classList.toggle("on", barsEven);
  $("#barsN").textContent = barsN;
  document.querySelectorAll("[data-detail]").forEach(b => b.classList.toggle("on", b.dataset.detail === detail));
  document.querySelectorAll("[data-style]").forEach(b => b.classList.toggle("on", b.dataset.style === look));
  store.set("style", look);
  // The instrument menu shows the preset, or e.g. "Alto sax E♭ +8va" / "+3 semitones" after transposing.
  const custom = $("#instSel option[value=custom]");
  custom.hidden = t in PRESETS;
  custom.textContent = label();
  $("#instSel").value = t in PRESETS ? String(t) : "custom";
  $("#pages").classList.add("loading");
  // While editing, draw the editor's note list (switching song or level loads that one's).
  if (editor.on && editor.slot !== file + "|" + detail && !(await editor.load())) return;
  if (id !== req) return;
  const r = editor.on
    ? await fetch("/api/render", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({f: file, t, clef, detail, lyrics, harmony, bars: barsParam(), style: look, enh: enh(), events: editor.wire()})})
    : await fetch(`/api/render?f=${encodeURIComponent(file)}&t=${t}&clef=${clef}&detail=${detail}&lyrics=${lyrics ? 1 : 0}&harmony=${harmony ? 1 : 0}&bars=${barsParam()}&style=${look}&enh=${enh() ? 1 : 0}`);
  if (id !== req) return;                       // a newer click superseded this one
  $("#pages").classList.remove("loading");
  if (!r.ok) { $("#pages").innerHTML = `<div id="msg">${await r.text()}</div>`; return; }
  const d = await r.json();
  $("#pages").innerHTML = d.svgs.map(s => `<div class="page">${s}</div>`).join("");
  applyZoom(zoom, undefined, undefined, false);
  player.load(d);
  $("#lyricsBtn").disabled = !d.hasLyrics;
  $("#harmonyGroup").hidden = !d.canHarmony;
  hasHarmony = d.hasHarmony;
  $("#harmonyBtn").classList.toggle("on", hasHarmony && harmony);
  $("#harmonyBtn").textContent = hasHarmony ? (harmony ? "On" : "Off") : "Add";
  $("#harmonyBtn").title = hasHarmony ? "Show the backing vocals' harmony as a second voice"
    : "Separate the backing vocals and add their harmony as a second voice (about a minute)";
  $("#lyricsBtn").title = d.hasLyrics ? "Show the words under the notes" : "No lyrics for this song (re-run it to add them)";
  $("#editBtn").hidden = !d.editable;
  if (editor.on) editor.drawn(d);
  const warn = [];
  if (d.belowAlto) warn.push(`${d.belowAlto} notes below alto sax range`);
  if (d.aboveAlto) warn.push(`${d.aboveAlto} notes above alto sax range`);
  $("#status").innerHTML =
    `<span>Showing <b>${label()}</b></span>` +
    `<span>Transposed <b>${d.interval}</b> (${t > 0 ? "+" : ""}${t})</span>` +
    `<span>Key signature <b>${d.keySig}</b>` +
      (d.altKeySig ? ` <button class="quiet inline" id="enhBtn" title="Same notes, spelled the other way; remembered for this instrument">Write in ${d.altKeySig}</button>` : "") + `</span>` +
    `<span>Range <b>${d.low} – ${d.high}</b></span>` +
    (warn.length ? `<span class="warn">${warn.join(", ")} (try ±8va)</span>` : "") +
    (editor.on
      ? `<span>Click a note to select it (<kbd>⇧</kbd>-click for more), or empty staff to add one.
         <kbd>↑</kbd><kbd>↓</kbd> pitch (<kbd>⇧</kbd> octave), <kbd>A</kbd>–<kbd>G</kbd> note name,
         <kbd>+</kbd> add after, <kbd>1</kbd>–<kbd>5</kbd> length, <kbd>.</kbd> dot, <kbd>T</kbd> triplet, <kbd>⌫</kbd> delete,
         <kbd>L</kbd> word, <kbd>S</kbd> slur, <kbd>/</kbd> grace, <kbd>←</kbd><kbd>→</kbd> next note
         (<kbd>⌥</kbd> move it), <kbd>⌘Z</kbd> undo</span>`
      : (detail === "detailed" ? `<span>Grey notes: ones the transcription is unsure of</span>` : "") +
        (d.edited ? `<span><b>Edited</b> (the original is kept: Edit, then Revert to original)</span>` : "") +
        (editor.hasDraft() ? `<span class="warn">Unsaved edits: click Edit to finish them</span>` : "") +
        `<span><kbd>←</kbd><kbd>→</kbd> half step <kbd>↑</kbd><kbd>↓</kbd> octave</span>`);
  if ($("#enhBtn")) $("#enhBtn").onclick = () => { store.set("enh:" + t, enh() ? "0" : "1"); render(); };
}

document.querySelectorAll("[data-clef]").forEach(b => b.onclick = () => { clef = b.dataset.clef; render(); });
document.querySelectorAll("[data-detail]").forEach(b => b.onclick = () => { detail = b.dataset.detail; render(); });
document.querySelectorAll("[data-style]").forEach(b => b.onclick = () => { look = b.dataset.style; render(); });
$("#lyricsBtn").onclick = () => { lyrics = !lyrics; render(); };
$("#barsAuto").onclick = () => { barsEven = false; render(); };
$("#barsN").onclick = () => { barsEven = true; render(); };
document.querySelectorAll("[data-bars]").forEach(b => b.onclick = () => {   // stepping turns even bars on
  if (barsEven) barsN = Math.max(1, Math.min(12, barsN + +b.dataset.bars));
  barsEven = true; render();
});
let hasHarmony = false;
$("#harmonyBtn").onclick = async () => {
  if (hasHarmony) { harmony = !harmony; store.set("harmony", harmony ? "1" : "0"); return render(); }
  const r = await fetch("/api/job", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({harmony: file})});
  if (!r.ok) { $("#new").hidden = false; return showProgress(await r.text(), true); }
  harmony = true; store.set("harmony", "1");
  startedAt = Date.now();
  poll();
};
function setT(v) { t = Math.max(-36, Math.min(36, v)); render(); }

$("#instSel").onchange = e => {
  if (e.target.value === "custom") return;
  inst = +e.target.value;            // what the player's instrument is (playback sounds at concert pitch)
  store.set("inst:" + file, inst);
  setT(inst);
};
// The View menu closes on a click elsewhere or Escape, like any dropdown.
document.addEventListener("click", e => { if (!e.target.closest("#viewMenu")) $("#viewMenu").open = false; });
document.addEventListener("keydown", e => { if (e.key === "Escape" && $("#viewMenu").open) { $("#viewMenu").open = false; $("#viewMenu summary").focus(); } });
document.querySelectorAll("[data-step]").forEach(b => b.onclick = () => setT(t + +b.dataset.step));
$("#dlFmt").value = store.get("dlFmt") || "pdf";
$("#dlFmt").onchange = e => store.set("dlFmt", e.target.value);
function download(fmt) {
  location.href = `/api/export?fmt=${fmt}&f=${encodeURIComponent(file)}&t=${t}&clef=${clef}&detail=${detail}&lyrics=${lyrics ? 1 : 0}&harmony=${harmony ? 1 : 0}&bars=${barsParam()}&style=${look}&enh=${enh() ? 1 : 0}&label=${encodeURIComponent(label())}`;
}
$("#pdf").onclick = () => download($("#dlFmt").value);
$("#score").onchange = e => loadList(e.target.value);
document.addEventListener("keydown", e => {
  const inField = ["SELECT", "INPUT", "TEXTAREA", "BUTTON"].includes(e.target.tagName);
  if (mode === "split") {
    if (e.key === " " && (!inField || ["mixScrub"].includes(e.target.id) || e.target.tagName === "BUTTON")) {
      e.preventDefault(); mixToggle();
    }
    return;
  }
  if (e.key === " " && (!inField || e.target.id === "scrub" || e.target.tagName === "BUTTON")) {
    e.preventDefault(); player.toggle(); return;
  }
  if (inField && e.target.tagName !== "BUTTON") return;   // let fields (and the scrub bar) use arrow keys
  if (editor.on && !(e.target.tagName === "BUTTON" && e.key === "Enter") && editor.key(e)) return;
  const step = {ArrowLeft:-1, ArrowRight:1, ArrowDown:-12, ArrowUp:12}[e.key];
  if (step) { e.preventDefault(); setT(t + step); }
});

// ---- zoom: the score's own zoom (⌘+/⌘−/⌘0, pinch, ctrl+scroll, the Zoom buttons). Browser
// zoom can't enlarge the score - it's sized to fit the window - so these zoom just the pages.
const ZOOMS = [0.5, 0.67, 0.75, 0.85, 1, 1.15, 1.3, 1.5, 1.75, 2, 2.5, 3];
let zoom = Math.min(3, Math.max(0.5, +(store.get("zoom") || 1)));
function applyZoom(next, anchorX, anchorY, keepPlace = true) {
  const main = $("#pages");
  const fit = Math.min(1040, main.clientWidth - 48);   // the page's width at 100%
  // Keep the point under the anchor (default: the middle of the window) where it is.
  const ax = anchorX ?? innerWidth / 2, ay = anchorY ?? innerHeight / 2;
  const box = main.getBoundingClientRect(), top = box.top + scrollY;
  const fx = (main.scrollLeft + ax - box.left) / Math.max(main.scrollWidth, 1);
  const fy = (scrollY + ay - top) / Math.max(main.scrollHeight, 1);
  zoom = Math.min(3, Math.max(0.5, next));
  store.set("zoom", zoom);
  main.style.setProperty("--page-w", `${Math.round(fit * zoom)}px`);
  $("#zoomPct").textContent = `${Math.round(zoom * 100)}%`;
  if (!keepPlace) return;
  main.scrollLeft = fx * main.scrollWidth - (ax - box.left);
  scrollTo({top: top + fy * main.scrollHeight - ay});
}
const zoomStep = dir => ZOOMS.find(z => dir > 0 ? z > zoom + 1e-6 : false) ?? (dir < 0 ? [...ZOOMS].reverse().find(z => z < zoom - 1e-6) : null);
function stepZoom(dir) { const z = zoomStep(dir); if (z) applyZoom(z); }
document.querySelectorAll("[data-zoom]").forEach(b => b.onclick = () => stepZoom(+b.dataset.zoom));
$("#zoomPct").onclick = () => applyZoom(1);
addEventListener("resize", () => applyZoom(zoom));
// pinch on a trackpad (and ctrl+scroll) arrive as ctrl+wheel events
addEventListener("wheel", e => {
  if (!e.ctrlKey || mode === "split") return;
  e.preventDefault();
  applyZoom(zoom * Math.exp(-e.deltaY * 0.002), e.clientX, e.clientY);   // ~20% per wheel notch
}, {passive: false});
addEventListener("keydown", e => {
  if (!(e.metaKey || e.ctrlKey) || e.altKey || mode === "split") return;
  if (e.key === "=" || e.key === "+") { e.preventDefault(); stepZoom(1); }
  else if (e.key === "-" || e.key === "_") { e.preventDefault(); stepZoom(-1); }
  else if (e.key === "0") { e.preventDefault(); applyZoom(1); }
}, true);

// ---- playback
// Notes come from Verovio's timemap (positions in quarter notes, written pitch). They sound at
// concert pitch: written pitch minus the instrument's transposition (alto sax sounds a M6 lower).
const NAMES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"];
const midiName = m => NAMES[m % 12] + (Math.floor(m / 12) - 1);

// A box around one measure, sized from its five staff lines only - not the notes, which
// would make measures with high/low notes taller than the rest. Hand-drawn (Jazz) staff
// lines wander, so use where they were ruled; every box on the page is then the same height.
function measureRect(measureEl, cls) {
  const staff = measureEl.querySelector("g.staff");
  const lines = staff ? [...staff.children].filter(c => c.tagName === "path") : [];
  const boxes = (lines.length ? lines : [staff || measureEl]).map(l => {
    const ruled = l.getAttribute && l.getAttribute("data-line");
    if (!ruled) return l.getBBox();
    const [x1, y, x2] = ruled.split(" ").map(Number);
    return {x: x1, y, width: x2 - x1, height: 0};
  });
  const x = Math.min(...boxes.map(b => b.x)), right = Math.max(...boxes.map(b => b.x + b.width));
  const y = Math.min(...boxes.map(b => b.y)), bottom = Math.max(...boxes.map(b => b.y + b.height));
  const pad = (bottom - y) * 0.5;
  const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  r.setAttribute("class", cls);
  r.style.stroke = "none";   // Verovio's stylesheet outlines every rect (an id rule beats ours)
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
    // Grace notes (ids ending "-g") are a few ms long in the score - shorter than a sample's
    // attack, so they'd be inaudible. Play each for ~90 ms ending on its note's beat, starting
    // past the attack, with the note slurred out of it (a scoop).
    const GRACE_S = 0.09;
    const isGrace = n => /-g$/.test(n.id);
    if (this.notesOn) this.notes.forEach((n, i) => {
      const endQ = n.q + n.dur;
      if (endQ <= this.startQ) return;
      let t0 = this.startTime + Math.max(0, n.q - this.startQ) * spq;
      const t1 = this.startTime + (endQ - this.startQ) * spq;
      const m = this.sounding(n), prev = this.notes[i - 1];
      const afterGrace = prev && isGrace(prev) && Math.abs(prev.q + prev.dur - n.q) < 0.01;
      const slurred = (n.legato && prev && Math.abs(prev.q + prev.dur - n.q) < 1e-6 || afterGrace) && n.q > this.startQ;
      const grace = isGrace(n);
      if (grace) t0 = Math.max(this.startTime, t1 - GRACE_S);
      const buf = sampled && this.buffers[this.sound + ":" + m];
      this.voices.push(buf ? this.sample(buf, t0, t1, slurred || grace) : this.synth(m, t0, t1, slurred || grace));
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
    $("#clock").innerHTML = `<b>${fmt(q)}</b> / ${fmt(this.totalQ)}, bar ${mi + 1}`;
  },

  setButton(text) {
    $("#play").textContent = text || (this.playing ? "❚❚ Pause" : "▶ Play");
    document.body.classList.toggle("playing", this.playing);
  },
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
  if (editor.on) return editor.click(e);
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

// ---- editing
// The editor changes the score's note list (see song.py events_score): concert pitch, one list
// per staff plus the harmony voice, positions and lengths in ticks (TPQ per quarter note) while
// editing and as fractions of a quarter note on the wire. Every drawn note's id names its list
// and index ("n0-12", "h-3"; "-2", "-b1", "-g" on tied pieces and grace notes), so a click on
// the page maps straight to the note it shows. Unsaved changes are kept in localStorage.
const TPQ = 96;
const gcd = (a, b) => b ? gcd(b, a % b) : Math.abs(a);
const toTicks = s => { const [a, b = "1"] = String(s).split("/"); return Math.round(+a * TPQ / +b); };
const fromTicks = n => { const g = gcd(n, TPQ) || 1; return TPQ / g === 1 ? String(n / g) : `${n / g}/${TPQ / g}`; };
const LETTER_STEPS = [0, 2, 4, 5, 7, 9, 11];               // C D E F G A B
const SHARP_ORDER = [3, 0, 4, 1, 5, 2, 6], FLAT_ORDER = [6, 2, 5, 1, 4, 0, 3];
const MAJOR = {"-7":"C♭","-6":"G♭","-5":"D♭","-4":"A♭","-3":"E♭","-2":"B♭","-1":"F","0":"C","1":"G","2":"D","3":"A","4":"E","5":"B","6":"F♯","7":"C♯"};
const MINOR = {"-7":"A♭","-6":"E♭","-5":"B♭","-4":"F","-3":"C","-2":"G","-1":"D","0":"A","1":"E","2":"B","3":"F♯","4":"C♯","5":"G♯","6":"D♯","7":"A♯"};
const m21Name = s => s.replace("♭", "-").replace("♯", "#");
const ID_RE = /^(n\d+|h)-(\d+)/;
const VOICES = ev => [...ev.staves.map((_, i) => "n" + i), ...(ev.harmony ? ["h"] : [])];

const editor = {
  on: false, ev: null, savedJSON: "", edited: false, slot: "", undoStack: [], redoStack: [],
  sel: [],                  // [{v, e}]: v = "n0"/"n1" (staff) or "h" (harmony), e = the note object
  voice: "n0", base: 48, dot: false, trip: false, keySharps: 0, revertArmed: 0, afterRender: null,

  list(v) { return v === "h" ? (this.ev.harmony = this.ev.harmony || []) : this.ev.staves[+v.slice(1)]; },
  inputDur() { return Math.round(this.base * (this.dot ? 1.5 : 1) * (this.trip ? 2 / 3 : 1)); },
  barTicks() { const [n, d] = this.ev.time_sig.split("/").map(Number); return n * 4 * TPQ / d; },
  draftKey() { return "draft:" + this.slot; },
  dirty() { return !!this.ev && JSON.stringify(this.wire()) !== this.savedJSON; },

  fromWire(ev) {
    const conv = list => list.map(e => ({...e, q: toTicks(e.q), d: toTicks(e.d), p: [...e.p]}));
    return {...ev, length: toTicks(ev.length), staves: ev.staves.map(conv), ...(ev.harmony ? {harmony: conv(ev.harmony)} : {})};
  },
  wire() {
    const conv = list => [...list].sort((a, b) => a.q - b.q).map(e => ({...e, q: fromTicks(e.q), d: fromTicks(e.d)}));
    const ev = this.ev;
    return {...ev, length: fromTicks(ev.length), staves: ev.staves.map(conv), ...(ev.harmony ? {harmony: conv(ev.harmony)} : {})};
  },

  // Load the note list for the current song and detail level (an unsaved draft wins).
  async load() {
    this.slot = file + "|" + detail;
    const r = await fetch(`/api/events?f=${encodeURIComponent(file)}&detail=${detail}`);
    if (!r.ok) { this.setOn(false); $("#status").textContent = await r.text(); return false; }
    const d = await r.json();
    this.edited = d.edited;
    this.ev = this.fromWire(d.events);
    this.savedJSON = JSON.stringify(this.wire());
    const draft = store.get(this.draftKey());
    if (draft) { try { this.ev = this.fromWire(JSON.parse(draft)); } catch { store.del(this.draftKey()); } }
    this.undoStack = []; this.redoStack = []; this.sel = [];
    this.loadHistory();
    return true;
  },
  async loadHistory() {
    const r = await fetch(`/api/history?f=${encodeURIComponent(file)}&detail=${detail}`);
    const saves = r.ok ? await r.json() : [];
    const when = ms => new Date(ms).toLocaleString([], {day: "numeric", month: "short", hour: "2-digit", minute: "2-digit"});
    $("#histSel").innerHTML = `<option value="">History (${saves.length} save${saves.length === 1 ? "" : "s"})</option>` +
      saves.map(s => `<option value="${s.id}">Saved ${when(s.time)}, ${s.notes} notes</option>`).join("") +
      `<option value="original">The original transcription</option>`;
  },
  // Open an earlier version as unsaved changes (undoable; Save keeps it).
  async openVersion(id) {
    $("#histSel").value = "";
    if (!id) return;
    const r = await fetch(`/api/history?f=${encodeURIComponent(file)}&detail=${detail}&version=${id}`);
    if (!r.ok) return this.updateBar("Couldn't open that version");
    const ev = this.fromWire(await r.json());
    this.change(() => { this.ev = ev; this.sel = []; });
    this.updateBar(id === "original" ? "Opened the original: Save to keep it" : "Opened that save: Save to keep it");
  },
  hasDraft() { return !!store.get("draft:" + file + "|" + detail); },

  setOn(on) {
    this.on = on;
    document.body.classList.toggle("editing", on);
    $("#editBtn").classList.toggle("on", on);
    $("#editBtn").textContent = on ? "Done" : "Edit";
    $("#editBtn").title = on ? "Stop editing (unsaved changes are kept for next time)" : "Change, add and delete notes, words and bars";
    if (!on) { this.sel = []; this.ev = null; this.slot = ""; }
  },
  async toggle() {
    if (this.on) { this.setOn(false); return render(); }
    player.stop();
    this.setOn(true);
    if (await this.load()) render();
  },

  // ---- changing the list: every change is undoable and redraws the page
  change(fn) {
    this.undoStack.push(this.snapshot());
    if (this.undoStack.length > 300) this.undoStack.shift();
    this.redoStack = [];
    fn();
    this.tidy();
    this.commit();
  },
  snapshot() {
    return {ev: JSON.stringify(this.wire()), sel: this.sel.map(s => ({v: s.v, i: this.list(s.v).indexOf(s.e)}))};
  },
  restore(snap) {
    this.ev = this.fromWire(JSON.parse(snap.ev));
    this.sel = snap.sel.map(s => ({v: s.v, e: this.list(s.v)[s.i]})).filter(s => s.e);
  },
  undo() { if (!this.undoStack.length) return; this.redoStack.push(this.snapshot()); this.restore(this.undoStack.pop()); this.commit(); },
  redo() { if (!this.redoStack.length) return; this.undoStack.push(this.snapshot()); this.restore(this.redoStack.pop()); this.commit(); },
  tidy() {
    let end = 0;
    for (const v of VOICES(this.ev)) {
      const list = this.list(v);
      const keep = list.filter(e => e.d > 0 && e.p.length).sort((a, b) => a.q - b.q);
      list.splice(0, list.length, ...keep);
      keep.forEach(e => end = Math.max(end, e.q + e.d));
    }
    const bar = this.barTicks();
    this.ev.length = Math.max(this.ev.length, Math.ceil(end / bar) * bar, bar);
    this.sel = this.sel.filter(s => this.list(s.v).includes(s.e));
  },
  commit() {
    if (this.dirty()) store.set(this.draftKey(), JSON.stringify(this.wire()));
    else store.del(this.draftKey());
    render();
  },
  // Make room for `keep`: notes it now overlaps are shortened (if they start earlier) or removed.
  clearSpace(list, keep) {
    const end = keep.q + keep.d;
    for (const o of [...list]) {
      if (o === keep) continue;
      if (o.q < keep.q && o.q + o.d > keep.q) o.d = keep.q - o.q;
      else if (o.q >= keep.q && o.q < end) {
        if (o.q === keep.q && o.lyric && !keep.lyric) keep.lyric = o.lyric;   // keep the word
        list.splice(list.indexOf(o), 1);
      }
    }
  },
  touched(e) { delete e.unsure; },   // a note you've edited is one you're sure of

  // ---- what's drawn where
  hitNote(target) {
    let g = target.closest && target.closest("g.note, g.chord");
    while (g) {
      const m = g.id.match(ID_RE);
      if (m) return {v: m[1], i: +m[2], grace: /-g$/.test(g.id)};
      g = g.parentElement && g.parentElement.closest("g.note, g.chord");
    }
    return null;
  },
  elementsOf(v, i) {
    const id = `${v}-${i}`, head = document.getElementById(id);
    return [...(head ? [head] : []), ...document.querySelectorAll(`#pages [id^="${id}-"]`)];
  },
  highlight() {
    document.querySelectorAll("#pages g.sel").forEach(el => el.classList.remove("sel"));
    for (const s of this.sel) this.elementsOf(s.v, this.list(s.v).indexOf(s.e)).forEach(el => el.classList.add("sel"));
    this.updateBar();
    this.announce();
  },
  // For screen readers: what's selected, as written on the page.
  describe(s) {
    const SHARPS = ["C", "C sharp", "D", "D sharp", "E", "F", "F sharp", "G", "G sharp", "A", "A sharp", "B"];
    const FLATS = ["C", "D flat", "D", "E flat", "E", "F", "G flat", "G", "A flat", "A", "B flat", "B"];
    const LENGTHS = {16: "triplet 16th", 24: "16th", 32: "triplet 8th", 36: "dotted 16th", 48: "8th", 64: "triplet quarter",
                     72: "dotted 8th", 96: "quarter", 128: "triplet half", 144: "dotted quarter", 192: "half", 288: "dotted half", 384: "whole"};
    const names = this.keySharps < 0 ? FLATS : SHARPS;
    const pitch = s.e.p.map(m => { const w = m + t; return `${names[((w % 12) + 12) % 12]} ${Math.floor(w / 12) - 1}`; }).join(" and ");
    const [, den] = this.ev.time_sig.split("/").map(Number), bar = this.barTicks(), beat = 4 * TPQ / den;
    const where = `bar ${Math.floor(s.e.q / bar) + 1} beat ${+((s.e.q % bar) / beat + 1).toFixed(2)}`;
    return [pitch, LENGTHS[s.e.d] || `${+(s.e.d / TPQ).toFixed(2)} beats`, where,
            s.e.lyric ? `word: ${s.e.lyric}` : "", s.v === "h" ? "harmony" : "", s.e.slur ? "slurred" : "",
            s.e.grace != null ? "with grace note" : ""].filter(Boolean).join(", ");
  },
  announce() {
    const text = !this.sel.length ? "" : this.sel.length > 1 ? `${this.sel.length} notes selected` : this.describe(this.sel[0]);
    if (text !== this.said) { this.said = text; $("#sr").textContent = text; }
  },
  keyAccidental(letter) {
    const ks = this.keySharps;
    if (ks > 0 && SHARP_ORDER.slice(0, ks).includes(letter)) return 1;
    if (ks < 0 && FLAT_ORDER.slice(0, -ks).includes(letter)) return -1;
    return 0;
  },
  // The staff, pitch and time under a click on empty staff space.
  spotAt(measure, x, y) {
    let best = null;
    [...measure.querySelectorAll("g.staff")].forEach((st, si) => {
      const lines = [...st.children].filter(c => c.tagName === "path").map(l => l.getBoundingClientRect());
      if (lines.length < 5) return;
      const ys = lines.map(b => b.top + b.height / 2).sort((a, b) => a - b);
      const top = ys[0], bottom = ys[ys.length - 1];
      const dist = y < top ? top - y : y > bottom ? y - bottom : 0;
      if (!best || dist < best.dist)
        best = {si, top, sp: (bottom - top) / 4, dist, left: Math.min(...lines.map(b => b.left)),
                right: Math.max(...lines.map(b => b.right))};
    });
    if (!best || best.dist > best.sp * 7) return null;
    const grand = this.ev.staves.length > 1;
    const v = grand ? "n" + best.si : this.voice;
    const treble = grand ? best.si === 0 : clef !== "bass";
    const steps = Math.round((y - best.top) / (best.sp / 2));
    const dia = (treble ? 38 : 26) - steps;                // top line: F5 (treble) or A3 (bass)
    const letter = ((dia % 7) + 7) % 7, octave = Math.floor(dia / 7);
    const written = 12 * (octave + 1) + LETTER_STEPS[letter] + this.keyAccidental(letter);
    // Time: interpolate between where things are drawn in this bar.
    const mi = player.measures.findIndex(m => m.id === measure.id);
    if (mi < 0) return null;
    const mq = player.measures[mi].q * TPQ;
    const nq = mi + 1 < player.measures.length ? player.measures[mi + 1].q * TPQ : mq + this.barTicks();
    // The bar's music starts at its first note or rest (rests fill every gap), or in an empty
    // bar (a whole-bar rest is drawn centred) just after any clef/key/time at the line's start.
    const anchors = [];
    let leftmost = Infinity, start = best.left;
    measure.querySelectorAll("g.note, g.rest, g.chord").forEach(el => {
      const b = el.getBoundingClientRect(); if (b.width) leftmost = Math.min(leftmost, b.left);
    });
    measure.querySelectorAll("g.clef, g.keySig, g.meterSig").forEach(el => {
      const b = el.getBoundingClientRect(); if (b.width) start = Math.max(start, b.right);
    });
    anchors.push([leftmost < Infinity ? leftmost : start + best.sp, mq]);
    measure.querySelectorAll("g.note, g.chord").forEach(el => {
      const m = el.id.match(/^(n\d+|h)-(\d+)$/);
      const e = m && this.list(m[1])[+m[2]];
      if (e && e.q >= mq && e.q < nq) anchors.push([el.getBoundingClientRect().left, e.q]);
    });
    anchors.push([best.right, nq]);
    anchors.sort((a, b) => a[0] - b[0]);
    let q = anchors[0][1];
    for (let i = 0; i + 1 < anchors.length; i++) {
      const [x0, q0] = anchors[i], [x1, q1] = anchors[i + 1];
      if (x >= x0 && x < x1) q = q0 + (x - x0) / Math.max(x1 - x0, 1) * (q1 - q0);
      else if (x >= x1) q = q1;
    }
    const step = this.trip ? TPQ / 3 : TPQ / 4;
    let ticks = Math.floor(q / step + 0.35) * step;        // just left of a note still counts for it
    ticks = Math.max(mq, Math.min(ticks, nq - step));
    return {v, q: ticks, pitch: Math.max(12, Math.min(120, written - t))};
  },

  // ---- clicks and keys
  click(e) {
    const hit = this.hitNote(e.target);
    if (hit) {
      const ev = this.list(hit.v)[hit.i];
      if (!ev) return;
      if (e.shiftKey || e.metaKey || e.ctrlKey) {
        const at = this.sel.findIndex(s => s.e === ev);
        if (at >= 0) this.sel.splice(at, 1); else this.sel.push({v: hit.v, e: ev});
      } else this.sel = [{v: hit.v, e: ev}];
      if (hit.v === "h" || hit.v === "n0") this.voice = hit.v;
      return this.highlight();
    }
    const measure = e.target.closest("g.measure");
    const spot = measure && this.spotAt(measure, e.clientX, e.clientY);
    if (!spot) { this.sel = []; return this.highlight(); }
    this.insert(spot.v, spot.q, spot.pitch);
  },
  key(e) {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && (e.key === "z" || e.key === "Z")) { e.preventDefault(); e.shiftKey ? this.redo() : this.undo(); return true; }
    if (mod && e.key === "y") { e.preventDefault(); this.redo(); return true; }
    if (mod && e.key === "s") { e.preventDefault(); this.save(); return true; }
    if (mod || e.altKey && !e.key.startsWith("Arrow")) return false;
    const k = e.key;
    const act = {
      ArrowUp: () => this.shift(e.shiftKey ? 12 : 1), ArrowDown: () => this.shift(e.shiftKey ? -12 : -1),
      ArrowLeft: () => e.altKey ? this.nudge(-1) : this.step(-1), ArrowRight: () => e.altKey ? this.nudge(1) : this.step(1),
      Delete: () => this.remove(), Backspace: () => this.remove(), "+": () => this.addAfter(), "=": () => this.addAfter(), Escape: () => { this.sel = []; this.highlight(); },
      Enter: () => this.editLyric(), l: () => this.editLyric(), L: () => this.editLyric(),
      s: () => this.slur(), S: () => this.slur(), "/": () => this.grace(-1), "?": () => this.grace(1),
      ".": () => this.setLength({dot: !this.dot}), t: () => this.setLength({trip: !this.trip}), T: () => this.setLength({trip: !this.trip}),
      1: () => this.setLength({base: 24}), 2: () => this.setLength({base: 48}), 3: () => this.setLength({base: 96}),
      4: () => this.setLength({base: 192}), 5: () => this.setLength({base: 384}),
    }[k];
    if (act) { e.preventDefault(); act(); return true; }
    const letter = "CDEFGAB".indexOf(k.toUpperCase());
    if (k.length === 1 && letter >= 0) { e.preventDefault(); this.letter(letter); return true; }
    return false;
  },

  // ---- edits
  insert(v, q, pitch) { this.insertAt(v, q, [pitch]); },
  insertAt(v, q, pitches) {
    const pitch = pitches[0];
    this.change(() => {
      const list = this.list(v), d = this.inputDur();
      if (this.ev.staves.length > 1 || !this.ev.mono) {          // chords: add to one that's there
        const same = list.find(e => e.q === q && e.d === d);
        if (same) {
          if (!same.p.includes(pitch)) same.p = [...same.p, pitch].sort((a, b) => a - b);
          this.touched(same); this.sel = [{v, e: same}]; return;
        }
      }
      const e = {q, d, p: [...pitches]};
      list.push(e);
      this.clearSpace(list, e);
      this.sel = [{v, e}];
    });
  },
  // A note straight after the selected one (same pitch, the chosen length). It goes in the
  // rest after the note if there's room, else takes the end of the selected note (splitting it
  // in two, e.g. where two sung notes were heard as one), else overwrites what follows.
  addAfter() {
    const s = this.sel[this.sel.length - 1];
    if (!s) return;
    const list = this.list(s.v), d = this.inputDur(), end = s.e.q + s.e.d;
    const next = list.filter(e => e.q >= end).sort((a, b) => a.q - b.q)[0];
    if (!next || next.q - end >= d) return this.insertAt(s.v, end, [...s.e.p]);
    if (s.e.d > d) {
      const e = s.e;
      return this.change(() => {
        e.d -= d; this.touched(e);
        const added = {q: e.q + e.d, d, p: [...e.p], slur: true};
        list.push(added);
        this.sel = [{v: s.v, e: added}];
      });
    }
    this.insertAt(s.v, end, [...s.e.p]);
  },
  shift(semis) {
    if (!this.sel.length) return;
    this.change(() => this.sel.forEach(s => { s.e.p = s.e.p.map(x => Math.max(12, Math.min(120, x + semis))); this.touched(s.e); }));
  },
  letter(L) {   // the nearest note with that letter name, as written (in the shown key)
    if (!this.sel.length) return;
    this.change(() => this.sel.forEach(s => {
      const written = s.e.p[s.e.p.length - 1] + t;
      let best = null;
      for (let o = 0; o < 10; o++) {
        const cand = 12 * (o + 1) + LETTER_STEPS[L] + this.keyAccidental(L);
        if (best === null || Math.abs(cand - written) < Math.abs(best - written)) best = cand;
      }
      const delta = best - t - s.e.p[s.e.p.length - 1];
      s.e.p = s.e.p.map(x => x + delta);
      this.touched(s.e);
    }));
  },
  setLength(opts) {
    Object.assign(this, opts);
    if (!this.sel.length) return this.updateBar();
    this.change(() => this.sel.forEach(s => { s.e.d = this.inputDur(); this.touched(s.e); this.clearSpace(this.list(s.v), s.e); }));
  },
  remove() {
    if (!this.sel.length) return;
    const first = this.sel[0], list = this.list(first.v), at = list.indexOf(first.e);
    this.change(() => {
      this.sel.forEach(s => { const l = this.list(s.v); l.splice(l.indexOf(s.e), 1); });
      this.sel = list[at - 1] ? [{v: first.v, e: list[at - 1]}] : [];
    });
  },
  step(dir) {   // select the next/previous note in the same voice
    const s = this.sel[this.sel.length - 1];
    const v = s ? s.v : this.voice, list = this.list(v);
    if (!list.length) return;
    const i = s ? list.indexOf(s.e) + dir : 0;
    if (i < 0 || i >= list.length) return;
    this.sel = [{v, e: list[i]}];
    this.highlight();
    this.elementsOf(v, i)[0]?.scrollIntoView({block: "nearest", behavior: "smooth"});
  },
  nudge(dir) {  // move earlier/later by a 16th (a triplet 8th in triplet mode)
    if (!this.sel.length) return;
    const by = (this.trip ? TPQ / 3 : TPQ / 4) * dir;
    this.change(() => this.sel.forEach(s => { s.e.q = Math.max(0, s.e.q + by); this.touched(s.e); this.clearSpace(this.list(s.v), s.e); }));
  },
  slur() {
    if (!this.sel.length) return;
    const on = !this.sel.every(s => s.e.slur);
    this.change(() => this.sel.forEach(s => { if (on) s.e.slur = true; else delete s.e.slur; }));
  },
  grace(dir) {
    if (!this.sel.length) return;
    this.change(() => this.sel.forEach(s => {
      if (s.e.grace != null) delete s.e.grace;
      else s.e.grace = s.e.p[0] + 2 * dir;
    }));
  },
  barOf() {
    const s = this.sel[0];
    return s ? Math.floor(s.e.q / this.barTicks()) * this.barTicks() : null;
  },
  insertBar() {
    const bar = this.barTicks(), at = this.barOf() ?? this.ev.length;
    this.change(() => {
      for (const v of VOICES(this.ev)) for (const e of this.list(v)) if (e.q >= at) e.q += bar;
      this.ev.length += bar;
    });
  },
  deleteBar() {
    const bar = this.barTicks(), at = this.barOf() ?? this.ev.length - bar;
    if (this.ev.length <= bar) return;
    this.change(() => {
      for (const v of VOICES(this.ev)) {
        const list = this.list(v);
        for (const e of [...list]) {
          if (e.q >= at && e.q < at + bar) list.splice(list.indexOf(e), 1);
          else if (e.q < at && e.q + e.d > at) e.d = at - e.q;
          else if (e.q >= at + bar) e.q -= bar;
        }
      }
      this.ev.length -= bar;
      this.sel = [];
    });
  },
  // `written` is the key signature as shown for the instrument; the note list keeps concert pitch.
  setKey(written) {
    const mode = this.ev.key[1] === "minor" ? "minor" : "major", table = mode === "minor" ? MINOR : MAJOR;
    const pc = name => ({C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11})[name[0]] + (name.includes("♯") ? 1 : name.includes("♭") ? -1 : 0);
    const concertPc = (((pc(table[written]) - t) % 12) + 12) % 12;
    const concert = Object.keys(table).filter(k => (pc(table[k]) % 12 + 12) % 12 === concertPc)
      .sort((a, b) => Math.abs(a) - Math.abs(b))[0];
    this.wantSharps = +written;   // if the page spells it the other way, flip it (see drawn)
    this.change(() => { this.ev.key = [m21Name(table[concert]), mode]; });
  },
  setTime(ts) { this.change(() => { this.ev.time_sig = ts; }); },

  editLyric() {
    const s = this.sel[0];
    if (!s || s.v === "h") return;
    if (!lyrics) { lyrics = true; this.afterRender = () => this.editLyric(); return render(); }
    const list = this.list(s.v), i = list.indexOf(s.e);
    const el = document.getElementById(`${s.v}-${i}`);
    if (!el) return;
    const measure = el.closest("g.measure"), staff = measure && measure.querySelector("g.staff");
    const nb = el.getBoundingClientRect(), sb = (staff || el).getBoundingClientRect();
    const inp = document.createElement("input");
    inp.className = "lyric-input"; inp.value = s.e.lyric || ""; inp.setAttribute("aria-label", "Word under this note");
    inp.style.left = `${nb.left + window.scrollX - 8}px`;
    inp.style.top = `${Math.max(nb.bottom, sb.bottom) + window.scrollY + 14}px`;
    document.body.append(inp);
    inp.focus(); inp.select();
    let done = false;
    const finish = (save, move) => {
      if (done) return;
      done = true; inp.remove();
      const text = inp.value.trim();
      const next = move && list[i + move] ? list[i + move] : null;
      if (next) { this.sel = [{v: s.v, e: next}]; this.afterRender = () => this.editLyric(); }
      if (save && text !== (s.e.lyric || "")) this.change(() => { if (text) s.e.lyric = text; else delete s.e.lyric; });
      else if (next) { this.highlight(); const go = this.afterRender; this.afterRender = null; go(); }
    };
    inp.onkeydown = ev => {
      ev.stopPropagation();
      if (ev.key === "Enter") { ev.preventDefault(); finish(true); }
      else if (ev.key === "Tab") { ev.preventDefault(); finish(true, ev.shiftKey ? -1 : 1); }
      else if (ev.key === "Escape") finish(false);
    };
    inp.onblur = () => finish(true);
  },

  // ---- saving
  async save() {
    if (!this.dirty()) return this.updateBar("Nothing to save");
    this.updateBar("Saving…");
    const r = await fetch("/api/edits", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({f: file, detail, events: this.wire()})});
    if (!r.ok) return this.updateBar(await r.text());
    this.savedJSON = JSON.stringify(this.wire());
    this.edited = true;
    store.del(this.draftKey());
    this.loadHistory();
    this.updateBar("Saved");
  },
  discard() {
    this.ev = this.fromWire(JSON.parse(this.savedJSON));
    store.del(this.draftKey());
    this.undoStack = []; this.redoStack = []; this.sel = [];
    render();
  },
  async revert() {
    if (Date.now() - this.revertArmed > 4000) {      // ask once more, in place
      this.revertArmed = Date.now();
      $("#revertBtn").classList.add("confirm");
      $("#revertBtn").textContent = "Click again to revert";
      setTimeout(() => { $("#revertBtn").classList.remove("confirm"); $("#revertBtn").textContent = "Revert to original"; }, 4000);
      return;
    }
    this.revertArmed = 0;
    $("#revertBtn").classList.remove("confirm"); $("#revertBtn").textContent = "Revert to original";
    const r = await fetch("/api/edits", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({f: file, detail, revert: true})});
    if (!r.ok) return this.updateBar(await r.text());
    store.del(this.draftKey());
    await this.load();
    render();
    this.updateBar("Back to the original (your saves are still in History)");
  },

  // ---- after each draw
  drawn(d) {
    this.keySharps = d.keySharps || 0;
    if (this.wantSharps != null) {
      const want = this.wantSharps;
      this.wantSharps = null;
      if (d.keySharps !== want && d.altSharps === want) { store.set("enh:" + t, enh() ? "0" : "1"); return render(); }
    }
    this.highlight();
    if (this.afterRender) { const go = this.afterRender; this.afterRender = null; go(); }
  },
  updateBar(message) {
    if (!this.ev) return;
    document.querySelectorAll("[data-base]").forEach(b => b.classList.toggle("on", +b.dataset.base === this.base));
    $("#dotBtn").classList.toggle("on", this.dot);
    $("#tripBtn").classList.toggle("on", this.trip);
    const showHarmony = harmony && !!this.ev.harmony && this.ev.staves.length === 1;
    $("#voiceGroup").hidden = !showHarmony;
    if (!showHarmony && this.voice === "h") this.voice = "n0";
    document.querySelectorAll("[data-voice]").forEach(b => b.classList.toggle("on", b.dataset.voice === this.voice));
    document.querySelectorAll("#editBar [data-act]").forEach(b => {
      if (["add", "delete", "lyric", "slur", "grace"].includes(b.dataset.act)) b.disabled = !this.sel.length;
    });
    $("#undoBtn").disabled = !this.undoStack.length;
    $("#redoBtn").disabled = !this.redoStack.length;
    const dirty = this.dirty();
    $("#saveBtn").disabled = !dirty;
    $("#discardBtn").disabled = !dirty;
    $("#revertBtn").hidden = !this.edited;
    const state = $("#editState");
    state.classList.toggle("dirty", dirty && !message);
    state.textContent = message || (dirty ? "Unsaved changes" : this.edited ? "Your saved edits" : "The original transcription");
    const mode = this.ev.key[1] === "minor" ? "minor" : "major";
    $("#keySel").innerHTML = Object.keys(MAJOR).sort((a, b) => a - b).map(k =>
      `<option value="${k}"${+k === this.keySharps ? " selected" : ""}>${(mode === "minor" ? MINOR : MAJOR)[k]} ${mode}</option>`).join("");
    $("#timeSel").value = this.ev.time_sig;
  },
};

$("#editBtn").onclick = () => editor.toggle();
document.querySelectorAll("[data-voice]").forEach(b => b.onclick = () => { editor.voice = b.dataset.voice; editor.updateBar(); });
document.querySelectorAll("[data-base]").forEach(b => b.onclick = () => editor.setLength({base: +b.dataset.base}));
$("#dotBtn").onclick = () => editor.setLength({dot: !editor.dot});
$("#tripBtn").onclick = () => editor.setLength({trip: !editor.trip});
document.querySelectorAll("#editBar [data-act]").forEach(b => b.onclick = () => ({
  add: () => editor.addAfter(), delete: () => editor.remove(), lyric: () => editor.editLyric(), slur: () => editor.slur(),
  grace: () => editor.grace(-1), insertBar: () => editor.insertBar(), deleteBar: () => editor.deleteBar(),
})[b.dataset.act]());
$("#keySel").onchange = e => editor.setKey(e.target.value);
$("#timeSel").onchange = e => editor.setTime(e.target.value);
$("#histSel").onchange = e => editor.openVersion(e.target.value);
$("#undoBtn").onclick = () => editor.undo();
$("#redoBtn").onclick = () => editor.redo();
$("#saveBtn").onclick = () => editor.save();
$("#discardBtn").onclick = () => editor.discard();
$("#revertBtn").onclick = () => editor.revert();

// ---- new song panel
let chosenFile = null, startedAt = 0;
function openNew() {
  $("#new").hidden = false; $("#newForm").hidden = false;
  $("#new h2").textContent = mode === "split" ? "Split a song" : "Add a song";
  $("#src").focus();
}
$("#newBtn").onclick = () => { if ($("#new").hidden) openNew(); else $("#new").hidden = true; };
// Introduction: shown until dismissed once, and again from "How it works".
function showWelcome(show) { $("#welcome").hidden = !show; if (!show) store.set("welcomed", "1"); }
$("#helpBtn").onclick = () => { showWelcome($("#welcome").hidden); if (!$("#welcome").hidden) window.scrollTo({top: 0}); };
$("#welcomeClose").onclick = () => showWelcome(false);
$("#welcomeNew").onclick = () => { showWelcome(false); openNew(); };
if (!store.get("welcomed")) showWelcome(true);
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
  const job = mode === "split"
    ? {source, split: {model: $("#newModel").value, vocal_layers: $("#newLayers").checked}}
    : {source, part: $("#part").value, bpm: $("#bpm").value || null};
  const r = await fetch("/api/job", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(job)});
  if (!r.ok) { $("#go").disabled = false; return showProgress(await r.text(), true); }
  startedAt = Date.now();
  poll();
};

function showProgress(html, isError) {
  $("#progress").innerHTML = isError ? `<span class="err">${esc(html)}</span>` : html;
}
const esc = s => s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

async function poll() {
  const j = await (await fetch("/api/job")).json();
  // A finished job only changes this page if this page started it or watched it run (not,
  // say, the result of an earlier job still remembered by the server after a refresh).
  const watched = startedAt !== 0 || j.state === "running";
  if (!watched) return;
  $("#new h2").textContent = j.kind === "harmony" ? "Adding harmony" : j.kind === "split" ? "Splitting" : "Add a song";
  $("#newForm").hidden = (j.kind === "harmony" || (j.kind === "split" && mode !== "split")) && j.state !== "idle";
  $("#splitGo").disabled = j.state === "running";
  if (j.state === "running") {
    $("#new").hidden = false; $("#go").disabled = true; $("#harmonyBtn").disabled = true;
    if (!startedAt) startedAt = Date.now();
    const secs = Math.round((Date.now() - startedAt) / 1000);
    const steps = j.log.map(esc);
    if (steps.length) steps[steps.length - 1] = `<b>${steps[steps.length - 1]}…</b>`;
    showProgress((steps.length ? steps.join("\n") : "<b>Starting…</b>") + `\n${secs}s elapsed`);
    return setTimeout(poll, 1000);
  }
  $("#go").disabled = false; $("#harmonyBtn").disabled = false; startedAt = 0;
  if (j.state === "error") return showProgress(j.error, true);
  if (j.state === "done" && j.result) {
    showProgress(j.kind === "harmony" ? "<b>Harmony added.</b> It shows as a second voice with stems down."
                 : j.kind === "split" ? `<b>Split.</b> Mix the parts of ${esc(j.result)} below.`
                 : `<b>Done.</b> Showing ${esc(j.result)}`);
    if (j.kind === "split") {
      if (j.model) split.model = j.model;
      if (mode !== "split") setMode("split");
      return loadSplit(j.result);
    }
    $("#src").value = ""; chosenFile = null; $("#file").value = ""; $("#filename").textContent = "";
    await loadList(j.result);
  }
}

async function loadList(select) {
  const list = await (await fetch("/api/list")).json();
  if (!list.length) {
    $("#new").hidden = false;
    $("#pages").innerHTML = `<div id="msg">No songs yet. Paste a YouTube link above, or choose an audio file, to make your first score.</div>`;
    return;
  }
  file = select && list.includes(select) ? select : (list.includes(file) ? file : list[list.length - 1]);
  // "Song name/vocals.musicxml" -> "Song name (vocals)"
  const songLabel = f => { const m = f.match(/^(.*)\/([^/]+)\.musicxml$/); return m ? `${m[1]} (${m[2]})` : f; };
  $("#score").innerHTML = list.map(f => `<option value="${esc(f)}"${f === file ? " selected" : ""}>${esc(songLabel(f))}</option>`).join("");
  const switched = file !== player.file;
  t = +(store.get("t:" + file) ?? 9);
  inst = +(store.get("inst:" + file) ?? 9);
  detail = store.get("level:" + file) || "balanced";
  if (switched) player.reset(file);
  render();
}


// ---- split: pull a song apart and mix its parts
let mode = store.get("mode") === "split" ? "split" : "sheet";
const PART_NAMES = {vocals: "Vocals", lead_vocals: "Lead vocal", backing_vocals: "Backing vocals", drums: "Drums",
                    bass: "Bass", guitar: "Guitar", piano: "Piano", other: "Other"};
const PART_TITLES = {other: "Everything else: keys, synths, strings, horns…"};
const split = {songs: [], song: null, model: store.get("splitModel") || "htdemucs", parts: [], gains: {}, on: {},
               audio: Object.assign(new Audio(), {preload: "auto"}), timer: 0, playing: false};

function setMode(m) {
  mode = m; store.set("mode", m);
  document.body.classList.toggle("split", m === "split");
  $("#splitView").hidden = m !== "split";
  document.querySelectorAll("[data-mode]").forEach(b => {
    b.classList.toggle("on", b.dataset.mode === m); b.setAttribute("aria-selected", b.dataset.mode === m); });
  $("#go").textContent = m === "split" ? "Split" : "Transcribe";
  if (!$("#new").hidden && !$("#newForm").hidden) $("#new h2").textContent = m === "split" ? "Split a song" : "Add a song";
  if (m === "split") { player.stop(); loadSplit(); } else { split.audio.pause(); split.playing = false; mixButton(); }
}
document.querySelectorAll("[data-mode]").forEach(b => b.onclick = () => setMode(b.dataset.mode));

async function loadSplit(select) {
  const d = await (await fetch("/api/songs")).json();
  split.songs = d.songs;
  showSites(d.sites);
  $("#noSongs").hidden = d.songs.length > 0;
  $("#splitEmpty").hidden = $("#mixer").hidden = true;
  if (!d.songs.length) { $("#splitSong").innerHTML = ""; return; }
  const name = select || split.song || store.get("splitSong");
  split.song = d.songs.some(x => x.name === name) ? name : d.songs[0].name;
  $("#splitSong").innerHTML = d.songs.map(x => `<option value="${esc(x.name)}"${x.name === split.song ? " selected" : ""}>${esc(x.name)}</option>`).join("");
  showSplit();
}

function showSplit() {
  store.set("splitSong", split.song); store.set("splitModel", split.model);
  document.querySelectorAll("[data-model]").forEach(b => b.classList.toggle("on", b.dataset.model === split.model));
  const entry = split.songs.find(x => x.name === split.song);
  const parts = (entry && entry.splits[split.model]) || [];
  $("#splitEmpty").hidden = parts.length > 0;
  $("#mixer").hidden = !parts.length;
  if (!parts.length) {
    $("#splitEmptyMsg").textContent = split.model === "htdemucs"
      ? "Not split yet. Splitting separates vocals, drums, bass and everything else, and takes a minute or two."
      : "Not split into 6 parts yet. This adds guitar and piano (those two come out less clean than the rest), and takes a couple of minutes.";
    split.audio.pause(); split.playing = false; mixButton();
    return;
  }
  // With lead/backing available, they stand in for "vocals".
  split.parts = parts.includes("lead_vocals") ? parts.filter(x => x !== "vocals") : parts;
  const saved = JSON.parse(store.get(`mix:${split.song}:${split.model}`) || "null") || {};
  split.parts.forEach(x => { split.gains[x] = saved[x]?.g ?? 1; split.on[x] = saved[x]?.on ?? true; });
  $("#stems").innerHTML = split.parts.map(x => `
    <div class="part-row" data-part="${x}">
      <div class="name"><button class="toggle" aria-label="${PART_NAMES[x]} on or off"></button>
        <span title="${PART_TITLES[x] || ""}">${PART_NAMES[x] || x}</span></div>
      <input type="range" min="0" max="200" step="5" aria-label="${PART_NAMES[x]} volume">
      <span class="pct"></span>
      <button class="solo quiet" title="Hear only this part">Solo</button>
    </div>`).join("");
  document.querySelectorAll("#stems .part-row").forEach(row => {
    const x = row.dataset.part;
    row.querySelector(".toggle").onclick = () => { split.on[x] = !split.on[x]; mixChanged(); };
    row.querySelector("input").oninput = e => { split.gains[x] = +e.target.value / 100; split.on[x] = true; mixChanged(); };
    row.querySelector(".solo").onclick = () => { split.parts.forEach(y => split.on[y] = y === x); mixChanged(); };
  });
  drawPresets();
  mixChanged(true);
}

function drawPresets() {
  const has = x => split.parts.includes(x), vox = split.parts.filter(x => x.includes("vocals"));
  const only = keep => () => split.parts.forEach(y => split.on[y] = keep.includes(y));
  const presets = [["Everything", only(split.parts)],
                   ["No vocals", only(split.parts.filter(x => !vox.includes(x)))],
                   ["Vocals only", only(vox)],
                   ["No drums", only(split.parts.filter(x => x !== "drums"))],
                   ["Drums and bass", only(["drums", "bass"])]];
  if (has("lead_vocals")) presets.splice(2, 0, ["Backing vocals and band", only(split.parts.filter(x => x !== "lead_vocals"))]);
  if (has("piano")) presets.push(["Piano and drums", only(["piano", "drums"])]);
  if (has("guitar")) presets.push(["No guitar", only(split.parts.filter(x => x !== "guitar"))]);
  $("#presets").innerHTML = presets.map(([name], i) => `<button data-preset-i="${i}">${name}</button>`).join("");
  document.querySelectorAll("[data-preset-i]").forEach(b => b.onclick = () => {
    presets[+b.dataset.presetI][1]();
    split.parts.forEach(y => { if (split.on[y] && !split.gains[y]) split.gains[y] = 1; });
    mixChanged();
  });
}

const mixGains = () => split.parts.map(x => `${x}:${split.on[x] ? split.gains[x] : 0}`).join(",");
const mixLabel = () => {
  const on = split.parts.filter(x => split.on[x] && split.gains[x] > 0);
  if (on.length === split.parts.length) return "full mix";
  if (!on.length) return "silence";
  const off = split.parts.filter(x => !on.includes(x));
  return off.length < on.length && off.length <= 2 ? "no " + off.map(x => PART_NAMES[x].toLowerCase()).join(", no ")
                                                    : on.map(x => PART_NAMES[x].toLowerCase()).join(" + ");
};
const mixQuery = () => `song=${encodeURIComponent(split.song)}&model=${split.model}&g=${mixGains()}`;

function mixChanged(immediate) {
  const saved = {};
  split.parts.forEach(x => saved[x] = {g: split.gains[x], on: split.on[x]});
  store.set(`mix:${split.song}:${split.model}`, JSON.stringify(saved));
  document.querySelectorAll("#stems .part-row").forEach(row => {
    const x = row.dataset.part, on = split.on[x];
    row.classList.toggle("off", !on);
    const t = row.querySelector(".toggle");
    t.textContent = on ? "On" : "Off"; t.classList.toggle("on", on); t.setAttribute("aria-pressed", on);
    row.querySelector("input").value = Math.round(split.gains[x] * 100);
    row.querySelector(".pct").textContent = `${Math.round(split.gains[x] * 100)}%`;
  });
  // Rebuild the mix shortly after the last change, then carry on from the same moment.
  clearTimeout(split.timer);
  split.timer = setTimeout(() => {
    const src = `/api/mix?${mixQuery()}`;
    if (split.audio.dataset.src === src) return;
    const at = split.audio.currentTime || 0, resume = split.playing;
    $("#mixState").textContent = "Mixing…";
    split.audio.src = src; split.audio.dataset.src = src;
    split.audio.addEventListener("loadedmetadata", () => {
      split.audio.currentTime = Math.min(at, split.audio.duration || at);
      if (resume) split.audio.play();
      $("#mixState").textContent = "";
    }, {once: true});
  }, immediate ? 0 : 300);
}

function mixButton() { $("#mixPlay").textContent = split.playing ? "❚❚ Pause" : "▶ Play"; }
function mixToggle() {
  if (!split.audio.src) return;
  if (split.playing) split.audio.pause(); else split.audio.play();
  split.playing = !split.playing; mixButton();
}
$("#mixPlay").onclick = mixToggle;
split.audio.addEventListener("ended", () => { split.playing = false; mixButton(); });
split.audio.addEventListener("error", () => { $("#mixState").textContent = "Couldn't build that mix"; });
const fmtTime = sec => `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;
split.audio.addEventListener("timeupdate", () => {
  const a = split.audio;
  if (!split.dragging) $("#mixScrub").value = a.currentTime;
  $("#mixScrub").max = a.duration || 1;
  $("#mixClock").textContent = `${fmtTime(a.currentTime)} / ${fmtTime(a.duration || 0)}`;
});
$("#mixScrub").oninput = e => { split.dragging = true; split.audio.currentTime = +e.target.value; };
$("#mixScrub").onchange = () => { split.dragging = false; };
$("#mixDownload").onclick = () => {
  location.href = `/api/mixdownload?${mixQuery()}&fmt=${$("#mixFmt").value}&label=${encodeURIComponent(mixLabel())}`;
};
$("#mixFmt").value = store.get("mixFmt") || "mp3";
$("#mixFmt").onchange = e => store.set("mixFmt", e.target.value);
$("#splitSong").onchange = e => { split.song = e.target.value; split.audio.pause(); split.playing = false; mixButton(); split.audio.removeAttribute("src"); delete split.audio.dataset.src; showSplit(); };
document.querySelectorAll("[data-model]").forEach(b => b.onclick = () => {
  split.model = b.dataset.model; split.audio.pause(); split.playing = false; mixButton(); showSplit(); });
$("#splitGo").onclick = async () => {
  const r = await fetch("/api/job", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({split: {song: split.song, model: split.model, vocal_layers: $("#splitLayers").checked}})});
  $("#new").hidden = false;
  if (!r.ok) return showProgress(await r.text(), true);
  startedAt = Date.now(); poll();
};

function showSites(sites) {
  if (!sites) return;
  $("#sites").textContent = `Links from ${sites.download.join(", ")} download directly. ` +
    `${sites.lookup.join(", ")} links are matched to the same song on YouTube. Files: any audio or video format.`;
}

(async () => {
  fetch("/api/songs").then(r => r.json()).then(d => showSites(d.sites)).catch(() => {});
  setMode(mode);
  await loadList();
  poll();   // pick up a job that was already running (e.g. after a page refresh)
})();

</script>
</body></html>
"""


if __name__ == "__main__":
    serve(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
