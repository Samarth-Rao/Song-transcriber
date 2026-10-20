"""Singing -> notes with GAME (https://github.com/openvpi/GAME, MIT), run from its ONNX release.

GAME is a model trained specifically to transcribe singing into note boundaries and pitches.
The upstream repo only ships a PyTorch inference path; this follows its documented ONNX
workflow (ONNX.md): encoder -> segmenter (D3PM refinement loop) -> estimator.

Model files: models/GAME-1.0.3-medium-onnx (release v1.0.3, sha256-verified on download).
"""
import json
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models" / "GAME-1.0.3-medium-onnx"
MODEL_URL = "https://github.com/openvpi/GAME/releases/download/v1.0.3/GAME-1.0.3-medium-onnx.zip"
MODEL_SHA256 = "5b7a21e64c6310efac399f5d12838fffa70565be162436b5a4a65f290721e7d8"  # GitHub release digest


def ensure_model(model_dir: Path = MODEL_DIR):
    """Download the release once; refuse it unless its sha256 matches the pinned digest."""
    import hashlib
    import io
    import urllib.request
    import zipfile

    if (model_dir / "config.json").exists():
        return model_dir
    with urllib.request.urlopen(MODEL_URL, timeout=600) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(f"GAME model checksum mismatch ({digest}); not using it")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        expected = {"config.json", "encoder.onnx", "segmenter.onnx", "estimator.onnx", "dur2bd.onnx", "bd2dur.onnx"}
        if {Path(n).name for n in names} != expected or any(".." in n or n.startswith("/") for n in names):
            raise RuntimeError(f"unexpected files in GAME model archive: {names}")
        z.extractall(model_dir.parent)
    return model_dir


def seeded(path: Path, seed: float = 1234.0) -> bytes:
    """The model with a fixed seed on its random ops (the segmenter's D3PM loop starts from
    random noise), so the same audio always gives the same notes. The file isn't modified."""
    import onnx

    model = onnx.load(str(path))
    for node in model.graph.node:
        if node.op_type.startswith("Random") or node.op_type in ("Bernoulli", "Multinomial"):
            node.attribute.extend([onnx.helper.make_attribute("seed", seed)])
    return model.SerializeToString()


class Game:
    def __init__(self, model_dir: Path = MODEL_DIR):
        import onnxruntime as ort

        self.config = json.loads((model_dir / "config.json").read_text())
        self.sr = self.config["samplerate"]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        load = lambda name: ort.InferenceSession(str(model_dir / f"{name}.onnx"), opts,
                                                 providers=["CPUExecutionProvider"])
        self.encoder, self.estimator = load("encoder"), load("estimator")
        self.segmenter = ort.InferenceSession(seeded(model_dir / "segmenter.onnx"), opts,
                                              providers=["CPUExecutionProvider"])
        self.dur2bd, self.bd2dur = load("dur2bd"), load("bd2dur")

    def run_chunk(self, wav, known_durations=None, language="en", steps=8,
                  seg_threshold=0.2, seg_radius=0.02, est_threshold=0.2):
        """Notes in one audio chunk: list of (start_s, end_s, midi_pitch_float).

        `known_durations` optionally fixes region boundaries (seconds, summing to the chunk
        length) - e.g. word spans - and GAME only decides the notes inside them.
        Defaults match upstream infer.py.
        """
        f32 = np.float32
        duration = len(wav) / self.sr
        x_seg, x_est, mask_t = self.encoder.run(None, {
            "waveform": wav[None].astype(f32), "duration": np.array([duration], f32)})
        known = np.array([known_durations or [duration]], f32)
        known_bd = self.dur2bd.run(None, {"durations": known, "maskT": mask_t})[0]
        lang = np.array([self.config["languages"].get(language, 0) if self.config["languages"] else 0], np.int64)
        bd = known_bd
        for i in range(steps if self.config.get("loop") else 1):   # D3PM: t from 0 (noise) towards 1
            feeds = {"x_seg": x_seg, "language": lang, "known_boundaries": known_bd, "maskT": mask_t,
                     "threshold": np.array(seg_threshold, f32),
                     "radius": np.array(round(seg_radius / self.config["timestep"]), np.int64)}
            if self.config.get("loop"):
                feeds.update(prev_boundaries=bd, t=np.array([i / steps], f32))
            bd = self.segmenter.run(None, feeds)[0]
        durations, mask_n = self.bd2dur.run(None, {"boundaries": bd, "maskT": mask_t})
        presence, scores = self.estimator.run(None, {
            "x_est": x_est, "boundaries": bd, "maskT": mask_t, "maskN": mask_n,
            "threshold": np.array(est_threshold, f32)})
        notes, t = [], 0.0
        for d, valid, voiced, pitch in zip(durations[0], mask_n[0], presence[0], scores[0]):
            if not valid:
                break
            if voiced:
                notes.append((t, t + float(d), float(pitch)))
            t += float(d)
        return notes

    def transcribe(self, path: Path, language="en"):
        """Whole file: sliced at silences like upstream, notes in seconds from the file start."""
        import librosa
        from game_slicer import Slicer

        wav, _ = librosa.load(str(path), sr=self.sr, mono=True)
        slicer = Slicer(sr=self.sr, threshold=-40., min_length=1000, min_interval=200, max_sil_kept=100)
        notes = []
        for chunk in slicer.slice(wav):
            off = chunk["offset"]
            notes += [(s + off, e + off, p) for s, e, p in self.run_chunk(chunk["waveform"], language=language)]
        return notes
