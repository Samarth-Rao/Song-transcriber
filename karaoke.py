"""Vocals -> lead + backing vocals with UVR's MDX-Net karaoke model, run from its ONNX file.

The model (UVR_MDXNET_KARA_2, from the Ultimate Vocal Remover model release) is a plain ONNX
graph of convolutions; this file is the minimal numpy STFT + onnxruntime wrapper around it,
following UVR's MDX-Net inference (python-audio-separator, MIT). No torch, no pickle.

Model file: models/UVR_MDXNET_KARA_2.onnx (sha256-verified on download).
"""
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "UVR_MDXNET_KARA_2.onnx"
MODEL_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_KARA_2.onnx"
MODEL_SHA256 = "bf32e15105a09c0f7dddd2b67346146334d6f3ecb399ed7638eba2ab07cbf5f4"

# From UVR's model parameter table for this file.
N_FFT, DIM_F, DIM_T, COMPENSATE = 5120, 2048, 256, 1.065
HOP, SR, OVERLAP = 1024, 44100, 0.25


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download the model once; refuse it unless its sha256 matches the pinned digest."""
    import hashlib
    import urllib.request

    if path.exists():
        return path
    with urllib.request.urlopen(MODEL_URL, timeout=600) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(f"karaoke model checksum mismatch ({digest}); not using it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _stft(x):
    """[channels, T] -> [channels, 2 (re/im), freq, frames]; matches torch.stft(center=True)."""
    win = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    xp = np.pad(x, ((0, 0), (N_FFT // 2, N_FFT // 2)), mode="reflect")
    frames = 1 + (xp.shape[1] - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    spec = np.transpose(np.fft.rfft(xp[:, idx] * win, axis=-1), (0, 2, 1))
    return np.stack([spec.real, spec.imag], 1).astype(np.float32)


def _istft(spec, length):
    """[channels, freq, frames] complex -> [channels, length] (overlap-add, window-normalised)."""
    win = np.hanning(N_FFT + 1)[:-1]
    frames = np.fft.irfft(np.transpose(spec, (0, 2, 1)), n=N_FFT, axis=-1) * win
    total = N_FFT + HOP * (frames.shape[1] - 1)
    out, wsum = np.zeros((spec.shape[0], total)), np.zeros(total)
    for i in range(frames.shape[1]):
        out[:, i * HOP:i * HOP + N_FFT] += frames[:, i]
        wsum[i * HOP:i * HOP + N_FFT] += win ** 2
    out /= np.where(wsum > 1e-11, wsum, 1)
    return out[:, N_FFT // 2:N_FFT // 2 + length]


def _run_chunk(sess, chunk):
    spec = _stft(chunk)[:, :, :DIM_F, :].reshape(1, 4, DIM_F, -1)   # L_re, L_im, R_re, R_im
    spec[:, :, :3, :] = 0
    out = sess.run(None, {"input": spec})[0][0].reshape(2, 2, DIM_F, -1)
    full = np.zeros((2, N_FFT // 2 + 1, out.shape[-1]), np.complex64)
    full[:, :DIM_F] = out[:, 0] + 1j * out[:, 1]
    return _istft(full, chunk.shape[1])


def split(vocals: Path):
    """Returns (lead, backing) as [2, T] float arrays at 44.1 kHz."""
    import librosa
    import onnxruntime as ort

    mix, _ = librosa.load(str(vocals), sr=SR, mono=False)
    if mix.ndim == 1:
        mix = np.stack([mix, mix])
    peak = float(np.abs(mix).max()) or 1.0
    gain = 0.9 / peak if peak > 0.9 else 1.0

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(ensure_model()), opts, providers=["CPUExecutionProvider"])

    # Overlapping chunks, Hann-weighted and averaged (UVR's MDX-Net chunking).
    trim = N_FFT // 2
    chunk_size = HOP * (DIM_T - 1)
    gen = chunk_size - 2 * trim
    padded = np.concatenate([np.zeros((2, trim)), mix * gain,
                             np.zeros((2, gen + trim - mix.shape[1] % gen))], 1).astype(np.float32)
    step = int((1 - OVERLAP) * chunk_size)
    res, div = np.zeros_like(padded), np.zeros_like(padded)
    for i in range(0, padded.shape[1], step):
        end = min(i + chunk_size, padded.shape[1])
        part = padded[:, i:end]
        if end - i < chunk_size:
            part = np.concatenate([part, np.zeros((2, chunk_size - (end - i)), np.float32)], 1)
        w = np.hanning(end - i)[None]
        res[:, i:end] += _run_chunk(sess, part)[:, :end - i] * w
        div[:, i:end] += w
    backing = (res / np.where(div > 0, div, 1))[:, trim:trim + mix.shape[1]] / gain
    lead = mix - backing * COMPENSATE
    return lead, backing
