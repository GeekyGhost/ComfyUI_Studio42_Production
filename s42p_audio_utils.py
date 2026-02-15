"""
S42 Production Suite — Shared Audio Utilities
==============================================
Common DSP helpers used across all audio nodes.
All processing is CPU/numpy/scipy — GPU stays free for Wan/Qwen.

Python 3.12 | ComfyUI Portable | torchaudio + scipy + numpy
"""

import numpy as np
import torch
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# ── Optional dependency flags ─────────────────────────────────────────────
try:
    import scipy.signal as signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available — EQ and filtering will be limited")

try:
    import torchaudio
    import torchaudio.functional as TAF
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logger.warning("torchaudio not available — resampling will use linear interpolation")

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False
    logger.warning("pyloudnorm not available — LUFS measurement will use RMS estimate")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available — beat detection will be limited")


# ── ComfyUI AUDIO format helpers ──────────────────────────────────────────

def audio_from_comfy(audio_dict: dict) -> Tuple[np.ndarray, int]:
    """
    Convert ComfyUI AUDIO dict → (numpy float32 array [channels, samples], sample_rate).
    Handles LazyAudioMap, plain dict with waveform tensor, etc.
    """
    waveform = audio_dict.get("waveform")
    sample_rate = int(audio_dict.get("sample_rate", 44100))

    if waveform is None:
        raise ValueError("AUDIO dict has no 'waveform' key")

    # waveform may be [B, C, S] or [C, S] — normalise to [C, S]
    if isinstance(waveform, torch.Tensor):
        w = waveform.detach().cpu().float()
    else:
        w = torch.as_tensor(waveform, dtype=torch.float32)

    if w.ndim == 3:
        w = w[0]          # drop batch dim → [C, S]
    elif w.ndim == 1:
        w = w.unsqueeze(0)  # mono [S] → [1, S]

    return w.numpy(), sample_rate


def audio_to_comfy(arr: np.ndarray, sample_rate: int) -> dict:
    """
    Convert (numpy float32 [C, S], sr) → ComfyUI AUDIO dict.
    Ensures output is [1, C, S] (batch=1).
    """
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]   # [S] → [1, S]
    tensor = torch.from_numpy(arr.astype(np.float32))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)   # [C, S] → [1, C, S]
    return {"waveform": tensor, "sample_rate": int(sample_rate)}


def ensure_stereo(arr: np.ndarray) -> np.ndarray:
    """Guarantee output is [2, S]. Duplicate mono if needed."""
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 2, axis=0)
    return arr[:2, :]   # trim to stereo if more channels


def ensure_float32(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32, copy=False)


# ── dB / linear helpers ───────────────────────────────────────────────────

def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float, floor_db: float = -120.0) -> float:
    if abs(linear) < 1e-10:
        return floor_db
    return 20.0 * np.log10(abs(linear))


def rms_db(arr: np.ndarray) -> float:
    rms = np.sqrt(np.mean(arr ** 2))
    return linear_to_db(rms)


def peak_db(arr: np.ndarray) -> float:
    return linear_to_db(np.max(np.abs(arr)))


# ── LUFS measurement ──────────────────────────────────────────────────────

def measure_lufs(arr: np.ndarray, sample_rate: int) -> float:
    """
    Measure integrated loudness in LUFS (EBU R128).
    Falls back to RMS-based estimate if pyloudnorm not available.
    """
    stereo = ensure_stereo(arr)
    audio_interleaved = stereo.T   # [S, 2]

    if PYLOUDNORM_AVAILABLE:
        try:
            meter = pyln.Meter(sample_rate)
            return float(meter.integrated_loudness(audio_interleaved))
        except Exception as e:
            logger.warning(f"pyloudnorm measurement failed ({e}), falling back to RMS estimate")

    # RMS-based LUFS estimate (roughly calibrated, not spec-accurate)
    rms = np.sqrt(np.mean(audio_interleaved ** 2))
    return float(20.0 * np.log10(rms + 1e-10) - 0.691)


def measure_true_peak_db(arr: np.ndarray) -> float:
    """Measure true peak in dBTP (oversample 4x for inter-sample peaks)."""
    stereo = ensure_stereo(arr)
    if SCIPY_AVAILABLE:
        # 4x oversample via polyphase resampling for true peak detection
        try:
            oversampled = signal.resample_poly(stereo, 4, 1, axis=1)
            return float(peak_db(oversampled))
        except Exception:
            pass
    return float(peak_db(stereo))


def lufs_gain(arr: np.ndarray, sample_rate: int, target_lufs: float) -> float:
    """Calculate linear gain needed to reach target_lufs."""
    current = measure_lufs(arr, sample_rate)
    if current <= -70.0:
        return 1.0   # silence, don't touch
    delta_db = target_lufs - current
    return db_to_linear(delta_db)


# ── Mid/Side processing ───────────────────────────────────────────────────

def to_mid_side(stereo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert [2, S] L/R to Mid and Side channels."""
    mid  = (stereo[0] + stereo[1]) * 0.5
    side = (stereo[0] - stereo[1]) * 0.5
    return mid, side


def from_mid_side(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Convert Mid/Side back to [2, S] L/R."""
    L = mid + side
    R = mid - side
    return np.stack([L, R], axis=0)


def stereo_width(arr: np.ndarray, width: float) -> np.ndarray:
    """
    Adjust stereo width via M/S processing.
    width=0.0 → full mono, width=1.0 → unchanged, width=2.0 → double width.
    """
    stereo = ensure_stereo(arr)
    mid, side = to_mid_side(stereo)
    side = side * width
    return from_mid_side(mid, side)


# ── Soft limiter ──────────────────────────────────────────────────────────

def soft_limit(arr: np.ndarray, threshold_db: float = -0.3) -> np.ndarray:
    """
    Transparent soft knee limiter using tanh saturation.
    Prevents hard clipping while preserving transients.
    """
    thresh = db_to_linear(threshold_db)
    return np.tanh(arr / thresh) * thresh


# ── Dynamic range measurement ─────────────────────────────────────────────

def measure_dynamic_range(arr: np.ndarray) -> float:
    """
    Crest factor-based dynamic range estimate in dB.
    Higher = more dynamic (less crushed).
    """
    peak = np.max(np.abs(arr))
    rms  = np.sqrt(np.mean(arr ** 2))
    if rms < 1e-10:
        return 0.0
    return float(20.0 * np.log10(peak / rms))
