"""
S42 Production Suite — Parametric EQ Node
==========================================
8-band parametric equalizer with full per-band control.
Uses scipy sosfilt (second-order sections) for stable, accurate filtering.

Bands: High-Pass → Low-Shelf → 4x Peak/Notch → High-Shelf → Low-Pass
All filters are zero-latency (causal), 2nd-order Butterworth/peaking designs.

Python 3.12 | ComfyUI Portable | scipy required
"""

import numpy as np
import logging
import json
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    import scipy.signal as sig
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.error("scipy is required for the Parametric EQ node. Install with: pip install scipy")

# Pull in shared utils (relative import works because __init__.py adds parent to path)
from s42p_audio_utils import (
    audio_from_comfy, audio_to_comfy, ensure_stereo, ensure_float32,
    rms_db, peak_db
)

CATEGORY = "S42 Production Suite 🎛️ Audio Mastering"


# ── DSP: individual filter builders ──────────────────────────────────────────

def _highpass_sos(freq: float, sr: int, order: int = 2) -> np.ndarray:
    nyq = sr / 2.0
    return sig.butter(order, freq / nyq, btype="high", output="sos")


def _lowpass_sos(freq: float, sr: int, order: int = 2) -> np.ndarray:
    nyq = sr / 2.0
    return sig.butter(order, freq / nyq, btype="low", output="sos")


def _low_shelf_sos(freq: float, gain_db: float, sr: int) -> np.ndarray:
    """Low shelf using bilinear-transform peaking shelf design."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    cos_w0 = np.cos(w0)
    S = 1.0
    alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)

    b0 =     A * ((A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
    b1 = 2 * A * ((A-1) - (A+1)*cos_w0)
    b2 =     A * ((A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
    a0 =          (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
    a1 =    -2 * ((A-1) + (A+1)*cos_w0)
    a2 =          (A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _high_shelf_sos(freq: float, gain_db: float, sr: int) -> np.ndarray:
    """High shelf using bilinear-transform peaking shelf design."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    cos_w0 = np.cos(w0)
    S = 1.0
    alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)

    b0 =      A * ((A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
    b1 = -2 * A * ((A-1) + (A+1)*cos_w0)
    b2 =      A * ((A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
    a0 =           (A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
    a1 =      2 * ((A-1) - (A+1)*cos_w0)
    a2 =           (A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _peak_sos(freq: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    """Peaking EQ band (boost or cut at a center frequency)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    alpha = np.sin(w0) / (2.0 * max(q, 0.01))
    cos_w0 = np.cos(w0)

    b0 =  1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 =  1.0 - alpha * A
    a0 =  1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 =  1.0 - alpha / A

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _notch_sos(freq: float, q: float, sr: int) -> np.ndarray:
    """Notch (band-reject) filter at a specific frequency."""
    w0 = 2.0 * np.pi * freq / sr
    alpha = np.sin(w0) / (2.0 * max(q, 0.01))
    cos_w0 = np.cos(w0)

    b0 =  1.0
    b1 = -2.0 * cos_w0
    b2 =  1.0
    a0 =  1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 =  1.0 - alpha

    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


def _apply_sos(audio: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Apply a second-order-section filter to [C, S] audio."""
    out = np.empty_like(audio)
    for ch in range(audio.shape[0]):
        out[ch] = sig.sosfilt(sos, audio[ch])
    return out


# ── Node ─────────────────────────────────────────────────────────────────────

class S42PParametricEQ:
    """
    🎛️ S42P Parametric EQ
    8-band parametric equalizer — the cornerstone of any mastering chain.
    Route your AceStep music or vocal audio through this before compression.
    Each band can be individually enabled/disabled.
    """

    BAND_TYPES = ["peak", "notch", "low_shelf", "high_shelf", "high_pass", "low_pass", "bypass"]

    @classmethod
    def INPUT_TYPES(cls):
        # Reusable tooltip fragments
        freq_tip   = lambda lo, hi: f"Center/cutoff frequency in Hz. Range {lo}–{hi}Hz. Lower = darker, higher = brighter."
        gain_tip   = "Gain in dB. Positive = boost, negative = cut. ±18dB range. Only applies to peak/shelf bands."
        q_tip      = "Q factor (bandwidth). Low Q = wide, musical. High Q = narrow, surgical. 0.1=very wide, 10=very narrow."
        type_tip   = ("Band type. Peak=boost/cut at frequency. Notch=deep cut. "
                      "Low/High Shelf=gentle tilt. High/Low Pass=remove a range entirely. Bypass=disable band.")
        enable_tip = "Enable or disable this band entirely. Disabled bands pass audio unchanged."

        def band(label: str, freq: float, gain: float, q: float,
                 btype: str, freq_lo: int, freq_hi: int) -> dict:
            return {
                f"{label}_type":    (cls.BAND_TYPES, {
                    "default": btype,
                    "tooltip": type_tip
                }),
                f"{label}_freq":    ("FLOAT", {
                    "default": freq, "min": float(freq_lo), "max": float(freq_hi),
                    "step": 1.0, "display": "slider",
                    "tooltip": freq_tip(freq_lo, freq_hi)
                }),
                f"{label}_gain":    ("FLOAT", {
                    "default": gain, "min": -18.0, "max": 18.0,
                    "step": 0.1, "display": "slider",
                    "tooltip": gain_tip
                }),
                f"{label}_q":       ("FLOAT", {
                    "default": q, "min": 0.1, "max": 10.0,
                    "step": 0.05, "display": "slider",
                    "tooltip": q_tip
                }),
                f"{label}_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": enable_tip
                }),
            }

        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio to equalize. Accepts any ComfyUI AUDIO output (mixer, loader, TTS, etc)."
                }),
                "output_gain": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0,
                    "step": 0.1, "display": "slider",
                    "tooltip": ("Master output gain applied after all EQ bands. "
                                "Use to compensate for loudness changes caused by boosting. "
                                "Tip: if you boosted a lot, trim here to prevent clipping.")
                }),
                "oversampling": (["1x", "2x", "4x"], {
                    "default": "1x",
                    "tooltip": ("Oversample the EQ processing to reduce aliasing artifacts at high frequencies. "
                                "2x or 4x recommended for material with lots of highs (cymbals, brightness). "
                                "Higher = better quality but slower.")
                }),
            },
            "optional": {
                # Band 1 — High Pass (remove low-end rumble)
                **band("b1", freq=80.0,   gain=0.0,  q=0.707, btype="high_pass",  freq_lo=10,   freq_hi=2000),
                # Band 2 — Low Shelf
                **band("b2", freq=120.0,  gain=0.0,  q=0.707, btype="low_shelf",  freq_lo=20,   freq_hi=500),
                # Band 3 — Low-Mid Peak
                **band("b3", freq=300.0,  gain=0.0,  q=1.0,   btype="peak",       freq_lo=50,   freq_hi=2000),
                # Band 4 — Mid Peak
                **band("b4", freq=1000.0, gain=0.0,  q=1.0,   btype="peak",       freq_lo=200,  freq_hi=8000),
                # Band 5 — Upper-Mid Peak
                **band("b5", freq=3000.0, gain=0.0,  q=1.0,   btype="peak",       freq_lo=500,  freq_hi=16000),
                # Band 6 — Presence Peak
                **band("b6", freq=5000.0, gain=0.0,  q=1.0,   btype="peak",       freq_lo=1000, freq_hi=20000),
                # Band 7 — High Shelf
                **band("b7", freq=8000.0, gain=0.0,  q=0.707, btype="high_shelf", freq_lo=2000, freq_hi=20000),
                # Band 8 — Low Pass (remove harsh top-end or noise)
                **band("b8", freq=20000.0,gain=0.0,  q=0.707, btype="low_pass",   freq_lo=4000, freq_hi=22000),
            }
        }

    RETURN_TYPES  = ("AUDIO", "STRING")
    RETURN_NAMES  = ("audio", "eq_info")
    FUNCTION      = "apply_eq"
    CATEGORY      = CATEGORY

    # ── main execute ─────────────────────────────────────────────────────────

    def apply_eq(self, audio: dict, output_gain: float = 0.0,
                 oversampling: str = "1x", **kwargs) -> Tuple[dict, str]:

        if not SCIPY_AVAILABLE:
            logger.error("scipy required for EQ — returning audio unchanged")
            return (audio, json.dumps({"error": "scipy not installed"}))

        arr, sr = audio_from_comfy(audio)
        arr     = ensure_stereo(ensure_float32(arr))

        os_factor = {"1x": 1, "2x": 2, "4x": 4}.get(oversampling, 1)
        if os_factor > 1:
            arr, sr = self._oversample(arr, sr, os_factor)

        bands_applied = []
        band_labels   = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]

        for label in band_labels:
            btype   = kwargs.get(f"{label}_type",    "bypass")
            enabled = kwargs.get(f"{label}_enabled", True)
            freq    = float(kwargs.get(f"{label}_freq",  1000.0))
            gain    = float(kwargs.get(f"{label}_gain",  0.0))
            q       = float(kwargs.get(f"{label}_q",     1.0))

            if not enabled or btype == "bypass":
                continue

            # Clamp freq safely away from DC and Nyquist
            nyq  = (sr / os_factor) / 2.0
            freq = float(np.clip(freq, 10.0, nyq * 0.98))

            try:
                sos = self._build_sos(btype, freq, gain, q, sr)
                if sos is not None:
                    arr = _apply_sos(arr, sos)
                    bands_applied.append({
                        "band": label, "type": btype,
                        "freq": freq, "gain": gain, "q": q
                    })
            except Exception as e:
                logger.warning(f"Band {label} ({btype} @ {freq:.0f}Hz) failed: {e}")

        # Downsample back if oversampled
        if os_factor > 1:
            arr, sr = self._downsample(arr, sr, os_factor)

        # Output gain
        if output_gain != 0.0:
            arr = arr * (10.0 ** (output_gain / 20.0))

        # Soft clip to prevent downstream clipping
        arr = np.clip(arr, -1.0, 1.0)

        info = {
            "bands_applied": len(bands_applied),
            "band_details":  bands_applied,
            "output_gain_db": output_gain,
            "oversampling":   oversampling,
            "sample_rate":    sr,
            "output_peak_db": float(f"{peak_db(arr):.2f}"),
            "output_rms_db":  float(f"{rms_db(arr):.2f}"),
        }

        return (audio_to_comfy(arr, sr), json.dumps(info, indent=2))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_sos(self, btype: str, freq: float, gain: float,
                   q: float, sr: int) -> Optional[np.ndarray]:
        if btype == "high_pass":
            return _highpass_sos(freq, sr)
        if btype == "low_pass":
            return _lowpass_sos(freq, sr)
        if btype == "low_shelf":
            return _low_shelf_sos(freq, gain, sr)
        if btype == "high_shelf":
            return _high_shelf_sos(freq, gain, sr)
        if btype == "peak":
            if abs(gain) < 0.01:
                return None   # flat = skip
            return _peak_sos(freq, gain, q, sr)
        if btype == "notch":
            return _notch_sos(freq, q, sr)
        return None

    def _oversample(self, arr: np.ndarray, sr: int, factor: int) -> Tuple[np.ndarray, int]:
        try:
            import scipy.signal as s
            up_arr = s.resample_poly(arr, factor, 1, axis=1)
            return up_arr.astype(np.float32), sr * factor
        except Exception as e:
            logger.warning(f"Oversampling failed: {e}")
            return arr, sr

    def _downsample(self, arr: np.ndarray, sr: int, factor: int) -> Tuple[np.ndarray, int]:
        try:
            import scipy.signal as s
            down_arr = s.resample_poly(arr, 1, factor, axis=1)
            return down_arr.astype(np.float32), sr // factor
        except Exception as e:
            logger.warning(f"Downsampling failed: {e}")
            return arr, sr


# ── ComfyUI registration ──────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "S42PParametricEQ": S42PParametricEQ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PParametricEQ": "🎛️ S42P Parametric EQ",
}
