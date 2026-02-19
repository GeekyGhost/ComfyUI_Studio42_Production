"""
S42 Production Suite — Audio Splitter Node v1.0
================================================
Splits one AUDIO input into separate frequency band AUDIO outputs.
Each output is a full ComfyUI AUDIO dict — pipe to different FX chains,
Mastering nodes, or Audio Reactive FX nodes independently.

Bands:
  sub_bass  — 20–80 Hz   (kick drum body, sub synth)
  bass      — 80–250 Hz  (bass guitar, low vocals)
  mid       — 250–4kHz   (vocals, most instruments)
  high      — 4–20kHz    (air, cymbals, hi-hats)
  full      — passthrough (all frequencies, for reference)

Use cases:
  - Drive different FX layers from different bands
    (sub → pulse FX, high → shimmer FX)
  - Apply different EQ/compression per band externally
  - Feed sub_bass into Beat Analyzer for ultra-clean kick detection

Uses Linkwitz-Riley crossover filters (same as Dynamics Processor) —
bands sum back to original with flat magnitude response.

Python 3.12 | ComfyUI Portable | scipy + numpy
"""

import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import scipy.signal as sig
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.error("scipy required for S42P Audio Splitter")

from s42p_audio_utils import (
    audio_from_comfy, audio_to_comfy, ensure_stereo, ensure_float32
)

CATEGORY = "S42 Production Suite/Audio Utility"


def _lr4_crossover(arr: np.ndarray, freq: float, sr: int):
    """
    Linkwitz-Riley 4th-order crossover at freq Hz.
    Returns (low_band, high_band) — sums to original.
    """
    nyq = sr / 2.0
    f   = float(np.clip(freq, 10.0, nyq * 0.98))
    lp  = sig.butter(2, f / nyq, btype="low",  output="sos")
    hp  = sig.butter(2, f / nyq, btype="high", output="sos")
    lr_lp = np.vstack([lp, lp])
    lr_hp = np.vstack([hp, hp])
    low_band  = np.stack([sig.sosfilt(lr_lp, arr[ch]) for ch in range(arr.shape[0])]).astype(np.float32)
    high_band = np.stack([sig.sosfilt(lr_hp, arr[ch]) for ch in range(arr.shape[0])]).astype(np.float32)
    return low_band, high_band


class S42PAudioSplitter:
    """
    🎚️ S42P Audio Splitter v1.0

    Splits one audio input into 4 frequency band outputs + full passthrough.

    Each output is a complete AUDIO signal — connect to:
      • S42P Audio Reactive FX (drive different effects from different bands)
      • S42P Parametric EQ (process bands independently)
      • S42P Beat Analyzer (sub_bass → cleaner kick detection)
      • S42P Dynamics Processor (apply different compression per band)
      • ShowText / Metering nodes

    Crossover frequencies are adjustable. Default values:
      sub/bass split:  80 Hz
      bass/mid split:  250 Hz
      mid/high split:  4000 Hz

    All bands use Linkwitz-Riley 4th-order filters — bands sum back to the
    original signal with a flat magnitude response (professional standard).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to split into frequency bands."}),
                "sub_bass_freq": ("FLOAT", {
                    "default": 80.0, "min": 20.0, "max": 200.0,
                    "step": 5.0, "display": "slider",
                    "tooltip": "Sub-bass / bass crossover frequency Hz. "
                               "Below this → sub_bass output. Default 80Hz."}),
                "bass_mid_freq": ("FLOAT", {
                    "default": 250.0, "min": 100.0, "max": 1000.0,
                    "step": 10.0, "display": "slider",
                    "tooltip": "Bass / mid crossover frequency Hz. Default 250Hz."}),
                "mid_high_freq": ("FLOAT", {
                    "default": 4000.0, "min": 500.0, "max": 12000.0,
                    "step": 100.0, "display": "slider",
                    "tooltip": "Mid / high crossover frequency Hz. Default 4000Hz."}),
            }
        }

    RETURN_TYPES  = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES  = ("sub_bass", "bass", "mid", "high", "full_passthrough")
    FUNCTION      = "split"
    CATEGORY      = CATEGORY

    def split(self, audio: dict,
              sub_bass_freq: float,
              bass_mid_freq: float,
              mid_high_freq: float) -> Tuple[dict, dict, dict, dict, dict]:

        if not SCIPY_AVAILABLE:
            logger.error("scipy required for Audio Splitter")
            return (audio, audio, audio, audio, audio)

        arr, sr = audio_from_comfy(audio)
        arr     = ensure_stereo(ensure_float32(arr))

        # Ensure crossover frequencies are ordered
        f1 = min(sub_bass_freq, bass_mid_freq * 0.9)
        f2 = min(bass_mid_freq, mid_high_freq * 0.9)
        f3 = mid_high_freq

        # Split: sub+rest at f1
        sub_bass_arr, rest1 = _lr4_crossover(arr, f1, sr)

        # Split rest at f2: bass + upper
        bass_arr, rest2 = _lr4_crossover(rest1, f2, sr)

        # Split upper at f3: mid + high
        mid_arr, high_arr = _lr4_crossover(rest2, f3, sr)

        return (
            audio_to_comfy(sub_bass_arr, sr),
            audio_to_comfy(bass_arr,     sr),
            audio_to_comfy(mid_arr,      sr),
            audio_to_comfy(high_arr,     sr),
            audio_to_comfy(arr,          sr),   # passthrough
        )


NODE_CLASS_MAPPINGS        = {"S42PAudioSplitter": S42PAudioSplitter}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PAudioSplitter": "🎚️ S42P Audio Splitter"}
