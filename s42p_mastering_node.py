"""
S42 Production Suite — Mastering Chain Node v5.0
=================================================
Final mastering stage in one node.

Signal flow:
Input → [bypass?] → [Stereo Width (M/S)] → [Harmonic Exciter] →
        [LUFS Loudness Targeting] → [True Peak Limiter] →
        [Dither + Noise Shape] → Output

CHANGES v5.0:
  - NEW:  bypass boolean — instant A/B without rewiring
  - IMPROVED: Asymmetric tape-style waveshaper in exciter
              (different curves for +/- cycles = more musical saturation)
  - Asymmetric shaper: positive=tanh(soft), negative=atan(firmer) → classic tape character

Python 3.12 | ComfyUI Portable | scipy + numpy
pyloudnorm strongly recommended (pip install pyloudnorm)
"""

import numpy as np
import logging
import json
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, np.bool_):     return bool(obj)
        return super().default(obj)


try:
    import scipy.signal as scipy_sig
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from s42p_audio_utils import (
    audio_from_comfy, audio_to_comfy, ensure_stereo, ensure_float32,
    db_to_linear, linear_to_db, rms_db, peak_db,
    measure_lufs, measure_true_peak_db, lufs_gain,
    stereo_width as ms_stereo_width, soft_limit,
    measure_dynamic_range, PYLOUDNORM_AVAILABLE
)

CATEGORY = "S42 Production Suite/Audio Mastering"

LUFS_PRESETS = {
    "Streaming — Spotify / Apple Music (-14 LUFS)": -14.0,
    "YouTube / Podcasts (-14 LUFS)":                -14.0,
    "TikTok / Instagram Reels (-14 LUFS)":          -14.0,
    "Broadcast / TV — EBU R128 (-23 LUFS)":         -23.0,
    "CD / Download — Max Loudness (no target)":      None,
    "Custom (use manual_lufs_target below)":         "custom",
}
LUFS_PRESET_KEYS = list(LUFS_PRESETS.keys())


# ── Asymmetric tape-style harmonic exciter ─────────────────────────────────────

def _harmonic_exciter(arr: np.ndarray, sr: int,
                      drive: float, blend: float,
                      high_freq_only: bool = True) -> np.ndarray:
    """
    Asymmetric tape-style harmonic exciter.

    Classic symmetric tanh saturation generates only odd harmonics (3rd, 5th…)
    which sounds harsh and transistor-like. Tape saturation is asymmetric:
      • Positive cycles → soft tanh curve (gentle even harmonics)
      • Negative cycles → firmer atan curve (adds 2nd harmonic character)

    This asymmetry produces the warm, musical 2nd-harmonic richness that
    makes tape and tube gear sound good. The difference is clearly audible
    on piano, vocals, and AceStep output.

    drive: 0.0–1.0 — saturation amount
    blend: 0.0–1.0 — wet/dry mix
    high_freq_only: restrict to >3kHz to avoid muddying the low end
    """
    if drive < 0.001 or blend < 0.001:
        return arr

    excite_src = arr.copy()

    if high_freq_only and SCIPY_AVAILABLE:
        try:
            nyq    = sr / 2.0
            hp_sos = scipy_sig.butter(2, 3000.0 / nyq, btype="high", output="sos")
            excite_src = np.stack(
                [scipy_sig.sosfilt(hp_sos, arr[ch]) for ch in range(arr.shape[0])],
                axis=0
            ).astype(np.float32)
        except Exception as e:
            logger.warning(f"Exciter HP filter failed: {e}")

    gain = 1.0 + drive * 4.0

    # Asymmetric waveshaper
    driven_pos = np.tanh(excite_src * gain)                          # soft positive
    driven_neg = (2.0 / np.pi) * np.arctan(excite_src * gain * 1.3) # firmer negative

    # Blend: positive half uses tanh, negative half uses atan
    driven = np.where(excite_src >= 0, driven_pos, driven_neg)

    # DC offset compensation (asymmetric shaper introduces slight DC)
    dc = driven.mean(axis=1, keepdims=True)
    driven = driven - dc

    harmonics = driven - excite_src   # difference = added harmonic content only

    return (arr + harmonics * blend).astype(np.float32)


# ── Dither + noise shaping ─────────────────────────────────────────────────────

def _apply_dither(arr: np.ndarray, bit_depth: int = 24,
                  noise_shape: bool = True) -> np.ndarray:
    """TPDF dither for final output."""
    if bit_depth not in (16, 24):
        return arr
    lsb    = 2.0 ** (-(bit_depth - 1))
    d1     = np.random.uniform(-lsb * 0.5, lsb * 0.5, arr.shape).astype(np.float32)
    d2     = np.random.uniform(-lsb * 0.5, lsb * 0.5, arr.shape).astype(np.float32)
    dither = d1 + d2

    if noise_shape and SCIPY_AVAILABLE:
        shaped = np.empty_like(dither)
        prev   = np.zeros(arr.shape[0], dtype=np.float32)
        coeff  = 0.85
        for i in range(dither.shape[1]):
            shaped[:, i] = dither[:, i] + coeff * prev
            prev          = -shaped[:, i]
        dither = shaped

    return (arr + dither).astype(np.float32)


# ── Node ───────────────────────────────────────────────────────────────────────

class S42PMasteringChain:
    """
    🎚️ S42P Mastering Chain v5.0
    Final mastering stage. Place last in your audio chain.

    Signal flow:
    Input → [bypass?] → Stereo Width → Harmonic Exciter (tape-style) →
            LUFS Targeting → True Peak Limiter → Dither → Output

    NEW v5.0:
      - bypass toggle for A/B comparison
      - Asymmetric tape waveshaper — adds musical 2nd harmonic warmth
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio for final mastering. "
                               "Connect from EQ → Dynamics Processor → here."}),
                # Global bypass
                "bypass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Bypass all mastering — pass audio through unchanged. "
                               "Use to instantly A/B the mastered vs unmastered signal."}),
                # Stereo Width
                "stereo_width": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.5, "step": 0.05,
                    "display": "slider",
                    "tooltip": "Stereo width via M/S. 1.0=unchanged. 0.0=mono. "
                               "1.5-2.0=wider. Above 2.0 may cause phase issues."}),
                "mid_gain":  ("FLOAT", {"default": 0.0, "min": -6.0, "max": 6.0, "step": 0.25,
                                         "display": "slider",
                                         "tooltip": "Mid channel gain dB. Boost for more centre presence."}),
                "side_gain": ("FLOAT", {"default": 0.0, "min": -6.0, "max": 6.0, "step": 0.25,
                                         "display": "slider",
                                         "tooltip": "Side channel gain dB. Boost for wider ambience."}),
                # Harmonic Exciter
                "exciter_drive": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Tape-style harmonic exciter drive. 0=off. 0.1-0.3=subtle air. "
                               "Uses asymmetric waveshaper: positive→tanh, negative→atan "
                               "for musical 2nd harmonic warmth (vs harsh symmetric tanh)."}),
                "exciter_blend": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Wet/dry blend of exciter. 0.2-0.4 is subtle and musical."}),
                "exciter_high_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Restrict exciter to >3kHz. Prevents muddying the low end. "
                               "Recommended for all music."}),
                # LUFS
                "lufs_preset": (LUFS_PRESET_KEYS, {
                    "default": LUFS_PRESET_KEYS[0],
                    "tooltip": "Target loudness for delivery platform."}),
                "manual_lufs_target": ("FLOAT", {
                    "default": -14.0, "min": -40.0, "max": -5.0, "step": 0.5,
                    "display": "slider",
                    "tooltip": "Manual LUFS target. Only used when preset = Custom."}),
                # Limiter
                "true_peak_ceiling": ("FLOAT", {
                    "default": -0.3, "min": -6.0, "max": 0.0, "step": 0.1,
                    "display": "slider",
                    "tooltip": "True peak ceiling dBTP. -0.3 is standard for streaming."}),
                # Dither
                "dither_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply TPDF dither at output. Enable for final renders only."}),
                "dither_bit_depth": (["16", "24"], {
                    "default": "24",
                    "tooltip": "Dither target bit depth."}),
            }
        }

    RETURN_TYPES  = ("AUDIO", "STRING")
    RETURN_NAMES  = ("audio", "mastering_report")
    FUNCTION      = "master"
    CATEGORY      = CATEGORY

    def master(
        self, audio: dict,
        bypass: bool,
        stereo_width: float, mid_gain: float, side_gain: float,
        exciter_drive: float, exciter_blend: float, exciter_high_only: bool,
        lufs_preset: str, manual_lufs_target: float,
        true_peak_ceiling: float,
        dither_enabled: bool, dither_bit_depth: str,
    ) -> Tuple[dict, str]:

        if bypass:
            return (audio, json.dumps({"bypass": True, "note": "All mastering bypassed."}, indent=2))

        arr, sr = audio_from_comfy(audio)
        arr     = ensure_stereo(ensure_float32(arr))

        report: dict = {
            "bypass": False,
            "input_peak_db":   round(peak_db(arr), 2),
            "input_rms_db":    round(rms_db(arr), 2),
        }

        # Measure input LUFS
        in_lufs = measure_lufs(arr, sr)
        report["input_lufs"] = round(in_lufs, 2) if in_lufs is not None else None

        # 1. Stereo width
        if stereo_width != 1.0 or mid_gain != 0.0 or side_gain != 0.0:
            arr = ms_stereo_width(arr, stereo_width,
                                   mid_gain_db=mid_gain, side_gain_db=side_gain)

        # 2. Harmonic exciter (tape-style asymmetric waveshaper)
        if exciter_drive > 0.001:
            arr = _harmonic_exciter(arr, sr, exciter_drive, exciter_blend, exciter_high_only)
            report["exciter_applied"] = True
            report["exciter_mode"] = "asymmetric_tape"

        # 3. LUFS targeting
        target_lufs = None
        preset_val  = LUFS_PRESETS.get(lufs_preset)
        if preset_val == "custom":
            target_lufs = float(manual_lufs_target)
        elif preset_val is not None:
            target_lufs = float(preset_val)

        gain_applied = 0.0
        if target_lufs is not None:
            g = lufs_gain(arr, sr, target_lufs)
            if g is not None:
                arr          = arr * g
                gain_applied = linear_to_db(g)
        report["lufs_target"]    = target_lufs
        report["lufs_gain_db"]   = round(gain_applied, 2)

        # 4. True peak limiter
        arr = soft_limit(arr, threshold_db=true_peak_ceiling)
        tp  = measure_true_peak_db(arr, sr)
        report["true_peak_db"] = round(tp, 2) if tp is not None else round(peak_db(arr), 2)

        # 5. Dither
        if dither_enabled:
            arr = _apply_dither(arr, int(dither_bit_depth))

        arr = np.clip(arr, -1.0, 1.0).astype(np.float32)

        # Output metrics
        out_lufs = measure_lufs(arr, sr)
        report["output_lufs"]     = round(out_lufs, 2) if out_lufs is not None else None
        report["output_peak_db"]  = round(peak_db(arr), 2)
        report["output_rms_db"]   = round(rms_db(arr), 2)
        report["dynamic_range_db"] = round(measure_dynamic_range(arr), 2)
        report["dither_applied"]  = dither_enabled

        return (audio_to_comfy(arr, sr),
                json.dumps(report, indent=2, cls=_NumpyEncoder))


NODE_CLASS_MAPPINGS        = {"S42PMasteringChain": S42PMasteringChain}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PMasteringChain": "🎚️ S42P Mastering Chain"}
