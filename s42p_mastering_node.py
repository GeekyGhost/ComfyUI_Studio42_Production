"""
S42 Production Suite â€” Mastering Chain Node
============================================
The final stage of audio mastering in one node.

Signal flow:
Input â†’ [Stereo Width (M/S)] â†’ [Harmonic Exciter] â†’
        [LUFS Loudness Targeting] â†’ [True Peak Limiter] â†’
        [Dither + Noise Shape] â†’ Output

LUFS presets:
  Streaming (Spotify/Apple/YouTube): -14 LUFS integrated
  TikTok / Reels:                    -14 LUFS integrated
  Broadcast / TV (EBU R128):         -23 LUFS integrated
  CD / Download:                      no LUFS target, max loudness mode
  Custom:                             user-specified value

Python 3.12 | ComfyUI Portable | scipy + numpy
pyloudnorm strongly recommended (pip install pyloudnorm)
"""

import numpy as np
import logging
import json
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars/arrays to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
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

CATEGORY = "S42 Production Suite ðŸŽ›ï¸ Audio Mastering"

# ”€”€ LUFS presets ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

LUFS_PRESETS = {
    "Streaming â€” Spotify / Apple Music (-14 LUFS)": -14.0,
    "YouTube / Podcasts (-14 LUFS)":                -14.0,
    "TikTok / Instagram Reels (-14 LUFS)":          -14.0,
    "Broadcast / TV â€” EBU R128 (-23 LUFS)":         -23.0,
    "CD / Download â€” Max Loudness (no target)":      None,   # None = skip LUFS, just limit
    "Custom (use manual_lufs_target below)":         "custom",
}

LUFS_PRESET_KEYS = list(LUFS_PRESETS.keys())


# ”€”€ Harmonic Exciter ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _harmonic_exciter(arr: np.ndarray, sr: int,
                      drive: float, blend: float,
                      high_freq_only: bool = True) -> np.ndarray:
    """
    Adds harmonic saturation (even + odd harmonics) for perceived brightness/presence.
    drive: 0.0â€“1.0 â€” saturation amount
    blend: 0.0â€“1.0 â€” wet/dry mix (0=dry, 1=fully excited)
    high_freq_only: if True, only excite content above 3kHz (avoids muddying lows)
    """
    if drive < 0.001 or blend < 0.001:
        return arr

    excite_src = arr.copy()

    # Isolate high frequencies if requested
    if high_freq_only and SCIPY_AVAILABLE:
        try:
            nyq = sr / 2.0
            hp_sos = scipy_sig.butter(2, 3000.0 / nyq, btype="high", output="sos")
            excite_src = np.stack(
                [scipy_sig.sosfilt(hp_sos, arr[ch]) for ch in range(arr.shape[0])],
                axis=0
            ).astype(np.float32)
        except Exception as e:
            logger.warning(f"Exciter HP filter failed: {e}")

    # Soft saturation via tanh with drive
    driven    = np.tanh(excite_src * (1.0 + drive * 4.0))
    harmonics = driven - excite_src   # difference = harmonic content only

    # Blend harmonics back into full signal
    return (arr + harmonics * blend).astype(np.float32)


# ”€”€ Dither + noise shaping ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _apply_dither(arr: np.ndarray, bit_depth: int = 24,
                  noise_shape: bool = True) -> np.ndarray:
    """
    TPDF dither for final output â€” critical when delivering at 16/24-bit.
    Noise shaping shifts dither noise to less audible frequencies.
    """
    if bit_depth not in (16, 24):
        return arr

    # TPDF dither amplitude = 1 LSB
    lsb = 2.0 ** (-(bit_depth - 1))
    # Two rectangular noise sources †’ triangular probability distribution
    d1  = np.random.uniform(-lsb * 0.5, lsb * 0.5, arr.shape).astype(np.float32)
    d2  = np.random.uniform(-lsb * 0.5, lsb * 0.5, arr.shape).astype(np.float32)
    dither = d1 + d2

    if noise_shape and SCIPY_AVAILABLE:
        # Simple first-order high-pass shaping: push dither noise above 10kHz
        # Coefficient from classic F-weighted noise shaper
        shaped = np.empty_like(dither)
        prev   = np.zeros(arr.shape[0], dtype=np.float32)
        coeff  = 0.85   # shapes energy toward Nyquist
        for i in range(dither.shape[1]):
            shaped[:, i] = dither[:, i] + coeff * prev
            prev          = -shaped[:, i]
        dither = shaped

    return (arr + dither).astype(np.float32)


# ”€”€ Node ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

class S42PMasteringChain:
    """
    ðŸŽšï¸ S42P Mastering Chain
    The final mastering stage. Place last in your audio chain before export.

    Complete signal flow:
    Input â†’ Stereo Width â†’ Harmonic Exciter â†’ LUFS Targeting â†’
            True Peak Limiter â†’ Dither â†’ Output

    Outputs the mastered audio plus a comprehensive metering report
    (LUFS, True Peak, Dynamic Range, gain applied).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": ("Input audio for final mastering. "
                                "Connect from Parametric EQ â†’ Dynamics Processor â†’ here. "
                                "This should be your final mix before export.")
                }),

                # ”€”€ Stereo Width ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
                "stereo_width": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.5,
                    "step": 0.05, "display": "slider",
                    "tooltip": ("Stereo width via Mid/Side processing. "
                                "1.0 = unchanged. "
                                "0.0 = full mono (useful for checking mono compatibility). "
                                "1.5â€“2.0 = wider stereo image for electronic/pop. "
                                "Above 2.0 can cause phase issues â€” use carefully. "
                                "Tip: check mono compatibility after widening by setting to 0.0.")
                }),
                "mid_gain": ("FLOAT", {
                    "default": 0.0, "min": -6.0, "max": 6.0,
                    "step": 0.25, "display": "slider",
                    "tooltip": ("Gain applied to the Mid (centre) channel in dB before recombining. "
                                "Boost for more vocal/bass presence. Cut to push focus to sides. "
                                "0dB = unchanged.")
                }),
                "side_gain": ("FLOAT", {
                    "default": 0.0, "min": -6.0, "max": 6.0,
                    "step": 0.25, "display": "slider",
                    "tooltip": ("Gain applied to the Side (stereo difference) channel in dB. "
                                "Boost to enhance ambient/reverb. Cut to tighten stereo field. "
                                "0dB = unchanged.")
                }),

                # ”€”€ Harmonic Exciter ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
                "exciter_drive": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0,
                    "step": 0.01, "display": "slider",
                    "tooltip": ("Harmonic exciter saturation drive. 0 = disabled (no effect). "
                                "0.1â€“0.3 = subtle air and presence. 0.5â€“0.8 = obvious brightness. "
                                "1.0 = heavy saturation. Use sparingly â€” small amounts go a long way. "
                                "Good for adding life to dull TTS vocals or flat AceStep output.")
                }),
                "exciter_blend": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0,
                    "step": 0.01, "display": "slider",
                    "tooltip": ("Wet/dry blend of the harmonic exciter. "
                                "0.0 = dry (no harmonics added). 1.0 = full wet. "
                                "0.2â€“0.4 is subtle and musical. "
                                "Only active when exciter_drive > 0.")
                }),
                "exciter_high_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": ("When enabled, the exciter only processes frequencies above 3kHz. "
                                "Recommended â€” prevents the exciter from muddying the low end. "
                                "Disable only if you want full-band saturation.")
                }),

                # ”€”€ LUFS Targeting ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
                "lufs_preset": (LUFS_PRESET_KEYS, {
                    "default": LUFS_PRESET_KEYS[0],
                    "tooltip": ("Target loudness preset for your delivery platform. "
                                "Streaming services (Spotify, Apple, YouTube) normalise to -14 LUFS â€” "
                                "delivering louder will be turned down, quieter will be turned up. "
                                "Broadcast (TV/radio) uses -23 LUFS (EBU R128). "
                                "CD has no target â€” uses max loudness before clipping. "
                                "Select 'Custom' to enter a manual target below.")
                }),
                "manual_lufs_target": ("FLOAT", {
                    "default": -14.0, "min": -40.0, "max": -5.0,
                    "step": 0.5, "display": "slider",
                    "tooltip": ("Manual LUFS target. Only used when lufs_preset is set to 'Custom'. "
                                "Common values: -14 LUFS (streaming), -16 LUFS (podcast), "
                                "-23 LUFS (broadcast), -6 LUFS (loud club mix). "
                                "Lower = quieter. More negative = quieter.")
                }),

                # ”€”€ True Peak Limiter ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
                "true_peak_ceiling": ("FLOAT", {
                    "default": -1.0, "min": -6.0, "max": -0.1,
                    "step": 0.1, "display": "slider",
                    "tooltip": ("True peak ceiling in dBTP (decibels True Peak). "
                                "-1.0 dBTP is the streaming standard (Spotify, Apple, YouTube). "
                                "-0.3 dBTP for less headroom sacrifice. "
                                "-3.0 dBTP for broadcast safety. "
                                "The limiter catches inter-sample peaks that can exceed 0dBFS after D/A conversion.")
                }),
                "limiter_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": ("Enable the final true peak limiter. "
                                "Strongly recommended for all delivery. "
                                "Disabling this risks clipping on playback devices.")
                }),

                # ”€”€ Dither ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
                "dither_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": ("Enable TPDF dither on output. "
                                "Required when delivering at 16-bit (CD, some streaming). "
                                "Adds tiny noise that prevents quantisation distortion at low levels. "
                                "Safe to leave on for 24-bit too â€” it's inaudible.")
                }),
                "dither_bit_depth": (["16", "24"], {
                    "default": "24",
                    "tooltip": ("Target bit depth for dither. "
                                "Use 16 for CD delivery or legacy 16-bit WAV. "
                                "Use 24 for professional delivery, streaming masters, and DAW exports.")
                }),
                "noise_shaping": ("BOOLEAN", {
                    "default": True,
                    "tooltip": ("Enable noise shaping with dither. "
                                "Shapes the dither noise spectrum toward high frequencies (above 10kHz) "
                                "where it's much less audible. Recommended. "
                                "Only active when dither is enabled.")
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO", "STRING")
    RETURN_NAMES  = ("mastered_audio", "mastering_report")
    FUNCTION      = "master_audio"
    CATEGORY      = CATEGORY

    def master_audio(self, audio: dict, stereo_width: float,
                     mid_gain: float, side_gain: float,
                     exciter_drive: float, exciter_blend: float,
                     exciter_high_only: bool, lufs_preset: str,
                     manual_lufs_target: float, true_peak_ceiling: float,
                     limiter_enabled: bool, dither_enabled: bool,
                     dither_bit_depth: str, noise_shaping: bool) -> Tuple[dict, str]:

        arr, sr = audio_from_comfy(audio)
        arr     = ensure_stereo(ensure_float32(arr))

        report = {
            "pyloudnorm_available": PYLOUDNORM_AVAILABLE,
            "input_lufs":           None,
            "input_true_peak_dbtp": None,
            "input_dynamic_range":  None,
        }

        # ”€”€ Measure input ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        try:
            report["input_lufs"]           = round(measure_lufs(arr, sr), 2)
            report["input_true_peak_dbtp"] = round(measure_true_peak_db(arr), 2)
            report["input_dynamic_range"]  = round(measure_dynamic_range(arr), 2)
        except Exception as e:
            logger.warning(f"Input measurement failed: {e}")

        # ”€”€ Step 1: Stereo Width (M/S) ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        from s42p_audio_utils import to_mid_side, from_mid_side
        mid, side = to_mid_side(arr)

        # Apply M/S gains
        mid  = mid  * db_to_linear(mid_gain)
        side = side * db_to_linear(side_gain)

        # Apply stereo width
        side = side * stereo_width

        arr = from_mid_side(mid, side).astype(np.float32)
        report["stereo_width_applied"] = stereo_width

        # ”€”€ Step 2: Harmonic Exciter ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        if exciter_drive > 0.001:
            arr = _harmonic_exciter(arr, sr, exciter_drive, exciter_blend, exciter_high_only)
            report["exciter_drive"]  = exciter_drive
            report["exciter_blend"]  = exciter_blend
        else:
            report["exciter_drive"] = 0.0

        # ”€”€ Step 3: LUFS Targeting ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        target_lufs = LUFS_PRESETS.get(lufs_preset, -14.0)
        if target_lufs == "custom":
            target_lufs = float(manual_lufs_target)

        gain_applied_db = 0.0
        if target_lufs is not None:
            try:
                gain_linear = lufs_gain(arr, sr, float(target_lufs))
                arr         = arr * gain_linear
                gain_applied_db = round(linear_to_db(gain_linear), 2)
                report["lufs_target"]       = target_lufs
                report["lufs_gain_applied_db"] = gain_applied_db
            except Exception as e:
                logger.warning(f"LUFS targeting failed: {e}")
                report["lufs_target_error"] = str(e)
        else:
            report["lufs_target"] = "CD/Max â€” no LUFS target"

        # ”€”€ Step 4: True Peak Limiter ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        if limiter_enabled:
            arr = soft_limit(arr, threshold_db=float(true_peak_ceiling))
            report["true_peak_ceiling_dbtp"] = true_peak_ceiling

        arr = np.clip(arr, -1.0, 1.0).astype(np.float32)

        # ”€”€ Step 5: Dither ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        if dither_enabled:
            bd  = int(dither_bit_depth)
            arr = _apply_dither(arr, bit_depth=bd, noise_shape=noise_shaping)
            report["dither_bit_depth"] = bd
            report["noise_shaping"]    = noise_shaping

        # ”€”€ Measure output ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        try:
            report["output_lufs"]           = round(measure_lufs(arr, sr), 2)
            report["output_true_peak_dbtp"] = round(measure_true_peak_db(arr), 2)
            report["output_dynamic_range"]  = round(measure_dynamic_range(arr), 2)
            report["output_peak_db"]        = round(peak_db(arr), 2)
            report["output_rms_db"]         = round(rms_db(arr), 2)
        except Exception as e:
            logger.warning(f"Output measurement failed: {e}")

        report["sample_rate"] = sr

        # ”€”€ Tip: mono compatibility warning ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        if stereo_width > 1.5:
            report["warning"] = (
                "Stereo width > 1.5 â€” check mono compatibility. "
                "Temporarily set stereo_width=0.0 to hear the mono sum."
            )

        return (audio_to_comfy(arr, sr), json.dumps(report, indent=2, cls=_NumpyEncoder))


# ”€”€ ComfyUI registration ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

NODE_CLASS_MAPPINGS = {
    "S42PMasteringChain": S42PMasteringChain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PMasteringChain": "ðŸŽšï¸ S42P Mastering Chain",
}
