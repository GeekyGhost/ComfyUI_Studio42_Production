"""
S42 Production Suite — Dynamics Processor Node
===============================================
Multiband compressor + broadband options with full sidechain support.
3 frequency bands (Low / Mid / High) each with independent:
  - Threshold, Ratio, Attack, Release, Knee, Makeup Gain
  - Mode: Compress | Expand | Gate | Bypass

Also includes a broadband (full-band) compressor before the multiband split,
useful for glue compression on full mixes.

Python 3.12 | ComfyUI Portable | scipy + numpy
"""

import numpy as np
import logging
import json
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import scipy.signal as sig
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.error("scipy required for multiband crossover filters")

from s42p_audio_utils import (
    audio_from_comfy, audio_to_comfy, ensure_stereo, ensure_float32,
    db_to_linear, linear_to_db, rms_db, peak_db, soft_limit
)

CATEGORY = "S42 Production Suite 🔊 Dynamics"


# ── Crossover filter pair ────────────────────────────────────────────────────

def _crossover_pair(freq: float, sr: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linkwitz-Riley crossover: returns (lowpass_sos, highpass_sos).
    LR-4 (order=4) is default — flat summed response, used in pro audio.
    """
    nyq = sr / 2.0
    f   = float(np.clip(freq, 20.0, nyq * 0.98))
    # LR-N = Butterworth(N/2) cascaded twice
    butter_order = order // 2
    lp_sos = sig.butter(butter_order, f / nyq, btype="low",  output="sos")
    hp_sos = sig.butter(butter_order, f / nyq, btype="high", output="sos")
    # Cascade for LR response
    lr_lp  = np.vstack([lp_sos, lp_sos])
    lr_hp  = np.vstack([hp_sos, hp_sos])
    return lr_lp, lr_hp


def _apply_sos_stereo(arr: np.ndarray, sos: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr)
    for ch in range(arr.shape[0]):
        out[ch] = sig.sosfilt(sos, arr[ch])
    return out


# ── Core dynamics engine ─────────────────────────────────────────────────────

def _compute_gain_reduction(
    arr: np.ndarray,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    knee_db: float,
    sr: int,
    mode: str,        # "compress" | "expand" | "gate"
    sidechain: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute per-sample gain reduction envelope for one band.
    Returns gain array [samples] to multiply against audio.
    Uses RMS level detection with envelope follower.
    """
    detector = sidechain if sidechain is not None else arr
    # Mix to mono for detection
    mono = np.mean(detector, axis=0) if detector.ndim == 2 else detector

    # Peak envelope follower
    sr_f       = float(sr)
    att_coeff  = np.exp(-1.0 / (max(attack_ms,  0.01) * 0.001 * sr_f))
    rel_coeff  = np.exp(-1.0 / (max(release_ms, 0.01) * 0.001 * sr_f))

    env = np.zeros(len(mono), dtype=np.float32)
    for i in range(1, len(mono)):
        level = abs(mono[i])
        if level > env[i-1]:
            env[i] = att_coeff * env[i-1] + (1.0 - att_coeff) * level
        else:
            env[i] = rel_coeff * env[i-1]

    # Convert envelope to dB
    env_db = np.where(env > 1e-10, 20.0 * np.log10(env), -120.0).astype(np.float32)
    thresh = float(threshold_db)
    knee_h = float(knee_db) * 0.5
    gain_db = np.zeros_like(env_db)

    if mode == "compress":
        # Below knee: no action; in knee: smooth transition; above: full ratio
        for i in range(len(env_db)):
            e = env_db[i]
            if e < thresh - knee_h:
                gain_db[i] = 0.0
            elif e <= thresh + knee_h and knee_db > 0:
                # Soft knee
                delta = e - (thresh - knee_h)
                gain_db[i] = (1.0/ratio - 1.0) * (delta ** 2) / (2.0 * max(knee_db, 0.001))
            else:
                gain_db[i] = (thresh + (e - thresh) / ratio) - e

    elif mode == "expand":
        # Downward expansion — signals above threshold pass, below get attenuated
        for i in range(len(env_db)):
            e = env_db[i]
            if e >= thresh:
                gain_db[i] = 0.0
            else:
                gain_db[i] = (ratio - 1.0) * (e - thresh)

    elif mode == "gate":
        # Hard gate — signals below threshold get full attenuation (-80dB floor)
        for i in range(len(env_db)):
            e = env_db[i]
            if e >= thresh:
                gain_db[i] = 0.0
            else:
                gain_db[i] = max((ratio * (e - thresh)), -80.0)

    # Convert back to linear gain
    gain_linear = np.power(10.0, gain_db / 20.0).astype(np.float32)
    return gain_linear


def _apply_dynamics_band(
    arr: np.ndarray,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    knee_db: float,
    makeup_db: float,
    sr: int,
    mode: str,
    sidechain: Optional[np.ndarray] = None
) -> np.ndarray:
    if mode == "bypass":
        return arr
    gain = _compute_gain_reduction(
        arr, threshold_db, ratio, attack_ms, release_ms, knee_db, sr, mode, sidechain
    )
    makeup = db_to_linear(makeup_db)
    # Apply gain to all channels
    out = arr * gain[np.newaxis, :] * makeup
    return out.astype(np.float32)


# ── Node ─────────────────────────────────────────────────────────────────────

class S42PDynamicsProcessor:
    """
    🔊 S42P Dynamics Processor
    3-band compressor/expander/gate with broadband glue compressor.

    Signal flow:
    Input → [Broadband Compressor] → [Crossover Split] →
        [Low Band Dynamics] ─┐
        [Mid Band Dynamics]  ├─ [Summed] → [Limiter] → Output
        [High Band Dynamics] ┘

    Use after the Parametric EQ, before the Mastering Chain.
    """

    MODES = ["bypass", "compress", "expand", "gate"]

    @classmethod
    def INPUT_TYPES(cls):

        def band_inputs(label: str, freq_label: str, default_thresh: float,
                        default_ratio: float, default_mode: str) -> dict:
            return {
                f"{label}_mode": (cls.MODES, {
                    "default": default_mode,
                    "tooltip": ("Processing mode for this band. "
                                "Compress=reduce loud peaks. Expand=increase dynamic range. "
                                "Gate=silence signals below threshold. Bypass=pass through unchanged.")
                }),
                f"{label}_threshold": ("FLOAT", {
                    "default": default_thresh, "min": -60.0, "max": 0.0,
                    "step": 0.5, "display": "slider",
                    "tooltip": (f"Threshold in dBFS for the {freq_label} band. "
                                "Signals above this level (compress mode) or below (gate/expand) are processed. "
                                "-18dB is a good starting point for music. Lower = more compression.")
                }),
                f"{label}_ratio": ("FLOAT", {
                    "default": default_ratio, "min": 1.0, "max": 20.0,
                    "step": 0.1, "display": "slider",
                    "tooltip": ("Compression/expansion ratio. "
                                "2:1 = gentle. 4:1 = moderate. 8:1+ = heavy limiting. "
                                "20:1 = hard limiter. For expand/gate: higher pushes quiet signals down faster.")
                }),
                f"{label}_attack": ("FLOAT", {
                    "default": 10.0, "min": 0.1, "max": 200.0,
                    "step": 0.5, "display": "slider",
                    "tooltip": ("Attack time in milliseconds — how fast the compressor clamps down on a loud signal. "
                                "Fast (1-5ms) catches transients tightly. "
                                "Slow (30-100ms) lets transients through for punch and snap.")
                }),
                f"{label}_release": ("FLOAT", {
                    "default": 80.0, "min": 5.0, "max": 2000.0,
                    "step": 5.0, "display": "slider",
                    "tooltip": ("Release time in milliseconds — how fast the compressor lets go after a loud signal. "
                                "Too fast causes pumping. Too slow sounds squashed. "
                                "80–300ms is typical for music.")
                }),
                f"{label}_knee": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 12.0,
                    "step": 0.5, "display": "slider",
                    "tooltip": ("Soft knee width in dB. "
                                "0 = hard knee (abrupt, more audible). "
                                "3–6 = soft knee (gradual, transparent). "
                                "Higher values sound more musical for most material.")
                }),
                f"{label}_makeup": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 24.0,
                    "step": 0.5, "display": "slider",
                    "tooltip": ("Makeup gain in dB applied after compression. "
                                "Compression reduces level — use makeup to restore it. "
                                "Tip: match output level to input level for fair A/B comparison.")
                }),
            }

        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio. Connect from the Parametric EQ output or any AUDIO source."
                }),
                # Crossover frequencies
                "low_mid_crossover": ("FLOAT", {
                    "default": 250.0, "min": 80.0, "max": 1000.0,
                    "step": 10.0, "display": "slider",
                    "tooltip": ("Crossover frequency between the Low and Mid bands in Hz. "
                                "250Hz is a classic split — lows get bass/kick, mids get guitars/vocals. "
                                "Adjust based on your material.")
                }),
                "mid_high_crossover": ("FLOAT", {
                    "default": 4000.0, "min": 1000.0, "max": 16000.0,
                    "step": 100.0, "display": "slider",
                    "tooltip": ("Crossover frequency between the Mid and High bands in Hz. "
                                "4kHz is a classic split — highs get air/cymbals, mids get vocals/guitars. "
                                "Lower this to treat more of the high frequencies.")
                }),
                # Broadband glue compressor
                "glue_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": ("Enable the broadband 'glue' compressor that runs before the multiband split. "
                                "Use this to gently compress the full mix before multiband processing. "
                                "Typical setting: 2:1 ratio, -20dB threshold, slow attack (30ms), 200ms release.")
                }),
                "glue_threshold": ("FLOAT", {
                    "default": -20.0, "min": -40.0, "max": 0.0,
                    "step": 0.5, "display": "slider",
                    "tooltip": ("Threshold for the broadband glue compressor. "
                                "Only active when glue is enabled. -20dB is a gentle starting point.")
                }),
                "glue_ratio": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 8.0,
                    "step": 0.1, "display": "slider",
                    "tooltip": ("Ratio for the glue compressor. 1.5–2:1 is gentle glue. "
                                "3–4:1 starts sounding noticeably compressed. Keep it subtle here.")
                }),
                "glue_attack": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 200.0,
                    "step": 1.0, "display": "slider",
                    "tooltip": "Attack for the glue compressor in ms. Slow attacks preserve transient punch."
                }),
                "glue_release": ("FLOAT", {
                    "default": 200.0, "min": 10.0, "max": 2000.0,
                    "step": 10.0, "display": "slider",
                    "tooltip": "Release for the glue compressor in ms. 150–300ms is typical for mix glue."
                }),
                # Low band
                **band_inputs("low",  "Low",  default_thresh=-24.0, default_ratio=3.0, default_mode="compress"),
                # Mid band
                **band_inputs("mid",  "Mid",  default_thresh=-20.0, default_ratio=2.5, default_mode="compress"),
                # High band
                **band_inputs("high", "High", default_thresh=-18.0, default_ratio=2.0, default_mode="compress"),
                # Output
                "output_limiter_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": ("Enable a final soft limiter at the output to prevent clipping after "
                                "the bands are summed. Strongly recommended. "
                                "Uses tanh soft saturation at -0.3dBFS ceiling.")
                }),
                "output_gain": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0,
                    "step": 0.5, "display": "slider",
                    "tooltip": ("Final output gain in dB applied after all processing. "
                                "Use to make up lost level or trim peaks. Applied before the output limiter.")
                }),
            },
            "optional": {
                "sidechain_audio": ("AUDIO", {
                    "tooltip": ("Optional sidechain input for all three bands. "
                                "When connected, the dynamics respond to this signal's level "
                                "rather than the main audio. Classic use: duck music under voiceover.")
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO", "STRING")
    RETURN_NAMES  = ("audio", "dynamics_info")
    FUNCTION      = "process_dynamics"
    CATEGORY      = CATEGORY

    def process_dynamics(self, audio: dict, low_mid_crossover: float,
                         mid_high_crossover: float, glue_enabled: bool,
                         glue_threshold: float, glue_ratio: float,
                         glue_attack: float, glue_release: float,
                         low_mode: str, low_threshold: float, low_ratio: float,
                         low_attack: float, low_release: float,
                         low_knee: float, low_makeup: float,
                         mid_mode: str, mid_threshold: float, mid_ratio: float,
                         mid_attack: float, mid_release: float,
                         mid_knee: float, mid_makeup: float,
                         high_mode: str, high_threshold: float, high_ratio: float,
                         high_attack: float, high_release: float,
                         high_knee: float, high_makeup: float,
                         output_limiter_enabled: bool, output_gain: float,
                         sidechain_audio: Optional[dict] = None) -> Tuple[dict, str]:

        if not SCIPY_AVAILABLE:
            logger.error("scipy required for dynamics processing")
            return (audio, json.dumps({"error": "scipy not installed"}))

        arr, sr = audio_from_comfy(audio)
        arr     = ensure_stereo(ensure_float32(arr))

        # Optional sidechain
        sc_arr = None
        if sidechain_audio is not None:
            try:
                sc_raw, sc_sr = audio_from_comfy(sidechain_audio)
                sc_arr = ensure_stereo(ensure_float32(sc_raw))
                # Match length to main audio
                if sc_arr.shape[1] != arr.shape[1]:
                    if sc_arr.shape[1] > arr.shape[1]:
                        sc_arr = sc_arr[:, :arr.shape[1]]
                    else:
                        pad = arr.shape[1] - sc_arr.shape[1]
                        sc_arr = np.pad(sc_arr, ((0,0),(0,pad)))
            except Exception as e:
                logger.warning(f"Sidechain processing failed: {e}")
                sc_arr = None

        # Step 1: Broadband glue compressor
        input_peak = peak_db(arr)
        if glue_enabled:
            arr = _apply_dynamics_band(
                arr, glue_threshold, glue_ratio, glue_attack, glue_release,
                3.0, 0.0, sr, "compress", sc_arr
            )

        # Step 2: Build crossover filters
        lp_low, hp_low   = _crossover_pair(low_mid_crossover,  sr)
        lp_high, hp_high = _crossover_pair(mid_high_crossover, sr)

        # Split into 3 bands
        band_low  = _apply_sos_stereo(arr, lp_low)
        band_rest = _apply_sos_stereo(arr, hp_low)
        band_mid  = _apply_sos_stereo(band_rest, lp_high)
        band_high = _apply_sos_stereo(band_rest, hp_high)

        # Sidechain per band (approximate split of sidechain too)
        sc_low = sc_mid = sc_high = None
        if sc_arr is not None:
            sc_low  = _apply_sos_stereo(sc_arr, lp_low)
            sc_rest = _apply_sos_stereo(sc_arr, hp_low)
            sc_mid  = _apply_sos_stereo(sc_rest, lp_high)
            sc_high = _apply_sos_stereo(sc_rest, hp_high)

        # Step 3: Process each band
        band_low  = _apply_dynamics_band(band_low,  low_threshold,  low_ratio,  low_attack,  low_release,  low_knee,  low_makeup,  sr, low_mode,  sc_low)
        band_mid  = _apply_dynamics_band(band_mid,  mid_threshold,  mid_ratio,  mid_attack,  mid_release,  mid_knee,  mid_makeup,  sr, mid_mode,  sc_mid)
        band_high = _apply_dynamics_band(band_high, high_threshold, high_ratio, high_attack, high_release, high_knee, high_makeup, sr, high_mode, sc_high)

        # Step 4: Sum bands
        out = band_low + band_mid + band_high

        # Step 5: Output gain + limiter
        if output_gain != 0.0:
            out = out * db_to_linear(output_gain)

        if output_limiter_enabled:
            out = soft_limit(out, threshold_db=-0.3)

        out = np.clip(out, -1.0, 1.0).astype(np.float32)

        info = {
            "input_peak_db":    round(input_peak, 2),
            "output_peak_db":   round(peak_db(out), 2),
            "output_rms_db":    round(rms_db(out), 2),
            "glue_enabled":     glue_enabled,
            "low_mid_crossover_hz":  low_mid_crossover,
            "mid_high_crossover_hz": mid_high_crossover,
            "band_modes": {
                "low":  low_mode,
                "mid":  mid_mode,
                "high": high_mode,
            },
            "sidechain_active": sc_arr is not None,
        }

        return (audio_to_comfy(out, sr), json.dumps(info, indent=2))


# ── ComfyUI registration ──────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "S42PDynamicsProcessor": S42PDynamicsProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PDynamicsProcessor": "🔊 S42P Dynamics Processor",
}
