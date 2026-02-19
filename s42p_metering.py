"""
S42 Production Suite — Metering Node v1.0
==========================================
Pure pass-through metering node — measures audio without modifying it.

Outputs individual FLOAT pins for LUFS, True Peak, Dynamic Range, RMS,
and Peak dB. Wire directly to ShowText nodes, downstream comparators,
or other nodes that accept FLOAT values.

Unlike the Mastering Chain's metering report (which only outputs a
JSON string), this node outputs machine-readable individual floats
suitable for downstream automation and branching.

Use at any point in your chain:
  VHS_LoadAudio → S42P Metering → S42P EQ → S42P Dynamics → S42P Mastering

No audio modification whatsoever — audio passes through unchanged.

Python 3.12 | ComfyUI Portable | scipy + numpy + pyloudnorm (optional)
"""

import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

from s42p_audio_utils import (
    audio_from_comfy, audio_to_comfy, ensure_stereo, ensure_float32,
    rms_db, peak_db, measure_lufs, measure_true_peak_db,
    measure_dynamic_range, PYLOUDNORM_AVAILABLE
)

CATEGORY = "S42 Production Suite/Audio Utility"


class S42PMetering:
    """
    📊 S42P Metering v1.0 — Audio analysis passthrough.

    Measures the audio signal without modifying it. Outputs:
      • lufs_integrated  — Integrated LUFS loudness (pyloudnorm recommended)
      • true_peak_db     — True peak in dBTP
      • dynamic_range_db — Crest factor / dynamic range in dB
      • rms_db           — RMS level in dBFS
      • peak_db          — Sample peak in dBFS
      • audio            — Unmodified audio passthrough
      • report           — Full JSON summary string

    Place anywhere in your chain to spot-check levels without
    adding or removing any other nodes.

    Typical use:
      • Check input levels before EQ/Compression
      • Verify LUFS after Mastering Chain
      • Monitor dynamic range to avoid over-compression
      • Feed LUFS into ShowText for visual display in ComfyUI
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to measure. Passes through unchanged."}),
                "label": ("STRING", {
                    "default": "Pre-Master",
                    "tooltip": "Optional label for the report (shown in JSON output)."}),
            }
        }

    RETURN_TYPES  = ("AUDIO", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES  = ("audio", "lufs_integrated", "true_peak_db",
                     "dynamic_range_db", "rms_db", "peak_db", "report")
    FUNCTION      = "meter"
    CATEGORY      = CATEGORY

    def meter(self, audio: dict, label: str = "Metering"):

        arr, sr = audio_from_comfy(audio)
        arr     = ensure_stereo(ensure_float32(arr))

        # ── Measure ───────────────────────────────────────────────────────────
        lufs = measure_lufs(arr, sr)
        lufs_val = round(lufs, 2) if lufs is not None else -99.0

        tp     = measure_true_peak_db(arr, sr)
        tp_val = round(tp, 2) if tp is not None else round(peak_db(arr), 2)

        dr_val  = round(measure_dynamic_range(arr), 2)
        rms_val = round(rms_db(arr), 2)
        pk_val  = round(peak_db(arr), 2)

        # ── Report ─────────────────────────────────────────────────────────────
        report = {
            "label":             label,
            "lufs_integrated":   lufs_val,
            "lufs_method":       "pyloudnorm" if PYLOUDNORM_AVAILABLE else "approx_rms",
            "true_peak_dbtp":    tp_val,
            "dynamic_range_db":  dr_val,
            "rms_dbfs":          rms_val,
            "peak_dbfs":         pk_val,
            "sample_rate":       sr,
            "duration_sec":      round(arr.shape[1] / sr, 3),
            "channels":          arr.shape[0],
            "warnings":          [],
        }

        # Helpful warnings
        if lufs_val > -8.0:
            report["warnings"].append("VERY LOUD — likely over-compressed or clipped.")
        if lufs_val < -30.0 and lufs_val > -90.0:
            report["warnings"].append("Very quiet — may need gain staging before mastering.")
        if tp_val > -0.1:
            report["warnings"].append(f"TRUE PEAK EXCEEDS -0.1 dBTP ({tp_val} dBTP) — will clip on some platforms.")
        if dr_val < 6.0:
            report["warnings"].append(f"Low dynamic range ({dr_val} dB) — may sound over-compressed.")

        report_str = json.dumps(report, indent=2)

        # Pass audio through completely unchanged
        return (
            audio_to_comfy(arr, sr),
            float(lufs_val),
            float(tp_val),
            float(dr_val),
            float(rms_val),
            float(pk_val),
            report_str,
        )


NODE_CLASS_MAPPINGS        = {"S42PMetering": S42PMetering}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PMetering": "📊 S42P Metering"}
