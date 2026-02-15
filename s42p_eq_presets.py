"""
S42 Production Suite — EQ Preset Loader
=========================================
Outputs a named EQ preset as a JSON string that can feed into the
S42P Parametric EQ node's preset_json optional input, or be read by
any downstream node via ShowText.

Also provides individual dB outputs for each frequency band so presets
can drive external EQ nodes or be used in other workflows.

PRESET LIBRARY (adapted from ULM eq_presets.py):
  flat              — Neutral pass-through
  vocal_clarity     — Enhance vocal presence and intelligibility
  warm_bass         — Add warmth and fullness to bass-heavy content
  bright_crisp      — Add sparkle and air for a modern sound
  radio_voice       — Classic broadcast vocal tone
  telephone         — Lo-fi telephone/vintage band-limited sound
  de_muddy          — Remove muddiness and boxiness
  mastering_gentle  — Subtle mastering-style spectral shaping
  bass_cut_vocal    — Remove low-end for cleaner vocal mix
  smiley_curve      — Boost lows & highs, scoop mids (modern pop)
  podcast_voice     — Optimised for spoken word clarity
  lo_fi             — Warm, reduced-bandwidth vintage sound
  sub_bass_focus    — Emphasise deep sub-bass frequencies
  de_ess            — Reduce harsh sibilant frequencies
  air_and_space     — Add openness and spatial quality
  music_safe        — Safe starting point for AceStep music (no HP damage)
  aceStep_balance   — Corrects typical AceStep spectral imbalance gently

HOW TO USE WITH S42P PARAMETRIC EQ:
  Connect preset_json (STRING) output → S42P Parametric EQ optional preset_json input.
  The EQ node reads this and applies band gains automatically.
  Individual outputs can be read with ShowText for reference.

Python 3.12 | ComfyUI Portable
"""

import json
import logging

logger = logging.getLogger(__name__)

CATEGORY = "S42 Production Suite 🎛️ Audio"


# ── Preset library ────────────────────────────────────────────────────────────
# All values in dB. Band definitions match S42P Parametric EQ band layout:
#   sub_bass  = 20-80Hz    (EQ band b1 low_shelf region)
#   bass      = 80-300Hz   (EQ band b2 low_shelf)
#   lo_mid    = 300-1kHz   (EQ band b3 peak)
#   mid       = 1-3kHz     (EQ band b4 peak)
#   hi_mid    = 3-8kHz     (EQ band b5 peak)
#   presence  = 5-10kHz    (EQ band b6 peak / presence)
#   brilliance = 8-20kHz   (EQ band b7 high_shelf)
#   hp_hz     = high-pass cutoff frequency (0 = off)
#   lp_hz     = low-pass cutoff (0 = off / 20000 = full range)

EQ_PRESETS = {
    "flat": {
        "name": "Flat / Neutral",
        "description": "No EQ adjustment — transparent pass-through.",
        "sub_bass": 0.0, "bass": 0.0, "lo_mid": 0.0, "mid": 0.0,
        "hi_mid": 0.0, "presence": 0.0, "brilliance": 0.0,
        "hp_hz": 0.0, "lp_hz": 0.0,
    },
    "music_safe": {
        "name": "Music Safe (AceStep)",
        "description": (
            "Conservative starting point for AceStep music. "
            "No sub-bass cut (preserves the 58-77% sub energy typical of AceStep). "
            "Gentle high shelf to add air without harshness."
        ),
        "sub_bass": 0.0, "bass": 0.0, "lo_mid": -0.5, "mid": 0.5,
        "hi_mid": 1.0, "presence": 1.5, "brilliance": 1.5,
        "hp_hz": 0.0, "lp_hz": 0.0,
    },
    "aceStep_balance": {
        "name": "AceStep Spectral Balance",
        "description": (
            "Corrects typical AceStep sub-dominance (58-77% sub energy). "
            "Gentle sub cut + mild bass/mid lift to re-balance the spectrum. "
            "Use AFTER checking S42P Audio Analyser — only apply if sub >55%."
        ),
        "sub_bass": -2.5, "bass": 1.0, "lo_mid": 1.5, "mid": 1.0,
        "hi_mid": 1.5, "presence": 2.0, "brilliance": 1.5,
        "hp_hz": 25.0, "lp_hz": 0.0,
    },
    "vocal_clarity": {
        "name": "Vocal Clarity",
        "description": "Enhance vocal presence and intelligibility.",
        "sub_bass": -3.0, "bass": -2.0, "lo_mid": -1.5, "mid": 2.0,
        "hi_mid": 2.5, "presence": 3.0, "brilliance": 1.5,
        "hp_hz": 80.0, "lp_hz": 0.0,
    },
    "warm_bass": {
        "name": "Warm & Full Bass",
        "description": "Add warmth and fullness to bass-heavy content.",
        "sub_bass": 3.0, "bass": 2.5, "lo_mid": 1.0, "mid": -0.5,
        "hi_mid": -1.0, "presence": 0.0, "brilliance": -0.5,
        "hp_hz": 0.0, "lp_hz": 0.0,
    },
    "bright_crisp": {
        "name": "Bright & Crisp",
        "description": "Add sparkle and air for a modern, polished sound.",
        "sub_bass": -2.0, "bass": -1.0, "lo_mid": -2.0, "mid": 1.0,
        "hi_mid": 2.5, "presence": 3.5, "brilliance": 4.0,
        "hp_hz": 40.0, "lp_hz": 0.0,
    },
    "radio_voice": {
        "name": "Radio Voice",
        "description": "Classic broadcast vocal tone.",
        "sub_bass": -6.0, "bass": -4.0, "lo_mid": 1.5, "mid": 4.0,
        "hi_mid": 3.0, "presence": 2.0, "brilliance": -2.0,
        "hp_hz": 120.0, "lp_hz": 8000.0,
    },
    "telephone": {
        "name": "Telephone Effect",
        "description": "Lo-fi telephone/vintage band-limited sound.",
        "sub_bass": -12.0, "bass": -10.0, "lo_mid": 2.0, "mid": 6.0,
        "hi_mid": 4.0, "presence": 0.0, "brilliance": -12.0,
        "hp_hz": 300.0, "lp_hz": 3500.0,
    },
    "de_muddy": {
        "name": "De-Muddy / Clean",
        "description": "Remove muddiness and boxiness for clarity.",
        "sub_bass": -1.0, "bass": -2.0, "lo_mid": -4.0, "mid": -1.5,
        "hi_mid": 1.0, "presence": 2.0, "brilliance": 1.5,
        "hp_hz": 30.0, "lp_hz": 0.0,
    },
    "mastering_gentle": {
        "name": "Mastering (Gentle)",
        "description": "Subtle mastering-style spectral shaping.",
        "sub_bass": 0.5, "bass": 0.3, "lo_mid": -0.5, "mid": 0.2,
        "hi_mid": 0.5, "presence": 0.8, "brilliance": 1.0,
        "hp_hz": 20.0, "lp_hz": 0.0,
    },
    "bass_cut_vocal": {
        "name": "Bass Cut (Vocals Only)",
        "description": "Remove low-end for cleaner vocal mix.",
        "sub_bass": -12.0, "bass": -8.0, "lo_mid": -3.0, "mid": 1.5,
        "hi_mid": 2.0, "presence": 2.5, "brilliance": 1.0,
        "hp_hz": 150.0, "lp_hz": 0.0,
    },
    "smiley_curve": {
        "name": "Smiley Curve",
        "description": "Boost lows & highs, scoop mids (modern pop).",
        "sub_bass": 3.0, "bass": 2.5, "lo_mid": -1.0, "mid": -2.5,
        "hi_mid": -1.0, "presence": 2.5, "brilliance": 3.5,
        "hp_hz": 0.0, "lp_hz": 0.0,
    },
    "podcast_voice": {
        "name": "Podcast Voice",
        "description": "Optimised for spoken word clarity.",
        "sub_bass": -6.0, "bass": -3.0, "lo_mid": -1.0, "mid": 2.5,
        "hi_mid": 3.0, "presence": 2.5, "brilliance": 0.5,
        "hp_hz": 100.0, "lp_hz": 0.0,
    },
    "lo_fi": {
        "name": "Lo-Fi / Vintage",
        "description": "Warm, reduced-bandwidth vintage sound.",
        "sub_bass": -8.0, "bass": 1.0, "lo_mid": 2.0, "mid": 1.5,
        "hi_mid": -1.0, "presence": -2.0, "brilliance": -6.0,
        "hp_hz": 60.0, "lp_hz": 7000.0,
    },
    "sub_bass_focus": {
        "name": "Sub-Bass Focus",
        "description": "Emphasise deep sub-bass frequencies.",
        "sub_bass": 6.0, "bass": 3.0, "lo_mid": -2.0, "mid": -3.0,
        "hi_mid": -2.0, "presence": -1.0, "brilliance": 0.0,
        "hp_hz": 0.0, "lp_hz": 0.0,
    },
    "de_ess": {
        "name": "De-Ess (Sibilance Reduction)",
        "description": "Reduce harsh sibilant frequencies.",
        "sub_bass": 0.0, "bass": 0.0, "lo_mid": 0.0, "mid": 0.0,
        "hi_mid": -2.0, "presence": -4.0, "brilliance": -1.5,
        "hp_hz": 0.0, "lp_hz": 0.0,
    },
    "air_and_space": {
        "name": "Air & Space",
        "description": "Add openness and spatial quality.",
        "sub_bass": -1.0, "bass": -0.5, "lo_mid": -1.0, "mid": 0.0,
        "hi_mid": 1.5, "presence": 2.5, "brilliance": 4.5,
        "hp_hz": 25.0, "lp_hz": 0.0,
    },
}

PRESET_KEYS = list(EQ_PRESETS.keys())


# ── ComfyUI Node ──────────────────────────────────────────────────────────────

class S42PEQPresetLoader:
    """
    S42P EQ Preset Loader — select a named EQ preset and output its
    band gain values as individual floats and a JSON string.

    The preset_json STRING output can be wired into the S42P Parametric EQ
    optional preset_json input to auto-apply all band settings at once.
    Individual dB outputs let you inspect or route specific band values.

    Includes AceStep-specific presets that account for the typical
    sub-bass dominance (58-77%) of AceStep 1.5 generated audio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (PRESET_KEYS, {
                    "default": "music_safe",
                    "tooltip": (
                        "Select an EQ preset. "
                        "'music_safe' and 'aceStep_balance' are designed specifically "
                        "for AceStep 1.5 output. All others are genre/use-case presets."
                    )
                }),
                "gain_trim_db": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5,
                    "tooltip": (
                        "Global gain trim applied to ALL bands uniformly. "
                        "Use to scale the preset effect up or down without editing individual bands."
                    )
                }),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING", "FLOAT", "FLOAT", "FLOAT",
                     "FLOAT",  "FLOAT",  "FLOAT", "FLOAT")
    RETURN_NAMES  = ("preset_json", "preset_name",
                     "sub_bass_db", "bass_db", "lo_mid_db", "mid_db",
                     "hi_mid_db", "presence_db", "brilliance_db")
    FUNCTION      = "load_preset"
    CATEGORY      = CATEGORY
    OUTPUT_NODE   = True

    def load_preset(self, preset: str, gain_trim_db: float) -> tuple:
        p = EQ_PRESETS.get(preset, EQ_PRESETS["flat"])

        def _trim(v: float) -> float:
            """Apply gain trim — zero trim = no change."""
            if abs(gain_trim_db) < 0.01 or abs(v) < 0.01:
                return v
            # Scale non-zero bands by trim ratio (additive in dB domain)
            return v + gain_trim_db * (1.0 if v >= 0 else -1.0) * min(1.0, abs(v) / 6.0)

        sub_bass   = round(_trim(p["sub_bass"]),   2)
        bass       = round(_trim(p["bass"]),       2)
        lo_mid     = round(_trim(p["lo_mid"]),     2)
        mid        = round(_trim(p["mid"]),        2)
        hi_mid     = round(_trim(p["hi_mid"]),     2)
        presence   = round(_trim(p["presence"]),   2)
        brilliance = round(_trim(p["brilliance"]), 2)

        payload = {
            "preset_key":   preset,
            "preset_name":  p["name"],
            "description":  p["description"],
            "bands_db": {
                "sub_bass":   sub_bass,
                "bass":       bass,
                "lo_mid":     lo_mid,
                "mid":        mid,
                "hi_mid":     hi_mid,
                "presence":   presence,
                "brilliance": brilliance,
            },
            "hp_hz":        p.get("hp_hz", 0.0),
            "lp_hz":        p.get("lp_hz", 0.0),
            "gain_trim_db": gain_trim_db,
        }

        preset_json = json.dumps(payload, indent=2)

        print(f"[S42P EQ Preset] '{p['name']}' — "
              f"sub={sub_bass:+.1f} bass={bass:+.1f} lo_mid={lo_mid:+.1f} "
              f"mid={mid:+.1f} hi_mid={hi_mid:+.1f} "
              f"presence={presence:+.1f} brilliance={brilliance:+.1f} dB")

        return (preset_json, p["name"],
                sub_bass, bass, lo_mid, mid, hi_mid, presence, brilliance)

    @classmethod
    def IS_CHANGED(cls, preset: str, gain_trim_db: float) -> str:
        return f"{preset}_{gain_trim_db}"


NODE_CLASS_MAPPINGS = {
    "S42PEQPresetLoader": S42PEQPresetLoader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PEQPresetLoader": "S42P EQ Preset Loader 🎚️"
}
