"""
S42 Production Suite — EQ Preset Loader v2.0
=============================================
HOW TO CONNECT:
  preset_json (STRING) → S42P Parametric EQ → optional input "preset_json"
  The Parametric EQ node reads this JSON and auto-applies all band settings.

  If you only want to inspect values, wire individual dB outputs to ShowText.

  ┌─────────────────────┐     ┌───────────────────────────────────────┐
  │ S42P EQ Preset      │     │ S42P Parametric EQ                    │
  │ Loader              │     │                                       │
  │  preset_json ───────┼─────┼→ preset_json (optional)               │
  │  (individual dBs)  │     │  (auto-applies all 8 bands from JSON) │
  └─────────────────────┘     └───────────────────────────────────────┘

PRESET LIBRARY:
  flat              – Neutral pass-through
  vocal_clarity     – Enhance vocal presence and intelligibility
  warm_bass         – Add warmth and fullness to bass-heavy content
  bright_crisp      – Add sparkle and air for a modern sound
  radio_voice       – Classic broadcast vocal tone
  telephone         – Lo-fi telephone/vintage band-limited sound
  de_muddy          – Remove muddiness and boxiness
  mastering_gentle  – Subtle mastering-style spectral shaping
  bass_cut_vocal    – Remove low-end for cleaner vocal mix
  smiley_curve      – Boost lows & highs, scoop mids (modern pop)
  podcast_voice     – Optimised for spoken word clarity
  lo_fi             – Warm, reduced-bandwidth vintage sound
  sub_bass_focus    – Emphasise deep sub-bass frequencies
  de_ess            – Reduce harsh sibilant frequencies
  air_and_space     – Add openness and spatial quality
  music_safe        – Safe AceStep starting point (no HP damage)
  aceStep_balance   – Corrects typical AceStep spectral imbalance gently

Python 3.12 | ComfyUI Portable
"""

import json
import logging

logger = logging.getLogger(__name__)

CATEGORY = "S42 Production Suite 🎛️ Audio Mastering"


# ── Preset library ────────────────────────────────────────────────────────
# Band layout mirrors S42P Parametric EQ:
#   sub_bass  20-80 Hz   (b1 – high-pass / low-shelf region)
#   bass      80-300 Hz  (b2 – low shelf)
#   lo_mid    300 Hz-1k  (b3 – peak)
#   mid       1-3 kHz    (b4 – peak)
#   hi_mid    3-8 kHz    (b5 – peak)
#   presence  5-10 kHz   (b6 – peak / presence)
#   brilliance 8-20 kHz  (b7 – high shelf)
#   hp_hz     high-pass cutoff (0 = disabled)
#   lp_hz     low-pass  cutoff (0 = disabled / full-range)

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
            "No sub-bass cut. Gentle high shelf adds air without harshness."
        ),
        "sub_bass": 0.0, "bass": 0.0, "lo_mid": -0.5, "mid": 0.5,
        "hi_mid": 1.0, "presence": 1.5, "brilliance": 1.5,
        "hp_hz": 0.0, "lp_hz": 0.0,
    },
    "aceStep_balance": {
        "name": "AceStep Spectral Balance",
        "description": (
            "Corrects typical AceStep sub-dominance (58-77% sub energy). "
            "Gentle sub cut + mild bass/mid lift. "
            "Only apply if S42P Audio Analyser reports sub >55%."
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


# ── ComfyUI Node ──────────────────────────────────────────────────────────

class S42PEQPresetLoader:
    """
    S42P EQ Preset Loader — select a named EQ preset and output values
    ready to wire into S42P Parametric EQ.

    CONNECTION GUIDE:
      ─ Quickest way:
          preset_json (STRING) → S42P Parametric EQ → optional "preset_json" input
          This applies all band values automatically.

      ─ Manual band control:
          Wire individual sub_bass_db / bass_db / … outputs
          to ShowText nodes to inspect values, then type them
          into the EQ node's band controls manually.

      ─ Audio passthrough:
          Connect audio_in to get a passthrough AUDIO output.
          Useful to keep the signal chain clean without breaking
          the audio wire between the preset loader and EQ.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (PRESET_KEYS, {
                    "default": "music_safe",
                    "tooltip": (
                        "'music_safe' and 'aceStep_balance' are designed for AceStep 1.5. "
                        "Connect preset_json → S42P Parametric EQ optional preset_json input."
                    )
                }),
                "gain_trim_db": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5,
                    "tooltip": (
                        "Global gain trim applied uniformly to all non-zero bands. "
                        "Scales preset intensity up or down."
                    )
                }),
            },
            "optional": {
                "audio_in": ("AUDIO", {
                    "tooltip": (
                        "Optional audio passthrough. Connect here to route audio "
                        "alongside the preset_json without breaking the signal chain."
                    )
                }),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING",
                     "FLOAT",  "FLOAT",  "FLOAT",  "FLOAT",
                     "FLOAT",  "FLOAT",  "FLOAT",
                     "AUDIO")
    RETURN_NAMES  = ("preset_json", "preset_name",
                     "sub_bass_db", "bass_db", "lo_mid_db", "mid_db",
                     "hi_mid_db",   "presence_db", "brilliance_db",
                     "audio_out")
    FUNCTION      = "load_preset"
    CATEGORY      = CATEGORY
    OUTPUT_NODE   = True

    def load_preset(self, preset: str, gain_trim_db: float,
                    audio_in: dict = None) -> tuple:
        p = EQ_PRESETS.get(preset, EQ_PRESETS["flat"])

        def _trim(v: float) -> float:
            if abs(gain_trim_db) < 0.01 or abs(v) < 0.01:
                return v
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

        # Pass audio through unchanged (or return None if not connected)
        audio_out = audio_in

        return (preset_json, p["name"],
                sub_bass, bass, lo_mid, mid, hi_mid, presence, brilliance,
                audio_out)

    @classmethod
    def IS_CHANGED(cls, preset: str, gain_trim_db: float,
                   audio_in=None) -> str:
        return f"{preset}_{gain_trim_db}"


NODE_CLASS_MAPPINGS = {
    "S42PEQPresetLoader": S42PEQPresetLoader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PEQPresetLoader": "S42P EQ Preset Loader 🎚️"
}
