"""
S42 Production Suite for ComfyUI
=================================
Professional audio mastering, 2D animation, video editing, and color grading.

Node Menu Category: S42 Production Suite

AUDIO MASTERING CHAIN:
  S42P Parametric EQ, Dynamics Processor, Mastering Chain, Beat Analyzer

ANIMATION & VIDEO EDITING:
  S42P Keyframe Animator, Transition, Video Tools, Color Grade

Works alongside Studio42 (BG Remover, Layer Composer) and GeekyGhost (TTS, LatentSync).

Version: 2.0.0 | Python 3.12 | ComfyUI Portable
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print("\n" + "="*60)
print("  S42 Production Suite v2.0.0 — Loading...")
print("="*60)

# ── Audio nodes ───────────────────────────────────────────────────────────────
_audio_nodes = [
    ("audio.s42p_eq_node",        "S42P Parametric EQ",      "pip install scipy"),
    ("audio.s42p_dynamics_node",  "S42P Dynamics Processor", "pip install scipy"),
    ("audio.s42p_mastering_node", "S42P Mastering Chain",    "pip install pyloudnorm"),
    ("audio.s42p_beat_analyzer",  "S42P Beat Analyzer",      "pip install librosa"),
]

for mod_path, label, fix in _audio_nodes:
    try:
        import importlib
        mod = importlib.import_module(f".{mod_path}", package=__name__)
        NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
        print(f"  OK {label}")
    except Exception as e:
        print(f"  !! {label} FAILED: {e}")
        print(f"       -> {fix}")

# ── Video / animation nodes ───────────────────────────────────────────────────
_video_nodes = [
    ("video.s42p_keyframe_animator", "S42P Keyframe Animator", "Requires Pillow (included with ComfyUI)"),
    ("video.s42p_transition",        "S42P Transition",        "Requires Pillow (included with ComfyUI)"),
    ("video.s42p_video_tools",       "S42P Video Tools",       "Requires torch (included with ComfyUI)"),
    ("video.s42p_color_grade",       "S42P Color Grade",       "Requires numpy (included with ComfyUI)"),
]

for mod_path, label, fix in _video_nodes:
    try:
        import importlib
        mod = importlib.import_module(f".{mod_path}", package=__name__)
        NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
        print(f"  OK {label}")
    except Exception as e:
        print(f"  !! {label} FAILED: {e}")
        print(f"       -> {fix}")

# ── Summary ───────────────────────────────────────────────────────────────────
total    = len(NODE_CLASS_MAPPINGS)
expected = 8
print("-"*60)
print(f"  {'All nodes loaded' if total == expected else str(expected-total)+' node(s) failed'}"
      f" — {total}/{expected} active")
print(f"  Find nodes under 'S42 Production Suite' in the node menu")
print("="*60 + "\n")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
