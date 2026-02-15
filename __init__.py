"""
S42 Production Suite for ComfyUI
=================================
Professional audio mastering, 2D animation, video editing, and color grading.

Repo structure: all node files are in the root directory (flat layout).
Uses spec_from_file_location() — no sys.path manipulation, no relative imports.
Safe for concurrent loading alongside other custom nodes.

Version: 2.0.3 | Python 3.12 | ComfyUI Portable
"""

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print("\n" + "="*60)
print("  S42 Production Suite v2.0.3 — Loading...")
print("="*60)


def _load(label: str, filename: str, fix_hint: str) -> bool:
    """Load a node file from the repo root by absolute path."""
    filepath = os.path.join(_HERE, filename)
    if not os.path.isfile(filepath):
        print(f"  !! {label} FAILED: file not found — {filepath}")
        return False
    try:
        mod_name = "s42prod_" + os.path.splitext(filename)[0]
        spec     = importlib.util.spec_from_file_location(mod_name, filepath)
        mod      = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        print(f"  OK {label}")
        return True
    except Exception as e:
        print(f"  !! {label} FAILED: {e}")
        print(f"       -> {fix_hint}")
        return False


def _preload(mod_name: str, filename: str) -> None:
    """Pre-load a shared utility into sys.modules so node files can import it."""
    filepath = os.path.join(_HERE, filename)
    if not os.path.isfile(filepath) or mod_name in sys.modules:
        return
    try:
        spec = importlib.util.spec_from_file_location(mod_name, filepath)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  !! Pre-load failed for {mod_name}: {e}")


# ── Pre-load shared utilities ─────────────────────────────────────────────────
_preload("s42p_audio_utils", "s42p_audio_utils.py")
_preload("s42p_video_utils", "s42p_video_utils.py")

# ── Audio nodes ───────────────────────────────────────────────────────────────
_load("S42P Parametric EQ",      "s42p_eq_node.py",       "pip install scipy")
_load("S42P Dynamics Processor", "s42p_dynamics_node.py", "pip install scipy")
_load("S42P Mastering Chain",    "s42p_mastering_node.py","pip install pyloudnorm")
_load("S42P Beat Analyzer",      "s42p_beat_analyzer.py", "pip install librosa")

# ── Video / animation nodes ───────────────────────────────────────────────────
_load("S42P Keyframe Animator",  "s42p_keyframe_animator.py", "Requires Pillow (included with ComfyUI)")
_load("S42P Transition",         "s42p_transition.py",        "Requires Pillow (included with ComfyUI)")
_load("S42P Video Tools",        "s42p_video_tools.py",       "Requires torch (included with ComfyUI)")
_load("S42P Color Grade",        "s42p_color_grade.py",       "Requires numpy (included with ComfyUI)")

# ── Summary ───────────────────────────────────────────────────────────────────
_total    = len(NODE_CLASS_MAPPINGS)
_expected = 8
print("-"*60)
print(f"  {'All nodes loaded' if _total == _expected else str(_expected - _total) + ' node(s) failed'}"
      f" — {_total}/{_expected} active")
print(f"  Find nodes under 'S42 Production Suite' in the node menu")
print("="*60 + "\n")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
