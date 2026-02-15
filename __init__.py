"""
S42 Production Suite — Node Registration
==========================================
Flat-file layout compatible with GitHub web upload (no subdirectories).
All node files are expected in the same directory as this __init__.py.

AUDIO NODES (8 original + 4 new = 9 audio total):
  s42p_eq_node            → S42P Parametric EQ
  s42p_dynamics_node      → S42P Dynamics Processor
  s42p_mastering_node     → S42P Mastering Chain
  s42p_beat_analyzer      → S42P Beat Analyzer
  s42p_audio_analyser     → S42P Audio Analyser      [NEW v2.1]
  s42p_audio_visualizer   → S42P Audio Visualizer    [NEW v2.1]
  s42p_latent_enhancer    → S42P Latent Enhancer     [NEW v2.1]
  s42p_songwriter         → S42P Songwriter          [NEW v2.1]
  s42p_eq_presets         → S42P EQ Preset Loader    [NEW v2.1]

VIDEO NODES (4):
  s42p_keyframe_animator  → S42P Keyframe Animator
  s42p_transition         → S42P Transition
  s42p_video_tools        → S42P Video Tools
  s42p_color_grade        → S42P Color Grade

UTILITIES (loaded into sys.modules, not exposed as nodes):
  s42p_audio_utils        → audio DSP helpers shared by audio nodes
  s42p_video_utils        → video DSP helpers shared by video nodes

Version: 2.1.0
Python:  3.12
Layout:  flat (all .py files in repo root, compatible with GitHub web upload)
"""

import sys
import os
import importlib.util
import traceback
import logging

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(filename, module_name):
    """
    Load a .py file from the same directory as __init__.py into sys.modules.
    Uses spec_from_file_location — zero sys.path manipulation, safe for
    concurrent loading alongside other custom node suites.
    """
    path = os.path.join(_HERE, filename)
    if not os.path.isfile(path):
        logger.warning(f"[S42P Suite] Missing file, skipping: {filename}")
        return None
    try:
        spec   = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"[S42P Suite] Failed to load {filename}: {e}")
        logger.debug(traceback.format_exc())
        return None


# Pre-load shared utility modules so audio/video nodes can import them
_load_module("s42p_audio_utils.py", "s42p_audio_utils")
_load_module("s42p_video_utils.py", "s42p_video_utils")


_NODE_FILES = [
    # filename,                 module_name,             label
    ("s42p_eq_node.py",          "s42p_eq_node",          "Parametric EQ"),
    ("s42p_dynamics_node.py",    "s42p_dynamics_node",    "Dynamics Processor"),
    ("s42p_mastering_node.py",   "s42p_mastering_node",   "Mastering Chain"),
    ("s42p_beat_analyzer.py",    "s42p_beat_analyzer",    "Beat Analyzer"),
    ("s42p_audio_analyser.py",   "s42p_audio_analyser",   "Audio Analyser"),
    ("s42p_audio_visualizer.py", "s42p_audio_visualizer", "Audio Visualizer"),
    ("s42p_latent_enhancer.py",  "s42p_latent_enhancer",  "Latent Enhancer"),
    ("s42p_songwriter.py",       "s42p_songwriter",       "Songwriter"),
    ("s42p_eq_presets.py",       "s42p_eq_presets",       "EQ Preset Loader"),
    ("s42p_keyframe_animator.py","s42p_keyframe_animator","Keyframe Animator"),
    ("s42p_transition.py",       "s42p_transition",       "Transition"),
    ("s42p_video_tools.py",      "s42p_video_tools",      "Video Tools"),
    ("s42p_color_grade.py",      "s42p_color_grade",      "Color Grade"),
]

NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
_loaded = []
_failed = []

for _fname, _modname, _label in _NODE_FILES:
    _mod = _load_module(_fname, _modname)
    if _mod is None:
        _failed.append(_label)
        continue
    _cm = getattr(_mod, "NODE_CLASS_MAPPINGS",        {})
    _dm = getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
    if not _cm:
        logger.warning(f"[S42P Suite] {_fname} exports no NODE_CLASS_MAPPINGS")
        _failed.append(_label)
        continue
    NODE_CLASS_MAPPINGS.update(_cm)
    NODE_DISPLAY_NAME_MAPPINGS.update(_dm)
    _loaded.append(_label)

print(f"\n{'─'*60}")
print(f"  S42 Production Suite  v2.1.0")
print(f"  Nodes loaded ({len(_loaded)}): {', '.join(_loaded)}")
if _failed:
    print(f"  FAILED ({len(_failed)}): {', '.join(_failed)}  ← check log above")
print(f"{'─'*60}\n")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
