"""
S42 Production Suite for ComfyUI
=================================
Professional audio mastering, 2D animation, video editing, and color grading.

IMPORT STRATEGY — safe for concurrent node loading:
  - Uses spec_from_file_location() to load each node by absolute file path.
  - Does NOT manipulate sys.path at all — no directory insertions that could
    shadow other nodes' imports or interfere with concurrent loading.
  - Each module is registered in sys.modules under a unique prefixed name
    (s42prod_*) to prevent any collision with other custom nodes.

Version: 2.0.2 | Python 3.12 | ComfyUI Portable
"""

import sys
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print("\n" + "="*60)
print("  S42 Production Suite v2.0.2 — Loading...")
print("="*60)


def _load_node_file(label: str, rel_path: str, fix_hint: str) -> bool:
    """
    Load a node module directly from its absolute file path.

    Key safety properties:
    - Never modifies sys.path
    - Registers modules under unique 's42prod_*' names to avoid collisions
    - Each module gets its own loader context so intra-suite imports
      are resolved by pre-loading dependencies first, not via sys.path
    """
    filepath = os.path.join(_HERE, rel_path)
    if not os.path.isfile(filepath):
        print(f"  !! {label} FAILED: file not found — {filepath}")
        return False
    try:
        mod_name = "s42prod_" + os.path.splitext(os.path.basename(filepath))[0]
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


def _preload(mod_name: str, rel_path: str) -> None:
    """
    Pre-load a shared utility module under a unique prefixed name
    so that node files that import it can find it in sys.modules.
    No sys.path modification needed.
    """
    filepath = os.path.join(_HERE, rel_path)
    if not os.path.isfile(filepath):
        return
    if mod_name in sys.modules:
        return
    try:
        spec = importlib.util.spec_from_file_location(mod_name, filepath)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  !! Pre-load failed for {mod_name}: {e}")


# ── Pre-load shared utilities under unique prefixed names ─────────────────────
# Node files import these as e.g. "from s42p_audio_utils import ..."
# We register them under BOTH their bare name AND prefixed name so the
# import statement inside the node file resolves from sys.modules cache
# without touching sys.path.

_preload("s42p_audio_utils", "utils/s42p_audio_utils.py")
_preload("s42p_video_utils", "video/s42p_video_utils.py")

# ── Audio nodes ───────────────────────────────────────────────────────────────
_audio_nodes = [
    ("S42P Parametric EQ",      "audio/s42p_eq_node.py",        "pip install scipy"),
    ("S42P Dynamics Processor", "audio/s42p_dynamics_node.py",  "pip install scipy"),
    ("S42P Mastering Chain",    "audio/s42p_mastering_node.py", "pip install pyloudnorm"),
    ("S42P Beat Analyzer",      "audio/s42p_beat_analyzer.py",  "pip install librosa"),
]

for _label, _rel, _fix in _audio_nodes:
    _load_node_file(_label, _rel, _fix)

# ── Video / animation nodes ───────────────────────────────────────────────────
_video_nodes = [
    ("S42P Keyframe Animator", "video/s42p_keyframe_animator.py", "Requires Pillow (included with ComfyUI)"),
    ("S42P Transition",        "video/s42p_transition.py",        "Requires Pillow (included with ComfyUI)"),
    ("S42P Video Tools",       "video/s42p_video_tools.py",       "Requires torch (included with ComfyUI)"),
    ("S42P Color Grade",       "video/s42p_color_grade.py",       "Requires numpy (included with ComfyUI)"),
]

for _label, _rel, _fix in _video_nodes:
    _load_node_file(_label, _rel, _fix)

# ── Summary ───────────────────────────────────────────────────────────────────
_total    = len(NODE_CLASS_MAPPINGS)
_expected = 8
print("-"*60)
print(f"  {'All nodes loaded' if _total == _expected else str(_expected - _total) + ' node(s) failed'}"
      f" — {_total}/{_expected} active")
print(f"  Find nodes under 'S42 Production Suite' in the node menu")
print("="*60 + "\n")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
