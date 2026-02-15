"""
S42 Production Suite for ComfyUI
=================================
Professional audio mastering, analysis, and production tools.

Node Menu Category: S42 Production Suite

Phase 1 — Audio Mastering Chain:
  🎛️ S42P Parametric EQ          — 8-band parametric equalizer
  🔊 S42P Dynamics Processor     — Multiband compressor / expander / gate
  🎚️ S42P Mastering Chain        — LUFS targeting, stereo width, exciter, limiter, dither
  🥁 S42P Beat Analyzer          — BPM, beat grid, key detection + visualization

Designed for:
  • Music video production (AceStep → EQ → Dynamics → Master → Beat Analyze → Wan)
  • Talking head / lip-sync (TTS → EQ → Dynamics → Master → LatentSync)
  • Full song stem mastering (per-stem EQ + Dynamics, final master on mixdown)

Hardware target: RTX 4090 16GB (laptop) — all DSP runs on CPU, GPU stays for Wan/Qwen.
Python: 3.12 | ComfyUI Portable

Author: Studio42 Production Suite
Version: 1.0.0
"""

import sys
import os

# Make relative imports work correctly in ComfyUI portable
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ── Parametric EQ ─────────────────────────────────────────────────────────────
try:
    from .audio.s42p_eq_node import (
        NODE_CLASS_MAPPINGS        as _EQ_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _EQ_DN,
    )
    NODE_CLASS_MAPPINGS.update(_EQ_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_EQ_DN)
    print("  ✅ S42P Parametric EQ loaded")
except Exception as e:
    print(f"  ⚠️  S42P Parametric EQ failed to load: {e}")
    print("       → Check scipy is installed: pip install scipy")

# ── Dynamics Processor ────────────────────────────────────────────────────────
try:
    from .audio.s42p_dynamics_node import (
        NODE_CLASS_MAPPINGS        as _DYN_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _DYN_DN,
    )
    NODE_CLASS_MAPPINGS.update(_DYN_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_DYN_DN)
    print("  ✅ S42P Dynamics Processor loaded")
except Exception as e:
    print(f"  ⚠️  S42P Dynamics Processor failed to load: {e}")
    print("       → Check scipy is installed: pip install scipy")

# ── Mastering Chain ───────────────────────────────────────────────────────────
try:
    from .audio.s42p_mastering_node import (
        NODE_CLASS_MAPPINGS        as _MAST_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _MAST_DN,
    )
    NODE_CLASS_MAPPINGS.update(_MAST_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_MAST_DN)
    print("  ✅ S42P Mastering Chain loaded")
except Exception as e:
    print(f"  ⚠️  S42P Mastering Chain failed to load: {e}")
    print("       → Check pyloudnorm (optional but recommended): pip install pyloudnorm")

# ── Beat Analyzer ─────────────────────────────────────────────────────────────
try:
    from .audio.s42p_beat_analyzer import (
        NODE_CLASS_MAPPINGS        as _BEAT_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _BEAT_DN,
    )
    NODE_CLASS_MAPPINGS.update(_BEAT_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_BEAT_DN)
    print("  ✅ S42P Beat Analyzer loaded")
except Exception as e:
    print(f"  ⚠️  S42P Beat Analyzer failed to load: {e}")
    print("       → Check librosa (recommended): pip install librosa")

# ── Startup summary ───────────────────────────────────────────────────────────
total = len(NODE_CLASS_MAPPINGS)
print(f"\n{'='*60}")
print(f"  S42 Production Suite v1.0.0 — {total} node(s) loaded")
print(f"  Category: 'S42 Production Suite' in ComfyUI node menu")
if total < 4:
    print(f"  ⚠️  {4 - total} node(s) failed — check console output above")
print(f"{'='*60}\n")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
