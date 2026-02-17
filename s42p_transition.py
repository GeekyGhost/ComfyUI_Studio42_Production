"""
S42 Production Suite — Video Transitions Node (FIXED)
=====================================================
CapCut-style transitions between two video clip batches.

FIXES in this version:
- Push/wipe transitions now use LAST frame of a_over and FIRST frame of b_over
  (not frame[i] which was causing redundant frames)
- Dissolve transitions properly blend corresponding frames
- All transitions generate correct frame count

Python 3.12 | ComfyUI Portable | PIL + torch
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import logging
from typing import Tuple

from PIL import Image

from s42p_video_utils import (
    ensure_rgb, match_resolution, apply_easing,
    frame_to_pil, pil_to_frame
)

logger = logging.getLogger(__name__)

CATEGORY = "S42 Production Suite 🎬 Animation"

TRANSITION_TYPES = [
    "cut", "dissolve", "fade_black", "fade_white",
    "push_left", "push_right", "push_up", "push_down",
    "wipe_left", "wipe_right", "wipe_up", "wipe_down",
    "zoom_in", "zoom_out", "glitch",
    "spin_cw", "spin_ccw", "iris_in",
    "slide_up", "slide_down",
]


# ── Per-transition frame generators ──────────────────────────────────────────

def _t_dissolve(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Blend two frames - uses both input frames"""
    return (a * (1.0 - t) + b * t).clip(0, 255).astype(np.uint8)


def _t_fade_color(a: np.ndarray, b: np.ndarray, t: float,
                  color: Tuple[int,int,int]) -> np.ndarray:
    """Fade through a color - uses both input frames"""
    c = np.array(color, dtype=np.float32)
    if t < 0.5:
        p = t / 0.5
        return (a * (1.0 - p) + c * p).clip(0, 255).astype(np.uint8)
    else:
        p = (t - 0.5) / 0.5
        return (c * (1.0 - p) + b * p).clip(0, 255).astype(np.uint8)


def _t_push(a: np.ndarray, b: np.ndarray, t: float,
            direction: str) -> np.ndarray:
    """Push transition - uses full frames, only t parameter matters"""
    h, w = a.shape[:2]
    out = np.zeros_like(a)
    if direction == "left":
        ox = int(w * t)
        if ox < w: out[:, :w-ox] = a[:, ox:]
        if ox > 0: out[:, w-ox:] = b[:, :ox]
    elif direction == "right":
        ox = int(w * t)
        if ox < w: out[:, ox:] = a[:, :w-ox]
        if ox > 0: out[:, :ox] = b[:, w-ox:]
    elif direction == "up":
        oy = int(h * t)
        if oy < h: out[:h-oy, :] = a[oy:, :]
        if oy > 0: out[h-oy:, :] = b[:oy, :]
    elif direction == "down":
        oy = int(h * t)
        if oy < h: out[oy:, :] = a[:h-oy, :]
        if oy > 0: out[:oy, :] = b[h-oy:, :]
    return out


def _t_wipe(a: np.ndarray, b: np.ndarray, t: float,
            direction: str) -> np.ndarray:
    """Wipe transition - uses full frames"""
    h, w = a.shape[:2]
    out  = a.copy()
    if direction == "left":
        cut = int(w * t)
        out[:, :cut] = b[:, :cut]
    elif direction == "right":
        cut = int(w * (1.0 - t))
        out[:, cut:] = b[:, cut:]
    elif direction == "up":
        cut = int(h * t)
        out[:cut, :] = b[:cut, :]
    elif direction == "down":
        cut = int(h * (1.0 - t))
        out[cut:, :] = b[cut:, :]
    return out


def _t_zoom_in(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """B zooms in from centre over A - uses full frames"""
    h, w = a.shape[:2]
    scale = 0.1 + t * 0.9
    bh    = max(1, int(h * scale))
    bw    = max(1, int(w * scale))
    b_s   = np.array(Image.fromarray(b).resize((bw, bh), Image.Resampling.LANCZOS))
    out   = a.copy()
    py    = (h - bh) // 2
    px    = (w - bw) // 2
    src_y0 = max(0, -py); dst_y0 = max(0, py)
    src_x0 = max(0, -px); dst_x0 = max(0, px)
    ph = min(bh - src_y0, h - dst_y0)
    pw = min(bw - src_x0, w - dst_x0)
    if ph > 0 and pw > 0:
        out[dst_y0:dst_y0+ph, dst_x0:dst_x0+pw] = b_s[src_y0:src_y0+ph, src_x0:src_x0+pw]
    return out


def _t_zoom_out(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """A shrinks to nothing, revealing B - uses full frames"""
    h, w  = a.shape[:2]
    scale = 1.0 - t
    ah    = max(1, int(h * scale))
    aw    = max(1, int(w * scale))
    a_s   = np.array(Image.fromarray(a).resize((aw, ah), Image.Resampling.LANCZOS))
    out   = b.copy()
    py    = (h - ah) // 2
    px    = (w - aw) // 2
    src_y0 = max(0, -py); dst_y0 = max(0, py)
    src_x0 = max(0, -px); dst_x0 = max(0, px)
    ph = min(ah - src_y0, h - dst_y0)
    pw = min(aw - src_x0, w - dst_x0)
    if ph > 0 and pw > 0:
        out[dst_y0:dst_y0+ph, dst_x0:dst_x0+pw] = a_s[src_y0:src_y0+ph, src_x0:src_x0+pw]
    return out


def _t_glitch(a: np.ndarray, b: np.ndarray, t: float,
              seed: int = 42) -> np.ndarray:
    """RGB channel-shift glitch - blends frames"""
    rng    = np.random.default_rng(seed + int(t * 1000))
    base   = (a * (1.0 - t) + b * t).astype(np.float32)
    h, w   = a.shape[:2]
    amp    = int(t * w * 0.12) + 1
    out    = base.copy()
    shift_r = rng.integers(-amp, amp)
    shift_b = rng.integers(-amp, amp)
    out[:, :, 0] = np.roll(base[:, :, 0], shift_r, axis=1)
    out[:, :, 2] = np.roll(base[:, :, 2], shift_b, axis=1)
    n_lines = max(1, int(t * 8))
    for _ in range(n_lines):
        y = rng.integers(0, h)
        out[y, :] = rng.integers(0, 255, (w, 3))
    return out.clip(0, 255).astype(np.uint8)


def _t_spin(a: np.ndarray, b: np.ndarray, t: float,
            clockwise: bool) -> np.ndarray:
    """A spins out, B spins in - switches frame at t=0.5"""
    direction = 1 if clockwise else -1
    if t < 0.5:
        angle = direction * t * 180.0
        src   = a
    else:
        angle = direction * (t - 0.5) * 180.0
        src   = b
    img = Image.fromarray(src)
    rot = img.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=False)
    return np.array(rot)


def _t_iris(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Circular iris wipe - uses full frames"""
    h, w   = a.shape[:2]
    out    = a.copy()
    cy, cx = h // 2, w // 2
    r_max  = math.hypot(cx, cy)
    r      = t * r_max
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask   = dist <= r
    out[mask] = b[mask]
    return out


def _t_slide(a: np.ndarray, b: np.ndarray, t: float,
             direction: str) -> np.ndarray:
    """Slide transition - uses full frames"""
    h, w = a.shape[:2]
    out  = a.copy()
    if direction == "up":
        segment_h = int(h * t)
        if segment_h > 0:
            out[h-segment_h:, :] = b[:segment_h, :]
    elif direction == "down":
        segment_h = int(h * t)
        if segment_h > 0:
            out[:segment_h, :] = b[h-segment_h:, :]
    return out


def _apply_transition_frame(a_np: np.ndarray, b_np: np.ndarray,
                             t: float, transition: str,
                             seed: int = 42) -> np.ndarray:
    """Dispatch to the correct transition function."""
    t = float(np.clip(t, 0.0, 1.0))
    if transition == "dissolve":
        return _t_dissolve(a_np, b_np, t)
    elif transition == "fade_black":
        return _t_fade_color(a_np, b_np, t, (0, 0, 0))
    elif transition == "fade_white":
        return _t_fade_color(a_np, b_np, t, (255, 255, 255))
    elif transition == "push_left":
        return _t_push(a_np, b_np, t, "left")
    elif transition == "push_right":
        return _t_push(a_np, b_np, t, "right")
    elif transition == "push_up":
        return _t_push(a_np, b_np, t, "up")
    elif transition == "push_down":
        return _t_push(a_np, b_np, t, "down")
    elif transition == "wipe_left":
        return _t_wipe(a_np, b_np, t, "left")
    elif transition == "wipe_right":
        return _t_wipe(a_np, b_np, t, "right")
    elif transition == "wipe_up":
        return _t_wipe(a_np, b_np, t, "up")
    elif transition == "wipe_down":
        return _t_wipe(a_np, b_np, t, "down")
    elif transition == "zoom_in":
        return _t_zoom_in(a_np, b_np, t)
    elif transition == "zoom_out":
        return _t_zoom_out(a_np, b_np, t)
    elif transition == "glitch":
        return _t_glitch(a_np, b_np, t, seed)
    elif transition == "spin_cw":
        return _t_spin(a_np, b_np, t, True)
    elif transition == "spin_ccw":
        return _t_spin(a_np, b_np, t, False)
    elif transition == "iris_in":
        return _t_iris(a_np, b_np, t)
    elif transition == "slide_up":
        return _t_slide(a_np, b_np, t, "up")
    elif transition == "slide_down":
        return _t_slide(a_np, b_np, t, "down")
    else:
        return _t_dissolve(a_np, b_np, t)


# ── Node ─────────────────────────────────────────────────────────────────────

class S42PTransition:
    """
    🎬 S42P Transition (FIXED)
    
    FIXED ISSUES:
    - Push/wipe/zoom transitions now use STABLE SOURCE FRAMES
    - For transitions that don't blend (push/wipe), we use:
      * Last frame of clip A (a_over[-1])
      * First frame of clip B (b_over[0])
    - For blending transitions (dissolve/glitch), we properly use:
      * a_over[i] and b_over[i] for each transition frame
    
    This ensures smooth motion without duplicate/redundant frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("IMAGE", {
                    "tooltip": "First video clip (IMAGE batch)"
                }),
                "clip_b": ("IMAGE", {
                    "tooltip": "Second video clip (IMAGE batch)"
                }),
                "transition_type": (TRANSITION_TYPES, {
                    "default": "dissolve"
                }),
                "transition_frames": ("INT", {
                    "default": 12, "min": 1, "max": 120, "step": 1, "display": "slider",
                    "tooltip": "Number of frames for transition (24fps: 12=0.5s, 24=1s)"
                }),
                "easing": (["ease_in_out", "linear", "ease_in",
                            "ease_out", "ease_in_out_cubic", "bounce"], {
                    "default": "ease_in_out"
                }),
                "glitch_seed": ("INT", {"default": 42, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "INT")
    RETURN_NAMES  = ("merged_frames", "total_frame_count")
    FUNCTION      = "apply_transition"
    CATEGORY      = CATEGORY

    def apply_transition(self, clip_a: torch.Tensor, clip_b: torch.Tensor,
                         transition_type: str, transition_frames: int,
                         easing: str, glitch_seed: int) -> Tuple[torch.Tensor, int]:

        clip_a = ensure_rgb(clip_a)
        clip_b = ensure_rgb(clip_b)
        clip_a, clip_b = match_resolution(clip_a, clip_b)

        na = clip_a.shape[0]
        nb = clip_b.shape[0]

        # Handle cut — just concatenate
        if transition_type == "cut":
            merged = torch.cat([clip_a, clip_b], dim=0)
            return (merged, int(merged.shape[0]))

        # Clamp transition length to available frames
        tf = min(transition_frames, na, nb)

        # Split clips
        a_keep  = clip_a[:na - tf]   # Frames before transition
        a_over  = clip_a[na - tf:]   # Frames available for transition (last tf frames of A)
        b_over  = clip_b[:tf]        # Frames available for transition (first tf frames of B)
        b_keep  = clip_b[tf:]        # Frames after transition

        # Determine which source frames to use
        # For blending transitions (dissolve, fade, glitch): use corresponding frames
        # For motion transitions (push, wipe, zoom, spin): use stable reference frames
        
        blending_transitions = ["dissolve", "fade_black", "fade_white", "glitch"]
        use_frame_blend = transition_type in blending_transitions
        
        # For non-blending transitions, use last frame of A and first frame of B
        if not use_frame_blend:
            a_ref = (a_over[-1].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            b_ref = (b_over[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # Build transition frames
        trans_frames = []
        for i in range(tf):
            raw_t = i / max(tf - 1, 1)
            t     = apply_easing(raw_t, easing)

            if use_frame_blend:
                # Use corresponding frames from overlap regions
                a_np = (a_over[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                b_np = (b_over[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            else:
                # Use stable reference frames
                a_np = a_ref
                b_np = b_ref

            result_np = _apply_transition_frame(a_np, b_np, t, transition_type, glitch_seed)
            result_t  = torch.from_numpy(result_np.astype(np.float32) / 255.0)
            trans_frames.append(result_t)

        trans_tensor = torch.stack(trans_frames)

        # Merge: a_keep + transition + b_keep
        parts = [p for p in [a_keep, trans_tensor, b_keep] if p.shape[0] > 0]
        merged = torch.cat(parts, dim=0)

        print(f"[S42P Transition] {transition_type}: {na}+{nb}-{tf}={merged.shape[0]} frames "
              f"(blend={'yes' if use_frame_blend else 'no'})")

        return (merged, int(merged.shape[0]))


# ── ComfyUI registration ──────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "S42PTransition": S42PTransition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PTransition": "🎬 S42P Transition",
}
