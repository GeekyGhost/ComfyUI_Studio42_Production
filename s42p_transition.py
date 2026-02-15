"""
S42 Production Suite — Video Transitions Node
=============================================
CapCut-style transitions between two video clip batches.

Signal flow:
  clip_a [B_a, H, W, 3]  ─┐
  clip_b [B_b, H, W, 3]  ─┴─ [Transition] → merged [B_a + t + B_b, H, W, 3]

The transition window overlaps the tail of clip_a with the head of clip_b.
Output = clip_a (minus overlap) + transition_frames + clip_b (minus overlap).
Resolutions are auto-matched to clip_a's size.

Transition Types:
  cut           — instant cut (no transition frames, just concatenate)
  dissolve      — classic crossfade opacity blend
  fade_black    — fade A to black, fade B from black
  fade_white    — fade A to white, fade B from white
  push_left     — B pushes A off-screen to the left
  push_right    — B pushes A off-screen to the right
  push_up       — B pushes A off-screen upward
  push_down     — B pushes A off-screen downward
  wipe_left     — wipe reveal from right to left
  wipe_right    — wipe reveal from left to right
  wipe_up       — wipe reveal bottom to top
  wipe_down     — wipe reveal top to bottom
  zoom_in       — B zooms in from centre, replacing A
  zoom_out      — A zooms out to reveal B
  glitch        — RGB channel-shift glitch effect
  spin_cw       — clockwise rotation transition
  spin_ccw      — counter-clockwise rotation transition
  iris_in       — circular iris wipe opening
  slide_up      — B slides up from bottom
  slide_down    — B slides down from top

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
    return (a * (1.0 - t) + b * t).clip(0, 255).astype(np.uint8)


def _t_fade_color(a: np.ndarray, b: np.ndarray, t: float,
                  color: Tuple[int,int,int]) -> np.ndarray:
    c = np.array(color, dtype=np.float32)
    if t < 0.5:
        p = t / 0.5
        return (a * (1.0 - p) + c * p).clip(0, 255).astype(np.uint8)
    else:
        p = (t - 0.5) / 0.5
        return (c * (1.0 - p) + b * p).clip(0, 255).astype(np.uint8)


def _t_push(a: np.ndarray, b: np.ndarray, t: float,
            direction: str) -> np.ndarray:
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
    """B zooms in from centre over A."""
    h, w = a.shape[:2]
    scale = 0.1 + t * 0.9   # grows from 10% to 100%
    bh    = max(1, int(h * scale))
    bw    = max(1, int(w * scale))
    b_s   = np.array(Image.fromarray(b).resize((bw, bh), Image.Resampling.LANCZOS))
    out   = a.copy()
    py    = (h - bh) // 2
    px    = (w - bw) // 2
    # Clip paste region to canvas bounds
    src_y0 = max(0, -py); dst_y0 = max(0, py)
    src_x0 = max(0, -px); dst_x0 = max(0, px)
    ph = min(bh - src_y0, h - dst_y0)
    pw = min(bw - src_x0, w - dst_x0)
    if ph > 0 and pw > 0:
        out[dst_y0:dst_y0+ph, dst_x0:dst_x0+pw] = b_s[src_y0:src_y0+ph, src_x0:src_x0+pw]
    return out


def _t_zoom_out(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """A shrinks to nothing, revealing B."""
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
    """RGB channel-shift glitch between A and B."""
    rng    = np.random.default_rng(seed + int(t * 1000))
    base   = (a * (1.0 - t) + b * t).astype(np.float32)
    h, w   = a.shape[:2]
    amp    = int(t * w * 0.12) + 1
    out    = base.copy()
    # Shift R channel left, B channel right
    shift_r = rng.integers(-amp, amp)
    shift_b = rng.integers(-amp, amp)
    out[:, :, 0] = np.roll(base[:, :, 0], shift_r, axis=1)
    out[:, :, 2] = np.roll(base[:, :, 2], shift_b, axis=1)
    # Random scanline corruption
    n_lines = max(1, int(t * 8))
    for _ in range(n_lines):
        y = rng.integers(0, h)
        out[y, :] = rng.integers(0, 255, (w, 3))
    return out.clip(0, 255).astype(np.uint8)


def _t_spin(a: np.ndarray, b: np.ndarray, t: float,
            clockwise: bool) -> np.ndarray:
    """A spins out, B spins in (90° style)."""
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
    """Circular iris wipe opens to reveal B."""
    h, w   = a.shape[:2]
    out    = a.copy()
    cy, cx = h // 2, w // 2
    r_max  = math.hypot(cx, cy)
    r      = t * r_max
    # Generate coordinate arrays for circle mask
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask   = dist <= r
    out[mask] = b[mask]
    return out


def _t_slide(a: np.ndarray, b: np.ndarray, t: float,
             direction: str) -> np.ndarray:
    h, w = a.shape[:2]
    out  = a.copy()
    if direction == "up":
        oy = int(h * (1.0 - t))
        out[h-oy:, :] = b[:oy, :] if oy > 0 else out[h:, :]
        out[:h-oy, :] = a[:h-oy, :]
        # B slides from bottom
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
    🎬 S42P Transition
    Applies a CapCut-style transition between two video clip batches.

    Output = clip_a (trimmed) + transition_frames + clip_b (trimmed)
    Total output length = len(clip_a) + len(clip_b) - transition_frames
    so your total runtime is preserved.

    Connect the output directly to VHS Video Combine or another
    S42P Transition to chain multiple clips.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("IMAGE", {
                    "tooltip": ("First video clip (IMAGE batch). "
                                "The tail of this clip overlaps with the transition. "
                                "Connect from a video loader, Wan output, or Keyframe Animator.")
                }),
                "clip_b": ("IMAGE", {
                    "tooltip": ("Second video clip (IMAGE batch). "
                                "The head of this clip overlaps with the transition. "
                                "Resolution is auto-matched to clip_a.")
                }),
                "transition_type": (TRANSITION_TYPES, {
                    "default": "dissolve",
                    "tooltip": ("The transition effect to apply between the two clips.\n"
                                "cut = instant, no effect.\n"
                                "dissolve = classic crossfade blend.\n"
                                "fade_black/white = dip through colour.\n"
                                "push_* = clip B pushes clip A off screen.\n"
                                "wipe_* = hard edge wipe reveal.\n"
                                "zoom_in = B zooms in from centre.\n"
                                "zoom_out = A shrinks to reveal B.\n"
                                "glitch = RGB channel-shift noise effect.\n"
                                "spin_cw/ccw = rotation transition.\n"
                                "iris_in = circular iris wipe.\n"
                                "slide_up/down = B slides in from edge.")
                }),
                "transition_frames": ("INT", {
                    "default": 12, "min": 1, "max": 120,
                    "step": 1, "display": "slider",
                    "tooltip": ("Number of frames the transition takes. "
                                "At 24fps: 12 frames = 0.5 seconds (snappy), "
                                "24 frames = 1 second (standard), "
                                "48 frames = 2 seconds (slow/dramatic). "
                                "Cannot exceed the length of either clip.")
                }),
                "easing": (["ease_in_out", "linear", "ease_in",
                            "ease_out", "ease_in_out_cubic", "bounce"], {
                    "default": "ease_in_out",
                    "tooltip": ("Easing curve for the transition progress. "
                                "ease_in_out = slow start, fast middle, slow end (most natural). "
                                "linear = constant speed. "
                                "ease_in = starts slow, ends fast. "
                                "ease_out = starts fast, ends slow. "
                                "bounce = overshoots slightly at end.")
                }),
                "glitch_seed": ("INT", {
                    "default": 42, "min": 0, "max": 9999,
                    "tooltip": ("Random seed for the glitch transition. "
                                "Change this to get different glitch patterns. "
                                "Only used when transition_type is 'glitch'.")
                }),
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

        # Split clips: non-overlapping parts + overlap tails/heads
        a_keep  = clip_a[:na - tf]   # tail of a that stays
        a_over  = clip_a[na - tf:]   # tail of a used in transition
        b_over  = clip_b[:tf]        # head of b used in transition
        b_keep  = clip_b[tf:]        # rest of b

        # Build transition frames
        trans_frames = []
        for i in range(tf):
            raw_t = i / max(tf - 1, 1)
            t     = apply_easing(raw_t, easing)

            a_np = (a_over[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            b_np = (b_over[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            result_np = _apply_transition_frame(a_np, b_np, t, transition_type, glitch_seed)
            result_t  = torch.from_numpy(result_np.astype(np.float32) / 255.0)
            trans_frames.append(result_t)

        trans_tensor = torch.stack(trans_frames)

        # Merge: a_keep + transition + b_keep
        parts = [p for p in [a_keep, trans_tensor, b_keep] if p.shape[0] > 0]
        merged = torch.cat(parts, dim=0)

        return (merged, int(merged.shape[0]))


# ── ComfyUI registration ──────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "S42PTransition": S42PTransition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PTransition": "🎬 S42P Transition",
}
