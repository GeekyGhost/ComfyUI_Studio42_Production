"""
S42 Production Suite â€” Shared Video Utilities
=============================================
Frame manipulation, easing functions, keyframe parsing, and
interpolation helpers used by all video nodes.

All operations work on ComfyUI IMAGE tensors: [B, H, W, C] float32 0-1.
No GPU models needed â€” pure torch/numpy/scipy/PIL operations.

Python 3.12 | ComfyUI Portable
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import re
import logging
from typing import List, Tuple, Optional, Dict, Union

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python not available â€” some transitions will use fallback")


# ”€”€ ComfyUI IMAGE tensor helpers ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def frames_to_np(frames: torch.Tensor) -> np.ndarray:
    """[B,H,W,C] float32 0-1 †’ [B,H,W,C] uint8 0-255"""
    return (frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def np_to_frames(arr: np.ndarray) -> torch.Tensor:
    """[B,H,W,C] uint8 0-255 †’ [B,H,W,C] float32 0-1"""
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


def frame_to_pil(frame: torch.Tensor) -> "Image.Image":
    """Single frame [H,W,C] float32 †’ PIL RGB"""
    return Image.fromarray((frame.cpu().numpy() * 255).clip(0,255).astype(np.uint8), "RGB")


def pil_to_frame(img: "Image.Image") -> torch.Tensor:
    """PIL RGB †’ [H,W,C] float32"""
    return torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0)


def ensure_rgb(frames: torch.Tensor) -> torch.Tensor:
    """Guarantee [B,H,W,3] €” drop alpha or expand grayscale."""
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    elif frames.shape[-1] == 1:
        frames = frames.repeat(1, 1, 1, 3)
    return frames


def match_resolution(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resize b to match a's HÃ—W if they differ.
    Uses bilinear interpolation. Returns (a, b_resized).
    """
    if a.shape[1:3] == b.shape[1:3]:
        return a, b
    h, w = a.shape[1], a.shape[2]
    b_p = b.permute(0, 3, 1, 2)                      # [B,C,H,W]
    b_r = F.interpolate(b_p, size=(h, w), mode="bilinear", align_corners=False)
    return a, b_r.permute(0, 2, 3, 1)                # back to [B,H,W,C]


# ”€”€ Easing functions ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def ease_linear(t: float) -> float:
    return t


def ease_in_quad(t: float) -> float:
    return t * t


def ease_out_quad(t: float) -> float:
    return t * (2.0 - t)


def ease_in_out_quad(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return -1.0 + (4.0 - 2.0 * t) * t


def ease_in_cubic(t: float) -> float:
    return t * t * t


def ease_out_cubic(t: float) -> float:
    p = t - 1.0
    return p * p * p + 1.0


def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4.0 * t * t * t
    p = 2.0 * t - 2.0
    return 0.5 * p * p * p + 1.0


def ease_in_elastic(t: float) -> float:
    if t in (0.0, 1.0):
        return t
    return -(2.0 ** (10.0 * t - 10.0)) * math.sin((t * 10.0 - 10.75) * (2.0 * math.pi) / 3.0)


def ease_out_elastic(t: float) -> float:
    if t in (0.0, 1.0):
        return t
    return 2.0 ** (-10.0 * t) * math.sin((t * 10.0 - 0.75) * (2.0 * math.pi) / 3.0) + 1.0


def ease_out_bounce(t: float) -> float:
    n, d = 7.5625, 2.75
    if t < 1.0 / d:
        return n * t * t
    elif t < 2.0 / d:
        t -= 1.5 / d
        return n * t * t + 0.75
    elif t < 2.5 / d:
        t -= 2.25 / d
        return n * t * t + 0.9375
    else:
        t -= 2.625 / d
        return n * t * t + 0.984375


EASING_FUNCTIONS = {
    "linear":         ease_linear,
    "ease_in":        ease_in_quad,
    "ease_out":       ease_out_quad,
    "ease_in_out":    ease_in_out_quad,
    "ease_in_cubic":  ease_in_cubic,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_cubic": ease_in_out_cubic,
    "elastic_in":     ease_in_elastic,
    "elastic_out":    ease_out_elastic,
    "bounce":         ease_out_bounce,
}

ALL_EASINGS = list(EASING_FUNCTIONS.keys())


def apply_easing(t: float, easing: str) -> float:
    """Apply named easing function to normalised time t ˆˆ [0,1]."""
    t = float(np.clip(t, 0.0, 1.0))
    fn = EASING_FUNCTIONS.get(easing, ease_in_out_quad)
    return fn(t)


# ”€”€ Keyframe string parser ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# Format: "frame:value, frame:value, ..."
# Optional per-keyframe easing: "frame:value:easing, ..."
# Examples:
#   "0:0.0, 24:1.0, 48:0.0"
#   "0:960:ease_in, 24:200:ease_out, 48:960"

def parse_keyframe_string(s: str) -> List[Tuple[int, float, str]]:
    """
    Parse a keyframe string into a list of (frame, value, easing) tuples.
    Tolerant of extra whitespace and trailing commas.
    Returns sorted list by frame number.
    """
    if not s or not s.strip():
        return []

    keyframes = []
    # Split on commas, handle optional easing tag
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) < 2:
            continue
        try:
            frame = int(parts[0].strip())
            value = float(parts[1].strip())
            easing = parts[2].strip() if len(parts) >= 3 else "ease_in_out"
            if easing not in EASING_FUNCTIONS:
                easing = "ease_in_out"
            keyframes.append((frame, value, easing))
        except (ValueError, IndexError):
            logger.warning(f"Skipping malformed keyframe token: '{token}'")
            continue

    return sorted(keyframes, key=lambda x: x[0])


def interpolate_keyframes(keyframes: List[Tuple[int, float, str]],
                           frame: int) -> float:
    """
    Interpolate the value at a given frame number from a keyframe list.
    - Before first keyframe: returns first value (hold)
    - After last keyframe: returns last value (hold)
    - Between keyframes: interpolates with the easing of the *from* keyframe
    """
    if not keyframes:
        return 0.0
    if frame <= keyframes[0][0]:
        return keyframes[0][1]
    if frame >= keyframes[-1][0]:
        return keyframes[-1][1]

    # Find surrounding keyframes
    for i in range(len(keyframes) - 1):
        f0, v0, easing = keyframes[i]
        f1, v1, _      = keyframes[i + 1]
        if f0 <= frame <= f1:
            if f1 == f0:
                return v1
            t   = (frame - f0) / (f1 - f0)
            t_e = apply_easing(t, easing)
            return v0 + (v1 - v0) * t_e

    return keyframes[-1][1]


def build_value_curve(keyframes: List[Tuple[int, float, str]],
                       n_frames: int) -> List[float]:
    """Build a per-frame value list from keyframes for n_frames total frames."""
    return [interpolate_keyframes(keyframes, f) for f in range(n_frames)]


# ”€”€ Framerate conversion ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def convert_fps(frames: torch.Tensor, src_fps: float, dst_fps: float,
                mode: str = "blend") -> torch.Tensor:
    """
    Convert a frame batch from src_fps to dst_fps.
    modes:
      'duplicate' â€” repeat/drop frames (fast, slight jitter)
      'blend'     â€” blend adjacent frames at fractional positions (smooth)
    """
    if abs(src_fps - dst_fps) < 0.01:
        return frames

    n_src    = frames.shape[0]
    duration = n_src / src_fps
    n_dst    = max(1, round(duration * dst_fps))

    if mode == "duplicate":
        indices = [min(int(i * src_fps / dst_fps), n_src - 1) for i in range(n_dst)]
        return frames[indices]

    # blend mode €” linear interpolation between adjacent frames
    out = []
    for i in range(n_dst):
        src_pos   = i * src_fps / dst_fps
        idx_lo    = min(int(src_pos), n_src - 1)
        idx_hi    = min(idx_lo + 1, n_src - 1)
        frac      = src_pos - int(src_pos)
        blended   = frames[idx_lo] * (1.0 - frac) + frames[idx_hi] * frac
        out.append(blended)
    return torch.stack(out)


# ”€”€ Simple image resize ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def resize_frame(frame: torch.Tensor, h: int, w: int,
                 mode: str = "bilinear") -> torch.Tensor:
    """Resize single [H,W,C] frame. Returns [h,w,C]."""
    x = frame.unsqueeze(0).permute(0, 3, 1, 2)   # [1,C,H,W]
    x = F.interpolate(x, size=(h, w), mode=mode,
                      align_corners=False if mode != "nearest" else None)
    return x.squeeze(0).permute(1, 2, 0)           # [h,w,C]
