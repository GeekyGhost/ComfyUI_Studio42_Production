"""
S42 Production Suite — Keyframe Animator Node
=============================================
2D animation compositor. Animates a layer IMAGE over a background
video (IMAGE batch) using keyframe strings for:
  position_x, position_y, scale_x, scale_y, opacity, rotation

Keyframe string format:
  "frame:value, frame:value, frame:value"
  Optional per-keyframe easing: "frame:value:easing_name"

Examples:
  position_x: "0:960, 24:200, 48:960"          ← ping-pong across screen
  opacity:    "0:0.0, 8:1.0, 40:1.0, 48:0.0"  ← fade in, hold, fade out
  scale_x:    "0:1.0, 12:1.5:ease_out, 48:1.0" ← bounce scale with custom ease
  rotation:   "0:0, 48:360"                    ← full spin over 48 frames

Available easing names:
  linear, ease_in, ease_out, ease_in_out,
  ease_in_cubic, ease_out_cubic, ease_in_out_cubic,
  elastic_in, elastic_out, bounce

Output is a standard IMAGE batch — connect directly to
VHS Video Combine, S42P Transition, or Save Image nodes.

Python 3.12 | ComfyUI Portable | PIL + torch
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import logging
from typing import Optional, Tuple

from PIL import Image

from .s42p_video_utils import (
    ensure_rgb, parse_keyframe_string, build_value_curve,
    frame_to_pil, pil_to_frame, ALL_EASINGS,
    EASING_FUNCTIONS
)

logger = logging.getLogger(__name__)

CATEGORY = "S42 Production Suite 🎬 Animation"

# ── Composite one frame ───────────────────────────────────────────────────────

def _composite_layer(bg_pil: Image.Image,
                     layer_pil: Image.Image,
                     mask_pil: Optional[Image.Image],
                     cx: float, cy: float,
                     sx: float, sy: float,
                     opacity: float,
                     rotation: float,
                     anchor_x: float, anchor_y: float) -> Image.Image:
    """
    Composite one animated layer frame onto a background PIL image.
    cx, cy = centre position in pixels (where layer anchor lands)
    sx, sy = scale multipliers
    opacity = 0.0-1.0
    rotation = degrees clockwise
    anchor_x/y = 0.0-1.0 (layer's own anchor point)
    """
    if opacity <= 0.001:
        return bg_pil  # nothing to draw

    # --- Scale layer ---------------------------------------------------------
    lw, lh  = layer_pil.size
    new_w   = max(1, int(lw * sx))
    new_h   = max(1, int(lh * sy))
    scaled  = layer_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # --- Scale mask alongside layer -----------------------------------------
    if mask_pil is not None:
        scaled_mask = mask_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        scaled_mask = None

    # --- Rotate around anchor -----------------------------------------------
    if abs(rotation) > 0.01:
        pivot_x = int(anchor_x * new_w)
        pivot_y = int(anchor_y * new_h)
        # PIL rotates around centre by default — expand canvas, then crop back
        expand_w = int(math.hypot(new_w, new_h)) + 4
        canvas   = Image.new("RGBA", (expand_w, expand_w), (0, 0, 0, 0))
        off_x    = (expand_w - new_w) // 2
        off_y    = (expand_w - new_h) // 2
        canvas.paste(scaled.convert("RGBA"), (off_x, off_y))
        rotated  = canvas.rotate(-rotation, resample=Image.Resampling.BICUBIC,
                                 center=(expand_w // 2, expand_w // 2), expand=False)

        # Mask rotation
        if scaled_mask is not None:
            mc = Image.new("L", (expand_w, expand_w), 0)
            mc.paste(scaled_mask, (off_x, off_y))
            mr = mc.rotate(-rotation, resample=Image.Resampling.BICUBIC,
                           center=(expand_w // 2, expand_w // 2), expand=False)
        else:
            mr = None

        # Trim back to original scaled size
        crop_x = (expand_w - new_w) // 2
        crop_y = (expand_w - new_h) // 2
        scaled  = rotated.crop((crop_x, crop_y, crop_x + new_w, crop_y + new_h)).convert("RGB")
        scaled_mask = mr.crop((crop_x, crop_y, crop_x + new_w, crop_y + new_h)) if mr else None
    else:
        scaled = scaled.convert("RGB")

    # --- Build alpha channel from mask + opacity ----------------------------
    if scaled_mask is not None:
        alpha_arr = np.array(scaled_mask).astype(np.float32) / 255.0
    else:
        alpha_arr = np.ones((new_h, new_w), dtype=np.float32)
    alpha_arr = (alpha_arr * opacity * 255.0).clip(0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha_arr, "L")

    # RGBA layer
    layer_rgba = scaled.convert("RGBA")
    layer_rgba.putalpha(alpha_img)

    # --- Calculate paste position from anchor + centre ----------------------
    paste_x = int(cx - anchor_x * new_w)
    paste_y = int(cy - anchor_y * new_h)

    # --- Composite onto background ------------------------------------------
    result = bg_pil.convert("RGBA")
    temp   = Image.new("RGBA", result.size, (0, 0, 0, 0))
    temp.paste(layer_rgba, (paste_x, paste_y), mask=layer_rgba)
    result = Image.alpha_composite(result, temp)
    return result.convert("RGB")


# ── Node ─────────────────────────────────────────────────────────────────────

class S42PKeyframeAnimator:
    """
    🎞️ S42P Keyframe Animator
    Composites an animated layer over a video frame batch.

    Keyframe String Format:
    ─────────────────────────────────────────────────
    "frame:value, frame:value, ..."
    Optionally add easing per keyframe:
    "frame:value:easing_name, ..."

    Frame numbers match your video's frame indices (0-based).
    Values before the first keyframe hold the first value.
    Values after the last keyframe hold the last value.

    Quick Examples:
    ─────────────────────────────────────────────────
    Slide in from right, hold, slide out:
      position_x: "0:1920, 12:960, 36:960, 48:0"

    Fade in, hold, fade out:
      opacity: "0:0.0, 8:1.0, 40:1.0, 48:0.0"

    Bounce scale:
      scale: "0:0.5, 6:1.2:ease_out, 12:1.0:bounce"

    Full spin:
      rotation: "0:0, 48:360"

    Available Easings:
      linear, ease_in, ease_out, ease_in_out,
      ease_in_cubic, ease_out_cubic, ease_in_out_cubic,
      elastic_in, elastic_out, bounce
    """

    @classmethod
    def INPUT_TYPES(cls):
        kf_tip = ("Keyframe string: 'frame:value, frame:value'\n"
                  "Optional easing: 'frame:value:ease_name'\n"
                  "Empty = property holds its default value for all frames.\n"
                  "Frame 0 = first frame of background video.")

        return {
            "required": {
                "background_frames": ("IMAGE", {
                    "tooltip": ("Video frame batch to composite onto. "
                                "Typically the output of a video loader, Wan generator, "
                                "or a previous Keyframe Animator node. "
                                "Must be a batch [B, H, W, 3].")
                }),
                "layer_image": ("IMAGE", {
                    "tooltip": ("The image/graphic to animate over the background. "
                                "Can be a single frame or a batch — if batch, frames cycle. "
                                "Use Background Remover output here for transparent subjects. "
                                "If the layer has no mask, a solid alpha is assumed.")
                }),

                # ── Position ─────────────────────────────────────────────────
                "position_x": ("STRING", {
                    "default": "0:960",
                    "multiline": False,
                    "tooltip": (f"Horizontal position keyframes (pixels from left edge). "
                                f"0 = left edge, canvas_width/2 = centre, canvas_width = right edge.\n{kf_tip}")
                }),
                "position_y": ("STRING", {
                    "default": "0:540",
                    "multiline": False,
                    "tooltip": (f"Vertical position keyframes (pixels from top edge). "
                                f"0 = top, canvas_height/2 = centre.\n{kf_tip}")
                }),

                # ── Scale ─────────────────────────────────────────────────────
                "scale_x": ("STRING", {
                    "default": "0:1.0",
                    "multiline": False,
                    "tooltip": (f"Horizontal scale keyframes. "
                                f"1.0 = original size. 2.0 = double width. 0.5 = half width.\n{kf_tip}")
                }),
                "scale_y": ("STRING", {
                    "default": "0:1.0",
                    "multiline": False,
                    "tooltip": (f"Vertical scale keyframes. "
                                f"Link to scale_x value for uniform scaling.\n{kf_tip}")
                }),
                "uniform_scale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": ("When enabled, scale_y is ignored and scale_x drives both axes equally. "
                                "Prevents unintentional stretching. "
                                "Disable only if you need non-uniform scaling (e.g. squeeze/stretch effects).")
                }),

                # ── Opacity ───────────────────────────────────────────────────
                "opacity": ("STRING", {
                    "default": "0:1.0",
                    "multiline": False,
                    "tooltip": (f"Opacity keyframes. 0.0 = fully transparent. 1.0 = fully opaque.\n"
                                f"Tip: fade in with '0:0.0, 8:1.0' — fade out with '40:1.0, 48:0.0'.\n{kf_tip}")
                }),

                # ── Rotation ──────────────────────────────────────────────────
                "rotation": ("STRING", {
                    "default": "0:0.0",
                    "multiline": False,
                    "tooltip": (f"Rotation keyframes in degrees (clockwise). "
                                f"0 = upright. 90 = rotated right. 180 = upside down. 360 = full spin.\n"
                                f"Tip: continuous spin: '0:0, {'{n}'}:360' where n = total frames.\n{kf_tip}")
                }),

                # ── Anchor ────────────────────────────────────────────────────
                "anchor_x": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "slider",
                    "tooltip": ("Horizontal anchor point on the layer (0=left edge, 0.5=centre, 1=right edge). "
                                "This is the point that stays pinned to position_x/position_y. "
                                "0.5/0.5 = centre anchor (most common). "
                                "0.0/0.0 = top-left anchor.")
                }),
                "anchor_y": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "slider",
                    "tooltip": ("Vertical anchor point on the layer (0=top, 0.5=centre, 1=bottom). "
                                "See anchor_x for details.")
                }),

                # ── Default easing ────────────────────────────────────────────
                "default_easing": (ALL_EASINGS, {
                    "default": "ease_in_out",
                    "tooltip": ("Default easing applied to all keyframe transitions "
                                "that don't specify their own easing in the string. "
                                "ease_in_out = smooth acceleration and deceleration (most natural). "
                                "linear = constant speed. "
                                "bounce = overshoots and bounces at destination. "
                                "elastic_out = spring effect at destination.")
                }),
            },
            "optional": {
                "layer_mask": ("MASK", {
                    "tooltip": ("Optional mask for the layer image. "
                                "White = opaque, Black = transparent. "
                                "Connect the MASK output from Background Remover here "
                                "to composite a subject with a transparent background. "
                                "If omitted, the full layer rectangle is used.")
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("animated_frames",)
    FUNCTION      = "animate"
    CATEGORY      = CATEGORY

    def animate(self, background_frames: torch.Tensor,
                layer_image: torch.Tensor,
                position_x: str, position_y: str,
                scale_x: str, scale_y: str, uniform_scale: bool,
                opacity: str, rotation: str,
                anchor_x: float, anchor_y: float,
                default_easing: str,
                layer_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:

        # Normalise inputs
        bg = ensure_rgb(background_frames)
        layer = ensure_rgb(layer_image)
        n_frames = bg.shape[0]
        h, w     = bg.shape[1], bg.shape[2]

        # ── Parse keyframe strings ────────────────────────────────────────────
        def _parse(s: str, default_val: float) -> list:
            kfs = parse_keyframe_string(s)
            # Apply default easing to any keyframe that has "ease_in_out" if
            # the user chose a different global default
            if default_easing != "ease_in_out":
                kfs = [(f, v, default_easing if e == "ease_in_out" else e)
                       for f, v, e in kfs]
            if not kfs:
                kfs = [(0, default_val, default_easing)]
            return kfs

        kf_px  = _parse(position_x, float(w / 2))
        kf_py  = _parse(position_y, float(h / 2))
        kf_sx  = _parse(scale_x,    1.0)
        kf_sy  = _parse(scale_y,    1.0)
        kf_op  = _parse(opacity,    1.0)
        kf_rot = _parse(rotation,   0.0)

        # Build per-frame value curves
        px_vals  = build_value_curve(kf_px,  n_frames)
        py_vals  = build_value_curve(kf_py,  n_frames)
        sx_vals  = build_value_curve(kf_sx,  n_frames)
        sy_vals  = build_value_curve(kf_sy,  n_frames)
        op_vals  = build_value_curve(kf_op,  n_frames)
        rot_vals = build_value_curve(kf_rot, n_frames)

        # Layer source frames (cycle if batch)
        n_layer = layer.shape[0]

        # Mask source
        has_mask = layer_mask is not None
        if has_mask:
            # Normalise mask dims [B,H,W] or [H,W]
            m = layer_mask
            if m.ndim == 2:
                m = m.unsqueeze(0)
            n_mask = m.shape[0]

        # ── Process each frame ────────────────────────────────────────────────
        out_frames = []
        for fi in range(n_frames):
            bg_pil    = frame_to_pil(bg[fi])
            layer_pil = frame_to_pil(layer[fi % n_layer])

            # Mask for this frame
            if has_mask:
                mask_np  = (m[fi % n_mask].cpu().numpy() * 255).clip(0,255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np, "L")
            else:
                mask_pil = None

            # Resolve animated values
            cx  = px_vals[fi]
            cy  = py_vals[fi]
            sx  = max(0.01, sx_vals[fi])
            sy  = sx if uniform_scale else max(0.01, sy_vals[fi])
            op  = float(np.clip(op_vals[fi], 0.0, 1.0))
            rot = rot_vals[fi]

            composited = _composite_layer(
                bg_pil, layer_pil, mask_pil,
                cx, cy, sx, sy, op, rot, anchor_x, anchor_y
            )
            out_frames.append(pil_to_frame(composited))

        return (torch.stack(out_frames),)


# ── ComfyUI registration ──────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "S42PKeyframeAnimator": S42PKeyframeAnimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PKeyframeAnimator": "🎞️ S42P Keyframe Animator",
}
