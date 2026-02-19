"""
S42 Production Suite — FX Compositor Node v1.0
===============================================
4-layer compositor with per-layer blend mode, opacity, and optional mask.

Replaces the single fx_image slot in S42P Layer Composer for complex
multi-FX workflows. Stack up to 4 FX layers in one node instead of
chaining multiple Layer Composer nodes.

Layer order (bottom to top):
  layer_a → layer_b → layer_c → layer_d → output

Each layer has:
  - image input (IMAGE, optional)
  - mask  input (MASK,  optional) — uses image alpha if mask absent
  - blend_mode (14 modes)
  - opacity (0–1)
  - enabled toggle

Blend modes: normal, screen, add, overlay, multiply, soft_light,
             hard_light, dodge, burn, darken, lighten,
             difference, exclusion, subtract

Output: IMAGE (RGBA or RGB based on input)

Python 3.12 | ComfyUI Portable | numpy + torch
"""

import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

CATEGORY = "S42 Production Suite/Video Composite"

BLEND_MODES = [
    "normal", "screen", "add", "overlay", "multiply",
    "soft_light", "hard_light", "dodge", "burn",
    "darken", "lighten", "difference", "exclusion", "subtract",
]


def _to_np(t: torch.Tensor) -> np.ndarray:
    """[H,W,C] tensor → float32 numpy [H,W,C]."""
    return t.detach().cpu().numpy().astype(np.float32)


def _blend(base: np.ndarray, layer: np.ndarray, mode: str) -> np.ndarray:
    """
    Blend layer onto base using Photoshop-compatible formulas.
    Both inputs are float32 [H,W,3] in 0..1 range.
    """
    b, l = base, layer
    if   mode == "normal":     return l
    elif mode == "screen":     return 1 - (1-b)*(1-l)
    elif mode == "add":        return np.clip(b + l, 0, 1)
    elif mode == "multiply":   return b * l
    elif mode == "darken":     return np.minimum(b, l)
    elif mode == "lighten":    return np.maximum(b, l)
    elif mode == "difference": return np.abs(b - l)
    elif mode == "exclusion":  return b + l - 2*b*l
    elif mode == "subtract":   return np.clip(b - l, 0, 1)
    elif mode == "dodge":      return np.clip(b / np.where(l < 1, 1-l, 1e-6), 0, 1)
    elif mode == "burn":       return np.clip(1 - (1-b) / np.where(l > 0, l, 1e-6), 0, 1)
    elif mode == "overlay":
        return np.where(b < 0.5, 2*b*l, 1 - 2*(1-b)*(1-l))
    elif mode == "hard_light":
        return np.where(l < 0.5, 2*b*l, 1 - 2*(1-b)*(1-l))
    elif mode == "soft_light":
        return np.where(l < 0.5,
                        b - (1-2*l)*b*(1-b),
                        b + (2*l-1)*(np.sqrt(np.clip(b,0,1)) - b))
    return l  # fallback


def _composite_layer(base: np.ndarray,
                     img_t: Optional[torch.Tensor],
                     mask_t: Optional[torch.Tensor],
                     mode: str,
                     opacity: float,
                     frame_idx: int,
                     out_h: int, out_w: int) -> np.ndarray:
    """
    Composite one optional layer onto base [H,W,3].
    Returns updated base [H,W,3].
    """
    if img_t is None or opacity < 0.001:
        return base

    n = img_t.shape[0]
    fi = min(frame_idx, n - 1)
    frm = _to_np(img_t[fi])   # [H,W,C]

    # Resize to output dimensions if needed
    if frm.shape[0] != out_h or frm.shape[1] != out_w:
        import cv2
        frm = cv2.resize(frm, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # Extract alpha / mask
    if frm.shape[2] == 4:
        layer_alpha = frm[..., 3:4]   # [H,W,1]
        layer_rgb   = frm[..., :3]
    else:
        layer_rgb   = frm
        layer_alpha = np.ones((out_h, out_w, 1), dtype=np.float32)

    # Override alpha with explicit mask if provided
    if mask_t is not None:
        mn = mask_t.shape[0]
        mi = min(frame_idx, mn - 1)
        m  = _to_np(mask_t[mi])
        if m.ndim == 2:
            m = m[..., np.newaxis]
        if m.shape[0] != out_h or m.shape[1] != out_w:
            import cv2
            m = cv2.resize(m.squeeze(), (out_w, out_h), interpolation=cv2.INTER_LINEAR)[..., np.newaxis]
        layer_alpha = np.clip(m, 0, 1)

    # Apply blend mode
    blended = _blend(base, layer_rgb, mode)

    # Composite with alpha + opacity
    effective_alpha = layer_alpha * opacity
    return base * (1 - effective_alpha) + blended * effective_alpha


class S42PFXCompositor:
    """
    ✨ S42P FX Compositor v1.0

    4-layer compositor — stack procedural FX, reactive overlays,
    and custom images without chaining multiple Layer Composer nodes.

    Layer order (bottom to top): A → B → C → D

    Each layer:
      • image  — connect any IMAGE (Procedural FX, Reactive FX, etc.)
      • mask   — optional alpha mask (or uses image alpha channel)
      • mode   — 14 Photoshop-compatible blend modes
      • opacity — 0–1
      • enabled — quick toggle without disconnecting

    Output size: set by output_width / output_height, or auto-detected
    from the first connected layer.

    Typical use:
      Starfield FX → layer_a
      Gold Dust FX → layer_b (screen blend, 0.7 opacity)
      Neon Scanlines → layer_c (add blend, 0.4 opacity)
      Plasma Orbs → layer_d (overlay blend, 0.5 opacity)
    """

    @classmethod
    def INPUT_TYPES(cls):
        def layer_inputs(label: str, default_mode: str, default_opacity: float) -> dict:
            return {
                f"{label}_image":   ("IMAGE",   {"tooltip": f"Layer {label.upper()} image input."}),
                f"{label}_mask":    ("MASK",    {"tooltip": f"Optional alpha mask for layer {label.upper()}."}),
                f"{label}_mode":    (BLEND_MODES, {"default": default_mode,
                    "tooltip": f"Blend mode for layer {label.upper()}."}),
                f"{label}_opacity": ("FLOAT",   {"default": default_opacity, "min": 0.0, "max": 1.0,
                    "step": 0.01, "display": "slider",
                    "tooltip": f"Opacity for layer {label.upper()}. 0=invisible, 1=fully visible."}),
                f"{label}_enabled": ("BOOLEAN", {"default": True,
                    "tooltip": f"Enable/disable layer {label.upper()} without disconnecting."}),
            }

        return {
            "required": {
                "output_width":  ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Output canvas width. 0 = auto from first layer."}),
                "output_height": ("INT", {"default": 576,  "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Output canvas height. 0 = auto from first layer."}),
                "batch_size":    ("INT", {"default": 1, "min": 1, "max": 9999,
                    "tooltip": "Number of output frames. Layers cycle if shorter."}),
            },
            "optional": {
                **layer_inputs("a", "normal",  1.0),
                **layer_inputs("b", "screen",  0.8),
                **layer_inputs("c", "add",     0.6),
                **layer_inputs("d", "overlay", 0.5),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("composite",)
    FUNCTION      = "composite"
    CATEGORY      = CATEGORY

    def composite(
        self,
        output_width: int, output_height: int, batch_size: int,
        # Layer A
        a_image=None, a_mask=None, a_mode="normal", a_opacity=1.0, a_enabled=True,
        # Layer B
        b_image=None, b_mask=None, b_mode="screen", b_opacity=0.8, b_enabled=True,
        # Layer C
        c_image=None, c_mask=None, c_mode="add",    c_opacity=0.6, c_enabled=True,
        # Layer D
        d_image=None, d_mask=None, d_mode="overlay",d_opacity=0.5, d_enabled=True,
    ):
        # Determine output size from first connected layer if auto
        w, h = output_width, output_height
        for img in [a_image, b_image, c_image, d_image]:
            if img is not None:
                h_det, w_det = img.shape[1], img.shape[2]
                if w == 0: w = w_det
                if h == 0: h = h_det
                break
        w = max(w, 64)
        h = max(h, 64)

        layers = [
            (a_image, a_mask, a_mode, a_opacity, a_enabled),
            (b_image, b_mask, b_mode, b_opacity, b_enabled),
            (c_image, c_mask, c_mode, c_opacity, c_enabled),
            (d_image, d_mask, d_mode, d_opacity, d_enabled),
        ]

        out_frames = []
        for fi in range(batch_size):
            # Start with transparent/black base
            base = np.zeros((h, w, 3), dtype=np.float32)

            for img_t, mask_t, mode, opacity, enabled in layers:
                if not enabled:
                    continue
                base = _composite_layer(base, img_t, mask_t, mode, opacity, fi, h, w)

            out_frames.append(torch.from_numpy(np.clip(base, 0, 1).astype(np.float32)))

        result = torch.stack(out_frames)
        return (result,)


NODE_CLASS_MAPPINGS        = {"S42PFXCompositor": S42PFXCompositor}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PFXCompositor": "✨ S42P FX Compositor"}
