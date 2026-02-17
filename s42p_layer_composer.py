"""
S42 Production Suite â€” Layer Composer v1.0
============================================
Multi-layer image/video compositing node.

Ported from Studio42 24oiduts LayerComposer.
Animation code removed â€” use S42P Keyframe Animator for transforms.

COMPOSITION MODES:
  Stacking:  bg â†’ fx â†’ fg  (background, effects layer, foreground)
  Each layer: IMAGE + optional MASK + blend_mode + opacity

BLEND MODES:
  normal, multiply, screen, overlay, hard_light, soft_light,
  dodge, burn, darken, lighten, difference, exclusion, add, subtract

FEATURES:
  â€¢ Per-layer opacity (0.0 â€“ 1.0)
  â€¢ Per-layer mask input (MASK type from Background Remover etc.)
  â€¢ Output resolution control
  â€¢ Batch-aware: processes all frames in sync

INPUT LAYOUT:
  bg_image       â€” bottom background layer  (IMAGE)
  bg_mask        â€” optional mask for bg     (MASK)
  fx_image       â€” middle effects layer     (IMAGE, e.g. Visualizer/Matrix Rain)
  fx_mask        â€” optional mask for fx     (MASK)
  fg_image       â€” top foreground layer     (IMAGE, e.g. background-removed subject)
  fg_mask        â€” foreground alpha mask    (MASK, from BG Remover)

  Each layer can be None â€” just leave the input unconnected.

OUTPUT:
  composited     â€” final IMAGE batch
  alpha_mask     â€” composite alpha (MASK)

Python 3.12 | ComfyUI Portable
"""

from __future__ import annotations

import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ”€”€ Blend mode implementations ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _blend(base: np.ndarray, layer: np.ndarray, mode: str) -> np.ndarray:
    """Blend two float32 [H,W,3] arrays, return float32 [H,W,3]."""
    b, l = np.clip(base,0,1), np.clip(layer,0,1)
    m = mode.lower()

    if m == "normal":     return l
    if m == "multiply":   return b * l
    if m == "screen":     return 1-(1-b)*(1-l)
    if m == "overlay":
        return np.where(b < 0.5, 2*b*l, 1-2*(1-b)*(1-l))
    if m == "hard_light":
        return np.where(l < 0.5, 2*b*l, 1-2*(1-b)*(1-l))
    if m == "soft_light":
        return np.where(l < 0.5,
                        b - (1-2*l)*b*(1-b),
                        b + (2*l-1)*(np.sqrt(np.clip(b,0,1)) - b))
    if m == "dodge":      return np.clip(b / np.clip(1-l,1e-6,1), 0, 1)
    if m == "burn":       return np.clip(1-(1-b)/np.clip(l,1e-6,1), 0, 1)
    if m == "darken":     return np.minimum(b, l)
    if m == "lighten":    return np.maximum(b, l)
    if m == "difference": return np.abs(b-l)
    if m == "exclusion":  return b+l-2*b*l
    if m == "add":        return np.clip(b+l, 0, 1)
    if m == "subtract":   return np.clip(b-l, 0, 1)
    return l  # fallback = normal


def _composite_layer(canvas: np.ndarray, canvas_alpha: np.ndarray,
                     layer_rgb: np.ndarray, layer_mask: np.ndarray,
                     opacity: float, blend_mode: str) -> tuple:
    """
    Alpha-composite one layer over canvas using Porter-Duff 'over'.

    canvas       [H,W,3] float32
    canvas_alpha [H,W]   float32  (0..1)
    layer_rgb    [H,W,3] float32
    layer_mask   [H,W]   float32  (0..1, pre-multiplied with opacity)
    Returns: (new_canvas, new_alpha)
    """
    a_l = np.clip(layer_mask * float(opacity), 0, 1)[..., np.newaxis]  # [H,W,1]
    a_b = canvas_alpha[..., np.newaxis]

    # Blended colour
    blended = _blend(canvas, layer_rgb, blend_mode)

    # Porter-Duff 'over': out_a = a_l + a_b*(1-a_l)
    out_alpha = a_l[:,:,0] + canvas_alpha * (1 - a_l[:,:,0])

    # Composite
    denom = np.clip(out_alpha, 1e-7, 1)[..., np.newaxis]
    out_rgb = (blended * a_l + canvas * a_b * (1 - a_l)) / denom

    return np.clip(out_rgb, 0, 1), np.clip(out_alpha, 0, 1)


# ”€”€ Frame helpers ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _get_frame(batch: Optional[torch.Tensor], idx: int,
               target_h: int, target_w: int) -> Optional[np.ndarray]:
    """Get frame idx from batch (or cycle), resize to target, return [H,W,3] float32."""
    if batch is None: return None
    fi = idx % batch.shape[0]
    frm = batch[fi].cpu().numpy().astype(np.float32)
    if frm.shape[-1] == 4: frm = frm[:,:,:3]  # drop alpha channel if present
    if frm.shape[0] != target_h or frm.shape[1] != target_w:
        if _HAS_CV2:
            frm = cv2.resize(frm, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            # Simple numpy resize fallback
            from PIL import Image
            pil = Image.fromarray((frm*255).astype(np.uint8))
            pil = pil.resize((target_w, target_h), Image.LANCZOS)
            frm = np.array(pil).astype(np.float32)/255.0
    return frm


def _get_mask(batch: Optional[torch.Tensor], idx: int,
              target_h: int, target_w: int) -> np.ndarray:
    """Get mask frame, return [H,W] float32 or ones if None."""
    if batch is None:
        return np.ones((target_h, target_w), dtype=np.float32)
    fi = idx % batch.shape[0]
    if batch.ndim == 3:
        m = batch[fi].cpu().numpy().astype(np.float32)
    elif batch.ndim == 2:
        m = batch.cpu().numpy().astype(np.float32)
    else:
        m = batch[fi].cpu().numpy().astype(np.float32)
        if m.ndim > 2: m = m[:,:,0]

    if m.shape[0] != target_h or m.shape[1] != target_w:
        if _HAS_CV2:
            m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            from PIL import Image
            pil = Image.fromarray((m*255).astype(np.uint8), "L")
            pil = pil.resize((target_w, target_h), Image.LANCZOS)
            m   = np.array(pil).astype(np.float32)/255.0
    return np.clip(m, 0, 1)


# ”€”€ Node ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

BLEND_MODES = [
    "normal","multiply","screen","overlay","hard_light","soft_light",
    "dodge","burn","darken","lighten","difference","exclusion","add","subtract"
]

class S42PLayerComposer:
    """S42P Layer Composer €” composite up to 3 layers (bg / fx / fg) for video and images.

    Standard music video workflow:
      bg_image  â† S42P Audio Visualizer or Matrix Rain (reactive background)
      fx_image  â† S42P Procedural FX or other effects layer
      fg_image  â† subject video (background removed with S42P Background Remover)
      fg_mask   â† alpha mask from S42P Background Remover â†’ result_mask

    Also works for still images â€” batch size 1 in, batch size 1 out.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_width":  ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Output width in pixels."}),
                "output_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Output height in pixels."}),
            },
            "optional": {
                # Background layer
                "bg_image":      ("IMAGE",  {"tooltip": "Bottom background layer."}),
                "bg_mask":       ("MASK",   {"tooltip": "Optional mask for background layer."}),
                "bg_opacity":    ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bg_blend_mode": (BLEND_MODES, {"default": "normal"}),

                # Effects layer (middle)
                "fx_image":      ("IMAGE",  {"tooltip": "Middle effects layer (visualizer, rain, etc.)."}),
                "fx_mask":       ("MASK",   {"tooltip": "Optional mask for FX layer."}),
                "fx_opacity":    ("FLOAT",  {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fx_blend_mode": (BLEND_MODES, {"default": "screen",
                    "tooltip": "screen blends additive light well. Try add or overlay for effects."}),

                # Foreground layer (top)
                "fg_image":      ("IMAGE",  {"tooltip": "Top foreground layer (subject video)."}),
                "fg_mask":       ("MASK",   {"tooltip": "Alpha mask for fg â€” connect BG Remover result_mask here."}),
                "fg_opacity":    ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fg_blend_mode": (BLEND_MODES, {"default": "normal"}),

                # Output
                "bg_fill_color": ("STRING", {"default": "#000000",
                    "tooltip": "Hex fill color for canvas before any layers. Default black."}),
                "output_format": (["rgb","rgba"], {"default": "rgb",
                    "tooltip": "rgb = 3-channel output. rgba = include composite alpha."}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK")
    RETURN_NAMES  = ("composited", "alpha_mask")
    FUNCTION      = "compose"
    CATEGORY      = "S42 Production Suite"

    def compose(self, output_width, output_height,
                bg_image=None, bg_mask=None, bg_opacity=1.0, bg_blend_mode="normal",
                fx_image=None, fx_mask=None, fx_opacity=0.8, fx_blend_mode="screen",
                fg_image=None, fg_mask=None, fg_opacity=1.0, fg_blend_mode="normal",
                bg_fill_color="#000000", output_format="rgb"):

        W, H = int(output_width), int(output_height)

        # Determine frame count from largest batch
        def batch_len(t): return t.shape[0] if t is not None else 0
        N = max(batch_len(bg_image), batch_len(fx_image), batch_len(fg_image), 1)

        # Parse fill color
        fill = self._parse_hex(bg_fill_color)

        out_frames = []
        out_alphas = []

        for i in range(N):
            # Canvas starts with fill color, full alpha = 0 (transparent)
            canvas       = np.full((H, W, 3), fill, dtype=np.float32)
            canvas_alpha = np.zeros((H, W), dtype=np.float32)

            # BG layer
            bg = _get_frame(bg_image, i, H, W)
            if bg is not None:
                bm = _get_mask(bg_mask, i, H, W)
                canvas, canvas_alpha = _composite_layer(
                    canvas, canvas_alpha, bg, bm, bg_opacity, bg_blend_mode)

            # FX layer
            fx = _get_frame(fx_image, i, H, W)
            if fx is not None:
                fm = _get_mask(fx_mask, i, H, W)
                canvas, canvas_alpha = _composite_layer(
                    canvas, canvas_alpha, fx, fm, fx_opacity, fx_blend_mode)

            # FG layer
            fg = _get_frame(fg_image, i, H, W)
            if fg is not None:
                fgm = _get_mask(fg_mask, i, H, W)
                canvas, canvas_alpha = _composite_layer(
                    canvas, canvas_alpha, fg, fgm, fg_opacity, fg_blend_mode)

            # If no layers at all, fill is fully opaque black
            if bg is None and fx is None and fg is None:
                canvas_alpha = np.ones((H, W), dtype=np.float32)

            if output_format == "rgba":
                rgba = np.dstack([canvas, canvas_alpha[..., np.newaxis]])
                out_frames.append(torch.from_numpy(rgba.astype(np.float32)))
            else:
                out_frames.append(torch.from_numpy(canvas.astype(np.float32)))

            out_alphas.append(torch.from_numpy(canvas_alpha.astype(np.float32)))

        composited = torch.stack(out_frames, dim=0)
        alpha_mask = torch.stack(out_alphas, dim=0)

        print(f"[S42P Layer Composer] âœ… {N} frames  {W}x{H}  format={output_format}")
        return (composited, alpha_mask)

    def _parse_hex(self, hex_str: str) -> tuple:
        try:
            s = hex_str.strip().lstrip("#")
            if len(s) == 3: s = "".join(c*2 for c in s)
            r, g, b = int(s[0:2],16)/255.0, int(s[2:4],16)/255.0, int(s[4:6],16)/255.0
            return (r, g, b)
        except Exception:
            return (0.0, 0.0, 0.0)


NODE_CLASS_MAPPINGS        = {"S42PLayerComposer": S42PLayerComposer}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PLayerComposer": "S42P Layer Composer ðŸŽ¬"}
