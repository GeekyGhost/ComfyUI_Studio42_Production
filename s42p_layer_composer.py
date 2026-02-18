"""
S42 Production Suite -- Layer Composer v3.0
============================================
Multi-layer image/video compositing node.

LAYERS (bottom to top):
  bg_image  — background, always stretched to fill canvas
  fx_image  — effects overlay, always stretched to fill canvas
  fg_image  — foreground subject; has fit/position/custom placement controls

FOREGROUND PLACEMENT:
  fit_mode:
    fill        — stretch to fill canvas (same as bg/fx)
    fit         — scale uniformly so whole image fits inside canvas (letterboxed)
    fit_width   — scale to match canvas width, crop/letterbox height
    fit_height  — scale to match canvas height, crop/letterbox width
    original    — keep source pixel dimensions
    custom      — manually set width, height, x, y via sliders

  position (ignored in fill / custom modes):
    center, top_left, top_center, top_right,
    center_left, center_right,
    bottom_left, bottom_center, bottom_right

  custom mode:
    fg_custom_w    — rendered width in pixels
    fg_custom_h    — rendered height in pixels
    fg_custom_x    — left edge in canvas pixels (0 = left edge, negative = off-screen left)
    fg_custom_y    — top edge in canvas pixels  (0 = top edge,  negative = off-screen up)

BLEND MODES (all layers):
  normal, multiply, screen, overlay, hard_light, soft_light,
  dodge, burn, darken, lighten, difference, exclusion, add, subtract

Python 3.12 | ComfyUI Portable
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_CHUNK = 64


# ---------------------------------------------------------------------------
# Blend modes
# ---------------------------------------------------------------------------

def _blend(base: np.ndarray, layer: np.ndarray, mode: str) -> np.ndarray:
    b, l = np.clip(base, 0, 1), np.clip(layer, 0, 1)
    m = mode.lower()
    if m == "normal":     return l
    if m == "multiply":   return b * l
    if m == "screen":     return 1 - (1-b)*(1-l)
    if m == "overlay":    return np.where(b < 0.5, 2*b*l, 1 - 2*(1-b)*(1-l))
    if m == "hard_light": return np.where(l < 0.5, 2*b*l, 1 - 2*(1-b)*(1-l))
    if m == "soft_light": return np.where(l < 0.5,
                                           b - (1-2*l)*b*(1-b),
                                           b + (2*l-1)*(np.sqrt(np.clip(b,0,1)) - b))
    if m == "dodge":      return np.clip(b / np.clip(1-l, 1e-6, 1), 0, 1)
    if m == "burn":       return np.clip(1 - (1-b) / np.clip(l, 1e-6, 1), 0, 1)
    if m == "darken":     return np.minimum(b, l)
    if m == "lighten":    return np.maximum(b, l)
    if m == "difference": return np.abs(b - l)
    if m == "exclusion":  return b + l - 2*b*l
    if m == "add":        return np.clip(b+l, 0, 1)
    if m == "subtract":   return np.clip(b-l, 0, 1)
    return l


def _composite_layer(canvas, canvas_alpha, layer_rgb, layer_mask, opacity, blend_mode):
    a_l = np.clip(layer_mask * float(opacity), 0, 1)[..., np.newaxis]
    a_b = canvas_alpha[..., np.newaxis]
    blended   = _blend(canvas, layer_rgb, blend_mode)
    out_alpha = a_l[:,:,0] + canvas_alpha * (1 - a_l[:,:,0])
    denom     = np.clip(out_alpha, 1e-7, 1)[..., np.newaxis]
    out_rgb   = (blended * a_l + canvas * a_b * (1 - a_l)) / denom
    return np.clip(out_rgb, 0, 1), np.clip(out_alpha, 0, 1)


# ---------------------------------------------------------------------------
# Resize helper
# ---------------------------------------------------------------------------

def _pil_resize(arr: np.ndarray, h: int, w: int, is_mask: bool) -> np.ndarray:
    if is_mask:
        pil = Image.fromarray((np.clip(arr, 0, 1)*255).astype(np.uint8), "L")
        return np.array(pil.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0
    pil = Image.fromarray((np.clip(arr, 0, 1)*255).astype(np.uint8))
    if pil.mode == "RGBA":
        pil = pil.convert("RGB")
    return np.array(pil.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0


def _get_frame(batch: Optional[torch.Tensor], idx: int, h: int, w: int) -> Optional[np.ndarray]:
    """Full-frame resize — used for bg and fx."""
    if batch is None:
        return None
    frm = batch[idx % batch.shape[0]].cpu().numpy().astype(np.float32)
    if frm.ndim == 2:
        frm = np.stack([frm, frm, frm], axis=-1)
    elif frm.shape[-1] == 4:
        frm = frm[:, :, :3]
    if frm.shape[0] != h or frm.shape[1] != w:
        frm = _pil_resize(frm, h, w, is_mask=False)
    return frm


def _get_mask_frame(batch: Optional[torch.Tensor], idx: int, h: int, w: int) -> np.ndarray:
    if batch is None:
        return np.ones((h, w), dtype=np.float32)
    m = (batch[idx % batch.shape[0]] if batch.ndim == 3 else batch).cpu().numpy().astype(np.float32)
    if m.ndim > 2:
        m = m[:, :, 0]
    if m.shape[0] != h or m.shape[1] != w:
        m = _pil_resize(m, h, w, is_mask=True)
    return np.clip(m, 0, 1)


# ---------------------------------------------------------------------------
# Foreground placement
# ---------------------------------------------------------------------------

_FIT_MODES = [
    "fill",
    "fit",
    "fit_width",
    "fit_height",
    "original",
    "custom",
]

_POSITIONS = [
    "center",
    "top_left",    "top_center",    "top_right",
    "center_left",                  "center_right",
    "bottom_left", "bottom_center", "bottom_right",
]


def _fg_place(fg_rgb: np.ndarray, fg_mask: np.ndarray,
              canvas_h: int, canvas_w: int,
              fit_mode: str, position: str,
              custom_w: int, custom_h: int,
              custom_x: int, custom_y: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale and place the foreground onto a canvas-sized buffer.
    Returns placed_rgb [H,W,3] and placed_mask [H,W], both float32.
    """
    src_h, src_w = fg_rgb.shape[:2]
    H, W = canvas_h, canvas_w

    # --- Target render size ---
    if fit_mode == "fill":
        rw, rh = W, H
    elif fit_mode == "fit":
        scale = min(W / src_w, H / src_h)
        rw, rh = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    elif fit_mode == "fit_width":
        scale = W / src_w
        rw, rh = W, max(1, int(src_h * scale))
    elif fit_mode == "fit_height":
        scale = H / src_h
        rw, rh = max(1, int(src_w * scale)), H
    elif fit_mode == "original":
        rw, rh = src_w, src_h
    else:  # custom
        rw, rh = max(1, custom_w), max(1, custom_h)

    # --- Resize ---
    if rw != src_w or rh != src_h:
        r_rgb  = _pil_resize(fg_rgb,  rh, rw, is_mask=False)
        r_mask = _pil_resize(fg_mask, rh, rw, is_mask=True)
    else:
        r_rgb  = fg_rgb
        r_mask = fg_mask if fg_mask.shape == (rh, rw) else _pil_resize(fg_mask, rh, rw, is_mask=True)

    # --- Offset ---
    if fit_mode == "custom":
        ox, oy = custom_x, custom_y
    elif fit_mode == "fill":
        ox, oy = 0, 0
    else:
        pos = position.lower()
        ox_map = {
            "top_left": 0,    "center_left": 0,    "bottom_left": 0,
            "top_center": (W-rw)//2, "center": (W-rw)//2, "bottom_center": (W-rw)//2,
            "top_right": W-rw, "center_right": W-rw, "bottom_right": W-rw,
        }
        oy_map = {
            "top_left": 0,    "top_center": 0,    "top_right": 0,
            "center_left": (H-rh)//2, "center": (H-rh)//2, "center_right": (H-rh)//2,
            "bottom_left": H-rh, "bottom_center": H-rh, "bottom_right": H-rh,
        }
        ox = ox_map.get(pos, (W-rw)//2)
        oy = oy_map.get(pos, (H-rh)//2)

    # --- Copy into canvas-sized buffer (with bounds clipping) ---
    placed_rgb  = np.zeros((H, W, 3), dtype=np.float32)
    placed_mask = np.zeros((H, W),    dtype=np.float32)

    sx0 = max(0, -ox);  sy0 = max(0, -oy)
    dx0 = max(0,  ox);  dy0 = max(0,  oy)
    cw  = min(rw - sx0, W - dx0)
    ch  = min(rh - sy0, H - dy0)

    if cw > 0 and ch > 0:
        placed_rgb [dy0:dy0+ch, dx0:dx0+cw] = r_rgb [sy0:sy0+ch, sx0:sx0+cw]
        placed_mask[dy0:dy0+ch, dx0:dx0+cw] = r_mask[sy0:sy0+ch, sx0:sx0+cw]

    return placed_rgb, placed_mask


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLEND_MODES = [
    "normal", "multiply", "screen", "overlay", "hard_light", "soft_light",
    "dodge", "burn", "darken", "lighten", "difference", "exclusion", "add", "subtract",
]
CATEGORY = "S42 Production Suite/Video Composite"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class S42PLayerComposer:
    """
    S42P Layer Composer v3.0

    Standard music video workflow:
      bg_image  <- audio visualizer / colour card  (fills canvas, no placement controls)
      fx_image  <- procedural FX overlay            (fills canvas, no placement controls)
      fg_image  <- subject with background removed  (fit + position + custom placement)
      fg_mask   <- alpha from S42P Background Remover
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_width":  ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8,
                                          "tooltip": "Canvas output width in pixels."}),
                "output_height": ("INT", {"default": 576,  "min": 64, "max": 8192, "step": 8,
                                          "tooltip": "Canvas output height in pixels."}),
            },
            "optional": {

                # ── Background (always fills canvas) ──────────────────────────
                "bg_image":      ("IMAGE", {
                    "tooltip": "Bottom layer. Always stretched to fill the full canvas. "
                               "Connect an audio visualizer, solid colour, or background video."}),
                "bg_mask":       ("MASK",  {
                    "tooltip": "Optional alpha mask for the background layer."}),
                "bg_opacity":    ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "Background layer opacity (0=invisible, 1=fully visible)."}),
                "bg_blend_mode": (BLEND_MODES, {"default": "normal",
                                                "tooltip": "Blend mode for the background layer."}),

                # ── FX overlay (always fills canvas) ──────────────────────────
                "fx_image":      ("IMAGE", {
                    "tooltip": "Middle effects layer. Always stretched to fill the full canvas. "
                               "Connect S42P Procedural FX, particle overlays, etc."}),
                "fx_mask":       ("MASK",  {
                    "tooltip": "Optional alpha mask for the FX layer. "
                               "Use a generated mask or the alpha channel from a transparent FX render."}),
                "fx_opacity":    ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "FX layer opacity."}),
                "fx_blend_mode": (BLEND_MODES, {"default": "screen",
                                                "tooltip": "Blend mode for the FX layer. "
                                                           "'screen' is ideal for light-based FX. "
                                                           "Try 'add' for intense glow or 'overlay' for contrast."}),

                # ── Foreground (with placement controls) ──────────────────────
                "fg_image":      ("IMAGE", {
                    "tooltip": "Top foreground layer. Usually a subject with its background removed. "
                               "Placement is controlled by fg_fit_mode, fg_position, and the fg_custom_* sliders."}),
                "fg_mask":       ("MASK",  {
                    "tooltip": "Alpha mask for the foreground. "
                               "Connect result_mask from S42P Background Remover to cut out the subject."}),
                "fg_opacity":    ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "Foreground layer opacity."}),
                "fg_blend_mode": (BLEND_MODES, {"default": "normal",
                                                "tooltip": "Blend mode for the foreground layer. "
                                                           "'normal' is correct for a cut-out subject."}),

                # Fit mode
                "fg_fit_mode":   (_FIT_MODES, {"default": "fit",
                                               "tooltip":
                                   "How the foreground image is scaled before placement:\n"
                                   "  fill       — stretch to fill the entire canvas (ignores fg_position)\n"
                                   "  fit        — scale uniformly so the whole image fits inside the canvas (letterboxed)\n"
                                   "  fit_width  — scale to match canvas width; height may be taller than canvas\n"
                                   "  fit_height — scale to match canvas height; width may be wider than canvas\n"
                                   "  original   — keep the source image at its original pixel size\n"
                                   "  custom     — set exact rendered width, height, and position via the fg_custom_* sliders"}),

                # Position preset
                "fg_position":   (_POSITIONS, {"default": "center",
                                               "tooltip":
                                   "Where to anchor the (scaled) foreground on the canvas. "
                                   "Not used when fg_fit_mode is 'fill' or 'custom'.\n"
                                   "Options: center, top_left, top_center, top_right, "
                                   "center_left, center_right, bottom_left, bottom_center, bottom_right."}),

                # Custom placement (only active when fg_fit_mode == "custom")
                "fg_custom_w":   ("INT", {
                    "default": 512, "min": 1, "max": 8192, "step": 1,
                    "tooltip": "Rendered width of the foreground image in pixels.\n"
                               "Active only when fg_fit_mode is 'custom'."}),
                "fg_custom_h":   ("INT", {
                    "default": 512, "min": 1, "max": 8192, "step": 1,
                    "tooltip": "Rendered height of the foreground image in pixels.\n"
                               "Active only when fg_fit_mode is 'custom'."}),
                "fg_custom_x":   ("INT", {
                    "default": 0, "min": -8192, "max": 8192, "step": 1,
                    "tooltip": "Horizontal position of the foreground's left edge on the canvas (pixels).\n"
                               "0 = left edge of canvas. Positive = move right. Negative = move off-screen left.\n"
                               "Active only when fg_fit_mode is 'custom'."}),
                "fg_custom_y":   ("INT", {
                    "default": 0, "min": -8192, "max": 8192, "step": 1,
                    "tooltip": "Vertical position of the foreground's top edge on the canvas (pixels).\n"
                               "0 = top edge of canvas. Positive = move down. Negative = move off-screen up.\n"
                               "Active only when fg_fit_mode is 'custom'."}),

                # ── Output ────────────────────────────────────────────────────
                "bg_fill_color": ("STRING", {"default": "#000000",
                                             "tooltip": "Hex background fill colour visible through transparent areas. "
                                                        "Default black (#000000). Supports shorthand like #F00."}),
                "output_format": (["rgb", "rgba"], {"default": "rgb",
                                                    "tooltip": "rgb = 3-channel output tensor (standard for most nodes). "
                                                               "rgba = includes composite alpha as a 4th channel."}),
                "chunk_size":    ("INT", {"default": 64, "min": 4, "max": 256, "step": 4,
                                          "tooltip": "Frames per processing chunk (memory optimisation). "
                                                     "64 uses ~110 MB/chunk at 512×288. "
                                                     "Reduce to 8–16 if you get out-of-memory errors on long batches."}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK")
    RETURN_NAMES  = ("composited", "alpha_mask")
    FUNCTION      = "compose"
    CATEGORY      = CATEGORY

    def compose(self,
                output_width=1024, output_height=576,
                bg_image=None, bg_mask=None, bg_opacity=1.0,  bg_blend_mode="normal",
                fx_image=None, fx_mask=None, fx_opacity=0.8,  fx_blend_mode="screen",
                fg_image=None, fg_mask=None, fg_opacity=1.0,  fg_blend_mode="normal",
                fg_fit_mode="fit", fg_position="center",
                fg_custom_w=512, fg_custom_h=512, fg_custom_x=0, fg_custom_y=0,
                bg_fill_color="#000000", output_format="rgb", chunk_size=64):

        W, H = int(output_width), int(output_height)
        CS   = max(4, int(chunk_size))

        def _blen(t): return t.shape[0] if t is not None else 0
        N = max(_blen(bg_image), _blen(fx_image), _blen(fg_image), 1)

        fill       = self._parse_hex(bg_fill_color)
        frame_list = []
        alpha_list = []
        chunk_frames: list[torch.Tensor] = []
        chunk_alphas: list[torch.Tensor] = []

        for i in range(N):
            canvas       = np.full((H, W, 3), fill, dtype=np.float32)
            canvas_alpha = np.zeros((H, W), dtype=np.float32)

            # Background — full canvas
            bg = _get_frame(bg_image, i, H, W)
            if bg is not None:
                canvas, canvas_alpha = _composite_layer(
                    canvas, canvas_alpha, bg,
                    _get_mask_frame(bg_mask, i, H, W),
                    bg_opacity, bg_blend_mode)

            # FX — full canvas
            fx = _get_frame(fx_image, i, H, W)
            if fx is not None:
                canvas, canvas_alpha = _composite_layer(
                    canvas, canvas_alpha, fx,
                    _get_mask_frame(fx_mask, i, H, W),
                    fx_opacity, fx_blend_mode)

            # Foreground — placed / fitted
            if fg_image is not None:
                fg_raw = fg_image[i % fg_image.shape[0]].cpu().numpy().astype(np.float32)
                if fg_raw.ndim == 2:
                    fg_raw = np.stack([fg_raw, fg_raw, fg_raw], axis=-1)
                elif fg_raw.shape[-1] == 4:
                    fg_raw = fg_raw[:, :, :3]

                src_h, src_w = fg_raw.shape[:2]

                # Build source-resolution mask
                if fg_mask is not None:
                    mi = i % fg_mask.shape[0]
                    fgm = (fg_mask[mi] if fg_mask.ndim == 3 else fg_mask).cpu().numpy().astype(np.float32)
                    if fgm.ndim > 2:
                        fgm = fgm[:, :, 0]
                    if fgm.shape != (src_h, src_w):
                        fgm = _pil_resize(fgm, src_h, src_w, is_mask=True)
                else:
                    fgm = np.ones((src_h, src_w), dtype=np.float32)

                placed_rgb, placed_mask = _fg_place(
                    fg_raw, fgm, H, W,
                    fg_fit_mode, fg_position,
                    int(fg_custom_w), int(fg_custom_h),
                    int(fg_custom_x), int(fg_custom_y),
                )
                canvas, canvas_alpha = _composite_layer(
                    canvas, canvas_alpha, placed_rgb, placed_mask,
                    fg_opacity, fg_blend_mode)

            if bg is None and fx is None and fg_image is None:
                canvas_alpha = np.ones((H, W), dtype=np.float32)

            frm_np = (np.dstack([canvas, canvas_alpha[..., np.newaxis]])
                      if output_format == "rgba" else canvas)

            chunk_frames.append(torch.from_numpy(frm_np.astype(np.float32)))
            chunk_alphas.append(torch.from_numpy(canvas_alpha.astype(np.float32)))

            if len(chunk_frames) >= CS:
                frame_list.append(torch.stack(chunk_frames, dim=0))
                alpha_list.append(torch.stack(chunk_alphas, dim=0))
                chunk_frames, chunk_alphas = [], []

            if (i+1) % 200 == 0 or i == N-1:
                logger.info(f"[S42P Layer Composer] {i+1}/{N} frames")

        if chunk_frames:
            frame_list.append(torch.stack(chunk_frames, dim=0))
            alpha_list.append(torch.stack(chunk_alphas, dim=0))

        composited = torch.cat(frame_list, dim=0)
        alpha_mask = torch.cat(alpha_list, dim=0)

        mb = composited.nbytes / 1024 / 1024
        print(f"[S42P Layer Composer] done: {N} frames  {W}×{H}  {mb:.1f} MB  "
              f"fg={fg_fit_mode}/{fg_position}  format={output_format}")
        return (composited, alpha_mask)

    def _parse_hex(self, hex_str: str) -> tuple:
        try:
            s = hex_str.strip().lstrip("#")
            if len(s) == 3:
                s = "".join(c*2 for c in s)
            return (int(s[0:2],16)/255.0, int(s[2:4],16)/255.0, int(s[4:6],16)/255.0)
        except Exception:
            return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "S42PLayerComposer": S42PLayerComposer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PLayerComposer": "S42P Layer Composer",
}