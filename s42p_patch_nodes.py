"""
S42 Production Suite ? Patch Lift & Drop
=========================================
Patch workflow: extract a region ? edit it ? replace it exactly where it was.

S42P PATCH LIFT  ? Extract a rectangular region from an image or video batch.
                   Outputs the patch AND stores original coordinates for Drop.

S42P PATCH DROP  ? Place a (possibly modified) patch back at the exact pixel
                   coordinates it was lifted from, with optional blend modes
                   and feathering for seamless integration.

WORKFLOW:
  Image/Video ? [S42P Patch Lift] ? patch ? [your nodes] ? [S42P Patch Drop]
                       ?  coord_data ?????????????????????????????? ?
                       ?? original_image ?????????????????????????? ?

IMPROVED vs Studio42 originals:
  ? Coord data carried as a compact JSON STRING (no separate INT outputs)
  ? Video batch handling with consistent coords across all frames
  ? Blend modes: replace, overlay, screen, multiply, soft_light
  ? Feather edge blending baked into Drop node (no separate mask needed)
  ? Handles out-of-bounds gracefully (clips to image boundary)

Python 3.12 | ComfyUI Portable | torch + Pillow
"""

from __future__ import annotations

import json
import logging
import math
import numpy as np
import torch
from typing import Optional

try:
    from PIL import Image, ImageFilter, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)
CATEGORY = "S42 Production Suite/Utilities"


# ?? Helpers ????????????????????????????????????????????????????????????????

def _t2np(t: torch.Tensor) -> np.ndarray:
    """[B,H,W,C] or [H,W,C] torch float32 ? numpy uint8."""
    arr = t.cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]  # take first in batch
    return (arr * 255).clip(0, 255).astype(np.uint8)


def _np2t(arr: np.ndarray) -> torch.Tensor:
    """HWC uint8 ? [1,H,W,C] float32 tensor."""
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def _blend(base: np.ndarray, patch: np.ndarray, mode: str,
           alpha: float) -> np.ndarray:
    """Blend patch onto base (HWC uint8). Returns HWC uint8."""
    b = base.astype(np.float32) / 255.0
    p = patch.astype(np.float32) / 255.0
    if mode == "replace":
        blended = p
    elif mode == "multiply":
        blended = b * p
    elif mode == "screen":
        blended = 1.0 - (1.0 - b) * (1.0 - p)
    elif mode == "overlay":
        mask    = b < 0.5
        blended = np.where(mask, 2 * b * p, 1 - 2 * (1 - b) * (1 - p))
    elif mode == "soft_light":
        blended = (1 - 2 * p) * b ** 2 + 2 * p * b
    else:
        blended = p

    result = b * (1.0 - alpha) + blended * alpha
    return (result * 255).clip(0, 255).astype(np.uint8)


def _make_feather_mask(w: int, h: int, feather: int) -> np.ndarray:
    """Create a float32 [H,W] feather mask (1=opaque centre, 0=transparent edge)."""
    mask = np.ones((h, w), dtype=np.float32)
    if feather <= 0:
        return mask
    for i in range(feather):
        val  = (i + 1) / (feather + 1)
        mask[i,    :] = np.minimum(mask[i,    :],  val)
        mask[h-1-i,:] = np.minimum(mask[h-1-i,:], val)
        mask[:,    i] = np.minimum(mask[:,    i],  val)
        mask[:, w-1-i] = np.minimum(mask[:, w-1-i], val)
    return mask


# ?? PatchLift ?????????????????????????????????????????????????????????????

class S42PPatchLift:
    """
    S42P Patch Lift ? extract a rectangular region from an image or video batch.

    The coord_data output (JSON STRING) stores all the information needed
    to place the patch back precisely with S42P Patch Drop.  Always wire
    coord_data through to the Drop node.

    For video: all frames are cropped at the same coordinates.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Source image or video batch [B,H,W,C]."
                }),
                "x": ("INT", {
                    "default": 128, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "Left edge of the patch region (pixels)."
                }),
                "y": ("INT", {
                    "default": 128, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "Top edge of the patch region (pixels)."
                }),
                "width": ("INT", {
                    "default": 256, "min": 8, "max": 4096, "step": 8,
                    "tooltip": "Width of the patch region (pixels)."
                }),
                "height": ("INT", {
                    "default": 256, "min": 8, "max": 4096, "step": 8,
                    "tooltip": "Height of the patch region (pixels)."
                }),
            },
            "optional": {
                "pad_to_multiple": ("INT", {
                    "default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": (
                        "Pad patch dimensions to a multiple of this value. "
                        "Set to 8 or 64 for compatibility with VAE/latent nodes."
                    )
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES  = ("patch", "original_image", "coord_data")
    FUNCTION      = "lift"
    CATEGORY      = CATEGORY

    def lift(self, image: torch.Tensor, x: int, y: int,
              width: int, height: int, pad_to_multiple: int = 8) -> tuple:

        B, H, W, C = image.shape

        # Clamp to image bounds
        x      = max(0, min(x,      W - 1))
        y      = max(0, min(y,      H - 1))
        width  = max(8, min(width,  W - x))
        height = max(8, min(height, H - y))

        # Pad to multiple (for VAE compatibility)
        if pad_to_multiple > 1:
            pw = math.ceil(width  / pad_to_multiple) * pad_to_multiple
            ph = math.ceil(height / pad_to_multiple) * pad_to_multiple
            # Clamp padded dimensions
            pw = min(pw, W - x)
            ph = min(ph, H - y)
        else:
            pw, ph = width, height

        # Extract patch batch
        patch = image[:, y:y+ph, x:x+pw, :].clone()

        # Encode coordinates for Drop node
        coord_data = json.dumps({
            "x": x, "y": y,
            "width": pw, "height": ph,
            "original_width": W,
            "original_height": H,
        })

        print(f"[S42P PatchLift] Extracted {pw}x{ph} patch at ({x},{y}) "
              f"from {W}x{H} source, batch={B}")

        return (patch, image, coord_data)


# ?? PatchDrop ?????????????????????????????????????????????????????????????

class S42PPatchDrop:
    """
    S42P Patch Drop ? composite a (possibly modified) patch back onto the
    original image at the exact pixel coordinates stored in coord_data.

    BLEND MODES:
      replace     ? Direct pixel replacement (default)
      multiply    ? Darkening blend
      screen      ? Lightening blend
      overlay     ? Contrast-boosting blend
      soft_light  ? Gentle soft-light blend

    FEATHER EDGE: blends the patch edges smoothly into the background.
    Works on both single images and video batches.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {
                    "tooltip": "Original image from S42P Patch Lift (original_image output)."
                }),
                "patch": ("IMAGE", {
                    "tooltip": "Modified patch to drop back. Can be any size ? will be resized."
                }),
                "coord_data": ("STRING", {
                    "forceInput": True,
                    "tooltip": "JSON coord_data from S42P Patch Lift."
                }),
            },
            "optional": {
                "blend_mode": (["replace", "multiply", "screen", "overlay", "soft_light"], {
                    "default": "replace",
                    "tooltip": "Blend mode for compositing patch onto original."
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Patch opacity (1.0 = fully opaque)."
                }),
                "feather_edge": ("INT", {
                    "default": 4, "min": 0, "max": 64, "step": 1,
                    "tooltip": (
                        "Feather/blend pixels at patch edges for seamless integration. "
                        "Higher = softer edge. 0 = hard edge."
                    )
                }),
                "patch_mask": ("MASK", {
                    "tooltip": "(Optional) External mask to control patch transparency."
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "drop"
    CATEGORY      = CATEGORY

    def drop(self, original_image: torch.Tensor, patch: torch.Tensor,
              coord_data: str,
              blend_mode: str = "replace", opacity: float = 1.0,
              feather_edge: int = 4,
              patch_mask: Optional[torch.Tensor] = None) -> tuple:

        # Parse coord_data
        try:
            coords = json.loads(coord_data)
            cx     = int(coords["x"])
            cy     = int(coords["y"])
            cw     = int(coords["width"])
            ch     = int(coords["height"])
        except Exception as e:
            logger.error(f"[S42P PatchDrop] Invalid coord_data: {e}")
            return (original_image,)

        B, H, W, C = original_image.shape
        result      = original_image.clone()

        # Clamp to image bounds
        x2 = min(cx + cw, W)
        y2 = min(cy + ch, H)
        aw = x2 - cx  # actual width to paste
        ah = y2 - cy  # actual height to paste

        if aw <= 0 or ah <= 0:
            logger.warning("[S42P PatchDrop] Patch region out of bounds, skipping.")
            return (result,)

        for b in range(B):
            # Get original slice as numpy
            orig_slice = (result[b, cy:y2, cx:x2, :].cpu().float().numpy() * 255).astype(np.uint8)

            # Get patch frame (wrap if patch batch < original batch)
            pi  = b % patch.shape[0]
            p_t = patch[pi]  # [H,W,C]

            # Resize patch to target region if needed
            if p_t.shape[0] != ah or p_t.shape[1] != aw:
                if PIL_AVAILABLE:
                    p_img  = Image.fromarray(
                        (p_t.cpu().float().numpy() * 255).clip(0,255).astype(np.uint8)
                    )
                    p_img  = p_img.resize((aw, ah), Image.LANCZOS)
                    p_np   = np.array(p_img)
                else:
                    # torch interpolate fallback
                    p_4d   = p_t.unsqueeze(0).permute(0,3,1,2).float()
                    p_4d   = torch.nn.functional.interpolate(
                        p_4d, size=(ah, aw), mode="bilinear", align_corners=False
                    )
                    p_np   = (p_4d[0].permute(1,2,0).numpy() * 255).astype(np.uint8)
            else:
                p_np = (p_t.cpu().float().numpy() * 255).clip(0,255).astype(np.uint8)

            # Ensure 3-channel
            if p_np.shape[2] == 4:
                p_np = p_np[..., :3]
            if orig_slice.shape[2] == 4:
                orig_slice = orig_slice[..., :3]

            # Apply external mask if provided
            eff_opacity = opacity
            if patch_mask is not None:
                mi  = b % patch_mask.shape[0]
                m_t = patch_mask[mi]
                if m_t.shape[0] != ah or m_t.shape[1] != aw:
                    m_4d = m_t.unsqueeze(0).unsqueeze(0).float()
                    m_4d = torch.nn.functional.interpolate(
                        m_4d, size=(ah, aw), mode="bilinear", align_corners=False
                    )
                    m_np = m_4d[0,0].numpy()
                else:
                    m_np = m_t.cpu().float().numpy()
                eff_opacity = float(np.mean(m_np)) * opacity

            # Feather mask
            feather_m = _make_feather_mask(aw, ah, feather_edge)

            # Blend
            blended = _blend(orig_slice, p_np, blend_mode, eff_opacity)

            # Apply feather: lerp between original and blended
            feather_3 = feather_m[:, :, np.newaxis]
            final     = (orig_slice.astype(np.float32) * (1 - feather_3) +
                         blended.astype(np.float32) * feather_3).clip(0, 255).astype(np.uint8)

            # Write back
            result[b, cy:y2, cx:x2, :C] = torch.from_numpy(
                final.astype(np.float32) / 255.0
            )

        print(f"[S42P PatchDrop] Placed {aw}x{ah} patch at ({cx},{cy}) "
              f"mode={blend_mode} opacity={opacity:.2f} feather={feather_edge}")

        return (result,)


NODE_CLASS_MAPPINGS = {
    "S42PPatchLift": S42PPatchLift,
    "S42PPatchDrop":  S42PPatchDrop,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PPatchLift": "S42P Patch Lift ??",
    "S42PPatchDrop":  "S42P Patch Drop ?",
}
