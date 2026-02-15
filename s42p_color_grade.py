"""
S42 Production Suite — Color Grade Node
========================================
Professional per-frame color grading for IMAGE batches (video or single images).

Three grading stages, applied in order:
  1. LUT       — 3D Look-Up Table (.cube file) for film/grade looks
  2. Curves    — Lift/Gamma/Gain per channel + master
  3. HSL       — Hue/Saturation/Lightness per color range (6 ranges)

Any stage can be bypassed independently.
Works on [B, H, W, C] float32 IMAGE tensors — fully batch-aware.

Python 3.12 | ComfyUI Portable | numpy + torch
"""

import torch
import numpy as np
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

CATEGORY = "S42 Production Suite 🎨 Color"

# ── LUT (.cube) parser ────────────────────────────────────────────────────────

def _parse_cube_lut(filepath: str) -> Optional[Tuple[np.ndarray, int]]:
    """
    Parse an Adobe/.cube 3D LUT file.
    Returns (lut_data [size, size, size, 3], size) or None on failure.
    """
    if not filepath or not os.path.isfile(filepath):
        return None

    size    = None
    data    = []
    domain_min = np.array([0.0, 0.0, 0.0])
    domain_max = np.array([1.0, 1.0, 1.0])

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.upper().startswith("LUT_3D_SIZE"):
                    size = int(line.split()[-1])
                elif line.upper().startswith("DOMAIN_MIN"):
                    domain_min = np.array([float(v) for v in line.split()[1:4]])
                elif line.upper().startswith("DOMAIN_MAX"):
                    domain_max = np.array([float(v) for v in line.split()[1:4]])
                else:
                    parts = line.split()
                    if len(parts) == 3:
                        try:
                            data.append([float(v) for v in parts])
                        except ValueError:
                            pass

        if size is None or len(data) != size ** 3:
            logger.warning(f"LUT parse failed — expected {size**3 if size else '?'} "
                           f"entries, got {len(data)}")
            return None

        # Shape: [size*size*size, 3] → [size, size, size, 3]
        lut = np.array(data, dtype=np.float32).reshape(size, size, size, 3)

        # Normalise domain if non-standard
        rng = domain_max - domain_min
        rng = np.where(rng == 0, 1.0, rng)
        lut = (lut - domain_min) / rng
        lut = lut.clip(0.0, 1.0)

        return lut, size

    except Exception as e:
        logger.error(f"LUT file read error: {e}")
        return None


def _apply_lut(img: np.ndarray, lut: np.ndarray, size: int,
               strength: float = 1.0) -> np.ndarray:
    """
    Apply a 3D LUT to a [H, W, 3] float32 image via trilinear interpolation.
    strength: 0.0 = no effect, 1.0 = full LUT.
    """
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3).clip(0.0, 1.0)

    # Scale to LUT coordinates
    coords = flat * (size - 1)
    lo     = np.floor(coords).astype(np.int32).clip(0, size - 2)
    hi     = lo + 1
    frac   = coords - lo   # [N, 3] — fractional parts

    # Trilinear interpolation — 8 corners of the LUT cube
    r0, g0, b0 = lo[:, 0],   lo[:, 1],   lo[:, 2]
    r1, g1, b1 = hi[:, 0],   hi[:, 1],   hi[:, 2]
    fr, fg, fb = frac[:, 0], frac[:, 1], frac[:, 2]

    def c(ri, gi, bi):
        return lut[ri, gi, bi]   # [N, 3]

    result = (
        c(r0,g0,b0) * (1-fr)[:,None] * (1-fg)[:,None] * (1-fb)[:,None] +
        c(r1,g0,b0) *    fr [:,None] * (1-fg)[:,None] * (1-fb)[:,None] +
        c(r0,g1,b0) * (1-fr)[:,None] *    fg [:,None] * (1-fb)[:,None] +
        c(r0,g0,b1) * (1-fr)[:,None] * (1-fg)[:,None] *    fb [:,None] +
        c(r1,g1,b0) *    fr [:,None] *    fg [:,None] * (1-fb)[:,None] +
        c(r1,g0,b1) *    fr [:,None] * (1-fg)[:,None] *    fb [:,None] +
        c(r0,g1,b1) * (1-fr)[:,None] *    fg [:,None] *    fb [:,None] +
        c(r1,g1,b1) *    fr [:,None] *    fg [:,None] *    fb [:,None]
    )

    result = result.reshape(h, w, 3).clip(0.0, 1.0).astype(np.float32)

    # Blend with original at strength
    if strength < 1.0:
        result = img * (1.0 - strength) + result * strength

    return result


# ── Curves / Levels ──────────────────────────────────────────────────────────

def _apply_curves(img: np.ndarray,
                  # Master
                  master_lift: float, master_gamma: float, master_gain: float,
                  # Red
                  r_lift: float, r_gamma: float, r_gain: float,
                  # Green
                  g_lift: float, g_gamma: float, g_gain: float,
                  # Blue
                  b_lift: float, b_gamma: float, b_gain: float,
                  # Contrast
                  contrast: float, contrast_pivot: float,
                  # Saturation
                  saturation: float) -> np.ndarray:
    """
    Apply lift/gamma/gain (ASC CDL-style) per channel + master.
    Then contrast and saturation.

    lift:    adds to shadows (negative = darker shadows)
    gamma:   midtone brightness (>1 = brighter mids)
    gain:    multiplies highlights (>1 = brighter highs)

    Formula per channel: output = (input * gain + lift) ^ (1/gamma)
    Then master applied on top.
    """
    out = img.astype(np.float32).copy()

    # Per-channel lift/gamma/gain
    gamma_floor = 0.001

    def _lgk(x, lift, gamma, gain):
        x = x * gain + lift
        x = x.clip(0.0, None)
        if abs(gamma - 1.0) > 0.001:
            x = np.power(x.clip(0.0, 1.0), 1.0 / max(gamma, gamma_floor))
        return x

    out[..., 0] = _lgk(out[..., 0], r_lift, r_gamma, r_gain)
    out[..., 1] = _lgk(out[..., 1], g_lift, g_gamma, g_gain)
    out[..., 2] = _lgk(out[..., 2], b_lift, b_gamma, b_gain)

    # Master lift/gamma/gain on top
    out = _lgk(out, master_lift, master_gamma, master_gain)

    # Contrast around pivot
    if abs(contrast - 1.0) > 0.001:
        out = (out - contrast_pivot) * contrast + contrast_pivot

    # Saturation (luminance-preserving)
    if abs(saturation - 1.0) > 0.001:
        luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        luma = luma[..., np.newaxis]
        out  = luma + (out - luma) * saturation

    return out.clip(0.0, 1.0).astype(np.float32)


# ── HSL Grading ──────────────────────────────────────────────────────────────

def _rgb_to_hsl(img: np.ndarray) -> np.ndarray:
    """[H,W,3] RGB float → [H,W,3] HSL float (H: 0-360, S: 0-1, L: 0-1)"""
    r, g, b = img[...,0], img[...,1], img[...,2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Lightness
    L = (cmax + cmin) * 0.5

    # Saturation
    S = np.where(delta == 0, 0.0, delta / (1.0 - np.abs(2.0 * L - 1.0) + 1e-10))

    # Hue
    H = np.zeros_like(r)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    H[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    H[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
    H[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

    return np.stack([H, S.clip(0,1), L.clip(0,1)], axis=-1)


def _hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """[H,W,3] HSL → [H,W,3] RGB float"""
    H, S, L = hsl[...,0], hsl[...,1], hsl[...,2]
    C = (1.0 - np.abs(2.0 * L - 1.0)) * S
    X = C * (1.0 - np.abs((H / 60.0) % 2.0 - 1.0))
    m = L - C * 0.5

    R = np.zeros_like(H); G = np.zeros_like(H); B = np.zeros_like(H)
    for cond, rv, gv, bv in [
        ((H < 60),              C, X, 0),
        ((H >= 60)  & (H<120), X, C, 0),
        ((H >= 120) & (H<180), 0, C, X),
        ((H >= 180) & (H<240), 0, X, C),
        ((H >= 240) & (H<300), X, 0, C),
        ((H >= 300),            C, 0, X),
    ]:
        R[cond] = rv if np.isscalar(rv) else rv[cond]
        G[cond] = gv if np.isscalar(gv) else gv[cond]
        B[cond] = bv if np.isscalar(bv) else bv[cond]

    return np.stack([R+m, G+m, B+m], axis=-1).clip(0.0, 1.0).astype(np.float32)


# HSL color range center hues (degrees) and half-widths
_HSL_RANGES = {
    "reds":    (0.0,   35.0),
    "yellows": (60.0,  25.0),
    "greens":  (120.0, 35.0),
    "cyans":   (180.0, 25.0),
    "blues":   (240.0, 35.0),
    "magentas":(300.0, 35.0),
}


def _hue_weight(H: np.ndarray, center: float, half_width: float) -> np.ndarray:
    """Soft mask for pixels whose hue falls within a colour range."""
    diff = np.abs(((H - center + 180) % 360) - 180)
    weight = (1.0 - diff / half_width).clip(0.0, 1.0)
    return weight ** 2   # smooth falloff


def _apply_hsl(img: np.ndarray,
               reds_h: float,    reds_s: float,    reds_l: float,
               yellows_h: float, yellows_s: float, yellows_l: float,
               greens_h: float,  greens_s: float,  greens_l: float,
               cyans_h: float,   cyans_s: float,   cyans_l: float,
               blues_h: float,   blues_s: float,   blues_l: float,
               magentas_h: float,magentas_s: float,magentas_l: float) -> np.ndarray:
    """Apply HSL adjustments per color range."""
    hsl = _rgb_to_hsl(img)

    adjustments = {
        "reds":     (reds_h,     reds_s,     reds_l),
        "yellows":  (yellows_h,  yellows_s,  yellows_l),
        "greens":   (greens_h,   greens_s,   greens_l),
        "cyans":    (cyans_h,    cyans_s,    cyans_l),
        "blues":    (blues_h,    blues_s,    blues_l),
        "magentas": (magentas_h, magentas_s, magentas_l),
    }

    H = hsl[..., 0]

    for name, (dh, ds, dl) in adjustments.items():
        if abs(dh) < 0.01 and abs(ds) < 0.01 and abs(dl) < 0.01:
            continue   # skip unchanged ranges for speed
        center, half_w = _HSL_RANGES[name]
        w = _hue_weight(H, center, half_w)[..., np.newaxis]
        hsl[..., 0:1] += w * dh
        hsl[..., 1:2]  = (hsl[..., 1:2] + w * ds).clip(0, 1)
        hsl[..., 2:3]  = (hsl[..., 2:3] + w * dl).clip(0, 1)

    hsl[..., 0] = hsl[..., 0] % 360.0
    return _hsl_to_rgb(hsl)


# ── Node ─────────────────────────────────────────────────────────────────────

class S42PColorGrade:
    """
    🎨 S42P Color Grade
    Three-stage professional color grading for video or images.

    Signal flow: Input → LUT → Curves/Levels → HSL → Output
    Each stage can be bypassed independently.

    Works on both single images and video batches.
    Connect after your animation/compositing chain, before export.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Reusable tooltip builders
        def lgk_tip(ch, param):
            tips = {
                "lift":  f"Lift (shadows) for {ch} channel. Positive = brighter shadows. Negative = crush blacks. 0.0 = neutral.",
                "gamma": f"Gamma (midtones) for {ch} channel. >1.0 = brighter mids. <1.0 = darker mids. 1.0 = neutral.",
                "gain":  f"Gain (highlights) for {ch} channel. >1.0 = brighter highlights. <1.0 = darker highlights. 1.0 = neutral.",
            }
            return tips[param]

        def hsl_tip(color, param):
            tips = {
                "hue": (f"Hue rotation for {color} (degrees). "
                        f"Positive = rotate hue clockwise. Negative = counter-clockwise. "
                        f"0 = unchanged. ±30° is a strong shift."),
                "sat": (f"Saturation adjustment for {color}. "
                        f"Positive = more vivid. Negative = desaturate toward grey. "
                        f"0 = unchanged. ±0.3 is a noticeable change."),
                "lum": (f"Lightness adjustment for {color}. "
                        f"Positive = lighter. Negative = darker. "
                        f"0 = unchanged."),
            }
            return tips[param]

        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": ("Input image or video batch to color grade. "
                                "Accepts any IMAGE output — single frames or full video batches. "
                                "Connect at the end of your compositing chain before export.")
                }),

                # ── LUT Stage ─────────────────────────────────────────────────
                "lut_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": ("Enable 3D LUT application. "
                                "LUT is applied first, before curves and HSL. "
                                "Disable to skip LUT entirely.")
                }),
                "lut_file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": ("Full path to a .cube LUT file. "
                                "Example: C:/LUTs/Kodak2383.cube\n"
                                "Adobe .cube format supported (3D LUTs only). "
                                "Free LUTs: search 'free .cube LUT download'. "
                                "Film LUTs: Kodak, Fuji, LOG-to-REC709 conversions, etc.")
                }),
                "lut_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0,
                    "step": 0.01, "display": "slider",
                    "tooltip": ("LUT blend strength. 1.0 = full LUT applied. "
                                "0.5 = half LUT, half original (subtle look). "
                                "0.0 = LUT disabled (same as unchecking lut_enabled). "
                                "Useful for taming strong film LUTs.")
                }),

                # ── Curves Stage ──────────────────────────────────────────────
                "curves_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable curves/levels adjustment. Applied after LUT, before HSL."
                }),

                # Master
                "master_lift":  ("FLOAT", {"default": 0.0,  "min": -0.5, "max": 0.5,  "step": 0.005, "display": "slider", "tooltip": lgk_tip("Master","lift")}),
                "master_gamma": ("FLOAT", {"default": 1.0,  "min": 0.1,  "max": 3.0,  "step": 0.01,  "display": "slider", "tooltip": lgk_tip("Master","gamma")}),
                "master_gain":  ("FLOAT", {"default": 1.0,  "min": 0.1,  "max": 3.0,  "step": 0.01,  "display": "slider", "tooltip": lgk_tip("Master","gain")}),
                "contrast":     ("FLOAT", {"default": 1.0,  "min": 0.5,  "max": 2.0,  "step": 0.01,  "display": "slider",
                    "tooltip": ("Contrast multiplier around the pivot point. "
                                ">1.0 = more contrast (punchier). "
                                "<1.0 = lower contrast (flat/faded). "
                                "1.0 = unchanged.")}),
                "contrast_pivot": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,  "display": "slider",
                    "tooltip": ("The luminance value around which contrast is applied. "
                                "0.5 = standard midpoint. "
                                "Lower values push contrast into shadows. "
                                "Higher values push contrast into highlights.")}),
                "saturation":   ("FLOAT", {"default": 1.0,  "min": 0.0,  "max": 3.0,  "step": 0.01,  "display": "slider",
                    "tooltip": ("Global saturation. 1.0 = unchanged. "
                                "0.0 = full greyscale. "
                                "1.5 = 50% more vivid. "
                                "Uses luminance-preserving method.")}),

                # Red channel
                "r_lift":  ("FLOAT", {"default": 0.0, "min": -0.3, "max": 0.3, "step": 0.005, "display": "slider", "tooltip": lgk_tip("Red","lift")}),
                "r_gamma": ("FLOAT", {"default": 1.0, "min": 0.1,  "max": 3.0, "step": 0.01,  "display": "slider", "tooltip": lgk_tip("Red","gamma")}),
                "r_gain":  ("FLOAT", {"default": 1.0, "min": 0.1,  "max": 3.0, "step": 0.01,  "display": "slider", "tooltip": lgk_tip("Red","gain")}),

                # Green channel
                "g_lift":  ("FLOAT", {"default": 0.0, "min": -0.3, "max": 0.3, "step": 0.005, "display": "slider", "tooltip": lgk_tip("Green","lift")}),
                "g_gamma": ("FLOAT", {"default": 1.0, "min": 0.1,  "max": 3.0, "step": 0.01,  "display": "slider", "tooltip": lgk_tip("Green","gamma")}),
                "g_gain":  ("FLOAT", {"default": 1.0, "min": 0.1,  "max": 3.0, "step": 0.01,  "display": "slider", "tooltip": lgk_tip("Green","gain")}),

                # Blue channel
                "b_lift":  ("FLOAT", {"default": 0.0, "min": -0.3, "max": 0.3, "step": 0.005, "display": "slider", "tooltip": lgk_tip("Blue","lift")}),
                "b_gamma": ("FLOAT", {"default": 1.0, "min": 0.1,  "max": 3.0, "step": 0.01,  "display": "slider", "tooltip": lgk_tip("Blue","gamma")}),
                "b_gain":  ("FLOAT", {"default": 1.0, "min": 0.1,  "max": 3.0, "step": 0.01,  "display": "slider", "tooltip": lgk_tip("Blue","gain")}),

                # ── HSL Stage ─────────────────────────────────────────────────
                "hsl_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable HSL per-color-range adjustment. Applied last, after curves."
                }),

                # Reds
                "reds_hue": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "display": "slider", "tooltip": hsl_tip("reds","hue")}),
                "reds_sat": ("FLOAT", {"default": 0.0, "min": -1.0,   "max": 1.0,   "step": 0.01,"display": "slider", "tooltip": hsl_tip("reds","sat")}),
                "reds_lum": ("FLOAT", {"default": 0.0, "min": -0.5,   "max": 0.5,   "step": 0.01,"display": "slider", "tooltip": hsl_tip("reds","lum")}),

                # Yellows
                "yellows_hue": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "display": "slider", "tooltip": hsl_tip("yellows","hue")}),
                "yellows_sat": ("FLOAT", {"default": 0.0, "min": -1.0,   "max": 1.0,   "step": 0.01,"display": "slider", "tooltip": hsl_tip("yellows","sat")}),
                "yellows_lum": ("FLOAT", {"default": 0.0, "min": -0.5,   "max": 0.5,   "step": 0.01,"display": "slider", "tooltip": hsl_tip("yellows","lum")}),

                # Greens
                "greens_hue": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "display": "slider", "tooltip": hsl_tip("greens","hue")}),
                "greens_sat": ("FLOAT", {"default": 0.0, "min": -1.0,   "max": 1.0,   "step": 0.01,"display": "slider", "tooltip": hsl_tip("greens","sat")}),
                "greens_lum": ("FLOAT", {"default": 0.0, "min": -0.5,   "max": 0.5,   "step": 0.01,"display": "slider", "tooltip": hsl_tip("greens","lum")}),

                # Cyans
                "cyans_hue": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "display": "slider", "tooltip": hsl_tip("cyans","hue")}),
                "cyans_sat": ("FLOAT", {"default": 0.0, "min": -1.0,   "max": 1.0,   "step": 0.01,"display": "slider", "tooltip": hsl_tip("cyans","sat")}),
                "cyans_lum": ("FLOAT", {"default": 0.0, "min": -0.5,   "max": 0.5,   "step": 0.01,"display": "slider", "tooltip": hsl_tip("cyans","lum")}),

                # Blues
                "blues_hue": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "display": "slider", "tooltip": hsl_tip("blues","hue")}),
                "blues_sat": ("FLOAT", {"default": 0.0, "min": -1.0,   "max": 1.0,   "step": 0.01,"display": "slider", "tooltip": hsl_tip("blues","sat")}),
                "blues_lum": ("FLOAT", {"default": 0.0, "min": -0.5,   "max": 0.5,   "step": 0.01,"display": "slider", "tooltip": hsl_tip("blues","lum")}),

                # Magentas
                "magentas_hue": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "display": "slider", "tooltip": hsl_tip("magentas","hue")}),
                "magentas_sat": ("FLOAT", {"default": 0.0, "min": -1.0,   "max": 1.0,   "step": 0.01,"display": "slider", "tooltip": hsl_tip("magentas","sat")}),
                "magentas_lum": ("FLOAT", {"default": 0.0, "min": -0.5,   "max": 0.5,   "step": 0.01,"display": "slider", "tooltip": hsl_tip("magentas","lum")}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("graded_images",)
    FUNCTION      = "grade"
    CATEGORY      = CATEGORY

    def grade(self, images: torch.Tensor,
              lut_enabled: bool, lut_file_path: str, lut_strength: float,
              curves_enabled: bool,
              master_lift: float, master_gamma: float, master_gain: float,
              contrast: float, contrast_pivot: float, saturation: float,
              r_lift: float, r_gamma: float, r_gain: float,
              g_lift: float, g_gamma: float, g_gain: float,
              b_lift: float, b_gamma: float, b_gain: float,
              hsl_enabled: bool,
              reds_hue: float,    reds_sat: float,    reds_lum: float,
              yellows_hue: float, yellows_sat: float, yellows_lum: float,
              greens_hue: float,  greens_sat: float,  greens_lum: float,
              cyans_hue: float,   cyans_sat: float,   cyans_lum: float,
              blues_hue: float,   blues_sat: float,   blues_lum: float,
              magentas_hue: float,magentas_sat: float,magentas_lum: float
              ) -> Tuple[torch.Tensor]:

        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] == 4:
            images = images[..., :3]

        # Load LUT once if enabled
        lut_data = None
        if lut_enabled and lut_file_path.strip():
            result = _parse_cube_lut(lut_file_path.strip())
            if result:
                lut_data, lut_size = result
                logger.info(f"LUT loaded: {os.path.basename(lut_file_path)} ({lut_size}³)")
            else:
                logger.warning(f"LUT failed to load: {lut_file_path}")

        arr = images.cpu().numpy().astype(np.float32)
        out = np.empty_like(arr)

        for i in range(arr.shape[0]):
            frame = arr[i]   # [H, W, 3]

            # Stage 1: LUT
            if lut_enabled and lut_data is not None:
                frame = _apply_lut(frame, lut_data, lut_size, lut_strength)

            # Stage 2: Curves
            if curves_enabled:
                frame = _apply_curves(
                    frame,
                    master_lift, master_gamma, master_gain,
                    r_lift, r_gamma, r_gain,
                    g_lift, g_gamma, g_gain,
                    b_lift, b_gamma, b_gain,
                    contrast, contrast_pivot, saturation
                )

            # Stage 3: HSL
            if hsl_enabled:
                frame = _apply_hsl(
                    frame,
                    reds_hue,    reds_sat,    reds_lum,
                    yellows_hue, yellows_sat, yellows_lum,
                    greens_hue,  greens_sat,  greens_lum,
                    cyans_hue,   cyans_sat,   cyans_lum,
                    blues_hue,   blues_sat,   blues_lum,
                    magentas_hue,magentas_sat,magentas_lum,
                )

            out[i] = frame.clip(0.0, 1.0)

        return (torch.from_numpy(out),)


# ── ComfyUI registration ──────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "S42PColorGrade": S42PColorGrade,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PColorGrade": "🎨 S42P Color Grade",
}
