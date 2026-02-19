"""
S42 Production Suite — Audio Reactive FX v5.0
==============================================
Apply audio-reactive visual effects to image/video frames.

EFFECTS:
  pulse     — scale/zoom pulse with audio envelope
  neon      — glowing edge highlight driven by audio
  cascade   — echo/shadow copies offset by envelope
  flicker   — color flash overlay on beats
  rotation  — rotate frame by envelope amount
  warp      — TRUE radial warp via cv2.remap polar displacement map

CHANGES v5.0:
  - FIX: warp now uses cv2.remap with polar displacement map
         (was np.roll = horizontal shift, not a real warp)
  - NEW: bypass toggle
  - NEW: chromatic_aberration effect (RGB channel split on beats)
  - NEW: zoom_blur effect (radial motion blur toward centre)
  - IMPROVED: All effects optionally audio-reactive via external envelope

Python 3.12 | ComfyUI Portable
"""

import math
import numpy as np
import torch
import cv2
import logging

logger = logging.getLogger(__name__)

try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False


# ── Audio helpers ──────────────────────────────────────────────────────────────

def _to_numpy_mono(audio_any):
    y = None; sr = None
    if isinstance(audio_any, dict):
        for k in ("samples","waveform","audio","y"):
            if k in audio_any: y = audio_any[k]; break
        for k in ("sample_rate","sr","rate"):
            if k in audio_any: sr = audio_any[k]; break
    elif isinstance(audio_any, (tuple, list)) and len(audio_any) >= 2:
        y, sr = audio_any[0], audio_any[1]
    else:
        y = audio_any

    if isinstance(sr, (list, tuple)) and len(sr) > 0: sr = sr[0]
    if isinstance(sr, torch.Tensor): sr = sr.detach().cpu().numpy()
    if isinstance(sr, np.ndarray):   sr = float(sr.flatten()[0]) if sr.size else 44100.0
    if sr is None: sr = 44100.0
    try: sr = int(round(float(sr)))
    except Exception: sr = 44100

    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
    y = np.array(y)

    if y.ndim == 2:
        if y.shape[0] <= 8:  y = y.mean(axis=0)
        elif y.shape[1] <= 8: y = y.mean(axis=1)
        else: y = y.flatten()
    elif y.ndim > 2:
        y = y.flatten()

    y = y.astype(np.float32, copy=False)
    if y.size and np.max(np.abs(y)) > 1.5:
        y = y / max(32768.0, float(np.max(np.abs(y))))
    if y.size == 0: y = np.zeros(1024, dtype=np.float32)
    return y, sr


def _rms_env(y: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 0: return np.zeros(1, dtype=np.float32)
    hop = max(1, len(y) // frames)
    env = np.array([
        float(np.sqrt(np.mean(y[i*hop:(i+1)*hop]**2)))
        for i in range(frames)
    ], dtype=np.float32)
    mx = env.max()
    return env / mx if mx > 1e-8 else env


# ── Effect helpers ─────────────────────────────────────────────────────────────

def _mask_for_frame(mask, idx, w, h):
    if mask is None: return None
    n = mask.shape[0] if hasattr(mask, 'shape') else 1
    m = mask[min(idx, n-1)].detach().cpu().numpy() if hasattr(mask, 'detach') else np.array(mask[min(idx, n-1)])
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(m.astype(np.float32), 0.0, 1.0)


def _hex_to_bgr(hex_color: str):
    h = hex_color.lstrip('#')
    if len(h) == 3: h = ''.join(c*2 for c in h)
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)


def _scale_frame(img, scale, quality_mode):
    if abs(scale - 1.0) < 0.001: return img
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    interp = cv2.INTER_LINEAR if quality_mode == "fast" else cv2.INTER_CUBIC
    scaled = cv2.resize(img, (new_w, new_h), interpolation=interp)
    if scale > 1.0:
        x1 = (new_w - w) // 2; y1 = (new_h - h) // 2
        return scaled[y1:y1+h, x1:x1+w]
    else:
        out = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
        x1 = (w - new_w) // 2; y1 = (h - new_h) // 2
        out[y1:y1+new_h, x1:x1+new_w] = scaled
        return out


def _scale_gray(img, scale, quality_mode):
    if img is None or abs(scale - 1.0) < 0.001: return img
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    interp = cv2.INTER_LINEAR if quality_mode == "fast" else cv2.INTER_CUBIC
    scaled = cv2.resize(img.astype(np.float32), (new_w, new_h), interpolation=interp)
    if scale > 1.0:
        x1 = (new_w - w) // 2; y1 = (new_h - h) // 2
        return scaled[y1:y1+h, x1:x1+w].astype(np.uint8)
    else:
        out = np.zeros((h, w), dtype=np.uint8)
        x1 = (w - new_w) // 2; y1 = (h - new_h) // 2
        out[y1:y1+new_h, x1:x1+new_w] = np.clip(scaled, 0, 255).astype(np.uint8)
        return out


def _apply_neon(frame_bgr, mask_np, hex_color, thickness, amp, intensity, quality_mode):
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur_k = max(3, int(thickness * 2) | 1)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    edges  = cv2.subtract(gray, blurred)
    color  = _hex_to_bgr(hex_color)
    layer  = np.zeros_like(frame_bgr)
    layer[:] = color
    alpha  = float(np.clip(amp * intensity, 0.0, 1.0))
    if mask_np is not None:
        mask_3d = np.stack([mask_np]*3, axis=-1)
        layer   = (layer * mask_3d).astype(np.uint8)
    result = cv2.addWeighted(frame_bgr, 1.0, layer, alpha, 0)
    if quality_mode == "high":
        result = cv2.GaussianBlur(result, (0, 0), 0.75 + 1.5*amp)
    return result


def _apply_cascade(frame_bgr, alpha, mask_np, spread, copies, amp, intensity):
    base_img = frame_bgr.copy()
    base_a   = alpha.copy() if alpha is not None else None
    for i in range(1, copies+1):
        offset = int(round(spread * amp * i))
        shifted = np.roll(frame_bgr, shift=offset, axis=1)
        la = (0.5/i) * amp * intensity
        if mask_np is not None:
            am = (mask_np * la).astype(np.float32)
            base_img = (base_img.astype(np.float32)*(1-am[...,None]) +
                        shifted.astype(np.float32)*(am[...,None])).astype(np.uint8)
        else:
            base_img = cv2.addWeighted(base_img, 1.0, shifted, la, 0)
        if base_a is not None:
            sa = np.roll(alpha, shift=offset, axis=1)
            if mask_np is not None:
                am = (mask_np * la).astype(np.float32)
                base_a = (base_a.astype(np.float32)*(1-am) + sa.astype(np.float32)*am).astype(np.uint8)
            else:
                base_a = np.maximum(base_a, (sa * np.clip(la,0,1)).astype(np.uint8))
    return base_img, base_a


def _apply_flicker(frame_bgr, hex_color, strength, amp, mask_np=None):
    color   = _hex_to_bgr(hex_color)
    overlay = np.full_like(frame_bgr, color)
    a       = float(strength) * float(amp)
    if a <= 1e-4: return frame_bgr
    if mask_np is None:
        return cv2.addWeighted(frame_bgr, 1.0, overlay, a, 0)
    out = frame_bgr.astype(np.float32)
    ov  = overlay.astype(np.float32)
    am  = (mask_np * a).astype(np.float32)
    return (out*(1-am[...,None]) + ov*(am[...,None])).astype(np.uint8)


def _rotate_frame(img, deg, quality_mode):
    if abs(deg) < 0.01: return img
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    flags = cv2.INTER_LINEAR if quality_mode == "fast" else cv2.INTER_CUBIC
    return cv2.warpAffine(img, M, (w, h), flags=flags)


def _rotate_gray(img, deg, quality_mode):
    if img is None or abs(deg) < 0.01: return img
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    flags = cv2.INTER_LINEAR if quality_mode == "fast" else cv2.INTER_CUBIC
    return cv2.warpAffine(img.astype(np.float32), M, (w, h), flags=flags).astype(np.uint8)


def _build_warp_maps(h: int, w: int) -> tuple:
    """Pre-build normalised coordinate grids for radial warp — cached per (h,w)."""
    cx, cy = w / 2.0, h / 2.0
    xs     = np.arange(w, dtype=np.float32)
    ys     = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    dx  = xg - cx
    dy  = yg - cy
    r   = np.sqrt(dx**2 + dy**2)          # radius from centre
    th  = np.arctan2(dy, dx)              # angle
    return cx, cy, r, th, dx, dy


_WARP_CACHE: dict = {}


def _warp(frame_bgr: np.ndarray, intensity: float, amp: float) -> np.ndarray:
    """
    True radial warp using cv2.remap with polar displacement.

    Each pixel is displaced radially — pixels near the centre move inward,
    pixels near edges move outward (or vice-versa depending on amp sign).
    This creates a genuine lens-distortion / ripple warp, not a horizontal shift.

    intensity: max pixel displacement in proportion to image diagonal
    amp: 0..1 audio envelope strength
    """
    h, w = frame_bgr.shape[:2]
    key  = (h, w)
    if key not in _WARP_CACHE:
        _WARP_CACHE[key] = _build_warp_maps(h, w)

    cx, cy, r, th, dx, dy = _WARP_CACHE[key]

    # Displacement magnitude: sinusoidal ripple with falloff from centre
    diag     = math.sqrt(w**2 + h**2)
    strength = float(intensity) * float(amp) * diag * 0.02
    # Ripple: displacement = strength * sin(r / ripple_freq * pi) * exp(-r/diag)
    ripple   = strength * np.sin(r / (diag * 0.15) * math.pi) * np.exp(-r / (diag * 0.6))

    map_x = (cx + dx - (dx / (r + 1e-6)) * ripple).astype(np.float32)
    map_y = (cy + dy - (dy / (r + 1e-6)) * ripple).astype(np.float32)

    return cv2.remap(frame_bgr, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _warp_gray(img, intensity, amp):
    if img is None: return img
    h, w = img.shape[:2]
    key  = (h, w)
    if key not in _WARP_CACHE:
        _WARP_CACHE[key] = _build_warp_maps(h, w)
    cx, cy, r, th, dx, dy = _WARP_CACHE[key]
    diag     = math.sqrt(w**2 + h**2)
    strength = float(intensity) * float(amp) * diag * 0.02
    ripple   = strength * np.sin(r / (diag * 0.15) * math.pi) * np.exp(-r / (diag * 0.6))
    map_x    = (cx + dx - (dx / (r + 1e-6)) * ripple).astype(np.float32)
    map_y    = (cy + dy - (dy / (r + 1e-6)) * ripple).astype(np.float32)
    return cv2.remap(img.astype(np.float32), map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT).astype(np.uint8)


def _chromatic_aberration(frame_bgr: np.ndarray, amp: float, intensity: float) -> np.ndarray:
    """
    RGB channel offset — splits channels apart radially on beats.
    Creates the classic glitch/lens-chromatic-aberration look.
    """
    shift = int(round(float(intensity) * float(amp) * max(frame_bgr.shape[0], frame_bgr.shape[1]) * 0.02))
    if shift < 1: return frame_bgr
    b, g, r = cv2.split(frame_bgr)
    b = np.roll(b,  shift, axis=1)   # blue shifts right
    r = np.roll(r, -shift, axis=1)   # red shifts left
    g = np.roll(g,  shift//2, axis=0) # green shifts down slightly
    return cv2.merge([b, g, r])


def _zoom_blur(frame_bgr: np.ndarray, amp: float, intensity: float, steps: int = 6) -> np.ndarray:
    """
    Radial motion blur — accumulates scaled copies toward centre.
    Creates an 'into the camera' zoom effect pulsing with the beat.
    """
    if amp < 0.01 or intensity < 0.01: return frame_bgr
    h, w   = frame_bgr.shape[:2]
    out    = frame_bgr.astype(np.float32)
    total  = 1.0
    scale_step = 1.0 - float(intensity) * float(amp) * 0.04 / max(steps, 1)
    s = scale_step
    for _ in range(steps):
        scaled = _scale_frame(frame_bgr, s, "fast").astype(np.float32)
        weight = float(amp) * 0.3
        out   += scaled * weight
        total += weight
        s     *= scale_step
    return np.clip(out / total, 0, 255).astype(np.uint8)


# ── Node ───────────────────────────────────────────────────────────────────────

class S42PAudioReactiveFX:
    """
    S42P Audio Reactive FX v5.0

    Apply audio-driven effects to image/video frames.

    NEW v5.0:
      - bypass toggle
      - TRUE radial warp via cv2.remap (not np.roll)
      - chromatic_aberration: RGB channel split on beats
      - zoom_blur: radial motion blur pulsing with energy
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio":  ("AUDIO",),
            },
            "optional": {
                "bypass": ("BOOLEAN", {"default": False,
                    "tooltip": "Pass images through unchanged. For A/B comparison."}),
                "mask": ("MASK", {"tooltip": "Restrict effects to foreground."}),
                "envelope_ema": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "EMA smoothing. Higher = smoother."}),
                "attack":  ("FLOAT", {"default": 0.2,  "min": 0.0, "max": 0.99, "step": 0.01}),
                "release": ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 0.99, "step": 0.01}),
                "curve": (["linear","smoothstep","ease_in_out","sine","cosine","sine_half"],
                          {"default": "ease_in_out"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.1}),
                # LFO
                "lfo_enabled":  ("BOOLEAN", {"default": False}),
                "lfo_waveform": (["sine","triangle","saw","square"], {"default": "sine"}),
                "lfo_rate_hz":  ("FLOAT", {"default": 1.0, "min": 0.01, "max": 30.0, "step": 0.01}),
                "lfo_mix":      ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                # External
                "external_envelope":     ("TENSOR",),
                "use_external_envelope": ("BOOLEAN", {"default": False}),
                "external_beats":        ("TENSOR",),
                "use_external_beats":    ("BOOLEAN", {"default": False}),
                "beat_boost": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 3.0, "step": 0.01}),
                # Pulse
                "pulse_enabled":   ("BOOLEAN", {"default": True}),
                "pulse_intensity": ("FLOAT",   {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Neon
                "neon_enabled":   ("BOOLEAN", {"default": False}),
                "neon_intensity": ("FLOAT",   {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01}),
                "neon_color":     ("STRING",  {"default": "#FF00FF"}),
                "neon_thickness": ("INT",     {"default": 5, "min": 1, "max": 50}),
                # Cascade
                "cascade_enabled":   ("BOOLEAN", {"default": False}),
                "cascade_intensity": ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "cascade_spread":    ("INT",     {"default": 20, "min": 0, "max": 400}),
                "cascade_copies":    ("INT",     {"default": 3, "min": 1, "max": 12}),
                # Flicker
                "flicker_enabled":   ("BOOLEAN", {"default": False}),
                "flicker_intensity": ("FLOAT",   {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.01}),
                "flicker_color":     ("STRING",  {"default": "#00FFFF"}),
                # Rotation
                "rotation_enabled":   ("BOOLEAN", {"default": False}),
                "rotation_intensity": ("FLOAT",   {"default": 8.0, "min": 0.0, "max": 90.0, "step": 0.1}),
                # Warp (TRUE radial via cv2.remap)
                "warp_enabled":   ("BOOLEAN", {"default": False}),
                "warp_intensity": ("FLOAT",   {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "True radial warp strength (polar displacement via cv2.remap)."}),
                # Chromatic Aberration (new)
                "chroma_enabled":   ("BOOLEAN", {"default": False}),
                "chroma_intensity": ("FLOAT",   {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "RGB channel split intensity. Glitch/lens effect on beats."}),
                # Zoom Blur (new)
                "zoom_blur_enabled":   ("BOOLEAN", {"default": False}),
                "zoom_blur_intensity": ("FLOAT",   {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Radial motion blur toward centre. Pulses with audio energy."}),
                "quality_mode": (["fast","high"], {"default": "fast"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "AUDIO")
    RETURN_NAMES  = ("fx_images", "audio_passthrough")
    FUNCTION      = "apply_fx"
    CATEGORY      = "S42 Production Suite/Audio Reactive"

    def apply_fx(
        self, images, audio,
        bypass=False, mask=None,
        envelope_ema=0.25, attack=0.2, release=0.5,
        curve="ease_in_out", fps=24.0,
        lfo_enabled=False, lfo_waveform="sine", lfo_rate_hz=1.0, lfo_mix=0.25,
        external_envelope=None, use_external_envelope=False,
        external_beats=None, use_external_beats=False, beat_boost=0.35,
        pulse_enabled=True, pulse_intensity=0.08,
        neon_enabled=False, neon_intensity=0.6, neon_color="#FF00FF", neon_thickness=5,
        cascade_enabled=False, cascade_intensity=1.0, cascade_spread=20, cascade_copies=3,
        flicker_enabled=False, flicker_intensity=0.4, flicker_color="#00FFFF",
        rotation_enabled=False, rotation_intensity=8.0,
        warp_enabled=False, warp_intensity=6.0,
        chroma_enabled=False, chroma_intensity=0.5,
        zoom_blur_enabled=False, zoom_blur_intensity=0.5,
        quality_mode="fast"
    ):
        if bypass:
            return (images, audio)

        device = images.device
        dtype  = images.dtype
        B      = int(images.shape[0])

        # ── Build envelope ─────────────────────────────────────────────────────
        if use_external_envelope and external_envelope is not None:
            try:
                ev = external_envelope.detach().cpu().numpy().flatten()
                if len(ev) != B:
                    ev = np.interp(np.linspace(0,1,B), np.linspace(0,1,len(ev)), ev)
                env = np.clip(ev.astype(np.float32), 0.0, 1.0)
            except Exception:
                env = np.zeros(B, dtype=np.float32)
        else:
            y, sr = _to_numpy_mono(audio)
            env   = _rms_env(y, B)

        # Smoothing
        smoothed = np.zeros_like(env)
        ema      = float(envelope_ema)
        for i, e in enumerate(env):
            smoothed[i] = ema * (smoothed[i-1] if i > 0 else e) + (1.0 - ema) * e

        # Curve shaping
        def _shape(x):
            x = float(np.clip(x, 0, 1))
            if   curve == "smoothstep":    return x*x*(3-2*x)
            elif curve == "ease_in_out":   return x*x*x*(x*(x*6-15)+10)
            elif curve == "sine":          return math.sin(x * math.pi / 2)
            elif curve == "cosine":        return (1.0 - math.cos(x * math.pi)) / 2
            elif curve == "sine_half":     return math.sin(x * math.pi)
            return x

        shaped = np.array([_shape(v) for v in smoothed], dtype=np.float32)

        # LFO blend
        if lfo_enabled and lfo_mix > 0:
            t = np.arange(B, dtype=np.float32) / max(fps, 1.0)
            ph = 2*math.pi * lfo_rate_hz * t
            if   lfo_waveform == "sine":     lfo = (np.sin(ph) + 1) / 2
            elif lfo_waveform == "triangle": lfo = 1 - 2*np.abs((ph/(2*math.pi)) % 1 - 0.5)
            elif lfo_waveform == "saw":      lfo = (ph / (2*math.pi)) % 1
            else:                            lfo = (np.sign(np.sin(ph)) + 1) / 2
            shaped = np.clip(shaped * (1-lfo_mix) + lfo.astype(np.float32) * lfo_mix, 0, 1)

        env = shaped

        # Beat set
        beat_set: set = set()
        if use_external_beats and external_beats is not None:
            try:
                bf = external_beats.detach().cpu().numpy().astype(np.int64).flatten()
                beat_set = {int(x) for x in bf if 0 <= int(x) < B}
            except Exception:
                pass

        no_effects = not any([pulse_enabled, neon_enabled, cascade_enabled,
                               flicker_enabled, rotation_enabled, warp_enabled,
                               chroma_enabled, zoom_blur_enabled])
        if no_effects:
            return (images, audio)

        out_frames = []
        for i in range(B):
            frm      = (images[i].detach().cpu().numpy() * 255.0).astype(np.uint8)
            h, w     = frm.shape[:2]
            has_alpha = frm.shape[2] == 4
            alpha     = frm[..., 3].copy() if has_alpha else None
            rgb       = frm[..., :3]
            frm_bgr   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            e = float(env[i])
            if i in beat_set and beat_boost > 0.0:
                e = float(np.clip(e + beat_boost, 0.0, 1.0))

            m = _mask_for_frame(mask, i, w, h)

            if pulse_enabled and pulse_intensity > 0.0:
                scale   = 1.0 + e * float(pulse_intensity)
                frm_bgr = _scale_frame(frm_bgr, scale, quality_mode)
                alpha   = _scale_gray(alpha, scale, quality_mode)

            if neon_enabled and neon_intensity > 0.0:
                frm_bgr = _apply_neon(frm_bgr, m, neon_color, int(neon_thickness),
                                      e, float(neon_intensity), quality_mode)

            if cascade_enabled and cascade_copies > 0 and cascade_intensity > 0.0:
                frm_bgr, alpha = _apply_cascade(frm_bgr, alpha, m, int(cascade_spread),
                                                int(cascade_copies), e, float(cascade_intensity))

            if flicker_enabled and flicker_intensity > 1e-4:
                frm_bgr = _apply_flicker(frm_bgr, flicker_color, float(flicker_intensity), e, m)

            if rotation_enabled and rotation_intensity > 0.0:
                deg     = (e - 0.5) * 2.0 * float(rotation_intensity)
                frm_bgr = _rotate_frame(frm_bgr, deg, quality_mode)
                alpha   = _rotate_gray(alpha, deg, quality_mode)

            if warp_enabled and warp_intensity > 0.0:
                frm_bgr = _warp(frm_bgr, float(warp_intensity), e)
                alpha   = _warp_gray(alpha, float(warp_intensity), e)

            if chroma_enabled and chroma_intensity > 0.0:
                frm_bgr = _chromatic_aberration(frm_bgr, e, float(chroma_intensity))

            if zoom_blur_enabled and zoom_blur_intensity > 0.0:
                frm_bgr = _zoom_blur(frm_bgr, e, float(zoom_blur_intensity))

            frm_rgb = cv2.cvtColor(frm_bgr, cv2.COLOR_BGR2RGB)
            out_np  = np.dstack([frm_rgb, alpha]) if has_alpha and alpha is not None else frm_rgb
            out_frames.append(torch.from_numpy(out_np.astype(np.float32) / 255.0))

        fx_images = torch.stack(out_frames).to(device=device, dtype=dtype)
        return (fx_images, audio)


NODE_CLASS_MAPPINGS        = {"S42PAudioReactiveFX": S42PAudioReactiveFX}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PAudioReactiveFX": "🎵 S42P Audio Reactive FX"}
