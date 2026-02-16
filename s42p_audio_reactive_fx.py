"""
S42 Production Suite — Audio Reactive FX v4.1
================================================
Apply audio-reactive visual effects to image/video frames.

Based on Studio42AudioReactiveFX reference implementation.
Outputs correctly-timed frame batches that match input frame count.

EFFECTS:
  pulse       — scale/zoom pulse with audio envelope
  neon        — glowing edge highlight driven by audio
  cascade     — echo/shadow copies offset by envelope
  flicker     — color flash overlay on beats
  rotation    — rotate frame by envelope amount
  warp        — radial ripple warp

SMOOTHING:
  envelope_ema, attack, release, curve — shape the audio response
  lfo_enabled — optionally blend an LFO with the audio envelope

EXTERNAL INTEGRATION:
  Plug in S42P Beat Analyzer or S42P Audio Analyser outputs:
  external_envelope, external_beats — override internal analysis

MASK INPUT:
  Connect a MASK to restrict effects to foreground only.
  Works great with S42P Background Remover output.

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


# ── Audio helpers ─────────────────────────────────────────────────────────

def _to_numpy_mono(audio_any):
    """Parse ComfyUI AUDIO into (y: float32 mono np.ndarray, sr: int)."""
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
    if isinstance(sr, np.ndarray): sr = float(sr.flatten()[0]) if sr.size else 44100.0
    if sr is None: sr = 44100.0
    try: sr = int(round(float(sr)))
    except Exception: sr = 44100

    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
    y = np.array(y)

    if y.ndim == 2:
        if y.shape[0] <= 8: y = y.mean(axis=0)
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
    """RMS envelope resampled to target frame count, normalized 0..1."""
    if frames <= 0: return np.zeros(0, dtype=np.float32)
    if _HAS_LIBROSA:
        rms = librosa.feature.rms(y=y)[0]
        if len(rms) <= 1:
            rms = np.array([float(np.sqrt(np.mean(y*y)+1e-9))], dtype=np.float32)
    else:
        n = len(y); block = max(1, n // frames)
        vals = []
        for i in range(frames):
            s, e = i*block, min(n, (i+1)*block)
            seg = y[s:e]
            vals.append(float(np.sqrt(np.mean(seg*seg)+1e-9)) if seg.size else 0.0)
        rms = np.array(vals, dtype=np.float32)

    x_src = np.linspace(0.0, 1.0, num=len(rms), dtype=np.float32)
    x_dst = np.linspace(0.0, 1.0, num=frames, dtype=np.float32)
    env = np.interp(x_dst, x_src, rms).astype(np.float32)
    mn, mx = float(env.min()), float(env.max())
    return np.zeros_like(env) if mx-mn < 1e-9 else (env-mn)/(mx-mn)


# ── Smoothing & shaping ───────────────────────────────────────────────────

def _ema(arr, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 1e-6: return arr
    out = np.copy(arr)
    for i in range(1, len(out)):
        out[i] = alpha*out[i-1] + (1.0-alpha)*out[i]
    return out


def _attack_release(arr, attack, release):
    a = float(np.clip(attack, 0.0, 1.0))
    r = float(np.clip(release, 0.0, 1.0))
    if (a <= 1e-6 and r <= 1e-6) or len(arr) == 0: return arr
    out = np.copy(arr)
    for i in range(1, len(out)):
        out[i] = (a*out[i-1] + (1.0-a)*out[i] if out[i] > out[i-1]
                  else r*out[i-1] + (1.0-r)*out[i])
    return out


def _shape_curve(x, mode):
    x = np.clip(x, 0.0, 1.0)
    m = (mode or "linear").lower()
    if m == "smoothstep":   return x*x*(3-2*x)
    if m == "ease_in_out":  return 0.5 - 0.5*np.cos(np.pi*x)
    if m == "sine":         return (np.sin(2*np.pi*x - np.pi/2)+1)/2.0
    if m == "cosine":       return (1-np.cos(2*np.pi*x))/2.0
    if m == "sine_half":    return np.sin(np.pi*x)
    return x


def _lfo_wave(n, rate_hz, fps, waveform):
    if fps <= 0: fps = 24.0
    t = np.arange(n, dtype=np.float32) / float(fps)
    phase = 2*np.pi*rate_hz*t
    w = (waveform or "sine").lower()
    if w == "triangle": return (2/np.pi*np.arcsin(np.sin(phase))+1)/2
    if w == "saw":      return (phase/(2*np.pi)) % 1.0
    if w == "square":   return (np.sign(np.sin(phase))+1)/2
    return (np.sin(phase)+1)/2


def _hex_to_bgr(hex_color: str):
    try:
        s = hex_color.strip().lstrip("#")
        if len(s) == 3: s = "".join(c*2 for c in s)
        r, g, b = int(s[0:2],16), int(s[2:4],16), int(s[4:6],16)
        return (b, g, r)
    except Exception:
        return (255, 0, 255)


def _mask_for_frame(mask_tensor, idx, w, h):
    if mask_tensor is None: return None
    if isinstance(mask_tensor, torch.Tensor):
        m = mask_tensor
        if m.ndim == 3: m = m[idx % m.shape[0]]
        m = m.detach().cpu().numpy().astype(np.float32)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(m, 0.0, 1.0)
    return None


# ── Effect implementations ────────────────────────────────────────────────

def _scale_frame(frame_bgr, scale, quality_mode):
    if abs(scale-1.0) < 1e-4: return frame_bgr
    h, w = frame_bgr.shape[:2]
    nh, nw = max(1, int(round(h*scale))), max(1, int(round(w*scale)))
    interp = cv2.INTER_LINEAR if quality_mode == "fast" else cv2.INTER_CUBIC
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=interp)
    if nh >= h and nw >= w:
        y0, x0 = (nh-h)//2, (nw-w)//2
        return resized[y0:y0+h, x0:x0+w]
    canvas = np.zeros_like(frame_bgr)
    y0, x0 = (h-nh)//2, (w-nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def _scale_gray(ch, scale, quality_mode):
    if ch is None: return None
    h, w = ch.shape[:2]
    nh, nw = max(1, int(round(h*scale))), max(1, int(round(w*scale)))
    interp = cv2.INTER_LINEAR if quality_mode == "fast" else cv2.INTER_CUBIC
    resized = cv2.resize(ch, (nw, nh), interpolation=interp)
    if nh >= h and nw >= w:
        y0, x0 = (nh-h)//2, (nw-w)//2
        return resized[y0:y0+h, x0:x0+w]
    canvas = np.zeros((h, w), dtype=ch.dtype)
    y0, x0 = (h-nh)//2, (w-nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def _rotate_frame(frame_bgr, degrees, quality_mode):
    if abs(degrees) < 1e-3: return frame_bgr
    h, w = frame_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), float(degrees), 1.0)
    flags = cv2.INTER_LINEAR if quality_mode == "fast" else cv2.INTER_CUBIC
    return cv2.warpAffine(frame_bgr, M, (w, h), flags=flags, borderMode=cv2.BORDER_REFLECT)


def _rotate_gray(ch, degrees, quality_mode):
    if ch is None: return None
    h, w = ch.shape[:2]
    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), float(degrees), 1.0)
    flags = cv2.INTER_LINEAR if quality_mode == "fast" else cv2.INTER_CUBIC
    return cv2.warpAffine(ch, M, (w, h), flags=flags, borderMode=cv2.BORDER_REFLECT)


def _warp(frame_bgr, intensity, amp):
    h, w = frame_bgr.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    cx, cy = (w-1)/2.0, (h-1)/2.0
    dx, dy = xx-cx, yy-cy
    r = np.sqrt(dx*dx + dy*dy) + 1e-6
    ripple = np.sin(r / (10.0+40.0*(1.0-amp))) * (intensity*amp)
    mx = (xx + (dx/r)*ripple).astype(np.float32)
    my = (yy + (dy/r)*ripple).astype(np.float32)
    return cv2.remap(frame_bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _warp_gray(ch, intensity, amp):
    if ch is None: return None
    h, w = ch.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    cx, cy = (w-1)/2.0, (h-1)/2.0
    dx, dy = xx-cx, yy-cy
    r = np.sqrt(dx*dx + dy*dy) + 1e-6
    ripple = np.sin(r / (10.0+40.0*(1.0-amp))) * (intensity*amp)
    mx = (xx + (dx/r)*ripple).astype(np.float32)
    my = (yy + (dy/r)*ripple).astype(np.float32)
    return cv2.remap(ch, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _apply_neon(frame_bgr, mask_np, hex_color, thickness, amp, intensity, quality_mode):
    color = _hex_to_bgr(hex_color)
    if mask_np is not None:
        edges_src = (mask_np * 255).astype(np.uint8)
        gx = cv2.Sobel(edges_src, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(edges_src, cv2.CV_32F, 0, 1, ksize=3)
        edges = (gx*gx + gy*gy)**0.5
        edges = (edges / (edges.max()+1e-6) * 255).astype(np.uint8)
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)

    result = frame_bgr.copy()
    base_alpha = float(intensity) * float(amp)
    if base_alpha <= 1e-4: return result
    for t in range(1, max(1, thickness)+1, 2):
        k = np.ones((t, t), np.uint8)
        dil = cv2.dilate(edges, k, iterations=1)
        layer = np.zeros_like(frame_bgr); layer[dil > 0] = color
        alpha = base_alpha * (0.6 if t <= 3 else 0.35)
        result = cv2.addWeighted(result, 1.0, layer, alpha, 0)
    if quality_mode == "high":
        result = cv2.GaussianBlur(result, (0, 0), 0.75 + 1.5*amp)
    return result


def _apply_cascade(frame_bgr, alpha, mask_np, spread, copies, amp, intensity):
    base_img = frame_bgr.copy()
    base_a = alpha.copy() if alpha is not None else None
    for i in range(1, copies+1):
        offset = int(round(spread * amp * i))
        shifted_img = np.roll(frame_bgr, shift=offset, axis=1)
        layer_alpha = (0.5/i) * amp * intensity
        if mask_np is not None:
            am = (mask_np * layer_alpha).astype(np.float32)
            base_img = (base_img.astype(np.float32)*(1-am[...,None]) +
                        shifted_img.astype(np.float32)*(am[...,None])).astype(np.uint8)
        else:
            base_img = cv2.addWeighted(base_img, 1.0, shifted_img, layer_alpha, 0)
        if base_a is not None:
            shifted_a = np.roll(alpha, shift=offset, axis=1)
            if mask_np is not None:
                am = (mask_np * layer_alpha).astype(np.float32)
                base_a = (base_a.astype(np.float32)*(1-am) +
                          shifted_a.astype(np.float32)*am).astype(np.uint8)
            else:
                base_a = np.maximum(base_a, (shifted_a * np.clip(layer_alpha,0,1)).astype(np.uint8))
    return base_img, base_a


def _apply_flicker(frame_bgr, hex_color, strength, amp, mask_np=None):
    color = _hex_to_bgr(hex_color)
    overlay = np.zeros_like(frame_bgr, dtype=np.uint8)
    overlay[:] = color
    a = float(strength) * float(amp)
    if a <= 1e-4: return frame_bgr
    if mask_np is None:
        return cv2.addWeighted(frame_bgr, 1.0, overlay, a, 0)
    out = frame_bgr.astype(np.float32)
    ov  = overlay.astype(np.float32)
    am  = (mask_np * a).astype(np.float32)
    out = out*(1-am[...,None]) + ov*(am[...,None])
    return out.astype(np.uint8)


# ── Node ──────────────────────────────────────────────────────────────────

class S42PAudioReactiveFX:
    """S42P Audio Reactive FX — apply audio-driven effects to image/video frames.

    Outputs a frame batch matching the input IMAGE count with effects applied
    in sync with the audio envelope.

    MASK input: restrict effects to foreground only (connect Background Remover mask).
    EXTERNAL inputs: use S42P Beat Analyzer / Audio Analyser for precise beat detection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio":  ("AUDIO",),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional foreground mask — effects apply only where mask > 0"}),

                # Smoothing / shaping
                "envelope_ema": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "EMA smoothing. Higher = smoother, slower response."}),
                "attack":  ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.99, "step": 0.01}),
                "release": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.99, "step": 0.01}),
                "curve": (["linear","smoothstep","ease_in_out","sine","cosine","sine_half"],
                          {"default": "ease_in_out"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.1}),

                # LFO blend
                "lfo_enabled":  ("BOOLEAN", {"default": False}),
                "lfo_waveform": (["sine","triangle","saw","square"], {"default": "sine"}),
                "lfo_rate_hz":  ("FLOAT", {"default": 1.0, "min": 0.01, "max": 30.0, "step": 0.01}),
                "lfo_mix":      ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),

                # External analyzer inputs
                "external_envelope":     ("TENSOR",),
                "use_external_envelope": ("BOOLEAN", {"default": False}),
                "external_beats":        ("TENSOR",),
                "use_external_beats":    ("BOOLEAN", {"default": False}),
                "beat_boost": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 3.0, "step": 0.01,
                    "tooltip": "Extra envelope boost on detected beat frames."}),

                # Pulse / scale
                "pulse_enabled":   ("BOOLEAN", {"default": True}),
                "pulse_intensity": ("FLOAT",   {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Scale pulse amount. 0.08 = gentle 8% zoom on loudest frames."}),

                # Neon glow
                "neon_enabled":   ("BOOLEAN", {"default": False}),
                "neon_intensity": ("FLOAT",   {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01}),
                "neon_color":     ("STRING",  {"default": "#FF00FF",
                    "tooltip": "Hex color for neon edge glow, e.g. #FF00FF, #00FFFF, #FF6600"}),
                "neon_thickness": ("INT",     {"default": 5, "min": 1, "max": 50}),

                # Cascade echo
                "cascade_enabled":   ("BOOLEAN", {"default": False}),
                "cascade_intensity": ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "cascade_spread":    ("INT",     {"default": 20, "min": 0, "max": 400,
                    "tooltip": "Pixel offset per echo copy."}),
                "cascade_copies":    ("INT",     {"default": 3, "min": 1, "max": 12}),

                # Color flicker
                "flicker_enabled":   ("BOOLEAN", {"default": False}),
                "flicker_intensity": ("FLOAT",   {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.01}),
                "flicker_color":     ("STRING",  {"default": "#00FFFF"}),

                # Rotation
                "rotation_enabled":   ("BOOLEAN", {"default": False}),
                "rotation_intensity": ("FLOAT",   {"default": 8.0, "min": 0.0, "max": 90.0, "step": 0.1,
                    "tooltip": "Max rotation degrees at peak envelope."}),

                # Warp
                "warp_enabled":   ("BOOLEAN", {"default": False}),
                "warp_intensity": ("FLOAT",   {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Ripple warp strength."}),

                "quality_mode": (["fast","high"], {"default": "fast"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "AUDIO")
    RETURN_NAMES  = ("fx_images", "audio_passthrough")
    FUNCTION      = "apply_fx"
    CATEGORY      = "S42 Production Suite"

    def apply_fx(self, images, audio, mask=None,
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
                 quality_mode="fast"):

        device = images.device
        dtype  = images.dtype
        B      = int(images.shape[0])

        # ── Envelope ────────────────────────────────────────────────────
        env = None
        if use_external_envelope and external_envelope is not None:
            try:
                env = external_envelope.detach().cpu().numpy().astype(np.float32)
                if env.ndim != 1: env = env.flatten()
                if len(env) != B:
                    xs = np.linspace(0.0,1.0,num=len(env),dtype=np.float32)
                    xd = np.linspace(0.0,1.0,num=B,dtype=np.float32)
                    env = np.interp(xd, xs, env).astype(np.float32)
                env = np.clip(env, 0.0, 1.0)
            except Exception:
                env = None

        if env is None:
            y, sr    = _to_numpy_mono(audio)
            base_env = _rms_env(y, B)
            env      = _ema(base_env, float(envelope_ema))
            env      = _attack_release(env, float(attack), float(release))
            env      = _shape_curve(env, curve)

        if lfo_enabled:
            lfo = _lfo_wave(B, float(lfo_rate_hz), float(fps), lfo_waveform)
            env = np.clip((1.0-float(lfo_mix))*env + float(lfo_mix)*lfo, 0.0, 1.0)

        # ── Beat set ─────────────────────────────────────────────────────
        beat_set: set = set()
        if use_external_beats and external_beats is not None:
            try:
                bf = external_beats.detach().cpu().numpy().astype(np.int64).flatten().tolist()
                beat_set = {int(x) for x in bf if 0 <= int(x) < B}
            except Exception:
                pass

        # Early out — nothing enabled
        if not any([pulse_enabled, neon_enabled, cascade_enabled,
                    flicker_enabled, rotation_enabled, warp_enabled]):
            return (images, audio)

        out_frames = []
        for i in range(B):
            frm = (images[i].detach().cpu().numpy() * 255.0).astype(np.uint8)
            h, w = frm.shape[:2]

            # Alpha split
            has_alpha = (frm.shape[2] == 4)
            alpha = frm[..., 3].copy() if has_alpha else None
            rgb   = frm[..., :3]

            frm_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            e = float(env[i])
            if i in beat_set and beat_boost > 0.0:
                e = float(np.clip(e + float(beat_boost), 0.0, 1.0))

            m = _mask_for_frame(mask, i, w, h)

            if pulse_enabled and pulse_intensity > 0.0:
                scale = 1.0 + e * float(pulse_intensity)
                frm_bgr = _scale_frame(frm_bgr, scale, quality_mode)
                alpha   = _scale_gray(alpha, scale, quality_mode)

            if neon_enabled and neon_intensity > 0.0:
                frm_bgr = _apply_neon(frm_bgr, m, neon_color, int(neon_thickness),
                                      e, float(neon_intensity), quality_mode)

            if cascade_enabled and cascade_copies > 0 and cascade_intensity > 0.0:
                frm_bgr, alpha = _apply_cascade(
                    frm_bgr, alpha, m, int(cascade_spread),
                    int(cascade_copies), e, float(cascade_intensity))

            if flicker_enabled and flicker_intensity > 1e-4:
                frm_bgr = _apply_flicker(frm_bgr, flicker_color,
                                         float(flicker_intensity), e, m)

            if rotation_enabled and rotation_intensity > 0.0:
                deg = (e - 0.5) * 2.0 * float(rotation_intensity)
                frm_bgr = _rotate_frame(frm_bgr, deg, quality_mode)
                alpha   = _rotate_gray(alpha, deg, quality_mode)

            if warp_enabled and warp_intensity > 0.0:
                frm_bgr = _warp(frm_bgr, float(warp_intensity), e)
                alpha   = _warp_gray(alpha, float(warp_intensity), e)

            frm_rgb = cv2.cvtColor(frm_bgr, cv2.COLOR_BGR2RGB)
            out_np  = np.dstack([frm_rgb, alpha]) if has_alpha and alpha is not None else frm_rgb
            out_frames.append(torch.from_numpy(out_np.astype(np.float32) / 255.0))

        fx_images = torch.stack(out_frames).to(device=device, dtype=dtype)
        return (fx_images, audio)


NODE_CLASS_MAPPINGS        = {"S42PAudioReactiveFX": S42PAudioReactiveFX}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PAudioReactiveFX": "S42P Audio Reactive FX 🎵"}
