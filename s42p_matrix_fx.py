"""
S42 Production Suite — Matrix Rain FX v1.0
============================================
Generates animated matrix-style binary rain frames (0s and 1s).

Outputs a batch of IMAGE frames you can pipe directly into VHS_VideoCombine,
S42P Layer Composer, or any ComfyUI video/image batch node.

OPTIONS:
  width / height / fps / frame_count — output dimensions and timing
  transparent_bg — RGBA output with black background removed (alpha channel)
  audio — optional AUDIO to sync drop speed and intensity to music
  char_size — pixel size of each character cell
  drop_speed — columns fall rate (cells per second at idle)
  density — fraction of columns active at once
  glow — enable soft glow blur on characters
  color_r/g/b — foreground character color (defaults to Matrix green)
  head_color — leading character highlight color

AUDIO REACTIVE:
  When audio connected, bass drives column speed and density.
  Works best at 24-30 fps for smooth animation.

Python 3.12 | ComfyUI Portable | No extra dependencies beyond numpy/torch
"""

from __future__ import annotations

import math
import random
import numpy as np
import torch
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


# ── Audio helper ──────────────────────────────────────────────────────────

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
    if isinstance(sr, (list,tuple)) and sr: sr = sr[0]
    if isinstance(sr, torch.Tensor): sr = sr.detach().cpu().numpy()
    if isinstance(sr, np.ndarray): sr = float(sr.flatten()[0]) if sr.size else 44100.0
    if sr is None: sr = 44100.0
    try: sr = int(round(float(sr)))
    except: sr = 44100
    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
    y = np.array(y)
    if y.ndim == 2:
        if y.shape[0] <= 8: y = y.mean(axis=0)
        elif y.shape[1] <= 8: y = y.mean(axis=1)
        else: y = y.flatten()
    elif y.ndim > 2: y = y.flatten()
    y = y.astype(np.float32, copy=False)
    if y.size and np.max(np.abs(y)) > 1.5:
        y = y / max(32768.0, float(np.max(np.abs(y))))
    if y.size == 0: y = np.zeros(1024, dtype=np.float32)
    return y, sr


def _rms_env(y, frames):
    if frames <= 0: return np.zeros(frames, dtype=np.float32)
    if _HAS_LIBROSA:
        rms = librosa.feature.rms(y=y)[0]
    else:
        n = len(y); block = max(1, n // frames)
        rms = np.array([
            float(np.sqrt(np.mean(y[i*block:min(n,(i+1)*block)]**2)+1e-9))
            for i in range(max(1,n//block))
        ], dtype=np.float32)
    xs = np.linspace(0,1,len(rms),dtype=np.float32)
    xd = np.linspace(0,1,frames,dtype=np.float32)
    env = np.interp(xd, xs, rms).astype(np.float32)
    mn, mx = env.min(), env.max()
    return np.zeros_like(env) if mx-mn < 1e-9 else (env-mn)/(mx-mn)


def _band_env(y, sr, frames, lo, hi):
    """Per-band energy envelope."""
    if not _HAS_LIBROSA or y is None: return np.zeros(frames, dtype=np.float32)
    try:
        hop = max(64, len(y)//max(frames,1))
        S = np.abs(librosa.stft(y, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0]*2-2)
        band_mask = (freqs >= lo) & (freqs <= hi)
        band_energy = S[band_mask].mean(axis=0) if band_mask.any() else np.zeros(S.shape[1])
        xs = np.linspace(0,1,len(band_energy),dtype=np.float32)
        xd = np.linspace(0,1,frames,dtype=np.float32)
        env = np.interp(xd, xs, band_energy).astype(np.float32)
        mn, mx = env.min(), env.max()
        return np.zeros_like(env) if mx-mn < 1e-9 else (env-mn)/(mx-mn)
    except Exception:
        return np.zeros(frames, dtype=np.float32)


# ── Matrix Rain Renderer ──────────────────────────────────────────────────

class _MatrixRenderer:
    """Stateful matrix rain column simulator."""

    BINARY_CHARS = "01"
    EXTENDED_CHARS = "01アイウエオカキクケコサシスセソタチツテトナニヌネノ"

    def __init__(self, width: int, height: int, char_size: int,
                 drop_speed: float, density: float, char_set: str = "binary"):
        self.w = width
        self.h = height
        self.cs = max(6, char_size)  # cell size in pixels
        self.cols = max(1, width // self.cs)
        self.rows = max(1, height // self.cs)
        self.speed = drop_speed  # rows per second
        self.density = density
        self.chars = self.EXTENDED_CHARS if char_set == "extended" else self.BINARY_CHARS

        # Per-column state: (position, trail_length, active, speed_mult, chars_list)
        self.state = []
        self._rng = random.Random(42)
        self._init_columns()

    def _init_columns(self):
        self.state = []
        for _ in range(self.cols):
            active = self._rng.random() < self.density
            self.state.append({
                "pos": self._rng.uniform(-self.rows, 0) if active else -self.rows * 2,
                "trail": self._rng.randint(6, min(30, self.rows)),
                "active": active,
                "speed_mult": self._rng.uniform(0.6, 1.6),
                "chars": [self._rng.choice(self.chars) for _ in range(self.rows + 30)],
                "next_spawn": self._rng.uniform(0, 3.0) if not active else 0.0,
            })

    def step(self, dt: float, speed_mult: float = 1.0, density_mult: float = 1.0):
        """Advance simulation by dt seconds."""
        for col in self.state:
            if col["active"]:
                col["pos"] += self.speed * speed_mult * col["speed_mult"] * dt
                # Randomly mutate a few characters
                if self._rng.random() < 0.15:
                    idx = self._rng.randint(0, len(col["chars"])-1)
                    col["chars"][idx] = self._rng.choice(self.chars)
                # Deactivate when off screen
                if col["pos"] - col["trail"] > self.rows + 2:
                    col["active"] = False
                    col["pos"] = -1
                    col["next_spawn"] = self._rng.uniform(0.2, 2.5) / max(0.1, density_mult)
            else:
                col["next_spawn"] -= dt
                if col["next_spawn"] <= 0:
                    # Respawn if density allows
                    if self._rng.random() < self.density * density_mult:
                        col["active"] = True
                        col["pos"]    = -self._rng.uniform(0, 3)
                        col["trail"]  = self._rng.randint(6, min(30, self.rows))
                        col["speed_mult"] = self._rng.uniform(0.6, 1.6)
                        col["chars"] = [self._rng.choice(self.chars) for _ in range(self.rows+30)]
                    col["next_spawn"] = self._rng.uniform(0.1, 1.5)

    def render(self, fg_color: Tuple[int,int,int], head_color: Tuple[int,int,int],
               transparent: bool, glow: bool) -> np.ndarray:
        """Render current state to RGBA numpy array."""
        cs = self.cs
        channels = 4 if transparent else 3
        canvas = np.zeros((self.h, self.w, channels), dtype=np.uint8)

        for ci, col in enumerate(self.state):
            if not col["active"]: continue
            pos = col["pos"]
            trail = col["trail"]
            x = ci * cs

            for ri in range(self.rows + 2):
                dist_from_head = pos - ri
                if dist_from_head < 0 or dist_from_head > trail: continue

                y = ri * cs
                if y + cs > self.h or x + cs > self.w: continue

                # Brightness falloff along trail
                t = 1.0 - (dist_from_head / max(1, trail))
                t = max(0.0, min(1.0, t))

                if dist_from_head < 0.8:
                    # Head character — bright highlight
                    r, g, b = head_color
                    alpha = 255
                else:
                    # Trail character — fade
                    r = int(fg_color[0] * t)
                    g = int(fg_color[1] * t)
                    b = int(fg_color[2] * t)
                    alpha = int(255 * min(1.0, t * 1.5))

                char_idx = ri % len(col["chars"])
                char = col["chars"][char_idx]

                # Draw character as a simple bright pixel block
                # (avoid font rendering dependency)
                pix_r = max(1, cs - 2)
                py1, px1 = y + 1, x + 1
                py2, px2 = min(self.h, py1+pix_r), min(self.w, px1+pix_r)

                if transparent:
                    # RGBA: set pixel block
                    canvas[py1:py2, px1:px2, 0] = r
                    canvas[py1:py2, px1:px2, 1] = g
                    canvas[py1:py2, px1:px2, 2] = b
                    canvas[py1:py2, px1:px2, 3] = alpha
                else:
                    canvas[py1:py2, px1:px2, 0] = r
                    canvas[py1:py2, px1:px2, 1] = g
                    canvas[py1:py2, px1:px2, 2] = b

                # Add character bit pattern for visual distinction
                # Use the 0/1 value as intensity modifier
                if char == "1":
                    # Draw a small bright dot inside the cell
                    mid_y = (py1 + py2) // 2
                    mid_x = (px1 + px2) // 2
                    s = max(1, cs//4)
                    hy1 = max(0, mid_y-s); hy2 = min(self.h, mid_y+s)
                    hx1 = max(0, mid_x-s); hx2 = min(self.w, mid_x+s)
                    if transparent:
                        canvas[hy1:hy2, hx1:hx2, :3] = np.minimum(
                            255,
                            canvas[hy1:hy2, hx1:hx2, :3].astype(int) + 60
                        ).astype(np.uint8)
                    else:
                        canvas[hy1:hy2, hx1:hx2] = np.minimum(
                            255,
                            canvas[hy1:hy2, hx1:hx2].astype(int) + 60
                        ).astype(np.uint8)

        # Optional glow effect
        if glow and _HAS_CV2 and canvas.max() > 0:
            if transparent:
                rgb = canvas[:,:,:3].astype(np.float32)
                blurred = cv2.GaussianBlur(rgb, (0,0), float(cs)*0.4)
                canvas[:,:,:3] = np.clip(rgb + blurred * 0.4, 0, 255).astype(np.uint8)
            else:
                blurred = cv2.GaussianBlur(canvas.astype(np.float32), (0,0), float(cs)*0.4)
                canvas = np.clip(canvas.astype(np.float32) + blurred*0.4, 0, 255).astype(np.uint8)

        return canvas


# ── Node ──────────────────────────────────────────────────────────────────

class S42PMatrixRainFX:
    """S42P Matrix Rain FX — animated binary (0/1) rain effect.

    Generates a batch of animation frames. Pipe output to:
    • VHS_VideoCombine — export as video
    • S42P Layer Composer — composite over other video
    • Any image batch node

    Set transparent_bg=True for RGBA output (black background becomes alpha)
    so you can composite the rain over your own background.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width":       ("INT", {"default": 512,  "min": 64, "max": 4096, "step": 8}),
                "height":      ("INT", {"default": 512,  "min": 64, "max": 4096, "step": 8}),
                "fps":         ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.5}),
                "frame_count": ("INT", {"default": 72, "min": 1, "max": 10000,
                    "tooltip": "Number of frames to generate. 72 = 3s at 24fps."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Optional audio — syncs drop speed and density to bass energy."}),

                "char_size": ("INT", {"default": 14, "min": 6, "max": 64, "step": 2,
                    "tooltip": "Character cell size in pixels. Larger = fewer, bolder characters."}),
                "drop_speed": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 60.0, "step": 0.5,
                    "tooltip": "Column fall rate (cells/second at idle)."}),
                "density": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Fraction of columns active at once."}),

                "color_r": ("INT", {"default": 0,   "min": 0, "max": 255,
                    "tooltip": "Foreground character Red component."}),
                "color_g": ("INT", {"default": 220, "min": 0, "max": 255,
                    "tooltip": "Foreground character Green component (default Matrix green)."}),
                "color_b": ("INT", {"default": 60,  "min": 0, "max": 255,
                    "tooltip": "Foreground character Blue component."}),
                "head_r":  ("INT", {"default": 180, "min": 0, "max": 255}),
                "head_g":  ("INT", {"default": 255, "min": 0, "max": 255}),
                "head_b":  ("INT", {"default": 180, "min": 0, "max": 255}),

                "transparent_bg": ("BOOLEAN", {"default": False,
                    "tooltip": "RGBA output — black background becomes transparent. "
                               "Use with S42P Layer Composer for compositing."}),
                "glow": ("BOOLEAN", {"default": True,
                    "tooltip": "Add subtle glow blur to characters."}),
                "char_set": (["binary","extended"], {"default": "binary",
                    "tooltip": "binary = 0s and 1s only. extended = binary + katakana symbols."}),

                "audio_speed_mult": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "How much bass boosts drop speed."}),
                "audio_density_mult": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "How much bass boosts column density."}),

                "seed": ("INT", {"default": 42, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("frames",)
    FUNCTION      = "generate"
    CATEGORY      = "S42 Production Suite"

    def generate(self, width, height, fps, frame_count,
                 audio=None, char_size=14, drop_speed=8.0, density=0.5,
                 color_r=0, color_g=220, color_b=60,
                 head_r=180, head_g=255, head_b=180,
                 transparent_bg=False, glow=True, char_set="binary",
                 audio_speed_mult=2.0, audio_density_mult=1.5,
                 seed=42):

        random.seed(seed)
        np.random.seed(seed % (2**31))

        fg_color   = (color_r, color_g, color_b)
        head_color = (head_r, head_g, head_b)
        dt = 1.0 / max(1.0, float(fps))
        N  = int(frame_count)

        # Audio envelopes (optional)
        bass_env  = np.zeros(N, dtype=np.float32)
        if audio is not None:
            try:
                y, sr = _to_numpy_mono(audio)
                bass_env = _band_env(y, sr, N, 20, 200)
                # EMA smooth
                for i in range(1, N):
                    bass_env[i] = 0.7*bass_env[i-1] + 0.3*bass_env[i]
            except Exception as e:
                logger.warning(f"[S42P Matrix Rain] Audio analysis failed: {e}")

        renderer = _MatrixRenderer(width, height, char_size, drop_speed, density, char_set)

        frames: List[torch.Tensor] = []
        channels = 4 if transparent_bg else 3

        for i in range(N):
            e = float(bass_env[i])
            speed_mult   = 1.0 + e * float(audio_speed_mult)
            density_mult = 1.0 + e * float(audio_density_mult)

            renderer.step(dt, speed_mult, density_mult)
            canvas = renderer.render(fg_color, head_color, transparent_bg, glow)

            # to float32 tensor [H,W,C]
            frame_f = canvas.astype(np.float32) / 255.0
            frames.append(torch.from_numpy(frame_f))

        # Stack to [B,H,W,C]
        out = torch.stack(frames, dim=0)
        print(f"[S42P Matrix Rain] Generated {N} frames  "
              f"{width}x{height}  {'RGBA' if transparent_bg else 'RGB'}  "
              f"{'audio-reactive' if audio is not None else 'static'}")
        return (out,)


NODE_CLASS_MAPPINGS        = {"S42PMatrixRainFX": S42PMatrixRainFX}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PMatrixRainFX": "S42P Matrix Rain FX 🟩"}
