"""
S42 Production Suite — Audio Visualizer
========================================
Generates a procedural, audio-reactive IMAGE batch (one frame per video frame)
from a ComfyUI AUDIO input. Connect the IMAGE output to VHS Video Combine
alongside the same audio for a complete music visualizer.

VISUAL DESIGN:
  Three-layer hybrid renderer driven by three frequency bands:
    • Background  — slow plasma/flow field driven by bass (20-300Hz)
    • Midground   — rotating geometric shapes driven by mids (300-4kHz)
    • Foreground  — particle sparkle driven by highs (4-20kHz)
  Beat events (from beat_data_json or internal detection) trigger burst pulses
  that affect all layers simultaneously.

AUDIO FEATURES EXTRACTED PER FRAME:
  • rms_bass    — bass energy   → background pulse scale, glow intensity
  • rms_mid     — mid energy    → shape rotation speed, polygon count
  • rms_high    — high energy   → particle density, sparkle brightness
  • beat_hit    — beat on/off   → burst flash event
  • onset_hit   — onset event   → particle spawn burst

DEPENDENCIES:
  PIL (Pillow) — included with ComfyUI
  numpy        — included with ComfyUI
  torch        — included with ComfyUI
  scipy        — installed for S42P suite

Python 3.12 | ComfyUI Portable | No extra dependencies
"""

import math
import random
import numpy as np
import torch
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available — Audio Visualizer will output black frames")

try:
    from scipy import signal as sp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

CATEGORY = "S42 Production Suite 🎬 Animation"

# ── Colour palettes ───────────────────────────────────────────────────────────

PALETTES = {
    "cyberpunk": {
        "bg":      (5,   2,  18),
        "bass":    (180,  0, 255),
        "mid":     (0,  220, 255),
        "high":    (255,  50, 150),
        "beat":    (255, 255, 255),
        "glow":    (100,   0, 200),
    },
    "solar": {
        "bg":      (10,   5,   0),
        "bass":    (255, 100,   0),
        "mid":     (255, 200,  20),
        "high":    (255, 255, 180),
        "beat":    (255, 255, 255),
        "glow":    (200,  60,   0),
    },
    "ocean": {
        "bg":      (0,    5,  20),
        "bass":    (0,   60, 180),
        "mid":     (0,  200, 220),
        "high":    (150, 240, 255),
        "beat":    (255, 255, 255),
        "glow":    (0,   80, 160),
    },
    "emerald": {
        "bg":      (2,   10,   5),
        "bass":    (0,  160,  60),
        "mid":     (80, 255, 120),
        "high":    (200, 255, 180),
        "beat":    (255, 255, 255),
        "glow":    (0,   100,  40),
    },
    "monochrome": {
        "bg":      (5,    5,   5),
        "bass":    (80,  80,  80),
        "mid":     (160, 160, 160),
        "high":    (230, 230, 230),
        "beat":    (255, 255, 255),
        "glow":    (40,  40,  40),
    },
    "inferno": {
        "bg":      (8,    0,   0),
        "bass":    (180,   0,   0),
        "mid":     (255,  80,   0),
        "high":    (255, 220,  50),
        "beat":    (255, 255, 200),
        "glow":    (120,   0,   0),
    },
    "aurora": {
        "bg":      (0,    5,  15),
        "bass":    (0,  140,  80),
        "mid":     (100, 50, 200),
        "high":    (180, 240, 200),
        "beat":    (255, 255, 255),
        "glow":    (40,  80, 120),
    },
}


# ── Audio feature extraction ──────────────────────────────────────────────────

def _extract_band_energy(mono: np.ndarray, sr: int,
                          lo: float, hi: float,
                          hop_n: int) -> np.ndarray:
    """Per-hop RMS energy in a frequency band using a bandpass filter."""
    if SCIPY_AVAILABLE and lo > 0:
        nyq = sr / 2.0
        lo_n = max(lo / nyq, 0.001)
        hi_n = min(hi / nyq, 0.999)
        if lo_n < hi_n:
            try:
                sos = sp.butter(2, [lo_n, hi_n], btype='band', output='sos')
                filtered = sp.sosfilt(sos, mono).astype(np.float32)
            except Exception:
                filtered = mono
        else:
            filtered = mono
    else:
        filtered = mono

    n_frames = len(mono) // hop_n
    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = filtered[i * hop_n:(i + 1) * hop_n]
        energy[i] = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
    return energy


def _detect_beats_internal(mono: np.ndarray, sr: int,
                             hop_n: int, n_frames: int) -> np.ndarray:
    """
    Simple onset-based beat detection without librosa.
    Returns binary array [n_frames] where 1 = beat hit.
    """
    # RMS envelope
    env = np.array([
        float(np.sqrt(np.mean(mono[i*hop_n:(i+1)*hop_n]**2) + 1e-12))
        for i in range(n_frames)
    ], dtype=np.float32)

    # Spectral flux for onset detection
    beats = np.zeros(n_frames, dtype=np.float32)
    if len(env) > 3:
        delta = np.diff(env, prepend=env[0])
        threshold = np.mean(env) + 1.2 * np.std(env)
        # Local maxima above threshold with minimum spacing
        min_gap = max(3, int(sr * 0.25 / hop_n))  # ~250ms min gap
        last_beat = -min_gap
        for i in range(1, len(env) - 1):
            if (env[i] > env[i-1] and env[i] > env[i+1] and
                    env[i] > threshold and (i - last_beat) >= min_gap):
                beats[i] = 1.0
                last_beat = i

    return beats


def _parse_beat_json(beat_json: str, n_frames: int,
                      fps: float, duration: float) -> np.ndarray:
    """Parse beat_data_json from S42P Beat Analyzer into frame-aligned beat array."""
    beats = np.zeros(n_frames, dtype=np.float32)
    try:
        import json
        data = json.loads(beat_json)
        beat_times = data.get("beat_times_sec", [])
        for t in beat_times:
            frame = int(t * fps)
            if 0 <= frame < n_frames:
                beats[frame] = 1.0
                # Spread beat impact over ±1 frame
                if frame > 0:
                    beats[frame - 1] = max(beats[frame - 1], 0.5)
                if frame < n_frames - 1:
                    beats[frame + 1] = max(beats[frame + 1], 0.5)
    except Exception as e:
        logger.warning(f"Could not parse beat_data_json: {e}")
    return beats


# ── Per-frame renderers ───────────────────────────────────────────────────────

def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    """Linear interpolate between two RGB tuples."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _draw_background(draw: 'ImageDraw.Draw', width: int, height: int,
                     bass_energy: float, frame_idx: int,
                     palette: dict, seed_val: float) -> None:
    """
    Plasma/radial background driven by bass energy.
    Uses overlapping radial gradients that pulse with bass.
    """
    bg   = palette["bg"]
    glow = palette["glow"]
    beat_col = palette["bass"]

    # Number of glow orbs scales with bass
    n_orbs = max(2, int(3 + bass_energy * 8))
    rng = random.Random(int(seed_val * 1000) + frame_idx // 8)

    for i in range(n_orbs):
        # Slowly drifting center positions
        angle = (frame_idx * 0.008 + i * 2.1 + rng.random() * 0.5) % (2 * math.pi)
        drift = 0.25 + rng.random() * 0.15
        cx = int(width * (0.5 + drift * math.cos(angle)))
        cy = int(height * (0.5 + drift * math.sin(angle * 0.7)))

        radius = int((width * 0.15) + bass_energy * width * 0.3)
        radius = min(radius, width)

        # Draw concentric ellipses fading from glow to bg
        steps = 12
        for s in range(steps, 0, -1):
            alpha = (s / steps) * bass_energy * 0.7
            col   = _lerp_color(bg, beat_col, alpha * (s / steps))
            r     = int(radius * s / steps)
            if r > 0:
                draw.ellipse([cx - r, cy - r * 2 // 3,
                              cx + r, cy + r * 2 // 3], fill=col)


def _polygon_points(cx: float, cy: float, r: float,
                    n: int, rotation: float, squash: float = 1.0) -> list:
    """Vertices of a regular polygon."""
    pts = []
    for i in range(n):
        angle = rotation + 2 * math.pi * i / n
        pts.append((
            cx + r * math.cos(angle),
            cy + r * math.sin(angle) * squash
        ))
    return pts


def _draw_midground(draw: 'ImageDraw.Draw', width: int, height: int,
                    mid_energy: float, bass_energy: float,
                    frame_idx: int, palette: dict, seed_val: float) -> None:
    """
    Rotating geometric shapes driven by mid energy.
    Multiple polygons at different radii, speeds, and sizes.
    """
    cx, cy = width / 2, height / 2
    mid_col  = palette["mid"]
    bass_col = palette["bass"]

    rng = random.Random(int(seed_val * 999))
    n_shapes = max(2, int(3 + mid_energy * 6))

    for i in range(n_shapes):
        sides = rng.randint(3, 8)
        # Orbit radius pulses with mid energy
        orbit_r = (0.15 + i * 0.07 + mid_energy * 0.08) * min(width, height)
        # Each shape rotates at a different speed
        rot_speed = (0.015 + i * 0.008) * (1.0 + mid_energy * 2.0)
        rotation  = frame_idx * rot_speed * (1 if i % 2 == 0 else -1)
        # Self-rotation
        self_rot  = frame_idx * (0.02 + i * 0.005) * (mid_energy + 0.3)

        # Orbit center
        ocx = cx + orbit_r * math.cos(rotation + rng.random() * 0.5)
        ocy = cy + orbit_r * math.sin(rotation * 0.8 + rng.random() * 0.5) * 0.6

        # Shape size
        shape_r = int((0.03 + mid_energy * 0.05) * min(width, height))
        shape_r = max(8, shape_r)

        pts = _polygon_points(ocx, ocy, shape_r, sides, self_rot)

        # Color: blend mid and bass colors based on energy ratio
        t = min(1.0, bass_energy / (mid_energy + 0.01))
        col = _lerp_color(mid_col, bass_col, t * 0.5)

        # Outline only (no fill) for the geometric look
        if len(pts) >= 3:
            draw.polygon(pts, outline=col)
            # Inner polygon at 60% scale for depth
            inner_pts = _polygon_points(ocx, ocy, shape_r * 0.6, sides, self_rot + math.pi / sides)
            draw.polygon(inner_pts, outline=_lerp_color(col, (255, 255, 255), 0.3))


def _draw_particles(draw: 'ImageDraw.Draw', width: int, height: int,
                    high_energy: float, beat_hit: float,
                    frame_idx: int, palette: dict,
                    particles: list, seed_val: float) -> list:
    """
    Particle system driven by high-frequency energy.
    Particles spawn on high energy / beats, drift, and fade.
    Mutates and returns the particle list.
    """
    high_col = palette["high"]
    beat_col = palette["beat"]
    cx, cy   = width / 2.0, height / 2.0
    rng      = random.Random(frame_idx + int(seed_val * 100))

    # Spawn new particles
    spawn_count = int(high_energy * 12) + (8 if beat_hit > 0.5 else 0)
    for _ in range(spawn_count):
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(1.0, 3.0 + high_energy * 5.0) * (2.0 if beat_hit > 0.5 else 1.0)
        # Spawn near center with some spread
        spread = rng.uniform(0, min(width, height) * 0.2)
        particles.append({
            "x":     cx + math.cos(angle) * spread,
            "y":     cy + math.sin(angle) * spread * 0.6,
            "vx":    math.cos(angle) * speed,
            "vy":    math.sin(angle) * speed * 0.7,
            "life":  rng.uniform(0.4, 1.0),
            "size":  rng.uniform(1.5, 3.5 + high_energy * 4.0),
            "beat":  beat_hit > 0.5,
        })

    # Update and draw surviving particles
    surviving = []
    for p in particles:
        p["x"]    += p["vx"]
        p["y"]    += p["vy"]
        p["vx"]   *= 0.96
        p["vy"]   *= 0.96
        p["life"] -= 0.04

        if p["life"] <= 0:
            continue

        alpha = p["life"]
        col   = beat_col if p["beat"] else high_col
        col   = _lerp_color((0, 0, 0), col, alpha)
        size  = max(1, int(p["size"] * alpha))

        x, y = int(p["x"]), int(p["y"])
        if 0 <= x < width and 0 <= y < height:
            draw.ellipse([x - size, y - size, x + size, y + size], fill=col)

        surviving.append(p)

    # Cap particle count to avoid slowdown
    if len(surviving) > 300:
        surviving = surviving[-300:]

    return surviving


def _draw_beat_flash(draw: 'ImageDraw.Draw', width: int, height: int,
                     beat_hit: float, palette: dict) -> None:
    """Brief radial flash on beat hit — drawn as a large dim overlay ellipse."""
    if beat_hit < 0.3:
        return
    col = _lerp_color((0, 0, 0), palette["beat"], beat_hit * 0.15)
    r = int(min(width, height) * 0.6 * beat_hit)
    cx, cy = width // 2, height // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)


def _draw_waveform_ring(draw: 'ImageDraw.Draw', width: int, height: int,
                         mono_chunk: np.ndarray, bass_e: float,
                         palette: dict, frame_idx: int) -> None:
    """
    Circular waveform: the audio waveform of the current frame displayed
    as a polar ring. Radius modulates with amplitude.
    """
    cx, cy = width / 2.0, height / 2.0
    base_r = min(width, height) * (0.25 + bass_e * 0.1)
    n_pts  = min(256, len(mono_chunk))
    step   = max(1, len(mono_chunk) // n_pts)
    samples = mono_chunk[::step][:n_pts]
    max_amp = max(float(np.max(np.abs(samples))), 1e-6)

    pts = []
    for i, s in enumerate(samples):
        angle = 2 * math.pi * i / len(samples)
        amp   = float(s) / max_amp
        r     = base_r + amp * base_r * 0.5
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    if len(pts) > 2:
        col = _lerp_color(palette["mid"], palette["high"], bass_e)
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=col, width=2)
        draw.line([pts[-1], pts[0]], fill=col, width=2)


def _render_frame(width: int, height: int,
                  bass_e: float, mid_e: float, high_e: float,
                  beat_hit: float, frame_idx: int,
                  palette: dict, particles: list,
                  mono_chunk: np.ndarray,
                  seed_val: float) -> Tuple[np.ndarray, list]:
    """Render one frame. Returns (H, W, 3) float32 array and updated particles."""
    if not PIL_AVAILABLE:
        return np.zeros((height, width, 3), dtype=np.float32), particles

    img  = Image.new("RGB", (width, height), palette["bg"])
    draw = ImageDraw.Draw(img)

    # Layer 1 — background plasma (bass)
    _draw_background(draw, width, height, bass_e, frame_idx, palette, seed_val)

    # Layer 2 — beat flash (before geometry, behind everything)
    _draw_beat_flash(draw, width, height, beat_hit, palette)

    # Layer 3 — midground geometry (mids)
    _draw_midground(draw, width, height, mid_e, bass_e, frame_idx, palette, seed_val)

    # Layer 4 — circular waveform ring
    _draw_waveform_ring(draw, width, height, mono_chunk, bass_e, palette, frame_idx)

    # Layer 5 — foreground particles (highs)
    particles = _draw_particles(draw, width, height, high_e, beat_hit,
                                 frame_idx, palette, particles, seed_val)

    arr = np.array(img, dtype=np.float32) / 255.0
    return arr, particles


# ── Smoothing ─────────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple moving average for energy arrays."""
    if window < 2 or len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr, kernel, mode='same')


def _normalise_energy(arr: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    """Normalise energy array to [0, headroom]."""
    peak = float(np.max(arr)) + 1e-8
    return np.clip(arr / peak, 0.0, headroom).astype(np.float32)


# ── ComfyUI Node ──────────────────────────────────────────────────────────────

class S42PAudioVisualizer:
    """
    S42P Audio Visualizer — generate a procedural, audio-reactive IMAGE batch.

    Three-layer hybrid renderer:
      • Background  — plasma orbs pulsing with bass energy
      • Midground   — rotating geometric polygons driven by mids
      • Foreground  — particle system driven by high frequencies + beats

    Connect IMAGE output to VHS Video Combine with the same audio to create
    a complete music visualizer. Connect beat_data_json from S42P Beat Analyzer
    for precise beat-locked events.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to visualize."
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 8.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Output frames per second. Match your target video FPS."
                }),
                "width": ("INT", {
                    "default": 1280, "min": 256, "max": 2048, "step": 64,
                }),
                "height": ("INT", {
                    "default": 720, "min": 144, "max": 1152, "step": 64,
                }),
                "palette": (list(PALETTES.keys()), {
                    "default": "cyberpunk",
                    "tooltip": "Colour palette for the visualizer."
                }),
                "bass_sensitivity": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Multiplier on bass energy — controls background intensity."
                }),
                "mid_sensitivity": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Multiplier on mid energy — controls geometry speed/scale."
                }),
                "high_sensitivity": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Multiplier on high energy — controls particle density."
                }),
                "motion_speed": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1,
                    "tooltip": "Global animation speed multiplier."
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 999999,
                    "tooltip": "Seed for reproducible geometry placement."
                }),
            },
            "optional": {
                "beat_data_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "JSON from S42P Beat Analyzer. When connected, uses precise "
                        "beat timestamps for flash events. Falls back to internal "
                        "onset detection if not connected."
                    )
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("frames",)
    FUNCTION      = "visualize"
    CATEGORY      = CATEGORY

    def visualize(self, audio: dict, fps: float, width: int, height: int,
                  palette: str, bass_sensitivity: float, mid_sensitivity: float,
                  high_sensitivity: float, motion_speed: float, seed: int,
                  beat_data_json: Optional[str] = None) -> Tuple[torch.Tensor]:

        if not PIL_AVAILABLE:
            logger.error("Pillow not available — cannot render visualizer frames")
            blank = torch.zeros(1, height, width, 3, dtype=torch.float32)
            return (blank,)

        sr       = audio["sample_rate"]
        waveform = audio["waveform"]
        mono     = waveform.numpy().mean(axis=(0, 1)).astype(np.float32) if \
                   hasattr(waveform, 'numpy') else \
                   np.asarray(waveform).mean(axis=(0, 1)).astype(np.float32)

        duration  = len(mono) / sr
        n_frames  = max(1, int(duration * fps))
        hop_n     = max(1, len(mono) // n_frames)
        pal       = PALETTES.get(palette, PALETTES["cyberpunk"])
        seed_val  = float(seed) / 999999.0

        print(f"[S42P Visualizer] {duration:.1f}s audio → {n_frames} frames @ {fps}fps "
              f"({width}×{height}) palette={palette}")

        # ── Extract per-frame energy bands ────────────────────────────────────
        bass_energy = _normalise_energy(
            _smooth(_extract_band_energy(mono, sr, 20.0, 300.0, hop_n), 5)
        ) * bass_sensitivity

        mid_energy  = _normalise_energy(
            _smooth(_extract_band_energy(mono, sr, 300.0, 4000.0, hop_n), 3)
        ) * mid_sensitivity

        high_energy = _normalise_energy(
            _smooth(_extract_band_energy(mono, sr, 4000.0, 20000.0, hop_n), 2)
        ) * high_sensitivity

        # Clamp after sensitivity scaling
        bass_energy = np.clip(bass_energy, 0.0, 1.0)
        mid_energy  = np.clip(mid_energy,  0.0, 1.0)
        high_energy = np.clip(high_energy, 0.0, 1.0)

        # Pad to n_frames if needed
        def _pad(arr, n):
            if len(arr) >= n: return arr[:n]
            return np.pad(arr, (0, n - len(arr)), mode='edge')

        bass_energy = _pad(bass_energy, n_frames)
        mid_energy  = _pad(mid_energy,  n_frames)
        high_energy = _pad(high_energy, n_frames)

        # ── Beat detection ────────────────────────────────────────────────────
        if beat_data_json:
            beat_hits = _parse_beat_json(beat_data_json, n_frames, fps, duration)
        else:
            beat_hits = _detect_beats_internal(mono, sr, hop_n, n_frames)

        # ── Render frames ─────────────────────────────────────────────────────
        frames    = []
        particles = []

        for i in range(n_frames):
            # Mono chunk for this frame's waveform ring
            chunk_start = i * hop_n
            chunk_end   = min(chunk_start + hop_n, len(mono))
            mono_chunk  = mono[chunk_start:chunk_end]

            # Apply motion speed by scaling the effective frame index
            eff_frame = int(i * motion_speed)

            frame_arr, particles = _render_frame(
                width=width, height=height,
                bass_e=float(bass_energy[i]),
                mid_e=float(mid_energy[i]),
                high_e=float(high_energy[i]),
                beat_hit=float(beat_hits[i]),
                frame_idx=eff_frame,
                palette=pal,
                particles=particles,
                mono_chunk=mono_chunk,
                seed_val=seed_val,
            )
            frames.append(frame_arr)

            if (i + 1) % 24 == 0:
                print(f"[S42P Visualizer] {i+1}/{n_frames} frames rendered...")

        print(f"[S42P Visualizer] Complete — {n_frames} frames")

        # Stack to [B, H, W, C] float32 tensor
        batch = np.stack(frames, axis=0)          # [N, H, W, 3]
        out   = torch.from_numpy(batch).float()   # [N, H, W, 3]
        return (out,)


NODE_CLASS_MAPPINGS = {
    "S42PAudioVisualizer": S42PAudioVisualizer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PAudioVisualizer": "S42P Audio Visualizer 🎆"
}
