"""
S42 Production Suite — Audio Visualizer v6.0
=============================================
Multi-mode audio-reactive visualizer for music videos.

IMPROVEMENTS in v6.0:
  - Wormhole: True 3D psychedelic tunnel like Doctor Who intro
  - Aurora: Replaced with "Plasma Waves" - flowing energy ribbons
  - Fractal Zoom: Smooth Mandelbrot/Julia zoom with rotation
  - Liquid Metal: Enhanced reflective chrome surface with dynamic lighting
  - NEW: Kaleidoscope, Particle Explosion, Waveform Tunnel effects

MODES:
  spectrum_classic  — bars + oscilloscope ring + particle burst (Winamp-inspired)
  lava_lamp         — physics blobs that rise/fall, morph with music
  wormhole          — psychedelic 3D tunnel with depth twisting (Doctor Who style)
  matrix            — falling character rain, columns pulse to bass
  stargate          — expanding portal rings with energy tendrils
  oscilloscope      — clean waveform scope with peak indicators ✓ PERFECT
  plasma_waves      — flowing energy ribbons (replaces aurora)
  fractal_zoom      — smooth Mandelbrot zoom with rotation
  dna_helix         — rotating 3D double helix synced to music
  liquid_metal      — enhanced chrome surface with dynamic lighting
  kaleidoscope      — symmetric pattern reflections (NEW)
  particle_explosion— radial burst particles (NEW)
  waveform_tunnel   — 3D waveform tunnel fly-through (NEW)

Python 3.12 | ComfyUI Portable | Pillow + numpy (CPU, no VRAM)
"""

from __future__ import annotations

import gc
import math
import random
import logging
from typing import Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Colour palettes
# ─────────────────────────────────────────────────────────────────────────────

PALETTES = {
    "Spectrum Classic":  {"bg":(0,0,0),    "bar_low":(0,255,0),   "bar_mid":(255,255,0),
                          "bar_top":(255,0,0),"wave":(0,255,128),  "particle":(255,255,255),
                          "beat":(255,64,0), "accent":(0,200,255), "p1":(0,64,128),"p2":(0,128,64)},
    "Neon Synthwave":    {"bg":(5,0,20),   "bar_low":(0,255,255), "bar_mid":(180,0,255),
                          "bar_top":(255,0,128),"wave":(0,255,200),"particle":(255,200,0),
                          "beat":(255,0,200),"accent":(255,255,0), "p1":(20,0,60),"p2":(0,20,60)},
    "Lava Lamp":         {"bg":(10,0,0),   "bar_low":(255,100,0),"bar_mid":(255,50,0),
                          "bar_top":(255,255,50),"wave":(255,140,0),"particle":(255,220,100),
                          "beat":(255,255,100),"accent":(200,80,0),"p1":(80,10,0),"p2":(40,0,0)},
    "Void Blue":         {"bg":(0,0,10),   "bar_low":(0,100,255),"bar_mid":(0,180,255),
                          "bar_top":(200,240,255),"wave":(0,200,255),"particle":(200,240,255),
                          "beat":(100,200,255),"accent":(0,255,200),"p1":(0,10,40),"p2":(0,20,60)},
    "Matrix Green":      {"bg":(0,0,0),    "bar_low":(0,180,0),  "bar_mid":(0,255,0),
                          "bar_top":(200,255,200),"wave":(0,220,0),"particle":(100,255,100),
                          "beat":(200,255,200),"accent":(0,255,80),"p1":(0,20,0),"p2":(0,30,0)},
    "Blood Moon":        {"bg":(5,0,0),    "bar_low":(180,0,0),  "bar_mid":(255,60,0),
                          "bar_top":(255,200,0),"wave":(220,40,0),"particle":(255,120,0),
                          "beat":(255,255,0),"accent":(255,80,0), "p1":(40,0,0),"p2":(20,0,0)},
    "Arctic Ice":        {"bg":(0,10,20),  "bar_low":(180,230,255),"bar_mid":(100,200,255),
                          "bar_top":(255,255,255),"wave":(200,240,255),"particle":(255,255,255),
                          "beat":(0,200,255),"accent":(180,240,255),"p1":(0,20,40),"p2":(0,30,60)},
    "Deep Space":        {"bg":(0,0,5),    "bar_low":(80,0,160), "bar_mid":(160,0,255),
                          "bar_top":(255,200,255),"wave":(120,0,255),"particle":(255,180,255),
                          "beat":(200,100,255),"accent":(255,0,200),"p1":(10,0,30),"p2":(5,0,20)},
    "Sunset":            {"bg":(8,2,0),    "bar_low":(255,80,0), "bar_mid":(255,160,0),
                          "bar_top":(255,240,100),"wave":(255,120,0),"particle":(255,200,100),
                          "beat":(255,255,150),"accent":(200,60,0),"p1":(60,10,0),"p2":(30,5,0)},
    "Cyberpunk":         {"bg":(0,5,8),    "bar_low":(0,255,180),"bar_mid":(255,0,180),
                          "bar_top":(255,255,0),"wave":(0,200,255),"particle":(255,0,200),
                          "beat":(255,255,100),"accent":(0,255,255),"p1":(0,15,25),"p2":(15,0,20)},
    "Gold Chrome":       {"bg":(5,5,0),    "bar_low":(200,150,0),"bar_mid":(255,200,50),
                          "bar_top":(255,255,200),"wave":(255,210,80),"particle":(255,255,200),
                          "beat":(255,240,100),"accent":(200,180,0),"p1":(30,20,0),"p2":(20,15,0)},
    "Infrared":          {"bg":(0,0,5),    "bar_low":(0,0,255),  "bar_mid":(0,255,100),
                          "bar_top":(255,50,0),"wave":(100,255,200),"particle":(255,100,50),
                          "beat":(255,0,50),"accent":(0,200,255), "p1":(0,0,20),"p2":(0,10,0)},
}

PALETTE_NAMES = list(PALETTES.keys())

VIZ_MODES = [
    "spectrum_classic", "lava_lamp",  "wormhole",   "matrix",
    "stargate",         "oscilloscope","plasma_waves", 
    "fractal_zoom",     "dna_helix",  "liquid_metal",
    "kaleidoscope",     "particle_explosion", "waveform_tunnel"
]

# ─────────────────────────────────────────────────────────────────────────────
# Shared: energy extraction + wave chunk
# ─────────────────────────────────────────────────────────────────────────────

def _extract_energy(audio_np: np.ndarray, sr: int, fps: float,
                    max_frames: int):
    total    = len(audio_np)
    n_frames = min(int(total / sr * fps), max_frames)
    n_frames = max(n_frames, 1)
    hop      = max(64, total // n_frames)
    fft_win  = max(512, 1 << (min(2048, hop * 4) - 1).bit_length())
    nyq      = sr / 2.0

    def _bin(hz): return max(1, int(hz / nyq * (fft_win // 2)))

    window  = np.hanning(fft_win).astype(np.float32)
    bass_e  = np.zeros(n_frames, np.float32)
    mid_e   = np.zeros(n_frames, np.float32)
    high_e  = np.zeros(n_frames, np.float32)

    for i in range(n_frames):
        s = i * hop; e = s + fft_win
        chunk = np.zeros(fft_win, np.float32)
        av    = min(fft_win, total - s)
        if av > 0: chunk[:av] = audio_np[s:s+av]
        chunk *= window
        spec   = np.abs(np.fft.rfft(chunk))[:fft_win // 2]
        bass_e[i] = float(np.mean(spec[:_bin(250)] ** 2)) ** 0.5
        mid_e[i]  = float(np.mean(spec[_bin(250):_bin(4000)] ** 2)) ** 0.5
        high_e[i] = float(np.mean(spec[_bin(4000):] ** 2)) ** 0.5

    beat   = np.zeros(n_frames, np.float32)
    win_b  = max(1, int(fps * 0.4))
    for i in range(n_frames):
        lo, hi = max(0, i-win_b), min(n_frames, i+win_b)
        if bass_e[i] > np.mean(bass_e[lo:hi]) * 1.5 + 1e-8:
            beat[i] = 1.0

    def _n(a): return np.clip(a / (float(np.max(a)) + 1e-8), 0, 1)
    return _n(bass_e), _n(mid_e), _n(high_e), beat, n_frames


def _wave_chunk(audio_np: np.ndarray, sr: int, i: int, fps: float, n_frames: int):
    t      = i / max(1, n_frames)
    centre = int(t * len(audio_np))
    cs     = max(64, sr // 60)
    s      = max(0, centre - cs // 2)
    e      = min(len(audio_np), s + cs)
    c      = audio_np[s:e].astype(np.float32)
    return c if len(c) >= 2 else np.zeros(64, np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lerp(c1, c2, t):
    t = max(0., min(1., float(t)))
    return (int(c1[0]+(c2[0]-c1[0])*t), int(c1[1]+(c2[1]-c1[1])*t),
            int(c1[2]+(c2[2]-c1[2])*t))

def _hsv(h, s, v):
    import colorsys
    r,g,b = colorsys.hsv_to_rgb(h%1.0, s, v)
    return (int(r*255), int(g*255), int(b*255))


# ─────────────────────────────────────────────────────────────────────────────
# MODE: SPECTRUM CLASSIC (kept as-is, already perfect)
# ─────────────────────────────────────────────────────────────────────────────

def _spectrum_classic_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal, bar_count, particles):
    # Background plasma
    step = max(8, w // 40); t = fi * 0.05
    for x in range(0, w, step):
        for y in range(0, h, step):
            v = (math.sin(x*.02+t)+math.sin(y*.02+t*.7)+math.sin((x+y)*.015+t*1.3)*(0.3+bass*.7)+3)/6
            draw.rectangle([x,y,x+step,y+step], fill=_lerp(pal["p1"],pal["p2"],v))
    
    # Beat flash
    if beat > 0.5:
        cx,cy = w//2,h//2
        r_max = int(math.sqrt(cx*cx+cy*cy)*beat)
        for r in range(0, r_max, max(2, r_max//8)):
            draw.ellipse([cx-r,cy-r,cx+r,cy+r], outline=pal["beat"])
    
    # Spectrum bars
    bar_w = max(4, w // bar_count)
    for i in range(bar_count):
        x = i * bar_w
        ht = int(h * 0.6 * (bass * 0.4 + mid * 0.4 + high * 0.2) * (0.3 + random.random() * 0.7))
        ht = min(ht, h)
        col = _lerp(pal["bar_low"], _lerp(pal["bar_mid"], pal["bar_top"], ht/max(1,h)), min(1, ht/(h*0.7)))
        draw.rectangle([x, h - ht, x + bar_w - 2, h], fill=col)
    
    # Oscilloscope ring
    cx2, cy2 = w // 2, h // 2
    r_ring = int(min(w, h) * 0.38 + bass * min(w,h) * 0.08)
    n_pts = min(len(wave), 256)
    pts = []
    for i in range(n_pts):
        ang = (i / n_pts) * 2 * math.pi
        amp = wave[i * len(wave) // n_pts]
        r = r_ring + amp * r_ring * 0.3
        pts.append((cx2 + int(r * math.cos(ang)), cy2 + int(r * math.sin(ang))))
    if len(pts) > 1:
        draw.line(pts + [pts[0]], fill=pal["wave"], width=2)
    
    # Particles
    if beat > 0.7:
        for _ in range(int(10 * beat)):
            particles.append({
                "x": cx2, "y": cy2,
                "vx": random.uniform(-8,8), "vy": random.uniform(-8,8),
                "life": 40
            })
    
    for p in particles:
        p["x"] += p["vx"]; p["y"] += p["vy"]
        p["life"] -= 1
        if p["life"] > 0:
            a = p["life"] / 40.0
            draw.ellipse([p["x"]-2, p["y"]-2, p["x"]+2, p["y"]+2],
                        fill=tuple(int(c*a) for c in pal["particle"]))
    
    particles[:] = [p for p in particles if p["life"] > 0]
    return particles


# ─────────────────────────────────────────────────────────────────────────────
# MODE: WORMHOLE (Doctor Who style psychedelic 3D tunnel)
# ─────────────────────────────────────────────────────────────────────────────

def _wormhole_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    """Psychedelic 3D tunnel with twisting, depth-mapped rings - Doctor Who intro style"""
    cx, cy = w // 2, h // 2
    num_rings = 30
    t = fi * 0.015
    
    # Draw rings from back to front for proper depth
    for ring_idx in range(num_rings, 0, -1):
        depth = ring_idx / num_rings
        z = depth + t % 1.0
        if z > 1.0:
            z -= 1.0
        
        # Perspective scaling
        scale = 0.2 + (1.0 - depth) * 2.5
        radius = int(min(w, h) * scale * (0.15 + bass * 0.1))
        
        # Twist rotation based on depth
        twist = depth * math.pi * 4 + t * 3
        
        # Color cycling through depth
        hue = (depth * 2 + t * 0.5 + bass * 0.3) % 1.0
        sat = 0.7 + high * 0.3
        val = 0.3 + depth * 0.7 + beat * 0.3
        ring_col = _hsv(hue, sat, val)
        
        # Draw segmented ring with gaps
        segments = 12 + int(mid * 8)
        for seg in range(segments):
            ang_start = (seg / segments) * 2 * math.pi + twist
            ang_end = ((seg + 0.7) / segments) * 2 * math.pi + twist
            
            # Draw thick arc
            pts = []
            for a in np.linspace(ang_start, ang_end, 8):
                x = cx + int(radius * math.cos(a))
                y = cy + int(radius * math.sin(a))
                pts.append((x, y))
            
            if len(pts) > 1:
                width = max(1, int(3 * (1.0 - depth) + beat * 2))
                draw.line(pts, fill=ring_col, width=width)
        
        # Add particles spiraling through tunnel
        if beat > 0.6 and ring_idx % 3 == 0:
            for p_ang in np.linspace(0, 2*math.pi, 6):
                px = cx + int(radius * 0.7 * math.cos(p_ang + twist))
                py = cy + int(radius * 0.7 * math.sin(p_ang + twist))
                p_size = max(1, int(4 * (1.0 - depth)))
                draw.ellipse([px-p_size, py-p_size, px+p_size, py+p_size],
                            fill=pal["particle"])


# ─────────────────────────────────────────────────────────────────────────────
# MODE: PLASMA WAVES (replaces aurora)
# ─────────────────────────────────────────────────────────────────────────────

def _plasma_waves_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    """Flowing plasma energy ribbons"""
    t = fi * 0.02
    
    # Create plasma field
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            # Multi-layer sine waves
            v1 = math.sin(x * 0.01 + t * 2)
            v2 = math.sin(y * 0.01 + t * 1.5)
            v3 = math.sin((x + y) * 0.007 + t * 3)
            v4 = math.sin(math.sqrt(x*x + y*y) * 0.015 + t)
            
            plasma = (v1 + v2 + v3 + v4 + 4) / 8
            plasma = plasma * (0.7 + bass * 0.3)
            
            hue = (plasma + t * 0.5 + mid * 0.2) % 1.0
            sat = 0.6 + high * 0.4
            val = 0.4 + plasma * 0.6 + beat * 0.3
            col = _hsv(hue, sat, val)
            
            draw.rectangle([x, y, x+4, y+4], fill=col)
    
    # Energy ribbons
    num_ribbons = 5
    for ribbon in range(num_ribbons):
        points = []
        phase = (ribbon / num_ribbons) * 2 * math.pi + t
        
        for x in range(0, w, 8):
            wave_y = h // 2 + int(
                math.sin(x * 0.02 + phase) * h * 0.2 * (1 + bass * 0.5) +
                math.sin(x * 0.01 + phase * 1.5) * h * 0.1 * (1 + mid * 0.5)
            )
            points.append((x, wave_y))
        
        if len(points) > 1:
            hue = (ribbon / num_ribbons + t * 0.3) % 1.0
            ribbon_col = _hsv(hue, 0.9, 0.8 + beat * 0.2)
            draw.line(points, fill=ribbon_col, width=max(2, int(3 + bass * 4)))


# ─────────────────────────────────────────────────────────────────────────────
# MODE: FRACTAL ZOOM (smooth continuous zoom with rotation)
# ─────────────────────────────────────────────────────────────────────────────

def _fractal_zoom_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    """Smooth Mandelbrot/Julia set zoom with rotation"""
    img = draw._image
    pixels = img.load()
    
    # Smooth zoom and rotation
    zoom = 0.5 ** (fi * 0.01 * (1 + bass * 0.5))
    rotation = fi * 0.02 * (1 + mid * 0.3)
    
    # Center offset - slowly drift
    cx_offset = -0.5 + math.sin(fi * 0.003) * 0.3
    cy_offset = math.cos(fi * 0.004) * 0.3
    
    for py in range(h):
        for px in range(w):
            # Convert to complex plane with rotation
            x = (px - w/2) / (w * 0.4)
            y = (py - h/2) / (h * 0.4)
            
            # Rotate
            ang = rotation
            xr = x * math.cos(ang) - y * math.sin(ang)
            yr = x * math.sin(ang) + y * math.cos(ang)
            
            # Apply zoom and offset
            c = complex(xr * zoom + cx_offset, yr * zoom + cy_offset)
            z = complex(0, 0)
            
            # Mandelbrot iteration
            max_iter = 50
            for i in range(max_iter):
                if abs(z) > 2.0:
                    break
                z = z*z + c
            
            # Smooth coloring
            if i < max_iter:
                smooth_i = i + 1 - math.log(math.log(abs(z) + 1)) / math.log(2)
                hue = (smooth_i * 0.05 + bass * 0.2 + fi * 0.002) % 1.0
                sat = 0.7 + high * 0.3
                val = min(1.0, smooth_i / 20.0 + beat * 0.3)
                col = _hsv(hue, sat, val)
            else:
                col = pal["bg"]
            
            if 0 <= px < w and 0 <= py < h:
                pixels[px, py] = col


# ─────────────────────────────────────────────────────────────────────────────
# MODE: LIQUID METAL (enhanced)
# ─────────────────────────────────────────────────────────────────────────────

def _liquid_metal_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    """Enhanced chrome/mercury surface with dynamic lighting and ripples"""
    img = draw._image
    pixels = img.load()
    
    t = fi * 0.03
    
    for py in range(h):
        for px in range(w):
            x = px / w
            y = py / h
            
            # Multi-layer liquid surface
            surf1 = math.sin(x * 10 + t * 2 + bass * 3) * 0.1
            surf2 = math.sin(y * 8 - t * 1.5 + mid * 2) * 0.08
            surf3 = math.sin((x + y) * 12 + t * 3) * 0.06
            
            # Beat ripples
            if beat > 0.5:
                dist = math.sqrt((x - 0.5)**2 + (y - 0.5)**2)
                ripple = math.sin(dist * 30 - t * 5) * 0.15 * beat
                surf1 += ripple
            
            surface = surf1 + surf2 + surf3
            
            # Chrome gradient
            normal = surface + 0.5
            
            # Reflective highlights
            highlight = max(0, normal - 0.6) * 3
            shadow = max(0, 0.4 - normal) * 2
            
            # Color based on surface normal
            base_bright = 0.3 + normal * 0.4 + high * 0.3
            hue = (t * 0.1 + surface * 0.5) % 1.0
            
            r = int(min(255, (base_bright + highlight - shadow + bass * 0.2) * 255))
            g = int(min(255, (base_bright + highlight * 0.9 - shadow + mid * 0.15) * 255))
            b = int(min(255, (base_bright + highlight * 0.8 - shadow * 0.5) * 255))
            
            # Add some color tint
            tint_col = _hsv(hue, 0.3, 1.0)
            r = int(r * 0.7 + tint_col[0] * 0.3)
            g = int(g * 0.7 + tint_col[1] * 0.3)
            b = int(b * 0.7 + tint_col[2] * 0.3)
            
            pixels[px, py] = (
                max(0, min(255, r)),
                max(0, min(255, g)),
                max(0, min(255, b))
            )


# ─────────────────────────────────────────────────────────────────────────────
# Other existing modes (lava_lamp, matrix, stargate, oscilloscope, dna_helix)
# Keeping implementation from original - they work well
# ─────────────────────────────────────────────────────────────────────────────

def _lava_frame(draw, w, h, bass, mid, high, beat, fi, pal, blobs):
    """Physics-based lava lamp blobs"""
    if not blobs:
        for _ in range(8):
            blobs.append({
                "x": random.randint(0, w),
                "y": random.randint(0, h),
                "vx": random.uniform(-1, 1),
                "vy": random.uniform(-2, 0),
                "r": random.randint(30, 80),
                "hue": random.random()
            })
    
    # Update blobs
    for blob in blobs:
        blob["vy"] += 0.1 - bass * 0.15  # Gravity + bass
        blob["vx"] *= 0.98
        blob["vy"] *= 0.98
        
        blob["x"] += blob["vx"]
        blob["y"] += blob["vy"]
        
        # Bounce
        if blob["y"] < blob["r"]:
            blob["y"] = blob["r"]
            blob["vy"] = abs(blob["vy"]) * 0.7
        if blob["y"] > h - blob["r"]:
            blob["y"] = h - blob["r"]
            blob["vy"] = -abs(blob["vy"]) * 0.7
        if blob["x"] < blob["r"] or blob["x"] > w - blob["r"]:
            blob["vx"] *= -1
            blob["x"] = max(blob["r"], min(w - blob["r"], blob["x"]))
        
        blob["r"] = int(30 + 50 * (0.5 + bass * 0.5))
        blob["hue"] = (blob["hue"] + 0.001) % 1.0
    
    # Draw blobs with glow
    for blob in blobs:
        col = _hsv(blob["hue"], 0.9, 0.9 + beat * 0.1)
        for r_offset in range(blob["r"], 0, -max(1, blob["r"]//10)):
            alpha = r_offset / blob["r"]
            glow_col = tuple(int(c * alpha) for c in col)
            draw.ellipse([
                blob["x"] - r_offset, blob["y"] - r_offset,
                blob["x"] + r_offset, blob["y"] + r_offset
            ], fill=glow_col)
    
    return blobs


def _matrix_frame(draw, w, h, bass, mid, high, beat, fi, pal, cols):
    """Matrix-style falling code"""
    if not cols:
        for x in range(0, w, 12):
            cols.append({
                "x": x,
                "y": random.randint(-h, 0),
                "speed": 2 + random.random() * 4,
                "chars": [random.choice("01") for _ in range(40)]
            })
    
    for col in cols:
        col["speed"] = 2 + mid * 6 + (4 if beat > 0.5 else 0)
        col["y"] += col["speed"]
        
        if col["y"] > h:
            col["y"] = -50
            col["chars"] = [random.choice("01") for _ in range(40)]
        
        for i, char in enumerate(col["chars"]):
            y_pos = int(col["y"] - i * 15)
            if 0 <= y_pos < h:
                alpha = 1.0 - (i / len(col["chars"]))
                col_char = tuple(int(c * alpha) for c in pal["bar_mid"])
                draw.text((col["x"], y_pos), char, fill=col_char)
    
    return cols


def _stargate_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    """Expanding stargate portal rings"""
    cx, cy = w // 2, h // 2
    t = fi * 0.05
    
    # Main portal rings
    for ring in range(10):
        r = int((ring * 40 + t * 50 + bass * 80) % (min(w, h) * 0.7))
        alpha = 1.0 - (r / (min(w, h) * 0.7))
        if alpha > 0:
            hue = (ring * 0.1 + t * 0.2) % 1.0
            ring_col = _hsv(hue, 0.7 + high * 0.3, 0.6 + alpha * 0.4 + beat * 0.2)
            width = max(1, int(5 * alpha + beat * 3))
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=ring_col, width=width)
    
    # Energy tendrils
    if beat > 0.4:
        for angle in np.linspace(0, 2*math.pi, 8):
            pts = []
            for dist in range(0, min(w,h)//2, 20):
                wobble = math.sin(dist * 0.1 + t) * 30 * bass
                x = cx + int((dist + wobble) * math.cos(angle))
                y = cy + int((dist + wobble) * math.sin(angle))
                pts.append((x, y))
            if pts:
                draw.line(pts, fill=pal["accent"], width=2)


def _osc_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal):
    """Clean oscilloscope waveform - PERFECT, don't change"""
    # Grid
    for i in range(0, h, h//8):
        draw.line([(0, i), (w, i)], fill=(20,20,20))
    for i in range(0, w, w//10):
        draw.line([(i, 0), (i, h)], fill=(20,20,20))
    
    # Waveform
    points = []
    for i, sample in enumerate(wave):
        x = int(i * w / len(wave))
        y = int(h/2 - sample * h * 0.4)
        points.append((x, y))
    
    if len(points) > 1:
        draw.line(points, fill=pal["wave"], width=2)
    
    # Peak indicators
    if beat > 0.5:
        draw.line([(0, 10), (w, 10)], fill=pal["beat"], width=3)
        draw.line([(0, h-10), (w, h-10)], fill=pal["beat"], width=3)


def _dna_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    """Double helix DNA strand"""
    cx, cy = w // 2, h // 2
    t = fi * 0.03
    
    # Draw double helix
    pts1, pts2 = [], []
    for i in range(60):
        z = i / 60.0
        y = int(h * z)
        
        angle = z * 4 * math.pi + t
        radius = int(w * 0.2 * (1 + bass * 0.3))
        
        x1 = cx + int(radius * math.cos(angle))
        x2 = cx + int(radius * math.cos(angle + math.pi))
        
        pts1.append((x1, y))
        pts2.append((x2, y))
        
        # Rungs
        if i % 3 == 0:
            hue = (z + t * 0.5) % 1.0
            rung_col = _hsv(hue, 0.8, 0.7 + beat * 0.3)
            draw.line([(x1, y), (x2, y)], fill=rung_col, width=2)
    
    # Strands
    if pts1:
        draw.line(pts1, fill=pal["bar_mid"], width=3)
        draw.line(pts2, fill=pal["bar_top"], width=3)


# ─────────────────────────────────────────────────────────────────────────────
# NEW MODES
# ─────────────────────────────────────────────────────────────────────────────

def _kaleidoscope_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    """Symmetric kaleidoscope pattern"""
    img = draw._image
    pixels = img.load()
    
    t = fi * 0.02
    segments = 8
    
    for py in range(h):
        for px in range(w):
            x = px - w/2
            y = py - h/2
            
            # Convert to polar
            r = math.sqrt(x*x + y*y) / (min(w,h) * 0.5)
            theta = math.atan2(y, x)
            
            # Kaleidoscope symmetry
            theta = (theta % (2 * math.pi / segments)) * segments
            
            # Pattern
            pattern = math.sin(r * 10 + t) * math.cos(theta * 3 + t * 2)
            pattern = pattern * (0.7 + bass * 0.3)
            
            hue = (pattern + r + t * 0.5 + mid * 0.2) % 1.0
            sat = 0.7 + high * 0.3
            val = 0.4 + abs(pattern) * 0.6 + beat * 0.2
            
            col = _hsv(hue, sat, val)
            pixels[px, py] = col


def _particle_explosion_frame(draw, w, h, bass, mid, high, beat, fi, pal, state):
    """Radial particle burst"""
    particles = state.get("particles", [])
    cx, cy = w // 2, h // 2
    
    # Spawn on beats
    if beat > 0.6:
        for _ in range(int(30 * beat)):
            angle = random.random() * 2 * math.pi
            speed = 2 + random.random() * 8 * bass
            particles.append({
                "x": cx, "y": cy,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": 60,
                "hue": random.random()
            })
    
    # Update and draw
    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vy"] += 0.2  # Gravity
        p["life"] -= 1
        
        if p["life"] > 0:
            alpha = p["life"] / 60.0
            col = _hsv(p["hue"], 0.9, alpha)
            size = max(1, int(4 * alpha))
            draw.ellipse([p["x"]-size, p["y"]-size, p["x"]+size, p["y"]+size], fill=col)
    
    particles[:] = [p for p in particles if p["life"] > 0]
    state["particles"] = particles


def _waveform_tunnel_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal):
    """3D waveform tunnel fly-through"""
    cx, cy = w // 2, h // 2
    t = fi * 0.05
    
    # Draw rings of waveform at different depths
    for ring in range(20, 0, -1):
        depth = ring / 20.0
        z_offset = (t + depth) % 1.0
        scale = 0.3 + (1.0 - depth) * 1.5
        
        # Sample waveform
        wave_idx = int((z_offset * len(wave))) % len(wave)
        n_samples = min(64, len(wave))
        
        pts = []
        for i in range(n_samples):
            angle = (i / n_samples) * 2 * math.pi
            sample_idx = (wave_idx + i) % len(wave)
            amplitude = wave[sample_idx] * h * 0.2 * scale * (1 + bass * 0.5)
            
            radius = min(w,h) * 0.15 * scale + amplitude
            x = cx + int(radius * math.cos(angle))
            y = cy + int(radius * math.sin(angle))
            pts.append((x, y))
        
        if len(pts) > 2:
            hue = (depth + t * 0.3) % 1.0
            col = _hsv(hue, 0.8, 0.5 + depth * 0.5 + beat * 0.2)
            width = max(1, int(2 * (1.0 - depth)))
            draw.line(pts + [pts[0]], fill=col, width=width)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

def _render_frame(mode: str, w: int, h: int, bass: float, mid: float, high: float,
                  beat: float, wave: np.ndarray, fi: int, pal: dict,
                  bar_count: int, state: dict):

    # Pixel-level modes that draw directly
    if mode in ("fractal_zoom", "liquid_metal", "kaleidoscope"):
        img  = Image.new("RGB", (w, h), pal["bg"])
        draw = ImageDraw.Draw(img)
        if mode == "fractal_zoom":
            _fractal_zoom_frame(draw, w, h, bass, mid, high, beat, fi, pal)
        elif mode == "liquid_metal":
            _liquid_metal_frame(draw, w, h, bass, mid, high, beat, fi, pal)
        elif mode == "kaleidoscope":
            _kaleidoscope_frame(draw, w, h, bass, mid, high, beat, fi, pal)
    else:
        img  = Image.new("RGB", (w, h), pal["bg"])
        draw = ImageDraw.Draw(img)

        if mode == "spectrum_classic":
            p = state.get("particles", [])
            p = _spectrum_classic_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal, bar_count, p)
            state["particles"] = p

        elif mode == "lava_lamp":
            bl = state.get("blobs", [])
            bl = _lava_frame(draw, w, h, bass, mid, high, beat, fi, pal, bl)
            state["blobs"] = bl

        elif mode == "wormhole":
            _wormhole_frame(draw, w, h, bass, mid, high, beat, fi, pal)

        elif mode == "matrix":
            cs = state.get("cols", [])
            cs = _matrix_frame(draw, w, h, bass, mid, high, beat, fi, pal, cs)
            state["cols"] = cs

        elif mode == "stargate":
            _stargate_frame(draw, w, h, bass, mid, high, beat, fi, pal)

        elif mode == "oscilloscope":
            _osc_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal)

        elif mode == "plasma_waves":
            _plasma_waves_frame(draw, w, h, bass, mid, high, beat, fi, pal)

        elif mode == "dna_helix":
            _dna_frame(draw, w, h, bass, mid, high, beat, fi, pal)
        
        elif mode == "particle_explosion":
            _particle_explosion_frame(draw, w, h, bass, mid, high, beat, fi, pal, state)
        
        elif mode == "waveform_tunnel":
            _waveform_tunnel_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal)

    arr = np.array(img, dtype=np.float32) / 255.0
    return arr, state


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI Node
# ─────────────────────────────────────────────────────────────────────────────

class S42PAudioVisualizer:
    """
    S42P Audio Visualizer — 13 audio-reactive modes for music videos.

    UPDATED MODES:
      wormhole         - Doctor Who style psychedelic 3D tunnel ✨ NEW
      plasma_waves     - Flowing energy ribbons (replaces aurora) ✨ NEW
      fractal_zoom     - Smooth Mandelbrot zoom with rotation ✨ IMPROVED
      liquid_metal     - Enhanced chrome surface ✨ IMPROVED
      kaleidoscope     - Symmetric pattern reflections ✨ NEW
      particle_explosion - Radial burst particles ✨ NEW
      waveform_tunnel  - 3D waveform fly-through ✨ NEW

    PERFECT (unchanged):
      spectrum_classic, oscilloscope

    12 PALETTES: Spectrum Classic, Neon Synthwave, Lava Lamp, Void Blue,
    Matrix Green, Blood Moon, Arctic Ice, Deep Space, Sunset, Cyberpunk,
    Gold Chrome, Infrared
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":    ("AUDIO",),
                "mode":     (VIZ_MODES, {"default": "spectrum_classic"}),
                "fps":      ("FLOAT", {"default": 24.0, "min": 8.0, "max": 60.0, "step": 1.0}),
                "width":    ("INT",   {"default": 512, "min": 256, "max": 1920, "step": 64}),
                "height":   ("INT",   {"default": 288, "min": 144, "max": 1080, "step": 32}),
                "palette":  (PALETTE_NAMES, {"default": "Spectrum Classic"}),
                "bar_count":("INT",   {"default": 64, "min": 16, "max": 128, "step": 8}),
                "max_frames":("INT",  {
                    "default": 1440, "min": 60, "max": 9000, "step": 60,
                    "tooltip": "24fps: 720=30s  1440=60s  2880=120s"
                }),
                "low_memory":("BOOLEAN", {
                    "default": False,
                    "tooltip": "Render at half resolution + upscale. Saves RAM."
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("visualization",)
    FUNCTION      = "visualize"
    CATEGORY      = "S42 Production Suite 🎨 Visualizer"

    def visualize(self, audio: dict, mode: str, fps: float, width: int,
                  height: int, palette: str, bar_count: int, max_frames: int,
                  low_memory: bool = False, seed: int = 42) -> tuple:

        if not PIL_AVAILABLE:
            dummy = np.zeros((1, height, width, 3), dtype=np.float32)
            return (torch.from_numpy(dummy) if TORCH_AVAILABLE else dummy,)

        waveform = audio.get("waveform")
        sr       = int(audio.get("sample_rate", 44100))

        wav_np   = waveform.cpu().float().numpy() if TORCH_AVAILABLE and hasattr(waveform, "numpy") else np.asarray(waveform, dtype=np.float32)
        if wav_np.ndim == 3: wav_np = wav_np.mean(axis=(0, 1))
        elif wav_np.ndim == 2: wav_np = wav_np.mean(axis=0)
        wav_np = wav_np.astype(np.float32)

        rw = width  // 2 if low_memory else width
        rh = height // 2 if low_memory else height

        print(f"[S42P Visualizer v6.0] mode={mode}  {len(wav_np)/sr:.1f}s  "
              f"{rw}x{rh}  max_frames={max_frames}")

        bass_e, mid_e, high_e, beat, n_frames = _extract_energy(wav_np, sr, fps, max_frames)
        print(f"[S42P Visualizer] Rendering {n_frames} frames ...")

        pal    = PALETTES.get(palette, PALETTES["Spectrum Classic"])
        state  = {}
        frames = []
        random.seed(seed)

        for i in range(n_frames):
            wc   = _wave_chunk(wav_np, sr, i, fps, n_frames)
            arr, state = _render_frame(
                mode, rw, rh,
                float(bass_e[i]), float(mid_e[i]), float(high_e[i]),
                float(beat[i]), wc, i, pal, bar_count, state
            )

            if low_memory and (rw != width or rh != height):
                up = Image.fromarray((arr*255).clip(0,255).astype(np.uint8))
                up = up.resize((width, height), Image.BILINEAR)
                arr = np.array(up, dtype=np.float32) / 255.0

            frames.append(arr)

            if i > 0 and i % 200 == 0:
                gc.collect()
                print(f"[S42P Visualizer]   {i}/{n_frames}")

        batch = np.stack(frames, axis=0).astype(np.float32)
        frames.clear(); gc.collect()
        print(f"[S42P Visualizer v6.0] ✓ {n_frames} frames  {batch.nbytes/1024/1024:.1f}MB")

        return (torch.from_numpy(batch) if TORCH_AVAILABLE else batch,)


NODE_CLASS_MAPPINGS        = {"S42PAudioVisualizer": S42PAudioVisualizer}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PAudioVisualizer": "S42P Audio Visualizer 🎨"}
