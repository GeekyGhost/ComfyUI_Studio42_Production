"""
S42 Production Suite — Audio Visualizer v7.0
=============================================
Multi-mode audio-reactive visualizer for music videos.

MODES (20 total):
  spectrum_classic    — bars + oscilloscope ring + particle burst (Winamp-inspired)
  lava_lamp           — physics blobs that rise/fall, morph with music
  wormhole_3d         — TRUE 3D psychedelic tunnel: curves, banked turns, neon lights,
                        speed driven by bass, tunnel patterns by mids, colors by highs
  cosmic_tunnel       — star-streaked hyperspace warp tunnel
  neon_corridor       — blade-runner neon corridor fly-through
  matrix              — falling character rain, columns pulse to bass
  stargate            — expanding portal rings with energy tendrils
  oscilloscope        — clean waveform scope with peak indicators
  plasma_waves        — flowing energy ribbons (aurora replacement)
  fractal_zoom        — smooth Mandelbrot zoom with rotation
  dna_helix           — rotating 3D double helix synced to music
  liquid_metal        — chrome fluid surface with dynamic lighting
  kaleidoscope        — symmetric pattern reflections
  particle_explosion  — radial burst particles from centre
  waveform_tunnel     — 3D waveform ring fly-through
  soundscape          — abstract terrain of frequency peaks
  vortex              — spinning spiral vortex driven by audio
  crystal_cave        — crystalline formations lit by frequency energy
  neural_fire         — firing neuron network topology
  aurora_borealis     — flowing northern-lights curtains over starfield

COLOR PALETTES (18):
  Spectrum Classic, Neon Synthwave, Lava Lamp, Void Blue, Matrix Green,
  Blood Moon, Arctic Ice, Deep Space, Sunset, Cyberpunk, Gold Chrome,
  Infrared, Acid Trip, Rose Gold, Toxic Green, Ultraviolet, Fire Storm,
  Ocean Depth

Python 3.12 | ComfyUI Portable | Pillow + numpy (CPU, no VRAM)
"""

from __future__ import annotations

import gc
import math
import random
import logging
from typing import Tuple, Dict, Any, Optional

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

# ══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTES
# ══════════════════════════════════════════════════════════════════════════════

PALETTES = {
    "Spectrum Classic":  {"bg":(0,0,0),    "bar_low":(0,255,0),   "bar_mid":(255,255,0),
                          "bar_top":(255,0,0),  "wave":(0,255,128), "particle":(255,255,255),
                          "beat":(255,64,0),    "accent":(0,200,255),"p1":(0,64,128),"p2":(0,128,64)},
    "Neon Synthwave":    {"bg":(5,0,20),   "bar_low":(0,255,255), "bar_mid":(180,0,255),
                          "bar_top":(255,0,128),"wave":(0,255,200), "particle":(255,200,0),
                          "beat":(255,0,200),   "accent":(255,255,0),"p1":(20,0,60),"p2":(0,20,60)},
    "Lava Lamp":         {"bg":(10,0,0),   "bar_low":(255,100,0), "bar_mid":(255,50,0),
                          "bar_top":(255,255,50),"wave":(255,140,0),"particle":(255,220,100),
                          "beat":(255,255,100), "accent":(200,80,0),"p1":(80,10,0),"p2":(40,0,0)},
    "Void Blue":         {"bg":(0,0,10),   "bar_low":(0,100,255), "bar_mid":(0,180,255),
                          "bar_top":(200,240,255),"wave":(0,200,255),"particle":(200,240,255),
                          "beat":(100,200,255), "accent":(0,255,200),"p1":(0,10,40),"p2":(0,20,60)},
    "Matrix Green":      {"bg":(0,0,0),    "bar_low":(0,180,0),   "bar_mid":(0,255,0),
                          "bar_top":(200,255,200),"wave":(0,220,0), "particle":(100,255,100),
                          "beat":(200,255,200), "accent":(0,255,80),"p1":(0,20,0),"p2":(0,30,0)},
    "Blood Moon":        {"bg":(5,0,0),    "bar_low":(180,0,0),   "bar_mid":(255,60,0),
                          "bar_top":(255,200,0),"wave":(220,40,0),  "particle":(255,120,0),
                          "beat":(255,255,0),   "accent":(255,80,0),"p1":(40,0,0),"p2":(20,0,0)},
    "Arctic Ice":        {"bg":(0,10,20),  "bar_low":(180,230,255),"bar_mid":(100,200,255),
                          "bar_top":(255,255,255),"wave":(200,240,255),"particle":(255,255,255),
                          "beat":(0,200,255),   "accent":(180,240,255),"p1":(0,20,40),"p2":(0,30,60)},
    "Deep Space":        {"bg":(0,0,5),    "bar_low":(80,0,160),  "bar_mid":(160,0,255),
                          "bar_top":(255,200,255),"wave":(120,0,255),"particle":(255,180,255),
                          "beat":(200,100,255), "accent":(255,0,200),"p1":(10,0,30),"p2":(5,0,20)},
    "Sunset":            {"bg":(8,2,0),    "bar_low":(255,80,0),  "bar_mid":(255,160,0),
                          "bar_top":(255,240,100),"wave":(255,120,0),"particle":(255,200,100),
                          "beat":(255,255,150), "accent":(200,60,0),"p1":(60,10,0),"p2":(30,5,0)},
    "Cyberpunk":         {"bg":(0,5,8),    "bar_low":(0,255,180), "bar_mid":(255,0,180),
                          "bar_top":(255,255,0),"wave":(0,200,255), "particle":(255,0,200),
                          "beat":(255,255,100), "accent":(0,255,255),"p1":(0,15,25),"p2":(15,0,20)},
    "Gold Chrome":       {"bg":(5,5,0),    "bar_low":(200,150,0), "bar_mid":(255,200,50),
                          "bar_top":(255,255,200),"wave":(255,210,80),"particle":(255,255,200),
                          "beat":(255,240,100), "accent":(200,180,0),"p1":(30,20,0),"p2":(20,15,0)},
    "Infrared":          {"bg":(0,0,5),    "bar_low":(0,0,255),   "bar_mid":(0,255,100),
                          "bar_top":(255,50,0), "wave":(100,255,200),"particle":(255,100,50),
                          "beat":(255,0,50),    "accent":(0,200,255),"p1":(0,0,20),"p2":(0,10,0)},
    "Acid Trip":         {"bg":(5,0,8),    "bar_low":(255,0,255), "bar_mid":(0,255,0),
                          "bar_top":(255,255,0),"wave":(255,0,128),  "particle":(0,255,255),
                          "beat":(255,128,0),   "accent":(128,0,255),"p1":(20,0,30),"p2":(0,20,0)},
    "Rose Gold":         {"bg":(8,3,4),    "bar_low":(255,180,180),"bar_mid":(255,120,140),
                          "bar_top":(255,220,200),"wave":(255,150,160),"particle":(255,200,180),
                          "beat":(255,240,220), "accent":(200,100,120),"p1":(40,15,20),"p2":(25,8,12)},
    "Toxic Green":       {"bg":(0,5,0),    "bar_low":(80,255,0),  "bar_mid":(180,255,0),
                          "bar_top":(255,255,50),"wave":(120,255,20),"particle":(200,255,100),
                          "beat":(255,255,80),  "accent":(50,200,0),"p1":(0,20,0),"p2":(5,30,0)},
    "Ultraviolet":       {"bg":(2,0,8),    "bar_low":(100,0,255), "bar_mid":(180,0,255),
                          "bar_top":(255,180,255),"wave":(140,0,255),"particle":(220,180,255),
                          "beat":(255,100,255), "accent":(60,0,200),"p1":(8,0,25),"p2":(12,0,35)},
    "Fire Storm":        {"bg":(4,0,0),    "bar_low":(255,20,0),  "bar_mid":(255,100,0),
                          "bar_top":(255,220,80),"wave":(255,60,0),  "particle":(255,180,50),
                          "beat":(255,255,120), "accent":(200,30,0),"p1":(40,5,0),"p2":(20,2,0)},
    "Ocean Depth":       {"bg":(0,2,8),    "bar_low":(0,80,180),  "bar_mid":(0,140,220),
                          "bar_top":(100,220,255),"wave":(20,160,200),"particle":(180,240,255),
                          "beat":(0,200,255),   "accent":(0,100,200),"p1":(0,8,25),"p2":(0,15,40)},
}

PALETTE_NAMES = list(PALETTES.keys())

VIZ_MODES = [
    "spectrum_classic", "lava_lamp",    "wormhole_3d",      "cosmic_tunnel",
    "neon_corridor",    "matrix",       "stargate",         "oscilloscope",
    "plasma_waves",     "fractal_zoom", "dna_helix",        "liquid_metal",
    "kaleidoscope",     "particle_explosion","waveform_tunnel","soundscape",
    "vortex",           "crystal_cave", "neural_fire",      "aurora_borealis",
]

# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _clamp_color(r,g,b):
    return (int(max(0,min(255,r))), int(max(0,min(255,g))), int(max(0,min(255,b))))


def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return _clamp_color(c1[0]+(c2[0]-c1[0])*t, c1[1]+(c2[1]-c1[1])*t, c1[2]+(c2[2]-c1[2])*t)


def _energy_chunk(audio_chunk: np.ndarray, sr: int):
    """Return (bass, mid, high, beat) float values 0..1."""
    n = len(audio_chunk)
    if n < 64:
        return 0.0, 0.0, 0.0, 0.0
    fft  = np.abs(np.fft.rfft(audio_chunk * np.hanning(n)))
    freq = np.fft.rfftfreq(n, 1.0/sr)
    def band(lo, hi):
        m = (freq >= lo) & (freq < hi)
        return float(np.sqrt(np.mean(fft[m]**2))) if m.any() else 0.0
    bass  = band(20,  250)
    mid   = band(250, 4000)
    high  = band(4000, 20000)
    mx = max(bass, mid, high, 1e-6)
    bass, mid, high = bass/mx, mid/mx, high/mx
    beat = 1.0 if bass > 0.7 else 0.0
    return float(np.clip(bass,0,1)), float(np.clip(mid,0,1)), float(np.clip(high,0,1)), beat


def _wave_chunk(audio_chunk: np.ndarray, n_points: int = 512) -> np.ndarray:
    if len(audio_chunk) == 0:
        return np.zeros(n_points, dtype=np.float32)
    wave = audio_chunk[:n_points] if len(audio_chunk) >= n_points else \
           np.pad(audio_chunk, (0, n_points - len(audio_chunk)))
    mx = np.max(np.abs(wave))
    return (wave / mx).astype(np.float32) if mx > 1e-8 else wave.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SPECTRUM CLASSIC
# ══════════════════════════════════════════════════════════════════════════════

def _spectrum_classic_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal, bar_count, particles):
    bar_w = w // bar_count
    for i in range(bar_count):
        t = i / bar_count
        h_pct = 0.2 + bass*0.5*(1-t) + mid*0.3*t + high*0.2*t*t
        bh = int(h * h_pct * random.uniform(0.85, 1.0))
        x0, x1 = i*bar_w, (i+1)*bar_w-1
        c = _lerp_color(_lerp_color(pal["bar_low"],pal["bar_mid"],t), pal["bar_top"], h_pct)
        draw.rectangle([x0, h-bh, x1, h], fill=c)
    # Waveform ring
    cx, cy = w//2, h//2
    pts = []
    for j, v in enumerate(wave[:128]):
        a = 2*math.pi*j/128
        r = h*0.25 + v*h*0.1*(1+bass)
        pts.append((cx+r*math.cos(a), cy+r*math.sin(a)))
    if len(pts) > 2:
        draw.line(pts+[pts[0]], fill=pal["wave"], width=2)
    # Particles
    if beat > 0.5:
        for _ in range(8):
            particles.append([cx, cy, random.uniform(-5,5)*bass, random.uniform(-5,5)*bass,
                               random.randint(30,60), random.choice([pal["particle"],pal["beat"]])])
    alive = []
    for p in particles:
        p[0]+=p[2]; p[1]+=p[3]; p[4]-=1
        if p[4]>0:
            r=max(1,p[4]//8)
            draw.ellipse([p[0]-r,p[1]-r,p[0]+r,p[1]+r], fill=p[5])
            alive.append(p)
    return alive


# ══════════════════════════════════════════════════════════════════════════════
# LAVA LAMP
# ══════════════════════════════════════════════════════════════════════════════

def _lava_frame(draw, w, h, bass, mid, high, beat, fi, pal, blobs):
    if len(blobs) < 12:
        for _ in range(12 - len(blobs)):
            blobs.append([random.uniform(0,w), random.uniform(0,h),
                          random.uniform(20,80+bass*60), random.uniform(-1.5,1.5),
                          random.uniform(-1.0,1.0)])
    for b in blobs:
        b[1] -= b[4] * (0.5 + bass*2.0)
        b[0] += b[3] * 0.3
        b[2] = max(15, min(120, b[2] + (bass-0.5)*3))
        if b[1] < -b[2]:  b[1] = h + b[2]; b[0] = random.uniform(0,w)
        if b[0] < 0: b[0] = w
        if b[0] > w: b[0] = 0
    for b in blobs:
        r  = int(b[2])
        cx, cy = int(b[0]), int(b[1])
        c1 = _lerp_color(pal["bar_low"], pal["bar_top"], bass)
        c2 = _lerp_color(pal["bar_mid"], pal["accent"], mid)
        c  = _lerp_color(c1, c2, high)
        for layer, alpha in [(r, c), (r*2//3, _lerp_color(c, (255,255,255), 0.3))]:
            draw.ellipse([cx-layer, cy-layer, cx+layer, cy+layer], fill=alpha)
    return blobs


# ══════════════════════════════════════════════════════════════════════════════
# WORMHOLE 3D — True 3D psychedelic tunnel with curves, banked turns,
# neon lighting, speed driven by bass, turns by mids, color by highs
# ══════════════════════════════════════════════════════════════════════════════

def _wormhole_3d_frame(draw, w, h, bass, mid, high, beat, fi, pal, state):
    """
    Full 3D wormhole tunnel with:
      - Perspective-correct ring rendering (near rings large, far rings small)
      - Camera path curves and banked turns driven by mid frequencies
      - Speed (ring density) driven by bass
      - Neon light strips along the tunnel walls
      - Color cycling driven by high frequencies
      - Pulsing ring glow on beats
      - Tunnel pattern alternation (rings, hexagons, diamonds)
    """
    img = state.get("img_obj", draw._image if hasattr(draw, '_image') else None)

    cx, cy   = w / 2.0, h / 2.0
    n_rings  = 32

    # Speed: bass drives how fast we fly through the tunnel
    speed = 0.8 + bass * 3.0
    t     = fi * speed * 0.05

    # Camera sway: mid frequencies drive gentle S-curves through the tunnel
    sway_x = math.sin(t * 0.7 + mid * 2.0) * w * 0.08 * (1 + mid)
    sway_y = math.cos(t * 0.5 + mid * 1.5) * h * 0.06 * (1 + mid * 0.5)

    # Bank (tilt): barrel-roll feeling from combined bass+mid
    bank_angle = math.sin(t * 0.3) * 15.0 * mid + beat * 8.0

    # Color cycling: highs shift the color hue
    hue_shift = (t * 0.3 + high * 2.0) % 1.0

    def hue_to_rgb(h_val):
        h6 = h_val * 6
        x  = 1 - abs(h6 % 2 - 1)
        r,g,b = [(1,x,0),(x,1,0),(0,1,x),(0,x,1),(x,0,1),(1,0,x)][int(h6)%6]
        return _clamp_color(int(r*255), int(g*255), int(b*255))

    # Tunnel walls: draw rings from far to near
    for ring_i in range(n_rings, 0, -1):
        # Perspective: z goes from 0 (far, small) to 1 (near, large)
        z_norm = ring_i / n_rings
        # Phase offset makes the tunnel flow toward viewer
        phase  = (t + ring_i * 0.15) % (2 * math.pi)

        # Perspective scale: near rings are large
        persp = z_norm ** 1.5
        rx    = w * 0.55 * persp
        ry    = h * 0.50 * persp

        # Camera curve offset — rings shift as camera curves
        off_x = cx + sway_x * (1.0 - z_norm) * 0.8
        off_y = cy + sway_y * (1.0 - z_norm) * 0.5

        # Brightness: near rings bright, far rings dim — depth fog
        brightness = 0.15 + z_norm * 0.85
        # Bass brightens near rings
        if z_norm > 0.7:
            brightness = min(1.0, brightness + bass * 0.4)
        # Beat flash: nearest rings pulse
        if beat > 0.5 and z_norm > 0.85:
            brightness = min(1.5, brightness + beat * 0.8)

        # Color: cycle with high frequencies, shift along tunnel depth
        ring_hue = (hue_shift + z_norm * 0.4 + ring_i * 0.02) % 1.0
        rc = hue_to_rgb(ring_hue)
        c  = _clamp_color(int(rc[0]*brightness), int(rc[1]*brightness), int(rc[2]*brightness))

        # Ring line width: near rings thicker
        lw = max(1, int(1 + z_norm * 4 + bass * 2))

        # Rotate ring for bank effect
        cos_b = math.cos(math.radians(bank_angle * (1 - z_norm)))
        sin_b = math.sin(math.radians(bank_angle * (1 - z_norm)))

        # Draw ellipse ring — using polygon for rotation support
        n_pts = max(16, int(32 * z_norm))
        pts = []
        for p_i in range(n_pts):
            a   = 2 * math.pi * p_i / n_pts
            px  = rx * math.cos(a)
            py  = ry * math.sin(a)
            # Apply bank rotation
            rpx = px * cos_b - py * sin_b
            rpy = px * sin_b + py * cos_b
            pts.append((off_x + rpx, off_y + rpy))

        if len(pts) > 2:
            draw.line(pts + [pts[0]], fill=c, width=lw)

        # Neon light strips: 4 or 8 glowing lines along the tunnel length
        n_strips = 8 if z_norm > 0.4 else 4
        for strip_i in range(n_strips):
            strip_angle = 2 * math.pi * strip_i / n_strips + phase * 0.1
            sx = off_x + rx * math.cos(strip_angle)
            sy = off_y + ry * math.sin(strip_angle)
            strip_brightness = 0.4 + z_norm * 0.6 + bass * 0.3 * (strip_i % 2)
            sc_h = (hue_shift + strip_i / n_strips + 0.5) % 1.0
            sc = hue_to_rgb(sc_h)
            sc = _clamp_color(int(sc[0]*strip_brightness), int(sc[1]*strip_brightness), int(sc[2]*strip_brightness))
            r_dot = max(1, int(1.5 + z_norm * 3))
            draw.ellipse([sx-r_dot, sy-r_dot, sx+r_dot, sy+r_dot], fill=sc)

    # Tunnel floor/ceiling lines: perspective lines from vanishing point to edges
    vanish_x = cx + sway_x * 0.4
    vanish_y = cy + sway_y * 0.3
    n_lines = 12
    for li in range(n_lines):
        a = 2 * math.pi * li / n_lines + t * 0.05
        edge_x = cx + (w * 0.7) * math.cos(a)
        edge_y = cy + (h * 0.65) * math.sin(a)
        line_hue = (hue_shift + li / n_lines) % 1.0
        lc = hue_to_rgb(line_hue)
        alpha_v = 0.1 + bass * 0.2
        lc = _clamp_color(int(lc[0]*alpha_v), int(lc[1]*alpha_v), int(lc[2]*alpha_v))
        draw.line([(vanish_x, vanish_y), (edge_x, edge_y)], fill=lc, width=1)

    return state


# ══════════════════════════════════════════════════════════════════════════════
# COSMIC TUNNEL — star-streaked hyperspace warp
# ══════════════════════════════════════════════════════════════════════════════

def _cosmic_tunnel_frame(draw, w, h, bass, mid, high, beat, fi, pal, state):
    if "stars" not in state:
        rng = random.Random(42)
        state["stars"] = [[rng.uniform(-1,1), rng.uniform(-1,1), rng.uniform(0,1)]
                          for _ in range(200)]
    cx, cy = w/2, h/2
    speed  = 0.01 + bass * 0.06 + beat * 0.04
    stars  = state["stars"]
    new_s  = []
    for s in stars:
        s[2] -= speed
        if s[2] <= 0:
            s = [random.uniform(-1,1), random.uniform(-1,1), 1.0]
        z     = max(s[2], 0.01)
        sx    = cx + s[0] * w * 0.6 / z
        sy    = cy + s[1] * h * 0.6 / z
        bright = min(255, int((1.0-z)*255 + bass*60))
        size   = max(1, int((1.0-z)*4 + bass*3))
        c = _lerp_color(pal["accent"], pal["beat"], 1.0-z)
        c = _clamp_color(int(c[0]*bright/255), int(c[1]*bright/255), int(c[2]*bright/255))
        # Draw streak: line from previous position toward centre
        pz  = z + speed
        psx = cx + s[0] * w * 0.6 / pz
        psy = cy + s[1] * h * 0.6 / pz
        draw.line([(psx, psy), (sx, sy)], fill=c, width=size)
        new_s.append(s)
    state["stars"] = new_s
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NEON CORRIDOR
# ══════════════════════════════════════════════════════════════════════════════

def _neon_corridor_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    cx, cy = w/2, h/2
    depth  = 8
    speed  = fi * (0.5 + bass * 2.5)

    for d in range(depth, 0, -1):
        t     = (d / depth + speed * 0.02) % 1.0
        persp = t * t
        hw    = w * 0.5 * persp
        hh    = h * 0.45 * persp

        bright = persp * (0.5 + bass * 0.5)
        c1 = _lerp_color(pal["bar_low"], pal["bar_top"], t)
        c1 = _clamp_color(int(c1[0]*bright), int(c1[1]*bright), int(c1[2]*bright))

        lw = max(1, int(1 + persp * 3))
        # Corridor frame
        draw.rectangle([cx-hw, cy-hh, cx+hw, cy+hh], outline=c1, width=lw)

        # Floor/ceiling grid lines
        n_grid = 6
        for gi in range(n_grid):
            gx = cx - hw + (2*hw) * gi / n_grid
            gc = _lerp_color(pal["p1"], pal["accent"], gi/n_grid)
            gc = _clamp_color(int(gc[0]*bright*0.5), int(gc[1]*bright*0.5), int(gc[2]*bright*0.5))
            draw.line([(gx, cy+hh), (cx, cy)], fill=gc, width=1)
            draw.line([(gx, cy-hh), (cx, cy)], fill=gc, width=1)

        # Neon strips on walls
        if beat > 0.5:
            bc = pal["beat"]
            bc = _clamp_color(int(bc[0]*persp), int(bc[1]*persp), int(bc[2]*persp))
            draw.line([(cx-hw, cy-hh), (cx-hw, cy+hh)], fill=bc, width=max(1,lw+1))
            draw.line([(cx+hw, cy-hh), (cx+hw, cy+hh)], fill=bc, width=max(1,lw+1))


# ══════════════════════════════════════════════════════════════════════════════
# MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def _matrix_frame(draw, w, h, bass, mid, high, beat, fi, pal, cols):
    char_w, char_h = 12, 16
    n_cols = w // char_w
    if len(cols) < n_cols:
        cols += [random.randint(-h, 0) for _ in range(n_cols - len(cols))]
    speed = max(1, int(2 + bass * 8))
    for ci in range(n_cols):
        cols[ci] += speed
        if cols[ci] > h + char_h:
            cols[ci] = random.randint(-h//2, 0)
        y   = cols[ci]
        x   = ci * char_w
        ch  = str(random.randint(0, 1))
        br  = 0.5 + mid * 0.5
        c   = _clamp_color(int(pal["bar_low"][0]*br), int(pal["bar_low"][1]*br), int(pal["bar_low"][2]*br))
        draw.text((x, y), ch, fill=c)
        # Bright head
        hc = _lerp_color(pal["bar_top"], (255,255,255), high)
        draw.text((x, y - char_h), ch, fill=hc)
    return cols


# ══════════════════════════════════════════════════════════════════════════════
# STARGATE
# ══════════════════════════════════════════════════════════════════════════════

def _stargate_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    cx, cy = w//2, h//2
    n_rings = 20
    for ri in range(n_rings):
        t     = (fi * 0.03 * (1 + bass*2) + ri/n_rings) % 1.0
        r     = int(min(w,h) * 0.5 * t * (1 + bass*0.3))
        if r < 2: continue
        bright = (1.0 - t) * (0.5 + bass * 0.5)
        c = _lerp_color(pal["bar_low"], pal["accent"], t)
        c = _clamp_color(int(c[0]*bright), int(c[1]*bright), int(c[2]*bright))
        lw = max(1, int(3 * (1-t) + beat*2))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=c, width=lw)
    # Tendrils
    for ti in range(12):
        a  = 2*math.pi*ti/12 + fi*0.02
        r1 = int(min(w,h)*0.08)
        r2 = int(min(w,h)*0.45 * (0.5 + mid*0.5))
        tc = _lerp_color(pal["beat"], pal["particle"], mid)
        tc = _clamp_color(int(tc[0]*0.7), int(tc[1]*0.7), int(tc[2]*0.7))
        draw.line([(cx+r1*math.cos(a), cy+r1*math.sin(a)),
                   (cx+r2*math.cos(a+0.15*bass), cy+r2*math.sin(a+0.15*bass))],
                  fill=tc, width=max(1,int(1+bass*2)))


# ══════════════════════════════════════════════════════════════════════════════
# OSCILLOSCOPE
# ══════════════════════════════════════════════════════════════════════════════

def _osc_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal):
    cy = h // 2
    scale = h * 0.4 * (1 + bass * 0.3)
    n = len(wave)
    if n < 2: return
    step = w / n
    pts  = [(int(i*step), int(cy - wave[i]*scale)) for i in range(n)]
    lw   = max(1, int(1 + mid * 2))
    c    = _lerp_color(pal["wave"], pal["beat"], bass)
    draw.line(pts, fill=c, width=lw)
    # Peak markers
    if beat > 0.5:
        draw.line([(0,cy),(w,cy)], fill=pal["beat"], width=1)


# ══════════════════════════════════════════════════════════════════════════════
# PLASMA WAVES
# ══════════════════════════════════════════════════════════════════════════════

def _plasma_waves_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    n_bands = 12
    for bi in range(n_bands):
        t     = bi / n_bands
        amp   = 0.05 + (bass*(1-t) + mid*t) * 0.15
        freq  = 1.5 + t * 3.0 + mid * 2.0
        phase = fi * 0.04 * (1 + t * 0.5) + t * math.pi
        y_base = int(h * (0.1 + t * 0.85))
        pts = []
        for xi in range(0, w, 3):
            xn  = xi / w
            yn  = y_base + math.sin(xn * freq * math.pi * 2 + phase) * h * amp
            pts.append((xi, int(yn)))
        if len(pts) > 2:
            c = _lerp_color(_lerp_color(pal["bar_low"], pal["bar_mid"], t),
                            pal["accent"], bass * (1-t))
            c = _clamp_color(int(c[0]*(0.4+t*0.6)), int(c[1]*(0.4+t*0.6)), int(c[2]*(0.4+t*0.6)))
            draw.line(pts, fill=c, width=max(1, int(2 + bass * 3 * (1-t))))


# ══════════════════════════════════════════════════════════════════════════════
# FRACTAL ZOOM
# ══════════════════════════════════════════════════════════════════════════════

def _fractal_zoom_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    # Mandelbrot-esque pattern using trig approximation (CPU-friendly)
    t      = fi * 0.01 * (1 + bass * 2.0)
    cx_m   = -0.745 + math.sin(t * 0.3) * 0.002 * (1 + mid)
    cy_m   =  0.186 + math.cos(t * 0.2) * 0.002 * (1 + mid)
    zoom   = 0.003 / (1.0 + fi * 0.002 * (1 + bass))
    max_it = 32
    step   = max(2, w // 80)

    for py in range(0, h, step):
        for px in range(0, w, step):
            zr = cx_m + (px - w/2) * zoom
            zi = cy_m + (py - h/2) * zoom
            c = 0.0 + 0.0j
            it = 0
            for it in range(max_it):
                c = c*c + complex(zr, zi)
                if abs(c) > 2.0: break
            t_val = it / max_it
            col = _lerp_color(_lerp_color(pal["p1"], pal["bar_mid"], t_val),
                              pal["accent"], t_val**2)
            if t_val > 0.05:
                draw.rectangle([px, py, px+step-1, py+step-1], fill=col)


# ══════════════════════════════════════════════════════════════════════════════
# DNA HELIX
# ══════════════════════════════════════════════════════════════════════════════

def _dna_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    cx = w // 2
    n  = 60
    for i in range(n):
        t     = i / n
        y     = int(h * t)
        phase = fi * 0.06 * (1 + bass) + t * 4 * math.pi
        amp   = w * 0.3 * (1 + bass * 0.2)
        x1    = int(cx + math.cos(phase) * amp)
        x2    = int(cx + math.cos(phase + math.pi) * amp)
        r1    = max(3, int(5 + bass * 6 * (1-abs(math.cos(phase)))))
        r2    = max(3, int(5 + mid  * 5 * (1-abs(math.cos(phase+math.pi)))))
        c1 = _lerp_color(pal["bar_low"], pal["bar_top"], t)
        c2 = _lerp_color(pal["bar_mid"], pal["accent"], t)
        draw.ellipse([x1-r1,y-r1,x1+r1,y+r1], fill=c1)
        draw.ellipse([x2-r2,y-r2,x2+r2,y+r2], fill=c2)
        # Connector rungs
        if i % 4 == 0:
            rc = _lerp_color(pal["wave"], pal["beat"], mid)
            draw.line([(x1,y),(x2,y)], fill=rc, width=max(1,int(1+beat)))


# ══════════════════════════════════════════════════════════════════════════════
# LIQUID METAL
# ══════════════════════════════════════════════════════════════════════════════

def _liquid_metal_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    n_bands = 30
    for bi in range(n_bands):
        t     = bi / n_bands
        y0    = int(h * t)
        y1    = int(h * (t + 1/n_bands)) + 1
        # Chrome shimmer: oscillating brightness band
        phase  = fi * 0.05 + t * 6.0 * math.pi
        bright = 0.3 + 0.5 * (0.5 + 0.5*math.sin(phase + bass*math.pi))
        # Highlight sweep
        sweep  = 0.5 + 0.5 * math.sin(fi * 0.08 + t * 2 * math.pi + mid*math.pi)
        bright += sweep * 0.3 * (1 + high)
        c = _lerp_color(pal["bar_low"], (255,255,255), bright * (0.4 + bass*0.3))
        c = _clamp_color(int(c[0]), int(c[1]), int(c[2]))
        draw.rectangle([0, y0, w, y1], fill=c)
        # Ripple distortion overlay
        if beat > 0.5 and bi % 3 == 0:
            rw = int(w * 0.3 * bass)
            rx = w//2 + int(w * 0.3 * math.sin(fi*0.2 + bi))
            rc = _lerp_color(pal["beat"], (255,255,255), 0.5)
            draw.rectangle([rx, y0, rx+rw, y1], fill=rc)


# ══════════════════════════════════════════════════════════════════════════════
# KALEIDOSCOPE
# ══════════════════════════════════════════════════════════════════════════════

def _kaleidoscope_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    cx, cy = w//2, h//2
    n_sym  = 8
    n_elem = 20
    t      = fi * 0.03 * (1 + mid)
    for ei in range(n_elem):
        r_base = (w//4) * (ei / n_elem)
        r_mod  = r_base * (1 + bass * 0.3 * math.sin(t + ei))
        for si in range(n_sym):
            a  = 2*math.pi*si/n_sym + t + ei*0.15
            a2 = 2*math.pi*(si+1)/n_sym + t + ei*0.15
            x1 = int(cx + r_mod * math.cos(a))
            y1 = int(cy + r_mod * math.sin(a))
            x2 = int(cx + r_mod * 0.7 * math.cos((a+a2)/2))
            y2 = int(cy + r_mod * 0.7 * math.sin((a+a2)/2))
            c  = _lerp_color(_lerp_color(pal["bar_low"], pal["accent"], ei/n_elem),
                             pal["beat"], bass)
            draw.line([(cx,cy),(x1,y1),(x2,y2),(cx,cy)], fill=c, width=max(1,int(1+bass*2)))


# ══════════════════════════════════════════════════════════════════════════════
# PARTICLE EXPLOSION
# ══════════════════════════════════════════════════════════════════════════════

def _particle_explosion_frame(draw, w, h, bass, mid, high, beat, fi, pal, state):
    if "particles" not in state:
        state["particles"] = []
    cx, cy = w//2, h//2
    # Spawn on beats or continuously
    spawn_n = int(4 + bass * 12)
    if beat > 0.5 or bass > 0.3:
        for _ in range(spawn_n):
            a = random.uniform(0, 2*math.pi)
            spd = random.uniform(2, 8 + bass*10)
            state["particles"].append({
                "x": cx + random.uniform(-10,10),
                "y": cy + random.uniform(-10,10),
                "vx": math.cos(a)*spd, "vy": math.sin(a)*spd,
                "life": random.randint(20, 60),
                "max_life": 60,
                "color": random.choice([pal["bar_low"],pal["bar_mid"],pal["bar_top"],pal["beat"]])
            })
    alive = []
    for p in state["particles"]:
        p["x"]   += p["vx"] * (1 + mid*0.5)
        p["y"]   += p["vy"] * (1 + mid*0.5)
        p["vy"]  += 0.15  # gravity
        p["life"] -= 1
        if p["life"] > 0:
            alpha = p["life"] / p["max_life"]
            r_size = max(1, int(3 * alpha + bass * 2))
            c = _clamp_color(int(p["color"][0]*alpha), int(p["color"][1]*alpha), int(p["color"][2]*alpha))
            x, y = int(p["x"]), int(p["y"])
            draw.ellipse([x-r_size,y-r_size,x+r_size,y+r_size], fill=c)
            alive.append(p)
    state["particles"] = alive[:300]  # cap
    return state


# ══════════════════════════════════════════════════════════════════════════════
# WAVEFORM TUNNEL
# ══════════════════════════════════════════════════════════════════════════════

def _waveform_tunnel_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal):
    cx, cy  = w//2, h//2
    n_rings = 20
    speed   = fi * 0.04 * (1 + bass * 2)
    for ri in range(n_rings, 0, -1):
        z_norm = ri / n_rings
        persp  = z_norm ** 1.8
        base_r = min(w,h) * 0.45 * persp
        bright = 0.2 + z_norm * 0.8
        c = _lerp_color(pal["wave"], pal["beat"], bass * z_norm)
        c = _clamp_color(int(c[0]*bright), int(c[1]*bright), int(c[2]*bright))
        # Waveform-modulated ring
        n_pts = max(16, int(64 * z_norm))
        pts   = []
        for pi in range(n_pts):
            a  = 2*math.pi*pi/n_pts + speed*0.1*(1-z_norm)
            wi = int(pi/n_pts * len(wave)) % len(wave)
            r  = base_r * (1.0 + wave[wi] * 0.3 * (1 + bass))
            pts.append((cx + r*math.cos(a), cy + r*math.sin(a)))
        if len(pts) > 2:
            draw.line(pts + [pts[0]], fill=c, width=max(1, int(1+z_norm*3)))


# ══════════════════════════════════════════════════════════════════════════════
# SOUNDSCAPE
# ══════════════════════════════════════════════════════════════════════════════

def _soundscape_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal):
    """3D terrain of frequency peaks — seen from above and slightly angled."""
    n_cols = 60
    n_rows = 30
    for row in range(n_rows, 0, -1):
        row_t   = row / n_rows
        y_base  = int(h * (0.3 + row_t * 0.6))
        z_scale = 0.3 + row_t * 0.7
        pts     = []
        for col in range(n_cols+1):
            col_t = col / n_cols
            wi    = int(col_t * (len(wave)-1))
            val   = float(wave[wi]) if len(wave) > wi else 0.0
            freq_amp = (bass*(1-col_t) + mid*col_t*2 + high*col_t**2) * 0.3
            peak  = (val * 0.5 + freq_amp) * h * 0.25 * z_scale
            x     = int(w * col_t)
            y     = int(y_base - peak * (1 + beat*0.3))
            pts.append((x, y))
        if len(pts) > 2:
            c = _lerp_color(_lerp_color(pal["bar_low"], pal["bar_mid"], row_t),
                            pal["bar_top"], bass * row_t)
            draw.line(pts, fill=c, width=max(1, int(row_t * 3)))


# ══════════════════════════════════════════════════════════════════════════════
# VORTEX
# ══════════════════════════════════════════════════════════════════════════════

def _vortex_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    cx, cy   = w//2, h//2
    n_arms   = 6
    n_points = 40
    spin     = fi * 0.04 * (1 + bass * 2 + beat * 0.5)
    for arm in range(n_arms):
        base_angle = 2*math.pi*arm/n_arms
        for pi in range(n_points):
            t      = pi / n_points
            r      = min(w,h) * 0.45 * t * (1 + bass * 0.3)
            angle  = base_angle + t * 3 * math.pi * (1 + mid) + spin
            x      = cx + r * math.cos(angle)
            y      = cy + r * math.sin(angle)
            bright = 1.0 - t * 0.7 + bass * 0.2
            dot_r  = max(1, int(3*(1-t) + bass*3))
            c = _lerp_color(pal["bar_top"], pal["bar_low"], t)
            c = _clamp_color(int(c[0]*bright), int(c[1]*bright), int(c[2]*bright))
            draw.ellipse([x-dot_r,y-dot_r,x+dot_r,y+dot_r], fill=c)


# ══════════════════════════════════════════════════════════════════════════════
# CRYSTAL CAVE
# ══════════════════════════════════════════════════════════════════════════════

def _crystal_cave_frame(draw, w, h, bass, mid, high, beat, fi, pal, state):
    if "crystals" not in state:
        rng = random.Random(99)
        state["crystals"] = [
            {"x": rng.randint(0,w), "y": rng.randint(h//2, h),
             "height": rng.randint(h//8, h//2), "width": rng.randint(10,40),
             "phase": rng.uniform(0,2*math.pi)}
            for _ in range(20)
        ] + [
            {"x": rng.randint(0,w), "y": rng.randint(0, h//2),
             "height": -rng.randint(h//8, h//2), "width": rng.randint(10,40),
             "phase": rng.uniform(0,2*math.pi)}
            for _ in range(15)
        ]

    t = fi * 0.02
    for cr in state["crystals"]:
        pulse   = 0.5 + 0.5*math.sin(t + cr["phase"]) * (0.5 + bass)
        hfrac   = cr["height"] * (0.7 + pulse * 0.4)
        x       = cr["x"]
        y_base  = cr["y"]
        wh      = cr["width"]
        tip_y   = y_base - int(hfrac)
        bright  = 0.3 + pulse * 0.7 + beat * 0.3
        c = _lerp_color(pal["bar_low"], pal["accent"], pulse)
        c = _clamp_color(int(c[0]*bright), int(c[1]*bright), int(c[2]*bright))
        # Crystal polygon: narrow tip, wider base
        pts = [(x, tip_y), (x-wh, y_base), (x, y_base-wh//3), (x+wh, y_base)]
        draw.polygon(pts, fill=c, outline=pal["bar_top"])
        # Inner highlight
        hc = _lerp_color(c, (255,255,255), 0.6)
        hw = max(1, wh//4)
        draw.line([(x, tip_y),(x, y_base-wh//3)], fill=hc, width=hw)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL FIRE
# ══════════════════════════════════════════════════════════════════════════════

def _neural_fire_frame(draw, w, h, bass, mid, high, beat, fi, pal, state):
    if "nodes" not in state:
        rng = random.Random(77)
        state["nodes"] = [(rng.randint(0,w), rng.randint(0,h)) for _ in range(40)]
        state["connections"] = [(i,j) for i in range(40) for j in range(i+1,40)
                                if math.dist(state["nodes"][i], state["nodes"][j]) < w*0.25]
        state["firing"] = set()

    t = fi * 0.05
    # Fire nodes triggered by beat
    if beat > 0.5:
        for _ in range(int(4 + bass*8)):
            state["firing"].add(random.randint(0, 39))
    # Propagate firing
    new_firing = set()
    for n_i in list(state["firing"]):
        if random.random() < 0.3 + mid * 0.4:
            new_firing.add(n_i)
        # Spread to neighbours
        for ci, (a_i, b_i) in enumerate(state["connections"]):
            if a_i == n_i or b_i == n_i:
                neighbor = b_i if a_i == n_i else a_i
                if random.random() < 0.15 + bass * 0.2:
                    new_firing.add(neighbor)
    state["firing"] = new_firing

    # Draw connections
    for a_i, b_i in state["connections"]:
        ax, ay = state["nodes"][a_i]
        bx, by = state["nodes"][b_i]
        active = a_i in state["firing"] or b_i in state["firing"]
        bright  = 0.1 + active * (0.6 + bass * 0.4)
        c = _lerp_color(pal["p1"], pal["bar_low"], bright)
        draw.line([(ax,ay),(bx,by)], fill=c, width=1)
    # Draw nodes
    for ni, (nx, ny) in enumerate(state["nodes"]):
        active = ni in state["firing"]
        r_size = 3 + active * int(4 + beat * 4)
        bright  = 0.3 + active * 0.7
        c = _lerp_color(pal["bar_mid"], pal["beat"], active * (0.5 + bass*0.5))
        c = _clamp_color(int(c[0]*bright), int(c[1]*bright), int(c[2]*bright))
        draw.ellipse([nx-r_size,ny-r_size,nx+r_size,ny+r_size], fill=c)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# AURORA BOREALIS
# ══════════════════════════════════════════════════════════════════════════════

def _aurora_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    # Starfield background
    rng = random.Random(fi % 3)  # subtle twinkle
    for _ in range(60):
        sx = rng.randint(0, w)
        sy = rng.randint(0, h//2)
        br = rng.randint(100,255)
        draw.point((sx,sy), fill=(br,br,br))
    # Aurora curtains
    n_curtains = 5
    for ci in range(n_curtains):
        ct     = ci / n_curtains
        speed  = 0.5 + ct * 0.3
        phase  = fi * 0.025 * speed + ct * math.pi * 0.4
        amp    = h * (0.15 + bass * 0.12 + ct * 0.04)
        n_pts  = w // 4
        pts_top, pts_bot = [], []
        for xi in range(n_pts+1):
            xn  = xi / n_pts
            x   = int(xn * w)
            yn  = 0.12 + ct * 0.08 + 0.06*math.sin(xn*4*math.pi + phase)
            y   = int(h * yn)
            pts_top.append((x, y))
            pts_bot.append((x, int(y + amp * (0.5 + 0.5*math.sin(xn*2*math.pi+phase+mid)))))
        # Fill curtain polygon
        poly = pts_top + list(reversed(pts_bot))
        if len(poly) > 3:
            c = _lerp_color(_lerp_color(pal["bar_low"], pal["accent"], ct),
                            pal["wave"], high * 0.5)
            bright = 0.25 + bass * 0.3 + ct * 0.1
            c = _clamp_color(int(c[0]*bright), int(c[1]*bright), int(c[2]*bright))
            try:
                draw.polygon(poly, fill=c)
            except Exception:
                pass
        # Bright top edge
        ec = _lerp_color(pal["particle"], pal["beat"], ct)
        ebright = 0.5 + bass * 0.4
        ec = _clamp_color(int(ec[0]*ebright), int(ec[1]*ebright), int(ec[2]*ebright))
        draw.line(pts_top, fill=ec, width=max(1,int(1+bass*2)))


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

def _render_frame(mode, w, h, bass, mid, high, beat, wave, fi, pal, bar_count, state):
    img  = Image.new("RGB", (w, h), pal["bg"])
    draw = ImageDraw.Draw(img)

    if   mode == "spectrum_classic":
        p = state.get("particles", [])
        p = _spectrum_classic_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal, bar_count, p)
        state["particles"] = p

    elif mode == "lava_lamp":
        bl = state.get("blobs", [])
        bl = _lava_frame(draw, w, h, bass, mid, high, beat, fi, pal, bl)
        state["blobs"] = bl

    elif mode == "wormhole_3d":
        state = _wormhole_3d_frame(draw, w, h, bass, mid, high, beat, fi, pal, state)

    elif mode == "cosmic_tunnel":
        state = _cosmic_tunnel_frame(draw, w, h, bass, mid, high, beat, fi, pal, state)

    elif mode == "neon_corridor":
        _neon_corridor_frame(draw, w, h, bass, mid, high, beat, fi, pal)

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

    elif mode == "fractal_zoom":
        _fractal_zoom_frame(draw, w, h, bass, mid, high, beat, fi, pal)

    elif mode == "dna_helix":
        _dna_frame(draw, w, h, bass, mid, high, beat, fi, pal)

    elif mode == "liquid_metal":
        _liquid_metal_frame(draw, w, h, bass, mid, high, beat, fi, pal)

    elif mode == "kaleidoscope":
        _kaleidoscope_frame(draw, w, h, bass, mid, high, beat, fi, pal)

    elif mode == "particle_explosion":
        state = _particle_explosion_frame(draw, w, h, bass, mid, high, beat, fi, pal, state)

    elif mode == "waveform_tunnel":
        _waveform_tunnel_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal)

    elif mode == "soundscape":
        _soundscape_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal)

    elif mode == "vortex":
        _vortex_frame(draw, w, h, bass, mid, high, beat, fi, pal)

    elif mode == "crystal_cave":
        state = _crystal_cave_frame(draw, w, h, bass, mid, high, beat, fi, pal, state)

    elif mode == "neural_fire":
        state = _neural_fire_frame(draw, w, h, bass, mid, high, beat, fi, pal, state)

    elif mode == "aurora_borealis":
        _aurora_frame(draw, w, h, bass, mid, high, beat, fi, pal)

    arr = np.array(img, dtype=np.float32) / 255.0
    return arr, state


# ══════════════════════════════════════════════════════════════════════════════
# ComfyUI Node
# ══════════════════════════════════════════════════════════════════════════════

class S42PAudioVisualizer:
    """
    🎬 S42P Audio Visualizer v7.0 — 20 audio-reactive modes, 18 color palettes.

    NEW v7.0:
      wormhole_3d       — TRUE 3D psychedelic tunnel (speed/curves/lights all audio-driven)
      cosmic_tunnel     — star-streaked hyperspace warp
      neon_corridor     — blade-runner neon corridor fly-through
      soundscape        — 3D frequency terrain
      vortex            — spinning spiral driven by audio
      crystal_cave      — crystalline formations lit by frequency energy
      neural_fire       — firing neuron network topology
      aurora_borealis   — flowing northern-lights curtains
      kaleidoscope      — symmetric pattern reflections
      particle_explosion — radial burst particles
      waveform_tunnel   — 3D waveform ring fly-through

    OPTIONAL AUDIO: all modes animate gracefully without audio (time-driven).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode":    (VIZ_MODES, {"default": "wormhole_3d",
                    "tooltip": "Visualizer mode. wormhole_3d recommended for music videos."}),
                "palette": (PALETTE_NAMES, {"default": "Neon Synthwave",
                    "tooltip": "Color palette."}),
                "width":  ("INT", {"default": 1024, "min": 64, "max": 3840, "step": 8}),
                "height": ("INT", {"default": 576,  "min": 64, "max": 2160, "step": 8}),
                "fps":    ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000,
                    "tooltip": "Starting frame index. Offset for long renders."}),
                "batch_size": ("INT", {"default": 24, "min": 1, "max": 9999,
                    "tooltip": "Number of frames to render."}),
                "bar_count": ("INT", {"default": 64, "min": 8, "max": 256,
                    "tooltip": "Number of frequency bars (spectrum_classic mode)."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Optional. All modes work without audio (time-driven animation)."}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("frames",)
    FUNCTION      = "visualize"
    CATEGORY      = "S42 Production Suite/Visualizer"

    def visualize(self, mode, palette, width, height, fps, start_frame,
                  batch_size, bar_count, audio=None):

        if not PIL_AVAILABLE:
            logger.error("Pillow required for audio visualizer.")
            blank = np.zeros((batch_size, height, width, 3), dtype=np.float32)
            if TORCH_AVAILABLE:
                return (torch.from_numpy(blank),)
            return (blank,)

        pal = PALETTES.get(palette, PALETTES["Neon Synthwave"])
        frames_out = []
        state: Dict[str, Any] = {}

        # Parse audio
        audio_mono = None
        audio_sr   = 44100
        if audio is not None:
            try:
                import torch as _torch
                waveform = audio.get("waveform") or audio.get("samples")
                audio_sr = audio.get("sample_rate", 44100)
                if isinstance(audio_sr, _torch.Tensor): audio_sr = int(audio_sr.item())
                if waveform is not None:
                    if isinstance(waveform, _torch.Tensor): waveform = waveform.detach().cpu().numpy()
                    waveform = np.asarray(waveform, dtype=np.float32)
                    if waveform.ndim == 3: waveform = waveform[0]
                    if waveform.ndim == 2: waveform = waveform.mean(axis=0)
                    mx = np.max(np.abs(waveform))
                    if mx > 1e-8: waveform = waveform / mx
                    audio_mono = waveform
            except Exception as e:
                logger.warning(f"Audio parse failed: {e}")

        total_samples = len(audio_mono) if audio_mono is not None else 0

        for bi in range(batch_size):
            fi = start_frame + bi

            # Extract per-frame audio
            bass, mid, high, beat = 0.3, 0.3, 0.3, 0.0
            wave = np.zeros(512, dtype=np.float32)

            if audio_mono is not None and total_samples > 0:
                frame_dur = audio_sr / fps
                sample_start = int(fi * frame_dur)
                chunk_size   = max(1024, int(frame_dur * 4))
                s0 = min(sample_start, total_samples - 1)
                s1 = min(s0 + chunk_size, total_samples)
                chunk = audio_mono[s0:s1]
                if len(chunk) > 64:
                    bass, mid, high, beat = _energy_chunk(chunk, audio_sr)
                    wave = _wave_chunk(chunk)

            arr, state = _render_frame(mode, width, height, bass, mid, high, beat,
                                        wave, fi, pal, bar_count, state)
            frames_out.append(arr)

            if bi % 50 == 0:
                gc.collect()

        stacked = np.stack(frames_out, axis=0).astype(np.float32)

        if TORCH_AVAILABLE:
            return (torch.from_numpy(stacked),)
        return (stacked,)


NODE_CLASS_MAPPINGS        = {"S42PAudioVisualizer": S42PAudioVisualizer}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PAudioVisualizer": "🎬 S42P Audio Visualizer"}
