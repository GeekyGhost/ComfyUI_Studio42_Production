"""
S42 Production Suite — Audio Visualizer v6.1
=============================================
Keeps every visual effect from v6.0 exactly as-is (they had life and motion).

NEW in v6.1:
  - duration_sec input: set seconds → node calculates frames = round(dur × fps).
    Set to 0.0 to auto-match full audio length (up to max_frames cap).
  - Improved per-frame audio extraction: each frame gets its own FFT window
    centred at the correct timestamp for accurate bass/mid/high tracking.
  - Adaptive beat detection: local-window onset detection, not just threshold.
  - Every mode's parameters (speed, size, colour, count) now accurately follow
    the extracted per-frame features.

MODES (unchanged from v6.0):
  spectrum_classic · lava_lamp · wormhole · matrix · stargate · oscilloscope
  plasma_waves · fractal_zoom · dna_helix · liquid_metal
  kaleidoscope · particle_explosion · waveform_tunnel

12 PALETTES: same as v6.0

Python 3.12 | ComfyUI Portable | Pillow + numpy (CPU, no VRAM)
"""

from __future__ import annotations

import gc
import math
import random
import colorsys
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
# Colour palettes (unchanged from v6.0)
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
# Audio extraction — accurate per-frame analysis
# ─────────────────────────────────────────────────────────────────────────────

def _extract_all_features(audio_np: np.ndarray, sr: int, fps: float, n_frames: int):
    """
    Returns (bass_e, mid_e, high_e, beat, wave_chunks) each length n_frames.
    Each frame gets its own FFT window centred at the correct timestamp so
    motion, colour, and size track the music accurately at every frame.
    Beat detection uses a local adaptive threshold for natural onset response.
    """
    total   = len(audio_np)
    fft_win = 2048
    nyq     = sr / 2.0
    window  = np.hanning(fft_win).astype(np.float32)

    def _bin(hz):
        return max(1, int(hz / nyq * (fft_win // 2)))

    bass_e = np.zeros(n_frames, np.float32)
    mid_e  = np.zeros(n_frames, np.float32)
    high_e = np.zeros(n_frames, np.float32)
    wave_c = []

    for i in range(n_frames):
        t_sec  = i / fps
        centre = int(t_sec * sr)
        half   = fft_win // 2
        s      = max(0, centre - half)
        e      = min(total, s + fft_win)

        chunk = np.zeros(fft_win, np.float32)
        av    = e - s
        if av > 0:
            chunk[:av] = audio_np[s:s + av]
        chunk *= window

        spec      = np.abs(np.fft.rfft(chunk))[:fft_win // 2]
        bass_e[i] = float(np.sqrt(np.mean(spec[:_bin(250)]              ** 2) + 1e-12))
        mid_e[i]  = float(np.sqrt(np.mean(spec[_bin(250):_bin(4000)]   ** 2) + 1e-12))
        high_e[i] = float(np.sqrt(np.mean(spec[_bin(4000):]             ** 2) + 1e-12))

        # Short waveform snippet for oscilloscope
        wcs = max(0, centre - sr // 120)
        wce = min(total, wcs + sr // 60)
        wc  = audio_np[wcs:wce].astype(np.float32) if wce > wcs else np.zeros(64, np.float32)
        wave_c.append(wc)

    # Normalise each band to [0,1] by global peak
    for arr in (bass_e, mid_e, high_e):
        peak = float(np.max(arr))
        if peak > 1e-8:
            arr /= peak
        np.clip(arr, 0, 1, out=arr)

    # Beat: bass exceeds local adaptive threshold, and is a local peak
    beat  = np.zeros(n_frames, np.float32)
    win_b = max(1, int(fps * 0.35))
    for i in range(n_frames):
        lo, hi       = max(0, i - win_b), min(n_frames, i + win_b + 1)
        local_mean   = float(np.mean(bass_e[lo:hi]))
        thresh       = local_mean * 1.45 + 0.04
        is_peak      = (i == 0) or (bass_e[i] >= bass_e[i - 1])
        if bass_e[i] > thresh and is_peak:
            beat[i] = min(1.0, (bass_e[i] - thresh) / max(thresh * 0.4, 1e-6))

    return bass_e, mid_e, high_e, beat, wave_c


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lerp(c1, c2, t):
    t = max(0., min(1., float(t)))
    return (int(c1[0]+(c2[0]-c1[0])*t), int(c1[1]+(c2[1]-c1[1])*t),
            int(c1[2]+(c2[2]-c1[2])*t))

def _hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0., min(1., s)), max(0., min(1., v)))
    return (int(r*255), int(g*255), int(b*255))


# ─────────────────────────────────────────────────────────────────────────────
# MODE: SPECTRUM CLASSIC
# ─────────────────────────────────────────────────────────────────────────────

def _spectrum_classic_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal, bar_count, particles):
    # Background plasma pulses with bass
    step = max(8, w // 40)
    t    = fi * 0.05
    for x in range(0, w, step):
        for y in range(0, h, step):
            v = (math.sin(x*.02+t) + math.sin(y*.02+t*.7) +
                 math.sin((x+y)*.015+t*1.3)*(0.3+bass*.7) + 3) / 6
            draw.rectangle([x, y, x+step, y+step], fill=_lerp(pal["p1"], pal["p2"], v))

    if beat > 0.5:
        cx, cy = w//2, h//2
        r_max  = int(math.sqrt(cx*cx+cy*cy) * beat)
        for r in range(0, r_max, max(2, r_max//8)):
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=pal["beat"])

    bar_w = max(4, w // bar_count)
    for i in range(bar_count):
        t_bar  = i / bar_count
        energy = bass * (1-t_bar) + mid * t_bar * 0.6 + high * t_bar * t_bar
        ht     = int(h * 0.7 * energy * (0.5 + random.random() * 0.5))
        ht     = min(ht, h)
        col    = _lerp(pal["bar_low"],
                        _lerp(pal["bar_mid"], pal["bar_top"], min(1, ht/(h*0.7))),
                        min(1, ht / max(1, h*0.7)))
        draw.rectangle([i*bar_w, h-ht, i*bar_w+bar_w-2, h], fill=col)

    cx2, cy2 = w//2, h//2
    r_ring   = int(min(w,h) * 0.38 + bass * min(w,h) * 0.08)
    n_pts    = min(len(wave), 256)
    pts      = []
    for i in range(n_pts):
        ang = (i / n_pts) * 2 * math.pi
        amp = wave[i * len(wave) // max(n_pts,1)] if len(wave) > 0 else 0
        r   = r_ring + amp * r_ring * 0.3
        pts.append((cx2 + int(r*math.cos(ang)), cy2 + int(r*math.sin(ang))))
    if len(pts) > 1:
        draw.line(pts + [pts[0]], fill=pal["wave"], width=2)

    if beat > 0.5:
        for _ in range(int(10 + beat * 20)):
            particles.append({
                "x": cx2, "y": cy2,
                "vx": random.uniform(-8,8)*(0.5+bass),
                "vy": random.uniform(-8,8)*(0.5+bass),
                "life": int(30 + beat*30)
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
# MODE: WORMHOLE
# ─────────────────────────────────────────────────────────────────────────────

def _wormhole_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    cx, cy    = w // 2, h // 2
    num_rings = 30
    # Tunnel speed driven by bass + beat
    t         = fi * (0.015 + bass * 0.04 + beat * 0.03)

    for ring_idx in range(num_rings, 0, -1):
        depth = ring_idx / num_rings
        z     = depth + t % 1.0
        if z > 1.0: z -= 1.0

        scale  = 0.2 + (1.0 - depth) * 2.5
        radius = int(min(w,h) * scale * (0.15 + bass * 0.12))

        twist    = depth * math.pi * 4 + t * 3
        hue      = (depth * 2 + t * 0.5 + bass * 0.3) % 1.0
        sat      = 0.7 + high * 0.3
        val      = 0.3 + depth * 0.7 + beat * 0.5
        ring_col = _hsv(hue, sat, val)

        segments = 12 + int(mid * 12)
        for seg in range(segments):
            ang_start = (seg / segments) * 2 * math.pi + twist
            ang_end   = ((seg + 0.7) / segments) * 2 * math.pi + twist
            pts = []
            for a in np.linspace(ang_start, ang_end, 8):
                pts.append((cx + int(radius*math.cos(a)), cy + int(radius*math.sin(a))))
            if len(pts) > 1:
                draw.line(pts, fill=ring_col, width=max(1, int(3*(1-depth)+beat*3)))

        if beat > 0.4 and ring_idx % 3 == 0:
            for p_ang in np.linspace(0, 2*math.pi, 6):
                px = cx + int(radius * 0.7 * math.cos(p_ang + twist))
                py = cy + int(radius * 0.7 * math.sin(p_ang + twist))
                ps = max(1, int(5 * (1-depth) * (1+beat)))
                draw.ellipse([px-ps, py-ps, px+ps, py+ps], fill=pal["particle"])


# ─────────────────────────────────────────────────────────────────────────────
# MODE: PLASMA WAVES
# ─────────────────────────────────────────────────────────────────────────────

def _plasma_waves_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    t = fi * (0.02 + mid * 0.03)

    for y in range(0, h, 4):
        for x in range(0, w, 4):
            v1 = math.sin(x * 0.01 + t * 2)
            v2 = math.sin(y * 0.01 + t * 1.5)
            v3 = math.sin((x+y) * 0.007 + t * 3)
            v4 = math.sin(math.sqrt(x*x+y*y) * 0.015 + t)
            plasma = (v1+v2+v3+v4+4) / 8 * (0.6 + bass * 0.4)
            hue = (plasma + t * 0.5 + mid * 0.2) % 1.0
            col = _hsv(hue, 0.6 + high * 0.4, 0.4 + plasma * 0.6 + beat * 0.4)
            draw.rectangle([x, y, x+4, y+4], fill=col)

    for ribbon in range(5 + int(bass * 4)):
        points = []
        phase  = (ribbon / 9) * 2 * math.pi + t
        for x in range(0, w, 8):
            wave_y = h//2 + int(
                math.sin(x*0.02+phase)    * h * (0.15+bass*0.10) +
                math.sin(x*0.01+phase*1.5)* h * (0.05+mid*0.08)
            )
            points.append((x, wave_y))
        if len(points) > 1:
            hue = (ribbon / 9 + t * 0.3) % 1.0
            draw.line(points, fill=_hsv(hue, 0.9, 0.8+beat*0.2),
                      width=max(2, int(2+bass*5)))


# ─────────────────────────────────────────────────────────────────────────────
# MODE: FRACTAL ZOOM (pixel-direct — same as v6)
# ─────────────────────────────────────────────────────────────────────────────

def _fractal_zoom_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    img    = draw._image
    pixels = img.load()
    zoom   = 0.5 ** (fi * 0.01 * (1 + bass * 0.5))
    rot    = fi * 0.02 * (1 + mid * 0.3)
    cxo    = -0.5 + math.sin(fi * 0.003) * 0.3
    cyo    = math.cos(fi * 0.004) * 0.3
    for py in range(h):
        for px in range(w):
            x  = (px - w/2) / (w * 0.4)
            y  = (py - h/2) / (h * 0.4)
            xr = x*math.cos(rot) - y*math.sin(rot)
            yr = x*math.sin(rot) + y*math.cos(rot)
            c  = complex(xr*zoom+cxo, yr*zoom+cyo)
            z  = 0j
            mi = 50
            i  = 0
            for i in range(mi):
                if abs(z) > 2.0: break
                z = z*z + c
            if i < mi-1:
                try:
                    sm = i + 1 - math.log(math.log(abs(z)+1e-9)) / math.log(2)
                except Exception:
                    sm = i
                col = _hsv((sm*0.05+bass*0.2+fi*0.002)%1.0, 0.7+high*0.3,
                            min(1.0, sm/20.0+beat*0.3))
            else:
                col = pal["bg"]
            if 0 <= px < w and 0 <= py < h:
                pixels[px, py] = col


# ─────────────────────────────────────────────────────────────────────────────
# MODE: LIQUID METAL (pixel-direct — same as v6)
# ─────────────────────────────────────────────────────────────────────────────

def _liquid_metal_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    img    = draw._image
    pixels = img.load()
    t      = fi * 0.03
    for py in range(h):
        for px in range(w):
            x = px/w; y = py/h
            s1 = math.sin(x*10+t*2+bass*3)*0.1
            s2 = math.sin(y*8-t*1.5+mid*2)*0.08
            s3 = math.sin((x+y)*12+t*3)*0.06
            if beat > 0.5:
                d  = math.sqrt((x-0.5)**2+(y-0.5)**2)
                s1 += math.sin(d*30-t*5)*0.15*beat
            surface = s1+s2+s3; normal = surface+0.5
            hl = max(0, normal-0.6)*3; sh = max(0, 0.4-normal)*2
            bb = 0.3+normal*0.4+high*0.3
            hue = (t*0.1+surface*0.5)%1.0
            r = int(min(255, (bb+hl-sh+bass*0.2)*255))
            g = int(min(255, (bb+hl*0.9-sh+mid*0.15)*255))
            b = int(min(255, (bb+hl*0.8-sh*0.5)*255))
            tn = _hsv(hue, 0.3, 1.0)
            pixels[px, py] = (max(0,min(255,int(r*.7+tn[0]*.3))),
                               max(0,min(255,int(g*.7+tn[1]*.3))),
                               max(0,min(255,int(b*.7+tn[2]*.3))))


# ─────────────────────────────────────────────────────────────────────────────
# MODE: LAVA LAMP
# ─────────────────────────────────────────────────────────────────────────────

def _lava_frame(draw, w, h, bass, mid, high, beat, fi, pal, blobs):
    if not blobs:
        for _ in range(8):
            blobs.append({"x": random.randint(0,w), "y": random.randint(0,h),
                           "vx": random.uniform(-1,1), "vy": random.uniform(-2,0),
                           "r": random.randint(30,80), "hue": random.random()})
    for b in blobs:
        b["vy"] += 0.1 - bass * 0.2
        b["vx"] *= 0.98; b["vy"] *= 0.98
        b["x"]  += b["vx"]; b["y"] += b["vy"]
        if b["y"] < b["r"]:           b["y"]=b["r"];       b["vy"]=abs(b["vy"])*0.7
        if b["y"] > h-b["r"]:         b["y"]=h-b["r"];     b["vy"]=-abs(b["vy"])*0.7
        if b["x"] < b["r"] or b["x"] > w-b["r"]:
            b["vx"] *= -1; b["x"] = max(b["r"], min(w-b["r"], b["x"]))
        b["r"]   = int(30 + 55*(0.4+bass*0.6))
        b["hue"] = (b["hue"] + 0.001 + mid*0.002) % 1.0
    for b in blobs:
        col = _hsv(b["hue"], 0.9, 0.9+beat*0.1)
        for r_off in range(b["r"], 0, -max(1, b["r"]//10)):
            alpha = r_off / b["r"]
            draw.ellipse([b["x"]-r_off, b["y"]-r_off, b["x"]+r_off, b["y"]+r_off],
                         fill=tuple(int(c*alpha) for c in col))
    return blobs


# ─────────────────────────────────────────────────────────────────────────────
# MODE: MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def _matrix_frame(draw, w, h, bass, mid, high, beat, fi, pal, cols):
    if not cols:
        for x in range(0, w, 12):
            cols.append({"x": x, "y": random.randint(-h,0), "speed": 2+random.random()*4,
                         "chars": [random.choice("01") for _ in range(40)]})
    for col in cols:
        col["speed"] = 2 + mid*8 + (5 if beat > 0.5 else 0)
        col["y"]    += col["speed"]
        if col["y"] > h:
            col["y"]=""; col["y"] = -50
            col["chars"] = [random.choice("01") for _ in range(40)]
        for i, char in enumerate(col["chars"]):
            y_pos = int(col["y"] - i*15)
            if 0 <= y_pos < h:
                alpha   = (1.0-(i/len(col["chars"]))) * (0.5+high*0.5)
                draw.text((col["x"], y_pos), char,
                          fill=tuple(int(c*alpha) for c in pal["bar_mid"]))
    return cols


# ─────────────────────────────────────────────────────────────────────────────
# MODE: STARGATE
# ─────────────────────────────────────────────────────────────────────────────

def _stargate_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    cx, cy = w//2, h//2
    t      = fi * (0.05 + bass*0.08)
    for ring in range(10):
        r     = int((ring*40 + t*50 + bass*80) % (min(w,h)*0.7))
        alpha = 1.0 - (r / (min(w,h)*0.7))
        if alpha > 0:
            hue = (ring*0.1 + t*0.2) % 1.0
            col = _hsv(hue, 0.7+high*0.3, 0.6+alpha*0.4+beat*0.3)
            draw.ellipse([cx-r,cy-r,cx+r,cy+r], outline=col,
                         width=max(1, int(5*alpha+beat*4)))
    if beat > 0.3:
        tlen = int(min(w,h)//2 * (0.4+bass*0.6))
        for angle in np.linspace(0, 2*math.pi, 8+int(mid*4)):
            pts = []
            for dist in range(0, tlen, 15):
                wobble = math.sin(dist*0.1+t)*30*bass
                pts.append((cx+int((dist+wobble)*math.cos(angle)),
                             cy+int((dist+wobble)*math.sin(angle))))
            if pts:
                draw.line(pts, fill=pal["accent"], width=max(1,int(1+beat*2)))


# ─────────────────────────────────────────────────────────────────────────────
# MODE: OSCILLOSCOPE
# ─────────────────────────────────────────────────────────────────────────────

def _osc_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal):
    for i in range(0, h, h//8):
        draw.line([(0,i),(w,i)], fill=(20,20,20))
    for i in range(0, w, w//10):
        draw.line([(i,0),(i,h)], fill=(20,20,20))
    pts = []
    for i, sample in enumerate(wave):
        x = int(i * w / len(wave))
        y = int(h/2 - sample * h * (0.35 + bass*0.15))
        pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill=_lerp(pal["wave"], pal["beat"], beat),
                  width=max(1, int(1+mid*2)))
    if beat > 0.5:
        draw.line([(0,10),(w,10)], fill=pal["beat"], width=3)
        draw.line([(0,h-10),(w,h-10)], fill=pal["beat"], width=3)


# ─────────────────────────────────────────────────────────────────────────────
# MODE: DNA HELIX
# ─────────────────────────────────────────────────────────────────────────────

def _dna_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    cx = w//2
    t  = fi * (0.03 + mid*0.03)
    p1, p2 = [], []
    for i in range(60):
        z      = i/60.0; y = int(h*z)
        angle  = z*4*math.pi + t
        radius = int(w * (0.18+bass*0.08))
        x1     = cx + int(radius*math.cos(angle))
        x2     = cx + int(radius*math.cos(angle+math.pi))
        p1.append((x1,y)); p2.append((x2,y))
        if i % 3 == 0:
            hue = (z + t*0.5 + bass*0.2) % 1.0
            draw.line([(x1,y),(x2,y)], fill=_hsv(hue,0.8,0.7+beat*0.3),
                      width=max(1,int(1+beat*2)))
    if p1:
        draw.line(p1, fill=pal["bar_mid"], width=3)
        draw.line(p2, fill=pal["bar_top"], width=3)


# ─────────────────────────────────────────────────────────────────────────────
# MODE: KALEIDOSCOPE (pixel-direct)
# ─────────────────────────────────────────────────────────────────────────────

def _kaleidoscope_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    img    = draw._image
    pixels = img.load()
    t      = fi * (0.02 + mid*0.04)
    segs   = 8
    for py in range(h):
        for px in range(w):
            x = px-w/2; y = py-h/2
            r     = math.sqrt(x*x+y*y) / (min(w,h)*0.5)
            theta = math.atan2(y,x)
            theta = (theta % (2*math.pi/segs)) * segs
            pat   = math.sin(r*(8+high*6)+t) * math.cos(theta*3+t*2) * (0.6+bass*0.4)
            hue   = (pat+r+t*0.5+mid*0.2)%1.0
            pixels[px,py] = _hsv(hue, 0.7+high*0.3, 0.35+abs(pat)*0.65+beat*0.25)


# ─────────────────────────────────────────────────────────────────────────────
# MODE: PARTICLE EXPLOSION
# ─────────────────────────────────────────────────────────────────────────────

def _particle_explosion_frame(draw, w, h, bass, mid, high, beat, fi, pal, state):
    particles = state.get("particles", [])
    cx, cy    = w//2, h//2
    if beat > 0.3:
        for _ in range(int(20 + beat*40 + bass*20)):
            ang  = random.random() * 2 * math.pi
            spd  = (2+random.random()*6) * (0.5+bass*1.5)
            life = int(40+beat*40)
            particles.append({"x":cx,"y":cy,"vx":math.cos(ang)*spd,"vy":math.sin(ang)*spd,
                               "life":life,"max_life":life,"hue":random.random()})
    for p in particles:
        p["x"] += p["vx"]*(1+mid*0.5); p["y"] += p["vy"]*(1+mid*0.5)
        p["vy"] += 0.2; p["life"] -= 1
        if p["life"] > 0:
            a    = p["life"]/max(1,p["max_life"])
            size = max(1, int(5*a*(1+bass)))
            col  = _hsv(p["hue"], 0.9, a)
            draw.ellipse([p["x"]-size,p["y"]-size,p["x"]+size,p["y"]+size], fill=col)
    particles[:] = [p for p in particles if p["life"] > 0]
    state["particles"] = particles


# ─────────────────────────────────────────────────────────────────────────────
# MODE: WAVEFORM TUNNEL
# ─────────────────────────────────────────────────────────────────────────────

def _waveform_tunnel_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal):
    cx, cy = w//2, h//2
    t      = fi * (0.05 + bass*0.06)
    for ring in range(20, 0, -1):
        depth    = ring/20.0
        z_offset = (t+depth) % 1.0
        scale    = 0.3 + (1.0-depth)*1.5
        wl       = max(1, len(wave))
        wi       = int((z_offset*wl)) % wl
        n_s      = min(64, wl)
        pts      = []
        for i in range(n_s):
            angle = (i/n_s)*2*math.pi
            si    = (wi+i) % wl
            amp   = wave[si] * h * (0.15+bass*0.10) * scale
            r     = min(w,h)*0.15*scale + amp
            pts.append((cx+int(r*math.cos(angle)), cy+int(r*math.sin(angle))))
        if len(pts) > 2:
            hue = (depth+t*0.3+bass*0.2)%1.0
            col = _hsv(hue, 0.8, 0.5+depth*0.5+beat*0.3)
            draw.line(pts+[pts[0]], fill=col, width=max(1,int(2*(1-depth)+beat)))


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

def _render_frame(mode, w, h, bass, mid, high, beat, wave, fi, pal, bar_count, state):
    img  = Image.new("RGB", (w, h), pal["bg"])
    draw = ImageDraw.Draw(img)

    if   mode == "spectrum_classic":
        p = _spectrum_classic_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal,
                                    bar_count, state.get("particles", []))
        state["particles"] = p
    elif mode == "lava_lamp":
        state["blobs"] = _lava_frame(draw, w, h, bass, mid, high, beat, fi, pal,
                                     state.get("blobs", []))
    elif mode == "wormhole":
        _wormhole_frame(draw, w, h, bass, mid, high, beat, fi, pal)
    elif mode == "matrix":
        state["cols"] = _matrix_frame(draw, w, h, bass, mid, high, beat, fi, pal,
                                      state.get("cols", []))
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
        _particle_explosion_frame(draw, w, h, bass, mid, high, beat, fi, pal, state)
    elif mode == "waveform_tunnel":
        _waveform_tunnel_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal)

    return np.array(img, dtype=np.float32) / 255.0, state


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI Node
# ─────────────────────────────────────────────────────────────────────────────

class S42PAudioVisualizer:
    """
    S42P Audio Visualizer v6.1 — 13 modes, 12 palettes, accurate audio reactivity.

    DURATION INPUT (new in v6.1):
      Set duration_sec > 0.0 → frame count = round(duration_sec × fps).
      Set to 0.0 → auto-match full audio length (capped at max_frames).

    AUDIO REACTIVITY (improved in v6.1):
      Each frame gets its own FFT window centred at the correct timestamp.
      Bass/mid/high track the music accurately at every single frame.
      Beat detection uses a local adaptive threshold (natural, not clicky).

    PER-MODE AUDIO MAPPING:
      bass  → tunnel speed, blob size, bar height, ring expansion, particle burst size
      mid   → rotation speed, ribbon count, glyph fall speed, saturation shift
      high  → brightness, pattern density, glow, colour vibrancy
      beat  → particle spawns, flash pulses, line width spikes, brightness flare
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":        ("AUDIO",),
                "mode":         (VIZ_MODES, {"default": "spectrum_classic"}),
                "fps":          ("FLOAT",   {"default": 24.0, "min": 8.0,  "max": 60.0,   "step": 1.0}),
                "width":        ("INT",     {"default": 512,  "min": 256,  "max": 1920,   "step": 64}),
                "height":       ("INT",     {"default": 288,  "min": 144,  "max": 1080,   "step": 32}),
                "palette":      (PALETTE_NAMES, {"default": "Spectrum Classic"}),
                "bar_count":    ("INT",     {"default": 64,   "min": 16,   "max": 128,    "step": 8}),
                "duration_sec": ("FLOAT",   {
                    "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.5,
                    "tooltip": (
                        "Output duration in seconds. Set > 0 to use instead of max_frames.\n"
                        "frames = round(duration_sec × fps).  Example: 30 s × 24 fps = 720 frames.\n"
                        "Set to 0.0 to auto-match the full audio length (up to max_frames cap)."
                    )
                }),
                "max_frames":   ("INT",     {
                    "default": 1440, "min": 60, "max": 9000, "step": 60,
                    "tooltip": "Cap when duration_sec=0.  24fps: 720=30s  1440=60s  2880=120s"
                }),
                "low_memory":   ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Render at half resolution then upscale. Saves RAM on long renders."
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("visualization",)
    FUNCTION      = "visualize"
    CATEGORY      = "S42 Production Suite/Audio Visualizer"

    def visualize(self, audio: dict, mode: str, fps: float, width: int, height: int,
                  palette: str, bar_count: int, duration_sec: float = 0.0,
                  max_frames: int = 1440, low_memory: bool = False, seed: int = 42):

        if not PIL_AVAILABLE:
            dummy = np.zeros((1, height, width, 3), dtype=np.float32)
            return (torch.from_numpy(dummy) if TORCH_AVAILABLE else dummy,)

        # ── Parse audio ───────────────────────────────────────────────────────
        waveform = audio.get("waveform")
        sr       = int(audio.get("sample_rate", 44100))

        if TORCH_AVAILABLE and hasattr(waveform, "cpu"):
            wav_np = waveform.cpu().float().numpy()
        else:
            wav_np = np.asarray(waveform, dtype=np.float32)

        if wav_np.ndim == 3: wav_np = wav_np.mean(axis=(0, 1))
        elif wav_np.ndim == 2: wav_np = wav_np.mean(axis=0)
        wav_np = wav_np.astype(np.float32)
        peak   = float(np.max(np.abs(wav_np)))
        if peak > 1e-8: wav_np /= peak

        # ── Determine frame count ─────────────────────────────────────────────
        audio_dur_s = len(wav_np) / max(sr, 1)
        if duration_sec > 0.0:
            n_frames = max(1, round(duration_sec * fps))
        else:
            n_frames = min(max_frames, max(1, int(audio_dur_s * fps)))

        rw = width  // 2 if low_memory else width
        rh = height // 2 if low_memory else height

        print(f"[S42P Visualizer v6.1] mode={mode}  audio={audio_dur_s:.1f}s  "
              f"frames={n_frames}  {rw}×{rh}")

        # ── Extract per-frame audio features ──────────────────────────────────
        bass_e, mid_e, high_e, beat, wave_chunks = _extract_all_features(
            wav_np, sr, fps, n_frames)

        print(f"[S42P Visualizer] Rendering {n_frames} frames ...")

        pal    = PALETTES.get(palette, PALETTES["Spectrum Classic"])
        state  = {}
        frames = []
        random.seed(seed)

        for i in range(n_frames):
            wc = wave_chunks[i] if i < len(wave_chunks) else np.zeros(64, np.float32)
            arr, state = _render_frame(
                mode, rw, rh,
                float(bass_e[i]), float(mid_e[i]), float(high_e[i]),
                float(beat[i]), wc, i, pal, bar_count, state
            )

            if low_memory and (rw != width or rh != height):
                up  = Image.fromarray((arr*255).clip(0,255).astype(np.uint8))
                up  = up.resize((width, height), Image.BILINEAR)
                arr = np.array(up, dtype=np.float32) / 255.0

            frames.append(arr)

            if i > 0 and i % 200 == 0:
                gc.collect()
                print(f"[S42P Visualizer]   {i}/{n_frames}")

        batch = np.stack(frames, axis=0).astype(np.float32)
        frames.clear(); gc.collect()
        print(f"[S42P Visualizer v6.1] ✓ {n_frames} frames  {batch.nbytes/1024/1024:.1f} MB")

        return (torch.from_numpy(batch) if TORCH_AVAILABLE else batch,)


NODE_CLASS_MAPPINGS        = {"S42PAudioVisualizer": S42PAudioVisualizer}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PAudioVisualizer": "S42P Audio Visualizer 🎬"}