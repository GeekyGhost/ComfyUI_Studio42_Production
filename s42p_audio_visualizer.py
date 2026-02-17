"""
S42 Production Suite â€” Audio Visualizer v5.0
=============================================
Multi-mode audio-reactive visualizer for music videos.

MODES:
  spectrum_classic  â€” bars + oscilloscope ring + particle burst (Winamp-inspired)
  lava_lamp         â€” physics blobs that rise/fall, morph with music
  wormhole          â€” true 3D tunnel perspective with depth-mapped rings
  matrix            â€” falling character rain, columns pulse to bass
  stargate          â€” expanding portal rings with energy tendrils
  oscilloscope      â€” clean waveform scope with peak indicators
  nebula            â€” fractal plasma cloud with beat explosions
  aurora            â€” flowing northern-lights curtains over starfield
  fractal_zoom      â€” mandelbrot-esque zoom driven by bass
  dna_helix         â€” rotating 3D double helix synced to music
  liquid_metal      â€” mercury/chrome fluid surface with beat ripples

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

# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# Colour palettes
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

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
    "stargate",         "oscilloscope","nebula",     "aurora",
    "fractal_zoom",     "dna_helix",  "liquid_metal",
]

# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# Shared: energy extraction + wave chunk
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

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


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# Colour helpers
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _lerp(c1, c2, t):
    t = max(0., min(1., float(t)))
    return (int(c1[0]+(c2[0]-c1[0])*t), int(c1[1]+(c2[1]-c1[1])*t),
            int(c1[2]+(c2[2]-c1[2])*t))

def _hsv(h, s, v):
    import colorsys
    r,g,b = colorsys.hsv_to_rgb(h%1.0, s, v)
    return (int(r*255), int(g*255), int(b*255))


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# MODE: SPECTRUM CLASSIC (Winamp-inspired, renamed)
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _spectrum_classic_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal, bar_count, particles):
    step = max(8, w // 40); t = fi * 0.05
    for x in range(0, w, step):
        for y in range(0, h, step):
            v = (math.sin(x*.02+t)+math.sin(y*.02+t*.7)+math.sin((x+y)*.015+t*1.3)*(0.3+bass*.7)+3)/6
            draw.rectangle([x,y,x+step,y+step], fill=_lerp(pal["p1"],pal["p2"],v))
    if beat > 0.5:
        cx,cy = w//2,h//2
        r_max = int(math.sqrt(cx*cx+cy*cy)*beat)
        for r in range(0, r_max, max(2, r_max//8)):
            draw.ellipse([cx-r,cy-r,cx+r,cy+r], outline=pal["beat"])
    bz_h = int(h*.45); bz_y = h-bz_h; bw = max(2,w//bar_count); sp=max(1,bw//4)
    rng  = random.Random(int(bass*1000+mid*500+high*200))
    for i in range(bar_count):
        n = i/bar_count
        e = (bass*(0.8+rng.uniform(-.2,.2)) if n<.15
             else mid*(0.7+rng.uniform(-.3,.3)) if n<.7
             else high*(0.6+rng.uniform(-.4,.4)))
        bh = max(2, int(bz_h*min(1,e))); x0=i*bw+sp; x1=x0+bw-sp*2
        y0=bz_y+bz_h-bh; y1=bz_y+bz_h
        for sy in range(bh):
            fy  = 1.-sy/bh
            col = _lerp(pal["bar_low"],pal["bar_mid"],fy*3) if fy<.33 else _lerp(pal["bar_mid"],pal["bar_top"],(fy-.33)*1.5)
            draw.rectangle([x0,y1-sy-1,x1,y1-sy], fill=col)
        if y0-2>=bz_y: draw.rectangle([x0,y0-2,x1,y0],fill=pal["bar_top"])
    n_pts=min(len(wave),256)
    if n_pts>=4:
        cx,cy=w//2,h//2; wr=wave[:n_pts].copy(); wr/=(np.max(np.abs(wr))+1e-8)
        r_min=int(min(w,h)*.12); r_max=int(min(w,h)*.28)+int(bass*30); pts=[]
        for j in range(n_pts):
            a=2*math.pi*j/n_pts; r=r_min+(r_max-r_min)*(0.5+0.5*float(wr[j]))
            pts.append((int(cx+r*math.cos(a)),int(cy+r*math.sin(a))))
        pts.append(pts[0])
        for j in range(len(pts)-1): draw.line([pts[j],pts[j+1]],fill=pal["wave"],width=2)
    new_p=[]; pts_d=[]; cx2,cy2=w//2,h//2
    n_spawn=int(high*6)+(8 if beat>.5 else 0); rng2=random.Random(fi*137+int(high*1000))
    for _ in range(min(n_spawn,120-len(particles))):
        a=rng2.uniform(0,2*math.pi); spd=rng2.uniform(1.5,5.)*(0.5+high)
        particles.append({"x":float(cx2),"y":float(cy2),"vx":spd*math.cos(a),"vy":spd*math.sin(a),"life":rng2.uniform(.5,1.),"size":rng2.randint(1,3)})
    for p in particles:
        p["x"]+=p["vx"]; p["y"]+=p["vy"]; p["life"]-=0.03
        if p["life"]<=0 or not(0<=p["x"]<=w) or not(0<=p["y"]<=h): continue
        new_p.append(p); pts_d.append((p["x"],p["y"],p["size"]))
    for px,py,ps in pts_d:
        draw.ellipse([int(px)-ps,int(py)-ps,int(px)+ps,int(py)+ps],fill=pal["particle"])
    return new_p


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# MODE: LAVA LAMP €” fixed: blobs travel full height, morph with music
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _lava_frame(draw, w, h, bass, mid, high, beat, fi, pal, blobs):
    # Background gradient
    for y in range(0, h, 4):
        draw.rectangle([0,y,w,y+4], fill=_lerp(pal["p2"],pal["p1"],y/h))

    rng = random.Random(fi * 31 + int(bass * 777))

    # Spawn blobs €” start below screen
    if beat > 0.5 or len(blobs) < 5:
        for _ in range(2 if beat > 0.5 else 1):
            if len(blobs) < 25:
                r = rng.randint(int(min(w,h)*0.05), int(min(w,h)*0.14))
                blobs.append({
                    "x":   rng.uniform(r*1.5, w - r*1.5),
                    "y":   float(h + r + rng.randint(0, h//3)),   # start BELOW screen
                    "vx":  rng.uniform(-0.5, 0.5),
                    "vy":  -rng.uniform(0.8, 2.2),                # upward velocity
                    "r":   float(r),
                    "r_base": float(r),
                    "hue": rng.uniform(0.0, 0.12),
                    "phase": rng.uniform(0, math.pi * 2),
                    "life": 1.0,
                })

    new_blobs = []
    for b in blobs:
        # Update physics
        b["vy"] += 0.015         # gravity â€” slow drift back down
        b["vy"] -= bass * 0.5    # bass pushes up hard
        b["vy"]  = max(-3.5, min(1.2, b["vy"]))  # clamp velocity
        b["vx"] += math.sin(fi * 0.06 + b["phase"]) * 0.08 * mid
        b["vx"] *= 0.98          # damping

        b["x"] += b["vx"]
        b["y"] += b["vy"]

        # Bounce off walls softly
        if b["x"] < b["r"]:      b["x"] = b["r"];    b["vx"] =  abs(b["vx"])
        if b["x"] > w - b["r"]:  b["x"] = w-b["r"];  b["vx"] = -abs(b["vx"])

        # Morphing radius €” pulses with bass and high
        wobble   = math.sin(fi * 0.18 + b["phase"]) * 0.18 * mid
        r_target = b["r_base"] * (1.0 + bass * 0.45 + high * 0.2 + wobble)
        b["r"]   = b["r"] * 0.85 + r_target * 0.15   # smooth lerp

        # Kill if off top (and not coming back) or life exhausted
        b["life"] -= 0.003
        if b["y"] < -b["r"] * 3 or b["life"] <= 0:
            continue
        # Also kill if too far below screen for too long
        if b["y"] > h + b["r"] * 4:
            continue

        # Draw
        bx, by, br = int(b["x"]), int(b["y"]), max(4, int(b["r"]))
        hue  = (b["hue"] + fi * 0.003) % 1.0
        col  = _hsv(hue, 0.9,  0.55 + b["life"] * 0.45)
        col2 = _hsv(hue, 1.0,  0.92)
        col3 = _hsv(hue, 0.5,  0.98)   # highlight

        glow_r = int(br * 1.5)
        for gr in range(glow_r, br, max(1, (glow_r-br)//5)):
            alpha_v = (gr - br) / max(1, glow_r - br)
            gc_col  = _lerp(col, pal["bg"], alpha_v)
            draw.ellipse([bx-gr,by-gr,bx+gr,by+gr], fill=gc_col)
        draw.ellipse([bx-br,by-br,bx+br,by+br], fill=col2)
        # Highlight
        hl = max(2, br // 3)
        draw.ellipse([bx-br+hl//2,by-br+hl//2, bx-br//2+hl,by-br//2+hl], fill=col3)

        new_blobs.append(b)

    return new_blobs


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# MODE: WORMHOLE €” true 3D perspective tunnel
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _wormhole_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    cx, cy   = w / 2, h / 2
    n_rings  = 32
    fov      = 0.9               # field of view factor
    speed    = 0.06 + bass * 0.12  # tunnel fly-through speed
    spin     = fi * 0.04 + mid * 0.08

    # Deep space background
    for y in range(0, h, 3):
        draw.rectangle([0,y,w,y+3], fill=_lerp(pal["bg"],pal["p1"],y/h*0.15))

    # Draw stars in tunnel background
    rng = random.Random(fi // 6)
    for _ in range(40):
        sx = rng.randint(0, w); sy = rng.randint(0, h)
        lum = rng.randint(100, 220)
        draw.point((sx, sy), fill=(lum,lum,lum))

    # Generate tunnel rings sorted back to front (far first)
    for ri in range(n_rings - 1, -1, -1):
        # z depth: 0 (far) to 1 (near) with speed offset
        z_norm = ((ri / n_rings) + (fi * speed * 0.01)) % 1.0
        # Perspective: objects closer (z_norm†’1) appear larger
        persp  = z_norm ** 1.8
        persp  = max(0.01, persp)

        # Ring radius in 3D space €” modulate with bass
        r3d    = 0.55 + bass * 0.2 * z_norm
        rx     = int(cx * r3d * persp * fov * (w / h))
        ry     = int(cy * r3d * persp * fov)
        rx     = max(1, rx); ry = max(1, ry)

        # Wormhole twist: rotation increases with depth
        twist  = spin * (1.0 - z_norm * 0.5)

        # Colour: hue shifts along tunnel + time
        hue    = (z_norm * 0.5 + fi * 0.004 + beat * 0.05) % 1.0
        val    = 0.15 + persp * 0.85
        sat    = 0.6 + high * 0.4
        col    = _hsv(hue, sat, val)

        # Draw ring as polygon (perspective ellipse with rotation)
        n_pts  = max(8, int(16 * persp + 8))
        pts    = []
        for j in range(n_pts):
            a    = 2 * math.pi * j / n_pts + twist
            # Add warp distortion driven by mid freq
            warp = math.sin(a * 3 + fi * 0.1) * mid * 0.15 * persp
            px   = int(cx + (rx + rx * warp) * math.cos(a))
            py   = int(cy + (ry + ry * warp * 0.5) * math.sin(a))
            pts.append((px, py))

        if len(pts) >= 3:
            lw = max(1, int(persp * 3 + beat * 3))
            draw.polygon(pts, outline=col)

        # Draw connecting lines to next ring (tunnel walls)
        if ri > 0 and ri % 4 == 0:
            # Sample 8 connection points
            for j in range(0, n_pts, max(1, n_pts // 8)):
                a_next = 2 * math.pi * j / n_pts + spin * (1.0 - (z_norm - 1/n_rings)*0.5)
                z2     = ((  (ri-4) / n_rings) + (fi * speed * 0.01)) % 1.0
                p2     = max(0.01, z2 ** 1.8)
                rx2    = max(1, int(cx * 0.55 * p2 * fov * (w/h)))
                ry2    = max(1, int(cy * 0.55 * p2 * fov))
                p2x    = int(cx + rx2 * math.cos(a_next))
                p2y    = int(cy + ry2 * math.sin(a_next))
                if len(pts) > j:
                    wall_col = _lerp(col, pal["bg"], 0.6)
                    draw.line([pts[j], (p2x, p2y)], fill=wall_col, width=1)

    # Beat flash: bright inner core
    if beat > 0.5:
        r_core = int(20 + bass * 40)
        draw.ellipse([cx-r_core,cy-r_core,cx+r_core,cy+r_core], fill=pal["beat"])
        r2 = r_core // 3
        draw.ellipse([cx-r2,cy-r2,cx+r2,cy+r2], fill=(255,255,255))

    # Bright tunnel mouth glow at centre
    r_glow = int(8 + bass * 15)
    draw.ellipse([cx-r_glow,cy-r_glow,cx+r_glow,cy+r_glow], fill=pal["accent"])
    draw.ellipse([cx-3,cy-3,cx+3,cy+3], fill=(255,255,255))


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# MODE: MATRIX
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

_MATRIX_CHARS = list("ï½¦ï½±ï½²ï½³ï½´ï½µï½¶ï½·ï½¸ï½¹ï½ºï½»ï½¼ï½½ï½¾ï½¿ï¾€ï¾ï¾‚ï¾ƒï¾„ï¾…ï¾†ï¾‡ï¾ˆï¾‰ï¾Šï¾‹ï¾Œï¾ï¾Žï¾ï¾ï¾‘ï¾’ï¾“ï¾”ï¾•ï¾–ï¾—ï¾˜ï¾™ï¾šï¾›ï¾œï¾0123456789ABCDEF")

def _matrix_frame(draw, w, h, bass, mid, high, beat, fi, pal, cols_state):
    char_w=14; char_h=18; n_cols=w//char_w
    rng = random.Random(fi*3+42)
    if not cols_state or len(cols_state)!=n_cols:
        cols_state=[{"y":rng.randint(-h,0),"speed":rng.uniform(.5,2.),"len":rng.randint(8,24)} for _ in range(n_cols)]
    for ci,col in enumerate(cols_state):
        col["y"]+=col["speed"]*(1.+bass*3.)
        if col["y"]>h+col["len"]*char_h:
            col["y"]=-col["len"]*char_h; col["speed"]=rng.uniform(.5,2.5)*(1.+mid); col["len"]=rng.randint(6,30)
        for row in range(col["len"]):
            cy_p=int(col["y"])-row*char_h
            if cy_p<0 or cy_p>h: continue
            frac=1.-row/col["len"]
            if row==0: cc=(200,255,200)
            elif row<3: cc=_lerp(pal["bar_top"],pal["bar_mid"],row/3)
            else: cc=_lerp(pal["bar_mid"],pal["p1"],max(0.,(row-3)/(col["len"]-3)))
            if high>.6 and row==0: cc=(255,255,255)
            draw.rectangle([ci*char_w,cy_p,ci*char_w+char_w-2,cy_p+char_h-2],
                            fill=(int(cc[0]*frac),int(cc[1]*frac),int(cc[2]*frac)))
        if beat>.5 and ci%max(1,int(n_cols*(1-bass)))==0:
            for row in range(col["len"]):
                cy_p=int(col["y"])-row*char_h
                if 0<=cy_p<=h: draw.rectangle([ci*char_w,cy_p,ci*char_w+char_w-2,cy_p+char_h-2],fill=pal["beat"])
    return cols_state


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# MODE: STARGATE
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _stargate_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    cx,cy=w//2,h//2
    rng=random.Random(fi//4)
    for _ in range(60):
        sx=rng.randint(0,w);sy=rng.randint(0,h);sr=1 if rng.random()>.9 else 0
        lum=rng.randint(100,255); draw.ellipse([sx-sr,sy-sr,sx+sr,sy+sr],fill=(lum,lum,lum))
    for ri in range(16):
        t=(ri/16); phase=((fi*.04)+t)%1.0; r=int((.1+phase*.9)*min(cx,cy))
        hue=(t*.4+fi*.004+bass*.1)%1.; col=_hsv(hue,1.,0.4+phase*.6)
        lw=max(1,int(3*(1-phase)+beat*4)); draw.ellipse([cx-r,cy-r,cx+r,cy+r],outline=col,width=lw)
    n_tend=12+int(mid*20)
    for ti in range(n_tend):
        a0=2*math.pi*ti/n_tend+fi*.06
        r0=int(min(cx,cy)*.12); r1=int(min(cx,cy)*(.5+bass*.4+high*.2))
        pts=[]
        for si in range(9):
            sr2=r0+(r1-r0)*si/8; jit=math.sin(fi*.1+si*.9+ti*2.7)*mid*15
            a=a0+jit*.05; pts.append((int(cx+sr2*math.cos(a)),int(cy+sr2*math.sin(a))))
        hue=(ti/n_tend+fi*.003)%1.; col=_hsv(hue,.8,.7+beat*.3)
        for si in range(len(pts)-1): draw.line([pts[si],pts[si+1]],fill=col,width=max(1,int(beat*2+1)))
    r_core=int(min(cx,cy)*.1+bass*25)
    draw.ellipse([cx-r_core,cy-r_core,cx+r_core,cy+r_core],fill=pal["bg"],outline=pal["accent"],width=3)
    r2=r_core//2; draw.ellipse([cx-r2,cy-r2,cx+r2,cy+r2],fill=(255,255,255))


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# MODE: OSCILLOSCOPE
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _osc_frame(draw, w, h, bass, mid, high, beat, wave, fi, pal):
    draw.rectangle([0,0,w,h],fill=pal["bg"])
    gc=_lerp(pal["bg"],pal["p2"],.4)
    for x in range(0,w,w//8): draw.line([(x,0),(x,h)],fill=gc,width=1)
    for y in range(0,h,h//4): draw.line([(0,y),(w,y)],fill=gc,width=1)
    draw.line([(0,h//2),(w,h//2)],fill=_lerp(gc,pal["wave"],.4),width=1)
    n=min(len(wave),w)
    if n<2: return
    wr=wave[:n].copy(); wr/=(np.max(np.abs(wr))+1e-8)
    pts=[(int(j*w/n), int(h//2+wr[j]*(h//2-10)*(0.6+bass*.4))) for j in range(n)]
    pts=[(px,max(2,min(h-2,py))) for px,py in pts]
    col=pal["beat"] if beat>.5 else pal["wave"]
    lw=max(1,int(1+beat*2))
    for j in range(len(pts)-1): draw.line([pts[j],pts[j+1]],fill=col,width=lw)
    pv=float(np.max(np.abs(wr))); py2=int(h//2-pv*(h//2-10))
    draw.line([(0,py2),(w,py2)],fill=_lerp(pal["wave"],pal["bar_top"],.5),width=1)
    draw.line([(0,h-py2),(w,h-py2)],fill=_lerp(pal["wave"],pal["bar_top"],.5),width=1)


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# MODE: NEBULA
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _nebula_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    t=fi*.04; step=max(6,w//50)
    for x in range(0,w,step):
        for y in range(0,h,step):
            v=(math.sin(x*.025+t)*math.cos(y*.015+t*.8)+math.sin(math.sqrt(max(0,(x-w*.5)**2+(y-h*.5)**2))*.04+t*.9))
            v=(v+2)/4+bass*.2*math.sin(x*.05+t*2)
            col=_hsv((v*.5+fi*.002)%1.,.8+mid*.2,.2+v*.8)
            draw.rectangle([x,y,x+step,y+step],fill=col)
    if beat>.5:
        cx,cy=w//2,h//2
        for r in range(0,int(min(w,h)*.4),max(4,int(min(w,h)*.04))):
            col=_hsv((fi*.005+r*.003)%1.,1.,1.)
            draw.ellipse([cx-r,cy-r,cx+r,cy+r],outline=col)
    rng=random.Random(fi//8)
    for _ in range(30):
        sx=rng.randint(0,w);sy=rng.randint(0,h);b2=rng.randint(150,255)
        draw.point((sx,sy),fill=(b2,b2,b2))


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# MODE: AURORA €” northern lights curtains over star field
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _aurora_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    # Starfield
    draw.rectangle([0,0,w,h],fill=(0,0,10))
    rng=random.Random(fi//10)
    for _ in range(120):
        sx=rng.randint(0,w);sy=rng.randint(0,h//2)
        lum=rng.randint(80,200); sr=0 if rng.random()>.8 else 1
        draw.ellipse([sx-sr,sy-sr,sx+sr,sy+sr],fill=(lum,lum,lum))

    # Aurora curtains €” vertical bands of colour drifting horizontally
    n_curtains=6
    for ci in range(n_curtains):
        base_x = (ci/n_curtains + fi*0.002 + ci*0.17) % 1.0
        curtain_w = int(w * (0.15 + mid * 0.1))
        cx_pos = int(base_x * w * 1.2 - curtain_w//2)

        # Height varies with bass and position
        curtain_h = int(h * (0.35 + bass*0.35 + math.sin(fi*.05+ci)*0.1))

        hue = (ci/n_curtains * 0.4 + fi*0.002) % 1.0
        n_strips = max(4, curtain_w // 4)
        for si in range(n_strips):
            x = cx_pos + si * 4
            if x < 0 or x > w: continue
            wave_y = math.sin(fi*0.08 + si*0.3 + ci*1.5) * high * 30
            top_y  = max(0, int(h*0.1 + wave_y))
            bot_y  = min(h, top_y + curtain_h)

            # Vertical gradient: bright in middle, fade top and bottom
            mid_y  = (top_y + bot_y) // 2
            seg_h  = max(1, bot_y - top_y)
            for py in range(top_y, bot_y, 2):
                dist = abs(py - mid_y) / (seg_h/2 + 1)
                alpha = max(0., 1. - dist**1.5) * (0.4 + bass*0.3)
                col = _hsv(hue, 0.7, alpha)
                draw.line([(x, py),(x, py+2)], fill=col)

        hue = (hue + 0.1) % 1.0

    # Beat flash horizon glow
    if beat > 0.5:
        for y in range(h//2, h//2 + int(h*0.08)):
            alpha = (1 - (y - h//2) / (h*0.08)) * beat * 0.8
            col = _hsv((fi*.003)%1., 0.5, alpha)
            draw.line([(0,y),(w,y)], fill=col)


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# MODE: FRACTAL ZOOM €” zooming complex plane driven by bass
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _fractal_zoom_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    # Use a simple julia/mandelbrot-lite at low resolution, upscale
    rw, rh = w // 6, h // 6
    # Zoom in slowly, bass changes rate
    zoom   = 1.5 + fi * (0.008 + bass * 0.015)
    cx_off = -0.7 + math.cos(fi * 0.007) * 0.15 * (1 + mid)
    cy_off =  0.3 + math.sin(fi * 0.005) * 0.1

    max_iter = 24
    arr = np.zeros((rh, rw, 3), dtype=np.uint8)

    xs = np.linspace(cx_off - 1.5/zoom, cx_off + 1.5/zoom, rw)
    ys = np.linspace(cy_off - 1.0/zoom, cy_off + 1.0/zoom, rh)
    X, Y  = np.meshgrid(xs, ys)
    C     = X + 1j * Y
    Z     = np.zeros_like(C)
    iters = np.zeros(C.shape, dtype=np.float32)
    mask  = np.ones(C.shape, dtype=bool)
    for it in range(max_iter):
        Z[mask]     = Z[mask] ** 2 + C[mask]
        escaped      = mask & (np.abs(Z) > 2)
        iters[escaped] = it + 1 - np.log2(np.log2(np.abs(Z[escaped]) + 1e-10))
        mask[escaped]  = False

    # Colour mapping
    t_arr = iters / max_iter
    hue_arr = (t_arr * 3.0 + fi * 0.004 + beat * 0.05) % 1.0
    import colorsys
    for ry in range(rh):
        for rx in range(rw):
            if iters[ry,rx] == 0:
                arr[ry,rx] = pal["bg"]
            else:
                h_v = float(hue_arr[ry,rx])
                s_v = 0.7 + float(t_arr[ry,rx]) * 0.3
                v_v = 0.3 + float(t_arr[ry,rx]) * 0.7
                r2,g2,b2 = colorsys.hsv_to_rgb(h_v,s_v,v_v)
                arr[ry,rx] = (int(r2*255),int(g2*255),int(b2*255))

    # Upscale to full frame
    img_small = Image.fromarray(arr)
    img_full  = img_small.resize((w, h), Image.NEAREST)
    img_blur  = img_full.filter(ImageFilter.GaussianBlur(radius=1))
    draw._image.paste(img_blur)

    # Overlay beat burst
    if beat > 0.5:
        cx2,cy2=w//2,h//2
        for r in range(0,int(min(w,h)*.3),max(4,int(min(w,h)*.04))):
            col=_hsv((fi*.005+r*.003)%1.,1.,1.)
            draw.ellipse([cx2-r,cy2-r,cx2+r,cy2+r],outline=col,width=1)


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# MODE: DNA HELIX €” rotating 3D double helix
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _dna_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    draw.rectangle([0,0,w,h], fill=pal["bg"])

    cx = w // 2
    n_nodes = 40
    # Helix parameters
    amplitude = w // 4 + int(bass * w // 6)
    v_speed   = 0.04 + mid * 0.02
    rotation  = fi * v_speed
    node_spacing = h / n_nodes

    # Sort nodes by z (depth) for correct drawing order
    nodes_3d = []
    for i in range(n_nodes):
        angle = 2 * math.pi * i / 10.0 + rotation
        y_pos = (i * node_spacing + fi * 3) % h

        # Strand A
        x_a = cx + math.cos(angle) * amplitude
        z_a = math.sin(angle)

        # Strand B (180 offset)
        x_b = cx + math.cos(angle + math.pi) * amplitude
        z_b = math.sin(angle + math.pi)

        nodes_3d.append(("A", i, x_a, y_pos, z_a, angle))
        nodes_3d.append(("B", i, x_b, y_pos, z_b, angle + math.pi))

    # Sort by z so back nodes draw first
    nodes_3d.sort(key=lambda x: x[4])

    # Draw rungs (connecting lines) behind strands
    for i in range(n_nodes - 1):
        angle = 2 * math.pi * i / 10.0 + rotation
        y_pos = (i * node_spacing + fi * 3) % h
        x_a   = cx + math.cos(angle) * amplitude
        x_b   = cx + math.cos(angle + math.pi) * amplitude
        z_mid = 0.0
        if i % 3 == 0:  # every 3rd rung
            hue = (i / n_nodes + fi * 0.002) % 1.0
            rung_col = _hsv(hue, 0.6, 0.5 + high * 0.5)
            draw.line([(int(x_a), int(y_pos)), (int(x_b), int(y_pos))],
                      fill=rung_col, width=max(1, int(2 + beat * 2)))

    # Draw nodes
    prev_a = prev_b = None
    for strand, i, x, y, z, angle in nodes_3d:
        scale = 0.5 + (z + 1) * 0.25 + bass * 0.15  # depth scaling
        r = max(2, int(scale * 8 * (0.8 + high * 0.4)))
        hue = ((i / n_nodes) * 0.5 + fi * 0.003) % 1.0
        col = _hsv(hue, 1.0, 0.4 + scale * 0.6)
        draw.ellipse([int(x)-r, int(y)-r, int(x)+r, int(y)+r], fill=col)

    # Draw strand backbones
    strand_a_pts = sorted([n for n in nodes_3d if n[0]=="A"], key=lambda x: x[1])
    strand_b_pts = sorted([n for n in nodes_3d if n[0]=="B"], key=lambda x: x[1])
    for pts_list, col_key in [(strand_a_pts, "wave"), (strand_b_pts, "accent")]:
        prev = None
        for nd in pts_list:
            curr = (int(nd[2]), int(nd[3]))
            if prev and abs(curr[1] - prev[1]) < node_spacing * 2:
                draw.line([prev, curr], fill=pal[col_key], width=max(1,int(2+beat)))
            prev = curr


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# MODE: LIQUID METAL €” mercury surface with ripples
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _liquid_metal_frame(draw, w, h, bass, mid, high, beat, fi, pal):
    rw, rh = w // 4, h // 4
    t = fi * 0.05

    # Generate height map
    height_map = np.zeros((rh, rw), dtype=np.float32)
    for y in range(rh):
        for x in range(rw):
            nx, ny = x / rw, y / rh
            # Multiple overlapping sine waves = liquid surface
            height_map[y,x] = (
                math.sin(nx*8 + t + bass*3)    * 0.35
              + math.sin(ny*6 + t*0.7 + mid*2) * 0.30
              + math.sin((nx+ny)*5 + t*1.3)     * 0.20
              + math.sin(nx*ny*20 + t*2 + high) * 0.15
            )

    # Beat ripple €” add radial wave from centre
    if beat > 0.3:
        cx_n, cy_n = 0.5, 0.5
        for y in range(rh):
            for x in range(rw):
                dist = math.sqrt((x/rw - cx_n)**2 + (y/rh - cy_n)**2)
                height_map[y,x] += math.sin(dist * 30 - fi * 0.3) * beat * 0.4

    height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min() + 1e-8)

    # Lighting: normal-map based highlight
    # Compute normals from height differences
    arr = np.zeros((rh, rw, 3), dtype=np.uint8)
    for y in range(rh):
        for x in range(rw):
            # Sobel-like normal
            dx = height_map[y, min(x+1,rw-1)] - height_map[y, max(x-1,0)]
            dy = height_map[min(y+1,rh-1), x] - height_map[max(y-1,0), x]
            # Light direction (shifts with time)
            lx = math.sin(t * 0.3); ly = math.cos(t * 0.2); lz = 0.6
            nl = math.sqrt(lx*lx + ly*ly + lz*lz)
            dot = (-dx*lx - dy*ly + lz) / (nl + 1e-8)
            dot = max(0., min(1., dot))

            h_val = float(height_map[y,x])
            specular = dot ** 12 * (0.3 + high * 0.7)
            diffuse  = dot * (0.4 + bass * 0.2)

            # Base metal colour from palette
            base = _lerp(pal["p1"], pal["bar_mid"], h_val)
            r2 = min(255, int(base[0] * (diffuse + 0.3) + 255 * specular))
            g2 = min(255, int(base[1] * (diffuse + 0.3) + 255 * specular))
            b2 = min(255, int(base[2] * (diffuse + 0.3) + 255 * specular))
            arr[y,x] = (r2,g2,b2)

    img_small = Image.fromarray(arr)
    img_full  = img_small.resize((w, h), Image.BILINEAR)
    img_blur  = img_full.filter(ImageFilter.GaussianBlur(radius=0.8))
    draw._image.paste(img_blur)

    # Edge highlight ring
    cx2, cy2 = w // 2, h // 2
    r_ring = int(min(w, h) * 0.38 + bass * min(w,h) * 0.08)
    hue_r  = (fi * 0.004) % 1.0
    draw.ellipse([cx2-r_ring,cy2-r_ring,cx2+r_ring,cy2+r_ring],
                 outline=_hsv(hue_r, 0.3, 0.9 + high * 0.1), width=max(1,int(2+beat*3)))


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# Dispatch
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _render_frame(mode: str, w: int, h: int, bass: float, mid: float, high: float,
                  beat: float, wave: np.ndarray, fi: int, pal: dict,
                  bar_count: int, state: dict):

    if mode in ("fractal_zoom", "liquid_metal"):
        # These modes draw directly onto the image via draw._image
        img  = Image.new("RGB", (w, h), pal["bg"])
        draw = ImageDraw.Draw(img)
        if mode == "fractal_zoom":
            _fractal_zoom_frame(draw, w, h, bass, mid, high, beat, fi, pal)
        else:
            _liquid_metal_frame(draw, w, h, bass, mid, high, beat, fi, pal)
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

        elif mode == "nebula":
            _nebula_frame(draw, w, h, bass, mid, high, beat, fi, pal)

        elif mode == "aurora":
            _aurora_frame(draw, w, h, bass, mid, high, beat, fi, pal)

        elif mode == "dna_helix":
            _dna_frame(draw, w, h, bass, mid, high, beat, fi, pal)

    arr = np.array(img, dtype=np.float32) / 255.0
    return arr, state


# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
# ComfyUI Node
# ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

class S42PAudioVisualizer:
    """
    S42P Audio Visualizer â€” 11 audio-reactive modes for music videos.

    MODES:
      spectrum_classic â€” bars + scope ring + particles (retro-inspired)
      lava_lamp        â€” morphing physics blobs, full-screen travel
      wormhole         â€” true 3D perspective tunnel with depth + twist
      matrix           â€” falling character rain columns
      stargate         â€” expanding portal rings + energy tendrils
      oscilloscope     â€” clean waveform with peak indicators
      nebula           â€” fractal plasma cloud with beat explosions
      aurora           â€” northern lights curtains over starfield
      fractal_zoom     â€” mandelbrot zoom accelerating with bass
      dna_helix        â€” rotating 3D double helix with rungs
      liquid_metal     â€” chrome/mercury surface with light + ripples

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
    CATEGORY      = "S42 Production Suite ðŸŽ¨ Visualizer"

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

        print(f"[S42P Visualizer] mode={mode}  {len(wav_np)/sr:.1f}s  "
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
        print(f"[S42P Visualizer] âœ“ {n_frames} frames  {batch.nbytes/1024/1024:.1f}MB")

        return (torch.from_numpy(batch) if TORCH_AVAILABLE else batch,)


NODE_CLASS_MAPPINGS        = {"S42PAudioVisualizer": S42PAudioVisualizer}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PAudioVisualizer": "S42P Audio Visualizer ðŸŽ¨"}
