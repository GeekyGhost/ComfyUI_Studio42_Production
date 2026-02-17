"""
S42 Production Suite — Procedural FX Pack v4.0
================================================
16 animated procedural background/overlay generators.
All nodes output IMAGE tensors compatible with ComfyUI batch pipelines.

Supports transparent RGBA output for compositing with S42P Layer Composer.
Batch-aware with deterministic animation via seed.

Python 3.12 | ComfyUI Portable
Deps: Pillow, numpy (ComfyUI default) | torch (ComfyUI default)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rng(seed: int) -> random.Random:
    r = random.Random()
    r.seed(seed & 0xFFFFFFFF)
    return r


def _hash(*parts: int) -> int:
    h = 2166136261
    for p in parts:
        h ^= (int(p) & 0xFFFFFFFF)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _safe_font(px: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def _to_tensor(img: Image.Image):
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    arr = np.array(img).astype(np.float32) / 255.0
    try:
        import torch as _torch
        return _torch.from_numpy(arr)
    except Exception:
        return arr


def _interp_color(c1: Tuple, c2: Tuple, t: float) -> Tuple[int, int, int]:
    return (
        int(c1[0] * (1 - t) + c2[0] * t),
        int(c1[1] * (1 - t) + c2[1] * t),
        int(c1[2] * (1 - t) + c2[2] * t),
    )


def _compose_glow(
    fg: Image.Image,
    glow_radius: float,
    glow_gain: float,
    bg_color: Tuple[int, int, int],
    bg_alpha: int,
) -> Image.Image:
    out = Image.new("RGBA", fg.size, (*bg_color, bg_alpha))
    if glow_radius > 0 and glow_gain > 0:
        glow = fg.filter(ImageFilter.GaussianBlur(glow_radius))
        if glow_gain != 1.0:
            r, g, b, a = glow.split()
            r = r.point(lambda v: min(int(v * glow_gain), 255))
            g = g.point(lambda v: min(int(v * glow_gain), 255))
            b = b.point(lambda v: min(int(v * glow_gain), 255))
            glow = Image.merge("RGBA", (r, g, b, a))
        out.alpha_composite(glow)
    out.alpha_composite(fg)
    return out


def _stack_batch(imgs: list):
    try:
        import torch as _torch
        return (_torch.stack(imgs, dim=0),)
    except Exception:
        return (np.stack(imgs, axis=0),)


# ── Shared INPUT_TYPES blocks ─────────────────────────────────────────────────

_BATCH_INPUTS = {
    "frame":         ("INT",     {"default": 0, "min": 0, "max": 10_000_000,
                                  "tooltip": "Starting frame index for animation. Connect to a frame counter node for video output."}),
    "animate_batch": ("BOOLEAN", {"default": True,
                                  "tooltip": "When True each image in the batch advances one frame. When False all images use the same frame (still)."}),
    "batch_size":    ("INT",     {"default": 8, "min": 1, "max": 256,
                                  "tooltip": "Number of frames to generate. Match your video frame count."}),
}

_BG_INPUTS = {
    "bg_transparent": ("BOOLEAN", {"default": False,
                                   "tooltip": "Output RGBA with transparent background for compositing in S42P Layer Composer."}),
    "bg_r":           ("INT",     {"default": 0, "min": 0, "max": 255,
                                   "tooltip": "Background red channel (0-255). Only used when bg_transparent is off."}),
    "bg_g":           ("INT",     {"default": 0, "min": 0, "max": 255,
                                   "tooltip": "Background green channel (0-255)."}),
    "bg_b":           ("INT",     {"default": 0, "min": 0, "max": 255,
                                   "tooltip": "Background blue channel (0-255)."}),
}

_COLOR_INPUTS = {
    "override_colors": ("BOOLEAN", {"default": False,
                                    "tooltip": "Enable custom color override. When False, effect uses its designed color palette."}),
    "col1_r": ("INT", {"default": 255, "min": 0, "max": 255, "tooltip": "Primary color red channel."}),
    "col1_g": ("INT", {"default": 255, "min": 0, "max": 255, "tooltip": "Primary color green channel."}),
    "col1_b": ("INT", {"default": 255, "min": 0, "max": 255, "tooltip": "Primary color blue channel."}),
    "col2_r": ("INT", {"default": 180, "min": 0, "max": 255, "tooltip": "Secondary color red channel. Particles interpolate between primary and secondary."}),
    "col2_g": ("INT", {"default": 180, "min": 0, "max": 255, "tooltip": "Secondary color green channel."}),
    "col2_b": ("INT", {"default": 180, "min": 0, "max": 255, "tooltip": "Secondary color blue channel."}),
}

_MOTION_INPUTS = {
    "speed_mult":          ("FLOAT",   {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05,
                                        "tooltip": "Global speed multiplier. 1.0 = default speed. 2.0 = twice as fast."}),
    "use_custom_direction":("BOOLEAN", {"default": False,
                                        "tooltip": "Override built-in direction with the angle_deg value below."}),
    "angle_deg":           ("FLOAT",   {"default": 90.0, "min": -360.0, "max": 360.0, "step": 1.0,
                                        "tooltip": "Direction angle in degrees. 0=right, 90=down, 180=left, 270=up. Only used when use_custom_direction is on."}),
    "reverse":             ("BOOLEAN", {"default": False,
                                        "tooltip": "Reverse the default direction of the effect (e.g. snow falls up, bubbles fall down)."}),
}


# ── Base node class ────────────────────────────────────────────────────────────

class _FXBase:
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "generate"
    CATEGORY      = "S42 Production Suite/Procedural FX"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "width":  ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8,
                               "tooltip": "Output image width in pixels."}),
            "height": ("INT", {"default": 576,  "min": 64, "max": 4096, "step": 8,
                               "tooltip": "Output image height in pixels."}),
            "seed":   ("INT", {"default": 12345, "min": 0, "max": 2_147_483_647,
                               "tooltip": "Random seed for particle placement. Same seed = same layout every time."}),
        }
        optional = {}
        optional.update(_BATCH_INPUTS)
        optional.update(_BG_INPUTS)
        optional.update(_COLOR_INPUTS)
        optional.update(_MOTION_INPUTS)
        return {"required": required, "optional": optional}

    def _render(self, width: int, height: int, seed: int, frame: int, **kw) -> Image.Image:
        raise NotImplementedError

    def generate(self, width, height, seed, frame=0, animate_batch=True, batch_size=8, **kw):
        imgs = []
        b = max(1, int(batch_size))
        for i in range(b):
            f = frame + i if animate_batch else frame
            imgs.append(_to_tensor(self._render(width, height, int(seed), f, **kw)))
        return _stack_batch(imgs)


def _bg_from_kw(kw: dict, default_color=(0, 0, 0)):
    bg_transparent = bool(kw.get("bg_transparent", False))
    bg_alpha = 0 if bg_transparent else 255
    bg_color = (
        int(kw.get("bg_r", default_color[0])),
        int(kw.get("bg_g", default_color[1])),
        int(kw.get("bg_b", default_color[2])),
    )
    return bg_color, bg_alpha


def _colors_from_kw(kw: dict, default1, default2):
    if bool(kw.get("override_colors", False)):
        c1 = (int(kw.get("col1_r", default1[0])),
              int(kw.get("col1_g", default1[1])),
              int(kw.get("col1_b", default1[2])))
        c2 = (int(kw.get("col2_r", default2[0])),
              int(kw.get("col2_g", default2[1])),
              int(kw.get("col2_b", default2[2])))
    else:
        c1, c2 = default1, default2
    return c1, c2


def _direction_from_kw(kw: dict, default_gx: float, default_gy: float):
    speed_mult = float(kw.get("speed_mult", 1.0))
    reverse    = bool(kw.get("reverse", False))
    if bool(kw.get("use_custom_direction", False)):
        ang = math.radians(float(kw.get("angle_deg", 90.0)) + (180.0 if reverse else 0.0))
        gx, gy = math.cos(ang), math.sin(ang)
    else:
        gx, gy = default_gx, default_gy
        if reverse:
            gx, gy = -gx, -gy
    return gx, gy, speed_mult


# ── Drawing primitives ─────────────────────────────────────────────────────────

def _draw_circle(draw, x, y, sz, col, a):
    draw.ellipse([x - sz / 2, y - sz / 2, x + sz / 2, y + sz / 2], fill=(*col, a))


def _draw_ring(draw, x, y, sz, col, a):
    w = max(1, int(sz * 0.12))
    draw.ellipse([x - sz / 2, y - sz / 2, x + sz / 2, y + sz / 2],
                 outline=(*col, a), width=w)


def _draw_star(draw, x, y, sz, rot, col, a):
    pts = []
    for i in range(10):
        R = sz / 2 if i % 2 == 0 else sz / 4
        ang = math.radians(-90 + i * 36 + rot)
        pts.append((x + R * math.cos(ang), y + R * math.sin(ang)))
    draw.polygon(pts, fill=(*col, a))


def _draw_heart(draw, x, y, sz, rot, col, a):
    s = max(8, int(sz))
    spr = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d2  = ImageDraw.Draw(spr)
    cx, cy, r = s / 2, s * 0.45, s * 0.22
    d2.pieslice([cx - r * 2, cy - r, cx, cy + r], 0, 360, fill=(*col, a))
    d2.pieslice([cx, cy - r, cx + r * 2, cy + r], 0, 360, fill=(*col, a))
    d2.polygon([(cx - r * 2, cy), (cx + r * 2, cy), (cx, s)], fill=(*col, a))
    spr = spr.rotate(rot, resample=Image.BICUBIC, expand=True)
    draw.bitmap((x - spr.width / 2, y - spr.height / 2), spr)


def _draw_diamond(draw, x, y, sz, rot, col, a):
    pts = []
    for i in range(4):
        ang = math.radians(i * 90 + rot)
        pts.append((x + sz / 2 * math.cos(ang), y + sz / 2 * math.sin(ang)))
    draw.polygon(pts, fill=(*col, a))


def _draw_confetti_piece(draw, x, y, sz, rot, col, a):
    s = max(6, sz)
    spr = Image.new("RGBA", (int(s), int(s)), (0, 0, 0, 0))
    d2  = ImageDraw.Draw(spr)
    if (int(x + y) & 1) == 0:
        d2.polygon([(0, s * 0.2), (s, s * 0.2), (s * 0.8, s), (s * 0.2, s)], fill=(*col, a))
    else:
        d2.polygon([(s * 0.5, 0), (s, s), (0, s)], fill=(*col, a))
    spr = spr.rotate(rot, resample=Image.BICUBIC, expand=True)
    draw.bitmap((x - spr.width / 2, y - spr.height / 2), spr)


def _draw_vline(draw, x, y, sz, col, a):
    w = max(1, int(sz * 0.18))
    draw.rectangle([x - w / 2, y - sz / 1.8, x + w / 2, y + sz / 1.8], fill=(*col, a))


def _draw_snowflake(draw, x, y, sz, rot, col, a):
    cx, cy = x, y
    arms = 6
    for arm in range(arms):
        ang = math.radians(arm * 60 + rot)
        ex = cx + math.cos(ang) * sz / 2
        ey = cy + math.sin(ang) * sz / 2
        w = max(1, int(sz * 0.08))
        draw.line([(cx, cy), (ex, ey)], fill=(*col, a), width=w)
        for bar_t in (0.4, 0.7):
            bx = cx + math.cos(ang) * sz / 2 * bar_t
            by = cy + math.sin(ang) * sz / 2 * bar_t
            perp = ang + math.pi / 3
            bl = sz * 0.12
            draw.line([(bx - math.cos(perp) * bl, by - math.sin(perp) * bl),
                       (bx + math.cos(perp) * bl, by + math.sin(perp) * bl)],
                      fill=(*col, a), width=w)


def _draw_lightning_bolt(draw, x, y, sz, col, a):
    w = max(1, int(sz * 0.14))
    pts = [
        (x + sz * 0.1,  y - sz * 0.5),
        (x - sz * 0.05, y),
        (x + sz * 0.12, y),
        (x - sz * 0.1,  y + sz * 0.5),
    ]
    draw.polygon(pts, fill=(*col, a))


def _draw_cross(draw, x, y, sz, rot, col, a):
    w = max(1, int(sz * 0.2))
    ang = math.radians(rot)
    def arm(dx, dy):
        return (x + dx * math.cos(ang) - dy * math.sin(ang),
                y + dx * math.sin(ang) + dy * math.cos(ang))
    corners = [arm(-sz/2, -w/2), arm(sz/2, -w/2), arm(sz/2, w/2), arm(-sz/2, w/2)]
    draw.polygon(corners, fill=(*col, a))
    corners2 = [arm(-w/2, -sz/2), arm(w/2, -sz/2), arm(w/2, sz/2), arm(-w/2, sz/2)]
    draw.polygon(corners2, fill=(*col, a))


def _draw_text_glyph(draw, glyph, x, y, sz, col, a, font):
    if a <= 0:
        return
    try:
        draw.text((x - sz * 0.33, y - sz * 0.55), glyph, font=font, fill=(*col, a))
    except Exception:
        draw.text((int(x), int(y)), glyph, fill=(*col, a))


# ── Particle renderer (shared) ────────────────────────────────────────────────

def _render_particles(
    width, height, seed, frame,
    count_per_mega, gx, gy, speed_min, speed_max,
    size_min, size_max, rot_min, rot_max, life_frames,
    color1, color2, alpha, trail, shape,
    glow_radius, glow_gain, bg_color, bg_alpha,
    glyphs="",
):
    fg   = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(fg, "RGBA")
    mega  = (width * height) / 1_000_000.0
    count = max(1, int(count_per_mega * mega))
    font  = _safe_font(int(max(10, size_max))) if shape == "text" else None

    for i in range(count):
        h   = _hash(seed, i, 42)
        rnd = _rng(h)
        life = max(1, int(life_frames))
        t    = (frame + (h % life)) % life
        sx   = rnd.random() * width
        sy   = rnd.random() * height
        spd  = rnd.uniform(speed_min, speed_max)
        sz   = rnd.uniform(size_min, size_max)
        rot  = rnd.uniform(rot_min, rot_max)
        x    = (sx + gx * t * spd) % width
        y    = (sy + gy * t * spd) % height
        col  = _interp_color(color1, color2, rnd.random())
        steps = max(0, int(trail))

        for k in range(steps, -1, -1):
            fade = 1.0 if steps == 0 else (k / steps)
            aa   = int(alpha * (fade ** 1.2))
            if aa <= 0:
                continue
            yy = (y - gy * k * spd) % height
            xx = (x - gx * k * spd) % width

            if shape == "circle":
                _draw_circle(draw, xx, yy, sz, col, aa)
            elif shape == "bubble":
                _draw_ring(draw, xx, yy, sz, col, aa)
            elif shape == "star":
                _draw_star(draw, xx, yy, sz, rot, col, aa)
            elif shape == "heart":
                _draw_heart(draw, xx, yy, sz, rot, col, aa)
            elif shape == "diamond":
                _draw_diamond(draw, xx, yy, sz, rot, col, aa)
            elif shape == "confetti":
                _draw_confetti_piece(draw, xx, yy, sz, rot, col, aa)
            elif shape == "line":
                _draw_vline(draw, xx, yy, sz, col, aa)
            elif shape == "snowflake":
                _draw_snowflake(draw, xx, yy, sz, rot, col, aa)
            elif shape == "lightning":
                _draw_lightning_bolt(draw, xx, yy, sz, col, aa)
            elif shape == "cross":
                _draw_cross(draw, xx, yy, sz, rot, col, aa)
            elif shape == "text" and glyphs and font:
                glyph = glyphs[h % len(glyphs)]
                _draw_text_glyph(draw, glyph, xx, yy, sz, col, aa, font)

    return _compose_glow(fg, glow_radius, glow_gain, bg_color, bg_alpha)


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Hearts Rain
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_HeartsRain(_FXBase):
    """Falling hearts — romance, Valentine, celebration overlays."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"]   = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                    "tooltip": "Particle density multiplier. Higher = more hearts on screen."})
        base["optional"]["glow"]      = ("FLOAT", {"default": 2.5, "min": 0.0, "max": 12.0, "step": 0.5,
                                                    "tooltip": "Glow blur radius in pixels. 0 = no glow."})
        base["optional"]["size_scale"]= ("FLOAT", {"default": 1.0, "min": 0.3, "max": 3.0, "step": 0.1,
                                                    "tooltip": "Scale heart size. 1.0 = default, 2.0 = twice as large."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (255, 70, 110), (255, 160, 190))
        gx, gy, sm = _direction_from_kw(kw, 0.0, 1.0)
        density    = float(kw.get("density", 1.0))
        glow       = float(kw.get("glow", 2.5))
        scale      = float(kw.get("size_scale", 1.0))
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=900 * density, gx=gx, gy=gy,
            speed_min=1.2 * sm, speed_max=3.3 * sm,
            size_min=10 * scale, size_max=28 * scale,
            rot_min=-20, rot_max=20, life_frames=600,
            color1=c1, color2=c2, alpha=235, trail=3,
            shape="heart", glow_radius=glow, glow_gain=1.4,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Snow
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_Snow(_FXBase):
    """Realistic snowfall with slight drift."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"]    = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                     "tooltip": "Snowflake density. Higher = heavier snowfall."})
        base["optional"]["shape"]      = (["circle", "snowflake"], {"default": "circle",
                                          "tooltip": "Circle = soft bokeh-style flakes. Snowflake = 6-arm crystalline shape."})
        base["optional"]["drift"]      = ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                                                    "tooltip": "Horizontal wind drift amount. 0 = straight down."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (240, 240, 255), (180, 200, 255))
        gx, gy, sm = _direction_from_kw(kw, float(kw.get("drift", 0.1)), 1.0)
        density = float(kw.get("density", 1.0))
        shape   = kw.get("shape", "circle")
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=1200 * density, gx=gx, gy=gy,
            speed_min=0.6 * sm, speed_max=1.5 * sm,
            size_min=6, size_max=16,
            rot_min=-10, rot_max=10, life_frames=900,
            color1=c1, color2=c2, alpha=220, trail=1,
            shape=shape, glow_radius=1.0, glow_gain=1.1,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Starfield
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_Starfield(_FXBase):
    """Space starfield with optional warp-speed parallax."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"]    = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                     "tooltip": "Star density. 1.0 = normal, 3.0 = very dense."})
        base["optional"]["warp_speed"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                                    "tooltip": "Warp trail length. 0 = no trails, 1.0 = full warp effect."})
        base["optional"]["shape"]      = (["star", "circle"], {"default": "star",
                                          "tooltip": "Star = 5-point star shapes. Circle = round dots (faster)."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (255, 240, 200), (180, 220, 255))
        gx, gy, sm = _direction_from_kw(kw, 0.0, 0.3)
        density    = float(kw.get("density", 1.0))
        warp       = float(kw.get("warp_speed", 0.0))
        shape      = kw.get("shape", "star")
        trail      = int(warp * 12)
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=800 * density, gx=gx, gy=gy,
            speed_min=0.3 * sm, speed_max=1.0 * sm,
            size_min=6, size_max=20,
            rot_min=0, rot_max=360, life_frames=800,
            color1=c1, color2=c2, alpha=230, trail=trail,
            shape=shape, glow_radius=2.0, glow_gain=1.5,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Bokeh Orbs
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_BokehOrbs(_FXBase):
    """Soft out-of-focus light circles for cinematic overlays."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"]    = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                     "tooltip": "Number of orbs. Keep low (0.3-1.0) for subtle cinematic look."})
        base["optional"]["orb_size"]   = ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.1,
                                                     "tooltip": "Size multiplier for orbs. Large orbs overlap beautifully at low density."})
        base["optional"]["glow"]       = ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.5,
                                                     "tooltip": "Glow blur radius. High values create soft, dreamy bokeh look."})
        base["optional"]["opacity"]    = ("INT",   {"default": 200, "min": 10, "max": 255,
                                                     "tooltip": "Orb opacity (0-255). Lower values allow layering over video."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (120, 255, 130), (80, 200, 255))
        gx, gy, sm = _direction_from_kw(kw, 0.0, 0.2)
        density = float(kw.get("density", 1.0))
        scale   = float(kw.get("orb_size", 1.0))
        glow    = float(kw.get("glow", 4.0))
        opacity = int(kw.get("opacity", 200))
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=500 * density, gx=gx, gy=gy,
            speed_min=0.2 * sm, speed_max=0.8 * sm,
            size_min=18 * scale, size_max=42 * scale,
            rot_min=0, rot_max=0, life_frames=1200,
            color1=c1, color2=c2, alpha=opacity, trail=0,
            shape="circle", glow_radius=glow, glow_gain=1.4,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Fireflies
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_Fireflies(_FXBase):
    """Drifting luminous fireflies with organic motion."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"] = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                  "tooltip": "Number of fireflies per megapixel."})
        base["optional"]["glow"]    = ("FLOAT", {"default": 3.0, "min": 0.0, "max": 12.0, "step": 0.5,
                                                  "tooltip": "Glow halo size. High values create warm lantern effect."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (230, 255, 120), (80, 255, 140))
        gx, gy, sm = _direction_from_kw(kw, 0.2, -0.1)
        density = float(kw.get("density", 1.0))
        glow    = float(kw.get("glow", 3.0))
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=650 * density, gx=gx, gy=gy,
            speed_min=0.5 * sm, speed_max=1.8 * sm,
            size_min=6, size_max=12,
            rot_min=0, rot_max=0, life_frames=700,
            color1=c1, color2=c2, alpha=255, trail=2,
            shape="circle", glow_radius=glow, glow_gain=1.8,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 6 — Confetti
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_Confetti(_FXBase):
    """Falling celebration confetti with tumbling rotation."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"] = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                  "tooltip": "Confetti piece density. 2.0+ for heavy party effects."})
        base["optional"]["shape"]   = (["confetti", "diamond", "star"], {"default": "confetti",
                                       "tooltip": "Confetti = flat rectangles/triangles. Diamond = rhombus. Star = 5-point stars."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (255, 200, 0), (0, 200, 255))
        gx, gy, sm = _direction_from_kw(kw, 0.0, 1.4)
        density = float(kw.get("density", 1.0))
        shape   = kw.get("shape", "confetti")
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=1800 * density, gx=gx, gy=gy,
            speed_min=1.0 * sm, speed_max=3.0 * sm,
            size_min=8, size_max=16,
            rot_min=0, rot_max=360, life_frames=500,
            color1=c1, color2=c2, alpha=230, trail=1,
            shape=shape, glow_radius=0.0, glow_gain=0.0,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 7 — Matrix Rain (Binary)
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_MatrixRain(_FXBase):
    """Classic digital rain — falling 0s and 1s with glowing trails."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"]   = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                                                    "tooltip": "Column density. 1.0 = classic Matrix look."})
        base["optional"]["trail"]     = ("INT",   {"default": 8, "min": 1, "max": 20,
                                                    "tooltip": "Trail length in characters. Higher = longer green streaks."})
        base["optional"]["glow"]      = ("FLOAT", {"default": 1.8, "min": 0.0, "max": 8.0, "step": 0.25,
                                                    "tooltip": "Glow radius for the digital rain effect."})
        base["optional"]["char_set"]  = (["01", "01アイウエオカキクケコサシスセソ", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"],
                                         {"default": "01",
                                          "tooltip": "Character set. Binary = 0s and 1s. Katakana = Matrix film style. Alphanumeric = hacker style."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (120, 255, 130), (50, 200, 90))
        gx, gy, sm = _direction_from_kw(kw, 0.0, 1.1)
        density  = float(kw.get("density", 1.0))
        trail    = int(kw.get("trail", 8))
        glow     = float(kw.get("glow", 1.8))
        char_set = kw.get("char_set", "01")
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=1400 * density, gx=gx, gy=gy,
            speed_min=0.8 * sm, speed_max=2.6 * sm,
            size_min=14, size_max=22,
            rot_min=0, rot_max=0, life_frames=700,
            color1=c1, color2=c2, alpha=235, trail=trail,
            shape="text", glow_radius=glow, glow_gain=1.3,
            bg_color=bg_color, bg_alpha=bg_alpha,
            glyphs=char_set,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 8 — Neon Scanlines
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_NeonScanlines(_FXBase):
    """Scrolling neon vertical bars — retro CRT / synthwave aesthetic."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"] = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                  "tooltip": "Scanline density. Low = sparse dramatic lines. High = busy CRT look."})
        base["optional"]["glow"]    = ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5,
                                                  "tooltip": "Neon glow halo radius."})
        base["optional"]["opacity"] = ("INT",   {"default": 180, "min": 10, "max": 255,
                                                  "tooltip": "Line opacity. Low values work well as video overlays."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (120, 255, 130), (30, 180, 255))
        gx, gy, sm = _direction_from_kw(kw, 0.0, 1.8)
        density = float(kw.get("density", 1.0))
        glow    = float(kw.get("glow", 2.0))
        opacity = int(kw.get("opacity", 180))
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=260 * density, gx=gx, gy=gy,
            speed_min=1.0 * sm, speed_max=2.2 * sm,
            size_min=24, size_max=48,
            rot_min=0, rot_max=0, life_frames=400,
            color1=c1, color2=c2, alpha=opacity, trail=0,
            shape="line", glow_radius=glow, glow_gain=1.6,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 9 — Rising Bubbles
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_RisingBubbles(_FXBase):
    """Transparent rising bubbles — underwater, dreamy, liquid overlays."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"]  = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                   "tooltip": "Bubble density. High values create champagne effect."})
        base["optional"]["size"]     = ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.1,
                                                   "tooltip": "Bubble size multiplier."})
        base["optional"]["filled"]   = ("BOOLEAN", {"default": False,
                                                     "tooltip": "Filled circles instead of rings. Rings look more like real bubbles."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (160, 220, 255), (120, 255, 220))
        gx, gy, sm = _direction_from_kw(kw, 0.0, -1.1)
        density = float(kw.get("density", 1.0))
        scale   = float(kw.get("size", 1.0))
        filled  = bool(kw.get("filled", False))
        shape   = "circle" if filled else "bubble"
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=900 * density, gx=gx, gy=gy,
            speed_min=0.7 * sm, speed_max=2.0 * sm,
            size_min=10 * scale, size_max=28 * scale,
            rot_min=0, rot_max=0, life_frames=900,
            color1=c1, color2=c2, alpha=220, trail=2,
            shape=shape, glow_radius=1.5, glow_gain=1.2,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 10 — Lightning Storm
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_LightningStorm(_FXBase):
    """Electric lightning bolts raining down — storm, energy, EDM."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"] = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                  "tooltip": "Lightning bolt density."})
        base["optional"]["glow"]    = ("FLOAT", {"default": 3.0, "min": 0.0, "max": 12.0, "step": 0.5,
                                                  "tooltip": "Electric glow radius. Higher = more intense energy."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (255, 255, 100), (100, 180, 255))
        gx, gy, sm = _direction_from_kw(kw, 0.0, 1.5)
        density = float(kw.get("density", 1.0))
        glow    = float(kw.get("glow", 3.0))
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=400 * density, gx=gx, gy=gy,
            speed_min=2.0 * sm, speed_max=5.0 * sm,
            size_min=16, size_max=36,
            rot_min=0, rot_max=0, life_frames=300,
            color1=c1, color2=c2, alpha=240, trail=0,
            shape="lightning", glow_radius=glow, glow_gain=2.0,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 11 — Geometric Storm
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_GeometricStorm(_FXBase):
    """Spinning geometric shapes — abstract, futuristic, motion graphics."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"] = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                  "tooltip": "Number of shapes on screen."})
        base["optional"]["shape"]   = (["star", "diamond", "cross", "confetti"], {"default": "diamond",
                                       "tooltip": "Geometric shape to use."})
        base["optional"]["glow"]    = ("FLOAT", {"default": 1.5, "min": 0.0, "max": 8.0, "step": 0.25,
                                                  "tooltip": "Shape glow/bloom effect."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (255, 80, 200), (80, 200, 255))
        gx, gy, sm = _direction_from_kw(kw, 0.0, 1.0)
        density = float(kw.get("density", 1.0))
        shape   = kw.get("shape", "diamond")
        glow    = float(kw.get("glow", 1.5))
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=700 * density, gx=gx, gy=gy,
            speed_min=0.8 * sm, speed_max=2.5 * sm,
            size_min=8, size_max=28,
            rot_min=0, rot_max=360, life_frames=600,
            color1=c1, color2=c2, alpha=220, trail=2,
            shape=shape, glow_radius=glow, glow_gain=1.4,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 12 — Meteor Shower
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_MeteorShower(_FXBase):
    """Diagonal meteor streaks across a dark sky."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"]     = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                      "tooltip": "Meteor frequency. High density = meteor storm."})
        base["optional"]["trail_length"]= ("INT",   {"default": 6, "min": 1, "max": 20,
                                                      "tooltip": "Streak trail length. Higher = longer shooting star tail."})
        base["optional"]["glow"]        = ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.25,
                                                      "tooltip": "Meteor glow halo."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (255, 255, 255), (200, 220, 255))
        gx, gy, sm = _direction_from_kw(kw, 0.7, 0.7)
        density = float(kw.get("density", 1.0))
        trail   = int(kw.get("trail_length", 6))
        glow    = float(kw.get("glow", 2.0))
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=300 * density, gx=gx, gy=gy,
            speed_min=3.0 * sm, speed_max=7.0 * sm,
            size_min=4, size_max=10,
            rot_min=0, rot_max=0, life_frames=400,
            color1=c1, color2=c2, alpha=255, trail=trail,
            shape="circle", glow_radius=glow, glow_gain=1.6,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 13 — Petal Rain
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_PetalRain(_FXBase):
    """Falling flower petals — sakura blossom, nature, soft organic."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"]  = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                   "tooltip": "Petal density. 1.0 = gentle shower, 3.0 = storm."})
        base["optional"]["drift"]    = ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.5, "step": 0.05,
                                                   "tooltip": "Horizontal wind drift. 0 = straight down, 1.0 = strong breeze."})
        base["optional"]["size"]     = ("FLOAT", {"default": 1.0, "min": 0.3, "max": 3.0, "step": 0.1,
                                                   "tooltip": "Petal size multiplier."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (255, 180, 200), (255, 220, 230))
        drift  = float(kw.get("drift", 0.3))
        gx, gy, sm = _direction_from_kw(kw, drift, 1.0)
        density = float(kw.get("density", 1.0))
        scale   = float(kw.get("size", 1.0))
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=700 * density, gx=gx, gy=gy,
            speed_min=0.5 * sm, speed_max=1.5 * sm,
            size_min=8 * scale, size_max=22 * scale,
            rot_min=-30, rot_max=30, life_frames=900,
            color1=c1, color2=c2, alpha=210, trail=1,
            shape="confetti", glow_radius=0.5, glow_gain=1.1,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 14 — Gold Dust
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_GoldDust(_FXBase):
    """Rising golden sparkle particles — luxury, awards, magic."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"] = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                                                  "tooltip": "Sparkle particle count."})
        base["optional"]["glow"]    = ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.25,
                                                  "tooltip": "Golden shimmer radius."})
        base["optional"]["shape"]   = (["star", "circle", "diamond"], {"default": "star",
                                       "tooltip": "Sparkle shape."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (255, 220, 60), (255, 180, 20))
        gx, gy, sm = _direction_from_kw(kw, 0.0, -0.8)
        density = float(kw.get("density", 1.0))
        glow    = float(kw.get("glow", 2.0))
        shape   = kw.get("shape", "star")
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=1000 * density, gx=gx, gy=gy,
            speed_min=0.4 * sm, speed_max=1.5 * sm,
            size_min=4, size_max=14,
            rot_min=0, rot_max=360, life_frames=800,
            color1=c1, color2=c2, alpha=240, trail=1,
            shape=shape, glow_radius=glow, glow_gain=2.0,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 15 — Cyber Grid
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_CyberGrid(_FXBase):
    """Scrolling neon crosshair/cross symbols — sci-fi HUD, cyberpunk UI."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["density"] = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                                                  "tooltip": "Grid element density."})
        base["optional"]["glow"]    = ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.25,
                                                  "tooltip": "HUD glow radius."})
        base["optional"]["opacity"] = ("INT",   {"default": 160, "min": 10, "max": 255,
                                                  "tooltip": "Grid opacity. Low values work well as video overlays."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (0, 255, 200), (0, 180, 255))
        gx, gy, sm = _direction_from_kw(kw, 0.0, 0.5)
        density = float(kw.get("density", 1.0))
        glow    = float(kw.get("glow", 2.0))
        opacity = int(kw.get("opacity", 160))
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=400 * density, gx=gx, gy=gy,
            speed_min=0.3 * sm, speed_max=0.8 * sm,
            size_min=12, size_max=24,
            rot_min=0, rot_max=90, life_frames=1000,
            color1=c1, color2=c2, alpha=opacity, trail=0,
            shape="cross", glow_radius=glow, glow_gain=1.5,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 16 — Plasma Orbs
# ═══════════════════════════════════════════════════════════════════════════════

class S42P_FX_PlasmaOrbs(_FXBase):
    """Large slow-moving plasma/lava orbs — psychedelic, ambient, motion art."""

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["count"]   = ("INT",   {"default": 8, "min": 2, "max": 40,
                                                  "tooltip": "Number of plasma orbs. Keep low (4-12) for classic lava lamp look."})
        base["optional"]["orb_size"]= ("FLOAT", {"default": 1.0, "min": 0.3, "max": 4.0, "step": 0.1,
                                                  "tooltip": "Orb size multiplier. Large overlapping orbs create plasma blending."})
        base["optional"]["glow"]    = ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 0.5,
                                                  "tooltip": "Plasma blur/glow amount. High values create soft lava lamp look."})
        base["optional"]["opacity"] = ("INT",   {"default": 180, "min": 20, "max": 255,
                                                  "tooltip": "Orb opacity. Lower allows layering."})
        return base

    def _render(self, width, height, seed, frame, **kw):
        bg_color, bg_alpha = _bg_from_kw(kw)
        c1, c2 = _colors_from_kw(kw, (255, 60, 180), (60, 100, 255))
        gx, gy, sm = _direction_from_kw(kw, 0.0, 0.1)
        count   = int(kw.get("count", 8))
        scale   = float(kw.get("orb_size", 1.0))
        glow    = float(kw.get("glow", 8.0))
        opacity = int(kw.get("opacity", 180))
        mega    = (width * height) / 1_000_000.0
        min_sz  = max(80, int(120 * scale * (mega ** 0.5)))
        max_sz  = max(160, int(280 * scale * (mega ** 0.5)))
        return _render_particles(
            width, height, seed, frame,
            count_per_mega=count / max(0.01, mega), gx=gx, gy=gy,
            speed_min=0.1 * sm, speed_max=0.4 * sm,
            size_min=min_sz, size_max=max_sz,
            rot_min=0, rot_max=0, life_frames=2000,
            color1=c1, color2=c2, alpha=opacity, trail=0,
            shape="circle", glow_radius=glow, glow_gain=1.3,
            bg_color=bg_color, bg_alpha=bg_alpha,
        )


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "S42P_FX_HeartsRain":      S42P_FX_HeartsRain,
    "S42P_FX_Snow":            S42P_FX_Snow,
    "S42P_FX_Starfield":       S42P_FX_Starfield,
    "S42P_FX_BokehOrbs":       S42P_FX_BokehOrbs,
    "S42P_FX_Fireflies":       S42P_FX_Fireflies,
    "S42P_FX_Confetti":        S42P_FX_Confetti,
    "S42P_FX_MatrixRain":      S42P_FX_MatrixRain,
    "S42P_FX_NeonScanlines":   S42P_FX_NeonScanlines,
    "S42P_FX_RisingBubbles":   S42P_FX_RisingBubbles,
    "S42P_FX_LightningStorm":  S42P_FX_LightningStorm,
    "S42P_FX_GeometricStorm":  S42P_FX_GeometricStorm,
    "S42P_FX_MeteorShower":    S42P_FX_MeteorShower,
    "S42P_FX_PetalRain":       S42P_FX_PetalRain,
    "S42P_FX_GoldDust":        S42P_FX_GoldDust,
    "S42P_FX_CyberGrid":       S42P_FX_CyberGrid,
    "S42P_FX_PlasmaOrbs":      S42P_FX_PlasmaOrbs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42P_FX_HeartsRain":      "S42P FX • Hearts Rain",
    "S42P_FX_Snow":            "S42P FX • Snow",
    "S42P_FX_Starfield":       "S42P FX • Starfield",
    "S42P_FX_BokehOrbs":       "S42P FX • Bokeh Orbs",
    "S42P_FX_Fireflies":       "S42P FX • Fireflies",
    "S42P_FX_Confetti":        "S42P FX • Confetti",
    "S42P_FX_MatrixRain":      "S42P FX • Matrix Rain",
    "S42P_FX_NeonScanlines":   "S42P FX • Neon Scanlines",
    "S42P_FX_RisingBubbles":   "S42P FX • Rising Bubbles",
    "S42P_FX_LightningStorm":  "S42P FX • Lightning Storm",
    "S42P_FX_GeometricStorm":  "S42P FX • Geometric Storm",
    "S42P_FX_MeteorShower":    "S42P FX • Meteor Shower",
    "S42P_FX_PetalRain":       "S42P FX • Petal Rain",
    "S42P_FX_GoldDust":        "S42P FX • Gold Dust",
    "S42P_FX_CyberGrid":       "S42P FX • Cyber Grid",
    "S42P_FX_PlasmaOrbs":      "S42P FX • Plasma Orbs",
}
