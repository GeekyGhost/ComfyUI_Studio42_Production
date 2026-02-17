# S42 Production Suite Procedural FX Pack (10 animated backgrounds)
# License: MIT
# Author: Studio42 (Willie G)
# Deps: Pillow>=10, numpy>=1.24 (ComfyUI default)
# Notes:
# - Transparent vs solid background supported (bg_transparent + bg_r/g/b)
# - User overrides for colors, speed multiplier, direction angle/reverse
# - Batch-aware animation; deterministic via seed

import math, random
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    import torch
except Exception:
    torch = None

# ----------------- Common helpers -----------------

def _rng(seed:int):
    r = random.Random()
    r.seed(seed & 0xFFFFFFFF)
    return r

def _hash(*parts:int)->int:
    h = 2166136261
    for p in parts:
        h ^= (int(p) & 0xFFFFFFFF)
        h = (h * 16777619) & 0xFFFFFFFF
    return h

def _safe_font(px:int, font_path:Optional[str]=None):
    if font_path:
        try:
            return ImageFont.truetype(font_path, px)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

def _to_tensor(img: Image.Image):
    """Preserve channels. If image has alpha, keep RGBA (4ch); else RGB (3ch)."""
    # Normalize to a supported mode while preserving alpha if present
    if img.mode not in ("RGB", "RGBA"):
        # Prefer RGBA so we can carry alpha through pipelines when available
        img = img.convert("RGBA")
    arr = np.array(img).astype(np.float32) / 255.0
    if torch is not None:
        return torch.from_numpy(arr)
    return arr

# --- glow + background compositing that honors transparency ---

def _compose_glow(fg_rgba: Image.Image, glow_radius: float, glow_gain: float,
                  bg_color: Tuple[int,int,int], bg_alpha: int) -> Image.Image:
    """
    Composites blurred glow under the crisp foreground sprites.
    Honors transparent backgrounds by using bg_alpha.
    """
    out = Image.new("RGBA", fg_rgba.size, (*bg_color, int(bg_alpha)))

    if glow_radius > 0 and glow_gain > 0:
        glow = fg_rgba.filter(ImageFilter.GaussianBlur(glow_radius))
        if glow_gain != 1.0:
            r, g, b, a = glow.split()
            r = r.point(lambda v: min(int(v * glow_gain), 255))
            g = g.point(lambda v: min(int(v * glow_gain), 255))
            b = b.point(lambda v: min(int(v * glow_gain), 255))
            glow = Image.merge("RGBA", (r, g, b, a))
        out.alpha_composite(glow)

    out.alpha_composite(fg_rgba)
    return out

# ---------- Parameters & overrides ----------

@dataclass
class FXParams:
    # spawn / motion
    count_per_mega: float = 1200.0   # particles per 1MP of canvas
    gravity: Tuple[float,float] = (0.0, 1.0) # direction (x,y) pixels/frame (unit-ish; combined with per-particle speed)
    speed_min: float = 1.0
    speed_max: float = 4.0
    size_min: float = 6.0
    size_max: float = 18.0
    rot_min_deg: float = 0.0
    rot_max_deg: float = 0.0
    life_frames: int = 240          # wrap-around cycle

    # rendering
    color1: Tuple[int,int,int] = (0,255,0)
    color2: Tuple[int,int,int] = (0,180,0)
    alpha: int = 220
    bg_alpha: int = 255
    bg_color: Tuple[int,int,int] = (0,0,0)  # background RGB when bg_alpha>0

    # look
    trail: int = 0                   # draw N faded steps behind
    glow_radius: float = 0.0
    glow_gain: float = 0.0

    # text glyph fallback / font
    glyphs: str = ""
    font_path: Optional[str] = None

    # shape mode
    mode: str = "circle"             # circle | heart | star | line | bubble | confetti | text


def _apply_overrides(p: FXParams, overrides: dict):
    # Background controls
    bg_transparent = bool(overrides.get("bg_transparent", False))
    p.bg_alpha = 0 if bg_transparent else 255
    p.bg_color = (
        int(overrides.get("bg_r", p.bg_color[0])),
        int(overrides.get("bg_g", p.bg_color[1])),
        int(overrides.get("bg_b", p.bg_color[2])),
    )

    # Color overrides
    if bool(overrides.get("override_colors", False)):
        p.color1 = (
            int(overrides.get("col1_r", p.color1[0])),
            int(overrides.get("col1_g", p.color1[1])),
            int(overrides.get("col1_b", p.color1[2])),
        )
        p.color2 = (
            int(overrides.get("col2_r", p.color2[0])),
            int(overrides.get("col2_g", p.color2[1])),
            int(overrides.get("col2_b", p.color2[2])),
        )

    # Direction + speed
    use_custom_dir = bool(overrides.get("use_custom_direction", False))
    angle_deg = float(overrides.get("angle_deg", 90.0))  # 0=right, 90=down
    reverse = bool(overrides.get("reverse", False))
    speed_mult = float(overrides.get("speed_mult", 1.0))

    if use_custom_dir:
        ang = math.radians(angle_deg + (180.0 if reverse else 0.0))
        gx, gy = math.cos(ang), math.sin(ang)
        p.gravity = (gx, gy)
    else:
        gx, gy = p.gravity
        if reverse:
            gx, gy = -gx, -gy
        p.gravity = (gx, gy)

    # Scale per-particle speeds
    p.speed_min *= speed_mult
    p.speed_max *= speed_mult

# ---------- Drawing primitives ----------

def _interp_color(c1,c2,t):
    return (int(c1[0]*(1-t)+c2[0]*t),
            int(c1[1]*(1-t)+c2[1]*t),
            int(c1[2]*(1-t)+c2[2]*t))


def _draw_sprite(draw:ImageDraw.ImageDraw, mode:str, x:float,y:float, sz:float, rot:float, col:Tuple[int,int,int], a:int, font):
    if a<=0: return
    if mode in ("circle","bubble"):
        bbox = [x-sz/2, y-sz/2, x+sz/2, y+sz/2]
        if mode=="bubble":
            # thin ring
            outline = (*col,a)
            draw.ellipse(bbox, outline=outline, width=max(1,int(sz*0.12)))
        else:
            draw.ellipse(bbox, fill=(*col,a))
        return
    if mode=="line":
        # vertical neon line segment
        w = max(1,int(sz*0.18))
        draw.rectangle([x-w/2, y-sz/1.8, x+w/2, y+sz/1.8], fill=(*col,a))
        return
    if mode=="star":
        # 5-point star
        pts=[]
        for i in range(10):
            R = sz/2 if i%2==0 else sz/4
            ang = math.radians(-90 + i*36 + rot)
            pts.append((x+R*math.cos(ang), y+R*math.sin(ang)))
        draw.polygon(pts, fill=(*col,a))
        return
    if mode=="heart":
        # heart via small rotated RGBA sprite
        s = int(max(8, sz))
        spr = Image.new("RGBA",(s,s),(0,0,0,0))
        d2 = ImageDraw.Draw(spr)
        w,h = s,s
        cx,cy = w/2, h*0.45
        r = s*0.22
        # two top circles + triangle bottom
        d2.pieslice([cx-r*2, cy-r, cx, cy+r], 0, 360, fill=(*col,a))
        d2.pieslice([cx, cy-r, cx+r*2, cy+r], 0, 360, fill=(*col,a))
        d2.polygon([(cx-r*2, cy), (cx+r*2, cy), (cx, h)], fill=(*col,a))
        spr = spr.rotate(rot, resample=Image.BICUBIC, expand=True)
        draw.bitmap((x-spr.width/2, y-spr.height/2), spr, fill=None)
        return
    if mode=="confetti":
        # rotated rectangle/triangle mix
        s = max(6, sz)
        spr = Image.new("RGBA",(int(s),int(s)),(0,0,0,0))
        d2 = ImageDraw.Draw(spr)
        if (int(x+y) & 1)==0:
            d2.polygon([(0,s*0.2),(s,s*0.2),(s*0.8,s),(s*0.2,s)], fill=(*col,a))
        else:
            d2.polygon([(s*0.5,0),(s, s),(0,s)], fill=(*col,a))
        spr = spr.rotate(rot, resample=Image.BICUBIC, expand=True)
        draw.bitmap((x-spr.width/2, y-spr.height/2), spr, fill=None)
        return
    # mode=="text" handled in caller with _draw_text


def _draw_text(draw, glyph:str, x:float,y:float, sz:float, col:Tuple[int,int,int], a:int, font):
    if a<=0: return
    # crude centering for monospace/default font
    draw.text((x - sz*0.33, y - sz*0.62), glyph, font=font, fill=(*col,a))

# ---------- Renderer ----------

def _render_fx(width:int, height:int, seed:int, frame:int, params:FXParams)->Image.Image:
    # draw sprites on a transparent foreground
    fg = Image.new("RGBA", (width, height), (0,0,0,0))
    draw = ImageDraw.Draw(fg, "RGBA")

    # particle count proportional to area
    mega = (width*height)/1_000_000.0
    count = max(1, int(params.count_per_mega*mega))

    gx,gy = params.gravity
    font = _safe_font(int(max(10, params.size_max)), params.font_path)

    for i in range(count):
        h = _hash(seed, i, 42)
        rnd = _rng(h)

        life = max(1, int(params.life_frames))
        t = (frame + (h % life)) % life

        sx = rnd.random()*width
        sy = rnd.random()*height

        spd = rnd.uniform(params.speed_min, params.speed_max)
        sz  = rnd.uniform(params.size_min, params.size_max)
        rot = rnd.uniform(params.rot_min_deg, params.rot_max_deg)

        x = (sx + gx * t * spd) % width
        y = (sy + gy * t * spd) % height

        ct = rnd.random()
        col = _interp_color(params.color1, params.color2, ct)

        steps = max(0, int(params.trail))
        for k in range(steps, -1, -1):
            fade = 1.0 if steps==0 else (k/steps)
            aa = int(params.alpha * (fade**1.2))
            yy = (y - gy * k * spd) % height
            xx = (x - gx * k * spd) % width

            if params.mode=="text" and params.glyphs:
                glyph = params.glyphs[ h % len(params.glyphs) ]
                _draw_text(draw, glyph, xx, yy, sz, col, aa, font)
            else:
                _draw_sprite(draw, params.mode, xx, yy, sz, rot, col, aa, font)

    # composite glow + background with transparency honored
    out = _compose_glow(fg, params.glow_radius, params.glow_gain,
                        params.bg_color, params.bg_alpha)
    return out

# ---------- Batch helper (fixed: no duplicate 'frame' arg) ----------

def _make_output(batch_size:int, generator, **kwargs):
    """
    Batch helper that advances frame per index when animate_batch=True.
    Important: remove (pop) frame/animate_batch from kwargs so we don't
    pass 'frame' twice to generator(**kwargs, frame=...).
    """
    imgs = []
    b = max(1, int(batch_size))
    call_kwargs = dict(kwargs)
    base_frame = int(call_kwargs.pop("frame", 0))
    animate = bool(call_kwargs.pop("animate_batch", True))

    for i in range(b):
        f = base_frame + i if animate else base_frame
        img = generator(frame=f, **call_kwargs)
        imgs.append(_to_tensor(img))

    if torch is not None:
        return (torch.stack(imgs, dim=0),)
    else:
        return (np.stack(imgs, axis=0),)

# -------------- Base Node class --------------

class _FXBase:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Studio42/Procedural FX"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height":("INT", {"default": 576,  "min": 64, "max": 4096, "step": 8}),
                "seed":  ("INT", {"default": 12345, "min": 0, "max": 2_147_483_647}),
            },
            "optional":{
                # timeline / batch
                "frame": ("INT", {"default": 0, "min": 0, "max": 10_000_000}),
                "animate_batch": ("BOOLEAN", {"default": True}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 128}),

                # background control
                "bg_transparent": ("BOOLEAN", {"default": False}),
                "bg_r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "bg_g": ("INT", {"default": 0, "min": 0, "max": 255}),
                "bg_b": ("INT", {"default": 0, "min": 0, "max": 255}),

                # color override
                "override_colors": ("BOOLEAN", {"default": False}),
                "col1_r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "col1_g": ("INT", {"default": 255, "min": 0, "max": 255}),
                "col1_b": ("INT", {"default": 0, "min": 0, "max": 255}),
                "col2_r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "col2_g": ("INT", {"default": 180, "min": 0, "max": 255}),
                "col2_b": ("INT", {"default": 0, "min": 0, "max": 255}),

                # direction + speed
                "use_custom_direction": ("BOOLEAN", {"default": False}),
                "angle_deg": ("FLOAT", {"default": 90.0, "min": -360.0, "max": 360.0, "step": 1.0}), # 0=right, 90=down
                "reverse": ("BOOLEAN", {"default": False}),
                "speed_mult": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    def _params(self):
        # override in subclasses
        return FXParams()

    def _render(self, width, height, seed, frame, **overrides):
        p = self._params()
        _apply_overrides(p, overrides)
        return _render_fx(width, height, seed, frame, p)

    def generate(self, width, height, seed, frame=0, animate_batch=True, batch_size=8, **kwargs):
        return _make_output(batch_size, self._render,
                            width=width, height=height, seed=int(seed),
                            frame=int(frame), animate_batch=animate_batch, **kwargs)

# -------------- Preset Nodes (10) --------------

class HeartsRain(_FXBase):
    def _params(self):
        return FXParams(
            count_per_mega=900,
            gravity=(0.0, 1.0),
            speed_min=1.2, speed_max=3.3,
            size_min=10, size_max=28,
            rot_min_deg=-20, rot_max_deg=20,
            life_frames=600,
            color1=(255,70,110), color2=(255,160,190),
            alpha=235, glow_radius=2.5, glow_gain=1.4,
            trail=3,
            mode="heart"
        )

class SnowFlakes(_FXBase):
    def _params(self):
        return FXParams(
            count_per_mega=1200,
            gravity=(0.1, 1.0),
            speed_min=0.6, speed_max=1.5,
            size_min=6, size_max=16,
            rot_min_deg=-10, rot_max_deg=10,
            life_frames=900,
            color1=(240,240,255), color2=(180,200,255),
            alpha=220, glow_radius=1.0, glow_gain=1.1,
            trail=1,
            mode="circle"
        )

class StarField(_FXBase):
    def _params(self):
        return FXParams(
            count_per_mega=800,
            gravity=(0.0, 0.3),
            speed_min=0.3, speed_max=1.0,
            size_min=8, size_max=20,
            rot_min_deg=0, rot_max_deg=360,
            life_frames=800,
            color1=(255,240,200), color2=(180,220,255),
            alpha=230, glow_radius=2.0, glow_gain=1.5,
            trail=0,
            mode="star"
        )

class BokehOrbs(_FXBase):
    def _params(self):
        return FXParams(
            count_per_mega=500,
            gravity=(0.0, 0.2),
            speed_min=0.2, speed_max=0.8,
            size_min=18, size_max=42,
            life_frames=1200,
            color1=(120,255,130), color2=(80,200,255),
            alpha=200, glow_radius=4.0, glow_gain=1.4,
            trail=0,
            mode="circle"
        )

class Fireflies(_FXBase):
    def _params(self):
        return FXParams(
            count_per_mega=650,
            gravity=(0.2, -0.1),
            speed_min=0.5, speed_max=1.8,
            size_min=6, size_max=12,
            life_frames=700,
            color1=(230,255,120), color2=(80,255,140),
            alpha=255, glow_radius=3.0, glow_gain=1.8,
            trail=2,
            mode="circle"
        )

class Confetti(_FXBase):
    def _params(self):
        return FXParams(
            count_per_mega=1800,
            gravity=(0.0, 1.4),
            speed_min=1.0, speed_max=3.0,
            size_min=8, size_max=16,
            rot_min_deg=0, rot_max_deg=360,
            life_frames=500,
            color1=(255,200,0), color2=(0,200,255),
            alpha=230, glow_radius=0.0, glow_gain=0.0,
            trail=1,
            mode="confetti"
        )

class AsciiRain(_FXBase):
    def _params(self):
        return FXParams(
            count_per_mega=1400,
            gravity=(0.0, 1.1),
            speed_min=0.8, speed_max=2.6,
            size_min=14, size_max=22,
            life_frames=700,
            color1=(120,255,130), color2=(50,200,90),
            alpha=235, glow_radius=1.8, glow_gain=1.3,
            trail=6,
            glyphs="01",
            mode="text"
        )

    def _render(self, width, height, seed, frame, **overrides):
        p = self._params()
        _apply_overrides(p, overrides)

        # Foreground layer
        fg = Image.new("RGBA",(width,height),(0,0,0,0))
        draw = ImageDraw.Draw(fg,"RGBA")
        font = _safe_font(int(p.size_max), p.font_path)
        mega = (width*height)/1_000_000.0
        count = max(1,int(p.count_per_mega*mega))
        gx,gy = p.gravity

        for i in range(count):
            h = _hash(seed, i, 7)
            rnd = _rng(h)
            life = max(1, int(p.life_frames))
            t = (frame + (h % life)) % life
            sx,sy = rnd.random()*width, rnd.random()*height
            spd = rnd.uniform(p.speed_min, p.speed_max)
            sz  = rnd.uniform(p.size_min, p.size_max)
            x = (sx + gx * t * spd) % width
            y = (sy + gy * t * spd) % height
            col = _interp_color(p.color1, p.color2, rnd.random())
            steps = max(0,int(p.trail))
            ch = p.glyphs[h % len(p.glyphs)]
            for k in range(steps, -1, -1):
                fade = 1.0 if steps==0 else (k/steps)
                aa = int(p.alpha * (fade**1.1))
                yy = (y - gy * k * spd) % height
                xx = (x - gx * k * spd) % width
                _draw_text(draw, ch, xx, yy, sz, col, aa, font)

        return _compose_glow(fg, p.glow_radius, p.glow_gain, p.bg_color, p.bg_alpha)

class EmojiPop(_FXBase):
    def _params(self):
        # Use text mode; supply a small emoji set (unicode render depends on font)
        return FXParams(
            count_per_mega=600,
            gravity=(0.0, -0.6),
            speed_min=0.5, speed_max=1.2,
            size_min=18, size_max=34,
            life_frames=600,
            color1=(255,255,255), color2=(255,255,255),
            alpha=255, glow_radius=0.0, glow_gain=0.0,
            trail=1,
            glyphs="â¤ï¸âœ¨ðŸŽˆâ­ï¸ðŸ’šðŸ’™ðŸ’›",
            mode="text"
        )

class NeonScanlines(_FXBase):
    def _params(self):
        return FXParams(
            count_per_mega=260,
            gravity=(0.0, 1.8),
            speed_min=1.0, speed_max=2.2,
            size_min=24, size_max=48,
            rot_min_deg=0, rot_max_deg=0,
            life_frames=400,
            color1=(120,255,130), color2=(30,180,255),
            alpha=180, glow_radius=2.0, glow_gain=1.6,
            trail=0,
            mode="line"
        )

class RisingBubbles(_FXBase):
    def _params(self):
        return FXParams(
            count_per_mega=900,
            gravity=(0.0, -1.1),
            speed_min=0.7, speed_max=2.0,
            size_min=10, size_max=28,
            life_frames=900,
            color1=(160,220,255), color2=(120,255,220),
            alpha=220, glow_radius=1.5, glow_gain=1.2,
            trail=2,
            mode="bubble"
        )

# ----------------- Node registration -----------------

NODE_CLASS_MAPPINGS = {
    "HeartsRain": HeartsRain,
    "SnowFlakes": SnowFlakes,
    "StarField": StarField,
    "BokehOrbs": BokehOrbs,
    "Fireflies": Fireflies,
    "Confetti": Confetti,
    "AsciiRain": AsciiRain,
    "EmojiPop": EmojiPop,
    "NeonScanlines": NeonScanlines,
    "RisingBubbles": RisingBubbles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartsRain": "FX â€¢ Hearts Rain",
    "SnowFlakes": "FX â€¢ Snow",
    "StarField": "FX â€¢ Starfield",
    "BokehOrbs": "FX â€¢ Bokeh Orbs",
    "Fireflies": "FX â€¢ Fireflies",
    "Confetti": "FX â€¢ Confetti",
    "AsciiRain": "FX â€¢ ASCII Rain (0/1)",
    "EmojiPop": "FX â€¢ Emoji Pop",
    "NeonScanlines": "FX â€¢ Neon Scanlines",
    "RisingBubbles": "FX â€¢ Rising Bubbles",
}
