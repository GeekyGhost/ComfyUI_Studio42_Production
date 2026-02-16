"""
S42 Production Suite — Background Remover v1.0
================================================
Professional background removal with AI and traditional methods.

Ported from Studio42 24oiduts BackgroundRemoverEnhanced.
Renamed to S42P conventions. Animation-related code removed
(handled by S42P Keyframe Animator).

METHODS (in order of quality):
  BiRefNet General/Portrait/Massive — state-of-the-art AI (2024-2025)
  BEN2 Base                         — confidence-guided matting
  U2Net / ISNet / SILUETA           — classic reliable AI models
  Chroma Key Professional           — multi-pass green/blue screen
  Luma Key Professional             — brightness-based removal
  Difference Key                    — background subtraction
  Hybrid AI+Traditional             — AI + chroma combined

OUTPUTS:
  result_rgb   — composited on black background (IMAGE)
  result_rgba  — full RGBA with alpha channel (IMAGE)
  result_mask  — binary/soft alpha mask (MASK)
  info         — processing summary string

OPTIONAL INPUTS:
  guidance_mask    — white=keep, black=remove hints
  depth_map        — closer objects = foreground
  reference_bg     — clean background plate (difference key)

Python 3.12 | ComfyUI Portable
"""

from __future__ import annotations

import os
import time
import logging
import numpy as np
import torch
from PIL import Image
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

try:
    import folder_paths
    _MODELS_DIR = os.path.join(folder_paths.models_dir, "rembg")
except Exception:
    _MODELS_DIR = os.path.join(os.path.expanduser("~"), ".u2net")

os.makedirs(_MODELS_DIR, exist_ok=True)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning("[S42P BG Remover] OpenCV not available — some methods disabled")

try:
    from rembg import remove, new_session
    _HAS_REMBG = True
    logger.info("[S42P BG Remover] ✅ rembg available")
except ImportError:
    _HAS_REMBG = False
    logger.warning("[S42P BG Remover] rembg not available — AI methods disabled")

try:
    from transformers import AutoModelForImageSegmentation
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    from ben2 import BEN_Base
    _HAS_BEN2 = True
except ImportError:
    _HAS_BEN2 = False

# ── Session cache ─────────────────────────────────────────────────────────

_SESSION_CACHE = {}
_CLEANUP_TIME  = 0
CLEANUP_INTERVAL = 300


def _get_session(model_name: str, use_gpu: bool, precision: str = "fp16"):
    global _SESSION_CACHE, _CLEANUP_TIME
    key = f"{model_name}_{use_gpu}_{precision}"
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]
    try:
        os.environ["U2NET_HOME"] = _MODELS_DIR
        if model_name.startswith("birefnet") and _HAS_TRANSFORMERS:
            sess = _make_transformers_session(model_name, use_gpu, precision)
            if sess:
                _SESSION_CACHE[key] = sess
                return sess
        if not _HAS_REMBG:
            return None
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if use_gpu else ["CPUExecutionProvider"])
        sess = new_session(model_name, providers=providers)
        _SESSION_CACHE[key] = sess
        logger.info(f"[S42P BG Remover] Session cached: {model_name}")
        return sess
    except Exception as e:
        logger.error(f"[S42P BG Remover] Session failed {model_name}: {e}")
        return None


def _make_transformers_session(model_name, use_gpu, precision):
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True)
        if use_gpu and torch.cuda.is_available():
            model = model.cuda()
            if precision == "fp16":
                model = model.half()
        model.eval()
        return model
    except Exception as e:
        logger.warning(f"[S42P BG Remover] Transformers BiRefNet failed: {e}")
        return None


# ── Enums & Config ────────────────────────────────────────────────────────

class Method(Enum):
    BIREFNET_GENERAL       = "birefnet-general"
    BIREFNET_PORTRAIT      = "birefnet-portrait"
    BIREFNET_MASSIVE       = "birefnet-massive"
    BIREFNET_GENERAL_LITE  = "birefnet-general-lite"
    BIREFNET_DIS           = "birefnet-dis"
    BIREFNET_HRSOD         = "birefnet-hrsod"
    BIREFNET_COD           = "birefnet-cod"
    BEN2_BASE              = "ben2-base"
    U2NET                  = "u2net"
    U2NET_HUMAN            = "u2net_human_seg"
    U2NET_CLOTH            = "u2net_cloth_seg"
    U2NETP                 = "u2netp"
    SILUETA                = "silueta"
    ISNET_GENERAL          = "isnet-general-use"
    ISNET_ANIME            = "isnet-anime"
    CHROMA_KEY             = "chroma_key_professional"
    LUMA_KEY               = "luma_key_professional"
    DIFFERENCE_KEY         = "difference_key"
    HYBRID                 = "hybrid_ai_traditional"


@dataclass
class BgConfig:
    method: Method = Method.BIREFNET_GENERAL
    use_gpu: bool = True
    model_precision: str = "fp16"
    processing_resolution: int = 1024
    maintain_aspect_ratio: bool = True
    # Mask / depth
    use_guidance_mask: bool = True
    guidance_mask_strength: float = 0.7
    use_depth_guidance: bool = True
    depth_threshold: float = 0.5
    depth_falloff: float = 0.2
    combine_depth_with_ai: bool = True
    # Chroma key
    chroma_color: str = "green"
    primary_threshold: float = 0.12
    secondary_threshold: float = 0.03
    saturation_min: float = 0.4
    brightness_min: float = 0.15
    brightness_max: float = 0.95
    multi_pass_keying: bool = True
    spill_suppression: float = 0.85
    spill_preserve_luminance: bool = True
    edge_softness: float = 1.5
    core_matte_choke: float = 0.0
    # Luma key
    luma_key_mode: str = "shadows_highlights"
    shadow_threshold: float = 0.08
    highlight_threshold: float = 0.92
    luma_softness: float = 0.05
    # Difference key
    difference_threshold: float = 0.08
    # Color space
    color_space_mode: str = "hsv"
    # Post-process
    garbage_matte_enabled: bool = False
    remove_small_objects: bool = True
    small_object_threshold: int = 300
    mask_blur: float = 0.8
    edge_detection_enabled: bool = True
    morphological_operations: bool = True
    mask_dilation: int = 1
    mask_erosion: int = 1


# ── Core processor ────────────────────────────────────────────────────────

class _Processor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ref_bg = None

    def process(self, image: Image.Image, cfg: BgConfig,
                 guidance_mask=None, depth_map=None,
                 reference_bg=None) -> Tuple[Image.Image, Image.Image]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if reference_bg is not None:
            self._ref_bg = reference_bg.convert("RGB")

        gm = self._prep_mask(guidance_mask, image.size) if guidance_mask is not None and cfg.use_guidance_mask else None
        dm = self._prep_depth(depth_map, image.size)    if depth_map   is not None and cfg.use_depth_guidance  else None

        try:
            if cfg.method in (Method.BIREFNET_GENERAL, Method.BIREFNET_PORTRAIT,
                              Method.BIREFNET_MASSIVE, Method.BIREFNET_GENERAL_LITE,
                              Method.BIREFNET_DIS, Method.BIREFNET_HRSOD, Method.BIREFNET_COD):
                return self._ai_birefnet(image, cfg, gm, dm)
            elif cfg.method == Method.BEN2_BASE:
                return self._ai_ben2(image, cfg, gm, dm)
            elif cfg.method in (Method.U2NET, Method.U2NET_HUMAN, Method.U2NET_CLOTH,
                                Method.U2NETP, Method.SILUETA,
                                Method.ISNET_GENERAL, Method.ISNET_ANIME):
                return self._ai_rembg(image, cfg, gm, dm)
            elif cfg.method == Method.CHROMA_KEY:
                return self._chroma_key(image, cfg, gm, dm)
            elif cfg.method == Method.LUMA_KEY:
                return self._luma_key(image, cfg, gm, dm)
            elif cfg.method == Method.DIFFERENCE_KEY:
                return self._difference_key(image, cfg, gm, dm)
            elif cfg.method == Method.HYBRID:
                return self._hybrid(image, cfg, gm, dm)
        except Exception as e:
            logger.error(f"[S42P BG Remover] Processing failed: {e}")
        # Fallback — return original with white mask
        mask = Image.new("L", image.size, 255)
        rgba = image.convert("RGBA")
        return rgba, mask

    # ── Helpers ───────────────────────────────────────────────────────────

    def _prep_mask(self, m, size):
        if m.mode != "L": m = m.convert("L")
        if m.size != size: m = m.resize(size, Image.LANCZOS)
        return np.array(m, dtype=np.float32) / 255.0

    def _prep_depth(self, d, size):
        if d.mode != "L": d = d.convert("L")
        if d.size != size: d = d.resize(size, Image.LANCZOS)
        return np.array(d, dtype=np.float32) / 255.0

    def _resize_aspect(self, image, max_px):
        w, h = image.size
        if w >= h:
            return image.resize((max_px, int(h*max_px/w)), Image.LANCZOS)
        return image.resize((int(w*max_px/h), max_px), Image.LANCZOS)

    def _post_process(self, mask: Image.Image, cfg: BgConfig) -> Image.Image:
        if not _HAS_CV2:
            if cfg.mask_blur > 0:
                mask = mask.filter(Image.Image.BLUR if hasattr(Image.Image, "BLUR") else mask)
            return mask

        arr = np.array(mask)
        if cfg.garbage_matte_enabled:
            contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                gm = np.zeros_like(arr)
                cv2.fillPoly(gm, [largest], 255)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
                gm = cv2.dilate(gm, k, iterations=1)
                arr = cv2.bitwise_and(arr, gm)

        if cfg.remove_small_objects and cfg.small_object_threshold > 0:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(arr, connectivity=8)
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] < cfg.small_object_threshold:
                    arr[labels == i] = 0

        if cfg.morphological_operations:
            if cfg.mask_dilation > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                    (cfg.mask_dilation*2+1, cfg.mask_dilation*2+1))
                arr = cv2.dilate(arr, k, iterations=1)
            if cfg.mask_erosion > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                    (cfg.mask_erosion*2+1, cfg.mask_erosion*2+1))
                arr = cv2.erode(arr, k, iterations=1)

        if cfg.edge_softness > 0:
            ks = int(cfg.edge_softness * 2) | 1
            arr = cv2.GaussianBlur(arr, (ks,ks), cfg.edge_softness/3.0)
        if cfg.mask_blur > 0:
            arr = cv2.GaussianBlur(arr, (0,0), cfg.mask_blur)

        return Image.fromarray(arr.astype(np.uint8), "L")

    def _apply_gm_dm(self, mask_arr, gm, dm, cfg):
        if gm is not None:
            mask_arr = mask_arr * gm * cfg.guidance_mask_strength + \
                       mask_arr * (1 - cfg.guidance_mask_strength)
        if dm is not None and cfg.use_depth_guidance:
            dw = np.clip((dm - cfg.depth_threshold) / max(0.01, cfg.depth_falloff), 0, 1)
            dc = np.abs(dm - 0.5) * 2
            mask_arr = mask_arr*(1-dc*0.3) + (mask_arr*dw)*(dc*0.3)
        return np.clip(mask_arr, 0, 1)

    def _pack_rgba(self, image, mask_pil):
        rgba = Image.new("RGBA", image.size, (0,0,0,0))
        rgba.paste(image, (0,0))
        rgba.putalpha(mask_pil)
        return rgba, mask_pil

    # ── AI methods ────────────────────────────────────────────────────────

    def _ai_birefnet(self, image, cfg, gm, dm):
        sess = _get_session(cfg.method.value, cfg.use_gpu, cfg.model_precision)
        if sess is None:
            logger.warning("[S42P BG Remover] BiRefNet unavailable → chroma key fallback")
            return self._chroma_key(image, cfg, gm, dm)

        orig = image.size
        proc = image
        if cfg.processing_resolution > 0 and max(orig) > cfg.processing_resolution:
            proc = self._resize_aspect(image, cfg.processing_resolution) if cfg.maintain_aspect_ratio \
                   else image.resize((cfg.processing_resolution,)*2, Image.LANCZOS)

        if hasattr(sess, "forward"):  # transformers model
            result = self._transformers_infer(proc, sess, cfg)
        else:
            result = remove(proc, session=sess)

        if proc.size != orig:
            result = result.resize(orig, Image.LANCZOS)

        result = result.convert("RGBA")
        mask   = result.split()[3]
        arr    = np.array(mask, dtype=np.float32) / 255.0
        arr    = self._apply_gm_dm(arr, gm, dm, cfg)
        mask   = Image.fromarray((arr*255).astype(np.uint8), "L")
        mask   = self._post_process(mask, cfg)
        return self._pack_rgba(image, mask)

    def _transformers_infer(self, image, model, cfg):
        try:
            from torchvision import transforms
            tf = transforms.Compose([
                transforms.Resize((1024,1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
            t = tf(image).unsqueeze(0)
            if cfg.use_gpu and torch.cuda.is_available():
                t = t.cuda()
                if cfg.model_precision == "fp16": t = t.half()
            with torch.no_grad():
                preds = model(t)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()
            mask = transforms.ToPILImage()(pred).resize(image.size)
            result = image.convert("RGBA")
            result.putalpha(mask)
            return result
        except Exception as e:
            logger.error(f"[S42P BG Remover] Transformers infer failed: {e}")
            return image.convert("RGBA")

    def _ai_ben2(self, image, cfg, gm, dm):
        if _HAS_REMBG:
            try:
                sess = _get_session("ben2-base", cfg.use_gpu, cfg.model_precision)
                if sess:
                    result = remove(image, session=sess)
                    result = result.convert("RGBA")
                    mask   = result.split()[3]
                    arr    = np.array(mask, dtype=np.float32) / 255.0
                    arr    = self._apply_gm_dm(arr, gm, dm, cfg)
                    mask   = Image.fromarray((arr*255).astype(np.uint8), "L")
                    mask   = self._post_process(mask, cfg)
                    return self._pack_rgba(image, mask)
            except Exception: pass
        if _HAS_BEN2:
            try:
                dev = torch.device("cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu")
                m   = BEN_Base.from_pretrained("PramaLLC/BEN2")
                m.to(dev).eval()
                fg = m.inference(image, refine_foreground=True)
                fg = fg.convert("RGBA")
                mask = fg.split()[3]
                arr  = np.array(mask, dtype=np.float32) / 255.0
                arr  = self._apply_gm_dm(arr, gm, dm, cfg)
                mask = Image.fromarray((arr*255).astype(np.uint8), "L")
                mask = self._post_process(mask, cfg)
                return self._pack_rgba(image, mask)
            except Exception as e:
                logger.error(f"[S42P BG Remover] BEN2 failed: {e}")
        logger.warning("[S42P BG Remover] BEN2 unavailable → BiRefNet fallback")
        return self._ai_birefnet(image, BgConfig(method=Method.BIREFNET_GENERAL,
                                                  use_gpu=cfg.use_gpu,
                                                  model_precision=cfg.model_precision), gm, dm)

    def _ai_rembg(self, image, cfg, gm, dm):
        if not _HAS_REMBG:
            return self._chroma_key(image, cfg, gm, dm)
        orig = image.size
        proc = image
        if cfg.processing_resolution > 0 and max(orig) > cfg.processing_resolution:
            proc = self._resize_aspect(image, cfg.processing_resolution) if cfg.maintain_aspect_ratio \
                   else image.resize((cfg.processing_resolution,)*2, Image.LANCZOS)
        sess = _get_session(cfg.method.value, cfg.use_gpu, cfg.model_precision)
        if sess is None:
            return self._chroma_key(image, cfg, gm, dm)
        result = remove(proc, session=sess)
        if proc.size != orig:
            result = result.resize(orig, Image.LANCZOS)
        result = result.convert("RGBA")
        mask   = result.split()[3]
        arr    = np.array(mask, dtype=np.float32) / 255.0
        arr    = self._apply_gm_dm(arr, gm, dm, cfg)
        mask   = Image.fromarray((arr*255).astype(np.uint8), "L")
        mask   = self._post_process(mask, cfg)
        return self._pack_rgba(image, mask)

    # ── Traditional methods ────────────────────────────────────────────────

    def _chroma_key(self, image, cfg, gm, dm):
        if not _HAS_CV2:
            logger.warning("[S42P BG Remover] OpenCV required for chroma key")
            mask = Image.new("L", image.size, 255)
            return self._pack_rgba(image, mask)

        img_f = np.array(image, dtype=np.float32) / 255.0

        if cfg.color_space_mode == "hsv":
            working = cv2.cvtColor(img_f, cv2.COLOR_RGB2HSV)
        elif cfg.color_space_mode == "lab":
            working = cv2.cvtColor(img_f, cv2.COLOR_RGB2LAB)
        else:
            working = img_f.copy()

        fg_mask = self._primary_chroma(working, cfg)

        if cfg.multi_pass_keying:
            edge_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
            edges = np.abs(cv2.filter2D(fg_mask, -1, edge_kernel)) > 0.1
            secondary = self._secondary_chroma(working, cfg, edges)
            fg_mask = np.maximum(fg_mask, secondary)

        despilled = self._spill_suppress(img_f, fg_mask, cfg)

        if cfg.edge_detection_enabled:
            fg_mask = self._edge_refine(despilled, fg_mask, cfg)
        if dm is not None:
            fg_mask = self._depth_blend(fg_mask, dm, cfg)
        if gm is not None:
            fg_mask = fg_mask*gm*cfg.guidance_mask_strength + fg_mask*(1-cfg.guidance_mask_strength)
        if abs(cfg.core_matte_choke) > 0.001:
            fg_mask = self._choke(fg_mask, cfg.core_matte_choke)

        mask_pil = Image.fromarray((fg_mask*255).astype(np.uint8), "L")
        mask_pil = self._post_process(mask_pil, cfg)
        desp_pil = Image.fromarray((np.clip(despilled,0,1)*255).astype(np.uint8), "RGB")
        rgba = Image.new("RGBA", image.size, (0,0,0,0))
        rgba.paste(desp_pil, (0,0))
        rgba.putalpha(mask_pil)
        return rgba, mask_pil

    def _primary_chroma(self, working, cfg):
        if cfg.color_space_mode != "hsv":
            img_f = working if cfg.color_space_mode == "rgb" else \
                    cv2.cvtColor(working, cv2.COLOR_LAB2RGB)
            r, g, b = img_f[:,:,0], img_f[:,:,1], img_f[:,:,2]
            ch = {"green":g,"blue":b,"red":r}.get(cfg.chroma_color, g)
            ch2_a = {"green":r,"blue":r,"red":g}.get(cfg.chroma_color, r)
            ch2_b = {"green":b,"blue":g,"red":b}.get(cfg.chroma_color, b)
            bg = np.clip((ch - np.maximum(ch2_a,ch2_b))/max(0.001,cfg.primary_threshold), 0, 1)
            return 1.0 - bg

        hue, sat, val = working[:,:,0]*360, working[:,:,1], working[:,:,2]
        ranges = {"green":(70,170),"blue":(190,270),"red":(340,20),
                  "cyan":(160,210),"magenta":(275,345),"yellow":(40,80)}
        lo, hi = ranges.get(cfg.chroma_color, (70,170))
        if cfg.chroma_color == "red":
            hue_dist = np.minimum(np.abs(hue-lo), np.abs(hue-hi))
        else:
            ctr = (lo+hi)/2; hue_dist = np.abs(hue-ctr)
        rng = (hi-lo)/2
        hue_match = np.exp(-(hue_dist / max(0.001, rng*cfg.primary_threshold))**2)
        sat_match = np.clip((sat-cfg.saturation_min)/0.3, 0, 1)
        bg = hue_match * sat_match
        return np.clip(1.0-bg, 0, 1)

    def _secondary_chroma(self, working, cfg, edge_areas):
        arr = self._primary_chroma(working, cfg)
        return arr * edge_areas.astype(np.float32)

    def _spill_suppress(self, img_f, mask, cfg):
        if cfg.spill_suppression <= 0: return img_f
        out = img_f.copy()
        ch_map = {"green":1,"blue":2,"red":0}
        sc = ch_map.get(cfg.chroma_color, 1)
        oc = [i for i in range(3) if i != sc]
        spill = np.maximum(0, out[:,:,sc] - np.maximum(out[:,:,oc[0]], out[:,:,oc[1]]))
        fg_spill = spill * (1.0-mask) * cfg.spill_suppression
        out[:,:,sc] -= fg_spill
        return np.clip(out, 0, 1)

    def _edge_refine(self, img_f, mask, cfg):
        if not _HAS_CV2: return mask
        gray = (np.dot(img_f,[0.299,0.587,0.114])*255).astype(np.uint8)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32)/255.0
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        edges = cv2.dilate(edges, k, iterations=1)
        edges = cv2.GaussianBlur(edges, (5,5), 1.0)
        grad  = cv2.GaussianBlur(mask, (15,15), 5.0)
        return np.clip(mask*(1-edges*0.3) + grad*edges*0.3, 0, 1)

    def _depth_blend(self, mask, dm, cfg):
        df = np.clip((dm-cfg.depth_threshold)/max(0.001,cfg.depth_falloff), 0, 1)
        dc = np.abs(dm-0.5)*2
        return np.clip(mask*(1-dc*0.3) + (mask*df)*(dc*0.3), 0, 1)

    def _choke(self, mask, amount):
        if not _HAS_CV2: return mask
        m8 = (mask*255).astype(np.uint8)
        ks = int(abs(amount)*5)+1
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
        r  = cv2.erode(m8, k) if amount > 0 else cv2.dilate(m8, k)
        return r.astype(np.float32)/255.0

    def _luma_key(self, image, cfg, gm, dm):
        img_f = np.array(image, dtype=np.float32)/255.0
        gray  = np.dot(img_f,[0.299,0.587,0.114])

        if cfg.luma_key_mode == "shadows":
            bg = gray < cfg.shadow_threshold
            if cfg.luma_softness > 0:
                fg = np.clip((gray-cfg.shadow_threshold)/max(0.001,cfg.luma_softness), 0, 1)
            else: fg = (~bg).astype(np.float32)
        elif cfg.luma_key_mode == "highlights":
            bg = gray > cfg.highlight_threshold
            if cfg.luma_softness > 0:
                fg = np.clip((cfg.highlight_threshold-gray)/max(0.001,cfg.luma_softness), 0, 1)
            else: fg = (~bg).astype(np.float32)
        else:
            if cfg.luma_softness > 0:
                sf = np.clip((gray-cfg.shadow_threshold)/max(0.001,cfg.luma_softness), 0, 1)
                hf = np.clip((cfg.highlight_threshold-gray)/max(0.001,cfg.luma_softness), 0, 1)
                fg = np.minimum(sf, hf)
            else:
                fg = ((gray >= cfg.shadow_threshold) & (gray <= cfg.highlight_threshold)).astype(np.float32)

        fg = self._apply_gm_dm(fg, gm, dm, cfg)
        mask = Image.fromarray((fg*255).astype(np.uint8), "L")
        mask = self._post_process(mask, cfg)
        return self._pack_rgba(image, mask)

    def _difference_key(self, image, cfg, gm, dm):
        if self._ref_bg is None:
            logger.warning("[S42P BG Remover] No reference_bg for difference key → chroma fallback")
            return self._chroma_key(image, cfg, gm, dm)
        ref = self._ref_bg.resize(image.size, Image.LANCZOS) if self._ref_bg.size != image.size else self._ref_bg
        cur = np.array(image, dtype=np.float32)/255.0
        bak = np.array(ref,   dtype=np.float32)/255.0
        diff = np.sqrt(np.sum((cur-bak)**2, axis=2))
        fg = np.clip((diff-cfg.difference_threshold+0.1)/0.1, 0, 1)
        if _HAS_CV2:
            m8 = (fg*255).astype(np.uint8)
            k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
            m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN,  k)
            fg = m8.astype(np.float32)/255.0
        fg = self._apply_gm_dm(fg, gm, dm, cfg)
        mask = Image.fromarray((fg*255).astype(np.uint8), "L")
        mask = self._post_process(mask, cfg)
        return self._pack_rgba(image, mask)

    def _hybrid(self, image, cfg, gm, dm):
        ai_cfg = BgConfig(method=Method.BIREFNET_GENERAL,
                          use_gpu=cfg.use_gpu, model_precision=cfg.model_precision)
        try:
            _, ai_mask = self._ai_birefnet(image, ai_cfg, None, None)
        except Exception:
            _, ai_mask = self._luma_key(image, cfg, None, None)

        _, trad_mask = self._chroma_key(image, cfg, gm, dm)

        ai_arr   = np.array(ai_mask,   dtype=np.float32)/255.0
        trad_arr = np.array(trad_mask, dtype=np.float32)/255.0
        conf = np.abs(ai_arr-0.5)*2
        final = ai_arr*conf + trad_arr*(1-conf)
        mask  = Image.fromarray((np.clip(final,0,1)*255).astype(np.uint8), "L")

        rgba = image.convert("RGBA")
        rgba.putalpha(mask)
        return rgba, mask


# ── Node ──────────────────────────────────────────────────────────────────

class S42PBackgroundRemover:
    """S42P Background Remover — Professional multi-method background removal.

    AI methods (BiRefNet, BEN2, U2Net) require rembg — install with:
      pip install rembg onnxruntime
    Traditional methods (Chroma Key, Luma Key) need only OpenCV.
    """

    _processor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "method": ([m.value for m in Method], {"default": Method.BIREFNET_GENERAL.value}),
            },
            "optional": {
                "guidance_mask":      ("MASK",),
                "depth_map":          ("IMAGE",),
                "reference_background": ("IMAGE",),
                "processing_resolution": ("INT",   {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "use_gpu":            ("BOOLEAN",  {"default": True}),
                "model_precision":    (["fp16","fp32"], {"default": "fp16"}),
                "use_guidance_mask":  ("BOOLEAN",  {"default": True}),
                "guidance_mask_strength": ("FLOAT",{"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "use_depth_guidance": ("BOOLEAN",  {"default": True}),
                "depth_threshold":    ("FLOAT",    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "depth_falloff":      ("FLOAT",    {"default": 0.2, "min": 0.05,"max": 0.5, "step": 0.02}),
                # Chroma key
                "chroma_color":       (["green","blue","red","cyan","magenta","yellow"], {"default": "green"}),
                "primary_threshold":  ("FLOAT",    {"default": 0.12, "min": 0.01, "max": 0.5,  "step": 0.01}),
                "secondary_threshold":("FLOAT",    {"default": 0.03, "min": 0.01, "max": 0.3,  "step": 0.01}),
                "saturation_threshold":("FLOAT",   {"default": 0.4,  "min": 0.1,  "max": 1.0,  "step": 0.05}),
                "spill_suppression":  ("FLOAT",    {"default": 0.85, "min": 0.0,  "max": 1.0,  "step": 0.05}),
                "edge_softness":      ("FLOAT",    {"default": 1.5,  "min": 0.0,  "max": 10.0, "step": 0.25}),
                "core_matte_choke":   ("FLOAT",    {"default": 0.0,  "min": -0.5, "max": 0.5,  "step": 0.02}),
                # Luma key
                "luma_key_mode":      (["shadows","highlights","shadows_highlights"], {"default": "shadows_highlights"}),
                "shadow_threshold":   ("FLOAT",    {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.02}),
                "highlight_threshold":("FLOAT",    {"default": 0.92, "min": 0.5, "max": 1.0, "step": 0.02}),
                "luma_softness":      ("FLOAT",    {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                # Difference key
                "difference_threshold":("FLOAT",   {"default": 0.08, "min": 0.01,"max": 0.5, "step": 0.01}),
                # Post
                "remove_small_objects":("BOOLEAN", {"default": True}),
                "small_object_threshold":("INT",   {"default": 300,  "min": 50, "max": 2000, "step": 50}),
                "mask_blur":          ("FLOAT",    {"default": 0.8,  "min": 0.0, "max": 5.0, "step": 0.1}),
                "color_space_mode":   (["hsv","lab","rgb"], {"default": "hsv"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES  = ("result_rgb", "result_rgba", "result_mask", "info")
    FUNCTION      = "remove_background"
    CATEGORY      = "S42 Production Suite"

    def remove_background(self, images, method, **kw):
        if self._processor is None:
            S42PBackgroundRemover._processor = _Processor()

        cfg = BgConfig(
            method=Method(method),
            use_gpu=kw.get("use_gpu", True),
            model_precision=kw.get("model_precision","fp16"),
            processing_resolution=kw.get("processing_resolution",1024),
            use_guidance_mask=kw.get("use_guidance_mask",True),
            guidance_mask_strength=kw.get("guidance_mask_strength",0.7),
            use_depth_guidance=kw.get("use_depth_guidance",True),
            depth_threshold=kw.get("depth_threshold",0.5),
            depth_falloff=kw.get("depth_falloff",0.2),
            chroma_color=kw.get("chroma_color","green"),
            primary_threshold=kw.get("primary_threshold",0.12),
            secondary_threshold=kw.get("secondary_threshold",0.03),
            saturation_min=kw.get("saturation_threshold",0.4),
            spill_suppression=kw.get("spill_suppression",0.85),
            edge_softness=kw.get("edge_softness",1.5),
            core_matte_choke=kw.get("core_matte_choke",0.0),
            luma_key_mode=kw.get("luma_key_mode","shadows_highlights"),
            shadow_threshold=kw.get("shadow_threshold",0.08),
            highlight_threshold=kw.get("highlight_threshold",0.92),
            luma_softness=kw.get("luma_softness",0.05),
            difference_threshold=kw.get("difference_threshold",0.08),
            remove_small_objects=kw.get("remove_small_objects",True),
            small_object_threshold=kw.get("small_object_threshold",300),
            mask_blur=kw.get("mask_blur",0.8),
            color_space_mode=kw.get("color_space_mode","hsv"),
        )

        B = images.shape[0]
        ref_pil = None
        if kw.get("reference_background") is not None:
            ref_pil = self._t2pil(kw["reference_background"][0])

        gm_tensors = kw.get("guidance_mask")
        dm_tensors = kw.get("depth_map")

        results_rgb, results_rgba, masks, infos = [], [], [], []
        t0 = time.time()

        for i in range(B):
            img_pil = self._t2pil(images[i])

            gm_pil = None
            if gm_tensors is not None:
                gm_t = gm_tensors[i % gm_tensors.shape[0]] if gm_tensors.ndim == 3 else gm_tensors
                gm_pil = self._mask2pil(gm_t)

            dm_pil = None
            if dm_tensors is not None:
                dm_t = dm_tensors[i % dm_tensors.shape[0]] if dm_tensors.ndim == 4 else dm_tensors
                dm_pil = self._t2pil(dm_t).convert("L")

            rgba, mask = self._processor.process(img_pil, cfg, gm_pil, dm_pil, ref_pil)

            # RGB composite on black
            rgb = Image.new("RGB", rgba.size, (0,0,0))
            if rgba.mode == "RGBA":
                rgb.paste(rgba, mask=rgba.split()[3])
            else:
                rgb = rgba.convert("RGB")

            results_rgb.append(self._pil2t(rgb))
            results_rgba.append(self._pil2t_rgba(rgba))
            masks.append(self._mask2t(mask))

        elapsed = time.time()-t0
        info = (f"S42P BG Remover: {B} img, method={method}, "
                f"avg={elapsed/max(1,B):.2f}s/img")
        print(f"[S42P BG Remover] ✅ {info}")

        rgb_batch  = torch.cat(results_rgb,  dim=0)
        rgba_batch = torch.cat(results_rgba, dim=0)
        mask_batch = torch.stack(masks, dim=0)
        return (rgb_batch, rgba_batch, mask_batch, info)

    # ── Tensor helpers ────────────────────────────────────────────────────

    def _t2pil(self, t: torch.Tensor) -> Image.Image:
        arr = (t.cpu().numpy()*255).astype(np.uint8)
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        if arr.shape[-1] == 4: arr = arr[:,:,:3]
        return Image.fromarray(arr, "RGB")

    def _mask2pil(self, t: torch.Tensor) -> Image.Image:
        return Image.fromarray((t.cpu().numpy()*255).astype(np.uint8), "L")

    def _pil2t(self, p: Image.Image) -> torch.Tensor:
        p = p.convert("RGB")
        return torch.from_numpy(np.array(p).astype(np.float32)/255.0).unsqueeze(0)

    def _pil2t_rgba(self, p: Image.Image) -> torch.Tensor:
        p = p.convert("RGBA")
        return torch.from_numpy(np.array(p).astype(np.float32)/255.0).unsqueeze(0)

    def _mask2t(self, p: Image.Image) -> torch.Tensor:
        p = p.convert("L")
        return torch.from_numpy(np.array(p).astype(np.float32)/255.0)


NODE_CLASS_MAPPINGS        = {"S42PBackgroundRemover": S42PBackgroundRemover}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PBackgroundRemover": "S42P Background Remover 🎭"}
