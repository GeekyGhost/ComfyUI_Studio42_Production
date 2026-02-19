"""
S42 Production Suite — Songwriter Node v5.0
============================================
AI song generation for AceStep 1.5 Turbo via Ollama.

Uses ComfyUI-Ollama for Ollama connectivity and model options.

CHANGES v5.0:
  - TWO-PASS generation:
      Pass 1: Creative concept generation (genre, mood, theme → musical vision)
      Pass 2: Format conversion (vision → strict AceStep JSON)
    Result: dramatically fewer "model didn't follow format" failures
    and more musically coherent output (model isn't trying to be creative
    AND follow a strict schema simultaneously).
  - Thinking model support: <think>...</think> blocks auto-stripped
  - Vision input for image-inspired songs
  - Iterative refinement via context input/output

Python 3.12 | ComfyUI Portable
"""

from __future__ import annotations

import re
import json
import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ollama import Client
    _OLLAMA_OK = True
except ImportError:
    Client = None
    _OLLAMA_OK = False

logger = logging.getLogger(__name__)
CATEGORY = "S42 Production Suite/Songwriter"

try:
    from PIL import Image as PILImage
    _PIL = True
except ImportError:
    _PIL = False


# ── Constants ──────────────────────────────────────────────────────────────────

VALID_KEYS = [
    "C Major", "C Minor", "C# Major", "C# Minor", "Db Major", "Db Minor",
    "D Major", "D Minor", "D# Major", "D# Minor", "Eb Major", "Eb Minor",
    "E Major", "E Minor", "F Major", "F Minor", "F# Major", "F# Minor",
    "Gb Major", "Gb Minor", "G Major", "G Minor", "G# Major", "G# Minor",
    "Ab Major", "Ab Minor", "A Major", "A Minor", "A# Major", "A# Minor",
    "Bb Major", "Bb Minor", "B Major", "B Minor",
]

BPM_HINTS: Dict[str, Tuple[int, int]] = {
    "ambient": (55, 80),   "drone": (45, 70),    "ballad": (60, 80),
    "blues": (60, 100),    "jazz": (80, 140),     "soul": (70, 110),
    "lo-fi": (70, 90),     "lofi": (70, 90),      "hip-hop": (75, 100),
    "hip hop": (75, 100),  "trap": (70, 100),     "r&b": (70, 100),
    "rnb": (70, 100),      "pop": (90, 130),      "indie": (90, 130),
    "rock": (90, 150),     "punk": (140, 200),    "metal": (130, 200),
    "edm": (120, 140),     "house": (120, 130),   "techno": (130, 150),
    "dubstep": (135, 145), "drum and bass": (160, 180), "dnb": (160, 180),
    "classical": (60, 160), "orchestral": (60, 160),
    "country": (90, 130),  "folk": (80, 120),     "reggae": (60, 90),
    "synthwave": (100, 130), "cinematic": (60, 120),
}

_FALLBACK_MODELS = [
    "qwen2.5:7b", "qwen2.5:14b", "qwen3:8b", "qwen3:14b",
    "mistral:7b", "llama3.1:8b", "llama3.2:3b", "gemma3:9b",
    "llava:13b", "qwen2.5vl:7b", "minicpm-v:latest",
]

_model_cache: List[str] = []
_cached_url: str = ""


# ── Model list ─────────────────────────────────────────────────────────────────

def _fetch_models(base_url: str) -> List[str]:
    global _model_cache, _cached_url
    if _model_cache and _cached_url == base_url:
        return _model_cache
    try:
        client = Client(host=base_url)
        raw    = client.list().get("models", [])
        models = [m.get("name") or m.get("model") for m in raw
                  if m.get("name") or m.get("model")]
        if models:
            _model_cache = models
            _cached_url  = base_url
            return models
    except Exception as e:
        logger.warning(f"[S42P Songwriter] Cannot fetch models: {e}")
    return _FALLBACK_MODELS


# ── Image helpers ──────────────────────────────────────────────────────────────

def _is_vision_model(model: str) -> bool:
    low = model.lower()
    return any(m in low for m in [
        "-vl", "vl:", "llava", "vision", "minicpm-v", "moondream",
        "cogvlm", "bakllava", "internvl", "qvq", "gemma3",
        "pixtral", "qwen2-vl", "qwen2.5vl", "qwen3-vl",
    ])


def _tensor_to_b64_jpeg(image_tensor, max_edge: int = 1024) -> str:
    if not _PIL:
        raise RuntimeError("Pillow required for image input.")
    arr = image_tensor
    if hasattr(arr, "cpu"):   arr = arr.cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4: arr = arr[0]
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    pil = PILImage.fromarray(arr, "RGB")
    w, h = pil.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        pil   = pil.resize((int(w*scale), int(h*scale)), PILImage.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Options helper ─────────────────────────────────────────────────────────────

def _enabled_options(opts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract active options from ComfyUI-Ollama OllamaOptions dict."""
    if not opts:
        return None
    out = {}
    enable_keys_found = False
    for k in opts:
        if k.startswith("enable_") and opts[k]:
            enable_keys_found = True
            real_key = k.replace("enable_", "")
            val = opts.get(real_key)
            if val is not None:
                out[real_key] = val
    if not enable_keys_found and opts:
        # Schema may have changed — log a warning but don't silently fail
        logger.warning("[S42P Songwriter] OllamaOptions dict contains no 'enable_*' keys. "
                       "ComfyUI-Ollama may have updated its schema. Options will be ignored.")
    return out or None


# ── Text parsing ───────────────────────────────────────────────────────────────

def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning models."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()


def _extract_json(text: str) -> Optional[dict]:
    if not text: return None
    # Try fenced JSON block first
    fence = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
    if fence:
        try: return json.loads(fence.group(1))
        except Exception: pass
    # Try raw JSON object
    obj = re.search(r"\{.+\}", text, re.DOTALL)
    if obj:
        try: return json.loads(obj.group(0))
        except Exception: pass
    return None


def _parse_bpm(text: str, genre: str) -> int:
    m = re.search(r"\b(\d{2,3})\s*(?:bpm|BPM)\b", text)
    if m:
        bpm = int(m.group(1))
        if 40 <= bpm <= 220: return bpm
    # Lookup by genre keyword
    low = (text + " " + genre).lower()
    for key, (lo, hi) in BPM_HINTS.items():
        if key in low:
            return (lo + hi) // 2
    return 120


def _parse_key(text: str) -> str:
    for k in VALID_KEYS:
        if k.lower() in text.lower():
            return k
    return "C Major"


def _clean_lyrics(raw: str) -> str:
    lines = []
    for s in re.split(r"\n", raw):
        s = s.strip()
        if not s: lines.append(""); continue
        if re.match(r"^\[(Verse|Chorus|Bridge|Pre-Chorus|Outro|Intro|Hook|Break|End)",
                    s, re.IGNORECASE):
            lines.append(s[1:-1].strip() if s.startswith("[") and s.endswith("]") else s)
        else:
            lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not re.search(r"\[End\]", text, re.IGNORECASE):
        text = text.rstrip() + "\n\n[End]"
    return text.strip()


def _ensure_style_fields(style: str, full_text: str, description: str) -> str:
    if "[Tempo:" not in style:
        bpm = _parse_bpm(style + " " + full_text, description)
        style = style.rstrip() + f"\n[Tempo: {bpm} bpm]"
    if "[Key:" not in style:
        key = _parse_key(style + " " + full_text)
        style = style.rstrip() + f"\n[Key: {key}]"
    return style.strip()


# ── TWO-PASS PROMPTS ───────────────────────────────────────────────────────────

def _build_pass1_prompt(genre: str, mood: str, theme: str, tone: str,
                         has_image: bool) -> str:
    """
    Pass 1: Creative concept generation.
    The model is free to be creative here — no strict JSON schema.
    Just describe the song in natural language.
    """
    img_note = "\nAn image has been provided. Let it directly inspire the musical vision.\n" if has_image else ""
    tone_map = {
        "Neutral": "balanced and versatile",
        "Energetic": "high-energy and punchy",
        "Melancholy": "introspective and emotionally heavy",
        "Cinematic": "sweeping and narrative-driven",
        "Aggressive": "raw and intense",
        "Uplifting": "hopeful and positive",
        "Dark": "brooding and atmospheric",
    }
    tone_desc = tone_map.get(tone, "balanced")

    return f"""You are a creative music producer. Generate a detailed musical concept for a song.
{img_note}
GENRE: {genre}
MOOD: {mood}
THEME: {theme}
TONE: {tone_desc}

Describe in 3-5 paragraphs:
1. The specific subgenre, influences, and production style
2. The instrumentation, tempo feel, and key
3. The vocal style, delivery, and lyrical themes
4. A complete set of lyrics with verse/chorus/bridge structure

Be specific, creative, and musical. Do not use JSON. Write naturally."""


def _build_pass2_prompt(concept: str, genre: str, mood: str) -> str:
    """
    Pass 2: Format conversion.
    The model receives the creative concept and converts it to strict JSON.
    No creativity needed here — just formatting.
    """
    return f"""Convert this music concept into STRICT JSON for AceStep 1.5 music generation.

CONCEPT:
{concept}

OUTPUT: Respond with ONLY this JSON (no commentary, no markdown outside JSON):

{{
  "style": "[Style: Genre, SubGenre, Mood, Instrument1, Instrument2]\\n[Vocal: VocalTimbre, Delivery]\\n[Tempo: BPM bpm]\\n[Key: Root Mode]\\n[Production: Style1, Style2]",
  "lyrics": "[Verse 1]\\n\\n4-8 lyric lines here\\n\\n[Chorus]\\n\\n4-6 lyric lines here\\n\\n[Verse 2]\\n\\n4-8 lyric lines here\\n\\n[Bridge]\\n\\n2-4 lyric lines here\\n\\n[Chorus]\\n\\n4-6 lyric lines here\\n\\n[Outro]\\n\\n2-4 lyric lines here\\n\\n[End]"
}}

RULES:
1. [Tempo:] MUST contain a specific integer BPM matching the genre (40-220).
2. Use literal \\n for newlines in the JSON string values.
3. Lyric section headers use [Verse 1] format (with square brackets).
4. End lyrics with [End].
5. NO text outside the JSON object.
6. Use the ACTUAL lyrics from the concept above — do not invent new ones."""


# ── Node ───────────────────────────────────────────────────────────────────────

class S42PSongwriter:
    """
    🎵 S42P Songwriter v5.0

    Two-pass AI song generation for AceStep 1.5 Turbo via Ollama.

    PASS 1: Creative concept (genre + mood + theme → musical vision, free-form)
    PASS 2: AceStep formatting (vision → strict JSON style_tags + lyrics)

    The two-pass approach separates creativity from formatting, dramatically
    reducing "model didn't follow format" failures while producing more
    musically coherent output.

    Requires ComfyUI-Ollama + ollama Python package.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "connection": ("OLLAMA_CONNECTIVITY", {
                    "tooltip": "From ComfyUI-Ollama OllamaConnectivity node. "
                               "Sets server URL and model."}),
                "genre": ("STRING", {
                    "default": "synthwave, electronic",
                    "tooltip": "Music genre and sub-genre. Be specific for better results."}),
                "mood": ("STRING", {
                    "default": "energetic, euphoric",
                    "tooltip": "Emotional tone and energy of the song."}),
                "theme": ("STRING", {
                    "default": "driving through neon city lights at night",
                    "tooltip": "Lyrical subject matter or narrative."}),
                "tone": (["Neutral","Energetic","Melancholy","Cinematic","Aggressive","Uplifting","Dark"], {
                    "default": "Energetic",
                    "tooltip": "Overall tonal direction."}),
                "two_pass": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable two-pass generation (recommended). "
                               "Pass 1=creative concept, Pass 2=AceStep formatting. "
                               "Disable for faster single-pass (may have formatting failures)."}),
            },
            "optional": {
                "options": ("OLLAMA_OPTIONS", {
                    "tooltip": "From ComfyUI-Ollama OllamaOptions node. "
                               "temperature, seed, context size, etc."}),
                "context": ("STRING", {
                    "tooltip": "Previous raw_response for iterative refinement."}),
                "image": ("IMAGE", {
                    "tooltip": "Optional image input for vision models. "
                               "The song will be inspired by the image content."}),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("style_tags", "lyrics", "raw_response", "context")
    FUNCTION      = "generate"
    CATEGORY      = CATEGORY

    def generate(
        self,
        connection: dict,
        genre: str, mood: str, theme: str, tone: str,
        two_pass: bool = True,
        options: Optional[dict] = None,
        context: Optional[str] = None,
        image=None,
    ) -> Tuple[str, str, str, str]:

        if not _OLLAMA_OK:
            msg = ("ollama package not installed. "
                   "Run: pip install ollama\n"
                   "Also install ComfyUI-Ollama from ComfyUI Manager.")
            logger.error(f"[S42P Songwriter] {msg}")
            return ("", "", msg, "")

        # Parse connection
        base_url = connection.get("url", "http://127.0.0.1:11434") if isinstance(connection, dict) else "http://127.0.0.1:11434"
        model    = connection.get("model") if isinstance(connection, dict) else None
        if not model:
            models = _fetch_models(base_url)
            model  = models[0] if models else "qwen2.5:7b"

        client    = Client(host=base_url)
        ollama_opts = _enabled_options(options)

        # Image
        img_b64   = None
        has_image = False
        if image is not None and _is_vision_model(model):
            try:
                img_b64   = _tensor_to_b64_jpeg(image)
                has_image = True
            except Exception as e:
                logger.warning(f"[S42P Songwriter] Image conversion failed: {e}")

        def _call(prompt: str, system: Optional[str] = None, images: Optional[list] = None) -> str:
            """Single Ollama API call with error handling."""
            kwargs: dict = {
                "model":  model,
                "prompt": prompt,
            }
            if system:
                kwargs["system"] = system
            if images:
                kwargs["images"] = images
            if ollama_opts:
                kwargs["options"] = ollama_opts

            try:
                resp = client.generate(**kwargs)
                raw  = resp.get("response", "") if isinstance(resp, dict) else str(resp)
                return _strip_think_blocks(raw).strip()
            except Exception as e:
                logger.error(f"[S42P Songwriter] API call failed: {e}")
                return ""

        # ── TWO-PASS GENERATION ────────────────────────────────────────────────
        if two_pass:
            # Pass 1: Creative concept (free-form)
            pass1_prompt = _build_pass1_prompt(genre, mood, theme, tone, has_image)
            images_arg   = [img_b64] if img_b64 else None
            concept      = _call(pass1_prompt, images=images_arg)

            if not concept:
                return ("", "",
                        "Pass 1 (concept) returned empty response. Check Ollama server.",
                        "")

            # Pass 2: Format conversion (no image needed — concept already encodes it)
            pass2_prompt = _build_pass2_prompt(concept, genre, mood)
            raw_response = _call(pass2_prompt)

            full_raw = f"=== PASS 1 (Concept) ===\n{concept}\n\n=== PASS 2 (Format) ===\n{raw_response}"

        else:
            # Single-pass (legacy behaviour)
            from s42p_songwriter_v4_compat import _build_system_prompt  # type: ignore
            sys_prompt   = _build_system_prompt(tone, has_image)
            user_prompt  = (f"Genre: {genre}\nMood: {mood}\nTheme: {theme}"
                            + (f"\nContext: {context}" if context else ""))
            images_arg   = [img_b64] if img_b64 else None
            raw_response = _call(user_prompt, system=sys_prompt, images=images_arg)
            full_raw     = raw_response

        if not raw_response:
            return ("", "", full_raw or "No response from model.", full_raw or "")

        # ── Parse JSON ────────────────────────────────────────────────────────
        parsed = _extract_json(raw_response)

        if not parsed:
            return ("(No style — model did not follow format)",
                    "(No lyrics — model did not follow format)",
                    full_raw, full_raw)

        style  = str(parsed.get("style",  "")).strip()
        lyrics = str(parsed.get("lyrics", "")).strip()

        if not style and not lyrics:
            return ("(Empty output)", "(Empty output)", full_raw, full_raw)

        # Ensure required AceStep fields
        style  = _ensure_style_fields(style, raw_response, genre)
        lyrics = _clean_lyrics(lyrics)

        return (style, lyrics, full_raw, full_raw)


NODE_CLASS_MAPPINGS        = {"S42PSongwriter": S42PSongwriter}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PSongwriter": "🎵 S42P Songwriter"}
