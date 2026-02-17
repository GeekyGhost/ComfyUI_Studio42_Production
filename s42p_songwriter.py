# S42 Production Suite €” Songwriter
# Uses the ollama Python client (same as OllamaSongwriterV3) for reliable comms.
# Outputs AceStep 1.5-compatible style_tags + lyrics via OLLAMA_CONNECTIVITY.
#
# Optional IMAGE input enables vision models (llava, qwen2.5vl, qwen3-vl, etc.)
# Images are base64-encoded and passed directly to client.generate(images=[...])
#
# OUTPUTS †’ TextEncodeAceStepAudio1.5:
#   style_tags   †’ tags / style_tags input
#   lyrics       †’ lyrics input
#   raw_response †’ ShowText (debug)
#   context      †’ feed back to context input for iterative refinement

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
CATEGORY = "S42 Production Suite"

try:
    from PIL import Image as PILImage
    _PIL = True
except ImportError:
    _PIL = False


# ”€”€ Constants ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

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
    "rock": (90, 150),     "punk": (140, 200),    "alt-punk": (130, 175),
    "metal": (130, 200),   "grunge": (90, 140),   "edm": (120, 140),
    "house": (120, 130),   "techno": (130, 150),  "dubstep": (135, 145),
    "drum and bass": (160, 180), "dnb": (160, 180),
    "classical": (60, 160), "orchestral": (60, 160),
    "country": (90, 130),  "folk": (80, 120),     "reggae": (60, 90),
    "ska": (100, 140),     "trip-hop": (70, 95),  "trip hop": (70, 95),
    "synthwave": (100, 130), "retrowave": (100, 130), "cinematic": (60, 120),
    "epic": (80, 140),
}

_FALLBACK_MODELS = [
    "qwen2.5:7b", "qwen2.5:14b", "qwen3:8b", "qwen3:14b", "qwen3-vl:8b",
    "mistral:7b", "llama3.1:8b", "llama3.2:3b", "gemma3:9b", "phi4:14b",
    "llava:13b", "qwen2.5vl:7b", "minicpm-v:latest",
]

_model_cache: List[str] = []
_cached_url: str = ""


# ”€”€ Model list ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _fetch_models(base_url: str) -> List[str]:
    global _model_cache, _cached_url
    if _model_cache and _cached_url == base_url:
        return _model_cache
    try:
        if not _OLLAMA_OK:
            return _FALLBACK_MODELS
        client = Client(host=base_url)
        raw_list = client.list().get("models", [])
        # Ollama SDK returns dicts with either "name" or "model" depending on version
        models = [m.get("name") or m.get("model") for m in raw_list if m.get("name") or m.get("model")]
        if models:
            _model_cache = models
            _cached_url = base_url
            return models
    except Exception as e:
        logger.warning(f"[S42P Songwriter] Cannot fetch models: {e}")
    return _FALLBACK_MODELS


# ”€”€ Image helpers ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _is_vision_model(model: str) -> bool:
    low = model.lower()
    return any(m in low for m in [
        "-vl", "vl:", "llava", "vision", "minicpm-v", "moondream",
        "cogvlm", "bakllava", "internvl", "qvq", "gemma3",
        "pixtral", "qwen2-vl", "qwen2.5vl", "qwen3-vl",
    ])


def _tensor_to_b64_jpeg(image_tensor, max_edge: int = 1024) -> str:
    if not _PIL:
        raise RuntimeError("Pillow is required for image input.")
    arr = image_tensor
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    pil = PILImage.fromarray(arr, "RGB")
    w, h = pil.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ”€”€ Options helper (matches ComfyUI-Ollama OllamaOptionsV2 pattern) ”€”€”€”€”€”€”€

def _enabled_options(opts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not opts:
        return None
    out = {}
    for k in opts:
        if k.startswith("enable_") and opts[k]:
            out[k.replace("enable_", "")] = opts.get(k.replace("enable_", ""))
    return out or None


# ”€”€ Text parsing ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except Exception:
            pass
    return None


def _split_fallback(raw: str) -> Tuple[str, str]:
    """Last-resort split on first lyric section header."""
    m = re.search(r"^\s*\[.*?\]", raw, re.MULTILINE)
    if m:
        return raw[:m.start()].strip(), raw[m.start():].strip()
    parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(parts) > 1:
        return parts[0], "\n\n".join(parts[1:])
    return "", raw.strip()


def _parse_bpm(text: str, genre_hint: str) -> int:
    m = re.search(r"\b(\d{2,3})\s*(?:bpm)?\b", text, re.IGNORECASE)
    if m:
        b = int(m.group(1))
        if 40 <= b <= 220:
            return b
    low = genre_hint.lower()
    for g, (lo, hi) in BPM_HINTS.items():
        if g in low:
            return (lo + hi) // 2
    return 100


def _parse_key(text: str) -> str:
    for k in VALID_KEYS:
        if k.lower() in text.lower():
            return k
    return "A Minor"


def _clean_lyrics(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\\n", "\n").replace("\r\n", "\n")
    text = re.sub(r"(\[[^\]]+\])", lambda m: f"\n{m.group(0).rstrip()}\n\n", text)
    # Unwrap lines that are incorrectly wrapped in parentheses
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("(") and s.endswith(")") and not re.match(
                r"^\[(?:Verse|Chorus|Bridge|Pre-Chorus|Outro|Intro|Hook|Break|End)",
                s, re.IGNORECASE):
            lines.append(s[1:-1].strip())
        else:
            lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not re.search(r"\[End\]", text, re.IGNORECASE):
        text = text.rstrip() + "\n\n[End]"
    return text.strip()


def _ensure_style_fields(style: str, full_text: str, description: str) -> str:
    """Inject [Tempo:] and [Key:] tags if the model omitted them."""
    if "[Tempo:" not in style:
        bpm = _parse_bpm(style + " " + full_text, description)
        style = style.rstrip() + f"\n[Tempo: {bpm} bpm]"
    if "[Key:" not in style:
        key = _parse_key(style + " " + full_text)
        style = style.rstrip() + f"\n[Key: {key}]"
    return style.strip()


# ”€”€ System prompt ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _build_system_prompt(tone: str, has_image: bool) -> str:
    tone_map = {
        "Neutral":    "Write balanced, versatile lyrics without strong emotional bias.",
        "Energetic":  "Write high-energy, punchy lyrics with strong momentum and drive.",
        "Melancholy": "Write introspective, emotionally heavy lyrics with depth and longing.",
        "Cinematic":  "Write sweeping, narrative-driven lyrics that evoke visual imagery.",
        "Aggressive": "Write raw, intense lyrics with power and conviction.",
        "Uplifting":  "Write hopeful, positive lyrics with an uplifting arc.",
        "Dark":       "Write brooding, tense, atmospheric lyrics.",
    }
    tone_instr = tone_map.get(tone, tone_map["Neutral"])
    img_instr = (
        "\nAN IMAGE HAS BEEN PROVIDED. Analyse it carefully â€” scene, colours, mood, "
        "subject, atmosphere. Let the image directly inspire all musical and lyrical choices.\n"
    ) if has_image else ""

    return f"""You are an expert music producer and lyricist for AceStep 1.5 AI music generation.
{img_instr}
TONE DIRECTIVE: {tone_instr}

OUTPUT: Respond with STRICT JSON only (no commentary, no markdown outside the JSON):

{{
  "style": "[Style: Genre, SubGenre, Mood, Instrument1, Instrument2]\\n[Vocal: VocalTimbre, Delivery]\\n[Tempo: BPM bpm]\\n[Key: Root Mode]\\n[Production: Style1, Style2]",
  "lyrics": "[Verse 1]\\n\\n4-8 lines\\n\\n[Pre-Chorus]\\n\\n2-4 lines\\n\\n[Chorus]\\n\\n4-6 lines\\n\\n[Verse 2]\\n\\n4-8 lines\\n\\n[Bridge]\\n\\n2-4 lines\\n\\n[Chorus]\\n\\n4-6 lines\\n\\n[Outro]\\n\\n2-4 lines\\n\\n[End]"
}}

RULES:
1. [Tempo: BPM bpm] must be a realistic integer 40-220 for the genre.
2. Use literal \\n for newlines inside the JSON string.
3. Lyric lines are PLAIN TEXT â€” do not wrap them in extra brackets.
4. End lyrics with [End].
5. NO text outside the JSON object.
"""


# ”€”€ Node ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

class S42PSongwriter:

    saved_context = None

    @classmethod
    def INPUT_TYPES(cls):
        try:
            models = _fetch_models("http://localhost:11434")
        except Exception:
            models = _FALLBACK_MODELS

        return {
            "required": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Override system prompt. Leave blank to use the built-in AceStep prompt. "
                        "Useful for custom songwriter personas."
                    ),
                }),
                "description": ("STRING", {
                    "multiline": True,
                    "default": (
                        "An uplifting synthwave track with driving bassline, "
                        "lush pads, and hopeful vocals about overcoming adversity."
                    ),
                    "tooltip": (
                        "Describe the music â€” genre, mood, instruments, themes. "
                        "If an image is connected, describe what aspect to focus on."
                    ),
                }),
                "tone": ([
                    "Neutral", "Energetic", "Melancholy", "Cinematic",
                    "Aggressive", "Uplifting", "Dark",
                ], {"default": "Neutral"}),
                "duration_sec": ("INT", {
                    "default": 30, "min": 10, "max": 240, "step": 5,
                    "tooltip": "Target song length â€” guides section count and lyric density.",
                }),
            },
            "optional": {
                "connectivity": ("OLLAMA_CONNECTIVITY", {
                    "forceInput": False,
                    "tooltip": "Connect an OllamaConnectivityV2 node. Required to run.",
                }),
                "options": ("OLLAMA_OPTIONS", {
                    "forceInput": False,
                    "tooltip": "Connect an OllamaOptionsV2 node for advanced inference settings.",
                }),
                "optional_image": ("IMAGE", {
                    "tooltip": (
                        "Connect an image with a vision model to write a song about the scene. "
                        "Works with qwen3-vl, llava, qwen2.5vl, minicpm-v, moondream."
                    ),
                }),
                "image_max_edge": ("INT", {
                    "default": 768, "min": 256, "max": 1536, "step": 128,
                    "tooltip": "Resize image longest edge before sending to model.",
                }),
                "keep_context": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Persist context between runs for iterative refinement.",
                }),
                "context": ("OLLAMA_CONTEXT", {
                    "forceInput": False,
                    "tooltip": "Previous context output â€” connect back for iterative refinement.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 999999,
                    "tooltip": "Variation seed hint passed to model. 0 = ignore.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "OLLAMA_CONTEXT")
    RETURN_NAMES = ("style_tags", "lyrics", "raw_response", "context")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "S42P Songwriter â€” generates AceStep 1.5 style tags + lyrics via Ollama."

    def generate(
        self,
        system_prompt: str,
        description: str,
        tone: str,
        duration_sec: int,
        connectivity=None,
        options=None,
        optional_image=None,
        image_max_edge: int = 768,
        keep_context: bool = False,
        context=None,
        seed: int = 0,
    ):
        if not connectivity:
            raise Exception(
                "S42P Songwriter requires an OLLAMA_CONNECTIVITY input. "
                "Connect an OllamaConnectivityV2 node."
            )

        url   = connectivity["url"]
        model = connectivity["model"]
        keep_alive_unit = "m" if connectivity.get("keep_alive_unit") == "minutes" else "h"
        keep_alive_val  = f"{connectivity.get('keep_alive', 5)}{keep_alive_unit}"

        if not _OLLAMA_OK:
            raise RuntimeError(
                "[S42P Songwriter] ollama package not installed. "
                "Run: pip install ollama  (or: python_embeded\\python.exe -m pip install ollama)"
            )
        client = Client(host=url)
        req_opts = _enabled_options(options)

        # Context handling €” identical to OllamaSongwriterV3
        if isinstance(context, str) and context.strip():
            try:
                context = [int(x.strip()) for x in context.split(",") if x.strip()]
            except Exception:
                context = None
        if keep_context and context is None:
            context = self.saved_context

        has_image = optional_image is not None
        is_vision = _is_vision_model(model)

        if has_image and not is_vision:
            print(f"[S42P Songwriter] âš   Image connected but '{model}' may not support vision. "
                  "Try qwen3-vl, llava, qwen2.5vl, or minicpm-v.")
        if is_vision and not has_image:
            print(f"[S42P Songwriter] â„¹  '{model}' is a vision model â€” no image connected.")

        # Encode image if applicable
        images_b64 = None
        if has_image and is_vision:
            try:
                images_b64 = [_tensor_to_b64_jpeg(optional_image, image_max_edge)]
            except Exception as e:
                print(f"[S42P Songwriter] âš   Image encoding failed: {e} â€” continuing without image")
                has_image = False

        # Build prompts
        active_system = system_prompt.strip() if system_prompt.strip() else _build_system_prompt(tone, has_image)

        seed_hint = f"\nVariation seed: {seed}" if seed > 0 else ""
        dur_hint  = (
            f"\nTarget duration: {duration_sec}s â€” "
            "adjust section count and lyric density accordingly."
        )
        img_guide = (
            "An image is attached. Analyse it carefully â€” your musical and lyrical choices "
            "must reflect what you see.\n\n"
        ) if has_image else ""

        user_prompt = (
            f"{img_guide}Create a complete AceStep 1.5 music generation prompt for:\n\n"
            f"{description}{seed_hint}{dur_hint}\n\n"
            "Respond with ONLY the JSON object. No other text."
        )

        mode_str = "VISION" if (has_image and is_vision) else "TEXT"
        print(f"[S42P Songwriter] {mode_str}  model={model}  tone={tone}  {duration_sec}s ...")

        # ”€”€ Call Ollama €” same pattern as OllamaSongwriterV3 ”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        response = client.generate(
            model=model,
            system=active_system,
            prompt=user_prompt,
            images=images_b64,
            context=context,
            options=req_opts,
            keep_alive=keep_alive_val,
            format="",
        )

        raw = response.get("response", "") or ""

        if keep_context:
            self.saved_context = response.get("context")

        context_out = response.get("context")

        if not raw:
            print("[S42P Songwriter] âš   Empty response from model.")
            return (
                "[Style: Electronic]\n[Vocal: Smooth]\n[Tempo: 100 bpm]\n[Key: A Minor]",
                "[Verse 1]\n\n(Empty response â€” check model is running)\n\n[End]",
                "EMPTY RESPONSE",
                context_out,
            )

        # Strip reasoning tokens emitted by qwen3 / deepseek-r1
        raw_clean = _strip_think_blocks(raw)

        # ”€”€ Parse JSON (primary path €” matches OllamaSongwriterV3) ”€”€”€”€”€”€”€”€
        style_raw  = ""
        lyrics_raw = ""
        parsed     = _extract_json(raw_clean)

        if parsed and isinstance(parsed, dict):
            style_raw  = str(parsed.get("style",      "") or "").strip()
            lyrics_raw = str(parsed.get("lyrics",     "") or "").strip()
            if not style_raw:
                style_raw  = str(parsed.get("style_tags", "") or "").strip()
            if not lyrics_raw:
                lyrics_raw = str(parsed.get("lyric",      "") or "").strip()

        # ”€”€ Fallback: paragraph split ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        if not style_raw and not lyrics_raw:
            style_raw, lyrics_raw = _split_fallback(raw_clean)

        # ”€”€ Repair style ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        if not style_raw:
            bpm = _parse_bpm(raw_clean, description)
            key = _parse_key(raw_clean)
            style_raw = (
                f"[Style: Electronic, Atmospheric]\n[Vocal: Smooth, Melodic]\n"
                f"[Tempo: {bpm} bpm]\n[Key: {key}]\n[Production: Modern]"
            )
            print("[S42P Songwriter] âš   Style parse failed â€” defaults inserted. Check raw_response.")
        else:
            style_raw = _ensure_style_fields(style_raw, raw_clean, description)

        # ”€”€ Repair lyrics ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        if not lyrics_raw:
            lyrics_raw = (
                "[Verse 1]\n\n(No lyrics â€” model did not follow format. "
                "Check raw_response or try a different model.)\n\n[End]"
            )
            print("[S42P Songwriter] âš   Lyrics parse failed. Connect raw_response â†’ ShowText.")

        style_tags = style_raw.strip()
        lyrics     = _clean_lyrics(lyrics_raw)

        tm = re.search(r"\[Tempo:\s*(\d+)", style_tags, re.IGNORECASE)
        km = re.search(r"\[Key:\s*([^\]]+)\]", style_tags, re.IGNORECASE)
        print(f"[S42P Songwriter] âœ“  "
              f"bpm={tm.group(1) if tm else '?'}  "
              f"key={km.group(1).strip() if km else '?'}")

        return (style_tags, lyrics, raw, context_out)


NODE_CLASS_MAPPINGS        = {"S42PSongwriter": S42PSongwriter}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PSongwriter": "S42P Songwriter ðŸŽµ"}
