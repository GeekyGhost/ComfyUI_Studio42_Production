"""
S42 Production Suite — Songwriter
====================================
Single-call Ollama node that generates AceStep 1.5-compatible style tags,
lyrics, and BPM from a simple genre/mood description. Designed to wire
directly into TextEncodeAceStepAudio1.5.

Ported from ULM Songwriter v3 (GeekyGhost) with the following changes:
  • Category updated to S42 Production Suite
  • Node/display names updated to S42P namespace
  • Ollama URL param added (default localhost:11434)
  • Tone control added (Neutral / Energetic / Melancholy / Cinematic / Aggressive)
  • context I/O preserved for multi-turn iterative refinement

WIRING:
  style_tags   → TextEncodeAceStepAudio1.5.style_tags   (STRING)
  lyrics       → TextEncodeAceStepAudio1.5.lyrics       (STRING)
  bpm          → TextEncodeAceStepAudio1.5.bpm          (INT)
  raw_response → ShowText node (debug / inspect full LLM output)
  context      → loop back to context input for iterative refinement

Key is embedded in style_tags (e.g. "A minor key") — AceStep reads it there.
No separate key output — wiring key to a COMBO widget causes type errors.

REQUIREMENTS:
  Ollama running locally (default: http://localhost:11434)
  Any Ollama model capable of following structured output instructions.
  Recommended: qwen2.5:7b, qwen3:8b, mistral:7b, llama3.1:8b

Python 3.12 | ComfyUI Portable | No extra dependencies (urllib only)
"""

from __future__ import annotations

import re
import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)

CATEGORY = "S42 Production Suite 🎵 Songwriter"


# ── Music theory constants ────────────────────────────────────────────────────

VALID_KEYS = [
    "C major",  "C minor",  "C# major", "C# minor",
    "Db major", "Db minor", "D major",  "D minor",
    "D# major", "D# minor", "Eb major", "Eb minor",
    "E major",  "E minor",  "F major",  "F minor",
    "F# major", "F# minor", "Gb major", "Gb minor",
    "G major",  "G minor",  "G# major", "G# minor",
    "Ab major", "Ab minor", "A major",  "A minor",
    "A# major", "A# minor", "Bb major", "Bb minor",
    "B major",  "B minor",
]

BPM_HINTS: Dict[str, Tuple[int, int]] = {
    "ambient":        (55,  80), "drone":          (45,  70),
    "ballad":         (60,  80), "blues":          (60, 100),
    "jazz":           (80, 140), "soul":           (70, 110),
    "lo-fi":          (70,  90), "lofi":           (70,  90),
    "hip-hop":        (75, 100), "hip hop":        (75, 100),
    "trap":           (70, 100), "r&b":            (70, 100),
    "rnb":            (70, 100), "pop":            (90, 130),
    "indie":          (90, 130), "rock":           (90, 150),
    "punk":          (140, 200), "alt-punk":      (120, 180),
    "metal":         (130, 200), "grunge":         (90, 140),
    "edm":           (120, 140), "house":         (120, 130),
    "techno":        (130, 150), "dubstep":       (135, 145),
    "drum and bass": (160, 180), "dnb":           (160, 180),
    "classical":      (60, 160), "orchestral":     (60, 160),
    "country":        (90, 130), "folk":           (80, 120),
    "reggae":         (60,  90), "ska":           (100, 140),
    "trip-hop":       (70,  95), "trip hop":       (70,  95),
    "synthwave":     (100, 130), "retrowave":     (100, 130),
}

_FALLBACK_MODELS = [
    "qwen2.5:7b", "qwen2.5:14b", "qwen3:8b", "qwen3:14b",
    "mistral:7b", "llama3.1:8b", "llama3.2:3b", "gemma3:9b", "phi4:14b",
]

_model_cache: List[str] = []
_cached_url:  str       = ""


# ── Ollama helpers ────────────────────────────────────────────────────────────

def _fetch_models(base_url: str) -> List[str]:
    global _model_cache, _cached_url
    if _model_cache and _cached_url == base_url:
        return _model_cache
    try:
        url  = f"{base_url.rstrip('/')}/api/tags"
        req  = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data   = json.loads(r.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            if models:
                _model_cache = models
                _cached_url  = base_url
                return models
    except Exception as e:
        logger.warning(f"[S42P Songwriter] Cannot fetch Ollama models from {base_url}: {e}")
    return _FALLBACK_MODELS


def _ollama_generate(base_url: str, model: str, prompt: str,
                      system: str, timeout: int = 120) -> str:
    url     = f"{base_url.rstrip('/')}/api/generate"
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.75, "top_p": 0.9, "num_predict": 1200},
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}. "
            f"Is Ollama running? Error: {e}"
        )


# ── Output parsers ────────────────────────────────────────────────────────────

def _extract_section(text: str, start_tag: str, end_tag: str) -> str:
    pattern = re.compile(
        rf'---\s*{re.escape(start_tag)}\s*---\s*(.*?)\s*---\s*{re.escape(end_tag)}\s*---',
        re.DOTALL | re.IGNORECASE
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _parse_bpm(text: str, genre_hint: str) -> int:
    # Try to find an explicit BPM number in the response
    m = re.search(r'\b(\d{2,3})\s*(?:bpm|BPM)\b', text)
    if m:
        bpm = int(m.group(1))
        if 40 <= bpm <= 220:
            return bpm
    # Fall back to genre range midpoint
    genre_lower = genre_hint.lower()
    for genre, (lo, hi) in BPM_HINTS.items():
        if genre in genre_lower:
            return (lo + hi) // 2
    return 100


def _parse_key(text: str) -> str:
    for k in VALID_KEYS:
        if k.lower() in text.lower():
            return k
    return "A minor"


def _clean_lyrics(raw: str) -> str:
    """Normalise escaped newlines and ensure section headers are on own lines."""
    text = raw.replace("\\n", "\n").replace("\r\n", "\n")
    # Ensure section tags like [Verse 1] start on a new line
    text = re.sub(r'(?<!\n)(\[[^\]]+\])', r'\n\1', text)
    # Remove consecutive blank lines beyond 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _try_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to extract JSON if model responded with JSON instead of ---blocks---."""
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt(tone: str) -> str:
    tone_instructions = {
        "Neutral":     "Write balanced, versatile lyrics without strong emotional bias.",
        "Energetic":   "Write high-energy, punchy lyrics with strong momentum and drive.",
        "Melancholy":  "Write introspective, emotionally heavy lyrics with depth and longing.",
        "Cinematic":   "Write epic, sweeping lyrics that evoke imagery and narrative scope.",
        "Aggressive":  "Write intense, confrontational lyrics with raw energy and edge.",
    }.get(tone, "Write balanced, versatile lyrics.")

    return f"""You are a professional music producer and songwriter specialising in generating 
structured prompts for the AceStep 1.5 AI music generation model.

TONE DIRECTIVE: {tone_instructions}

Your response MUST use EXACTLY this format with no deviations:

---STYLE---
<comma-separated list: genre, mood descriptors, instruments, vocal style, key (e.g. A minor key), BPM>
---STYLE---

---VOCAL---
<voice description: gender, texture, delivery style, e.g. "warm female vocals, breathy delivery">
---VOCAL---

---BPM---
<integer only, e.g. 128>
---BPM---

---KEY---
<key name only, e.g. A minor>
---KEY---

---LYRICS---
[Verse 1] [Energy: Low/Mid/High]
<4-8 lines>

[Pre-Chorus] [Energy: Mid]
<2-4 lines>

[Chorus] [Energy: High]
<4-6 lines>

[Verse 2] [Energy: Mid]
<4-8 lines>

[Bridge] [Energy: Mid]
<2-4 lines>

[Chorus] [Energy: High]
<repeat or variation>

[Outro] [Energy: Low]
<2-4 lines>
[End]
---LYRICS---

RULES:
- The key MUST appear in STYLE as "<key> key", e.g. "A minor key"
- BPM must be a realistic integer for the genre (40-220 range)
- Lyrics must have section headers in [Square Brackets]
- Every section must include [Energy: Low/Mid/High] tag
- Do NOT include explanations, apologies, or preamble — output the blocks only
- Style tags should be 6-12 descriptors separated by commas
"""


# ── ComfyUI Node ──────────────────────────────────────────────────────────────

class S42PSongwriter:
    """
    S42P Songwriter — generate AceStep 1.5 style tags, lyrics, and BPM
    from a simple description using a local Ollama model.

    Wire outputs directly to TextEncodeAceStepAudio1.5:
      style_tags → style_tags
      lyrics     → lyrics
      bpm        → bpm

    Use the context output → context input loop for iterative refinement
    (each generation builds on the previous conversation context).
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Try to get live model list; fall back to defaults silently
        try:
            models = _fetch_models("http://localhost:11434")
        except Exception:
            models = _FALLBACK_MODELS

        return {
            "required": {
                "description": ("STRING", {
                    "multiline": True,
                    "default": (
                        "An uplifting synthwave track with driving bassline, "
                        "lush pads, and hopeful vocals about overcoming adversity."
                    ),
                    "tooltip": (
                        "Describe the music you want. Include genre, mood, instruments, "
                        "vocal style, themes. The more specific, the better the output."
                    )
                }),
                "model": (models, {
                    "default": models[0] if models else "qwen2.5:7b",
                    "tooltip": "Ollama model to use. Refresh ComfyUI to update this list."
                }),
                "tone": ([
                    "Neutral", "Energetic", "Melancholy", "Cinematic", "Aggressive"
                ], {
                    "default": "Neutral",
                    "tooltip": "Overall emotional tone directive for the songwriter."
                }),
                "duration_sec": ("INT", {
                    "default": 30, "min": 10, "max": 240, "step": 5,
                    "tooltip": "Target track duration in seconds. Affects lyric density."
                }),
            },
            "optional": {
                "ollama_url": ("STRING", {
                    "default": "http://localhost:11434",
                    "multiline": False,
                    "tooltip": "Ollama base URL. Change if running on a different port."
                }),
                "context": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Previous context for iterative refinement. "
                        "Connect the context output back here to build on prior generations."
                    )
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 999999,
                    "tooltip": "Seed hint passed to the description. 0 = ignore."
                }),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES  = ("style_tags", "lyrics", "bpm", "raw_response", "context")
    FUNCTION      = "generate"
    CATEGORY      = CATEGORY
    OUTPUT_NODE   = True

    def generate(self, description: str, model: str, tone: str,
                 duration_sec: int, ollama_url: str = "http://localhost:11434",
                 context: str = "", seed: int = 0) -> tuple:

        system = _build_system_prompt(tone)

        # Build user prompt
        seed_hint = f"\nSeed hint for variation: {seed}" if seed > 0 else ""
        dur_hint  = (f"\nTarget duration: {duration_sec} seconds — "
                     f"adjust lyric density accordingly (shorter = fewer sections).")
        prior     = (f"\n\nPrevious context for refinement:\n{context}" if context else "")

        user_prompt = (
            f"Create a complete AceStep 1.5 music generation prompt for:\n\n"
            f"{description}{seed_hint}{dur_hint}{prior}\n\n"
            f"Output ONLY the structured blocks. No explanations."
        )

        print(f"[S42P Songwriter] Generating with model={model}, tone={tone}, "
              f"duration={duration_sec}s ...")

        try:
            raw = _ollama_generate(ollama_url, model, user_prompt, system)
        except RuntimeError as e:
            error_msg = str(e)
            logger.error(f"[S42P Songwriter] {error_msg}")
            return (
                "electronic, cinematic, atmospheric",
                "[Verse 1]\nOllama connection failed — check Ollama is running.\n[End]",
                100,
                f"ERROR: {error_msg}",
                ""
            )

        # ── Parse sections ────────────────────────────────────────────────────
        style_raw  = _extract_section(raw, "STYLE",  "STYLE")
        vocals_raw = _extract_section(raw, "VOCAL",  "VOCAL")
        lyrics_raw = _extract_section(raw, "LYRICS", "LYRICS")
        bpm_raw    = _extract_section(raw, "BPM",    "BPM")
        key_raw    = _extract_section(raw, "KEY",    "KEY")

        # JSON fallback if model ignored the ---BLOCK--- format
        if not style_raw:
            parsed = _try_json_parse(raw)
            if parsed:
                style_raw  = parsed.get("style_tags", "")
                vocals_raw = parsed.get("vocals",     "")
                lyrics_raw = parsed.get("lyrics",     "")
                bpm_raw    = str(parsed.get("bpm",    "100"))
                key_raw    = parsed.get("key",        "")

        # ── Assemble style_tags ───────────────────────────────────────────────
        key   = _parse_key(key_raw or style_raw or raw)
        bpm   = _parse_bpm(bpm_raw or raw, description)
        key_tag = f"{key} key"

        style_parts = [s.strip() for s in style_raw.split(",") if s.strip()]
        if vocals_raw.strip():
            style_parts.append(vocals_raw.strip())
        # Ensure key is in style tags
        if not any(key.lower() in p.lower() for p in style_parts):
            style_parts.append(key_tag)

        style_tags = ", ".join(style_parts)

        # ── Assemble lyrics ───────────────────────────────────────────────────
        lyrics = _clean_lyrics(lyrics_raw) if lyrics_raw else (
            "[Verse 1]\n(No lyrics generated — check raw_response for model output)\n[End]"
        )

        # ── Build context for next iteration ─────────────────────────────────
        new_context = (
            f"Previous generation:\n"
            f"Style: {style_tags}\n"
            f"BPM: {bpm}\nKey: {key}\n"
            f"Lyrics (first 200 chars): {lyrics[:200]}..."
        )

        print(f"[S42P Songwriter] ✓ style_tags={style_tags[:80]}...")
        print(f"[S42P Songwriter] ✓ bpm={bpm}, key={key}")

        return (style_tags, lyrics, bpm, raw, new_context)


NODE_CLASS_MAPPINGS = {
    "S42PSongwriter": S42PSongwriter
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PSongwriter": "S42P Songwriter 🎵"
}
