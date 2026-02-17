# S42 Production Suite for ComfyUI — v4.0

**Professional audio mastering, AI music generation, background removal, reactive video production, and procedural FX nodes.**

Designed for **AceStep 1.5 Turbo + Wan + ComfyUI Portable** on RTX-class hardware.  
All DSP runs on **CPU** — your GPU stays free for generation.

---

## What's New in v4.0

| Change | Detail |
|--------|--------|
| **Ollama nodes removed** | `s42p_ollama_nodes.py` removed. Use **ComfyUI-Ollama** for connectivity/options nodes (see Songwriter section). |
| **Procedural FX expanded** | 10 → **16 nodes** with new effects: Lightning Storm, Geometric Storm, Meteor Shower, Petal Rain, Gold Dust, Cyber Grid, Plasma Orbs. All nodes have full tooltips. |
| **FX node IDs updated** | All procedural FX now use `S42P_FX_` prefix to avoid conflicts with other suites. Old names auto-warn if workflows reference them. |
| **Cleaner startup** | Songwriter skipped gracefully if `ollama` package not installed, with helpful console message. |

---

## Installation

### Step 1 — Clone / Download

Place all files in:
```
ComfyUI/custom_nodes/S42-Production-Suite/
```

All `.py` files must be in the **same flat directory** as `__init__.py`. No subdirectories.

### Step 2 — Install Core Dependencies

**Windows (ComfyUI Portable):**
```batch
.\python_embeded\python.exe -m pip install -r custom_nodes\S42-Production-Suite\requirements.txt
```

**Linux / Mac (ComfyUI Portable):**
```bash
./python_embeds/bin/python -m pip install -r custom_nodes/S42-Production-Suite/requirements.txt
```

### Step 3 — Songwriter / Ollama Setup

The S42P Songwriter node uses **ComfyUI-Ollama** for Ollama connectivity and model options.

**Install ComfyUI-Ollama:**

Option A — via ComfyUI Manager (recommended):
1. Open ComfyUI Manager → Custom Nodes → Search `ComfyUI-Ollama`
2. Install and restart ComfyUI

Option B — manual:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/stavsap/comfyui-ollama.git
```
Then install its requirements:
```bash
.\python_embeded\python.exe -m pip install ollama
```

**Install and run Ollama:**
1. Download from [ollama.com](https://ollama.com)
2. Pull a model: `ollama pull qwen2.5:7b`
3. Ollama runs at `http://127.0.0.1:11434` by default

**ComfyUI-Ollama nodes used by S42P Songwriter:**
- `OllamaConnectivity` — set server URL and model
- `OllamaOptions` *(optional)* — temperature, seed, context size, etc.

**Workflow:**
```
OllamaConnectivity ──→ S42P Songwriter ──→ TextEncodeAceStepAudio1.5
OllamaOptions ───────↗
```

### Step 4 — Optional: AI Background Removal

```bash
# U2Net / ISNet / BiRefNet via rembg
pip install rembg onnxruntime

# GPU acceleration
pip install onnxruntime-gpu

# BiRefNet via HuggingFace (highest quality)
pip install transformers torchvision

# BEN2
pip install ben2
```
Without these, Background Remover falls back to Chroma Key / Luma Key (OpenCV only).

### Step 5 — Restart ComfyUI

All nodes appear under **"S42 Production Suite"** in the node menu.

---

## Nodes Reference

---

### 🎵 S42P Songwriter

AI song generation for AceStep 1.5 Turbo via Ollama.

**Requires:** ComfyUI-Ollama installed (see above). `ollama` Python package (`pip install ollama`).

**Inputs:**
- `connection` — from ComfyUI-Ollama `OllamaConnectivity` node
- `options` *(optional)* — from ComfyUI-Ollama `OllamaOptions` node
- `genre` — music style/genre description
- `mood` — emotional tone and energy
- `theme` — lyrical subject matter
- `image` *(optional)* — vision input for image-inspired songs (requires vision model)

**Outputs:** `style_tags` + `lyrics` → `TextEncodeAceStepAudio1.5`

**Reasoning model support:** `<think>...</think>` blocks auto-stripped for qwen3, deepseek-r1, etc.

**Recommended models:**
- Fast text: `qwen2.5:7b`, `qwen3:8b`, `mistral:7b`, `llama3.1:8b`
- Vision (with image input): `llava:13b`, `qwen2.5vl:7b`, `minicpm-v:latest`
- Thinking/reasoning: `qwen3:8b`, `qwen3:14b`, `deepseek-r1:8b`

---

### ✨ Procedural FX (16 Nodes)

All FX nodes share these common inputs:

| Input | Description |
|-------|-------------|
| `width` / `height` | Output resolution |
| `seed` | Deterministic particle placement |
| `frame` | Animation frame index (connect to frame counter) |
| `animate_batch` | Each batch image advances one frame |
| `batch_size` | Number of frames to generate |
| `bg_transparent` | RGBA output for compositing |
| `bg_r/g/b` | Solid background color |
| `override_colors` | Enable custom color palette |
| `col1/col2 r/g/b` | Primary/secondary particle colors |
| `speed_mult` | Global speed multiplier |
| `use_custom_direction` | Override built-in direction |
| `angle_deg` | Custom direction angle (0=right, 90=down) |
| `reverse` | Flip default direction |

| Node | Effect | Best For |
|------|--------|----------|
| **S42P FX • Hearts Rain** | Falling hearts with glow | Romance, Valentine |
| **S42P FX • Snow** | Circle or 6-arm snowflakes | Winter, ambient |
| **S42P FX • Starfield** | Stars with optional warp trails | Sci-fi, space |
| **S42P FX • Bokeh Orbs** | Soft out-of-focus light circles | Cinematic overlay |
| **S42P FX • Fireflies** | Drifting luminous dots | Nature, dreamy |
| **S42P FX • Confetti** | Tumbling celebration pieces | Party, celebration |
| **S42P FX • Matrix Rain** | 0/1 digital rain (3 char sets) | Cyberpunk, tech |
| **S42P FX • Neon Scanlines** | Scrolling vertical bars | Synthwave, retro |
| **S42P FX • Rising Bubbles** | Transparent ring bubbles | Underwater, liquid |
| **S42P FX • Lightning Storm** | Electric bolt streaks | EDM, energy |
| **S42P FX • Geometric Storm** | Spinning diamonds/stars/crosses | Abstract, futuristic |
| **S42P FX • Meteor Shower** | Diagonal streak trails | Space, dramatic |
| **S42P FX • Petal Rain** | Drifting flower petals | Sakura, nature |
| **S42P FX • Gold Dust** | Rising sparkle particles | Luxury, awards |
| **S42P FX • Cyber Grid** | HUD crosshair elements | Sci-fi overlay |
| **S42P FX • Plasma Orbs** | Large slow lava-lamp orbs | Psychedelic, ambient |

**Transparent compositing workflow:**
```
S42P FX • Starfield (bg_transparent=True)
    ↓
S42P Layer Composer (fx_image)
    ↑
Subject Video → S42P BG Remover → result_rgb (fg_image)
```

---

### 🎵 S42P Audio Reactive FX

Apply audio-driven effects to image/video frame batches.

**Effects:**

| Effect | Description |
|--------|-------------|
| `pulse` | Scale/zoom with audio envelope |
| `neon` | Glowing edge highlight |
| `cascade` | Echo copies offset by envelope |
| `flicker` | Color flash on beats |
| `rotation` | Frame rotation by envelope |
| `warp` | Radial ripple distortion |

**Key inputs:**
- `audio` — ComfyUI AUDIO (from VHS_LoadAudio or similar)
- `images` — frame batch to process
- `mask` *(optional)* — restrict to foreground (from S42P BG Remover)
- `external_envelope` / `external_beats` — from S42P Beat Analyzer / Audio Analyser

---

### 🟩 S42P Matrix Rain FX

Dedicated high-quality matrix rain generator with audio reactivity.

**Separate from the Procedural FX version** — this node generates columnar rain with proper column state simulation.

- `transparent_bg` — RGBA output for compositing
- `char_set` — `binary` (0/1) or `extended` (binary + katakana)
- `audio` — bass drives speed and density reactively

---

### 🎭 S42P Background Remover

Multi-method background removal.

| Method | Requires | Best For |
|--------|----------|----------|
| `birefnet-general` | rembg / transformers | General — best quality |
| `birefnet-portrait` | rembg / transformers | Human portraits |
| `birefnet-massive` | rembg | Highest AI quality |
| `ben2-base` | ben2 | Confidence-guided matting |
| `u2net` / `u2net_human_seg` | rembg | Classic reliable |
| `isnet-anime` | rembg | Anime characters |
| `chroma_key_professional` | OpenCV only | Green/blue screen |
| `luma_key_professional` | OpenCV only | Brightness-based |
| `difference_key` | OpenCV only | With reference background |

**Outputs:** `result_rgb` + `result_rgba` + `result_mask` + `info`

---

### 🎬 S42P Layer Composer

Multi-layer compositor for music video production.

**Layer stack (bottom to top):**
1. `bg_image` — background (visualizer, matrix rain, static image)
2. `fx_image` — effects overlay (procedural FX, any RGBA)
3. `fg_image` — foreground subject (background-removed video)

**Blend modes:** normal, multiply, screen, overlay, hard_light, soft_light, dodge, burn, darken, lighten, difference, exclusion, add, subtract

Batch-aware — handles mismatched frame counts by cycling shorter batches.

---

### 🎛️ S42P Parametric EQ

8-band parametric equalizer.  
**Band types:** High Pass, Low Pass, Low Shelf, High Shelf, Peak, Notch.

### 📊 S42P Dynamics Processor

3-band multiband compressor + glue compressor + output limiter.

### 🎚️ S42P Mastering Chain

Full mastering pipeline: EQ → Stereo Width → Multiband Compression → Limiter → LUFS normalization.

### 🥁 S42P Beat Analyzer

Beat detection, BPM tracking, onset detection.  
**Outputs:** `envelope` + `beat_frames` tensors → Audio Reactive FX.

### 🎚️ S42P Audio Analyser

RMS envelope extraction + beat detection.  
**Outputs:** `envelope` + `beat_frames` → Audio Reactive FX.

### 🎬 S42P Audio Visualizer

11 animated visualizer modes:  
`spectrum_classic`, `lava_lamp`, `wormhole`, `matrix`, `stargate`, `oscilloscope`, `nebula`, `aurora`, `fractal_zoom`, `dna_helix`, `liquid_metal`

### ✨ S42P Latent Enhancer

Spectral balance correction in latent space before VAE decode.  
Addresses sub-bass loss and spectral imbalance introduced during generation.

### 🎵 S42P EQ Preset Loader

AceStep-specific EQ correction presets. Connect to S42P Parametric EQ.

### 🎬 S42P Keyframe Animator

Keyframe-driven transform animation: scale, translate, rotate, opacity.

### 🔄 S42P Transition

Video transitions: cut, dissolve, wipe, zoom, slide.

### 🛠️ S42P Video Tools

Resize, crop, pad, flip, speed change, frame extraction.

### 🎨 S42P Color Grade

LUT application, curves, HSL, temperature/tint, vignette.

### 🎵 S42P Audio Mixer

Multi-track audio mixing with per-track gain, pan, and mute.

---

## Typical Workflows

### Music Video (Full Pipeline)

```
VHS_LoadAudio
    ├── S42P Beat Analyzer → envelope + beat_frames
    │       ↓
    │   S42P Audio Reactive FX ← VHS_LoadVideo → S42P BG Remover ← result_mask
    │                          ← result_rgb
    │
    ├── S42P Audio Visualizer → bg_image
    │
    └── S42P Procedural FX (bg_transparent=True) → fx_image
                    ↓
            S42P Layer Composer → VHS_VideoCombine
```

### AI Song → Music Video

```
OllamaConnectivity
    ↓
S42P Songwriter → style_tags + lyrics
    ↓
TextEncodeAceStepAudio1.5 → AceStep Sampler → VHS_SaveAudio
    ↓
VHS_LoadAudio → S42P Audio Visualizer → S42P Layer Composer → VHS_VideoCombine
```

### Compositing with Procedural FX

```
S42P FX • Starfield (bg_transparent=False) → bg_image
    ↓
S42P FX • Gold Dust (bg_transparent=True)  → fx_image
    ↓
VHS_LoadVideo → S42P BG Remover → fg_image + fg_mask
    ↓
S42P Layer Composer → result
```

---

## Troubleshooting

**Songwriter fails / "ollama not found"**  
Install ComfyUI-Ollama and the ollama Python package. See Installation Step 3.

**Songwriter outputs `(No lyrics — model did not follow format)`**  
Check the `raw_response` output pin. Try a different model (`qwen2.5:7b` is reliable).

**Background Remover falls back to Chroma Key**  
Install `rembg` and `onnxruntime`. See Installation Step 4.

**Procedural FX nodes not appearing after upgrade from v3.x**  
Node IDs changed to `S42P_FX_*` prefix. Delete old nodes from workflow and re-add.

**Batch size mismatch with audio**  
Set `batch_size` to match your video frame count. Use `animate_batch=True` to advance frames.

---

## License

MIT — See LICENSE file.  
Author: Studio42 (Willie G)
