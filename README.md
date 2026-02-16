# S42 Production Suite for ComfyUI — v3.0

**Professional audio mastering, AI music generation, background removal, and reactive video production nodes.**

Designed for **AceStep 1.5 Turbo + Wan + ComfyUI Portable** on RTX class hardware.
All DSP runs on **CPU** — your GPU stays free for generation.

---

## What's New in v3.0

| Node | Change |
|------|--------|
| **S42P Songwriter** | Fixed `<think>` block stripping for qwen3/deepseek-r1 reasoning models. Multi-strategy parser. Better error messages. |
| **S42P Audio Reactive FX** | Refactored from Studio42 reference — correct frame output, alpha-aware, full effect suite |
| **S42P Matrix Rain FX** | NEW — animated binary (0/1) rain, audio-reactive, transparent or solid background |
| **S42P Background Remover** | NEW (ported from 24oiduts) — BiRefNet/BEN2/U2Net/Chroma Key/Luma Key |
| **S42P Layer Composer** | NEW (ported from 24oiduts) — bg/fx/fg layer stack with blend modes |
| **Audio Loader** | **REMOVED** — use your preferred audio loader (VHS_LoadAudio recommended) |

---

## Installation

### Step 1 — Clone
```
ComfyUI/custom_nodes/S42-Production-Suite/
```

### Step 2 — Install dependencies
```batch
# Windows ComfyUI Portable
.\python_embeded\python.exe -m pip install -r custom_nodes\S42-Production-Suite\requirements.txt
```
```bash
# Linux/Mac ComfyUI Portable
./python_embeds/bin/python -m pip install -r custom_nodes/S42-Production-Suite/requirements.txt
```

### Step 3 — Optional: AI background removal
```bash
pip install rembg onnxruntime          # for U2Net, ISNet, BiRefNet (rembg)
pip install onnxruntime-gpu            # GPU acceleration for rembg
pip install transformers torchvision   # BiRefNet via HuggingFace (highest quality)
pip install ben2                       # BEN2 model
```
Without these, Background Remover falls back to Chroma Key (OpenCV only).

### Step 4 — Restart ComfyUI

All nodes appear under **"S42 Production Suite"** in the node menu.

---

## Nodes

### 🎵 S42P Songwriter
Ollama-powered song generation for AceStep 1.5 Turbo.

**Fixed in v4.1:**
- `<think>...</think>` reasoning blocks auto-stripped (qwen3, deepseek-r1, etc.)
- Multi-strategy parser: delimited blocks → style heuristics → lyrics heuristics → fallback
- If lyrics show `(No lyrics — model did not follow format)`, check `raw_response` output

**Models:**
- Text (fast): `qwen2.5:7b`, `qwen3:8b`, `mistral:7b`, `llama3.1:8b`
- Vision (with image): `llava:13b`, `qwen2.5vl:7b`, `minicpm-v:latest`
- Thinking (auto-handled): `qwen3:8b`, `qwen3:14b`, `deepseek-r1:8b`

**Outputs:** `style_tags` + `lyrics` → `TextEncodeAceStepAudio1.5`

---

### 🎵 S42P Audio Reactive FX
Apply audio-driven effects to any image or video frame batch.

**Effects:**
| Effect | Description |
|--------|-------------|
| `pulse` | Scale/zoom with envelope |
| `neon` | Glowing edge highlight |
| `cascade` | Echo copies offset by envelope |
| `flicker` | Color flash overlay |
| `rotation` | Frame rotation by envelope |
| `warp` | Radial ripple distortion |

**Key inputs:**
- `mask` — restrict effects to foreground only (from BG Remover)
- `external_envelope` + `external_beats` — from S42P Beat Analyzer/Audio Analyser

---

### 🟩 S42P Matrix Rain FX
Animated binary (0/1) digital rain effect generator.

**Features:**
- Generates frame batch (pipe to VHS_VideoCombine or Layer Composer)
- `transparent_bg=True` → RGBA output for compositing over other video
- Audio-reactive: bass drives column speed and density
- `char_set`: `binary` (0s and 1s) or `extended` (binary + katakana symbols)
- Fully configurable color (default Matrix green)
- Optional glow blur

**Usage:**
```
VHS_LoadAudio → S42P Matrix Rain FX → S42P Layer Composer (bg_image)
                                    ↗
Subject Video → S42P BG Remover ──→ Layer Composer (fg_image + fg_mask)
```

---

### 🎭 S42P Background Remover
Multi-method background removal for video and images.

**Methods (best to compatible):**
| Method | Requires | Best For |
|--------|----------|----------|
| `birefnet-general` | rembg/transformers | General — best quality |
| `birefnet-portrait` | rembg/transformers | Human portraits |
| `birefnet-massive` | rembg | Highest quality AI |
| `ben2-base` | ben2 or rembg | Confidence-guided matting |
| `u2net` / `u2net_human_seg` | rembg | Classic reliable |
| `isnet-anime` | rembg | Anime characters |
| `chroma_key_professional` | OpenCV | Green/blue screen |
| `luma_key_professional` | OpenCV | Brightness-based |
| `difference_key` | OpenCV | With reference background |
| `hybrid_ai_traditional` | rembg + OpenCV | Best of both |

**Outputs:** `result_rgb` + `result_rgba` + `result_mask` + `info`

Connect `result_mask` → `S42P Audio Reactive FX (mask)` and `S42P Layer Composer (fg_mask)`.

---

### 🎬 S42P Layer Composer
Multi-layer image/video compositor.

**Layer stack (bottom to top):**
1. `bg_image` — background (visualizer, matrix rain, static image)
2. `fx_image` — effects layer (procedural FX, overlays)
3. `fg_image` — foreground subject (background-removed video)

**Per-layer controls:** opacity, blend mode, optional mask

**Blend modes:** normal, multiply, screen, overlay, hard_light, soft_light, dodge, burn, darken, lighten, difference, exclusion, add, subtract

**Batch-aware:** handles mismatched frame counts by cycling shorter batches.

---

### 🎛️ S42P Parametric EQ
8-band parametric equalizer. Types: High Pass, Low Pass, Low Shelf, High Shelf, Peak, Notch.

### 📊 S42P Dynamics Processor
3-band multiband compressor + glue compressor + output limiter.

### 🎚️ S42P Mastering Chain
Full mastering: EQ → Stereo Width → Multiband → Limiter → LUFS normalization.

### 🥁 S42P Beat Analyzer
Beat detection, BPM tracking, onset detection. Outputs `envelope` + `beat_frames` tensors.

### 🎚️ S42P Audio Analyser
RMS envelope + beat detection. `envelope` + `beat_frames` → Audio Reactive FX.

### 🎬 S42P Audio Visualizer
11 animated visualizer modes: `spectrum_classic`, `lava_lamp`, `wormhole`, `matrix`, `stargate`, `oscilloscope`, `nebula`, `aurora`, `fractal_zoom`, `dna_helix`, `liquid_metal`.

### ✨ S42P Latent Enhancer
Spectral balance correction in latent space before VAE decode.

### 🎵 S42P EQ Preset Loader
AceStep-specific EQ correction presets.

### 🎬 S42P Keyframe Animator
Keyframe-driven transform animation (scale, translate, rotate, opacity).

### 🔄 S42P Transition
Video transitions: cut, dissolve, wipe, zoom, slide.

### 🛠️ S42P Video Tools
Resize, crop, pad, flip, speed change, frame extraction.

### 🎨 S42P Color Grade
LUT application, curves, HSL, temperature, vignette.

---

## Typical Workflows

### Music Video
```
VHS_LoadAudio
    ├── S42P Matrix Rain FX ──────────────┐
    │                                     │
    └── VHS_LoadVideo                     │
            └── S42P BG Remover           │
                    ├── result_mask ──┐   │
                    └── result_rgb   │   │
                            └── S42P Audio Reactive FX ──┐
                                        │               │
                                        └─ S42P Layer Composer ← ┘
                                                └── VHS_VideoCombine
```

### Full AceStep + Mastering
```
S42P Songwriter (style_tags, lyrics)
    └── TextEncodeAceStepAudio1.5
            └── AceStep Sampler
                    └── S42P Mastering Chain
                            └── S42P Audio Analyser
                                    └── SaveAudio
```

---

## Troubleshooting

**Songwriter outputs `(No lyrics — model did not follow format)`**
→ Connect `raw_response` → ShowText to see what the model returned.
→ Try a different model — some don't follow strict formatting instructions.
→ qwen3 models need the `<think>` stripping that v4.1 adds — update if on older version.

**Background Remover shows `rembg not available`**
→ `pip install rembg onnxruntime`
→ Traditional methods (chroma key, luma key) still work without rembg.

**Audio Reactive FX outputs wrong frame count**
→ The node outputs exactly as many frames as `images` input — this is correct.
→ If you see 1 frame, your input IMAGE was 1 frame.

**Matrix Rain looks blocky**
→ Reduce `char_size` (smaller cells) or enable `glow=True`.
→ The node uses pixel blocks rather than font rendering to avoid font dependencies.

---

## Requirements

```
scipy>=1.11.0
numpy>=1.24.0
opencv-python>=4.8.0    # NEW in v3.0 — required for BG Remover, FX, Layer Composer
pyloudnorm>=0.1.1
librosa>=0.10.0
```

Optional (AI background removal):
```
rembg>=2.0.50
onnxruntime>=1.16.0     # or onnxruntime-gpu
transformers            # BiRefNet via HuggingFace
torchvision             # BiRefNet via HuggingFace
ben2                    # BEN2 model
```

---

*S42 Production Suite | Python 3.12 | ComfyUI Portable*
