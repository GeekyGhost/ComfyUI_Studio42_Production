# S42 Production Suite for ComfyUI

**Professional audio mastering, analysis, and production nodes.**
*Phase 1: Audio Mastering Chain*

---

## What This Is

A brand-new ComfyUI custom node suite focused on professional audio production.
Built specifically for:
- **Music video production** — AceStep → master → beat-sync → Wan video
- **Talking head / lip-sync** — TTS → EQ → master → LatentSync
- **Full song stem mastering** — per-stem dynamics, final master on mixdown

All DSP runs on **CPU** — your GPU stays free for Wan and Qwen generation.
Designed for **ComfyUI Portable + Python 3.12** on RTX 4090 (laptop).

---

## Installation

### Step 1 — Clone into custom_nodes
```
ComfyUI/custom_nodes/Studio42-Production/
```

### Step 2 — Install dependencies (ComfyUI Portable, Windows)
```batch
.\python_embeded\python.exe -m pip install -r custom_nodes\Studio42-Production\requirements.txt
```

**Linux/Mac Portable:**
```bash
./python_embeds/bin/python -m pip install -r custom_nodes/Studio42-Production/requirements.txt
```

### Step 3 — Restart ComfyUI

Nodes appear under **"S42 Production Suite"** in the node menu.

---

## Nodes

### 🎛️ S42P Parametric EQ
**8-band parametric equalizer** — the foundation of any mastering chain.

| Band | Default Purpose | Default Type |
|------|----------------|--------------|
| B1   | Low-cut / rumble removal | High Pass @ 80Hz |
| B2   | Bass body control | Low Shelf @ 120Hz |
| B3   | Low-mid muddiness | Peak @ 300Hz |
| B4   | Midrange presence | Peak @ 1kHz |
| B5   | Upper-mid clarity | Peak @ 3kHz |
| B6   | Presence/air | Peak @ 5kHz |
| B7   | High shelf air | High Shelf @ 8kHz |
| B8   | High-cut / noise | Low Pass @ 20kHz |

Each band has: `type` / `frequency` / `gain` / `Q` / `enabled` controls.
All controls have hover tooltips explaining what they do.

Outputs: `AUDIO` + `eq_info` (JSON — peak/RMS levels, bands applied)

---

### 🔊 S42P Dynamics Processor
**3-band multiband compressor** with broadband glue option.

Signal flow:
```
Input → [Glue Compressor (optional)] → [LR-4 Crossover Split] →
    Low Band  ─┐
    Mid Band   ├─ [Summed] → [Output Limiter] → Output
    High Band  ┘
```

Per-band controls: `mode` (compress/expand/gate/bypass), `threshold`, `ratio`,
`attack`, `release`, `knee`, `makeup gain`.

Optional `sidechain_audio` input — classic duck-music-under-voice setup.

Outputs: `AUDIO` + `dynamics_info` (JSON — input/output levels, settings)

---

### 🎚️ S42P Mastering Chain
**Final mastering stage.** Place last before export.

Signal flow:
```
Input → Stereo Width (M/S) → Harmonic Exciter →
        LUFS Targeting → True Peak Limiter → Dither → Output
```

**LUFS Presets:**
| Platform | Target |
|----------|--------|
| Spotify / Apple Music | -14 LUFS |
| YouTube / Podcasts | -14 LUFS |
| TikTok / Instagram Reels | -14 LUFS |
| Broadcast / TV (EBU R128) | -23 LUFS |
| CD / Download | Max loudness, no LUFS target |
| Custom | Your value |

Both preset dropdown AND manual override are available.

Outputs: `mastered_audio` + `mastering_report` (JSON — input/output LUFS,
true peak, dynamic range, gain applied, any warnings)

---

### 🥁 S42P Beat Analyzer
**BPM detection, beat grid, onset detection, musical key analysis.**

Pass audio through this node inline — it **does not modify the signal**.

**Three outputs:**
1. `audio` — passthrough, unchanged
2. `beat_data_json` — structured JSON with beat times, bar times, onset times,
   BPM, key, and **pre-calculated frame numbers at your chosen FPS values**
   (for direct use with Wan S2V, LatentSync, and the future Video Sequencer)
3. `beat_visualization` — waveform image with beat/bar/onset markers,
   previewable directly in ComfyUI

**Beat data JSON schema:**
```json
{
  "bpm": 128.0,
  "bpm_confidence": 0.85,
  "time_signature": 4,
  "beat_times": [0.0, 0.469, 0.938, ...],
  "bar_times": [0.0, 1.875, 3.75, ...],
  "onset_times": [0.0, 0.234, 0.469, ...],
  "key": "A minor",
  "key_confidence": 0.72,
  "duration_sec": 180.0,
  "sample_rate": 44100,
  "frames_at_bpm": {
    "24fps": [0, 11, 22, ...],
    "30fps": [0, 14, 28, ...],
    "60fps": [0, 28, 56, ...]
  }
}
```

---

## Recommended Workflow

### Music Video (AceStep → Wan)
```
[AceStep Music] → [S42P Parametric EQ] → [S42P Dynamics Processor]
               → [S42P Mastering Chain] → [S42P Beat Analyzer]
               → [Wan S2V / MMAudio / Video Combine]
```

### Talking Head / Lip-Sync (TTS → LatentSync)
```
[Kokoro TTS] → [S42P Parametric EQ] → [S42P Dynamics Processor]
             → [S42P Mastering Chain] → [Geeky LatentSync / VHS Combine]
```

### Full Song Production
```
Vocal Stem:   [Load Audio] → [S42P EQ] → [S42P Dynamics] →┐
Drum Stem:    [Load Audio] → [S42P EQ] → [S42P Dynamics] →├─ [Geeky AudioMixer]
Bass Stem:    [Load Audio] → [S42P EQ] → [S42P Dynamics] →┘       ↓
                                                         [S42P Mastering Chain]
                                                         [S42P Beat Analyzer]
                                                              ↓
                                                         [Export / Video]
```

---

## Dependencies

| Package | Required? | Purpose |
|---------|-----------|---------|
| `scipy` | **Required** | EQ filters, crossover, resampling |
| `numpy` | **Required** | Already in ComfyUI |
| `pyloudnorm` | Strongly recommended | Accurate LUFS measurement (EBU R128) |
| `librosa` | Strongly recommended | Beat detection, onset, key analysis |
| `Pillow` | Already in ComfyUI | Beat visualization rendering |

Without `pyloudnorm`: LUFS uses RMS estimate (less accurate but functional).
Without `librosa`: Beat detection uses autocorrelation fallback (less accurate).

---

## Phase 2 (Coming Next)

- 🎬 S42P Video Sequencer — multi-clip timeline with beat-snap
- ✂️ S42P Video Transitions — cut, dissolve, wipe, push between clips
- 🎨 S42P Color Grade — LUT loader, curves, HSL, film emulation
- 🎵 S42P Stem Splitter — Demucs integration (vocals/drums/bass/other)
- 📊 S42P Audio Visualizer — technical waveform/spectrum/VU for music videos

---

*S42 Production Suite — built for music video, lip-sync, and full song production.*
*DON'T PANIC. Your audio will sound professional.*

---

## Phase 2 — Animation & Video (Added)

### 🎞️ S42P Keyframe Animator
**2D animation compositor** — animates a layer image over a video frame batch.

**Keyframe String Format:** `"frame:value, frame:value"`
Optional per-keyframe easing: `"frame:value:easing_name"`

| Property | Example | Effect |
|----------|---------|--------|
| `position_x` | `"0:1920, 12:960"` | Slide in from right |
| `position_y` | `"0:540"` | Hold at vertical centre |
| `scale_x` | `"0:0.5, 12:1.0:bounce"` | Bounce scale up |
| `opacity` | `"0:0.0, 8:1.0, 40:1.0, 48:0.0"` | Fade in, hold, fade out |
| `rotation` | `"0:0, 48:360"` | Full spin over 48 frames |

**Available easings:** `linear`, `ease_in`, `ease_out`, `ease_in_out`,
`ease_in_cubic`, `ease_out_cubic`, `ease_in_out_cubic`, `elastic_in`, `elastic_out`, `bounce`

**Typical workflow:**
```
[BG Remover] → MASK  ─┐
[BG Remover] → IMAGE  ─┤→ [Keyframe Animator] → [Transition] → [VHS Video Combine]
[Wan Video]  → IMAGE  ─┘
```

---

### 🎬 S42P Transition
**20 CapCut-style transitions** between two video clips.

| Category | Types |
|----------|-------|
| Basic | `cut`, `dissolve`, `fade_black`, `fade_white` |
| Push | `push_left`, `push_right`, `push_up`, `push_down` |
| Wipe | `wipe_left`, `wipe_right`, `wipe_up`, `wipe_down` |
| Motion | `zoom_in`, `zoom_out`, `spin_cw`, `spin_ccw` |
| Creative | `glitch`, `iris_in`, `slide_up`, `slide_down` |

Output = clip_a (trimmed) + transition frames + clip_b (trimmed).
Chain multiple Transition nodes to build a multi-clip sequence.

---

### 🛠️ S42P Video Tools
**Four modes** via a single mode selector:

- **trim** — cut to in/out points (frame or seconds)
- **concatenate** — join up to 6 clips (auto-matches resolution)
- **reverse** — flip playback direction
- **fps_convert** — e.g. 8fps Wan output → 24fps smooth (blend or duplicate)

---

### 🎨 S42P Color Grade
**Three grading stages**, independently bypassable:

1. **LUT** — load any `.cube` file (film emulation, LOG→REC709, custom looks)
   - `lut_strength` slider lets you blend the LUT (1.0 = full, 0.5 = half)
2. **Curves/Levels** — ASC CDL-style lift/gamma/gain per channel + master contrast + saturation
3. **HSL** — hue/saturation/lightness per colour range (reds, yellows, greens, cyans, blues, magentas)

Signal flow: `LUT → Curves → HSL`

---

## Full Recommended Workflow

### Music Video (AceStep → beat-synced → Wan → graded → export)
```
[AceStep]           → [S42P EQ] → [S42P Dynamics] → [S42P Mastering Chain]
                    → [S42P Beat Analyzer] ─── beat_data_json → [Wan S2V]
[Wan S2V frames]    → [S42P Keyframe Animator] (logo/text overlays)
                    → [S42P Transition] (between scenes)
                    → [S42P Color Grade]
                    → [VHS Video Combine + mastered_audio]
```

### Talking Head (TTS → lip sync → grade → export)
```
[Kokoro TTS]        → [S42P EQ] → [S42P Dynamics] → [S42P Mastering Chain]
[Portrait image]    → [BG Remover] → [Layer Composer] (background plate)
                    → [Geeky LatentSync]
                    → [S42P Color Grade]
                    → [VHS Video Combine]
```

### Multi-clip Edit
```
[Wan clip 1] → [S42P Video Tools: fps_convert 8→24]
             → [S42P Transition: dissolve] ──────────────────┐
[Wan clip 2] → [S42P Video Tools: trim 0s–10s]              │
             → [S42P Transition: push_left] ─────────────────┤
[Wan clip 3] ────────────────────────────────────────────────┘
             → [S42P Color Grade]
             → [VHS Video Combine]
```
