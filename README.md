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
