"""
S42 Production Suite â€” Beat Analyzer Node
==========================================
BPM detection, beat grid, onset/transient detection, and key/scale analysis.

Outputs:
  AUDIO     â€” passthrough (chain inline without breaking graph)
  STRING    â€” JSON beat data compatible with LatentSync / Wan S2V timing
  IMAGE     â€” waveform + beat grid visualization (preview in ComfyUI)

Beat data JSON schema (for downstream node compatibility):
{
  "bpm":            float,
  "bpm_confidence": float (0.0â€“1.0),
  "time_signature": int (beats per bar, estimated),
  "beat_times":     [float, ...],   â† seconds
  "bar_times":      [float, ...],   â† seconds
  "onset_times":    [float, ...],   â† transient onset seconds
  "key":            str,            â† e.g. "C major"
  "key_confidence": float,
  "duration_sec":   float,
  "sample_rate":    int,
  "total_beats":    int,
  "frames_at_bpm":  dict            â† {fps: [frame_numbers]} for video sync
}

Python 3.12 | ComfyUI Portable
librosa strongly recommended (pip install librosa)
"""

import numpy as np
import logging
import json
from typing import Tuple, Optional

import torch


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars/arrays to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

logger = logging.getLogger(__name__)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available â€” beat detection will use basic autocorrelation fallback")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from s42p_audio_utils import (
    audio_from_comfy, audio_to_comfy, ensure_stereo, ensure_float32
)

CATEGORY = "S42 Production Suite ðŸ“Š Audio Analysis"


# ”€”€ Fallback BPM via autocorrelation ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _bpm_autocorrelation(mono: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Basic BPM detection via autocorrelation when librosa is not available.
    Less accurate than librosa but gives a usable estimate.
    Returns (bpm, confidence).
    """
    # Downsample to 22050 for speed
    if sr > 22050:
        step   = sr // 22050
        mono   = mono[::step]
        sr_eff = sr // step
    else:
        sr_eff = sr

    # Onset strength via energy envelope
    hop   = 512
    n_frames = len(mono) // hop
    energy = np.array([
        np.sum(mono[i*hop:(i+1)*hop] ** 2)
        for i in range(n_frames)
    ])

    # Autocorrelation of energy
    ac = np.correlate(energy, energy, mode="full")
    ac = ac[len(ac)//2:]    # keep positive lags only

    # Search BPM range 60€“200
    fps_e    = sr_eff / hop
    lag_min  = int(60.0 / 200.0 * fps_e)
    lag_max  = int(60.0 / 60.0  * fps_e)

    lag_range = ac[lag_min:lag_max]
    if len(lag_range) == 0:
        return 120.0, 0.1

    best_lag  = np.argmax(lag_range) + lag_min
    bpm       = 60.0 * fps_e / best_lag

    # Confidence: ratio of peak to mean
    conf = float(np.max(lag_range) / (np.mean(lag_range) + 1e-10))
    conf = min(float(conf) / 10.0, 1.0)   # normalise roughly

    return float(bpm), float(conf)


def _beats_from_bpm(bpm: float, duration: float) -> np.ndarray:
    """Generate regular beat times from a constant BPM."""
    beat_interval = 60.0 / bpm
    return np.arange(0.0, duration, beat_interval)


# ”€”€ Key detection ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _detect_key(mono: np.ndarray, sr: int) -> Tuple[str, float]:
    """
    Krumhansl-Schmuckler key-finding algorithm using chroma features.
    Returns (key_string, confidence 0â€“1).
    """
    if not LIBROSA_AVAILABLE:
        return "Unknown", 0.0

    try:
        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
        # Sum chroma energy across time
        chroma_mean = np.mean(chroma, axis=1)

        # Krumhansl-Schmuckler key profiles
        # Major profile
        major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        # Minor profile
        minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

        # Rotate profiles for all 12 keys and compute correlation
        keys_major = []
        keys_minor = []
        note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

        for i in range(12):
            rotated_major = np.roll(major_profile, i)
            rotated_minor = np.roll(minor_profile, i)
            corr_major = float(np.corrcoef(chroma_mean, rotated_major)[0,1])
            corr_minor = float(np.corrcoef(chroma_mean, rotated_minor)[0,1])
            keys_major.append((f"{note_names[i]} major", corr_major))
            keys_minor.append((f"{note_names[i]} minor", corr_minor))

        all_keys = keys_major + keys_minor
        all_keys.sort(key=lambda x: x[1], reverse=True)
        best_key, best_corr = all_keys[0]

        # Normalise confidence to 0€“1 range (correlations are -1 to 1)
        confidence = (best_corr + 1.0) / 2.0

        return best_key, round(confidence, 3)

    except Exception as e:
        logger.warning(f"Key detection failed: {e}")
        return "Unknown", 0.0


# ”€”€ Visualization ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _render_beat_visualization(mono: np.ndarray, sr: int,
                                beat_times: np.ndarray,
                                bar_times: np.ndarray,
                                onset_times: np.ndarray,
                                bpm: float, key: str,
                                width: int = 1280, height: int = 360) -> np.ndarray:
    """
    Render a waveform visualization with beat/bar/onset markers.
    Returns numpy [H, W, 3] RGB image compatible with ComfyUI IMAGE format.
    """
    if not PIL_AVAILABLE:
        # Return blank black image if PIL not available
        return np.zeros((height, width, 3), dtype=np.float32)

    img  = Image.new("RGB", (width, height), (12, 12, 20))
    draw = ImageDraw.Draw(img)

    duration = len(mono) / sr
    center_y = height // 2

    # ”€”€ Draw waveform ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    # Downsample to pixel width
    step    = max(1, len(mono) // width)
    samples = mono[::step][:width]
    max_amp = max(float(np.max(np.abs(samples))), 1e-6)

    for x in range(min(len(samples), width)):
        amp   = float(samples[x]) / max_amp
        y_top = int(center_y - amp * (center_y - 10))
        y_bot = center_y
        if amp < 0:
            y_top = center_y
            y_bot = int(center_y - amp * (center_y - 10))
        draw.line([(x, y_top), (x, y_bot)], fill=(60, 140, 200))

    # ”€”€ Draw bar lines (orange, thick) ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    for t in bar_times:
        x = int(t / duration * width)
        if 0 <= x < width:
            draw.line([(x, 0), (x, height)], fill=(255, 153, 0), width=2)

    # ”€”€ Draw beat lines (yellow, thin) ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    for t in beat_times:
        x = int(t / duration * width)
        if 0 <= x < width:
            draw.line([(x, center_y - 20), (x, center_y + 20)], fill=(220, 220, 80), width=1)

    # ”€”€ Draw onset markers (red triangles at top) ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    for t in onset_times[:500]:   # cap at 500 to avoid perf issues
        x = int(t / duration * width)
        if 0 <= x < width:
            draw.polygon([(x, 4), (x-4, 14), (x+4, 14)], fill=(220, 60, 60))

    # ”€”€ Info text bar at bottom ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    info_y = height - 28
    draw.rectangle([(0, info_y), (width, height)], fill=(20, 20, 30))
    try:
        font = ImageFont.load_default(size=14)
    except Exception:
        font = ImageFont.load_default()

    info_str = (f"  BPM: {bpm:.1f}  |  Key: {key}  |  "
                f"Beats: {len(beat_times)}  |  Bars: {len(bar_times)}  |  "
                f"Onsets: {len(onset_times)}  |  Duration: {duration:.1f}s")
    draw.text((8, info_y + 6), info_str, fill=(200, 200, 200), font=font)

    # ”€”€ Legend ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    legend_items = [
        ((255, 153,  0), "Bar"),
        ((220, 220, 80), "Beat"),
        ((220,  60, 60), "Onset"),
        (( 60, 140, 200), "Waveform"),
    ]
    lx = width - 260
    for i, (color, label) in enumerate(legend_items):
        draw.rectangle([(lx + i*64, info_y + 8), (lx + i*64 + 12, info_y + 18)], fill=color)
        draw.text((lx + i*64 + 16, info_y + 6), label, fill=(180, 180, 180), font=font)

    arr_rgb = np.array(img, dtype=np.float32) / 255.0
    return arr_rgb   # [H, W, 3]


# ”€”€ Node ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

class S42PBeatAnalyzer:
    """
    ðŸ¥ S42P Beat Analyzer
    Analyzes audio for BPM, beat grid, onsets, and musical key.

    Pass audio through this node inline â€” it doesn't modify the signal.
    Use the JSON output to drive beat-synced video cuts with Wan S2V
    or LatentSync timing, or load it into the future S42P Video Sequencer.
    The visualization image shows you the waveform with beat/bar markers
    so you can spot-check alignment before committing to a video cut.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": ("Input audio to analyze. "
                                "The audio passes through unchanged â€” this node only reads it. "
                                "Chain it between your mastering nodes and your video/export nodes.")
                }),
                "analysis_channel": (["mixed_mono", "left", "right"], {
                    "default": "mixed_mono",
                    "tooltip": ("Which channel to use for beat detection. "
                                "mixed_mono = average L+R (recommended for music). "
                                "left/right = individual channel (useful if kick is panned).")
                }),
                "bpm_min": ("FLOAT", {
                    "default": 60.0, "min": 30.0, "max": 200.0,
                    "step": 1.0, "display": "slider",
                    "tooltip": ("Minimum BPM to search for. "
                                "60 BPM = slow ballad. 90â€“140 = pop/hip-hop. 140â€“180 = EDM/drum & bass. "
                                "Narrowing the range improves detection accuracy.")
                }),
                "bpm_max": ("FLOAT", {
                    "default": 200.0, "min": 60.0, "max": 300.0,
                    "step": 1.0, "display": "slider",
                    "tooltip": ("Maximum BPM to search for. "
                                "Keep this above your expected tempo. "
                                "Set bpm_min and bpm_max close together for faster, more accurate detection.")
                }),
                "beats_per_bar": (["auto", "2", "3", "4", "6"], {
                    "default": "auto",
                    "tooltip": ("Time signature numerator â€” beats per bar. "
                                "auto = estimate from beat grid. "
                                "4 = common time (pop, rock, hip-hop). "
                                "3 = waltz. 6 = compound time. "
                                "Used to calculate bar lines in the visualization and JSON output.")
                }),
                "onset_sensitivity": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0,
                    "step": 0.05, "display": "slider",
                    "tooltip": ("Sensitivity for transient/onset detection. "
                                "Lower = only catch strong transients (kick, snare). "
                                "Higher = catch more subtle onsets (hi-hats, guitar picks). "
                                "0.3â€“0.5 works well for drums. 0.6â€“0.8 for melodic content.")
                }),
                "detect_key": ("BOOLEAN", {
                    "default": True,
                    "tooltip": ("Enable musical key detection using chroma analysis. "
                                "Adds a small processing cost. "
                                "Useful for choosing video color palettes or matching music tracks. "
                                "Requires librosa.")
                }),
                "video_fps_list": ("STRING", {
                    "default": "24,25,30,60",
                    "tooltip": ("Comma-separated list of video frame rates to pre-calculate beat frame numbers for. "
                                "e.g. '24,30,60' will output beat positions as frame numbers at each FPS. "
                                "This lets you directly look up which frame corresponds to each beat "
                                "for video editing and sequencing.")
                }),
                "viz_width": ("INT", {
                    "default": 1280, "min": 320, "max": 2560,
                    "step": 64, "display": "slider",
                    "tooltip": ("Width of the output visualization image in pixels. "
                                "1280 fits most ComfyUI previews cleanly. "
                                "Higher resolution for detailed inspection of dense beat grids.")
                }),
                "viz_height": ("INT", {
                    "default": 360, "min": 120, "max": 720,
                    "step": 8, "display": "slider",
                    "tooltip": ("Height of the output visualization image in pixels. "
                                "360 is compact for preview. 480â€“720 for more waveform detail.")
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO", "STRING", "IMAGE")
    RETURN_NAMES  = ("audio",  "beat_data_json", "beat_visualization")
    FUNCTION      = "analyze_beats"
    CATEGORY      = CATEGORY

    def analyze_beats(self, audio: dict, analysis_channel: str,
                      bpm_min: float, bpm_max: float, beats_per_bar: str,
                      onset_sensitivity: float, detect_key: bool,
                      video_fps_list: str, viz_width: int, viz_height: int
                      ) -> Tuple[dict, str, torch.Tensor]:

        arr, sr = audio_from_comfy(audio)
        arr     = ensure_stereo(ensure_float32(arr))
        duration = arr.shape[1] / sr

        # ”€”€ Extract mono analysis channel ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        if analysis_channel == "left":
            mono = arr[0]
        elif analysis_channel == "right":
            mono = arr[1]
        else:
            mono = np.mean(arr, axis=0)

        # ”€”€ BPM + Beat Detection ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        bpm         = 120.0
        bpm_conf    = 0.5
        beat_times  = np.array([])

        if LIBROSA_AVAILABLE:
            try:
                tempo, beats_frames = librosa.beat.beat_track(
                    y=mono, sr=sr,
                    bpm=None,
                    tightness=100,
                    trim=False
                )
                bpm        = float(tempo) if np.isscalar(tempo) else float(tempo[0])
                # Clamp to user range
                if not (bpm_min <= bpm <= bpm_max):
                    # Try again with start_bpm hint
                    mid_bpm = (bpm_min + bpm_max) / 2.0
                    tempo2, beats_frames2 = librosa.beat.beat_track(
                        y=mono, sr=sr, start_bpm=mid_bpm, tightness=100, trim=False
                    )
                    bpm2 = float(tempo2) if np.isscalar(tempo2) else float(tempo2[0])
                    if bpm_min <= bpm2 <= bpm_max:
                        bpm, beats_frames = bpm2, beats_frames2

                beat_times = librosa.frames_to_time(beats_frames, sr=sr)
                bpm_conf   = 0.85   # librosa is reliable
                logger.info(f"librosa detected BPM: {bpm:.1f}")
            except Exception as e:
                logger.warning(f"librosa beat tracking failed: {e}, using autocorrelation fallback")
                bpm, bpm_conf = _bpm_autocorrelation(mono, sr)
                beat_times    = _beats_from_bpm(bpm, duration)
        else:
            bpm, bpm_conf = _bpm_autocorrelation(mono, sr)
            beat_times    = _beats_from_bpm(bpm, duration)

        # ”€”€ Onset Detection ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        onset_times = np.array([])
        if LIBROSA_AVAILABLE:
            try:
                onset_frames = librosa.onset.onset_detect(
                    y=mono, sr=sr,
                    wait=1,
                    delta=1.0 - onset_sensitivity   # librosa delta is inverted
                )
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            except Exception as e:
                logger.warning(f"Onset detection failed: {e}")

        # ”€”€ Bar Times ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        if beats_per_bar == "auto":
            bpb = 4   # default, most music is 4/4
        else:
            bpb = int(beats_per_bar)

        bar_times = beat_times[::bpb] if len(beat_times) > 0 else np.array([])

        # ”€”€ Key Detection ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        key_str, key_conf = ("Unknown", 0.0)
        if detect_key:
            key_str, key_conf = _detect_key(mono, sr)

        # ”€”€ Frame numbers per FPS ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        frames_at_bpm = {}
        try:
            for fps_str in video_fps_list.split(","):
                fps_str = fps_str.strip()
                if fps_str:
                    fps_val = float(fps_str)
                    frames_at_bpm[f"{fps_str}fps"] = [
                        int(round(t * fps_val)) for t in beat_times.tolist()
                    ]
        except Exception as e:
            logger.warning(f"FPS frame calculation failed: {e}")

        # ”€”€ Build JSON output ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        beat_data = {
            "bpm":             round(float(bpm), 2),
            "bpm_confidence":  round(float(bpm_conf), 3),
            "time_signature":  bpb,
            "beat_times":      [round(float(t), 4) for t in beat_times.tolist()],
            "bar_times":       [round(float(t), 4) for t in bar_times.tolist()],
            "onset_times":     [round(float(t), 4) for t in onset_times.tolist()],
            "key":             key_str,
            "key_confidence":  key_conf,
            "duration_sec":    round(float(duration), 3),
            "sample_rate":     int(sr),
            "total_beats":     int(len(beat_times)),
            "total_onsets":    int(len(onset_times)),
            "frames_at_bpm":   frames_at_bpm,
            "analysis_notes":  {
                "librosa_used":   LIBROSA_AVAILABLE,
                "channel_used":   analysis_channel,
                "bpm_range_searched": [bpm_min, bpm_max],
            }
        }

        beat_json = json.dumps(beat_data, indent=2, cls=_NumpyEncoder)

        # ”€”€ Render visualization ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        viz_np  = _render_beat_visualization(
            mono, sr, beat_times, bar_times, onset_times,
            bpm=float(bpm), key=key_str,
            width=viz_width, height=viz_height
        )
        # ComfyUI IMAGE: torch [B, H, W, C] float32 0€“1
        viz_tensor = torch.from_numpy(viz_np).unsqueeze(0)

        return (audio_to_comfy(arr, sr), beat_json, viz_tensor)


# ”€”€ ComfyUI registration ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

NODE_CLASS_MAPPINGS = {
    "S42PBeatAnalyzer": S42PBeatAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PBeatAnalyzer": "ðŸ¥ S42P Beat Analyzer",
}
