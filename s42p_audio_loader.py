"""
S42 Production Suite — Audio Loader v3.2
==========================================
FIXED: Audio upload button and file selection dropdown.

This version dynamically scans the ComfyUI input folder so that 
uploaded files appear in the dropdown menu and the 'Upload' button 
is correctly rendered by the ComfyUI frontend.

Python 3.12 | ComfyUI Portable
"""

from __future__ import annotations

import json
import logging
import os
import math

import numpy as np
import torch

logger = logging.getLogger(__name__)
CATEGORY = "S42 Production Suite 🎵 Audio"

# --- Backend Availability Checks ---
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aiff", ".aif", ".wma", ".opus")
CHANNEL_MODES    = ["stereo", "mono", "preserve"]

# ── Helper Functions ──────────────────────────────────────────────────────────

def _resolve_path(audio_file: str) -> str:
    if os.path.isabs(audio_file) and os.path.isfile(audio_file):
        return audio_file
    if FOLDER_PATHS_AVAILABLE:
        inp = folder_paths.get_input_directory()
        cand = os.path.join(inp, audio_file)
        if os.path.isfile(cand):
            return cand
        # Deep search for files in subfolders
        base = os.path.basename(audio_file)
        for root, _, files in os.walk(inp):
            if base in files:
                return os.path.join(root, base)
    return audio_file

def _load_audio(filepath: str, target_sr: int = 0):
    if TORCHAUDIO_AVAILABLE:
        try:
            wav, sr = torchaudio.load(filepath)
            if target_sr > 0 and target_sr != sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
                sr  = target_sr
            return wav.float(), int(sr)
        except Exception as e:
            logger.warning(f"[S42P AudioLoader] torchaudio: {e}")

    if SOUNDFILE_AVAILABLE:
        try:
            data, sr = sf.read(filepath, dtype="float32")
            data = data[np.newaxis, :] if data.ndim == 1 else data.T
            wav  = torch.from_numpy(data.copy())
            if target_sr > 0 and target_sr != sr and TORCHAUDIO_AVAILABLE:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
                sr  = target_sr
            return wav.float(), int(sr)
        except Exception as e:
            raise RuntimeError(f"soundfile: {e}") from e

    raise RuntimeError("No audio backend found. Ensure torchaudio or soundfile is installed.")

def _analyse(wav: torch.Tensor, sr: int, enhanced: bool) -> dict:
    mono  = wav.mean(0)
    n     = mono.shape[0]
    peak  = float(mono.abs().max())
    rms   = float(mono.pow(2).mean().sqrt())
    info  = {
        "duration_sec":     round(n / max(sr, 1), 3),
        "sample_rate":      sr,
        "channels":         int(wav.shape[0]),
        "peak_db":          round(20 * math.log10(max(peak, 1e-10)), 2),
        "rms_db":           round(20 * math.log10(max(rms,  1e-10)), 2),
    }
    if enhanced and LIBROSA_AVAILABLE:
        try:
            m = mono.numpy()
            tempo, _ = librosa.beat.beat_track(y=m, sr=sr)
            info["estimated_bpm"] = round(float(tempo[0] if isinstance(tempo, np.ndarray) else tempo), 1)
        except Exception: pass
    return info

# ── ComfyUI Node ──────────────────────────────────────────────────────────────

class S42PAudioLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Scan the input directory for existing audio files
        files = []
        if FOLDER_PATHS_AVAILABLE:
            input_dir = folder_paths.get_input_directory()
            if os.path.exists(input_dir):
                files = [f for f in os.listdir(input_dir) 
                         if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(AUDIO_EXTENSIONS)]
        
        # Fallback if no files found
        if not files:
            files = ["upload_audio_here.wav"]

        return {
            "required": {
                # Using a list (COMBO) with "audio_upload": True triggers the UI button
                "audio_file": (sorted(files), {"audio_upload": True}),
                "channel_mode": (CHANNEL_MODES, {"default": "stereo"}),
                "target_sr": ([0, 22050, 44100, 48000], {"default": 44100}),
            },
            "optional": {
                "trim_silence": ("BOOLEAN", {"default": False}),
                "enhanced_analysis": ("BOOLEAN", {"default": False}),
                "gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5}),
            }
        }

    RETURN_TYPES  = ("AUDIO", "STRING", "INT", "FLOAT")
    RETURN_NAMES  = ("audio", "analysis_json", "sample_rate", "duration_sec")
    FUNCTION      = "load_audio"
    CATEGORY      = CATEGORY

    @classmethod
    def IS_CHANGED(cls, audio_file, **kw):
        try:
            p = _resolve_path(audio_file)
            return str(os.path.getmtime(p))
        except: return audio_file

    def load_audio(self, audio_file: str, channel_mode: str, target_sr: int,
                   trim_silence: bool = False, enhanced_analysis: bool = False,
                   gain_db: float = 0.0):

        fp = _resolve_path(audio_file)
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Audio file not found: {fp}")

        wav, sr = _load_audio(fp, target_sr)
        
        # Apply Channel Mode
        if channel_mode == "mono":
            wav = wav.mean(0, keepdim=True)
        elif channel_mode == "stereo":
            wav = wav.repeat(2, 1) if wav.shape[0] == 1 else wav[:2]

        # Apply Gain
        if abs(gain_db) > 0.01:
            wav = wav * (10.0 ** (gain_db / 20.0))

        # Peak Normalization (Safety)
        peak = float(wav.abs().max())
        if peak > 1.0:
            wav = wav / peak

        info = _analyse(wav, sr, enhanced_analysis)

        return (
            {"waveform": wav.unsqueeze(0), "sample_rate": sr},
            json.dumps(info, indent=2),
            sr,
            float(info["duration_sec"]),
        )

NODE_CLASS_MAPPINGS        = {"S42PAudioLoader": S42PAudioLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"S42PAudioLoader": "S42P Audio Loader 🎵"}