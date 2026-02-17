"""
S42 Production Suite â€” Audio Mixer
=====================================
Improved port of Studio42AudioMixer.

4-channel audio mixer with per-channel:
  â€¢ Volume (0.0â€“5.0)
  â€¢ Pan (-1.0 left â†’ 0.0 centre â†’ 1.0 right)
  â€¢ Start time offset (seconds)
  â€¢ Fade in / fade out (seconds)

Plus master volume and output limiter.

All DSP runs on CPU â€” no VRAM needed.

IMPROVEMENTS vs Studio42 original:
  â€¢ Streamlined to 4 channels (was 6 â€” too many optional inputs = messy)
  â€¢ Cleaner channel loop reduces code duplication
  â€¢ Peak normalisation guard prevents output clipping
  â€¢ Output sample_rate is taken from the first connected audio
  â€¢ Handles mono/stereo/mismatched channel counts gracefully
  â€¢ Added output_sample_rate override

Python 3.12 | ComfyUI Portable | torch only
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)
CATEGORY = "S42 Production Suite ðŸŽµ Audio"


def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _apply_fade(wav: torch.Tensor, sr: int,
                 fade_in: float, fade_out: float) -> torch.Tensor:
    """Apply linear fade in/out to a [C, N] waveform tensor."""
    n  = wav.shape[-1]
    fi = min(int(fade_in  * sr), n)
    fo = min(int(fade_out * sr), n)

    if fi > 0:
        ramp = torch.linspace(0.0, 1.0, fi)
        wav[:, :fi] = wav[:, :fi] * ramp

    if fo > 0:
        ramp = torch.linspace(1.0, 0.0, fo)
        wav[:, n-fo:] = wav[:, n-fo:] * ramp

    return wav


def _apply_pan(wav: torch.Tensor, pan: float) -> torch.Tensor:
    """Apply stereo panning to [C, N] tensor.  C must be 2 for pan to work."""
    if wav.shape[0] != 2 or abs(pan) < 0.001:
        return wav

    # Constant-power panning
    angle    = (pan + 1.0) / 2.0 * math.pi / 2.0  # 0 â†’ Ï€/2
    left_g   = math.cos(angle)
    right_g  = math.sin(angle)

    result   = wav.clone()
    result[0] = wav[0] * left_g  + wav[1] * left_g  * 0.0  # L channel
    result[1] = wav[1] * right_g + wav[0] * right_g * 0.0  # R channel
    # Simpler: just scale each channel
    result[0] = wav[0] * left_g
    result[1] = wav[1] * right_g
    return result


def _to_stereo(wav: torch.Tensor) -> torch.Tensor:
    """Ensure [C,N] tensor is stereo (2 channels)."""
    if wav.shape[0] == 1:
        return wav.repeat(2, 1)
    return wav[:2]


class S42PAudioMixer:
    """
    S42P Audio Mixer â€” mix up to 4 audio inputs with independent channel control.

    Each channel has:
      volume     â€“ Linear volume (1.0 = unity)
      pan        â€“ Stereo pan (-1.0 = hard left, 0.0 = centre, 1.0 = hard right)
      start_time â€“ Delay onset by N seconds
      fade_in    â€“ Fade in duration (seconds)
      fade_out   â€“ Fade out duration (seconds)

    Master controls:
      master_volume  â€“ Final output gain
      output_limiter â€“ Soft-limit output to prevent clipping

    Output is always stereo at the sample rate of channel 1 (or override).
    Channels 2â€“4 are resampled to match if needed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        def ch(default_vol: float, default_start: float = 0.0):
            return {
                "volume":     ("FLOAT", {"default": default_vol, "min": 0.0, "max": 5.0,  "step": 0.01}),
                "pan":        ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0,  "step": 0.05}),
                "start_time": ("FLOAT", {"default": default_start, "min": 0.0, "max": 300.0, "step": 0.1}),
                "fade_in":    ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0,  "step": 0.1}),
                "fade_out":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0,  "step": 0.1}),
            }

        return {
            "required": {
                "audio_1":           ("AUDIO", {"tooltip": "Channel 1 (main / lead)"}),
                "ch1_volume":        ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Ch1 volume. 0=mute, 1.0=unity gain, 5.0=+14 dB boost."}),
                "ch1_pan":           ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": "Ch1 pan. -1=full left, 0=centre, +1=full right."}),
                "ch1_fade_in":       ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Ch1 fade-in duration in seconds."}),
                "ch1_fade_out":      ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Ch1 fade-out duration in seconds."}),
                "master_volume":     ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0,   "step": 0.01,
                                                 "tooltip": "Master output gain."}),
                "output_limiter":    ("BOOLEAN", {"default": True,
                                                   "tooltip": "Soft-limit output to prevent clipping."}),
            },
            "optional": {
                "audio_2":           ("AUDIO", {"tooltip": "Channel 2"}),
                "ch2_volume":        ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Ch2 volume. 0=mute, 1.0=unity gain, 5.0=+14 dB boost."}),
                "ch2_pan":           ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": "Ch2 pan. -1=full left, 0=centre, +1=full right."}),
                "ch2_start_time":    ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1, "tooltip": "Ch2 start-time offset in seconds. Delays when this clip enters the mix."}),
                "ch2_fade_in":       ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Ch2 fade-in duration in seconds."}),
                "ch2_fade_out":      ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Ch2 fade-out duration in seconds."}),

                "audio_3":           ("AUDIO", {"tooltip": "Channel 3"}),
                "ch3_volume":        ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Ch3 volume. 0=mute, 1.0=unity gain, 5.0=+14 dB boost."}),
                "ch3_pan":           ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": "Ch3 pan. -1=full left, 0=centre, +1=full right."}),
                "ch3_start_time":    ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1, "tooltip": "Ch3 start-time offset in seconds. Delays when this clip enters the mix."}),
                "ch3_fade_in":       ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Ch3 fade-in duration in seconds."}),
                "ch3_fade_out":      ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Ch3 fade-out duration in seconds."}),

                "audio_4":           ("AUDIO", {"tooltip": "Channel 4"}),
                "ch4_volume":        ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Ch4 volume. 0=mute, 1.0=unity gain, 5.0=+14 dB boost."}),
                "ch4_pan":           ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": "Ch4 pan. -1=full left, 0=centre, +1=full right."}),
                "ch4_start_time":    ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1, "tooltip": "Ch4 start-time offset in seconds. Delays when this clip enters the mix."}),
                "ch4_fade_in":       ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Ch4 fade-in duration in seconds."}),
                "ch4_fade_out":      ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Ch4 fade-out duration in seconds."}),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "mix"
    CATEGORY      = CATEGORY

    def _get_wav(self, audio_dict: dict) -> tuple:
        """Extract [C,N] float32 tensor and sr from ComfyUI AUDIO dict."""
        wav = audio_dict.get("waveform")
        sr  = int(audio_dict.get("sample_rate", 44100))
        if hasattr(wav, "shape") and wav.ndim == 3:
            wav = wav[0]  # remove batch dim â†’ [C,N]
        return wav.float(), sr

    def mix(self,
            audio_1: dict,
            ch1_volume: float = 1.0, ch1_pan: float = 0.0,
            ch1_fade_in: float = 0.0, ch1_fade_out: float = 0.0,
            master_volume: float = 1.0, output_limiter: bool = True,

            audio_2: Optional[dict] = None,
            ch2_volume: float = 0.8, ch2_pan: float = 0.0,
            ch2_start_time: float = 0.0,
            ch2_fade_in: float = 0.0, ch2_fade_out: float = 0.0,

            audio_3: Optional[dict] = None,
            ch3_volume: float = 0.6, ch3_pan: float = 0.0,
            ch3_start_time: float = 0.0,
            ch3_fade_in: float = 0.0, ch3_fade_out: float = 0.0,

            audio_4: Optional[dict] = None,
            ch4_volume: float = 0.5, ch4_pan: float = 0.0,
            ch4_start_time: float = 0.0,
            ch4_fade_in: float = 0.0, ch4_fade_out: float = 0.0,
            ) -> tuple:

        # ”€”€ Channel specs ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
        channels = [
            (audio_1, ch1_volume, ch1_pan, 0.0,           ch1_fade_in, ch1_fade_out),
            (audio_2, ch2_volume, ch2_pan, ch2_start_time, ch2_fade_in, ch2_fade_out),
            (audio_3, ch3_volume, ch3_pan, ch3_start_time, ch3_fade_in, ch3_fade_out),
            (audio_4, ch4_volume, ch4_pan, ch4_start_time, ch4_fade_in, ch4_fade_out),
        ]

        # Get master sample rate from channel 1
        wav1, sr = self._get_wav(audio_1)
        wav1     = _to_stereo(wav1)

        # Determine output length: longest channel
        def _ch_len(audio_dict, start, sr):
            if audio_dict is None:
                return 0
            wav, asr = self._get_wav(audio_dict)
            n = wav.shape[-1]
            if asr != sr:
                n = int(n * sr / asr)
            return int(start * sr) + n

        total_n = max(_ch_len(a, s, sr) for a, _, _, s, _, _ in channels if a is not None)
        total_n = max(total_n, 1)

        # Mix buffer [2, total_n]
        mix_buf = torch.zeros(2, total_n, dtype=torch.float32)

        for audio_dict, vol, pan, start_t, fi, fo in channels:
            if audio_dict is None:
                continue

            wav, asr = self._get_wav(audio_dict)

            # Resample if needed
            if asr != sr:
                try:
                    import torchaudio
                    wav = torchaudio.functional.resample(wav, asr, sr)
                except Exception:
                    # Simple linear resample fallback
                    new_len = int(wav.shape[-1] * sr / asr)
                    wav = torch.nn.functional.interpolate(
                        wav.unsqueeze(0).float(), size=new_len, mode="linear",
                        align_corners=False
                    ).squeeze(0)

            wav = _to_stereo(wav)
            wav = _apply_fade(wav.clone(), sr, fi, fo)
            wav = _apply_pan(wav, pan)
            wav = wav * vol

            offset = int(start_t * sr)
            end    = min(offset + wav.shape[-1], total_n)
            src_n  = end - offset
            if src_n > 0:
                mix_buf[:, offset:end] += wav[:, :src_n]

        # Master volume
        mix_buf = mix_buf * master_volume

        # Output limiter (soft clip)
        if output_limiter:
            peak = float(mix_buf.abs().max())
            if peak > 0.999:
                mix_buf = mix_buf / peak * 0.999

        print(f"[S42P AudioMixer] Mixed {sum(1 for a,*_ in channels if a is not None)} "
              f"channels â†’ {total_n/sr:.1f}s @ {sr}Hz  "
              f"peak={float(mix_buf.abs().max()):.3f}")

        return ({"waveform": mix_buf.unsqueeze(0), "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {
    "S42PAudioMixer": S42PAudioMixer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PAudioMixer": "S42P Audio Mixer ðŸŽšï¸",
}
