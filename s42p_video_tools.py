"""
S42 Production Suite â€” Video Tools Node
========================================
Multi-function video utility node. Mode selector at top drives behaviour.

Modes:
  trim        â€” cut a clip to in/out points (frame or second based)
  concatenate â€” join up to 6 clips in sequence (auto-matches resolution)
  reverse     â€” reverse frame order
  fps_convert â€” convert frame rate (duplicate or blend modes)

All operations work on standard ComfyUI IMAGE batches [B, H, W, C].
Output is an IMAGE batch ready for VHS Video Combine or further nodes.

Python 3.12 | ComfyUI Portable | torch
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple

from s42p_video_utils import (
    ensure_rgb, match_resolution, convert_fps, frames_to_np, np_to_frames
)

logger = logging.getLogger(__name__)

CATEGORY = "S42 Production Suite ðŸŽ¬ Animation"

MODES = ["trim", "concatenate", "reverse", "fps_convert"]


class S42PVideoTools:
    """
    ðŸ› ï¸ S42P Video Tools
    Multi-purpose video utility node.
    Select a mode at the top â€” only the relevant controls apply.

    Modes:
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    trim        Cut a clip to specific in/out points.
                Specify by frame number or by seconds.

    concatenate Join up to 6 clips in sequence.
                Resolutions are auto-matched to clip_a.
                Connect clips in order â€” gaps (unconnected) are skipped.

    reverse     Flip the clip so last frame plays first.
                Useful for ping-pong loops or creative effects.

    fps_convert Change the playback frame rate.
                Wan generates at ~8fps â€” convert to 24/30fps for smooth playback.
                'blend' mode interpolates between frames for smooth motion.
                'duplicate' mode is faster but can look jittery.
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("IMAGE", {
                    "tooltip": ("Primary video clip (IMAGE batch). "
                                "Used by all modes. In concatenate mode, this is the first clip.")
                }),
                "mode": (MODES, {
                    "default": "trim",
                    "tooltip": ("Operation to perform:\n"
                                "trim = cut to in/out points.\n"
                                "concatenate = join multiple clips.\n"
                                "reverse = flip clip playback direction.\n"
                                "fps_convert = change frame rate.")
                }),

                # ”€”€ Trim controls ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
                "trim_unit": (["frames", "seconds"], {
                    "default": "frames",
                    "tooltip": ("Whether in_point and out_point are in frame numbers or seconds. "
                                "frames = exact frame indices (0-based). "
                                "seconds = time-based (requires specifying source_fps).")
                }),
                "in_point": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 99999.0,
                    "step": 1.0, "display": "slider",
                    "tooltip": ("Start point for trim. "
                                "In 'frames' mode: frame index (0 = first frame). "
                                "In 'seconds' mode: time in seconds from start. "
                                "Only used in 'trim' mode.")
                }),
                "out_point": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 99999.0,
                    "step": 1.0, "display": "slider",
                    "tooltip": ("End point for trim (inclusive). "
                                "-1 = trim to end of clip. "
                                "In 'frames' mode: frame index. "
                                "In 'seconds' mode: time in seconds. "
                                "Only used in 'trim' mode.")
                }),

                # ”€”€ FPS controls ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
                "source_fps": ("FLOAT", {
                    "default": 8.0, "min": 0.5, "max": 120.0,
                    "step": 0.5, "display": "slider",
                    "tooltip": ("Source frame rate of clip_a. "
                                "Wan generates video at approximately 8fps by default. "
                                "Set this to match the actual generation FPS. "
                                "Also used in 'trim' mode when trim_unit='seconds'.")
                }),
                "target_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0,
                    "step": 1.0, "display": "slider",
                    "tooltip": ("Target frame rate for fps_convert mode. "
                                "24fps = standard cinema. "
                                "30fps = standard video/web. "
                                "60fps = smooth/high motion. "
                                "Only used in 'fps_convert' mode.")
                }),
                "fps_mode": (["blend", "duplicate"], {
                    "default": "blend",
                    "tooltip": ("Frame interpolation method for fps_convert:\n"
                                "blend = linearly interpolates between adjacent frames. "
                                "Smooth motion, slightly softens the image.\n"
                                "duplicate = repeats or drops frames. "
                                "Faster but can look choppy with large FPS differences.")
                }),
            },
            "optional": {
                # Additional clips for concatenation
                "clip_b": ("IMAGE", {
                    "tooltip": "Second clip for concatenate mode. Leave unconnected to skip."
                }),
                "clip_c": ("IMAGE", {
                    "tooltip": "Third clip for concatenate mode. Leave unconnected to skip."
                }),
                "clip_d": ("IMAGE", {
                    "tooltip": "Fourth clip for concatenate mode. Leave unconnected to skip."
                }),
                "clip_e": ("IMAGE", {
                    "tooltip": "Fifth clip for concatenate mode. Leave unconnected to skip."
                }),
                "clip_f": ("IMAGE", {
                    "tooltip": "Sixth clip for concatenate mode. Leave unconnected to skip."
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE", "INT", "STRING")
    RETURN_NAMES  = ("video_frames", "frame_count", "info")
    FUNCTION      = "process"
    CATEGORY      = CATEGORY

    def process(self, clip_a: torch.Tensor, mode: str,
                trim_unit: str, in_point: float, out_point: float,
                source_fps: float, target_fps: float, fps_mode: str,
                clip_b: Optional[torch.Tensor] = None,
                clip_c: Optional[torch.Tensor] = None,
                clip_d: Optional[torch.Tensor] = None,
                clip_e: Optional[torch.Tensor] = None,
                clip_f: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int, str]:

        clip_a = ensure_rgb(clip_a)

        if mode == "trim":
            result, info = self._trim(clip_a, trim_unit, in_point, out_point, source_fps)
        elif mode == "concatenate":
            clips = [clip_a] + [c for c in [clip_b, clip_c, clip_d, clip_e, clip_f]
                                if c is not None]
            result, info = self._concatenate(clips)
        elif mode == "reverse":
            result = clip_a.flip(0)
            info = f"Reversed {clip_a.shape[0]} frames."
        elif mode == "fps_convert":
            result = convert_fps(clip_a, source_fps, target_fps, fps_mode)
            info   = (f"FPS convert: {source_fps}fps â†’ {target_fps}fps | "
                      f"{clip_a.shape[0]} frames â†’ {result.shape[0]} frames | "
                      f"mode={fps_mode}")
        else:
            result = clip_a
            info   = "Unknown mode â€” returning clip_a unchanged."

        return (result, int(result.shape[0]), info)

    # ”€”€ Trim ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

    def _trim(self, clip: torch.Tensor, unit: str,
              in_pt: float, out_pt: float, fps: float) -> Tuple[torch.Tensor, str]:
        n = clip.shape[0]

        if unit == "seconds":
            in_frame  = int(round(in_pt  * fps))
            out_frame = int(round(out_pt * fps)) if out_pt >= 0 else n - 1
        else:
            in_frame  = int(in_pt)
            out_frame = int(out_pt) if out_pt >= 0 else n - 1

        in_frame  = max(0, min(in_frame,  n - 1))
        out_frame = max(in_frame, min(out_frame, n - 1))

        result = clip[in_frame : out_frame + 1]
        info   = (f"Trimmed: frames {in_frame}â€“{out_frame} of {n} | "
                  f"output = {result.shape[0]} frames")
        return result, info

    # ”€”€ Concatenate ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

    def _concatenate(self, clips: list) -> Tuple[torch.Tensor, str]:
        if not clips:
            raise ValueError("No clips provided to concatenate.")

        reference = ensure_rgb(clips[0])
        parts     = [reference]
        counts    = [reference.shape[0]]

        for i, c in enumerate(clips[1:], 2):
            c = ensure_rgb(c)
            _, c = match_resolution(reference, c)
            parts.append(c)
            counts.append(c.shape[0])

        merged = torch.cat(parts, dim=0)
        info   = (f"Concatenated {len(parts)} clips: "
                  f"{' + '.join(str(x) for x in counts)} = {merged.shape[0]} frames")
        return merged, info


# ”€”€ ComfyUI registration ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

NODE_CLASS_MAPPINGS = {
    "S42PVideoTools": S42PVideoTools,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PVideoTools": "ðŸ› ï¸ S42P Video Tools",
}
