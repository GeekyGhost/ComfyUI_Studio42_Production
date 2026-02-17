"""
S42 Production Suite â€” Audio Analyser
======================================
Analyses one or two ComfyUI AUDIO inputs and produces a scored quality report
with concrete, per-dimension recommendations pointing to S42P nodes.

Ported from ULM Audio Analyser (GeekyGhost) with the following changes:
  â€¢ Recommendations rewritten to reference S42P node names + parameters
  â€¢ Category updated to S42 Production Suite
  â€¢ Node/display names updated to S42P namespace
  â€¢ _NumpyEncoder added for safe JSON serialisation
  â€¢ Scoring thresholds unchanged (calibrated against real AceStep 1.5 output)

METRIC â†’ S42P NODE MAPPING:
  Dynamic Range       â†’ S42P Dynamics Processor (output_gain, output_limiter)
  Spectral Balance    â†’ S42P Parametric EQ (low_shelf / high_shelf / bands)
  HF Presence         â†’ S42P Parametric EQ (high_shelf, presence band ~5kHz)
  Transient Clarity   â†’ S42P Dynamics Processor (attack/release times)
  Perceived Loudness  â†’ S42P Mastering Chain (lufs_preset / output_gain)
  Rhythmic Coherence  â†’ S42P Beat Analyzer (informational)
  Sub-Bass Control    â†’ S42P Parametric EQ (high_pass, low_shelf)
  Stereo Image        â†’ S42P Mastering Chain (stereo_width)

DESIGN CONSTRAINTS:
  â€¢ numpy + scipy only â€” guaranteed present in any ComfyUI install
  â€¢ ComfyUI AUDIO: {"waveform": tensor[B,C,N], "sample_rate": int}
  â€¢ All scoring thresholds calibrated against real AceStep 1.5 output

Python 3.12 | ComfyUI Portable
"""

import json
import math
import numpy as np
from typing import Optional, Tuple
from scipy import signal as sp
from scipy.signal import lfilter

import logging
logger = logging.getLogger(__name__)


# ”€”€ JSON encoder ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
        return super().default(obj)


# ”€”€ Waveform helpers ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _to_mono(waveform) -> np.ndarray:
    """[B,C,N] tensor or ndarray †’ mono float32 [N]."""
    arr = waveform.numpy() if hasattr(waveform, 'numpy') else np.asarray(waveform)
    return arr.mean(axis=(0, 1)).astype(np.float32)

def _to_stereo(waveform) -> np.ndarray:
    """[B,C,N] †’ [C,N] float32."""
    arr = waveform.numpy() if hasattr(waveform, 'numpy') else np.asarray(waveform)
    return arr[0].astype(np.float32)


# ”€”€ Core DSP ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _stft_spectrum(y: np.ndarray, sr: int, n_fft: int = 4096, hop: int = 1024):
    """
    STFT with scaling='spectrum' so magnitudes are comparable to time-domain
    amplitude. Without this, scipy returns spectral density (V/âˆšHz) which
    makes all band energies appear ~1000Ã— smaller than expected.
    """
    f, t, Zxx = sp.stft(
        y.astype(np.float32), fs=sr,
        nperseg=n_fft, noverlap=n_fft - hop,
        window='hann', scaling='spectrum'
    )
    return f, t, np.abs(Zxx)


def _k_weighted_lufs(y: np.ndarray, sr: int) -> float:
    """
    ITU-R BS.1770 K-weighted integrated loudness (LUFS).
    Coefficients are for 48 kHz but remain valid for relative comparison
    at other sample rates.
    """
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0,              -1.69065929318241,  0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    y1 = lfilter(b1, a1, y)
    y2 = lfilter(b2, a2, y1)
    power = float(np.mean(y2 ** 2))
    return -0.691 + 10.0 * math.log10(power + 1e-12)


def _crest_factor(y: np.ndarray) -> float:
    rms = math.sqrt(float(np.mean(y ** 2)) + 1e-12)
    return float(np.max(np.abs(y))) / rms


def _crest_sections(y: np.ndarray, sr: int, section_sec: float = 1.0) -> np.ndarray:
    n   = int(section_sec * sr)
    cfs = [_crest_factor(y[i:i+n]) for i in range(0, len(y) - n, n)
           if _crest_factor(y[i:i+n]) < 100]
    return np.array(cfs, dtype=np.float32) if cfs else np.array([3.0], dtype=np.float32)


def _band_pct(mag: np.ndarray, freqs: np.ndarray) -> dict:
    """Band energy percentages (sum ‰ˆ 100%)."""
    bands = [
        (20,    80,    'sub'),
        (80,    300,   'bass'),
        (300,   1000,  'lo_mid'),
        (1000,  3000,  'mid'),
        (3000,  8000,  'hi_mid'),
        (8000,  20000, 'hi'),
    ]
    energies = {}
    for lo, hi, name in bands:
        mask = (freqs >= lo) & (freqs < hi)
        energies[name] = float(np.mean(mag[mask] ** 2)) if mask.any() else 0.0
    total = sum(energies.values()) + 1e-12
    return {k: v / total * 100.0 for k, v in energies.items()}


def _spectral_centroid(mag: np.ndarray, freqs: np.ndarray) -> Tuple[float, float]:
    w  = mag + 1e-8
    sc = np.sum(freqs[:, None] * w, axis=0) / np.sum(w, axis=0)
    return float(np.mean(sc)), float(np.std(sc))


def _envelope_autocorr(y: np.ndarray, sr: int,
                        frame_ms: float = 50.0,
                        hop_ms: float = 10.0) -> Tuple[np.ndarray, float]:
    """
    RMS amplitude envelope at hop_ms resolution.
    Beat-period envelope autocorr is the correct way to measure rhythmic
    periodicity â€” raw waveform autocorr at any fixed lag is near-zero for
    all real music because the waveform oscillates at audio frequencies.
    """
    frame_n = int(frame_ms / 1000 * sr)
    hop_n   = int(hop_ms   / 1000 * sr)
    env = np.array([
        math.sqrt(float(np.mean(y[i:i + frame_n] ** 2)) + 1e-12)
        for i in range(0, len(y) - frame_n, hop_n)
    ], dtype=np.float32)
    return env, sr / hop_n


def _estimate_bpm_and_coherence(env: np.ndarray,
                                 env_fps: float) -> Tuple[float, int, float]:
    """BPM estimation via beat-period envelope autocorrelation (60€“200 BPM)."""
    xn     = (env - env.mean()) / (env.std() + 1e-8)
    lo_lag = max(1, int(env_fps * 60.0 / 200.0))
    hi_lag = min(int(env_fps * 60.0 / 60.0), len(xn) - 2)
    if lo_lag >= hi_lag:
        return 120.0, int(env_fps * 0.5), 0.0
    lags = np.arange(lo_lag, hi_lag + 1)
    acs  = np.array([
        float(np.corrcoef(xn[:-l], xn[l:])[0, 1]) if l < len(xn) else 0.0
        for l in lags
    ])
    best_idx = int(np.argmax(acs))
    best_lag = int(lags[best_idx])
    best_ac  = float(acs[best_idx])
    bpm      = env_fps * 60.0 / best_lag
    return bpm, best_lag, best_ac


def _lr_correlation(stereo: np.ndarray) -> float:
    """L/R channel correlation: 1=mono, 0=independent, negative=phase issues."""
    if stereo.shape[0] < 2:
        return 1.0
    l = stereo[0] - stereo[0].mean()
    r = stereo[1] - stereo[1].mean()
    return float(np.mean(l * r) / (np.std(l) * np.std(r) + 1e-8))


def _onset_density(y: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> float:
    """Spectral flux onsets per second (no librosa required)."""
    f, _, mag = _stft_spectrum(y, sr, n_fft=n_fft, hop=hop)
    flux  = np.sum(np.maximum(0.0, np.diff(mag, axis=1)), axis=0)
    mu, sigma = flux.mean(), flux.std() + 1e-8
    peaks = np.where(
        (flux[1:-1] > flux[:-2]) &
        (flux[1:-1] > flux[2:]) &
        (flux[1:-1] > mu + 0.5 * sigma)
    )[0]
    return float(len(peaks) / (len(y) / sr))


def _cross_corr(y_a: np.ndarray, y_b: np.ndarray,
                window_sec: float = 5.0, sr: int = 48000) -> float:
    """Normalised zero-lag cross-correlation over first window_sec."""
    n    = int(window_sec * sr)
    a, b = y_a[:n].astype(np.float64), y_b[:n].astype(np.float64)
    n    = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a    = (a - a.mean()) / (a.std() + 1e-8)
    b    = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.dot(a, b) / n)


def _rms_envelope_1s(y: np.ndarray, sr: int) -> list:
    n = int(sr)
    return [float(math.sqrt(float(np.mean(y[i:i+n]**2)) + 1e-12))
            for i in range(0, len(y) - n, n)]


# ”€”€ Scoring ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _score(value, lo_bad, lo_good, hi_good, hi_bad) -> float:
    """Trapezoid scorer †’ 0€“10. Plateau [lo_good, hi_good] = 10."""
    if lo_good <= value <= hi_good: return 10.0
    if value < lo_bad or value > hi_bad: return 0.0
    if value < lo_good:
        return 10.0 * (value - lo_bad) / (lo_good - lo_bad + 1e-9)
    return 10.0 * (hi_bad - value) / (hi_bad - hi_good + 1e-9)

def _score_low_best(value, excellent, good, bad) -> float:
    """For metrics where lower is better (e.g. spectral imbalance)."""
    if value <= excellent: return 10.0
    if value >= bad:       return 0.0
    if value <= good:
        return 10.0 - 4.0 * (value - excellent) / (good - excellent + 1e-9)
    return  6.0 - 6.0 * (value - good)     / (bad  - good      + 1e-9)


# ”€”€ Full analysis ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _analyse(waveform, sr: int, label: str) -> dict:
    m      = _to_mono(waveform)
    stereo = _to_stereo(waveform)

    f, _, mag        = _stft_spectrum(m, sr)
    bp               = _band_pct(mag, f)
    sc_mean, sc_std  = _spectral_centroid(mag, f)
    cf_global        = _crest_factor(m)
    cf_secs          = _crest_sections(m, sr)
    cf_std           = float(np.std(cf_secs))
    lufs             = _k_weighted_lufs(m, sr)
    lr_corr          = _lr_correlation(stereo)
    onset_rate       = _onset_density(m, sr)
    env, env_fps     = _envelope_autocorr(m, sr)
    bpm_est, beat_lag, beat_ac = _estimate_bpm_and_coherence(env, env_fps)
    rms_env_1s       = _rms_envelope_1s(m, sr)
    rms_global       = float(math.sqrt(float(np.mean(m**2)) + 1e-12))
    peak_global      = float(np.max(np.abs(m)))

    hf_score_val = bp['hi_mid'] + bp['hi'] * 3.0
    band_db      = [10.0 * math.log10(bp[k] / 100.0 + 1e-12)
                    for k in ['sub', 'bass', 'lo_mid', 'mid', 'hi_mid', 'hi']]
    spec_imbal   = float(np.std(band_db))

    scores = {
        'dynamic_range':     _score(cf_global,   2.0,  3.5,  9.0, 14.0),
        'spectral_balance':  _score_low_best(spec_imbal, 12.0, 20.0, 32.0),
        'hf_presence':       _score(hf_score_val, 0.005, 0.03, 0.50, 2.0),
        'transient_clarity': _score(cf_std,        0.1,  0.4,  3.0,  6.0),
        'perceived_loudness':_score(lufs,         -30.0,-20.0, -8.0, -3.0),
        'rhythmic_coherence':_score(beat_ac,       0.1,  0.4,  0.97, 1.001),
        'sub_bass_control':  _score(bp['sub'],    15.0, 30.0, 87.0, 95.0),
        'stereo_image':      _score(lr_corr,      -0.2,  0.4,  0.96, 1.001),
    }
    overall = float(np.mean(list(scores.values())))

    return {
        'label':             label,
        'sr':                sr,
        'duration':          len(m) / sr,
        'rms':               rms_global,
        'peak':              peak_global,
        'lufs':              lufs,
        'crest_factor':      cf_global,
        'crest_std':         cf_std,
        'band_pct':          bp,
        'spectral_imbalance':spec_imbal,
        'sc_mean':           sc_mean,
        'sc_std':            sc_std,
        'hf_score_val':      hf_score_val,
        'bpm_estimate':      bpm_est,
        'beat_autocorr':     beat_ac,
        'onset_rate':        onset_rate,
        'lr_correlation':    lr_corr,
        'rms_envelope_1s':   rms_env_1s,
        'scores':            scores,
        'overall_score':     overall,
    }


# ”€”€ Recommendation engine ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _recommendations(d: dict) -> list:
    """
    Returns list of (priority, node, param, message).
    priority: 1=HIGH, 2=MED, 3=INFO
    All recommendations reference S42P node names and parameters.
    """
    recs = []
    s    = d['scores']
    bp   = d['band_pct']

    # ”€”€ Dynamic Range ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    cf = d['crest_factor']
    if s['dynamic_range'] < 4:
        if cf < 3.0:
            recs.append((1, 'S42P Dynamics Processor', 'low_ratio / mid_ratio',
                f"OVER-COMPRESSED (crest={cf:.2f}). Reduce compression ratios â€” "
                f"try low_ratio=1.5, mid_ratio=1.5. Disable glue_enabled if active."))
        elif cf > 11.0:
            recs.append((1, 'S42P Mastering Chain', 'true_peak_ceiling + output_limiter',
                f"CLIPPING RISK (crest={cf:.2f}). "
                f"Lower true_peak_ceiling to -3.0 dB. Confirm output_limiter_enabled=True."))
    elif s['dynamic_range'] < 6 and cf < 3.5:
        recs.append((2, 'S42P Dynamics Processor', 'low_ratio / mid_ratio',
            f"Slightly compressed (crest={cf:.2f}). "
            f"Reduce band ratios slightly or increase low_threshold by 3â€“6 dB."))

    # ”€”€ Spectral Balance ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    imbal = d['spectral_imbalance']
    sub   = bp['sub']
    if s['spectral_balance'] < 4:
        recs.append((1, 'S42P Parametric EQ', 'b1 (high_pass) + low_shelf',
            f"SEVERE SPECTRAL IMBALANCE ({imbal:.1f} dB std). "
            f"Sub energy={sub:.1f}%. If sub-dominated: raise HP freq from 20â†’60Hz, "
            f"add low_shelf at 80Hz with -3 to -6 dB gain. "
            f"Then re-run analyser to verify."))
    elif s['spectral_balance'] < 6:
        recs.append((2, 'S42P Parametric EQ', 'low_shelf / high_shelf',
            f"Moderate spectral imbalance ({imbal:.1f} dB std). "
            f"Gentle low_shelf cut (-2 dB at 100Hz) and high_shelf boost (+1.5 dB at 8kHz) "
            f"will flatten the curve."))

    # ”€”€ HF Presence ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    hf = d['hf_score_val']
    if s['hf_presence'] < 4:
        if hf < 0.03:
            recs.append((1, 'S42P Parametric EQ', 'b7 (high_shelf) + presence peak',
                f"DULL OUTPUT (hi_mid={bp['hi_mid']:.4f}%, hi={bp['hi']:.5f}%). "
                f"Add high_shelf boost: +3 dB at 8kHz. "
                f"Add peak band at 5kHz: +2.5 dB, Q=1.0. "
                f"Also check S42P Mastering Chain exciter_drive > 0."))
        else:
            recs.append((1, 'S42P Parametric EQ', 'high_shelf cut',
                f"TOO BRIGHT (hf_score={hf:.4f}). "
                f"Cut high_shelf: -3 to -5 dB at 8kHz."))
    elif s['hf_presence'] < 7 and hf < 0.03:
        recs.append((2, 'S42P Parametric EQ', 'presence peak ~5kHz',
            f"Modest HF presence (hi_mid={bp['hi_mid']:.4f}%). "
            f"Gentle peak at 5kHz (+1.5 dB, Q=1.2) and exciter_drive=0.15 in Mastering Chain."))

    # ”€”€ Transient Clarity ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    cf_std = d['crest_std']
    if s['transient_clarity'] < 4:
        if cf_std < 0.3:
            recs.append((1, 'S42P Dynamics Processor', 'low_attack / mid_attack',
                f"SMEARED TRANSIENTS (CF std={cf_std:.3f}). "
                f"Slow the attack: low_attack=30ms, mid_attack=20ms. "
                f"This lets transient peaks through before compression engages."))
        else:
            recs.append((1, 'S42P Dynamics Processor', 'low_release / mid_release',
                f"ERRATIC TRANSIENTS (CF std={cf_std:.3f} â€” too spiky). "
                f"Shorten release: low_release=60ms, mid_release=50ms. "
                f"Engage glue_enabled with glue_ratio=2.0."))

    # ”€”€ Perceived Loudness ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    lufs = d['lufs']
    if s['perceived_loudness'] < 4:
        if lufs < -22.0:
            gain = round(-14.0 - lufs, 1)
            recs.append((1, 'S42P Mastering Chain', 'lufs_preset / manual_lufs_target',
                f"TOO QUIET ({lufs:.1f} LUFS, streaming target -14). "
                f"Set lufs_preset='Streaming â€” Spotify / Apple Music (-14 LUFS)' "
                f"OR manual_lufs_target={-14.0:.1f} for ~{gain:+.1f} dB gain."))
        else:
            recs.append((1, 'S42P Mastering Chain', 'manual_lufs_target + true_peak_ceiling',
                f"TOO LOUD ({lufs:.1f} LUFS). "
                f"Set lufs_preset='Broadcast / TV (EBU R128, -23 LUFS)' or lower manually. "
                f"Reduce true_peak_ceiling to -3.0 dB."))
    elif s['perceived_loudness'] < 7:
        gain_nudge = round(-14.0 - lufs, 1)
        recs.append((2, 'S42P Mastering Chain', 'manual_lufs_target',
            f"Loudness {lufs:.1f} LUFS (streaming target -14). "
            f"A {gain_nudge:+.1f} dB adjustment via lufs_preset would hit target."))

    # ”€”€ Rhythmic Coherence ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    bpm   = d['bpm_estimate']
    b_ac  = d['beat_autocorr']
    onset = d['onset_rate']
    if s['rhythmic_coherence'] < 4:
        if b_ac < 0.4:
            recs.append((1, 'S42P Beat Analyzer', 'mode / min_bpm',
                f"WEAK RHYTHMIC STRUCTURE (beat autocorr={b_ac:.3f} @ est {bpm:.0f} BPM). "
                f"Run S42P Beat Analyzer to confirm BPM detection. "
                f"If percussive: check Dynamics attack settings aren't killing transients."))
    else:
        recs.append((3, 'INFO', 'rhythmic_coherence',
            f"Good rhythmic lock (beat autocorr={b_ac:.3f} @ ~{bpm:.0f} BPM). "
            f"Beat structure well-defined â€” no intervention needed."))

    # ”€”€ Sub-Bass Control ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    if s['sub_bass_control'] < 4:
        if sub > 88.0:
            recs.append((1, 'S42P Parametric EQ', 'b1 (high_pass freq)',
                f"SUB-DOMINATED ({sub:.1f}% energy below 80Hz). "
                f"Raise b1 high_pass from 20Hz to 50-80Hz to trim excess sub. "
                f"Also try low_shelf at 80Hz with -3 dB."))
        elif sub < 20.0:
            recs.append((2, 'S42P Parametric EQ', 'b1 (high_pass freq)',
                f"THIN LOW END ({sub:.1f}% sub energy). "
                f"Lower or disable the high_pass (set to 20Hz / bypass). "
                f"Add low_shelf at 80Hz with +2 dB."))

    # ”€”€ Stereo Image ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    lr = d['lr_correlation']
    if s['stereo_image'] < 5:
        if lr > 0.97:
            recs.append((3, 'INFO', 'stereo image',
                f"Near-mono (L/R corr={lr:.3f}). "
                f"Set S42P Mastering Chain stereo_width > 1.0 to widen. "
                f"Normal for mono sources or centre-panned vocals."))
        elif lr < 0.2:
            recs.append((2, 'S42P Mastering Chain', 'stereo_width',
                f"Low L/R correlation ({lr:.3f}) â€” possible phase issue. "
                f"Reduce stereo_width toward 1.0 in Mastering Chain."))

    # ”€”€ Content-aware suggestion ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€
    if onset > 7.0 and s['transient_clarity'] >= 7:
        recs.append((3, 'S42P Dynamics Processor', 'mode / attack settings',
            f"Percussive content ({onset:.1f} onsets/s) with strong transients. "
            f"Use slow attack (20-40ms) on all bands to preserve punch. "
            f"Glue compressor (glue_enabled=True, glue_ratio=1.5) adds cohesion."))
    elif onset < 2.0 and s['rhythmic_coherence'] >= 7:
        recs.append((3, 'S42P Mastering Chain', 'exciter_blend',
            f"Sustained/ambient content ({onset:.1f} onsets/s). "
            f"exciter_drive=0.1, exciter_high_only=True adds air without muddying lows."))

    recs.sort(key=lambda x: x[0])
    return recs


# ”€”€ Report formatter ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

def _bar(score, width=10):
    n = max(0, min(width, int(round(score / 10.0 * width))))
    return '[' + 'â–ˆ' * n + 'â–‘' * (width - n) + ']'

def _bband(pct, width=22):
    n = max(0, min(width, int(round(pct / 100.0 * width))))
    return 'â–ˆ' * n + 'â–‘' * (width - n)

def _wrap(text, prefix, width=62):
    words = text.split()
    lines, cur = [], prefix
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = ' ' * len(prefix) + w
        else:
            cur += (' ' if cur.strip() else '') + w
    if cur.strip():
        lines.append(cur)
    return '\n'.join(lines)

def _sep(title='', width=62):
    if not title: return 'â”€' * width
    pad = width - len(title) - 4
    return f'â”€â”€ {title} ' + 'â”€' * max(0, pad)

_SCORE_LABELS = {
    'dynamic_range':      'Dynamic Range      ',
    'spectral_balance':   'Spectral Balance   ',
    'hf_presence':        'HF Presence        ',
    'transient_clarity':  'Transient Clarity  ',
    'perceived_loudness': 'Loudness (LUFS)    ',
    'rhythmic_coherence': 'Rhythmic Coherence  ',
    'sub_bass_control':   'Sub-Bass Control   ',
    'stereo_image':       'Stereo Image       ',
}


def _format_report(d_a: dict, d_b: Optional[dict] = None,
                   xcorr: Optional[float] = None) -> str:
    W = 62
    L = []
    L += ['â•' * W,
          '  S42P AUDIO ANALYSER  Â·  S42 Production Suite ðŸ”¬',
          'â•' * W]

    def scores_block(d):
        L.append(_sep(f'SCORES  [{d["label"]}]'))
        for k, lbl in _SCORE_LABELS.items():
            v = d['scores'][k]
            L.append(f'  {lbl}  {_bar(v)} {v:4.1f}/10')
        L.append('  ' + 'â”€' * 56)
        v = d['overall_score']
        L.append(f'  {"OVERALL":<22s}  {_bar(v)} {v:4.1f}/10')

    def metrics_block(d):
        L.append(_sep(f'MEASUREMENTS  [{d["label"]}]'))
        rms_db  = 20 * math.log10(d['rms']  + 1e-8)
        peak_db = 20 * math.log10(d['peak'] + 1e-8)
        L.extend([
            f'  Duration          {d["duration"]:.1f}s',
            f'  RMS               {d["rms"]:.4f}  ({rms_db:.1f} dBFS)',
            f'  Peak              {d["peak"]:.4f}  ({peak_db:.1f} dBFS)',
            f'  LUFS (K-weighted) {d["lufs"]:.2f}',
            f'  Crest Factor      {d["crest_factor"]:.2f}  (Ïƒ/s={d["crest_std"]:.2f})',
            f'  Est. BPM          {d["bpm_estimate"]:.1f}  (beat autocorr={d["beat_autocorr"]:.3f})',
            f'  Onset Rate        {d["onset_rate"]:.1f}/s',
            f'  L/R Correlation   {d["lr_correlation"]:.3f}',
            f'  Spec Centroid     {d["sc_mean"]:.0f} Hz  (Ïƒ={d["sc_std"]:.0f})',
            f'  Spec Imbalance    {d["spectral_imbalance"]:.1f} dB std',
            f'  HF Score Val      {d["hf_score_val"]:.5f}  (hi_mid={d["band_pct"]["hi_mid"]:.4f}%)',
        ])

    def bands_block(d):
        L.append(_sep(f'SPECTRAL DISTRIBUTION  [{d["label"]}]'))
        names = {
            'sub':    'Sub   20-80Hz  ',
            'bass':   'Bass  80-300Hz ',
            'lo_mid': 'Lo-Mid 300-1k  ',
            'mid':    'Mid   1-3 kHz  ',
            'hi_mid': 'Hi-Mid 3-8kHz  ',
            'hi':     'Hi    8-20kHz  ',
        }
        for k, lbl in names.items():
            pct = d['band_pct'][k]
            L.append(f'  {lbl}  {_bband(pct)}  {pct:6.3f}%')

    scores_block(d_a)
    L.append('')
    metrics_block(d_a)
    L.append('')
    bands_block(d_a)

    if d_b is not None:
        L.append('')
        scores_block(d_b)
        L.append('')
        metrics_block(d_b)
        L.append('')
        bands_block(d_b)

        L.append('')
        L.append(_sep('COMPARISON  [A vs B]'))
        if xcorr is not None:
            if xcorr > 0.95:
                L += [f'  Same generation : YES  (xcorr={xcorr:.4f})',
                      f'  Processing delta is isolated to S42P nodes.']
            elif xcorr > 0.5:
                L.append(f'  Same generation : LIKELY  (xcorr={xcorr:.4f})')
            else:
                L += [f'  Same generation : NO  (xcorr={xcorr:.4f})',
                      f'  Scores reflect absolute quality per track.']
        L.append('')
        for k, lbl in _SCORE_LABELS.items():
            sa, sb = d_a['scores'][k], d_b['scores'][k]
            delta  = sb - sa
            arr    = 'â–²' if delta > 0.3 else ('â–¼' if delta < -0.3 else 'â•')
            tag    = 'â† B wins' if delta > 0.5 else ('â† A wins' if delta < -0.5 else '')
            L.append(f'  {lbl}  A:{sa:4.1f}  B:{sb:4.1f}  {arr}{abs(delta):.1f}  {tag}')
        L.append('  ' + 'â”€' * 56)
        oa, ob = d_a['overall_score'], d_b['overall_score']
        dov = ob - oa
        win = 'B' if dov > 0.2 else ('A' if dov < -0.2 else 'TIE')
        L.append(f'  {"OVERALL":<20s}  A:{oa:4.1f}  B:{ob:4.1f}  Î”{dov:+.1f}  â†’ {win}')

    L.append('')
    L.append(_sep('RECOMMENDATIONS'))
    pairs = [(d_a, d_a['label'])] + ([(d_b, d_b['label'])] if d_b else [])
    for d, tag in pairs:
        recs = _recommendations(d)
        L.append(f'  â–¸ {tag}')
        if not recs:
            L.append('    No critical issues. Output quality is acceptable.')
        else:
            for pri, node, param, msg in recs:
                pri_lbl = {1: 'âš  HIGH', 2: 'Â· MED ', 3: 'Â· INFO'}[pri]
                L.append(f'  {pri_lbl}  {node}')
                L.append(f'          â””â”€ {param}')
                L.append(_wrap(msg, '             '))
        L.append('')

    L.append('â•' * W)
    return '\n'.join(L)


# ”€”€ ComfyUI Node ”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€”€

CATEGORY = "S42 Production Suite ðŸ”¬ Analysis"


class S42PAudioAnalyser:
    """
    S42P Audio Analyser â€” analyse one or two audio inputs and produce a
    scored quality report with concrete S42P node parameter recommendations.

    Outputs:
      report_text â€” human-readable scored report
      json_data   â€” machine-readable full analysis
      score_a     â€” overall quality 0.0â€“10.0
      score_b     â€” overall quality 0.0â€“10.0 (-1.0 if audio_b not connected)
      winner      â€” "A", "B", "TIE", or "N/A"
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("AUDIO", {
                    "tooltip": (
                        "Primary audio. Works standalone for single-track analysis "
                        "or use as 'before' in A/B comparison."
                    )
                }),
            },
            "optional": {
                "audio_b": ("AUDIO", {
                    "tooltip": (
                        "Second audio for comparison. "
                        "Connect e.g. unprocessed audio to A and mastered to B."
                    )
                }),
                "label_a": ("STRING", {"default": "A  (Baseline)",   "multiline": False}),
                "label_b": ("STRING", {"default": "B  (Processed)",  "multiline": False}),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES  = ("report_text", "json_data", "score_a", "score_b", "winner")
    FUNCTION      = "analyse"
    CATEGORY      = CATEGORY
    OUTPUT_NODE   = True

    def analyse(self, audio_a: dict, audio_b: Optional[dict] = None,
                label_a: str = "A  (Baseline)",
                label_b: str = "B  (Processed)"):

        sr  = audio_a["sample_rate"]
        wa  = audio_a["waveform"]
        d_a = _analyse(wa, sr, label_a)

        d_b    = None
        xcorr  = None
        winner = "N/A"
        score_b = -1.0

        if audio_b is not None:
            wb   = audio_b["waveform"]
            sr_b = audio_b["sample_rate"]
            d_b  = _analyse(wb, sr_b, label_b)
            xcorr = _cross_corr(_to_mono(wa), _to_mono(wb), sr=sr)
            oa, ob = d_a["overall_score"], d_b["overall_score"]
            winner  = "B" if ob - oa > 0.2 else ("A" if oa - ob > 0.2 else "TIE")
            score_b = float(d_b["overall_score"])

        report = _format_report(d_a, d_b, xcorr)

        def _safe(d):
            out = dict(d)
            out["rms_envelope_1s"] = out["rms_envelope_1s"][:60]
            return out

        payload = {"a": _safe(d_a)}
        if d_b is not None:
            payload["b"]                  = _safe(d_b)
            payload["winner"]             = winner
            payload["score_delta"]        = d_b["overall_score"] - d_a["overall_score"]
            payload["cross_correlation"]  = xcorr

        json_str = json.dumps(payload, indent=2, cls=_NumpyEncoder)
        score_a  = float(d_a["overall_score"])

        print("\n" + report)
        return (report, json_str, score_a, score_b, winner)


NODE_CLASS_MAPPINGS = {
    "S42PAudioAnalyser": S42PAudioAnalyser
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "S42PAudioAnalyser": "S42P Audio Analyser ðŸ”¬"
}
