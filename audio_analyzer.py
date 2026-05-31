#!/usr/bin/env python3
"""Localized audio analysis for the Phonk pipeline.

Distilled from BeatSync-Engine's auto_mode stages (stage1_audio, stage2_features,
stage3_sections). All Gradio, caching, progress-callback, and logging glue has
been stripped. What remains is the pure librosa signal analysis:

    * tempo (BPM) + a stable beat grid
    * primary downbeat timestamps (bar-aligned)
    * the section with the highest RMS energy (the "main drop")

Usage:
    analyzer = AudioAnalyzer()
    result = analyzer.analyze("track.wav")
    print(result.tempo, result.downbeats, result.main_drop)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import librosa
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioConfig:
    """Signal-analysis tuning. Defaults match BeatSync-Engine Auto Mode."""

    sr: int = 22050
    hop_length: int = 512
    n_fft: int = 2048

    # Beat tracking
    start_bpm: float = 120.0
    tightness: float = 120.0

    # Number of beats per bar; the first beat of each bar is a downbeat.
    beats_per_bar: int = 4

    # Structural segmentation
    section_target_seconds: float = 32.0  # aim for one section per ~32s
    section_min_count: int = 3
    section_max_count: int = 8
    section_min_seconds: float = 10.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class Section:
    index: int
    start: float
    end: float
    duration: float
    mean_rms: float  # normalized 0..1 average energy across the section


@dataclass
class AnalysisResult:
    tempo: float
    beat_times: np.ndarray
    downbeats: np.ndarray
    sections: List[Section]
    main_drop: Optional[Section]
    duration: float


# ---------------------------------------------------------------------------
# Numerical helpers (ported from auto_mode/__init__.py)
# ---------------------------------------------------------------------------


def _to_float(value, default: float = 0.0) -> float:
    """librosa>=0.10 returns tempo as an array; coerce to a scalar safely."""
    try:
        arr = np.asarray(value).reshape(-1)
        if arr.size:
            return float(arr[0])
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _normalize(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Robust 2nd/98th-percentile min-max normalization into [0, 1]."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=default, posinf=default, neginf=default)
    lo = float(np.percentile(arr, 2))
    hi = float(np.percentile(arr, 98))
    if hi - lo < 1e-8:
        return np.zeros_like(arr) + default
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Edge-padded moving-average smoothing with an odd kernel width."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 3 or width <= 1:
        return arr
    width = int(max(1, min(width, max(1, arr.size))))
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class AudioAnalyzer:
    """Computes tempo, downbeats, and the main-drop section from an audio file."""

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        self.cfg = config or AudioConfig()

    # -- public API --------------------------------------------------------

    def analyze(
        self,
        audio_file: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> AnalysisResult:
        """Run the full analysis on a (slice of) an audio file."""
        y, sr = self.load(audio_file, start_time, end_time)
        duration = len(y) / sr

        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
        except Exception:
            y_harmonic, y_percussive = y, y

        tempo, beat_times = self.detect_tempo_and_beats(y_percussive, sr)
        downbeats = self.detect_downbeats(beat_times)
        sections = self.detect_sections(y, y_harmonic, y_percussive, sr, beat_times)
        main_drop = max(sections, key=lambda s: s.mean_rms) if sections else None

        return AnalysisResult(
            tempo=tempo,
            beat_times=beat_times,
            downbeats=downbeats,
            sections=sections,
            main_drop=main_drop,
            duration=duration,
        )

    def load(
        self,
        audio_file: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Load + peak-normalize mono audio at the configured sample rate."""
        duration = None
        if end_time is not None and end_time > start_time:
            duration = end_time - start_time

        y, sr = librosa.load(
            audio_file,
            sr=self.cfg.sr,
            offset=start_time,
            duration=duration,
            mono=True,
        )
        if y.size == 0:
            raise ValueError("Audio file is empty or could not be decoded.")
        return librosa.util.normalize(y), sr

    # -- 1. tempo + beat grid ---------------------------------------------

    def detect_tempo_and_beats(
        self, y_percussive: np.ndarray, sr: int
    ) -> Tuple[float, np.ndarray]:
        """Estimate global tempo (BPM) and a stable beat grid.

        Returns (tempo_bpm, beat_times_seconds).
        """
        cfg = self.cfg
        onset_env = librosa.onset.onset_strength(
            y=y_percussive, sr=sr, hop_length=cfg.hop_length, aggregate=np.median
        )
        onset_env = _normalize(_smooth(onset_env, 3))

        try:
            tempo_raw, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=cfg.hop_length,
                units="frames",
                start_bpm=cfg.start_bpm,
                tightness=cfg.tightness,
                trim=False,
            )
        except TypeError:  # older librosa signatures
            tempo_raw, beat_frames = librosa.beat.beat_track(
                y=y_percussive,
                sr=sr,
                hop_length=cfg.hop_length,
                units="frames",
                start_bpm=cfg.start_bpm,
                tightness=cfg.tightness,
            )

        tempo = _to_float(tempo_raw, cfg.start_bpm)
        beat_frames = np.asarray(beat_frames, dtype=int)

        # Fallback to onset peaks if beat tracking found almost nothing.
        if beat_frames.size < 2:
            beat_frames = np.asarray(
                librosa.onset.onset_detect(
                    onset_envelope=onset_env,
                    sr=sr,
                    hop_length=cfg.hop_length,
                    backtrack=True,
                    wait=4,
                ),
                dtype=int,
            )
            tempo = cfg.start_bpm

        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=cfg.hop_length)
        beat_times = np.asarray(beat_times, dtype=float)
        valid = np.isfinite(beat_times) & (beat_times >= 0.0)
        return tempo, beat_times[valid]

    # -- 2. downbeats ------------------------------------------------------

    def detect_downbeats(self, beat_times: np.ndarray) -> np.ndarray:
        """Primary downbeats = the first beat of every bar.

        BeatSync marks bar anchors as every ``beats_per_bar``-th beat in the
        grid; the downbeat timestamps are those anchors.
        """
        beat_times = np.asarray(beat_times, dtype=float)
        if beat_times.size == 0:
            return beat_times
        step = max(1, self.cfg.beats_per_bar)
        return beat_times[::step]

    # -- 3. sections + main drop (highest RMS energy) ----------------------

    def detect_sections(
        self,
        y: np.ndarray,
        y_harmonic: np.ndarray,
        y_percussive: np.ndarray,
        sr: int,
        beat_times: np.ndarray,
    ) -> List[Section]:
        """Segment the track structurally and score each section by RMS energy.

        Uses chroma + MFCC + onset-strength agglomerative clustering (the same
        feature stack BeatSync uses) to find boundaries, then averages a
        normalized RMS envelope within each segment. The section with the
        highest ``mean_rms`` is the song's main drop.
        """
        cfg = self.cfg
        duration = len(y) / sr
        if duration <= 0:
            return []

        # Normalized RMS envelope and its frame timestamps.
        rms_curve = librosa.feature.rms(
            y=y, frame_length=cfg.n_fft, hop_length=cfg.hop_length
        )[0]
        rms_curve = _normalize(_smooth(rms_curve, 7))
        rms_times = librosa.frames_to_time(
            np.arange(rms_curve.size), sr=sr, hop_length=cfg.hop_length
        )

        boundaries = self._segment_boundaries(
            y, y_harmonic, y_percussive, sr, duration
        )

        sections: List[Section] = []
        for i in range(len(boundaries) - 1):
            start = float(boundaries[i])
            end = float(boundaries[i + 1])
            if end - start < 1.0:
                continue
            mask = (rms_times >= start) & (rms_times < end)
            mean_rms = float(np.mean(rms_curve[mask])) if np.any(mask) else 0.0
            sections.append(
                Section(
                    index=len(sections),
                    start=start,
                    end=end,
                    duration=end - start,
                    mean_rms=mean_rms,
                )
            )

        if not sections:
            sections.append(
                Section(0, 0.0, duration, duration, float(np.mean(rms_curve)))
            )
        return sections

    def _segment_boundaries(
        self,
        y: np.ndarray,
        y_harmonic: np.ndarray,
        y_percussive: np.ndarray,
        sr: int,
        duration: float,
    ) -> np.ndarray:
        """Structural boundary timestamps via agglomerative clustering."""
        cfg = self.cfg
        boundaries: List[float] = [0.0, duration]

        try:
            chroma = librosa.feature.chroma_stft(
                y=y_harmonic, sr=sr, hop_length=cfg.hop_length, n_fft=cfg.n_fft
            )
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=10, hop_length=cfg.hop_length
            )
            onset = librosa.onset.onset_strength(
                y=y_percussive, sr=sr, hop_length=cfg.hop_length
            )[np.newaxis, :]

            min_frames = min(chroma.shape[1], mfcc.shape[1], onset.shape[1])
            frame_features = np.vstack(
                [
                    librosa.util.normalize(chroma[:, :min_frames], axis=1),
                    librosa.util.normalize(mfcc[:, :min_frames], axis=1),
                    librosa.util.normalize(onset[:, :min_frames], axis=1),
                ]
            )
            target = int(
                np.clip(
                    round(duration / cfg.section_target_seconds),
                    cfg.section_min_count,
                    cfg.section_max_count,
                )
            )
            boundary_frames = librosa.segment.agglomerative(frame_features, k=target)
            boundary_times = librosa.frames_to_time(
                boundary_frames, sr=sr, hop_length=cfg.hop_length
            )
            boundaries.extend(float(t) for t in boundary_times if 0.0 < t < duration)
        except Exception:
            # Even spacing fallback if clustering fails.
            step = cfg.section_target_seconds
            boundaries.extend(float(t) for t in np.arange(step, duration, step))

        return self._merge_boundaries(np.asarray(boundaries, dtype=float), duration)

    def _merge_boundaries(self, boundaries: np.ndarray, duration: float) -> np.ndarray:
        """Clamp, dedupe, and enforce a minimum gap between boundaries."""
        min_gap = self.cfg.section_min_seconds
        arr = np.asarray(boundaries, dtype=float)
        arr = arr[np.isfinite(arr)]
        arr = np.unique(np.round(np.clip(arr, 0.0, duration), 3))

        merged: List[float] = []
        for t in arr:
            if not merged:
                merged.append(float(t))
            elif t in (0.0, duration) or t - merged[-1] >= min_gap:
                merged.append(float(t))

        if not merged or abs(merged[0]) > 1e-6:
            merged.insert(0, 0.0)
        if abs(merged[-1] - duration) > 1e-6:
            merged.append(float(duration))
        return np.asarray(merged, dtype=float)


__all__ = ["AudioAnalyzer", "AudioConfig", "AnalysisResult", "Section"]
