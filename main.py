#!/usr/bin/env python3
"""main.py — Master orchestrator for the Phonk gaming-montage pipeline.

Ties the three distilled root modules into one end-to-end render:

    1. audio_analyzer  — find the song's primary drop + the beat/downbeat grid.
    2. (chunking)      — pre-slice the raw gameplay VOD into 3s candidate clips.
    3. semantic_filter — score every candidate against the user's action prompt
                         and keep the highest-relevance clips.
    4. (beat match)    — lay the best clips onto a strict 30s timeline whose cuts
                         snap to the detected downbeats of the drop.
    5. phonk_fx        — punch-zoom each clip on its downbeat and bass-boost the
                         drop-window audio, then mux to a polished 30s MP4.

Usage:
    python3 main.py --video gameplay.mp4 --audio phonk.mp3 \
        --prompt "shotgun close-range kills" --output final_edit.mp4

Runtime deps are inherited from the modules it drives: librosa, torch +
transformers + a local Qwen3-VL checkpoint, moviepy + opencv, and a system
ffmpeg on PATH. See CLAUDE.md — none are installed by default.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from typing import Dict, List

# --- pipeline geometry ------------------------------------------------------
WINDOW_BEFORE_DROP = 10.0   # seconds of timeline before the drop
WINDOW_AFTER_DROP = 20.0    # seconds after — total strict window = 30s
TIMELINE_SECONDS = WINDOW_BEFORE_DROP + WINDOW_AFTER_DROP

CHUNK_SECONDS = 3.0         # length of each raw "candidate clip"
MIN_SEGMENT_SECONDS = 0.75  # don't cut faster than this even on dense beats
DEFAULT_MODEL_PATH = "bin/models/Qwen3-VL-2B-Instruct"


def log(step: str, message: str) -> None:
    """Single-line, prefixed progress print so each stage is easy to follow."""
    print(f"[{step}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Stage 1 — audio analysis + drop window
# ---------------------------------------------------------------------------
def analyze_audio(audio_path: str):
    """Run AudioAnalyzer and derive the strict 30s window around the main drop.

    Returns (result, window_start, window_end).
    """
    from audio_analyzer import AudioAnalyzer

    log("AUDIO", f"Analyzing '{audio_path}' (tempo, beats, sections)...")
    result = AudioAnalyzer().analyze(audio_path)

    if result.main_drop is None:
        raise RuntimeError("No sections detected in the audio; cannot locate a drop.")

    drop = result.main_drop
    log("AUDIO", f"Tempo ~{result.tempo:.1f} BPM, {len(result.beat_times)} beats, "
                 f"{len(result.downbeats)} downbeats over {result.duration:.1f}s.")
    log("AUDIO", f"Main drop section #{drop.index}: {drop.start:.2f}s–{drop.end:.2f}s "
                 f"(energy {drop.mean_rms:.2f}).")

    # 10s before the drop onset, 20s after — clamped into the track, fixed 30s.
    window_start = max(0.0, drop.start - WINDOW_BEFORE_DROP)
    window_end = window_start + TIMELINE_SECONDS
    if window_end > result.duration:
        window_end = result.duration
        window_start = max(0.0, window_end - TIMELINE_SECONDS)

    log("AUDIO", f"Drop window locked: {window_start:.2f}s–{window_end:.2f}s "
                 f"({window_end - window_start:.1f}s).")
    return result, window_start, window_end


def downbeat_cuts(result, window_start: float, window_end: float) -> List[float]:
    """Cut points (relative to the window) snapped to detected downbeats.

    These become the clip boundaries so every cut lands on a downbeat. Falls
    back to the full beat grid, then to an even grid, if downbeats are sparse.
    """
    import numpy as np

    def in_window(times) -> List[float]:
        rel = [float(t) - window_start for t in np.asarray(times)
               if window_start <= float(t) <= window_end]
        return rel

    grid = in_window(result.downbeats)
    if len(grid) < 3:                       # too few bars in the window
        grid = in_window(result.beat_times)
    if len(grid) < 3:                       # analyzer found almost no beats
        step = 2.0
        grid = list(np.arange(0.0, TIMELINE_SECONDS, step))

    # Always anchor the window's start and end, dedupe, enforce a minimum gap.
    cuts = [0.0]
    for t in sorted(grid):
        t = min(max(t, 0.0), TIMELINE_SECONDS)
        if t - cuts[-1] >= MIN_SEGMENT_SECONDS:
            cuts.append(t)
    if TIMELINE_SECONDS - cuts[-1] >= MIN_SEGMENT_SECONDS:
        cuts.append(TIMELINE_SECONDS)
    else:
        cuts[-1] = TIMELINE_SECONDS

    log("BEATS", f"{len(cuts) - 1} downbeat-aligned timeline slots.")
    return cuts


# ---------------------------------------------------------------------------
# Stage 2 — chunk the gameplay VOD into candidate clips
# ---------------------------------------------------------------------------
def chunk_video(video_duration: float) -> List[Dict]:
    """Pre-slice the VOD into back-to-back 3s candidate segments.

    Each candidate is a dict {id, start, end} — the shape semantic_filter's
    score_video_segments expects.
    """
    segments: List[Dict] = []
    start = 0.0
    idx = 0
    while start + CHUNK_SECONDS <= video_duration:
        segments.append({"id": str(idx), "start": start, "end": start + CHUNK_SECONDS})
        start += CHUNK_SECONDS
        idx += 1

    if not segments:
        raise RuntimeError(
            f"Gameplay video is only {video_duration:.1f}s; need at least "
            f"{CHUNK_SECONDS:.0f}s to chunk into candidate clips."
        )
    log("CHUNK", f"Sliced gameplay into {len(segments)} x {CHUNK_SECONDS:.0f}s "
                 f"candidate clips.")
    return segments


# ---------------------------------------------------------------------------
# Stage 3 — semantic relevance scoring
# ---------------------------------------------------------------------------
def rank_candidates(video_path: str, segments: List[Dict], prompt: str,
                    fps: float, model_path: str) -> List[Dict]:
    """Score every candidate against the prompt and return them best-first.

    Each returned dict carries the source {start, end} plus the AI confidence.
    """
    from semantic_filter import SemanticActionFilter

    log("FILTER", f"Loading Qwen-VL from '{model_path}'...")
    flt = SemanticActionFilter(model_path)

    log("FILTER", f"Scoring {len(segments)} clips against: \"{prompt}\"")
    scores = flt.score_video_segments(video_path, segments, prompt, fps=fps)

    by_id = {seg["id"]: seg for seg in segments}
    ranked: List[Dict] = []
    for s in scores:
        seg = by_id.get(s.id)
        if seg is None:
            continue
        ranked.append({
            "start": seg["start"],
            "end": seg["end"],
            "confidence": s.confidence,
            "match": s.match,
            "description": s.description,
        })

    # Highest relevance first; matched clips outrank non-matches at equal score.
    ranked.sort(key=lambda c: (c["match"], c["confidence"]), reverse=True)

    if ranked:
        matches = sum(1 for c in ranked if c["match"])
        log("FILTER", f"{matches}/{len(ranked)} clips matched the prompt. "
                      f"Top score {ranked[0]['confidence']:.2f}.")
    else:
        log("FILTER", "No clips scored.")
    return ranked


# ---------------------------------------------------------------------------
# Stage 4 + 5 — assemble the beat-matched, phonk-styled timeline
# ---------------------------------------------------------------------------
def build_montage(video_path: str, audio_path: str, output_path: str,
                  ranked: List[Dict], cuts: List[float],
                  window_start: float, window_end: float, downbeat_count: int):
    """Velocity-map the best clips onto the downbeat grid, apply FX, render MP4.

    Each downbeat slot is filled with a top-ranked clip that is velocity-ramped
    (slow-mo landing on the beat, then sped up into the next beat) and
    punch-zoomed on the beat. Each slot's *output* length is held fixed, so the
    stitched timeline stays locked to the strict 30s budget despite all the
    speed changes. The drop-window music is bass-boosted with an accent pulsed
    on every beat. Every clip is closed in a ``finally`` before the temp dir is
    removed, which keeps it leak-free and safe on Windows (an open file can't be
    deleted there).
    """
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
    from phonk_fx import (
        apply_beat_zoom,
        apply_velocity_ramp,
        apply_beat_synced_bass_boost,
        DEFAULT_SLOW_FACTOR,
        DEFAULT_FAST_FACTOR,
    )

    if not ranked:
        raise RuntimeError("No candidate clips to assemble the timeline from.")

    # One beat per slot start; cuts are already downbeat-snapped + window-relative.
    beat_grid = [round(float(c), 3) for c in cuts[:-1]]

    # The temp dir holds the processed-audio WAV. It is torn down only after every
    # clip is closed (below), so Windows can delete the no-longer-open file.
    with tempfile.TemporaryDirectory(prefix="phonk_render_") as work:
        opened = []  # every clip to .close(); closed in reverse in the finally
        try:
            # Drop the gameplay audio up front: we score on visuals and the final
            # track is the music, so this saves memory and open readers.
            source = VideoFileClip(video_path, audio=False)
            opened.append(source)

            n_slots = len(cuts) - 1
            log("EDIT", f"Velocity-mapping top clips onto {n_slots} downbeat slots "
                        f"(ramp {DEFAULT_SLOW_FACTOR}x on the beat -> "
                        f"{DEFAULT_FAST_FACTOR}x into the next).")

            timeline_clips = []
            for i in range(n_slots):
                slot_dur = float(cuts[i + 1] - cuts[i])

                # Cycle through the best clips so a short reel still fills 30s.
                cand = ranked[i % len(ranked)]
                src_start = min(float(cand["start"]),
                                max(0.0, source.duration - 0.05))
                src_end = min(float(cand["end"]), source.duration)
                base = source.subclip(src_start, src_end)
                opened.append(base)

                # Velocity ramp: slow-mo lands on the beat, then rushes the next.
                # Output is exactly slot_dur, so the 30s budget never drifts.
                ramped = apply_velocity_ramp(base, slot_dur)
                opened.append(ramped)

                # Punch-zoom fires at the slot start == the beat.
                punched = apply_beat_zoom(ramped)
                opened.append(punched)
                timeline_clips.append(punched)

            log("EDIT", "Stitching velocity-mapped clips into the 30s timeline...")
            video = concatenate_videoclips(timeline_clips, method="compose")
            opened.append(video)
            # Hard-clamp to the strict 30s budget regardless of any rounding.
            video = video.set_duration(min(TIMELINE_SECONDS, video.duration))
            opened.append(video)

            # Audio: drop window of the phonk track, bass-boosted with an accent
            # pulsed on every beat (single FFmpeg pass; see phonk_fx).
            log("EDIT", f"Bass-boosting drop audio with an accent on "
                        f"{len(beat_grid)} beats (FFmpeg)...")
            window_audio = AudioFileClip(audio_path).subclip(window_start, window_end)
            opened.append(window_audio)
            boosted_wav = os.path.join(work, "drop_boosted.wav")
            boosted = apply_beat_synced_bass_boost(
                window_audio, beat_grid, out_path=boosted_wav
            )
            opened.append(boosted)

            boosted = boosted.set_duration(video.duration)
            opened.append(boosted)
            final = video.set_audio(boosted)
            opened.append(final)

            log("RENDER", f"Writing polished montage to '{output_path}'...")
            final.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=source.fps or 30,
                threads=os.cpu_count() or 4,
                logger=None,
            )
            log("RENDER", f"Done: {output_path} ({final.duration:.1f}s, "
                          f"{downbeat_count} downbeats).")
        finally:
            # Reverse order: close composites before the readers they wrap.
            for clip in reversed(opened):
                try:
                    clip.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Assemble a 30s beat-synced phonk gaming montage.",
    )
    parser.add_argument("--video", required=True, help="Raw gameplay VOD (mp4).")
    parser.add_argument("--audio", required=True, help="Phonk music track (mp3/wav).")
    parser.add_argument("--prompt", required=True,
                        help='Action to highlight, e.g. "shotgun close-range kills".')
    parser.add_argument("--output", required=True, help="Output montage path (mp4).")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help=f"Local Qwen-VL checkpoint (default: {DEFAULT_MODEL_PATH}).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    for label, path in (("video", args.video), ("audio", args.audio)):
        if not os.path.isfile(path):
            print(f"error: --{label} file not found: {path}", file=sys.stderr)
            return 2

    try:
        # Stage 1: audio analysis + drop window.
        result, win_start, win_end = analyze_audio(args.audio)
        cuts = downbeat_cuts(result, win_start, win_end)

        # Need the gameplay duration + fps before chunking/scoring.
        from moviepy.editor import VideoFileClip
        probe = VideoFileClip(args.video)
        try:
            video_duration = probe.duration
            video_fps = probe.fps or 24.0
        finally:
            probe.close()
        log("CHUNK", f"Gameplay VOD: {video_duration:.1f}s @ {video_fps:.0f}fps.")

        # Stage 2: chunk into candidate clips.
        segments = chunk_video(video_duration)

        # Stage 3: semantic relevance ranking.
        ranked = rank_candidates(args.video, segments, args.prompt,
                                 video_fps, args.model_path)

        # Stages 4 + 5: beat-match, phonk FX, render.
        build_montage(args.video, args.audio, args.output, ranked, cuts,
                      win_start, win_end, downbeat_count=len(cuts) - 1)

        print(f"\n✅ Montage complete: {args.output}")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n❌ Pipeline failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
