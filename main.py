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
        --prompt "shotgun close-range kills" --output final_edit.mp4 \
        [--style_json style.json]

Engine B: pass --style_json to drive dynamic styling (velocity_ramp_peak,
bass_boost_intensity, screen_shake_multiplier, color_grade_preset, cut_density).
Omitting it reproduces the Engine A defaults.

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

CHUNK_SECONDS = 3.0              # length of each raw "candidate clip"
MIN_SEGMENT_SECONDS = 0.75      # min slot length for "low" cut density (downbeats)
MIN_SEGMENT_SECONDS_HIGH = 0.25  # min slot length for "high" cut density (all beats)
DEFAULT_MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"


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


def downbeat_cuts(result, window_start: float, window_end: float,
                  cut_density: str = "low") -> List[float]:
    """Cut points (relative to the window), snapped to the beat grid.

    ``cut_density`` (Engine B):
      * ``"low"``  — cut only on major downbeats (sparser, lands on the bars).
      * ``"high"`` — cut on every detected beat/transient (rapid-fire phonk).

    Each layer falls back across downbeats -> beats -> an even grid if it is too
    sparse, and a density-appropriate minimum gap keeps cuts from getting
    impossibly short. Window start/end are always anchored.
    """
    import numpy as np

    def in_window(times) -> List[float]:
        rel = [float(t) - window_start for t in np.asarray(times)
               if window_start <= float(t) <= window_end]
        return rel

    if cut_density == "high":
        # All transients: prefer the full beat grid, fall back to downbeats.
        grid = in_window(result.beat_times)
        if len(grid) < 3:
            grid = in_window(result.downbeats)
        min_gap = MIN_SEGMENT_SECONDS_HIGH
    else:
        # Major downbeats only (the default "low" density).
        grid = in_window(result.downbeats)
        if len(grid) < 3:
            grid = in_window(result.beat_times)
        min_gap = MIN_SEGMENT_SECONDS

    if len(grid) < 3:                       # analyzer found almost no beats
        grid = list(np.arange(0.0, TIMELINE_SECONDS, 2.0))

    # Always anchor the window's start and end, dedupe, enforce the minimum gap.
    cuts = [0.0]
    for t in sorted(grid):
        t = min(max(t, 0.0), TIMELINE_SECONDS)
        if t - cuts[-1] >= min_gap:
            cuts.append(t)
    if TIMELINE_SECONDS - cuts[-1] >= min_gap:
        cuts.append(TIMELINE_SECONDS)
    else:
        cuts[-1] = TIMELINE_SECONDS

    log("BEATS", f"{len(cuts) - 1} timeline slots (cut_density={cut_density}).")
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
                  window_start: float, window_end: float, downbeat_count: int,
                  style):
    """Velocity-map the best clips onto the beat grid, apply FX, render MP4.

    Each slot is filled with a top-ranked clip that is velocity-ramped (slow-mo
    landing on the beat, then sped up into the next beat at ``style``'s peak)
    and punch-zoomed on the beat (scaled by ``style.screen_shake_multiplier``).
    Each slot's *output* length is held fixed, so the stitched timeline stays
    locked to the strict 30s budget despite all the speed changes. The whole
    timeline is optionally color-graded, and the drop-window music is
    bass-boosted (intensity from ``style``) with an accent pulsed on every beat.
    Every clip is closed in a ``finally`` before the temp dir is removed, which
    keeps it leak-free and safe on Windows (an open file can't be deleted there).
    """
    from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
    from phonk_fx import (
        apply_beat_zoom,
        apply_velocity_ramp,
        apply_color_grade,
        apply_beat_synced_bass_boost,
        resolve_bass_gain,
        DEFAULT_SLOW_FACTOR,
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
            log("EDIT", f"Velocity-mapping top clips onto {n_slots} beat slots "
                        f"(ramp {DEFAULT_SLOW_FACTOR}x on the beat -> "
                        f"{style.velocity_ramp_peak}x into the next; "
                        f"shake x{style.screen_shake_multiplier}).")

            timeline_clips = []
            for i in range(n_slots):
                slot_dur = float(cuts[i + 1] - cuts[i])

                # Cycle through the best clips so a short reel still fills 30s.
                cand = ranked[i % len(ranked)]
                src_start = min(float(cand["start"]),
                                max(0.0, source.duration - 0.05))
                src_end = min(float(cand["end"]), source.duration)
                base = source.subclipped(src_start, src_end)
                opened.append(base)

                # Velocity ramp: slow-mo lands on the beat, then rushes the next
                # at the style's peak speed. Output is exactly slot_dur, so the
                # 30s budget never drifts.
                ramped = apply_velocity_ramp(
                    base, slot_dur, fast_factor=style.velocity_ramp_peak
                )
                opened.append(ramped)

                # Punch-zoom fires at the slot start == the beat, scaled by the
                # style's screen-shake multiplier.
                punched = apply_beat_zoom(
                    ramped, shake_multiplier=style.screen_shake_multiplier
                )
                opened.append(punched)
                timeline_clips.append(punched)

            log("EDIT", "Stitching velocity-mapped clips into the 30s timeline...")
            video = concatenate_videoclips(timeline_clips, method="compose")
            opened.append(video)
            # Hard-clamp to the strict 30s budget regardless of any rounding.
            video = video.with_duration(min(TIMELINE_SECONDS, video.duration))
            opened.append(video)

            # Optional color grade over the whole timeline (one pass).
            if style.color_grade_preset:
                log("EDIT", f"Color grading: {style.color_grade_preset}.")
                video = apply_color_grade(video, style.color_grade_preset)
                opened.append(video)

            # Audio: drop window of the phonk track, bass-boosted (intensity from
            # the style) with an accent pulsed on every beat (single FFmpeg pass).
            base_gain = resolve_bass_gain(style.bass_boost_intensity)
            log("EDIT", f"Bass-boosting drop audio ({style.bass_boost_intensity}/"
                        f"{base_gain}dB) with an accent on {len(beat_grid)} beats...")
            window_audio = AudioFileClip(audio_path).subclipped(window_start, window_end)
            opened.append(window_audio)
            boosted_wav = os.path.join(work, "drop_boosted.wav")
            boosted = apply_beat_synced_bass_boost(
                window_audio, beat_grid, out_path=boosted_wav, base_gain=base_gain
            )
            opened.append(boosted)

            boosted = boosted.with_duration(video.duration)
            opened.append(boosted)
            final = video.with_audio(boosted)
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
    parser.add_argument("--style_json", default=None,
                        help="Path to a JSON style config (Engine B dynamic "
                             "styling: velocity_ramp_peak, bass_boost_intensity, "
                             "screen_shake_multiplier, color_grade_preset, "
                             "cut_density). Optional; omit to use Engine A defaults.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    for label, path in (("video", args.video), ("audio", args.audio)):
        if not os.path.isfile(path):
            print(f"error: --{label} file not found: {path}", file=sys.stderr)
            return 2
    if args.style_json and not os.path.isfile(args.style_json):
        print(f"error: --style_json file not found: {args.style_json}", file=sys.stderr)
        return 2

    try:
        # Engine B: load dynamic styling (or Engine A defaults if no JSON given).
        from phonk_fx import StyleConfig
        style = StyleConfig.from_file(args.style_json) if args.style_json else StyleConfig()
        log("STYLE", f"velocity_peak={style.velocity_ramp_peak}, "
                     f"bass={style.bass_boost_intensity}, "
                     f"shake={style.screen_shake_multiplier}, "
                     f"grade={style.color_grade_preset or 'none'}, "
                     f"cut_density={style.cut_density}.")

        # Stage 1: audio analysis + drop window.
        result, win_start, win_end = analyze_audio(args.audio)
        cuts = downbeat_cuts(result, win_start, win_end, style.cut_density)

        # Need the gameplay duration + fps before chunking/scoring.
        from moviepy import VideoFileClip
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
                      win_start, win_end, downbeat_count=len(cuts) - 1,
                      style=style)

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
