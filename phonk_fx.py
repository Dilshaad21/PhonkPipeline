"""phonk_fx.py — Intense gaming-montage editing effects.

Refactored and generalized from ``auto-gaming-montage-maker`` (edit_video.py).
The original applied effects inline on each highlight subclip:

    target_clip = target_clip.resize(lambda t: 1 + 0.3 * t)   # creeping zoom
    target_audio = target_clip.audio.volumex(0.5)             # duck game audio
    background_sfx = AudioFileClip("editing_sfx/bass_boosted_fixed.mp3")
    combined_audio = CompositeAudioClip([adjusted_audio_clip, background_sfx])

Here those ideas are turned into two standalone, parameterized effects:

    * apply_beat_zoom        — a sudden, smooth "screen jump" punch on a beat.
    * apply_phonk_bass_boost — a real low-shelf bass boost (FFmpeg) that also
                               ducks the peripheral game volume.

Both functions are pure: they take a clip and return a new clip, leaving the
input untouched, so they compose cleanly inside a montage pipeline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

# MoviePy v2.0+ public API (no moviepy.editor; effects live under `vfx`).
from moviepy import AudioFileClip, concatenate_videoclips, vfx


# --------------------------------------------------------------------------- #
# Default phonk tuning. Centralized so a montage pipeline can tweak the feel
# in one place instead of hunting through inline magic numbers.
# --------------------------------------------------------------------------- #
DEFAULT_ZOOM_FACTOR = 1.15      # peak scale at the beat frame
DEFAULT_PUNCH_DURATION = 0.18   # seconds for the zoom to decay back to 1.0
DEFAULT_LOW_GAIN_DB = 12        # low-shelf boost on the kick/bass band
DEFAULT_LOW_FREQ_HZ = 110       # corner frequency of the low shelf
DEFAULT_DUCK_DB = -6            # how much to dip peripheral game volume

# Velocity-ramp feel: slow-motion lands on the beat, then the footage rushes
# (speeds up) into the next beat. Output slot length is held constant so the
# overall 30s timeline budget never drifts (see apply_velocity_ramp).
DEFAULT_FAST_FACTOR = 1.5       # playback speed accelerating into the next beat
DEFAULT_SLOW_FACTOR = 0.5       # slow-motion playback landing on the beat
DEFAULT_SLOW_FRACTION = 0.4     # fraction of each slot spent in slow-motion

# Per-beat low-end accent pulsed on top of the base bass boost.
DEFAULT_ACCENT_GAIN_DB = 6      # extra low-shelf gain in the window around a beat
DEFAULT_ACCENT_PRE = 0.03       # seconds the accent opens before the beat
DEFAULT_ACCENT_POST = 0.14      # seconds the accent stays open after the beat


# ========================================================================== #
#  BEAT ZOOM
# ========================================================================== #
def _punch_scale(t: float, zoom_factor: float, punch_duration: float) -> float:
    """Scale at time ``t``: spikes to ``zoom_factor`` on the beat, then eases
    back to 1.0 over ``punch_duration`` seconds (ease-out / quadratic decay).

    A linear ramp (as in the original ``1 + 0.3 * t``) drifts forever; this
    snaps in on the beat and settles, which reads as a hit rather than a creep.
    """
    if t >= punch_duration:
        return 1.0
    progress = t / punch_duration              # 0 -> 1 across the punch
    eased = (1.0 - progress) ** 2              # 1 -> 0, fast-out
    return 1.0 + (zoom_factor - 1.0) * eased


def apply_beat_zoom(video_clip, zoom_factor: float = DEFAULT_ZOOM_FACTOR,
                    punch_duration: float = DEFAULT_PUNCH_DURATION):
    """Apply a sudden, smooth scale "screen jump" on the beat to ``video_clip``.

    The clip is scaled up at t=0 and decays back to its native size, then
    centre-cropped back to the original resolution so the output stream keeps
    the exact same width/height (no letterboxing, no dimension changes
    downstream). Built on MoviePy so it slots straight into a clip pipeline.

    Parameters
    ----------
    video_clip : moviepy VideoClip
        The highlight clip to punch on a beat.
    zoom_factor : float
        Peak scale at the beat frame (1.15 == a 15 % jump).
    punch_duration : float
        Seconds for the zoom to relax back to 1.0.

    Returns
    -------
    moviepy VideoClip
        A new clip; the input is not mutated.
    """
    w, h = video_clip.size

    # MoviePy 2.x: `resized` accepts a time-varying scale callable.
    zoomed = video_clip.resized(
        lambda t: _punch_scale(t, zoom_factor, punch_duration)
    )

    # Re-crop each (variably upscaled) frame to the original resolution so the
    # output stream keeps constant dimensions. `image_transform` runs a pure
    # frame->frame function in 2.x (the old `.fl` is gone).
    return zoomed.image_transform(lambda frame: _center_crop_frame(frame, w, h))


def _center_crop_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Centre-crop a single (possibly upscaled) frame to ``target_w`` x ``target_h``."""
    fh, fw = frame.shape[:2]
    x0 = max((fw - target_w) // 2, 0)
    y0 = max((fh - target_h) // 2, 0)
    return frame[y0:y0 + target_h, x0:x0 + target_w]


# --------------------------------------------------------------------------- #
#  Raw-stream variant: same punch on a single OpenCV/numpy frame.
#  Useful when manipulating the decoded video stream directly rather than
#  through MoviePy (e.g. an ffmpeg rawvideo pipe like video_remove_silence.py).
# --------------------------------------------------------------------------- #
def zoom_frame(frame: np.ndarray, zoom: float) -> np.ndarray:
    """Scale a single BGR frame about its centre by ``zoom`` and crop back to
    the original size. ``zoom`` of 1.0 is a no-op.

    Drive this per-frame from a beat schedule using :func:`_punch_scale` to
    reproduce :func:`apply_beat_zoom` on a raw decoded stream.
    """
    if zoom == 1.0:
        return frame
    h, w = frame.shape[:2]
    scaled = cv2.resize(frame, None, fx=zoom, fy=zoom,
                        interpolation=cv2.INTER_LINEAR)
    sh, sw = scaled.shape[:2]
    x0 = (sw - w) // 2
    y0 = (sh - h) // 2
    return scaled[y0:y0 + h, x0:x0 + w]


# ========================================================================== #
#  VELOCITY RAMP  (dynamic speed mapping)
# ========================================================================== #
def apply_velocity_ramp(video_clip, target_duration: float,
                        slow_factor: float = DEFAULT_SLOW_FACTOR,
                        fast_factor: float = DEFAULT_FAST_FACTOR,
                        slow_fraction: float = DEFAULT_SLOW_FRACTION):
    """Map a clip onto a beat: slow-motion on the beat, then rush into the next.

    The output is split into two phases that read as the signature phonk
    "impact then chase" feel:

      * a slow-motion *head* (``slow_factor``, e.g. 0.5x) that lands exactly on
        the beat at the clip's start, and
      * a sped-up *tail* (``fast_factor``, e.g. 1.5x) that accelerates into the
        next beat.

    Crucially the returned clip is **exactly** ``target_duration`` seconds long
    regardless of the speed factors: the source footage consumed by each phase
    is sized so that, after ``vfx.MultiplySpeed``, the phase plays for its slice
    of the slot. This keeps a stitched sequence locked to its time budget even
    though the footage inside is speeding up and slowing down. A trailing
    ``with_duration`` clamps away any sub-millisecond rounding drift.

    Pure: returns a new clip; ``video_clip`` is not mutated.

    Parameters
    ----------
    video_clip : moviepy VideoClip
        Source highlight footage (should be at least ~``target_duration``
        seconds of usable material; it is clamped if shorter).
    target_duration : float
        The exact output length this slot must occupy on the timeline.
    slow_factor, fast_factor : float
        Playback multipliers for the slow-mo head and the sped-up tail.
    slow_fraction : float
        Portion of the output spent in slow-motion (0..1).
    """
    if target_duration <= 0:
        raise ValueError("target_duration must be positive.")

    slow_fraction = min(max(slow_fraction, 0.0), 1.0)
    out_slow = target_duration * slow_fraction      # output secs of slow-mo
    out_fast = target_duration - out_slow           # output secs of speed-up

    src_total = float(video_clip.duration or 0.0)
    # Source footage each phase needs: output_secs * speed_factor.
    need_slow = out_slow * slow_factor
    need_fast = out_fast * fast_factor
    need_total = need_slow + need_fast

    # If the clip is too short, shrink both phases proportionally; the final
    # set_duration then stretches the result back to fill the slot.
    if need_total > src_total and need_total > 1e-6:
        scale = src_total / need_total
        need_slow *= scale
        need_fast *= scale

    parts = []
    if need_slow > 1e-3 and out_slow > 1e-3:
        head = video_clip.subclipped(0, need_slow).with_effects(
            [vfx.MultiplySpeed(slow_factor)]
        )
        parts.append(head)
    if need_fast > 1e-3 and out_fast > 1e-3:
        start = need_slow
        tail = video_clip.subclipped(start, start + need_fast).with_effects(
            [vfx.MultiplySpeed(fast_factor)]
        )
        parts.append(tail)

    if not parts:
        ramped = video_clip.subclipped(
            0, min(target_duration, src_total or target_duration)
        )
    elif len(parts) == 1:
        ramped = parts[0]
    else:
        ramped = concatenate_videoclips(parts, method="compose")

    # Enforce the exact slot budget so the 30s total can never drift.
    return ramped.with_duration(target_duration)


# ========================================================================== #
#  PHONK BASS BOOST
# ========================================================================== #
def build_bass_boost_filter(low_gain: int = DEFAULT_LOW_GAIN_DB,
                            low_freq: int = DEFAULT_LOW_FREQ_HZ,
                            duck_db: int = DEFAULT_DUCK_DB) -> str:
    """Return the raw FFmpeg ``-af`` filter chain for a phonk bass boost.

    Chain:
      * ``bass``       — low-shelf gain of ``low_gain`` dB below ``low_freq``,
                         emphasizing the low-end kicks.
      * ``volume``     — dips the peripheral game audio by ``duck_db`` dB so the
                         boosted low end sits forward on the highlight.
      * ``alimiter``   — soft brick-wall so the boost can't clip the master.

    Exposed separately so callers can reuse the exact parameters with their own
    FFmpeg invocation if they aren't going through MoviePy.
    """
    return (
        f"bass=g={low_gain}:f={low_freq}:width_type=q:width=0.7,"
        f"volume={duck_db}dB,"
        f"alimiter=limit=0.95"
    )


def apply_phonk_bass_boost(audio_clip, low_gain: int = DEFAULT_LOW_GAIN_DB,
                           low_freq: int = DEFAULT_LOW_FREQ_HZ,
                           duck_db: int = DEFAULT_DUCK_DB):
    """Emphasize low-end kicks while ducking peripheral game volume.

    The clip's audio is rendered out to a temp WAV, processed through FFmpeg's
    EQ/limiter chain (see :func:`build_bass_boost_filter`), and read back as a
    fresh ``AudioFileClip``. This replaces the original's pre-baked
    ``bass_boosted_fixed.mp3`` SFX with a real, parameterized low-shelf boost.

    Parameters
    ----------
    audio_clip : moviepy AudioClip
        The highlight-moment audio to boost.
    low_gain : int
        Low-shelf gain in dB on the bass band (the headline "bass boost" knob).
    low_freq : int
        Corner frequency (Hz) of the low shelf.
    duck_db : int
        Negative dB applied to the whole clip to dip peripheral game volume.

    Returns
    -------
    moviepy AudioFileClip
        A new clip backed by the processed WAV.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; required for bass boost.")

    fd_in, src_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd_in)
    fd_out, dst_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd_out)

    try:
        audio_clip.write_audiofile(src_path, logger=None)

        filter_chain = build_bass_boost_filter(low_gain, low_freq, duck_db)
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", src_path,
            "-af", filter_chain,
            dst_path,
        ]
        subprocess.run(cmd, check=True)

        # AudioFileClip lazily reads dst_path; keep the temp file alive by
        # returning a clip bound to it (caller owns the lifetime via .close()).
        return AudioFileClip(dst_path)
    finally:
        if os.path.exists(src_path):
            os.remove(src_path)
        # dst_path is intentionally left for the returned clip to read.


def build_beat_synced_bass_filter(beat_times,
                                  base_gain: int = DEFAULT_LOW_GAIN_DB,
                                  accent_gain: int = DEFAULT_ACCENT_GAIN_DB,
                                  low_freq: int = DEFAULT_LOW_FREQ_HZ,
                                  duck_db: int = DEFAULT_DUCK_DB,
                                  accent_pre: float = DEFAULT_ACCENT_PRE,
                                  accent_post: float = DEFAULT_ACCENT_POST) -> str:
    """FFmpeg ``-af`` chain: a base bass boost plus a low-end accent on each beat.

    Unlike :func:`build_bass_boost_filter`, this varies *in time*: a second
    low-shelf stage adds ``accent_gain`` dB only inside short windows around each
    beat (via FFmpeg's timeline ``enable`` expression), so the kicks punch
    harder exactly on the beat. ``beat_times`` are seconds **relative to the
    audio clip** being processed (i.e. relative to the drop window, not the full
    song). Everything is one pass — no per-segment concatenation, so there are
    no boundary clicks.
    """
    chain = [f"bass=g={base_gain}:f={low_freq}:width_type=q:width=0.7"]

    windows = [
        f"between(t,{max(0.0, float(b) - accent_pre):.3f},{float(b) + accent_post:.3f})"
        for b in beat_times
    ]
    if windows:
        # Disjoint between() terms summed → nonzero (enabled) only on a beat.
        enable = "+".join(windows)
        chain.append(
            f"bass=g={accent_gain}:f={low_freq}:width_type=q:width=0.7"
            f":enable='{enable}'"
        )

    chain.append(f"volume={duck_db}dB")
    chain.append("alimiter=limit=0.95")
    return ",".join(chain)


def apply_beat_synced_bass_boost(audio_clip, beat_times, out_path=None,
                                 base_gain: int = DEFAULT_LOW_GAIN_DB,
                                 accent_gain: int = DEFAULT_ACCENT_GAIN_DB,
                                 low_freq: int = DEFAULT_LOW_FREQ_HZ,
                                 duck_db: int = DEFAULT_DUCK_DB):
    """Bass-boost ``audio_clip`` with an extra low-end pulse on every beat.

    Renders the clip to a temp WAV, runs the beat-synced FFmpeg chain (see
    :func:`build_beat_synced_bass_filter`) in a single pass, and returns a fresh
    ``AudioFileClip`` backed by the processed WAV. If the timeline-gated accent
    filter isn't supported by the local FFmpeg build, it falls back to the plain
    whole-clip boost so the render still succeeds.

    Parameters
    ----------
    audio_clip : moviepy AudioClip
        The drop-window music to process.
    beat_times : sequence of float
        Beat timestamps in seconds *relative to this clip's start*.
    out_path : str or os.PathLike, optional
        Where to write the processed WAV. If omitted a temp file is created.
        The returned clip reads from this path lazily, so the **caller owns its
        lifetime**: close the returned clip before deleting the file (important
        on Windows, where an open file cannot be removed).

    Returns
    -------
    moviepy AudioFileClip
        A new clip backed by the processed WAV.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; required for bass boost.")

    fd_in, src_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd_in)

    created_out = False
    if out_path is None:
        fd_out, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd_out)
        created_out = True
    else:
        out_path = os.fspath(out_path)

    try:
        audio_clip.write_audiofile(src_path, logger=None)

        filter_chain = build_beat_synced_bass_filter(
            beat_times, base_gain, accent_gain, low_freq, duck_db
        )
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", src_path, "-af", filter_chain, out_path,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            # Older FFmpeg without timeline 'enable' support on the bass filter:
            # fall back to a plain, whole-clip boost so the pipeline still runs.
            fallback = build_bass_boost_filter(base_gain, low_freq, duck_db)
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y",
                 "-i", src_path, "-af", fallback, out_path],
                check=True,
            )

        return AudioFileClip(out_path)
    except BaseException:
        # On failure, don't leak the temp output we created.
        if created_out and os.path.exists(out_path):
            os.remove(out_path)
        raise
    finally:
        if os.path.exists(src_path):
            os.remove(src_path)


__all__ = [
    "apply_beat_zoom",
    "apply_velocity_ramp",
    "apply_phonk_bass_boost",
    "apply_beat_synced_bass_boost",
    "build_bass_boost_filter",
    "build_beat_synced_bass_filter",
    "zoom_frame",
    "DEFAULT_ZOOM_FACTOR",
    "DEFAULT_PUNCH_DURATION",
    "DEFAULT_LOW_GAIN_DB",
    "DEFAULT_LOW_FREQ_HZ",
    "DEFAULT_DUCK_DB",
    "DEFAULT_FAST_FACTOR",
    "DEFAULT_SLOW_FACTOR",
    "DEFAULT_SLOW_FRACTION",
    "DEFAULT_ACCENT_GAIN_DB",
]
