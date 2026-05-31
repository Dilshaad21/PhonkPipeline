# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

PhonkPipeline assembles an automated gaming-montage pipeline by **distilling** logic out of three vendored upstream projects into clean, standalone, dependency-light modules at the repo root. The vendored directories are *reference source*, not the deliverable — the deliverable is the small set of root-level `*.py` modules.

The core workflow has two stages:

1. **Pack** — `pack_repos.py` walks a vendored repo and concatenates every `.py` file into a single `*_source.txt` bundle (skipping hidden dirs, `venv`, `__pycache__`). This produces a flat, greppable view of an entire upstream project.
2. **Distill** — the packed `*_source.txt` is read to harvest a specific capability, which is then rewritten as a focused root module. Distilled modules strip the upstream's framework glue (Gradio UIs, worker-process IPC, env-var batch tuning, progress callbacks, logging) and keep only the pure algorithm. Each module's top docstring records *what was stripped and what core was kept* — read it before changing a module.

When asked to "harvest"/"extract"/"distill" a capability, the expected pattern is: locate it in the relevant `*_source.txt`, then produce a new standalone root module (pure functions/classes, minimal deps, docstring noting provenance and what was stripped).

## Layout

**Vendored upstream sources** (read-only references; regenerate packs from these):
- `BeatSync-Engine/` — beat-synced editing engine (librosa analysis + Qwen-VL scene scoring). Packed into `beatsync_source.txt`.
- `auto-gaming-montage-maker/` — YOLO/template-matching highlight detector + MoviePy effects. Packed into `montage_fx_source.txt`.
- `crispy/` — additional reference project (`crispy-api`, `crispy-frontend`); not yet packed.

**Distilled root modules** (the actual product):
- `audio_analyzer.py` — from BeatSync-Engine's `auto_mode` stages (stage1–3). `AudioAnalyzer.analyze(path)` → `AnalysisResult` with tempo/BPM, a beat grid, bar-aligned downbeats, and the highest-RMS "main drop" section. Pure librosa.
- `semantic_filter.py` — from BeatSync-Engine's `stage5_qwen_scene_worker`. `SemanticActionFilter` scores frames/video segments against a free-text action query (e.g. "shotgun kill") via a local Qwen3-VL model. PyTorch/Transformers.
- `phonk_fx.py` — from auto-gaming-montage-maker's `packagefiles/edit_video.py`. `apply_beat_zoom` (MoviePy punch-zoom + a raw-frame `zoom_frame` variant) and `apply_phonk_bass_boost` (real FFmpeg low-shelf EQ + game-audio duck; `build_bass_boost_filter` exposes the raw `-af` string).

## Conventions for distilled modules

- **Standalone and importable** — each module is a library used via the API shown in its docstring's `Usage:` block; there is no root CLI/build/test harness. Functions are pure (return new objects, don't mutate inputs) so they compose in a pipeline.
- **Config as a frozen dataclass** — tuning lives in a `*Config` dataclass (`AudioConfig`, `SemanticConfig`) or module-level `DEFAULT_*` constants, not inline magic numbers. Defaults are chosen to match the upstream's behavior; preserve that when refactoring.
- **Results as dataclasses** — analysis outputs are dataclasses (`AnalysisResult`, `Section`, `FrameScore`), not loose tuples/dicts.

## Commands

```bash
# Regenerate the packed source bundles after pulling/changing vendored repos:
python pack_repos.py        # -> beatsync_source.txt, montage_fx_source.txt
```

There is no root-level test/build/lint setup or `requirements.txt`. Runtime deps are per-module and heavy: `librosa` (audio_analyzer), `moviepy` + `opencv-python` + a system `ffmpeg` on PATH (phonk_fx), `torch` + `transformers` + a local Qwen3-VL checkpoint under `bin/models/` (semantic_filter). These are not installed in the working environment by default — syntax-check edits, but expect imports to fail without the deps.
