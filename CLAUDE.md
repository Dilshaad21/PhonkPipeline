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

There is no root-level test/build/lint setup or `requirements.txt`. Runtime deps are per-module and heavy: `librosa` (audio_analyzer), `moviepy` + `opencv-python` + a system `ffmpeg` on PATH (phonk_fx), `torch` + `transformers` + a Qwen-VL model (semantic_filter). These are not installed in the working environment by default — syntax-check edits, but expect imports to fail without the deps.

## Windows setup

Developed on macOS, deployed on Windows. The code is cross-platform — all path handling goes through `os.path`/`pathlib`, and render temp files are explicitly closed before their containing dir is removed so Windows can delete them (an open file can't be unlinked on Windows). The environment, however, must be provisioned per-OS. There is no `requirements.txt`; install the per-module deps directly.

1. **Python + virtual environment** (Python 3.9+; on Windows the launcher is `python`, not `python3`):
   ```bat
   python -m venv phonk_env
   phonk_env\Scripts\activate
   python -m pip install --upgrade pip
   ```
   (In PowerShell the activate script is `phonk_env\Scripts\Activate.ps1`; if it's blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)

2. **Python dependencies:**
   ```bat
   pip install librosa "moviepy>=2.0" opencv-python numpy Pillow transformers
   :: CPU-only PyTorch:
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   :: ...or for an NVIDIA GPU (CUDA 12.1), use this instead of the CPU line:
   :: pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
   **MoviePy 2.0+ is required.** `phonk_fx.py`/`main.py` use the v2 API (`clip.with_effects([vfx.MultiplySpeed(...)])`, `subclipped`, `with_audio`, `with_duration`, `resized`, `image_transform`); the 1.x API (`.fx`, `.subclip`, `.set_audio`) will not work.

3. **FFmpeg on PATH** (required by `phonk_fx`'s bass boost and by MoviePy's reader/writer):
   ```bat
   winget install Gyan.FFmpeg
   :: ...or: choco install ffmpeg
   ```
   Open a **new** terminal afterward (so the PATH change takes effect) and confirm `ffmpeg -version` resolves.

4. **Qwen-VL model.** `main.py`'s default `--model-path` is the Hugging Face Hub repo id `Qwen/Qwen2-VL-2B-Instruct`, downloaded on first run into the HF cache (`%USERPROFILE%\.cache\huggingface`; relocate by setting `HF_HOME`). `semantic_filter` auto-detects: a Hub repo id is allowed to download, while an existing **local directory** path is loaded with `local_files_only=True` (fully offline). To run offline, pre-download the model (or copy a checkpoint dir) and pass that directory to `--model-path`. If a tokenizer load fails with `expected str, bytes or os.PathLike object, not NoneType` (a partial cache), delete `%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen2-VL-2B-Instruct` and re-run.

Validate the environment, then run the pipeline:
```bat
python smoke_test.py
python main.py --video gameplay.mp4 --audio phonk.mp3 ^
    --prompt "shotgun close-range kills" --output final_edit.mp4
```
`smoke_test.py` reports `[PASS]`/`[FAIL]` per stage (dependency imports, file layout, CLI/pipeline flow). `^` above is the **cmd.exe** line-continuation; in PowerShell use a backtick `` ` `` (or just put it all on one line).
