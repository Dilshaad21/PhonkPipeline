#!/usr/bin/env python3
"""Localized Qwen-VL semantic action filter for the Phonk pipeline.

Distilled from BeatSync-Engine's stage5_qwen_scene_worker. The standalone
worker process, request/response JSON files, env-var batch tuning sprawl, and
console logging have been stripped. What remains is the core PyTorch /
Transformers inference path, repurposed for a single job:

    Given a free-text action query (e.g. "shotgun kill", "sniper execution")
    and a sequence of image frames or video snippets, score each one on whether
    it depicts that action and optionally filter to the matches.

Usage:
    flt = SemanticActionFilter("bin/models/Qwen3-VL-2B-Instruct")

    # From in-memory frames:
    scores = flt.score_images(pil_images, "shotgun kill")

    # From a video, sampling the mid-frame of each (start, end) segment:
    matches = flt.filter_video_segments(
        "clip.mp4", segments=[{"id": "0", "start": 1.0, "end": 2.5}],
        query="sniper execution", fps=24.0, threshold=0.6,
    )
"""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Config + result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticConfig:
    frame_max_width: int = 512   # downscale wide frames before encoding
    max_new_tokens: int = 96
    batch_size: int = 8          # frames per forward pass
    torch_dtype: str = "auto"    # auto | float32 | float16 | bfloat16
    attn_implementation: str = "sdpa"


@dataclass
class FrameScore:
    """One frame's verdict against the action query."""

    id: str
    match: bool
    confidence: float          # 0..1, model's certainty the action is present
    description: str = ""

    @property
    def is_match(self) -> bool:
        return self.match


# ---------------------------------------------------------------------------
# Parsing / frame helpers (ported from the worker)
# ---------------------------------------------------------------------------


def _clamp(value, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not np.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _parse_json_object(text: str) -> Dict:
    """Best-effort extraction of a JSON object from raw model output."""
    if not text:
        return {}
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(
        frame, (max_width, max(2, int(h * scale))), interpolation=cv2.INTER_AREA
    )


def _frame_to_image(frame: Optional[np.ndarray], max_width: int) -> Optional[Image.Image]:
    if frame is None:
        return None
    frame = _resize_frame(frame, max_width=max_width)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _segment_mid_frame(fps: float, segment: Dict) -> int:
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    t = start + max(0.01, end - start) * 0.5
    return max(0, int(round(t * fps)))


def _is_out_of_memory(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_action_prompt(query: str) -> str:
    """Ask Qwen-VL to judge a single frame against a free-text action query."""
    query = query.strip()
    return (
        "You are reviewing one video frame for an action-matching task. "
        f'Decide whether the frame depicts the following action: "{query}". '
        "Judge only what is visibly happening in the image. "
        "Return JSON only, no markdown. Keys: "
        "match as true or false (true only if the action is clearly present); "
        "confidence as a number 0..1 (your certainty the action is present); "
        "description under 12 words summarizing what is happening."
    )


def _normalize_verdict(data: Dict) -> Dict:
    match_raw = data.get("match", False)
    if isinstance(match_raw, str):
        match = match_raw.strip().lower() in {"true", "yes", "1"}
    else:
        match = bool(match_raw)
    return {
        "match": match,
        "confidence": _clamp(data.get("confidence", 1.0 if match else 0.0)),
        "description": str(data.get("description", ""))[:160],
    }


# ---------------------------------------------------------------------------
# Qwen-VL Transformers client
# ---------------------------------------------------------------------------


class _QwenClient:
    """Thin wrapper over a local Qwen-VL image-text-to-text model."""

    def __init__(self, model_path: str, cfg: SemanticConfig) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        dtype = self._resolve_dtype()
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["attn_implementation"] = cfg.attn_implementation

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, **model_kwargs
            )
        except TypeError:  # older transformers used torch_dtype, no attn kwarg
            model_kwargs.pop("attn_implementation", None)
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, **model_kwargs
            )

        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()
        self.device = self._model_device()

    def _resolve_dtype(self):
        torch = self.torch
        requested = self.cfg.torch_dtype.lower()
        explicit = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if requested in explicit:
            return explicit[requested]
        if not torch.cuda.is_available():
            return torch.float32
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            return torch.float16

    def _model_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.torch.device(
                "cuda:0" if self.torch.cuda.is_available() else "cpu"
            )

    def _build_inputs(self, images: List[Image.Image], prompt: str) -> Dict[str, Any]:
        texts = []
        for image in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        return self.processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        )

    def generate(self, images: List[Image.Image], prompt: str) -> List[str]:
        """Greedy-decode one shared prompt over a batch of images."""
        torch = self.torch
        inputs = self._build_inputs(images, prompt)
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        generated = output_ids[:, prompt_len:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)

    def clear_cache(self) -> None:
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Public filter
# ---------------------------------------------------------------------------


class SemanticActionFilter:
    """Scores/filters frames and video snippets against a free-text action query."""

    def __init__(
        self, model_path: str, config: Optional[SemanticConfig] = None
    ) -> None:
        self.cfg = config or SemanticConfig()
        self._client = _QwenClient(model_path, self.cfg)

    # -- in-memory frames --------------------------------------------------

    def score_images(
        self, images: Sequence[Image.Image], query: str
    ) -> List[FrameScore]:
        """Score each PIL image on whether it depicts ``query``.

        Returns one FrameScore per input image, in order. IDs are the input
        index as a string.
        """
        items = [
            {"id": str(i), "image": img}
            for i, img in enumerate(images)
            if img is not None
        ]
        verdicts = self._score_items(items, query)
        return [verdicts[item["id"]] for item in items if item["id"] in verdicts]

    def filter_images(
        self,
        images: Sequence[Image.Image],
        query: str,
        threshold: float = 0.5,
    ) -> List[FrameScore]:
        """Keep only frames that match ``query`` at/above ``threshold`` confidence."""
        return [
            s
            for s in self.score_images(images, query)
            if s.match and s.confidence >= threshold
        ]

    # -- video snippets ----------------------------------------------------

    def score_video_segments(
        self,
        video_file: str,
        segments: Sequence[Dict],
        query: str,
        fps: float = 24.0,
    ) -> List[FrameScore]:
        """Sample each segment's mid-frame from ``video_file`` and score it.

        Each segment is a dict with ``start``/``end`` seconds and an optional
        ``id`` (defaults to the segment index).
        """
        items = self._sample_segment_frames(video_file, segments, fps)
        verdicts = self._score_items(items, query)
        return [verdicts[item["id"]] for item in items if item["id"] in verdicts]

    def filter_video_segments(
        self,
        video_file: str,
        segments: Sequence[Dict],
        query: str,
        fps: float = 24.0,
        threshold: float = 0.5,
    ) -> List[FrameScore]:
        """Score video segments and return only the matches above ``threshold``."""
        return [
            s
            for s in self.score_video_segments(video_file, segments, query, fps)
            if s.match and s.confidence >= threshold
        ]

    # -- internals ---------------------------------------------------------

    def _sample_segment_frames(
        self, video_file: str, segments: Sequence[Dict], fps: float
    ) -> List[Dict]:
        cap = cv2.VideoCapture(video_file)
        items: List[Dict] = []
        try:
            for index, segment in enumerate(segments):
                seg_id = str(segment.get("id", index))
                cap.set(cv2.CAP_PROP_POS_FRAMES, _segment_mid_frame(fps, segment))
                ok, frame = cap.read()
                image = _frame_to_image(frame if ok else None, self.cfg.frame_max_width)
                if image is not None:
                    items.append({"id": seg_id, "image": image})
        finally:
            cap.release()
        return items

    def _score_items(self, items: List[Dict], query: str) -> Dict[str, FrameScore]:
        """Batch the items through the model with OOM-safe splitting."""
        prompt = build_action_prompt(query)
        verdicts: Dict[str, FrameScore] = {}
        idx = 0
        while idx < len(items):
            batch = items[idx : idx + self.cfg.batch_size]
            verdicts.update(self._score_batch(batch, prompt))
            idx += self.cfg.batch_size
        return verdicts

    def _score_batch(self, batch: List[Dict], prompt: str) -> Dict[str, FrameScore]:
        if not batch:
            return {}
        images = [item["image"] for item in batch]
        try:
            texts = self._client.generate(images, prompt)
        except RuntimeError as exc:
            # Halve the batch and retry on CUDA OOM; re-raise anything else.
            if len(batch) <= 1 or not _is_out_of_memory(exc):
                raise
            self._client.clear_cache()
            split = max(1, len(batch) // 2)
            merged = self._score_batch(batch[:split], prompt)
            merged.update(self._score_batch(batch[split:], prompt))
            return merged

        results: Dict[str, FrameScore] = {}
        for item, text in zip(batch, texts):
            verdict = _normalize_verdict(_parse_json_object(text))
            results[item["id"]] = FrameScore(
                id=item["id"],
                match=verdict["match"],
                confidence=verdict["confidence"],
                description=verdict["description"],
            )
        return results


__all__ = [
    "SemanticActionFilter",
    "SemanticConfig",
    "FrameScore",
    "build_action_prompt",
]
