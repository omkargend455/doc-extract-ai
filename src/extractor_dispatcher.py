"""Unified extractor dispatcher supporting Auto provider routing, NVIDIA NIM, and deterministic offline baseline."""

from __future__ import annotations

from typing import Any, Optional

from src.baseline_regex import RegexBaselineExtractor
from src.nvidia_extractor import NvidiaExtractor
from src.schemas import ExtractionResult


def extract_with_provider(
    text: str,
    doc_type: str = "receipt",
    doc_id: str = "doc_unknown",
    mode: str = "auto",  # "auto", "nvidia", "baseline"
    nvidia_api_key: Optional[str] = None,
    nvidia_model: Optional[str] = None,
    nvidia_base_url: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
) -> ExtractionResult:
    """Execute document extraction according to selected provider mode.

    Modes:
      - 'auto': Try NVIDIA NIM when configured and valid; otherwise fallback to offline regex baseline.
      - 'nvidia': Explicitly invoke NVIDIA NIM hosted LLM (e.g. nvidia/nemotron-3.5-lightning-30b-a3b).
                  Falls back to regex with explicit diagnostic on failure.
      - 'baseline': Explicitly run deterministic regex baseline with zero network calls.
      - Unknown / obsolete modes (e.g. 'groq', 'gemini'): route directly to baseline.
    """
    normalized_mode = mode.lower().strip()

    baseline_ext = RegexBaselineExtractor()

    # 1. Explicit Offline Baseline Mode
    if normalized_mode in {"baseline", "offline", "regex"}:
        fields = baseline_ext.extract(text, doc_type=doc_type)
        return ExtractionResult(
            doc_id=doc_id,
            doc_type=doc_type,
            source="baseline",
            provider="regex",
            model_used=None,
            fields=fields,
            raw_text=text,
            error_message="Offline mode forced: Used deterministic regex baseline.",
            metadata={"input_mode": "offline_baseline"},
        )

    # 2. Explicit NVIDIA NIM Mode
    if normalized_mode == "nvidia":
        nvidia_ext = NvidiaExtractor(api_key=nvidia_api_key, model=nvidia_model, base_url=nvidia_base_url)
        return nvidia_ext.extract(text=text, doc_type=doc_type, doc_id=doc_id, image_bytes=image_bytes)

    # 3. Auto Mode (Priority: NVIDIA NIM -> Offline Baseline)
    if normalized_mode == "auto":
        nvidia_ext = NvidiaExtractor(api_key=nvidia_api_key, model=nvidia_model, base_url=nvidia_base_url)
        if nvidia_ext.is_available():
            res_nvidia = nvidia_ext.extract(text=text, doc_type=doc_type, doc_id=doc_id, image_bytes=image_bytes)
            if res_nvidia.source == "nvidia":
                return res_nvidia
            # NVIDIA was configured but failed; report error and fallback
            fields = baseline_ext.extract(text, doc_type=doc_type)
            return ExtractionResult(
                doc_id=doc_id,
                doc_type=doc_type,
                source="baseline_fallback",
                provider="regex",
                model_used=None,
                fields=fields,
                raw_text=text,
                error_message=f"Auto mode: NVIDIA NIM failed ({res_nvidia.error_message}). Switched to offline baseline.",
                metadata={"input_mode": "offline_fallback"},
            )

        # Default fallback: No NVIDIA API key configured; clean offline baseline
        fields = baseline_ext.extract(text, doc_type=doc_type)
        return ExtractionResult(
            doc_id=doc_id,
            doc_type=doc_type,
            source="baseline",
            provider="regex",
            model_used=None,
            fields=fields,
            raw_text=text,
            error_message="Auto mode: NVIDIA API key not configured. Used deterministic offline baseline.",
            metadata={"input_mode": "offline_baseline"},
        )

    # 4. Unknown or obsolete provider modes (e.g. 'groq', 'gemini') -> route directly to baseline
    fields = baseline_ext.extract(text, doc_type=doc_type)
    return ExtractionResult(
        doc_id=doc_id,
        doc_type=doc_type,
        source="baseline",
        provider="regex",
        model_used=None,
        fields=fields,
        raw_text=text,
        error_message=f"Unknown or obsolete provider mode '{mode}'. Routed directly to deterministic baseline.",
        metadata={"input_mode": "offline_baseline"},
    )
