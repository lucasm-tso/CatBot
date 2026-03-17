"""Pipeline adapter around legacy OCR + Ollama conversion logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig
from .legacy_bridge import load_legacy_module


def run_conversion_pipeline(
    pdf_path: Path,
    config: AppConfig,
) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
    """Convert one PDF by reusing the proven legacy pipeline implementation."""
    legacy = load_legacy_module()
    metrics: dict[str, Any] = {}

    with legacy.fitz.open(pdf_path) as probe_doc:
        if probe_doc.page_count == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

    if config.ocr_engine == "qwen":
        metrics["ocr_engine"] = "qwen"
        pages_data = legacy.extract_text_by_page_with_qwen_ocr(
            pdf_path=pdf_path,
            pages=None,
            dpi=config.dpi,
            ollama_url=config.ollama_url,
            ocr_model=config.resolved_ocr_model(),
            connect_timeout_s=config.connect_timeout,
            read_timeout_s=config.read_timeout,
            retries=config.qwen_retries,
            max_image_side=config.qwen_max_image_side,
            stream=config.stream,
            show_thinking=config.show_thinking,
            guided_zoom=config.guided_zoom,
            zoom_crop_dpi=config.zoom_crop_dpi,
            max_zoom_requests_per_page=config.max_zoom_requests_per_page,
            zoom_min_box_size=config.zoom_min_box_size,
            zoom_max_total_area=config.zoom_max_total_area,
            structured_crop_attempts=config.structured_crop_attempts,
            structured_confidence_threshold=config.structured_confidence_threshold,
            metrics=metrics,
        )
    else:
        metrics["ocr_engine"] = "paddle"
        pages_data = legacy.extract_text_by_page(
            pdf_path=pdf_path,
            pages=None,
            dpi=config.dpi,
            lang=config.lang,
        )
        metrics["pages_processed"] = len(pages_data)

    model_input = legacy.build_model_input(
        prompt=legacy.DEFAULT_PROMPT,
        pages_data=pages_data,
    )
    answer = legacy.call_ollama(
        ollama_url=config.ollama_url,
        model=config.model,
        content=model_input,
        timeout_s=config.final_timeout,
        stream=config.stream,
        show_thinking=config.show_thinking,
        metrics=metrics,
        call_label="final_reasoning",
    )
    return pages_data, answer, metrics
