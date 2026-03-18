"""Application configuration and defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Runtime options for conversion and summarization."""

    ollama_url: str = "http://localhost:7869"
    model: str = "qwen3.5:27b"
    ocr_engine: str = "qwen"
    ocr_model: str | None = None

    stream: bool = True
    show_thinking: bool = True
    guided_zoom: bool = True
    use_page_images_in_review: bool = True

    dpi: int = 600
    lang: str = "en"
    final_timeout: int = 180
    review_batch_size: int = 1
    review_image_max_side: int = 2400
    connect_timeout: int = 15
    read_timeout: int = 900

    qwen_retries: int = 1
    qwen_max_image_side: int = 6000

    zoom_crop_dpi: int = 450
    max_zoom_requests_per_page: int = 3
    zoom_min_box_size: float = 0.03
    zoom_max_total_area: float = 0.45

    structured_crop_attempts: int = 2
    structured_confidence_threshold: float = 0.8

    debug_metrics: bool = True

    summary_model: str | None = None
    summary_timeout: int = 90

    def resolved_ocr_model(self) -> str:
        return self.ocr_model or self.model

    def resolved_summary_model(self) -> str:
        return self.summary_model or self.model
