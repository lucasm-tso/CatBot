"""Internal OCR services used by the adapter app."""

from __future__ import annotations

import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import fitz
import numpy as np
import requests
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw

from .metrics import metrics_add_time, metrics_inc
from .ollama_client import call_ollama_chat


def render_page_to_image(page: fitz.Page, dpi: int = 200) -> Image.Image:
    """Render a PDF page to a PIL image."""
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def iter_ocr_lines_from_legacy(ocr_result: list[Any]) -> Iterable[str]:
    """Yield recognized lines from PaddleOCR 2.x list output format."""
    if not ocr_result:
        return
    for line in ocr_result:
        if not line or len(line) < 2:
            continue
        text_part = line[1]
        if isinstance(text_part, (list, tuple)) and text_part:
            text = str(text_part[0]).strip()
            if text:
                yield text


def extract_lines_from_result_item(item: Any) -> list[str]:
    """Extract text lines from a PaddleOCR result item across API versions."""
    if hasattr(item, "get"):
        rec_texts = item.get("rec_texts")
        if isinstance(rec_texts, list):
            return [str(t).strip() for t in rec_texts if str(t).strip()]

    if isinstance(item, dict):
        rec_texts = item.get("rec_texts")
        if isinstance(rec_texts, list):
            return [str(t).strip() for t in rec_texts if str(t).strip()]

    if isinstance(item, list):
        return list(iter_ocr_lines_from_legacy(item))

    return []


def extract_text_by_page(
    pdf_path: Path,
    pages: list[int] | None,
    dpi: int,
    lang: str,
) -> list[dict[str, str]]:
    """Run PaddleOCR for selected pages and return text by page."""
    doc = fitz.open(pdf_path)
    selected_pages = pages if pages is not None else list(range(doc.page_count))

    ocr_engine = PaddleOCR(
        use_doc_orientation_classify=True,
        use_textline_orientation=True,
        lang=lang,
    )
    extracted: list[dict[str, str]] = []

    for page_idx in selected_pages:
        page = doc.load_page(page_idx)
        image = render_page_to_image(page=page, dpi=dpi)
        arr = np.array(image)
        ocr_result = ocr_engine.predict(arr)

        lines: list[str] = []
        if isinstance(ocr_result, list) and ocr_result:
            lines = extract_lines_from_result_item(ocr_result[0])

        extracted.append({"page": str(page_idx + 1), "text": "\n".join(lines).strip()})

    doc.close()
    return extracted


def image_to_base64_png(image: Image.Image) -> str:
    """Encode a PIL image as base64 PNG for Ollama vision payload."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def resize_for_vision(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale large images to reduce vision model latency and memory."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    ratio = max_side / float(longest)
    new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def call_ollama_ocr_on_image(
    ollama_url: str,
    model: str,
    image: Image.Image,
    connect_timeout_s: int,
    read_timeout_s: int,
    retries: int,
    max_image_side: int,
    stream: bool,
    show_thinking: bool,
    metrics: dict[str, Any],
) -> str:
    """Use a vision-capable Ollama model to OCR one page image."""
    prepared_image = resize_for_vision(image, max_side=max_image_side)
    image_b64 = image_to_base64_png(prepared_image)
    content = "Extrais tout le texte de cette page. Retourne uniquement le texte extrait."

    for attempt in range(1, retries + 2):
        try:
            start = time.perf_counter()
            text = call_ollama_chat(
                ollama_url=ollama_url,
                model=model,
                content=content,
                timeout_s=(connect_timeout_s, read_timeout_s),
                stream=stream,
                show_thinking=show_thinking,
                metrics=metrics,
                call_label="qwen_ocr",
                images=[image_b64],
            )
            metrics_add_time(metrics, "qwen_ocr_seconds", time.perf_counter() - start)
            if attempt > 1:
                metrics_inc(metrics, "qwen_ocr_retry_successes")
            return text.strip()
        except requests.exceptions.ReadTimeout as exc:
            metrics_inc(metrics, "qwen_ocr_timeouts")
            if attempt > retries:
                raise RuntimeError(
                    "Le OCR Ollama a expire en attendant la sortie du modele. "
                    f"Model='{model}', read_timeout={read_timeout_s}s."
                ) from exc
            time.sleep(min(2 * attempt, 8))


def extract_text_by_page_with_qwen_ocr(
    pdf_path: Path,
    pages: list[int] | None,
    dpi: int,
    ollama_url: str,
    ocr_model: str,
    connect_timeout_s: int,
    read_timeout_s: int,
    retries: int,
    max_image_side: int,
    stream: bool,
    show_thinking: bool,
    guided_zoom: bool,
    zoom_crop_dpi: int,
    max_zoom_requests_per_page: int,
    zoom_min_box_size: float,
    zoom_max_total_area: float,
    structured_crop_attempts: int,
    structured_confidence_threshold: float,
    metrics: dict[str, Any],
) -> list[dict[str, str]]:
    """Run Qwen OCR by page.

    This internal implementation keeps the same signature as the previous bridge
    to avoid touching calling code; advanced zoom params are currently retained
    for API compatibility.
    """
    _ = (
        guided_zoom,
        zoom_crop_dpi,
        max_zoom_requests_per_page,
        zoom_min_box_size,
        zoom_max_total_area,
        structured_crop_attempts,
        structured_confidence_threshold,
    )

    doc = fitz.open(pdf_path)
    selected_pages = pages if pages is not None else list(range(doc.page_count))
    extracted: list[dict[str, str]] = []

    for page_idx in selected_pages:
        page_start = time.perf_counter()
        metrics_inc(metrics, "pages_processed")
        page = doc.load_page(page_idx)
        image = render_page_to_image(page=page, dpi=dpi)

        text = call_ollama_ocr_on_image(
            ollama_url=ollama_url,
            model=ocr_model,
            image=image,
            connect_timeout_s=connect_timeout_s,
            read_timeout_s=read_timeout_s,
            retries=retries,
            max_image_side=max_image_side,
            stream=stream,
            show_thinking=show_thinking,
            metrics=metrics,
        )
        metrics_add_time(metrics, "page_total_seconds", time.perf_counter() - page_start)
        extracted.append({"page": str(page_idx + 1), "text": text})

    doc.close()
    return extracted
