"""Pipeline adapter around internal OCR + Ollama conversion logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig
from .markdown_service import build_model_input
from .ocr_service import (
    extract_text_by_page,
    extract_text_by_page_with_qwen_ocr,
    image_to_base64_png,
    render_page_to_image,
    resize_for_vision,
)
from .ollama_client import call_ollama_chat
import fitz


APP_DEFAULT_PROMPT = (
    "Retranscris ce document en Markdown en restant tres fidele au contenu et a la structure d'origine. "
    "N'omets aucune section, meme si certaines zones sont difficiles a lire. "
    "Si le document contient des dessins techniques (schemas, plans, coupes, vues annotees, tableaux de cotes, legendes), "
    "decris-les le plus fidelement possible: elements visibles, etiquettes, dimensions/valeurs, unites, reperes et relations spatiales, "
    "sans inventer d'information absente."
)

def _safe_page_no(entry: dict[str, str], fallback: int) -> str:
    page_no = str(entry.get("page", "")).strip()
    return page_no if page_no else str(fallback)


def _summarize_chunk(
    page_answers: list[dict[str, str]],
    config: AppConfig,
    metrics: dict[str, Any],
    chunk_index: int,
) -> str:
    parts = [
        "Tu recois des transcriptions OCR page par page.",
        "Produis une version nettoyee en Markdown de ce lot de pages.",
        "Reste tres fidele au contenu et a la structure.",
        "N'omets aucune section visible et n'invente rien.",
        "",
        f"Lot de pages {chunk_index} :",
    ]

    for entry in page_answers:
        page_no = _safe_page_no(entry, fallback=0)
        answer = entry.get("answer", "").strip() or "[Empty response]"
        parts.append(f"\n--- PAGE {page_no} ---\n{answer}")

    return call_ollama_chat(
        ollama_url=config.ollama_url,
        model=config.model,
        content="\n".join(parts).strip(),
        timeout_s=config.final_timeout,
        stream=config.stream,
        show_thinking=config.show_thinking,
        metrics=metrics,
        call_label="summary_chunk",
    )


def _build_document_summary(
    page_answers: list[dict[str, str]],
    config: AppConfig,
    metrics: dict[str, Any],
) -> str:
    if not page_answers:
        return "[Empty summary]"

    chunk_size = max(1, int(config.review_batch_size))

    if len(page_answers) <= chunk_size:
        return _summarize_chunk(
            page_answers=page_answers,
            config=config,
            metrics=metrics,
            chunk_index=1,
        ).strip() or "[Empty chunk summary 1]"

    chunk_summaries: list[str] = []
    total_pages = len(page_answers)
    total_chunks = (total_pages + chunk_size - 1) // chunk_size

    for idx in range(0, total_pages, chunk_size):
        chunk = page_answers[idx : idx + chunk_size]
        chunk_number = (idx // chunk_size) + 1
        first_page = _safe_page_no(chunk[0], fallback=idx + 1)
        last_page = _safe_page_no(chunk[-1], fallback=min(total_pages, idx + chunk_size))
        chunk_summary = _summarize_chunk(
            page_answers=chunk,
            config=config,
            metrics=metrics,
            chunk_index=chunk_number,
        )
        cleaned = chunk_summary.strip() or f"[Empty chunk summary {chunk_number}]"
        chunk_summaries.append(
            f"--- PAGE BATCH {chunk_number}/{total_chunks} (pages {first_page}-{last_page}) ---\n\n{cleaned}"
        )

    return "\n\n".join(chunk_summaries).strip()


def _build_review_image_payload(
    review_doc: fitz.Document,
    page_no: str,
    config: AppConfig,
    metrics: dict[str, Any],
) -> list[str] | None:
    """Build base64 image payload for page-level clean review."""
    try:
        page_idx = max(0, int(page_no) - 1)
        page = review_doc.load_page(page_idx)
        page_image = render_page_to_image(page=page, dpi=config.dpi)
        prepared = resize_for_vision(page_image, max_side=config.review_image_max_side)
        metrics["review_images_sent"] = int(metrics.get("review_images_sent", 0)) + 1
        return [image_to_base64_png(prepared)]
    except Exception:  # noqa: BLE001
        metrics["review_image_build_failures"] = int(metrics.get("review_image_build_failures", 0)) + 1
        return None


def run_conversion_pipeline(
    pdf_path: Path,
    config: AppConfig,
) -> tuple[list[dict[str, str]], str, list[dict[str, str]], dict[str, Any]]:
    """Convert one PDF using internal OCR and Ollama services."""
    metrics: dict[str, Any] = {}

    with fitz.open(pdf_path) as probe_doc:
        if probe_doc.page_count == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

    if config.ocr_engine == "qwen":
        metrics["ocr_engine"] = "qwen"
        pages_data = extract_text_by_page_with_qwen_ocr(
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
        pages_data = extract_text_by_page(
            pdf_path=pdf_path,
            pages=None,
            dpi=config.dpi,
            lang=config.lang,
        )
        metrics["pages_processed"] = len(pages_data)

    page_answers: list[dict[str, str]] = []
    review_doc: fitz.Document | None = None
    if config.use_page_images_in_review:
        review_doc = fitz.open(pdf_path)

    try:
        for idx, page_entry in enumerate(pages_data, start=1):
            page_no = _safe_page_no(page_entry, fallback=idx)
            page_input = build_model_input(prompt=APP_DEFAULT_PROMPT, pages_data=[page_entry])
            review_images = None
            if review_doc is not None:
                review_images = _build_review_image_payload(
                    review_doc=review_doc,
                    page_no=page_no,
                    config=config,
                    metrics=metrics,
                )

            page_answer = call_ollama_chat(
                ollama_url=config.ollama_url,
                model=config.model,
                content=page_input,
                timeout_s=config.final_timeout,
                stream=config.stream,
                show_thinking=config.show_thinking,
                metrics=metrics,
                call_label="page_reasoning",
                images=review_images,
            )
            page_answers.append({"page": page_no, "answer": page_answer})
    finally:
        if review_doc is not None:
            review_doc.close()

    document_summary = _build_document_summary(
        page_answers=page_answers,
        config=config,
        metrics=metrics,
    )
    return pages_data, document_summary, page_answers, metrics
