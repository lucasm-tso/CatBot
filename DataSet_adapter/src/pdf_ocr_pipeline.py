"""Pipeline adapter around internal OCR + Ollama conversion logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig
from .markdown_service import build_model_input
from .ocr_service import extract_text_by_page, extract_text_by_page_with_qwen_ocr
from .ollama_client import call_ollama_chat
import fitz


APP_DEFAULT_PROMPT = (
    "Retranscris ce document en Markdown en restant tres fidele au contenu et a la structure d'origine. "
    "N'omets aucune section, meme si certaines zones sont difficiles a lire. "
    "Si le document contient des dessins techniques (schemas, plans, coupes, vues annotees, tableaux de cotes, legendes), "
    "decris-les le plus fidelement possible: elements visibles, etiquettes, dimensions/valeurs, unites, reperes et relations spatiales, "
    "sans inventer d'information absente."
)

SUMMARY_CHUNK_SIZE = 8


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
        "Tu recois des transcriptions markdown page par page.",
        "Produis un resume fidele de ce lot de pages.",
        "Conserve les points techniques, chiffres, references de plans et sections importantes.",
        "Reste concis (6-14 puces max) et ecris en Markdown.",
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

    chunk_summaries: list[str] = []
    for idx in range(0, len(page_answers), SUMMARY_CHUNK_SIZE):
        chunk = page_answers[idx : idx + SUMMARY_CHUNK_SIZE]
        chunk_number = (idx // SUMMARY_CHUNK_SIZE) + 1
        chunk_summary = _summarize_chunk(
            page_answers=chunk,
            config=config,
            metrics=metrics,
            chunk_index=chunk_number,
        )
        chunk_summaries.append(chunk_summary.strip() or f"[Empty chunk summary {chunk_number}]")

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    final_parts = [
        "Tu recois des resumes partiels d'un document PDF.",
        "Produis un resume global propre et coherent en Markdown.",
        "Inclure : 1) structure du document, 2) points techniques majeurs, 3) references importantes, 4) risques/ambiguities.",
        "N'invente aucune information.",
        "",
        "Resumes partiels :",
    ]
    for idx, chunk_summary in enumerate(chunk_summaries, start=1):
        final_parts.append(f"\n--- CHUNK {idx} ---\n{chunk_summary}")

    return call_ollama_chat(
        ollama_url=config.ollama_url,
        model=config.model,
        content="\n".join(final_parts).strip(),
        timeout_s=config.final_timeout,
        stream=config.stream,
        show_thinking=config.show_thinking,
        metrics=metrics,
        call_label="final_reasoning",
    )


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
    for idx, page_entry in enumerate(pages_data, start=1):
        page_no = _safe_page_no(page_entry, fallback=idx)
        page_input = build_model_input(prompt=APP_DEFAULT_PROMPT, pages_data=[page_entry])
        page_answer = call_ollama_chat(
            ollama_url=config.ollama_url,
            model=config.model,
            content=page_input,
            timeout_s=config.final_timeout,
            stream=config.stream,
            show_thinking=config.show_thinking,
            metrics=metrics,
            call_label="page_reasoning",
        )
        page_answers.append({"page": page_no, "answer": page_answer})

    document_summary = _build_document_summary(
        page_answers=page_answers,
        config=config,
        metrics=metrics,
    )
    return pages_data, document_summary, page_answers, metrics
