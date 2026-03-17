#!/usr/bin/env python3
"""Convert a PDF to OCR text with PaddleOCR, send to Ollama, and save response to Markdown."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF
import numpy as np
import requests
from paddleocr import PaddleOCR
from PIL import Image


DEFAULT_PROMPT = (
    "Retranscrit ce document en format mark down. Sois aussi fidèle que possible au contenu et à la structure du document original. "
    "Si tu ne comprends pas une partie du document, fais de ton mieux pour la retranscrire de manière lisible et structurée. "
    "Ne saute aucune partie du document, même si elle est difficile à lire, et utilise l'outil OCR pour les zones de faible confiance."
)


def metrics_inc(metrics: dict[str, Any], key: str, value: int = 1) -> None:
    """Increment an integer metric."""
    metrics[key] = int(metrics.get(key, 0)) + value


def metrics_add_time(metrics: dict[str, Any], key: str, seconds: float) -> None:
    """Accumulate timing metric in seconds."""
    metrics[key] = float(metrics.get(key, 0.0)) + float(seconds)


def round_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Round float metrics for cleaner JSON output."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            out[key] = round(value, 4)
        else:
            out[key] = value
    return out


def print_debug_metrics(metrics: dict[str, Any]) -> None:
    """Print a compact, human-readable metrics block to stderr."""
    print("\n=== METRICS DEBUG ===", file=sys.stderr)
    for key in sorted(metrics.keys()):
        print(f"{key}: {metrics[key]}", file=sys.stderr)


def parse_pages_spec(spec: str, page_count: int) -> list[int]:
    """Parse a page spec like '1,3-5' into zero-based page indexes."""
    pages: set[int] = set()
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start < 1 or end < 1 or start > end:
                raise ValueError(f"Invalid page range: {token}")
            for one_based in range(start, end + 1):
                if one_based > page_count:
                    continue
                pages.add(one_based - 1)
        else:
            one_based = int(token)
            if one_based < 1:
                raise ValueError(f"Invalid page number: {token}")
            if one_based <= page_count:
                pages.add(one_based - 1)

    if not pages:
        raise ValueError("No valid pages selected with --pages.")
    return sorted(pages)


def render_page_to_image(page: fitz.Page, dpi: int = 200) -> Image.Image:
    """Render a PDF page to a PIL image."""
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image


def iter_ocr_lines_from_legacy(ocr_result: list) -> Iterable[str]:
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
    # PaddleOCR 3.x returns OCRResult objects that expose dict-like keys.
    if hasattr(item, "get"):
        rec_texts = item.get("rec_texts")
        if isinstance(rec_texts, list):
            return [str(t).strip() for t in rec_texts if str(t).strip()]

    if isinstance(item, dict):
        rec_texts = item.get("rec_texts")
        if isinstance(rec_texts, list):
            return [str(t).strip() for t in rec_texts if str(t).strip()]

    # PaddleOCR 2.x legacy nested list format.
    if isinstance(item, list):
        return list(iter_ocr_lines_from_legacy(item))

    return []


def extract_text_by_page(
    pdf_path: Path,
    pages: list[int] | None,
    dpi: int,
    lang: str,
) -> list[dict[str, str]]:
    """Run OCR for selected pages and return structured text."""
    doc = fitz.open(pdf_path)
    selected_pages = pages if pages is not None else list(range(doc.page_count))

    # Enable full-page + text-line orientation to better handle rotated scans.
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

        page_text = "\n".join(lines).strip()
        extracted.append(
            {
                "page": str(page_idx + 1),
                "text": page_text,
            }
        )

    doc.close()
    return extracted


def image_to_base64_png(image: Image.Image) -> str:
    """Encode a PIL image as base64 PNG for Ollama vision payload."""
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def resize_for_vision(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale large page images to reduce vision-model latency and memory load."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    ratio = max_side / float(longest)
    new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a string."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    end = -1
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end == -1:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def clamp_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[float, float, float, float]:
    """Clamp bbox coordinates into [0, 1] and ensure x1<x2, y1<y2."""
    ax1 = max(0.0, min(1.0, float(x1)))
    ay1 = max(0.0, min(1.0, float(y1)))
    ax2 = max(0.0, min(1.0, float(x2)))
    ay2 = max(0.0, min(1.0, float(y2)))
    if ax1 > ax2:
        ax1, ax2 = ax2, ax1
    if ay1 > ay2:
        ay1, ay2 = ay2, ay1
    return ax1, ay1, ax2, ay2


def render_pdf_crop_by_normalized_bbox(
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    dpi: int,
    pad: float = 0.01,
) -> Image.Image:
    """Render a high-DPI crop from a PDF page with normalized coordinates."""
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(1.0, x2 + pad)
    y2 = min(1.0, y2 + pad)

    rect = page.rect
    clip = fitz.Rect(
        rect.x0 + x1 * rect.width,
        rect.y0 + y1 * rect.height,
        rect.x0 + x2 * rect.width,
        rect.y0 + y2 * rect.height,
    )
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def propose_zoom_regions(
    ollama_url: str,
    model: str,
    page_image: Image.Image,
    coarse_text: str,
    connect_timeout_s: int,
    read_timeout_s: int,
    max_image_side: int,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Ask model for uncertain regions to inspect with high-res crops."""
    base = ollama_url.rstrip("/")
    endpoint = f"{base}/api/chat"
    prepared_image = resize_for_vision(page_image, max_side=max_image_side)
    image_b64 = image_to_base64_png(prepared_image)

    schema_hint = (
        '{"uncertain_regions": ['
        '{"x1":0.1,"y1":0.1,"x2":0.4,"y2":0.2,"reason":"texte trop petit"}'
        "]}"
    )
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu identifies les zones de faible confiance pour affiner l'OCR. "
                    "Retourne uniquement du JSON, sans markdown ni prose."
                ),
            },
            {
                "role": "user",
                "content": (
                    "A partir de cette page et du texte OCR brouillon, propose jusqu'a 8 regions incertaines "
                    "avec des coordonnees normalisees entre 0 et 1. "
                    "Retourne strictement ce format JSON : "
                    f"{schema_hint}\n\n"
                    "Texte OCR brouillon :\n"
                    f"{coarse_text[:12000]}"
                ),
                "images": [image_b64],
            },
        ],
    }

    start = time.perf_counter()
    response = requests.post(
        endpoint,
        json=payload,
        timeout=(connect_timeout_s, read_timeout_s),
    )
    response.raise_for_status()
    metrics_inc(metrics, "zoom_proposal_calls")
    metrics_add_time(metrics, "zoom_proposal_seconds", time.perf_counter() - start)
    body = response.json()
    raw = str(body.get("message", {}).get("content", "")).strip()
    parsed = extract_first_json_object(raw)
    if not parsed or not isinstance(parsed, dict):
        return []

    regions = parsed.get("uncertain_regions")
    if not isinstance(regions, list):
        return []
    valid: list[dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        if not all(k in region for k in ("x1", "y1", "x2", "y2")):
            continue
        valid.append(region)
    metrics_inc(metrics, "zoom_regions_proposed", len(valid))
    return valid


def synthesize_page_text_with_refinements(
    ollama_url: str,
    model: str,
    coarse_text: str,
    refinements: list[dict[str, Any]],
    timeout_s: int,
    stream: bool,
    show_thinking: bool,
    metrics: dict[str, Any],
) -> str:
    """Merge coarse OCR text and region refinements into a final page transcription."""
    refinement_lines = []
    for idx, item in enumerate(refinements, start=1):
        bbox = item["bbox"]
        reason = str(item.get("reason", ""))
        text = str(item.get("text", "")).strip()
        refinement_lines.append(
            f"Region {idx} bbox={bbox} raison={reason}\n{text}"
        )

    merge_prompt = (
        "Produis une transcription OCR corrigee unique pour cette page. "
        "Utilise l'OCR brouillon comme base et corrige les erreurs avec les raffinements regionaux. "
        "Retourne uniquement le texte final corrige, sans explication.\n\n"
        "OCR BROUILLON :\n"
        f"{coarse_text}\n\n"
        "RAFFINEMENTS OCR PAR REGION :\n"
        f"{'\n\n'.join(refinement_lines)}"
    )
    return call_ollama(
        ollama_url=ollama_url,
        model=model,
        content=merge_prompt,
        timeout_s=timeout_s,
        stream=stream,
        show_thinking=show_thinking,
        metrics=metrics,
        call_label="zoom_synthesis",
    )


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
    """Use a vision-capable Ollama model to OCR one image page."""
    base = ollama_url.rstrip("/")
    endpoint = f"{base}/api/chat"
    prepared_image = resize_for_vision(image, max_side=max_image_side)
    image_b64 = image_to_base64_png(prepared_image)

    payload = {
        "model": model,
        "stream": stream,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un moteur OCR. Extrais tout le texte visible de l'image. "
                    "Respecte autant que possible l'ordre de lecture d'origine. "
                    "Retourne uniquement le texte extrait brut, sans commentaire."
                ),
            },
            {
                "role": "user",
                "content": "Extrais tout le texte de cette page.",
                "images": [image_b64],
            },
        ],
    }
    for attempt in range(1, retries + 2):
        try:
            start = time.perf_counter()
            response = requests.post(
                endpoint,
                json=payload,
                timeout=(connect_timeout_s, read_timeout_s),
                stream=stream,
            )
            response.raise_for_status()
            elapsed = time.perf_counter() - start
            metrics_inc(metrics, "qwen_ocr_calls")
            metrics_add_time(metrics, "qwen_ocr_seconds", elapsed)
            if attempt > 1:
                metrics_inc(metrics, "qwen_ocr_retry_successes")
            if not stream:
                body = response.json()
                message = body.get("message", {})
                return str(message.get("content", "")).strip()

            content_chunks: list[str] = []
            thinking_chunks: list[str] = []
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                event = json.loads(raw_line)
                message = event.get("message", {})
                chunk = str(message.get("content", ""))
                if chunk:
                    content_chunks.append(chunk)
                    metrics_inc(metrics, "qwen_ocr_stream_chunks")
                    print(chunk, end="", flush=True, file=sys.stderr)

                # Some models expose reasoning in a dedicated field.
                think = event.get("thinking") or message.get("thinking") or message.get("reasoning")
                if think:
                    think_text = str(think)
                    thinking_chunks.append(think_text)
                    metrics_inc(metrics, "qwen_ocr_thinking_chunks")
                    if show_thinking:
                        print(think_text, end="", flush=True, file=sys.stderr)

            if content_chunks or thinking_chunks:
                print("", file=sys.stderr)

            return "".join(content_chunks).strip()
        except requests.exceptions.ReadTimeout as exc:
            metrics_inc(metrics, "qwen_ocr_timeouts")
            if attempt > retries:
                raise RuntimeError(
                    "Le OCR Ollama a expire en attendant la sortie du modele. "
                    f"Model='{model}', read_timeout={read_timeout_s}s. "
                    "Essaie un --read-timeout plus grand, diminue --qwen-max-image-side, "
                    "ou utilise un modele vision plus leger pour --ocr-model."
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
    metrics: dict[str, Any],
) -> list[dict[str, str]]:
    """Run OCR for selected pages using a vision-capable Ollama model."""
    doc = fitz.open(pdf_path)
    selected_pages = pages if pages is not None else list(range(doc.page_count))
    extracted: list[dict[str, str]] = []

    for page_idx in selected_pages:
        page_start = time.perf_counter()
        metrics_inc(metrics, "pages_processed")
        page = doc.load_page(page_idx)
        image = render_page_to_image(page=page, dpi=dpi)
        coarse_start = time.perf_counter()
        coarse_text = call_ollama_ocr_on_image(
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
        metrics_add_time(metrics, "qwen_coarse_page_seconds", time.perf_counter() - coarse_start)

        page_text = coarse_text
        if guided_zoom:
            metrics_inc(metrics, "guided_zoom_pages")
            regions = propose_zoom_regions(
                ollama_url=ollama_url,
                model=ocr_model,
                page_image=image,
                coarse_text=coarse_text,
                connect_timeout_s=connect_timeout_s,
                read_timeout_s=read_timeout_s,
                max_image_side=max_image_side,
                metrics=metrics,
            )
            refinements: list[dict[str, Any]] = []
            total_area = 0.0
            for region in regions:
                if len(refinements) >= max_zoom_requests_per_page:
                    metrics_inc(metrics, "zoom_regions_rejected_limit")
                    break
                x1, y1, x2, y2 = clamp_bbox(
                    region["x1"], region["y1"], region["x2"], region["y2"]
                )
                w = x2 - x1
                h = y2 - y1
                area = w * h
                if w < zoom_min_box_size or h < zoom_min_box_size:
                    metrics_inc(metrics, "zoom_regions_rejected_small")
                    continue
                if total_area + area > zoom_max_total_area:
                    metrics_inc(metrics, "zoom_regions_rejected_area")
                    continue

                crop_start = time.perf_counter()
                crop = render_pdf_crop_by_normalized_bbox(
                    page=page,
                    bbox=(x1, y1, x2, y2),
                    dpi=zoom_crop_dpi,
                )
                crop_text = call_ollama_ocr_on_image(
                    ollama_url=ollama_url,
                    model=ocr_model,
                    image=crop,
                    connect_timeout_s=connect_timeout_s,
                    read_timeout_s=read_timeout_s,
                    retries=retries,
                    max_image_side=max_image_side,
                    stream=stream,
                    show_thinking=show_thinking,
                    metrics=metrics,
                )
                metrics_inc(metrics, "zoom_crop_ocr_calls")
                metrics_add_time(metrics, "zoom_crop_ocr_seconds", time.perf_counter() - crop_start)
                refinements.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "reason": str(region.get("reason", "")),
                        "text": crop_text,
                    }
                )
                total_area += area
                metrics_add_time(metrics, "zoom_total_area_used", area)

            if refinements:
                synth_start = time.perf_counter()
                page_text = synthesize_page_text_with_refinements(
                    ollama_url=ollama_url,
                    model=ocr_model,
                    coarse_text=coarse_text,
                    refinements=refinements,
                    timeout_s=read_timeout_s,
                    stream=stream,
                    show_thinking=show_thinking,
                    metrics=metrics,
                )
                metrics_add_time(metrics, "zoom_synthesis_seconds", time.perf_counter() - synth_start)
                metrics_inc(metrics, "zoom_synthesis_calls")
                metrics_inc(metrics, "zoom_regions_used", len(refinements))

        metrics_add_time(metrics, "page_total_seconds", time.perf_counter() - page_start)

        extracted.append(
            {
                "page": str(page_idx + 1),
                "text": page_text,
            }
        )

    doc.close()
    return extracted


def build_model_input(prompt: str, pages_data: list[dict[str, str]]) -> str:
    """Build a single prompt containing OCR text by page."""
    parts = [prompt.strip(), "", "Pages source OCR :"]
    for entry in pages_data:
        page_no = entry["page"]
        text = entry["text"].strip() or "[No text detected on this page]"
        parts.append(f"\n--- PAGE {page_no} ---\n{text}")
    return "\n".join(parts).strip()


def call_ollama(
    ollama_url: str,
    model: str,
    content: str,
    timeout_s: int,
    stream: bool,
    show_thinking: bool,
    metrics: dict[str, Any],
    call_label: str,
) -> str:
    """Call Ollama /api/chat and return assistant content."""
    base = ollama_url.rstrip("/")
    endpoint = f"{base}/api/chat"
    payload = {
        "model": model,
        "stream": stream,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant qui aide à analyser le contenu de documents PDF. "
                    "Sois précis et structure la réponse en Markdown."
                ),
            },
            {"role": "user", "content": content},
        ],
    }
    start = time.perf_counter()
    response = requests.post(endpoint, json=payload, timeout=timeout_s, stream=stream)
    response.raise_for_status()
    elapsed = time.perf_counter() - start
    metrics_inc(metrics, f"{call_label}_calls")
    metrics_add_time(metrics, f"{call_label}_seconds", elapsed)
    if not stream:
        body = response.json()
        message = body.get("message", {})
        answer = message.get("content", "")
        return str(answer).strip()

    content_chunks: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        event = json.loads(raw_line)
        message = event.get("message", {})
        chunk = str(message.get("content", ""))
        if chunk:
            content_chunks.append(chunk)
            metrics_inc(metrics, f"{call_label}_stream_chunks")
            print(chunk, end="", flush=True, file=sys.stderr)

        think = event.get("thinking") or message.get("thinking") or message.get("reasoning")
        if think and show_thinking:
            metrics_inc(metrics, f"{call_label}_thinking_chunks")
            print(str(think), end="", flush=True, file=sys.stderr)

    if content_chunks:
        print("", file=sys.stderr)
    return "".join(content_chunks).strip()


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    return cleaned or "output"


def write_markdown(
    output_path: Path,
    pdf_path: Path,
    model: str,
    prompt: str,
    pages_data: list[dict[str, str]],
    answer: str,
) -> None:
    generated_at = dt.datetime.now().isoformat(timespec="seconds")
    total_pages = len(pages_data)

    md_lines = [
        f"# Ollama Result - {pdf_path.name}",
        "",
        "## Metadata",
        f"- Generated at: {generated_at}",
        f"- PDF: `{pdf_path}`",
        f"- Model: `{model}`",
        f"- Pages processed: {total_pages}",
        "",
        "## Prompt",
        "```text",
        prompt.strip(),
        "```",
        "",
        "## Model Response",
        answer.strip() if answer.strip() else "[Empty response]",
        "",
        "## OCR Extract (by page)",
    ]

    for entry in pages_data:
        md_lines.extend(
            [
                "",
                f"### Page {entry['page']}",
                "```text",
                entry["text"].strip() if entry["text"].strip() else "[No text detected]",
                "```",
            ]
        )

    output_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")


def load_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt_file is not None:
        return prompt_file.read_text(encoding="utf-8").strip()
    if prompt:
        return prompt.strip()
    return DEFAULT_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract PDF text with PaddleOCR and ask Ollama model, saving output as Markdown."
    )
    parser.add_argument("pdf", type=Path, help="Path to input PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .md path (default: ./<pdf_name>_ollama.md)",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:35b",
        help="Ollama model name (default: qwen3.5:35b)",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["paddle", "qwen"],
        default="paddle",
        help="OCR engine: paddle or qwen (default: paddle)",
    )
    parser.add_argument(
        "--ocr-model",
        default=None,
        help=(
            "Ollama model used only for OCR when --ocr-engine qwen. "
            "Defaults to --model."
        ),
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt to send with OCR text.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Path to a text file containing the prompt.",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Pages to process, 1-based (example: 1,3-5). Default: all pages.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rendering DPI for PDF pages (default: 200)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="PaddleOCR language (default: en)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds for Ollama call (default: 180)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=15,
        help="HTTP connect timeout in seconds for Ollama requests (default: 15)",
    )
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=900,
        help="HTTP read timeout in seconds for qwen OCR page calls (default: 900)",
    )
    parser.add_argument(
        "--qwen-retries",
        type=int,
        default=1,
        help="Retry count per page for qwen OCR on read timeout (default: 1)",
    )
    parser.add_argument(
        "--qwen-max-image-side",
        type=int,
        default=1400,
        help="Max page image side (pixels) sent to qwen OCR (default: 1400)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream model output live to stderr while generating.",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="When available, print model thinking/reasoning stream to stderr.",
    )
    parser.add_argument(
        "--debug-metrics",
        action="store_true",
        help="Print detailed timing and tool-usage metrics to stderr.",
    )
    parser.add_argument(
        "--guided-zoom",
        action="store_true",
        help="Enable model-guided high-res crop OCR refinement with bounding boxes.",
    )
    parser.add_argument(
        "--zoom-crop-dpi",
        type=int,
        default=450,
        help="DPI used to render zoom crops from PDF (default: 450)",
    )
    parser.add_argument(
        "--max-zoom-requests-per-page",
        type=int,
        default=3,
        help="Maximum model-requested zoom crops per page (default: 3)",
    )
    parser.add_argument(
        "--zoom-min-box-size",
        type=float,
        default=0.03,
        help="Minimum normalized bbox width/height accepted for zoom (default: 0.03)",
    )
    parser.add_argument(
        "--zoom-max-total-area",
        type=float,
        default=0.45,
        help="Maximum cumulative normalized zoom area per page (default: 0.45)",
    )

    args = parser.parse_args()
    metrics: dict[str, Any] = {}
    t0_total = time.perf_counter()

    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Input must be an existing PDF file: {pdf_path}")

    if args.prompt and args.prompt_file:
        raise SystemExit("Use either --prompt or --prompt-file, not both.")

    output_path = args.output
    if output_path is None:
        name = sanitize_filename(pdf_path.stem)
        output_path = Path.cwd() / f"{name}_ollama.md"
    output_path = output_path.expanduser().resolve()

    prompt_text = load_prompt(args.prompt, args.prompt_file)

    with fitz.open(pdf_path) as probe_doc:
        if probe_doc.page_count == 0:
            raise SystemExit("The PDF has no pages.")
        selected_pages = (
            parse_pages_spec(args.pages, probe_doc.page_count) if args.pages else None
        )

    if args.ocr_engine == "qwen":
        ocr_model = args.ocr_model or args.model
        metrics["ocr_engine"] = "qwen"
        pages_data = extract_text_by_page_with_qwen_ocr(
            pdf_path=pdf_path,
            pages=selected_pages,
            dpi=args.dpi,
            ollama_url=args.ollama_url,
            ocr_model=ocr_model,
            connect_timeout_s=args.connect_timeout,
            read_timeout_s=args.read_timeout,
            retries=args.qwen_retries,
            max_image_side=args.qwen_max_image_side,
            stream=args.stream,
            show_thinking=args.show_thinking,
            guided_zoom=args.guided_zoom,
            zoom_crop_dpi=args.zoom_crop_dpi,
            max_zoom_requests_per_page=args.max_zoom_requests_per_page,
            zoom_min_box_size=args.zoom_min_box_size,
            zoom_max_total_area=args.zoom_max_total_area,
            metrics=metrics,
        )
    else:
        metrics["ocr_engine"] = "paddle"
        paddle_start = time.perf_counter()
        pages_data = extract_text_by_page(
            pdf_path=pdf_path,
            pages=selected_pages,
            dpi=args.dpi,
            lang=args.lang,
        )
        metrics_inc(metrics, "pages_processed", len(pages_data))
        metrics_add_time(metrics, "paddle_ocr_seconds", time.perf_counter() - paddle_start)

    model_input = build_model_input(prompt=prompt_text, pages_data=pages_data)
    answer = call_ollama(
        ollama_url=args.ollama_url,
        model=args.model,
        content=model_input,
        timeout_s=args.timeout,
        stream=args.stream,
        show_thinking=args.show_thinking,
        metrics=metrics,
        call_label="final_reasoning",
    )

    write_markdown(
        output_path=output_path,
        pdf_path=pdf_path,
        model=args.model,
        prompt=prompt_text,
        pages_data=pages_data,
        answer=answer,
    )

    summary = {
        "output": str(output_path),
        "pages_processed": len(pages_data),
        "model": args.model,
        "metrics": round_metrics({
            **metrics,
            "total_runtime_seconds": time.perf_counter() - t0_total,
        }),
    }
    if args.debug_metrics:
        print_debug_metrics(summary["metrics"])
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
