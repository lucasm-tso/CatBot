"""Markdown formatting utilities for conversion outputs."""

from __future__ import annotations

import datetime as dt
from pathlib import Path


def build_model_input(prompt: str, pages_data: list[dict[str, str]]) -> str:
    """Build a single prompt containing OCR text by page."""
    parts = [prompt.strip(), "", "Pages source OCR :"]
    for entry in pages_data:
        page_no = entry["page"]
        text = entry["text"].strip() or "[No text detected on this page]"
        parts.append(f"\n--- PAGE {page_no} ---\n{text}")
    return "\n".join(parts).strip()


def write_markdown(
    output_path: Path,
    pdf_path: Path,
    model: str,
    prompt: str,
    pages_data: list[dict[str, str]],
    summary: str,
    page_answers: list[dict[str, str]],
) -> None:
    """Persist summary, OCR extract, and raw per-page model output into a markdown report."""
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
        f"- Page evaluations: {len(page_answers)}",
        "",
        "## Prompt",
        "```text",
        prompt.strip(),
        "```",
        "",
        "## Model Summary",
        summary.strip() if summary.strip() else "[Empty summary]",
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

    md_lines.extend(["", "## Raw Model Output (by page)"])

    for entry in page_answers:
        page_no = entry.get("page", "?")
        answer_text = entry.get("answer", "").strip() or "[Empty response]"
        md_lines.extend(
            [
                "",
                f"### Page {page_no}",
                "```markdown",
                answer_text,
                "```",
            ]
        )

    output_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
