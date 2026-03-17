"""Single-file conversion orchestration."""

from __future__ import annotations

import time
from pathlib import Path

from .config import AppConfig
from .legacy_bridge import load_legacy_module
from .metrics import round_metrics
from .models import FileConversionResult
from .pdf_ocr_pipeline import run_conversion_pipeline


def output_markdown_path_for_pdf(pdf_path: Path) -> Path:
    """Return output markdown path next to source PDF."""
    return pdf_path.with_name(f"{pdf_path.stem}_ollama.md")


def convert_pdf_to_markdown(pdf_path: Path, config: AppConfig) -> FileConversionResult:
    """Convert one PDF and persist markdown output beside the source file."""
    start = time.perf_counter()
    output_path = output_markdown_path_for_pdf(pdf_path)

    try:
        pages_data, answer, metrics = run_conversion_pipeline(pdf_path=pdf_path, config=config)
        legacy = load_legacy_module()
        legacy.write_markdown(
            output_path=output_path,
            pdf_path=pdf_path,
            model=config.model,
            prompt=legacy.DEFAULT_PROMPT,
            pages_data=pages_data,
            answer=answer,
        )
        metrics = round_metrics(metrics)
        metrics["total_runtime_seconds"] = round(time.perf_counter() - start, 4)
        return FileConversionResult(
            pdf_path=pdf_path,
            output_md_path=output_path,
            summary_md_path=None,
            status="success",
            error_message=None,
            pages_processed=len(pages_data),
            model=config.model,
            metrics=metrics,
            duration_seconds=time.perf_counter() - start,
            uncertainty_score=0.0,
            uncertain_points=[],
        )
    except Exception as exc:  # noqa: BLE001
        return FileConversionResult(
            pdf_path=pdf_path,
            output_md_path=None,
            summary_md_path=None,
            status="failed",
            error_message=str(exc),
            pages_processed=0,
            model=config.model,
            metrics={"error": str(exc)},
            duration_seconds=time.perf_counter() - start,
            uncertainty_score=1.0,
            uncertain_points=["La conversion a echoue avant la generation du resume."],
        )
