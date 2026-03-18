"""Batch execution service for recursive PDF conversion."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import AppConfig
from .conversion_service import convert_pdf_to_markdown
from .discovery import discover_pdfs
from .metrics import aggregate_metric_dicts
from .models import BatchRunResult, FileConversionResult
from .reporting import write_global_summary
from .summary_service import write_pdf_summary


ProgressCallback = Callable[[int, int, Path], None]
LogCallback = Callable[[str], None]


def run_batch_conversion(
    root_folder: Path,
    config: AppConfig,
    progress_cb: ProgressCallback | None = None,
    log_cb: LogCallback | None = None,
    pdf_paths: list[Path] | None = None,
) -> BatchRunResult:
    """Run sequential conversion for all discovered PDFs under root folder."""
    started_at = datetime.now()
    queue = pdf_paths if pdf_paths is not None else discover_pdfs(root_folder)
    results: list[FileConversionResult] = []

    total = len(queue)
    for idx, pdf_path in enumerate(queue, start=1):
        if progress_cb is not None:
            progress_cb(idx, total, pdf_path)
        if log_cb is not None:
            log_cb(f"[{idx}/{total}] Processing: {pdf_path}")

        result = convert_pdf_to_markdown(pdf_path=pdf_path, config=config)
        result = write_pdf_summary(result=result, config=config)
        results.append(result)

        if log_cb is not None:
            if result.status == "success":
                log_cb(f"[{idx}/{total}] Success: {result.output_md_path}")
            else:
                log_cb(f"[{idx}/{total}] Failed: {result.error_message}")

    ended_at = datetime.now()
    success_count = sum(1 for r in results if r.status == "success")
    failure_count = sum(1 for r in results if r.status != "success")
    aggregate_metrics = aggregate_metric_dicts([r.metrics for r in results])

    batch_result = BatchRunResult(
        root_folder=root_folder.expanduser().resolve(),
        started_at=started_at,
        ended_at=ended_at,
        total_files=total,
        success_count=success_count,
        failure_count=failure_count,
        results=results,
        aggregate_metrics=aggregate_metrics,
        global_summary_path=None,
    )
    batch_result.global_summary_path = write_global_summary(batch_result)
    return batch_result
