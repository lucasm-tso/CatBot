"""Typed models for conversion and batch results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class FileConversionResult:
    """Outcome for one PDF conversion."""

    pdf_path: Path
    output_md_path: Path | None
    summary_md_path: Path | None
    status: str
    error_message: str | None
    pages_processed: int
    model: str
    metrics: dict[str, Any]
    duration_seconds: float
    uncertainty_score: float
    uncertain_points: list[str] = field(default_factory=list)


@dataclass
class BatchRunResult:
    """Aggregate outcome for one batch execution."""

    root_folder: Path
    started_at: datetime
    ended_at: datetime
    total_files: int
    success_count: int
    failure_count: int
    results: list[FileConversionResult]
    aggregate_metrics: dict[str, Any]
    global_summary_path: Path | None
