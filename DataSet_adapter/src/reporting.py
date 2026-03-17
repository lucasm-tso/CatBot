"""Global batch summary reporting."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .models import BatchRunResult


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_global_summary(batch: BatchRunResult) -> Path:
    """Write global markdown summary for the batch run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = batch.root_folder / f"_batch_conversion_summary_{timestamp}.md"

    lines = [
        "# Batch Conversion Summary",
        "",
        "## Overview",
        f"- Root folder: `{batch.root_folder}`",
        f"- Started at: {batch.started_at.isoformat(timespec='seconds')}",
        f"- Ended at: {batch.ended_at.isoformat(timespec='seconds')}",
        f"- Total files: {batch.total_files}",
        f"- Success: {batch.success_count}",
        f"- Failed: {batch.failure_count}",
        "",
        "## Aggregate Metrics",
        "```json",
        json.dumps(batch.aggregate_metrics, ensure_ascii=True, indent=2),
        "```",
        "",
        "## Per-file Status",
        "| File | Status | Pages | Uncertainty | Output | Summary |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]

    for result in batch.results:
        output_text = str(result.output_md_path) if result.output_md_path else "-"
        summary_text = str(result.summary_md_path) if result.summary_md_path else "-"
        lines.append(
            "| "
            f"{_safe_rel(result.pdf_path, batch.root_folder)} | "
            f"{result.status} | "
            f"{result.pages_processed} | "
            f"{result.uncertainty_score:.3f} | "
            f"{output_text} | "
            f"{summary_text} |"
        )

    lines.extend(["", "## Debug Blocks (Per File)"])
    for result in batch.results:
        lines.extend(
            [
                "",
                f"### {result.pdf_path.name}",
                "```json",
                json.dumps(
                    {
                        "output": str(result.output_md_path) if result.output_md_path else None,
                        "pages_processed": result.pages_processed,
                        "model": result.model,
                        "metrics": result.metrics,
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                "```",
            ]
        )

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path
