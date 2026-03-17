from datetime import datetime
from pathlib import Path

from src.models import BatchRunResult, FileConversionResult
from src.reporting import write_global_summary


def test_write_global_summary(tmp_path: Path) -> None:
    result = FileConversionResult(
        pdf_path=tmp_path / "doc.pdf",
        output_md_path=tmp_path / "doc_ollama.md",
        summary_md_path=tmp_path / "doc_summary.md",
        status="success",
        error_message=None,
        pages_processed=2,
        model="qwen3.5:27b",
        metrics={"pages_processed": 2},
        duration_seconds=1.2,
        uncertainty_score=0.1,
        uncertain_points=[],
    )
    batch = BatchRunResult(
        root_folder=tmp_path,
        started_at=datetime.now(),
        ended_at=datetime.now(),
        total_files=1,
        success_count=1,
        failure_count=0,
        results=[result],
        aggregate_metrics={"pages_processed": 2},
        global_summary_path=None,
    )
    out = write_global_summary(batch)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Batch Conversion Summary" in text
    assert "doc.pdf" in text
