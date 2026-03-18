from pathlib import Path
from typing import Any

from src.config import AppConfig
from src import pdf_ocr_pipeline as pipeline


class _DummyPdf:
    def __init__(self, page_count: int) -> None:
        self.page_count = page_count

    def __enter__(self) -> "_DummyPdf":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def test_run_conversion_pipeline_uses_one_page_calls(monkeypatch: Any) -> None:
    calls: list[str] = []

    def fake_open(_pdf_path: Path) -> _DummyPdf:
        return _DummyPdf(page_count=3)

    def fake_extract_text_by_page(**_kwargs: Any) -> list[dict[str, str]]:
        return [
            {"page": "1", "text": "ocr-1"},
            {"page": "2", "text": "ocr-2"},
            {"page": "3", "text": "ocr-3"},
        ]

    def fake_call_ollama_chat(**kwargs: Any) -> str:
        label = kwargs["call_label"]
        calls.append(label)
        if label == "page_reasoning":
            content = str(kwargs.get("content", ""))
            if "--- PAGE 1 ---" in content:
                return "page-1-answer"
            if "--- PAGE 2 ---" in content:
                return "page-2-answer"
            return "page-3-answer"
        if label == "summary_chunk":
            return "document-summary"
        return "unexpected"

    monkeypatch.setattr(pipeline.fitz, "open", fake_open)
    monkeypatch.setattr(pipeline, "extract_text_by_page", fake_extract_text_by_page)
    monkeypatch.setattr(pipeline, "call_ollama_chat", fake_call_ollama_chat)

    config = AppConfig(ocr_engine="paddle", stream=False, show_thinking=False)
    pages_data, summary, page_answers, metrics = pipeline.run_conversion_pipeline(
        pdf_path=Path("dummy.pdf"),
        config=config,
    )

    assert len(pages_data) == 3
    assert summary == "document-summary"
    assert [entry["answer"] for entry in page_answers] == [
        "page-1-answer",
        "page-2-answer",
        "page-3-answer",
    ]
    assert calls.count("page_reasoning") == 3
    assert calls.count("summary_chunk") == 1
    assert calls.count("final_reasoning") == 0
    assert metrics["pages_processed"] == 3
