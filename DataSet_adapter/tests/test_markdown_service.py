from pathlib import Path

from src.markdown_service import write_markdown


def test_write_markdown_includes_summary_ocr_and_raw_outputs(tmp_path: Path) -> None:
    output_path = tmp_path / "doc_ollama.md"
    pages_data = [
        {"page": "1", "text": "OCR page one"},
        {"page": "2", "text": "OCR page two"},
    ]
    page_answers = [
        {"page": "1", "answer": "# Page 1\nContent 1"},
        {"page": "2", "answer": "# Page 2\nContent 2"},
    ]

    write_markdown(
        output_path=output_path,
        pdf_path=tmp_path / "doc.pdf",
        model="qwen3.5:27b",
        prompt="Prompt text",
        pages_data=pages_data,
        summary="## Summary\n- Key point",
        page_answers=page_answers,
    )

    text = output_path.read_text(encoding="utf-8")

    assert "## Model Summary" in text
    assert "## OCR Extract (by page)" in text
    assert "## Raw Model Output (by page)" in text
    assert "### Page 1" in text
    assert "### Page 2" in text
    assert "```markdown" in text
    assert "# Page 1" in text
    assert "# Page 2" in text
