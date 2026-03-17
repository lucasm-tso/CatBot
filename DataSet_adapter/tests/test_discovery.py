from pathlib import Path

from src.discovery import discover_pdfs


def test_discover_pdfs_recursive_sorted(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "z.pdf").write_bytes(b"x")
    (tmp_path / "b" / "a.pdf").write_bytes(b"x")
    (tmp_path / "b" / "ignore.txt").write_text("x", encoding="utf-8")

    out = discover_pdfs(tmp_path)
    assert [p.name for p in out] == ["a.pdf", "z.pdf"]


def test_discover_pdfs_invalid_folder() -> None:
    bad = Path("/definitely/not/found")
    try:
        discover_pdfs(bad)
    except ValueError:
        return
    assert False, "Expected ValueError"
