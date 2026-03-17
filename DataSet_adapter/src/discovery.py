"""Recursive PDF discovery utilities."""

from __future__ import annotations

from pathlib import Path


def discover_pdfs(root_folder: Path) -> list[Path]:
    """Recursively discover PDF files under a folder in deterministic order."""
    root = root_folder.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid folder: {root}")

    pdfs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(pdfs, key=lambda p: (p.name.lower(), str(p).lower()))
