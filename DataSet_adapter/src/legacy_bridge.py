"""Helpers to load legacy conversion functions from run_pdf_to_md.py."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def _legacy_script_path() -> Path:
    return Path(__file__).resolve().parents[2] / "pdf_to_ollama_md" / "run_pdf_to_md.py"


@lru_cache(maxsize=1)
def load_legacy_module() -> ModuleType:
    """Load the existing conversion script as a module.

    Keeping this bridge avoids copying a very large, already validated pipeline.
    """
    script_path = _legacy_script_path()
    if not script_path.exists():
        raise FileNotFoundError(f"Legacy script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("legacy_run_pdf_to_md", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load legacy module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
