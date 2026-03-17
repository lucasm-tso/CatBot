"""Ollama call wrappers for future extension.

Current implementation uses legacy wrappers from run_pdf_to_md.py to preserve behavior.
"""

from __future__ import annotations

from typing import Any

from .legacy_bridge import load_legacy_module


def call_ollama_chat(
    ollama_url: str,
    model: str,
    content: str,
    timeout_s: int,
    stream: bool,
    show_thinking: bool,
    metrics: dict[str, Any],
    call_label: str,
) -> str:
    """Delegate chat call to legacy implementation for behavior parity."""
    legacy = load_legacy_module()
    return legacy.call_ollama(
        ollama_url=ollama_url,
        model=model,
        content=content,
        timeout_s=timeout_s,
        stream=stream,
        show_thinking=show_thinking,
        metrics=metrics,
        call_label=call_label,
    )
