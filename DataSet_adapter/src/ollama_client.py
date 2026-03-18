"""Native Ollama chat client used by the adapter app."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import requests

from .metrics import metrics_add_time, metrics_inc


def call_ollama_chat(
    ollama_url: str,
    model: str,
    content: str,
    timeout_s: int | tuple[int, int],
    stream: bool,
    show_thinking: bool,
    metrics: dict[str, Any],
    call_label: str,
    images: list[str] | None = None,
) -> str:
    """Call Ollama /api/chat and return assistant content."""
    base = ollama_url.rstrip("/")
    endpoint = f"{base}/api/chat"
    user_message: dict[str, Any] = {"role": "user", "content": content}
    if images:
        user_message["images"] = images

    payload = {
        "model": model,
        "stream": stream,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant qui aide a analyser le contenu de documents PDF. "
                    "Sois precis et structure la reponse en Markdown."
                ),
            },
            user_message,
        ],
    }

    start = time.perf_counter()
    response = requests.post(endpoint, json=payload, timeout=timeout_s, stream=stream)
    response.raise_for_status()
    elapsed = time.perf_counter() - start
    metrics_inc(metrics, f"{call_label}_calls")
    metrics_add_time(metrics, f"{call_label}_seconds", elapsed)

    if not stream:
        body = response.json()
        message = body.get("message", {})
        answer = message.get("content", "")
        return str(answer).strip()

    content_chunks: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        event = json.loads(raw_line)
        message = event.get("message", {})
        chunk = str(message.get("content", ""))
        if chunk:
            content_chunks.append(chunk)
            metrics_inc(metrics, f"{call_label}_stream_chunks")
            print(chunk, end="", flush=True, file=sys.stderr)

        think = event.get("thinking") or message.get("thinking") or message.get("reasoning")
        if think and show_thinking:
            metrics_inc(metrics, f"{call_label}_thinking_chunks")
            print(str(think), end="", flush=True, file=sys.stderr)

    if content_chunks:
        print("", file=sys.stderr)
    return "".join(content_chunks).strip()
