"""Metrics helper functions."""

from __future__ import annotations

from typing import Any


def metrics_inc(metrics: dict[str, Any], key: str, value: int = 1) -> None:
    """Increment an integer metric."""
    metrics[key] = int(metrics.get(key, 0)) + int(value)


def metrics_add_time(metrics: dict[str, Any], key: str, seconds: float) -> None:
    """Accumulate timing metric in seconds."""
    metrics[key] = float(metrics.get(key, 0.0)) + float(seconds)


def round_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Round float metric values for stable reporting."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            out[key] = round(value, 4)
        else:
            out[key] = value
    return out


def aggregate_metric_dicts(metric_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numeric metrics across files."""
    out: dict[str, Any] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                out[key] = float(out.get(key, 0.0)) + float(value)
    return round_metrics(out)
