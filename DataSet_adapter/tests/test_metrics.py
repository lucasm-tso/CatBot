from src.metrics import aggregate_metric_dicts, round_metrics


def test_round_metrics_float_precision() -> None:
    out = round_metrics({"a": 1.234567, "b": 2, "c": "x"})
    assert out["a"] == 1.2346
    assert out["b"] == 2
    assert out["c"] == "x"


def test_aggregate_metric_dicts_numeric_only() -> None:
    agg = aggregate_metric_dicts([
        {"x": 1, "y": 2.5, "k": "ignore"},
        {"x": 3, "y": 1.5},
    ])
    assert agg["x"] == 4.0
    assert agg["y"] == 4.0
