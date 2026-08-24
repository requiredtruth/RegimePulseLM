"""Causal feature extraction, regime probabilities, and drift measurement."""

from __future__ import annotations

import math
from typing import Any, Sequence

from . import __version__
from .core import RegimePulseError, digest

FEATURES = ("short_return", "long_return", "realized_volatility", "volume_ratio")
REGIMES = ("bull", "bear", "range", "turbulent")


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_config(raw: Any) -> dict[str, Any]:
    required = {"schema_version", "windows", "split", "threshold_quantiles", "drift_bins"}
    if not isinstance(raw, dict) or set(raw) != required or raw.get("schema_version") != 1:
        raise RegimePulseError("config fields do not match schema version 1")
    windows = raw["windows"]
    if not isinstance(windows, dict) or set(windows) != {"short", "long", "volatility", "volume"}:
        raise RegimePulseError("windows requires short, long, volatility, volume")
    for field in windows:
        if isinstance(windows[field], bool) or not isinstance(windows[field], int) or not 2 <= windows[field] <= 10_000:
            raise RegimePulseError(f"windows.{field} must be an integer in 2..10000")
    if windows["short"] >= windows["long"]:
        raise RegimePulseError("short window must be less than long window")
    split = raw["split"]
    fields = {"warmup_rows", "calibration_rows", "embargo_rows", "test_rows"}
    if not isinstance(split, dict) or set(split) != fields:
        raise RegimePulseError("split requires warmup_rows, calibration_rows, embargo_rows, test_rows")
    minimum_warmup = max(windows.values())
    minimums = {"warmup_rows": minimum_warmup, "calibration_rows": 20, "embargo_rows": 1, "test_rows": 10}
    for field, minimum in minimums.items():
        if isinstance(split[field], bool) or not isinstance(split[field], int) or split[field] < minimum:
            raise RegimePulseError(f"split.{field} must be an integer >= {minimum}")
    quantiles = raw["threshold_quantiles"]
    if not isinstance(quantiles, dict) or set(quantiles) != {"trend", "volatility"}:
        raise RegimePulseError("threshold_quantiles requires trend and volatility")
    for field in quantiles:
        if isinstance(quantiles[field], bool) or not isinstance(quantiles[field], (int, float)) or not 0.05 <= quantiles[field] <= 0.95:
            raise RegimePulseError(f"threshold_quantiles.{field} must be in 0.05..0.95")
    drift_bins = raw["drift_bins"]
    if isinstance(drift_bins, bool) or not isinstance(drift_bins, int) or not 2 <= drift_bins <= 20:
        raise RegimePulseError("drift_bins must be an integer in 2..20")
    return {"schema_version": 1, "windows": dict(windows), "split": dict(split),
            "threshold_quantiles": {key: float(value) for key, value in quantiles.items()},
            "drift_bins": drift_bins}


def features(prices: Sequence[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute every row from current and earlier values only."""
    windows = config["windows"]
    warmup = config["split"]["warmup_rows"]
    result = []
    log_returns = [0.0]
    for index in range(1, len(prices)):
        log_returns.append(math.log(prices[index]["close"] / prices[index - 1]["close"]))
    for index in range(warmup, len(prices)):
        short = prices[index]["close"] / prices[index - windows["short"]]["close"] - 1.0
        long = prices[index]["close"] / prices[index - windows["long"]]["close"] - 1.0
        recent = log_returns[index - windows["volatility"] + 1:index + 1]
        volatility = math.sqrt(sum(value * value for value in recent) / len(recent))
        earlier_volume = [row["volume"] for row in prices[index - windows["volume"]:index]]
        volume_mean = sum(earlier_volume) / len(earlier_volume)
        ratio = prices[index]["volume"] / volume_mean if volume_mean > 0 else 1.0
        result.append({"timestamp": prices[index]["timestamp"], "short_return": short,
                       "long_return": long, "realized_volatility": volatility,
                       "volume_ratio": ratio})
    return result


def _softmax(scores: Sequence[float]) -> list[float]:
    peak = max(scores)
    values = [math.exp(max(-50.0, min(50.0, score - peak))) for score in scores]
    total = sum(values)
    return [value / total for value in values]


def classify(row: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    trend = max(thresholds["trend_abs"], 1e-12)
    volatility = max(thresholds["volatility_high"], 1e-12)
    long, short, vol = row["long_return"], row["short_return"], row["realized_volatility"]
    scores = [max(0.0, long / trend) + max(0.0, short / trend),
              max(0.0, -long / trend) + max(0.0, -short / trend),
              max(0.0, 1.0 - abs(long) / trend) + max(0.0, 1.0 - vol / volatility),
              vol / volatility]
    probabilities = _softmax(scores)
    chosen = max(range(len(REGIMES)), key=lambda index: (probabilities[index], -index))
    return {**row, "probabilities": {name: probabilities[index] for index, name in enumerate(REGIMES)},
            "regime": REGIMES[chosen], "confidence": probabilities[chosen]}


def _psi(calibration: Sequence[float], test: Sequence[float], bins: int) -> dict[str, Any]:
    edges = [percentile(calibration, index / bins) for index in range(1, bins)]
    def counts(values: Sequence[float]) -> list[int]:
        result = [0] * bins
        for value in values:
            bucket = 0
            while bucket < len(edges) and value > edges[bucket]:
                bucket += 1
            result[bucket] += 1
        return result
    left, right = counts(calibration), counts(test)
    epsilon = 1e-6
    left_p = [(value + epsilon) / (len(calibration) + epsilon * bins) for value in left]
    right_p = [(value + epsilon) / (len(test) + epsilon * bins) for value in right]
    psi = sum((b - a) * math.log(b / a) for a, b in zip(left_p, right_p))
    return {"psi": psi, "edges": edges, "calibration_counts": left, "test_counts": right}


def _transitions(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result = {left: {right: 0 for right in REGIMES} for left in REGIMES}
    for previous, current in zip(rows, rows[1:]):
        result[previous["regime"]][current["regime"]] += 1
    return result


def run_pipeline(raw_config: Any, prices: Sequence[dict[str, Any]]) -> dict[str, Any]:
    config = validate_config(raw_config)
    expected = sum(config["split"].values())
    if len(prices) != expected:
        raise RegimePulseError(f"price CSV has {len(prices)} rows; split requires exactly {expected}")
    feature_rows = features(prices, config)
    split = config["split"]
    calibration = feature_rows[:split["calibration_rows"]]
    test = feature_rows[-split["test_rows"]:]
    thresholds = {"trend_abs": percentile([abs(row["long_return"]) for row in calibration],
                                          config["threshold_quantiles"]["trend"]),
                  "volatility_high": percentile([row["realized_volatility"] for row in calibration],
                                                config["threshold_quantiles"]["volatility"])}
    classified_calibration = [classify(row, thresholds) for row in calibration]
    classified_test = [classify(row, thresholds) for row in test]
    drift = {name: _psi([row[name] for row in calibration], [row[name] for row in test],
                        config["drift_bins"]) for name in FEATURES}
    return {"schema_version": 1, "tool_version": __version__,
            "evidence": {"config_sha256": digest(config), "prices_sha256": digest(list(prices))},
            "split": split, "windows": config["windows"], "thresholds": thresholds,
            "regimes": list(REGIMES), "calibration": classified_calibration,
            "test": classified_test, "drift": drift,
            "transitions": {"calibration": _transitions(classified_calibration),
                            "test": _transitions(classified_test)},
            "claims": {"causal_features": True, "test_used_for_thresholds": False,
                       "mode": "offline_research_only", "trade_signal": False}}
