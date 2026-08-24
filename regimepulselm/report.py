"""Full recomputation, terminal summary, and aggregate prompt export."""

from __future__ import annotations

from typing import Any

from .core import RegimePulseError, canonical_bytes, digest
from .engine import REGIMES, run_pipeline


def verify(config: Any, prices: list[dict[str, Any]], report: Any) -> dict[str, Any]:
    expected = run_pipeline(config, prices)
    if canonical_bytes(expected) != canonical_bytes(report):
        raise RegimePulseError("report does not match deterministic recomputation")
    return report


def summary(report: Any) -> str:
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise RegimePulseError("report schema_version must be 1")
    counts = {name: 0 for name in REGIMES}
    for row in report["test"]:
        counts[row["regime"]] += 1
    drift = sorted(report["drift"].items(), key=lambda item: (-item[1]["psi"], item[0]))
    rows = ["RegimePulseLM causal regime report",
            f"rows calibration={report['split']['calibration_rows']} embargo={report['split']['embargo_rows']} test={report['split']['test_rows']}",
            "test regimes " + " ".join(f"{name}={counts[name]}" for name in REGIMES),
            f"threshold trend_abs={report['thresholds']['trend_abs']:.8f}",
            f"threshold volatility_high={report['thresholds']['volatility_high']:.8f}",
            "drift " + " ".join(f"{name}={value['psi']:.6f}" for name, value in drift),
            "mode=offline_research_only not_a_trade_signal"]
    return "\n".join(rows) + "\n"


def prompt(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise RegimePulseError("report schema_version must be 1")
    counts = {name: sum(row["regime"] == name for row in report["test"]) for name in REGIMES}
    facts = {"partition_counts": {key: report["split"][key]
                                  for key in ("calibration_rows", "embargo_rows", "test_rows")},
             "thresholds": report["thresholds"], "test_regime_counts": counts,
             "feature_drift_psi": {name: value["psi"] for name, value in report["drift"].items()},
             "test_transitions": report["transitions"]["test"], "claims": report["claims"]}
    return {"facts_sha256": digest(facts), "messages": [
        {"role": "system", "content": "Explain only the supplied causal regime and drift facts. Do not predict a future regime, recommend a trade, imply profit, or call these labels ground truth. Discuss uncertainty and sample size."},
        {"role": "user", "content": canonical_bytes(facts).decode().rstrip("\n")}]}
