# Project specification

RegimePulseLM 0.1.0 transforms one strictly ordered historical `timestamp,close,volume` CSV into causal features, calibration-fitted thresholds, four rule-based regime probabilities, transition counts, and calibration-to-test feature drift.

The raw sequence is exhaustively partitioned as `warmup | calibration | embargo | test`. Warmup supports trailing features. Only calibration rows set thresholds. Embargo rows are not classified in the public report. Test rows measure drift and receive labels without changing earlier state.

Stable commands are `run`, `verify`, `summary`, `prompt`, and `demo`. JSON output is canonical and fully reproducible. The project never exposes a live-trading interface.
