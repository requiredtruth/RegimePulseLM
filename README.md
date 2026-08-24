# RegimePulseLM

Dependency-free, causal market-regime labeling with explicit probabilities, transition counts, and feature-drift diagnostics. It processes a declared historical CSV offline; it does not download prices, predict the next regime, produce a trade signal, or place orders.

```console
$ ./install.sh
...
RegimePulseLM causal regime report
rows calibration=20 embargo=2 test=10
test regimes bull=0 bear=0 range=0 turbulent=10
...
mode=offline_research_only not_a_trade_signal
```

The command compiles the package, runs every test, and executes a deterministic synthetic demonstration with Python 3.10+ and no runtime dependencies.

## Concrete distinction

Hidden Markov models are a common market-regime approach, including hierarchical methods intended to separate short- and long-term behavior ([Oelschläger and Adam](https://arxiv.org/abs/2007.14874)). RegimePulseLM is deliberately narrower and more auditable: it is not an HMM and does not claim latent states are truth. It computes four trailing-only features, learns two thresholds from a declared calibration prefix, excludes embargo and test rows from threshold fitting, emits interpretable regime probabilities, and measures population-stability-index drift for every feature.

It directly handles searches and failures such as **"market regime data leakage"**, **"future data changed historical regime labels"**, **"timestamp order invalid"**, **"crypto volatility regime drift"**, and **"regime classifier without sklearn"**.

## Input

CSV header:

```csv
timestamp,close,volume
2026-01-01T00:00:00Z,100.25,1200
```

Configuration:

```json
{
  "schema_version": 1,
  "windows": {"short": 3, "long": 8, "volatility": 5, "volume": 5},
  "split": {"warmup_rows": 8, "calibration_rows": 100, "embargo_rows": 5, "test_rows": 50},
  "threshold_quantiles": {"trend": 0.5, "volatility": 0.75},
  "drift_bins": 5
}
```

The row count must exactly equal the four partition counts. Timestamps must be unique, increasing, canonical UTC seconds. Close must be positive; volume must be non-negative.

```bash
./run.sh run config.json prices.csv report.json
./run.sh verify config.json prices.csv report.json
./run.sh summary report.json
./run.sh prompt report.json local-commentary-prompt.json
```

`verify` recomputes the complete report. `prompt` contains aggregate facts only and tells an optional local LLM not to predict regimes, recommend trades, or imply profit.

## Causal features and labels

- Short and long trailing returns use only current and earlier closes.
- Realized volatility uses trailing log returns only.
- Volume ratio compares current volume with earlier trailing volume.
- Calibration determines the absolute-trend and high-volatility thresholds.
- Softmax probabilities express rule-score ambiguity; they are not statistical posterior probabilities.
- Labels are `bull`, `bear`, `range`, and `turbulent`; they describe this rule set, not objective market truth.

Tests explicitly prove that modifying a future row cannot alter earlier features, and that modifying test data cannot alter thresholds or calibration labels.

## Limitations and safety

- Offline historical research only; no credentials, exchange connection, live feed, wallet, signing, orders, custody, or trading.
- No profit, alpha, return, or investment claim.
- No fees, spread, slippage, funding, survivorship, or execution model.
- PSI is sample- and bin-dependent; small partitions are unstable.
- Fixed trailing windows and two calibration thresholds cannot capture all structural changes.
- The tool cannot detect leakage already embedded in input prices or upstream processing.

## Support

[Donations fund additional development time](SUPPORT.md). A confirmed public transaction hash may accompany a funded-direction issue, but cannot purchase ownership, returns, priority, deadlines, acceptance, or prohibited work.

Apache-2.0 licensed. See [LICENSE](LICENSE).
