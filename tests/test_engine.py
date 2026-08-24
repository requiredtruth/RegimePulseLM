import copy
import csv
import io
import json
import unittest
from importlib.resources import files

from regimepulselm.core import RegimePulseError
from regimepulselm.engine import REGIMES, features, percentile, run_pipeline, validate_config
from regimepulselm.report import prompt, verify

CONFIG = json.loads(files("regimepulselm.data").joinpath("demo_config.json").read_text())
PRICES = [{"timestamp": row["timestamp"], "close": float(row["close"]), "volume": float(row["volume"])}
          for row in csv.DictReader(io.StringIO(files("regimepulselm.data").joinpath("demo.csv").read_text()))]


class EngineTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([0, 10], 0.5), 5)

    def test_pipeline_has_probabilities_and_drift(self):
        report = run_pipeline(CONFIG, PRICES)
        self.assertEqual(len(report["calibration"]), 20)
        self.assertEqual(len(report["test"]), 10)
        self.assertEqual(set(report["drift"]), {"short_return", "long_return", "realized_volatility", "volume_ratio"})
        for row in report["test"]:
            self.assertAlmostEqual(sum(row["probabilities"].values()), 1.0)
            self.assertIn(row["regime"], REGIMES)

    def test_test_changes_do_not_change_thresholds_or_calibration(self):
        baseline = run_pipeline(CONFIG, PRICES)
        changed = copy.deepcopy(PRICES)
        changed[-1]["close"] *= 1.2
        rerun = run_pipeline(CONFIG, changed)
        self.assertEqual(baseline["thresholds"], rerun["thresholds"])
        self.assertEqual(baseline["calibration"], rerun["calibration"])

    def test_future_change_cannot_change_past_features(self):
        baseline = features(PRICES, validate_config(CONFIG))
        changed = copy.deepcopy(PRICES)
        changed[-1]["close"] *= 1.2
        rerun = features(changed, validate_config(CONFIG))
        self.assertEqual(baseline[:-1], rerun[:-1])

    def test_exact_row_count_is_required(self):
        with self.assertRaisesRegex(RegimePulseError, "exactly 40"):
            run_pipeline(CONFIG, PRICES[:-1])

    def test_invalid_window_and_unknown_field_fail(self):
        invalid = copy.deepcopy(CONFIG)
        invalid["windows"]["short"] = invalid["windows"]["long"]
        with self.assertRaisesRegex(RegimePulseError, "short window"):
            run_pipeline(invalid, PRICES)
        invalid = copy.deepcopy(CONFIG)
        invalid["symbol"] = "EXAMPLE"
        with self.assertRaises(RegimePulseError):
            run_pipeline(invalid, PRICES)

    def test_verify_detects_tampering(self):
        report = run_pipeline(CONFIG, PRICES)
        self.assertIs(verify(CONFIG, PRICES, report), report)
        changed = copy.deepcopy(report)
        changed["claims"]["trade_signal"] = True
        with self.assertRaisesRegex(RegimePulseError, "recomputation"):
            verify(CONFIG, PRICES, changed)

    def test_prompt_omits_rows_and_timestamps(self):
        material = json.dumps(prompt(run_pipeline(CONFIG, PRICES)))
        self.assertNotIn(PRICES[0]["timestamp"], material)
        self.assertNotIn("close", material)
        self.assertIn("Do not predict", material)


if __name__ == "__main__":
    unittest.main()
