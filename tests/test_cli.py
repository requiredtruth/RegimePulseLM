import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "regimepulselm/data/demo_config.json"
PRICES = ROOT / "regimepulselm/data/demo.csv"


class CliTests(unittest.TestCase):
    def invoke(self, *args, check=True):
        return subprocess.run([sys.executable, "-m", "regimepulselm", *map(str, args)], cwd=ROOT,
                              text=True, capture_output=True, check=check)

    def test_demo_is_deterministic_plain_text(self):
        first, second = self.invoke("demo").stdout, self.invoke("demo").stdout
        self.assertEqual(first, second)
        self.assertIn("not_a_trade_signal", first)
        self.assertNotIn("\x1b", first)

    def test_run_verify_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            self.invoke("run", CONFIG, PRICES, report)
            self.assertEqual(self.invoke("verify", CONFIG, PRICES, report).stdout, "report verified\n")
            self.assertIn("test regimes", self.invoke("summary", report).stdout)

    def test_bad_csv_fails_without_traceback(self):
        result = self.invoke("run", CONFIG, CONFIG, "out.json", check=False)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
