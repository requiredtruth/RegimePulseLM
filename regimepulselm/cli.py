"""Command-line interface."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from importlib.resources import files

from .core import RegimePulseError, atomic_json, canonical_bytes, load_json, load_prices
from .engine import run_pipeline
from .report import prompt, summary, verify


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="regimepulselm", description="Causal market-regime labels and drift diagnostics")
    sub = result.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("config")
    run.add_argument("prices")
    run.add_argument("output")
    check = sub.add_parser("verify")
    check.add_argument("config")
    check.add_argument("prices")
    check.add_argument("report")
    show = sub.add_parser("summary")
    show.add_argument("report")
    export = sub.add_parser("prompt")
    export.add_argument("report")
    export.add_argument("output", nargs="?", default="-")
    sub.add_parser("demo")
    return result


def _demo_prices() -> list[dict]:
    text = files("regimepulselm.data").joinpath("demo.csv").read_text()
    rows = []
    for raw in csv.DictReader(io.StringIO(text)):
        rows.append({"timestamp": raw["timestamp"], "close": float(raw["close"]),
                     "volume": float(raw["volume"])})
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            report = run_pipeline(load_json(args.config), load_prices(args.prices))
            atomic_json(args.output, report)
            print(f"wrote deterministic report: {args.output}")
        elif args.command == "verify":
            verify(load_json(args.config), load_prices(args.prices), load_json(args.report))
            print("report verified")
        elif args.command == "summary":
            sys.stdout.write(summary(load_json(args.report)))
        elif args.command == "prompt":
            material = prompt(load_json(args.report))
            if args.output == "-":
                sys.stdout.buffer.write(canonical_bytes(material))
            else:
                atomic_json(args.output, material)
        elif args.command == "demo":
            config = json.loads(files("regimepulselm.data").joinpath("demo_config.json").read_text())
            sys.stdout.write(summary(run_pipeline(config, _demo_prices())))
        return 0
    except (RegimePulseError, OSError, KeyError, TypeError) as exc:
        print(f"regimepulselm: {exc}", file=sys.stderr)
        return 2
