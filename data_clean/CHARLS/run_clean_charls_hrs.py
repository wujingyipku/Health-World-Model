"""Entry point for CHARLS to HRS external-validation cleaning.

Usage:
  python run_clean_charls_hrs.py
  python run_clean_charls_hrs.py --waves 1 2 3 4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "data" / "charls"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parents[1] / "data" / "charls"
DEFAULT_WAVES = [1, 2, 3, 4, 5]
HRS_PREPROCESS = SCRIPT_DIR.parents[1] / "data" / "hrs" / "HRS_preprocessing.json"
HRS_CONFIG = SCRIPT_DIR.parents[1] / "data" / "hrs" / "HRS_model_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Harmonized CHARLS Waves 1-5 into HRS-compatible tables."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--dictionary", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--waves", nargs="+", type=int, default=DEFAULT_WAVES)
    parser.add_argument("--no-hrs-scaler", action="store_true")
    parser.add_argument("--csv-copy", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(SCRIPT_DIR / "clean_charls_hrs.py"),
        "--data-dir",
        str(Path(args.data_dir)),
        "--output-dir",
        str(output_dir),
        "--waves",
        *[str(w) for w in args.waves],
        "--hrs-preprocessing",
        str(HRS_PREPROCESS),
        "--hrs-model-config",
        str(HRS_CONFIG),
        "--log-level",
        args.log_level,
    ]
    if str(args.dictionary).strip():
        command.extend(["--dictionary", str(Path(args.dictionary))])
    if args.no_hrs_scaler:
        command.append("--no-hrs-scaler")
    if args.csv_copy:
        command.append("--csv-copy")

    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    print()
    print("CHARLS HRS external-validation cleaning completed successfully.")
    print(f"Output folder: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
