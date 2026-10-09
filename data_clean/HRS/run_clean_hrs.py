"""Entry point for HRS data cleaning.

Usage:
  python run_clean_hrs.py
  python run_clean_hrs.py --validation-size 0.10 --test-size 0.20
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_HRS = SCRIPT_DIR.parents[1] / "data" / "hrs"
DEFAULT_RAND_FILE = DATA_HRS / "randhrs1992_2022v1.dta"
DEFAULT_HARMONIZED_FILE = DATA_HRS / "H_HRS_d.dta"
DEFAULT_OUTPUT_DIR = DATA_HRS
DEFAULT_WAVES = [8, 9, 10, 11, 12, 13, 14]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean HRS Waves 8-14 into train/validation/test tables."
    )
    parser.add_argument("--rand-file", default=str(DEFAULT_RAND_FILE))
    parser.add_argument("--harmonized-file", default=str(DEFAULT_HARMONIZED_FILE))
    parser.add_argument("--dictionary", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--waves", nargs="+", type=int, default=DEFAULT_WAVES)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--no-csv-copy",
        action="store_true",
        help="Do not write CSV copies of transition tables.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(SCRIPT_DIR / "clean_hrs.py"),
        "--rand-file",
        str(Path(args.rand_file)),
        "--harmonized-file",
        str(Path(args.harmonized_file)),
        "--output-dir",
        str(output_dir),
        "--waves",
        *[str(wave) for wave in args.waves],
        "--validation-size",
        str(args.validation_size),
        "--test-size",
        str(args.test_size),
        "--seed",
        str(args.seed),
        "--log-level",
        args.log_level,
    ]
    if str(args.dictionary).strip():
        command.extend(["--dictionary", str(Path(args.dictionary))])
    if not args.no_csv_copy:
        command.append("--csv-copy")

    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)

    print()
    print("HRS cleaning completed successfully.")
    print(f"Output folder: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
