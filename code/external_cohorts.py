"""ELSA and CHARLS labels for external evaluation.

Person-level data are not included in this repository. The paths below are the
files written by the HRS, ELSA, and CHARLS cleaners. See ``data/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_HRS = PACKAGE_ROOT / "data" / "hrs"
DATA_ELSA = PACKAGE_ROOT / "data" / "elsa"
DATA_CHARLS = PACKAGE_ROOT / "data" / "charls"
HRS_TEST_PARQUET = DATA_HRS / "HRS_test.parquet"


@dataclass(frozen=True)
class ExternalCohort:
    name: str
    label: str
    parquet: Path
    output_subdir: str
    note: str


COHORTS: dict[str, ExternalCohort] = {
    "elsa": ExternalCohort(
        name="elsa",
        label="ELSA (England, Waves 1-10)",
        parquet=DATA_ELSA / "ELSA_external.parquet",
        output_subdir="evaluation_elsa",
        note=(
            "HRS-trained model and HRS-trained Logistic/MLP evaluated on ELSA. "
            "Training history is from HRS, not ELSA."
        ),
    ),
    "charls": ExternalCohort(
        name="charls",
        label="CHARLS (China, Waves 1-5)",
        parquet=DATA_CHARLS / "CHARLS_external.parquet",
        output_subdir="evaluation_charls",
        note=(
            "HRS-trained model and HRS-trained Logistic/MLP evaluated on CHARLS. "
            "Training history is from HRS, not CHARLS. "
            "Use the HRS-aligned table."
        ),
    ),
}


def get_cohort(name: str) -> ExternalCohort:
    key = str(name).strip().lower()
    if key not in COHORTS:
        known = ", ".join(sorted(COHORTS))
        raise KeyError(f"Unknown external cohort {name!r}. Choose one of: {known}")
    return COHORTS[key]
