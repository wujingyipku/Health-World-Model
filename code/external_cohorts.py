"""ELSA and CHARLS labels for external evaluation.

Full cohort tables are not part of this release. The paths below sit inside
this package and are absent unless a controlled-access table is placed there.
See ``data/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_COHORT_DIR = PACKAGE_ROOT / "data" / "full_cohorts"
HRS_TEST_PARQUET = _COHORT_DIR / "HRS_test.parquet"


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
        parquet=_COHORT_DIR / "ELSA_external.parquet",
        output_subdir="evaluation_elsa",
        note=(
            "HRS-trained model and HRS-trained Logistic/MLP evaluated on ELSA. "
            "Training history is from HRS, not ELSA."
        ),
    ),
    "charls": ExternalCohort(
        name="charls",
        label="CHARLS (China, Waves 1-5)",
        parquet=_COHORT_DIR / "CHARLS_external.parquet",
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
