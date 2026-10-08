"""Replot manuscript Figures 2–5 from the aggregate tables in ``data/figures/``.

- Figure 2 — HRS open-loop dynamics
- Figure 3 — ELSA open-loop dynamics
- Figure 4 — CHARLS open-loop dynamics
- Figure 5 — HRS / ELSA / CHARLS light-activity open-loop

These tables are cohort-level summaries (AUROC, sensitivity, risk tertiles,
Kaplan–Meier curves, age-standardized rates). They do not contain person-level
records. Drawing them does not load the checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import evaluate_and_plot_hrs_jepa as ep


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "data" / "figures"

DYNAMICS_SOURCES = (
    ("figure2_hrs_openloop.png", FIGURES / "fig2_hrs"),
    ("figure3_elsa_openloop.png", FIGURES / "fig3_elsa"),
    ("figure4_charls_openloop.png", FIGURES / "fig4_charls"),
)

FIG5_COHORTS = (
    ("HRS", FIGURES / "fig5_light_activity" / "hrs"),
    ("ELSA", FIGURES / "fig5_light_activity" / "elsa"),
    ("CHARLS", FIGURES / "fig5_light_activity" / "charls"),
)

FIG5_TABLES = {
    "observed_strata": "fig5_observed_action_strata.csv",
    "sim_observed_action": "fig5_sim_observed_action.csv",
    "observed_h1_counterfactual": "fig5_observed_h1_counterfactual.csv",
    "sim_h1_counterfactual": "fig5_sim_h1_counterfactual.csv",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _figure_prefix(src: Path) -> str:
    """``fig2_hrs`` -> ``fig2``. Matches the CSV prefix inside that folder."""
    head = src.name.split("_", 1)[0]
    if not head.startswith("fig"):
        raise ValueError(f"Expected a figN_* folder, got {src.name}")
    return head


def plot_dynamics(src: Path, dest: Path, dpi: int) -> Path:
    """Open-loop AUROC, sensitivity, Sen@Spec≥0.8, risk tertiles, and death KM."""
    prefix = _figure_prefix(src)
    scratch = dest.parent / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    produced = ep.plot_figS3_dynamic(
        worsening_dynamics=_read_csv(src / f"{prefix}_openloop_dynamics.csv"),
        clinical_dynamics=_read_csv(src / f"{prefix}_event_openloop_dynamics.csv"),
        km=_read_csv(src / f"{prefix}_kaplan_meier.csv"),
        worsening_risk_summary=_read_csv(src / f"{prefix}_openloop_risk_summary.csv"),
        clinical_risk_summary=_read_csv(src / f"{prefix}_clinical_openloop_risk_summary.csv"),
        figures_dir=scratch,
        dpi=dpi,
    )
    if produced is None:
        raise RuntimeError(f"plot_figS3_dynamic wrote nothing for {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    produced.replace(dest)
    return dest


def plot_light_activity(dest: Path, dpi: int) -> Path:
    """Age-standardized death risk under observed light activity and a switch at horizon 2."""
    cohorts = []
    for label, folder in FIG5_COHORTS:
        tables = {key: _read_csv(folder / name) for key, name in FIG5_TABLES.items()}
        cohorts.append((label, tables))
    dest.parent.mkdir(parents=True, exist_ok=True)
    produced = ep.plot_fig4_cohort3(
        cohorts,
        dest.parent,
        dpi,
        action="light_activity_frequency",
        event="death_event",
        output_name=dest.name,
    )
    if produced is None:
        raise RuntimeError("plot_fig4_cohort3 wrote nothing")
    return Path(produced)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HERE / "results",
        help="Directory for the four PNG files (default: results/).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ep.set_style()

    written: list[Path] = []
    for name, src in DYNAMICS_SOURCES:
        path = plot_dynamics(src, out / name, args.dpi)
        written.append(path)
        print(path)
    fig5 = plot_light_activity(out / "figure5_light_activity.png", args.dpi)
    written.append(fig5)
    print(fig5)

    scratch = out / "_scratch"
    if scratch.exists() and not any(scratch.iterdir()):
        scratch.rmdir()


if __name__ == "__main__":
    main()
