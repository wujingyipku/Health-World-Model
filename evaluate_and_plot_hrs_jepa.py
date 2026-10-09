
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch

from code.data import build_loader
from code.external_cohorts import (
    COHORTS,
    HRS_TEST_PARQUET,
    ExternalCohort,
    get_cohort,
)
from code.specs import ModelSpec
from code.evaluation import (
    CLINICAL_EVENT_REWARDS,
    DEFAULT_BINARY_POS_WEIGHT_MAX,
    EVENT_LABELS,
    FIG6_SIMULATION_EVENTS,
    KM_PANEL_ORDER,
    WORSENING_REWARDS,
    attach_baselines,
    collect_one_step_predictions,
    discover_binary_event_names,
    load_agent,
    prepare_baselines,
    read_table,
    resolve_device,
    selected_checkpoint,
)
from code.fig_report import (
    FIG2_PCA_REWARDS,
    FIG2_TEST_MAX_ROWS,
    FIG2_TRAIN_MAX_ROWS,
    FIG5_EXCLUDED_ACTIONS,
    FIG5_SIMULATION_ACTIONS,
    FIG5_SWITCH_HORIZON,
    PRIMARY_LEVELS,
    build_fig2_cache_manifest,
    calibration_curve_frame,
    clinical_compare_table,
    collect_persistent_action_validation,
    collect_h1_low_counterfactual_validation,
    collect_fig5_action_death_km,
    collect_fig5_smoking_death_km,
    load_fig5_km_cache,
    save_fig5_km_cache,
    FIG4_ACTION_ORDER,
    FIG4_DEFAULT_ACTION,
    FIG5_VALIDATION_OUTCOMES,
    FIG5_H1_CF_SWITCH_HORIZON,
    FIG5_KM_ACTION,
    get_fig4_action_spec,
    resolve_fig5_simulation_actions,
    collect_jepa_event_rollouts,
    collect_case_study_rollout,
    pick_case_study_person,
    FIG6_CASE_STUDY_OUTCOMES,
    FIG6_CASE_STUDY_MAX_HORIZON,
    FIG6_CASE_STUDY_SWITCH_HORIZON,
    FIG6_CASE_STUDY_SCENARIO_LABELS,
    FIG6_CASE_STUDY_SCENARIO_ORDER,
    collect_jepa_latent_prediction,
    collect_jepa_level_rollouts,
    collect_latent_change_vs_decline,
    collect_latent_rollout_mse,
    collect_latent_variance,
    collect_latents_and_health,
    compute_latent_2d,
    compute_latent_pca_by_rewards,
    filter_fig5_actions,
    fit_binary_probe_metrics,
    kaplan_meier_openloop_event_horizons,
    level_compare_metrics,
    load_or_fit_mlp_tabular,
    load_or_fit_mlp_horizon_star,
    attach_mlp_horizon_star_probs,
    mlp_horizon_star_cache_paths,
    load_training_history_curves,
    fig1c_export_table,
    collect_one_step_person_trajectories,
    mlp_events_long_from_wide,
    mlp_tabular_cache_paths,
    pick_example_trajectories,
    prepare_linear_and_persistence,
    risk_stratification_frame,
    summarize_event_openloop_dynamics,
    summarize_event_openloop_dynamics_hstar,
    summarize_event_risk_rollout,
    summarize_latent_by_horizon,
    summarize_level_openloop_mse,
    collect_openloop_event_risk,
    try_load_fig2_cache,
    try_load_fig5_validation_cache,
    write_fig2_cache,
    write_fig5_validation_cache,
    build_fig5_validation_cache_manifest,
)


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = DEFAULT_ROOT / "weights"
DEFAULT_DATA_DIR = DEFAULT_ROOT / "data" / "hrs"

# Nature Communications / ggsci "nature" palette (colorblind-friendly).
NATURE = {
    "red": "#E64B35",
    "blue": "#4DBBD5",
    "green": "#00A087",
    "navy": "#3C5488",
    "salmon": "#F39B7F",
    "grey": "#8491B4",
    "mint": "#91D1C2",
    "light_grey": "#DCDFE6",
    "neutral": "#ECECEC",
}
BLUE = NATURE["navy"]  # JEPA / primary model
ORANGE = NATURE["red"]  # MLP / high risk
GREEN = NATURE["green"]  # low risk / safer pole
GREY = NATURE["grey"]  # baselines / reference
PURPLE = NATURE["salmon"]
YELLOW = NATURE["blue"]  # mid risk
NATURE_CYCLE = [
    NATURE["navy"],
    NATURE["red"],
    NATURE["green"],
    NATURE["blue"],
    NATURE["salmon"],
    NATURE["grey"],
    NATURE["mint"],
]


def nature_color(index: int) -> str:
    return NATURE_CYCLE[index % len(NATURE_CYCLE)]


# Risk tertiles and KM: Nature sequential (green → blue → red).
RISK_COLORS = {
    "low": NATURE["green"],
    "mid": NATURE["blue"],
    "high": NATURE["red"],
    "all": NATURE["navy"],
}
RISK_STRATUM_LABELS = {
    "low": "Low risk",
    "mid": "Mid risk",
    "high": "High risk",
    "all": "All",
}
OBS_STRATUM_COLORS = {
    "observed event": NATURE["navy"],
    "observed death": NATURE["navy"],
    "no observed event": NATURE["grey"],
    "no observed death": NATURE["grey"],
}
# Open-loop combined panels: color by series type (same across outcomes).
OPENLOOP_REL_COLOR = NATURE["blue"]
OPENLOOP_JEPA_ABS_COLOR = BLUE
OPENLOOP_MLP_ABS_COLOR = ORANGE
WM_LEGEND = "WM"


def _legend_label(name: str) -> str:
    return WM_LEGEND if name == "JEPA" else name


# Fig 3 unified series style (Nature palette).
FIG3_REL_COLOR = "#6E6E6E"
FIG3_SERIES = {
    "relative": {
        "color": FIG3_REL_COLOR,
        "marker": "x",
        "linestyle": "--",
        "linewidth": 1.3,
        "markersize": 4.0,
        "label": "Relative (vs MLP h=1)",
    },
    "wm": {
        "color": NATURE["navy"],
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.2,
        "markersize": 3.5,
        "label": WM_LEGEND,
    },
    "mlp": {
        "color": NATURE["blue"],
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.1,
        "markersize": 4.0,
        "label": "MLP h=1",
    },
}
FIGS3_SERIES = {
    "relative": {
        **FIG3_SERIES["relative"],
        "label": "Relative (vs MLP-h*)",
    },
    "wm": FIG3_SERIES["wm"],
    "mlp": {
        **FIG3_SERIES["mlp"],
        "label": "MLP-h*",
    },
}
FIG3_HORIZONS = (1, 2, 3, 4, 5)
FIG3_STRATUM_ORDER = ("low", "mid", "high", "all")
FIG3_OBS_LABELS = {
    "observed event": "Observed",
    "observed death": "Observed",
    "no observed event": "Not observed",
    "no observed death": "Not observed",
}
FIG3_OBS_STYLE = {
    "observed event": {"color": "#6E6E6E", "linestyle": "--", "marker": "x"},
    "observed death": {"color": "#6E6E6E", "linestyle": "--", "marker": "x"},
    "no observed event": {"color": "#9E9E9E", "linestyle": ":", "marker": "+"},
    "no observed death": {"color": "#9E9E9E", "linestyle": ":", "marker": "+"},
}


def _fig3_ordered_strata(strata: Sequence[str] | pd.Series | np.ndarray) -> list[str]:
    present = [str(s) for s in strata]
    ordered = [s for s in FIG3_STRATUM_ORDER if s in present]
    ordered += [s for s in present if s not in ordered]
    return ordered


def _fig3_style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color=NATURE["neutral"], linewidth=0.6, zorder=0)
    ax.tick_params(direction="out", labelsize=7, length=3, width=0.6)


def _fig3_style_twin(ax2: plt.Axes, *, color: str) -> None:
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(color)
    ax2.tick_params(
        axis="y",
        direction="out",
        labelsize=7,
        length=3,
        width=0.6,
        colors=color,
        labelcolor=color,
    )
    ax2.yaxis.label.set_color(color)


def _fig3_collect_handles(ax: plt.Axes, *extra_axes: plt.Axes | None) -> tuple[list, list]:
    lines, labels = ax.get_legend_handles_labels()
    for extra in extra_axes:
        if extra is None:
            continue
        l2, lab2 = extra.get_legend_handles_labels()
        lines += l2
        labels += lab2
    seen: set[str] = set()
    uniq_lines: list = []
    uniq_labels: list[str] = []
    for ln, lb in zip(lines, labels):
        if lb in seen:
            continue
        seen.add(lb)
        uniq_lines.append(ln)
        uniq_labels.append(lb)
    return uniq_lines, uniq_labels


def _fig3_place_row_legend(
    fig: plt.Figure,
    gs: matplotlib.gridspec.GridSpec,
    row: int,
    legend_col: int,
    handles: list,
    labels: list[str],
) -> None:
    if not handles:
        return
    leg_ax = fig.add_subplot(gs[row, legend_col])
    leg_ax.axis("off")
    leg_ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        fontsize=6.5,
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.0,
    )


def _fig3_place_subplot_legend(
    ax: plt.Axes | Sequence[plt.Axes],
    *extra_axes: plt.Axes | None,
    ncol: int | None = None,
    fontsize: float = 7.5,
) -> None:
    """Place one legend centred under a row, spanning beyond the middle panel."""
    axes = [ax] if isinstance(ax, plt.Axes) else list(ax)
    if not axes:
        return
    mid = axes[len(axes) // 2]
    handles, labels = _fig3_collect_handles(mid, *extra_axes)
    if not handles:
        return
    n = len(handles)
    if ncol is None:
        ncol = n
    mid.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=ncol,
        fontsize=fontsize,
        frameon=False,
        handlelength=1.8,
        columnspacing=2.6,
        handletextpad=0.55,
        borderaxespad=0.0,
    )
    legend = mid.get_legend()
    if legend is not None:
        legend.set_in_layout(False)
        legend.set_clip_on(False)


def _fig3_set_integer_xticks(ax: plt.Axes, values=None) -> None:
    """Force integer major ticks; optional ``values`` set the inclusive range."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8, min_n_ticks=2))
    if values is None:
        return
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    lo = int(np.floor(vals.min()))
    hi = int(np.ceil(vals.max()))
    if hi < lo:
        return
    ax.set_xticks(list(range(lo, hi + 1)))


def _fig3_set_horizon_axis(ax: plt.Axes, *, show_label: bool) -> None:
    ax.set_xlim(0.8, 5.2)
    _fig3_set_integer_xticks(ax, FIG3_HORIZONS)
    if show_label:
        ax.set_xlabel("Horizon", fontsize=7.5)
# Open-loop thresholded metrics (prob >= 0.5) after AUPRC panels in Fig 3/4.
OPENLOOP_CLASSIFICATION_METRICS: tuple[tuple[str, str], ...] = (
    ("balanced_accuracy", "Balanced accuracy"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
)
# Fig 3/4: Neyman–Pearson open-loop row after Specificity panels.
OPENLOOP_EXTRA_CLASSIFICATION_METRICS: tuple[tuple[str, str], ...] = (
    ("sensitivity_at_spec80", "Sen@Spec≥0.8"),
)
OPENLOOP_REL_BASELINE_TEX = r"\mathrm{MLP\,h=1}"
OPENLOOP_ABS_BASELINE_TEX = r"\mathrm{Logistic\,h=1}"
_SHORT_WORSENING_XTICK = {
    "adl_worsening": "ADL",
    "iadl_worsening": "IADL",
    "death_event": "Death",
}
# Action simulation: low exposure → high outcome risk (red); high → low risk (green).
ACTION_COLORS = {
    "low": ORANGE,
    "baseline": GREY,
    "high": GREEN,
    "switch_low": ORANGE,
    "switch_high": GREEN,
}
# Fig 2 PCA: two-class coloring for Bernoulli 0/1 labels (0 safe, 1 event).
FIG2_BINARY_CMAP = ListedColormap([NATURE["light_grey"], ORANGE])
# Panel letters for the single-column Fig 2 layout (Death / ADL / IADL).
FIG2_PCA_PANEL_LABELS = ("A", "B", "C")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Health world model evaluation. "
            "Redraw manuscript Figures 2–5 with plot_manuscript_figures.py. "
            "Full cohort tables are not included in this release."
        )
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--train-data",
        default=str(DEFAULT_DATA_DIR / "HRS_train.parquet"),
    )
    parser.add_argument(
        "--test-data",
        default=str(DEFAULT_DATA_DIR / "HRS_test.parquet"),
    )
    parser.add_argument(
        "--val-data",
        default=str(DEFAULT_DATA_DIR / "HRS_validation.parquet"),
    )
    parser.add_argument(
        "--preprocessing",
        default=str(DEFAULT_DATA_DIR / "HRS_preprocessing.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <run-dir>/evaluation/",
    )
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--horizons",
        default="1,2,3,5",
        help="Comma-separated latent / rollout horizons.",
    )
    parser.add_argument("--max-persons-traj", type=int, default=4)
    parser.add_argument(
        "--skip-mlp",
        action="store_true",
        help="Skip the tabular MLP baseline (Fig 3A / Fig 4).",
    )
    parser.add_argument(
        "--skip-fig2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip Fig 2 (default True). Pass --no-skip-fig2 to enable Fig 2.",
    )
    parser.add_argument(
        "--refit-fig4-validation-cache",
        action="store_true",
        help="Force recompute Fig 4 open-loop validation cache.",
    )
    parser.add_argument(
        "--only-fig4",
        action="store_true",
        help=(
            "Only collect/plot Fig 4 (open-loop validation for --fig4-action). "
            "Skips baselines, Fig 1/3/5/2/7."
        ),
    )
    parser.add_argument(
        "--fig4-action",
        default=FIG4_DEFAULT_ACTION,
        choices=list(FIG4_ACTION_ORDER),
        help=(
            "Primary intervention: fig4.png and fig4_{action}.png. "
            f"Default: {FIG4_DEFAULT_ACTION}."
        ),
    )
    parser.add_argument(
        "--fig4-all-actions",
        action="store_true",
        help="Also collect/plot Fig 4 for every intervention (not only --fig4-action).",
    )
    parser.add_argument(
        "--fig4-validation-cache-dir",
        default=None,
        help="Default: <output-dir>/fig4_validation_cache/",
    )
    parser.add_argument(
        "--only-fig5",
        action="store_true",
        help=(
            "Only collect/plot Fig 5 (death KM for every Fig 4 intervention). "
            "Skips baselines, Fig 1/3/4/2/7."
        ),
    )
    parser.add_argument(
        "--only-figS3",
        action="store_true",
        help=(
            "Only collect/plot supplementary Fig S3 (MLP-h* open-loop dynamics "
            "plus Fig 3 risk/KM panels). Skips Fig 1/2/4/5/7."
        ),
    )
    parser.add_argument(
        "--figS3-from-cache",
        action="store_true",
        help=(
            "With --only-figS3: replot from source_data_report/ without JEPA "
            "rollout collection. Recomputes MLP-h* metrics when rollout CSVs "
            "exist but figS3 dynamics CSVs are missing, or when "
            "--refit-figS3-hstar is set."
        ),
    )
    parser.add_argument(
        "--refit-figS3-hstar",
        action="store_true",
        help=(
            "Force refit MLP-h* baselines when using --figS3-from-cache "
            "(or during --only-figS3 full collection via --refit-baselines)."
        ),
    )
    parser.add_argument(
        "--only-fig6",
        action="store_true",
        help=(
            "Only collect/plot Fig 6 case study (one HRS test person, open-loop "
            "risk trajectories under intervention paths). Skips Fig 1/2/3/4/5/7."
        ),
    )
    parser.add_argument(
        "--fig6-from-cache",
        action="store_true",
        help="With --only-fig6: replot from fig6_case_study_*.csv without JEPA forward pass.",
    )
    parser.add_argument(
        "--fig6-person-id",
        default=None,
        help="Override auto-selected Fig 6 case-study person_id from HRS test.",
    )
    parser.add_argument(
        "--fig6-exclude-person-id",
        default=None,
        help="With auto Fig 6 selection: skip this person_id (e.g. prior case study).",
    )
    parser.add_argument(
        "--skip-fig5",
        action="store_true",
        help="Skip Fig 5 death KM in the full report.",
    )
    parser.add_argument(
        "--refit-fig5-cache",
        action="store_true",
        help="Force recompute Fig 5 death KM cache.",
    )
    parser.add_argument(
        "--fig5-km-cache-dir",
        default=None,
        help="Default: <output-dir>/fig5_km_cache/",
    )
    parser.add_argument(
        "--skip-fig7",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip Fig 7 (default True). Pass --no-skip-fig7 to plot trajectories.",
    )
    parser.add_argument(
        "--history-only",
        action="store_true",
        help="Only plot Fig 1C from training_history.csv (no checkpoint).",
    )
    parser.add_argument(
        "--refit-baselines",
        action="store_true",
        help="Force re-fit linear / tabular MLP baselines.",
    )
    parser.add_argument(
        "--binary-pos-weight-max",
        type=float,
        default=None,
        help=(
            "Cap for Logistic/MLP positive-class weight ≈ n_neg/n_pos "
            "(same as JEPA world_loss.binary_reward_pos_weight_max). "
            "Default: checkpoint config, else "
            f"{DEFAULT_BINARY_POS_WEIGHT_MAX}. <=0 disables."
        ),
    )
    parser.add_argument(
        "--refit-fig2-cache",
        action="store_true",
        help="Force recompute Fig 2 latent/probe cache under evaluation/fig2_cache/.",
    )
    parser.add_argument(
        "--fig2-cache-dir",
        default=None,
        help="Default: <output-dir>/fig2_cache/",
    )
    parser.add_argument(
        "--skip-linear-baseline",
        action="store_true",
        help="Use prevalence / mean instead of SGD logistic.",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Fit/cache linear + tabular MLP baselines then exit.",
    )
    parser.add_argument(
        "--external",
        default=None,
        choices=sorted(COHORTS),
        help=(
            "Out-of-country cohort. Replaces --test-data with that parquet "
            "(unless --test-data / --external-data is set) and writes to "
            "<run-dir>/evaluation_<cohort>/ unless --output-dir is set. "
            "HRS train parquet is still used to fit Logistic/MLP."
        ),
    )
    parser.add_argument(
        "--external-data",
        default=None,
        help="Override parquet path for --external (must match HRS test columns).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def apply_external_cohort(args: argparse.Namespace) -> ExternalCohort | None:
    """Point --test-data / --output-dir at an out-of-country parquet."""
    cohort_name = args.external
    override = Path(args.external_data).resolve() if args.external_data else None
    if not cohort_name:
        args.external_cohort = None
        return None
    cohort = get_cohort(cohort_name)
    parquet = override if override is not None else cohort.parquet
    if not parquet.exists():
        raise FileNotFoundError(
            f"External {cohort.name} table not found: {parquet}. "
            "Person-level data are not included in this repository. See data/README.md."
        )
    default_test = HRS_TEST_PARQUET.resolve()
    if Path(args.test_data).resolve() == default_test:
        args.test_data = str(parquet)
    if args.output_dir is None:
        args.output_dir = str((Path(args.run_dir) / cohort.output_subdir).resolve())
    args.external = cohort.name
    args.external_cohort = cohort
    return cohort


def report_md_kwargs(args: argparse.Namespace) -> dict[str, object]:
    cohort = getattr(args, "external_cohort", None)
    if cohort is None:
        return {}
    return {
        "title": f"# Health world model external validation ({cohort.label})",
        "extra_header": [
            f"- External cohort: {cohort.label}",
            f"- Test table: `{Path(args.test_data)}`",
            f"- {cohort.note}",
        ],
    }


def parse_horizons(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titlepad": 4,
            "axes.linewidth": 0.7,
            "figure.constrained_layout.h_pad": 0.12,
            "figure.constrained_layout.w_pad": 0.06,
            "figure.constrained_layout.hspace": 0.28,
            "figure.constrained_layout.wspace": 0.10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.3,
            "savefig.dpi": 200,
        }
    )


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=NATURE["neutral"], linewidth=0.6, zorder=0)
    ax.tick_params(direction="out")


def panel_label(ax: plt.Axes, label: str) -> None:
    """Bold panel letter above the top-left of the axes (no title on this axes)."""
    ax.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
        annotation_clip=False,
        zorder=10,
    )


def panel_title(ax: plt.Axes, label: str, title: str = "", **kwargs) -> None:
    """Left-aligned title with a bold panel letter in a dedicated left slot.

    A separate letter at ``y=1.08`` used to share the header band with a
    centered title, so ``C`` ran into ``Open-loop ...`` on half-width panels.
    """
    fontsize = kwargs.pop("fontsize", plt.rcParams.get("axes.titlesize", 9))
    pad = kwargs.pop("pad", 6)
    title = "" if title is None else str(title)
    txt = ax.set_title(title, loc="left", fontsize=fontsize, pad=pad, **kwargs)
    if title:
        txt.set_x(0.14 if label else 0.0)
    if label:
        ax.annotate(
            label,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0, pad),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
            annotation_clip=False,
            zorder=10,
        )


def save_fig(fig: plt.Figure, path: Path, dpi: int, *, pad_inches: float = 0.1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Wrote {path}", flush=True)
    return path


def skip_notice(msg: str) -> None:
    print(f"[skip] {msg}", flush=True)


def _subset_named(
    frame: pd.DataFrame,
    names: tuple[str, ...] | list[str],
    col: str = "event",
) -> pd.DataFrame:
    if frame.empty or col not in frame.columns:
        return frame
    return frame[frame[col].isin(list(names))].copy()


def write_csv(frame: pd.DataFrame, path: Path) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None or frame.empty:
        return None
    frame.to_csv(path, index=False)
    print(f"Wrote {path}", flush=True)
    return path


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _figS3_event_names(spec: ModelSpec) -> list[str]:
    return [
        n
        for n in list(CLINICAL_EVENT_REWARDS) + list(WORSENING_REWARDS)
        if n in spec.reward_binary
    ]


def _figS3_csv_specs(
    *,
    worsening_roll: pd.DataFrame,
    event_roll: pd.DataFrame,
    worsening_dyn_hstar: pd.DataFrame,
    event_dyn_hstar: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("figS3_worsening_openloop_dynamics.csv", worsening_dyn_hstar),
        ("figS3_clinical_openloop_dynamics.csv", event_dyn_hstar),
        ("figS3_event_rollout_rows_worsening.csv", worsening_roll),
        ("figS3_event_rollout_rows_clinical.csv", event_roll),
    ]


def _collect_openloop_risk_km(
    agent,
    test_loader,
    device: torch.device,
    spec: ModelSpec,
    *,
    event_names_all: Sequence[str],
    max_h: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    """Open-loop risk rows/summary and KM curves shared by Fig 3 and Fig S3."""
    notices: list[str] = []
    event_risk = collect_openloop_event_risk(
        agent,
        test_loader,
        device,
        event_names=event_names_all,
        max_horizon=max_h,
    )
    event_risk_summary = summarize_event_risk_rollout(event_risk)
    worsening_risk_rows = _subset_named(event_risk, list(WORSENING_REWARDS))
    acute_risk_rows = _subset_named(event_risk, list(CLINICAL_EVENT_REWARDS))
    worsening_risk_summary = (
        event_risk_summary[event_risk_summary["event"].isin(list(WORSENING_REWARDS))]
        if not event_risk_summary.empty
        else pd.DataFrame()
    )
    acute_risk_summary = (
        event_risk_summary[event_risk_summary["event"].isin(list(CLINICAL_EVENT_REWARDS))]
        if not event_risk_summary.empty
        else pd.DataFrame()
    )
    if worsening_risk_summary.empty:
        notices.append("Fig 3/S3 risk empty: open-loop worsening risk unavailable")
    if acute_risk_summary.empty:
        notices.append("Fig 3/S3 risk empty: open-loop acute-event risk unavailable")

    km_events = [n for n in KM_PANEL_ORDER if n in spec.reward_binary]
    km_parts = [
        kaplan_meier_openloop_event_horizons(event_risk, event=name, horizons=(1, 2, 3))
        for name in km_events
    ]
    km = (
        pd.concat([p for p in km_parts if not p.empty], ignore_index=True)
        if any(not p.empty for p in km_parts)
        else pd.DataFrame()
    )
    if km.empty:
        notices.append(
            "Fig 3/S3 KM empty: open-loop KM unavailable (need binary events + enough persons)"
        )
    return (
        event_risk,
        worsening_risk_rows,
        acute_risk_rows,
        worsening_risk_summary,
        acute_risk_summary,
        km,
        notices,
    )


def _collect_figS3_hstar_dynamics(
    agent,
    test_loader,
    device: torch.device,
    *,
    spec: ModelSpec,
    baseline: pd.DataFrame,
    mlp_tabular: pd.DataFrame,
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    baseline_dir: Path,
    seed: int,
    max_h: int,
    pos_max: float,
    allowed_features: Sequence[str],
    refit_baselines: bool,
    refit_figS3_hstar: bool,
    skip_mlp: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    notices: list[str] = []
    event_names_all = _figS3_event_names(spec)
    mlp_hstar = load_or_fit_mlp_horizon_star(
        train=train_table,
        test=test_table,
        baseline_dir=baseline_dir,
        seed=seed,
        event_names=event_names_all,
        max_horizon=max_h,
        refit=refit_baselines or refit_figS3_hstar,
        skip=skip_mlp,
        binary_pos_weight_max=pos_max,
        allowed_feature_names=allowed_features,
    )
    event_roll_all = collect_jepa_event_rollouts(
        agent,
        test_loader,
        device,
        max_horizon=max_h,
        event_names=event_names_all,
        baseline=baseline,
        mlp_tabular=mlp_tabular,
    )
    event_roll_all = attach_mlp_horizon_star_probs(event_roll_all, mlp_hstar)
    worsening_roll = _subset_named(event_roll_all, list(WORSENING_REWARDS))
    event_roll = _subset_named(event_roll_all, list(CLINICAL_EVENT_REWARDS))
    worsening_dyn_hstar = summarize_event_openloop_dynamics_hstar(worsening_roll)
    event_dyn_hstar = summarize_event_openloop_dynamics_hstar(event_roll)
    if worsening_dyn_hstar.empty and event_dyn_hstar.empty:
        notices.append("Fig S3 empty: MLP-h* open-loop dynamics unavailable")
    return worsening_roll, event_roll, worsening_dyn_hstar, event_dyn_hstar, notices


def _write_figS3_csvs(
    source_dir: Path,
    *,
    worsening_roll: pd.DataFrame,
    event_roll: pd.DataFrame,
    worsening_dyn_hstar: pd.DataFrame,
    event_dyn_hstar: pd.DataFrame,
    worsening_risk_summary: pd.DataFrame | None = None,
    acute_risk_summary: pd.DataFrame | None = None,
    km: pd.DataFrame | None = None,
) -> list[Path]:
    csv_paths: list[Path] = []
    specs = _figS3_csv_specs(
        worsening_roll=worsening_roll,
        event_roll=event_roll,
        worsening_dyn_hstar=worsening_dyn_hstar,
        event_dyn_hstar=event_dyn_hstar,
    )
    if worsening_risk_summary is not None:
        specs.append(("fig3_openloop_risk_summary.csv", worsening_risk_summary))
    if acute_risk_summary is not None:
        specs.append(("fig3_clinical_openloop_risk_summary.csv", acute_risk_summary))
    if km is not None:
        specs.append(("fig3_kaplan_meier.csv", km))
    for name, frame in specs:
        p = write_csv(frame, source_dir / name)
        if p:
            csv_paths.append(p)
    return csv_paths


def _plot_figS3_bundle(
    *,
    worsening_dyn_hstar: pd.DataFrame,
    event_dyn_hstar: pd.DataFrame,
    km: pd.DataFrame,
    worsening_risk_summary: pd.DataFrame,
    acute_risk_summary: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> Path | None:
    return plot_figS3_dynamic(
        worsening_dynamics=worsening_dyn_hstar,
        clinical_dynamics=event_dyn_hstar,
        km=km,
        worsening_risk_summary=worsening_risk_summary,
        clinical_risk_summary=acute_risk_summary,
        figures_dir=figures_dir,
        dpi=dpi,
    )


def _rebuild_figS3_hstar_from_rollout_csvs(
    *,
    source_dir: Path,
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    baseline_dir: Path,
    spec: ModelSpec,
    seed: int,
    max_h: int,
    pos_max: float,
    allowed_features: Sequence[str] | None,
    refit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wors_roll = read_csv_if_exists(source_dir / "figS3_event_rollout_rows_worsening.csv")
    if wors_roll.empty:
        wors_roll = read_csv_if_exists(source_dir / "fig3_event_rollout_rows.csv")
    clin_roll = read_csv_if_exists(source_dir / "figS3_event_rollout_rows_clinical.csv")
    if clin_roll.empty:
        clin_roll = read_csv_if_exists(source_dir / "fig4_event_rollout_rows.csv")
    if wors_roll.empty and clin_roll.empty:
        raise SystemExit(
            "Fig S3 cache rebuild needs rollout CSVs under source_data_report/ "
            "(figS3_event_rollout_rows_* or fig3/fig4_event_rollout_rows.csv)."
        )
    event_names = _figS3_event_names(spec)
    if not event_names:
        parts: list[str] = []
        for frame in (wors_roll, clin_roll):
            if not frame.empty and "event" in frame.columns:
                parts.extend(frame["event"].astype(str).unique().tolist())
        event_names = sorted(set(parts))
    for frame in (wors_roll, clin_roll):
        if not frame.empty and "horizon" in frame.columns:
            max_h = max(max_h, int(pd.to_numeric(frame["horizon"], errors="coerce").max()))
    mlp_hstar = load_or_fit_mlp_horizon_star(
        train=train_table,
        test=test_table,
        baseline_dir=baseline_dir,
        seed=seed,
        event_names=event_names,
        max_horizon=max_h,
        refit=refit,
        skip=False,
        binary_pos_weight_max=pos_max,
        allowed_feature_names=allowed_features,
    )
    wors_roll = attach_mlp_horizon_star_probs(wors_roll, mlp_hstar)
    clin_roll = attach_mlp_horizon_star_probs(clin_roll, mlp_hstar)
    return (
        wors_roll,
        clin_roll,
        summarize_event_openloop_dynamics_hstar(wors_roll),
        summarize_event_openloop_dynamics_hstar(clin_roll),
    )


def _load_figS3_risk_km_from_cache(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    km = read_csv_if_exists(source_dir / "fig3_kaplan_meier.csv")
    if km.empty:
        km = read_csv_if_exists(source_dir / "fig4_kaplan_meier.csv")
    wors_risk = read_csv_if_exists(source_dir / "fig3_openloop_risk_summary.csv")
    clin_risk = read_csv_if_exists(source_dir / "fig3_clinical_openloop_risk_summary.csv")
    if clin_risk.empty:
        clin_risk = read_csv_if_exists(source_dir / "fig4_openloop_risk_summary.csv")
    return wors_risk, clin_risk, km


def merge_fig4_action_csv(path: Path, frame: pd.DataFrame) -> Path | None:
    """Replace rows for actions in ``frame``; keep other actions already on disk."""
    if frame is None or frame.empty or "action" not in frame.columns:
        return write_csv(frame, path)
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except (OSError, ValueError):
            existing = pd.DataFrame()
        if not existing.empty and "action" in existing.columns:
            keep = existing[~existing["action"].astype(str).isin(set(frame["action"].astype(str)))]
            frame = pd.concat([keep, frame], ignore_index=True)
    return write_csv(frame, path)


# ---------------------------------------------------------------------------
# Figure plotting
# ---------------------------------------------------------------------------


def plot_fig1(
    *,
    latent_summary: pd.DataFrame,
    history: pd.DataFrame,
    variance: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> Path | None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # 1A/1B are JEPA self-diagnostics. The reference is latent persistence
    # (predict z_{t+h} = z_t): same embedding space, same EMA target, so the
    # two are comparable. A separately-trained model's latent error is not.
    jepa_lat = (
        latent_summary[latent_summary["model"] == "JEPA"]
        if not latent_summary.empty
        else latent_summary
    )
    has_pers = not jepa_lat.empty and "persistence_mse" in jepa_lat.columns

    # 1A: one-step predictor vs latent persistence
    panel_label(ax_a, "A")
    one = jepa_lat[jepa_lat["horizon"] == 1] if not jepa_lat.empty else jepa_lat
    if one.empty or not has_pers:
        ax_a.text(0.5, 0.5, "No latent metrics", ha="center", va="center", transform=ax_a.transAxes)
    else:
        row = one.iloc[0]
        labels = ["Predictor", "Persistence\n(z fixed)"]
        cosines = [float(row["cosine_sim"]), float(row["persistence_cosine"])]
        mses = [float(row["latent_mse"]), float(row["persistence_mse"])]
        x = np.arange(len(labels))
        w = 0.35
        ax_a.bar(x - w / 2, cosines, width=w, color=BLUE)
        ax2 = ax_a.twinx()
        ax2.bar(x + w / 2, mses, width=w, color=ORANGE)
        # Headroom on both axes so the legend does not sit on the bars.
        ax_a.set_ylim(0, 1.32)
        ax2.set_ylim(0, max(mses) * 1.32)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(labels, fontsize=8)
        ax_a.set_ylabel("Cosine similarity")
        ax2.set_ylabel("Latent MSE")
        ax_a.set_title("JEPA latent prediction (h=1)", fontsize=9)
        lines = [
            plt.Line2D([0], [0], color=BLUE, lw=6),
            plt.Line2D([0], [0], color=ORANGE, lw=6),
        ]
        ax_a.legend(
            lines,
            ["Cosine ↑", "MSE ↓"],
            loc="upper center",
            ncol=2,
            fontsize=7,
            frameon=False,
        )
    clean_axis(ax_a)

    # 1B: multi-horizon predictor vs latent persistence, with sample sizes
    panel_label(ax_b, "B")
    if jepa_lat.empty or not has_pers:
        ax_b.text(0.5, 0.5, "No multi-horizon data", ha="center", va="center", transform=ax_b.transAxes)
    else:
        sub = jepa_lat.sort_values("horizon")
        ax_b.plot(
            sub["horizon"], sub["latent_mse"], marker="o", color=BLUE, label="Predictor"
        )
        ax_b.plot(
            sub["horizon"],
            sub["persistence_mse"],
            marker="x",
            linestyle=":",
            color=GREY,
            label="Persistence (z fixed)",
        )
        # Horizons draw on shrinking, healthier-survivor subsets; show n.
        top = float(max(sub["latent_mse"].max(), sub["persistence_mse"].max()))
        for _, r in sub.iterrows():
            ax_b.annotate(
                f"n={int(r['n'])}",
                (r["horizon"], top * 1.19),
                ha="center",
                va="center",
                fontsize=6,
                color=GREY,
            )
        ax_b.set_ylim(0, top * 1.42)
        ax_b.set_xlabel("Horizon (waves)")
        ax_b.set_ylabel("Latent MSE")
        ax_b.set_title("Open-loop latent error (JEPA space)", fontsize=9)
        ax_b.legend(fontsize=7)
    clean_axis(ax_b)

    # 1C: train / val / test world_loss only
    panel_label(ax_c, "C")
    hist_cols = ("train_world_loss", "val_world_loss")
    test_world_loss = history.attrs.get("test_world_loss", float("nan"))
    has_test = np.isfinite(float(test_world_loss))
    if history.empty or (
        not has_test
        and not any(
            c in history.columns and history[c].notna().any() for c in hist_cols
        )
    ):
        ax_c.text(0.5, 0.5, "No training_history.csv", ha="center", va="center", transform=ax_c.transAxes)
    else:
        series = [
            ("train_world_loss", BLUE, "-", None, "Train (HRS train)"),
            ("val_world_loss", ORANGE, "--", "o", "Validation (HRS val)"),
        ]
        for col, color, ls, marker, label in series:
            if col not in history.columns or not history[col].notna().any():
                continue
            sub = history.dropna(subset=["epoch", col]).sort_values("epoch")
            plot_kwargs = {
                "color": color,
                "linestyle": ls,
                "linewidth": 1.4,
                "label": label,
            }
            if marker:
                plot_kwargs["marker"] = marker
                plot_kwargs["markersize"] = 3.5
            ax_c.plot(sub["epoch"], sub[col], **plot_kwargs)
        if has_test:
            ax_c.axhline(
                float(test_world_loss),
                color=GREEN,
                linestyle=":",
                linewidth=1.4,
                label="Test (HRS test, best_final)",
            )
        ax_c.set_xlabel("Training epoch")
        ax_c.set_ylabel("World loss")
        ax_c.set_title("World-model loss by data split", fontsize=9)
        ax_c.legend(fontsize=7, ncol=1, loc="upper right")
    clean_axis(ax_c)

    # 1D: latent variance
    panel_label(ax_d, "D")
    if variance.empty:
        ax_d.text(0.5, 0.5, "No latent variance", ha="center", va="center", transform=ax_d.transAxes)
    else:
        ax_d.bar(variance["dim"], variance["std"], color=GREEN, width=1.0)
        ax_d.set_xlabel("Latent dimension")
        ax_d.set_ylabel("Std (test)")
        ax_d.set_title("Latent variance per dim")
    clean_axis(ax_d)

    return save_fig(fig, figures_dir / "fig1_representation.png", dpi)


def plot_fig2(
    *,
    probe: pd.DataFrame,
    binary_probe: pd.DataFrame,
    latent_2d: pd.DataFrame,
    method_2d: str,
    change: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    latent_pca_rewards: pd.DataFrame | None = None,
) -> Path | None:
    """Fig 2: one column of shared-PCA scatters colored by Death / ADL / IADL."""
    _ = probe, binary_probe, latent_2d, method_2d, change
    pca_rewards = latent_pca_rewards if latent_pca_rewards is not None else pd.DataFrame()
    present = (
        set(pca_rewards["outcome"].tolist())
        if (not pca_rewards.empty and "outcome" in pca_rewards.columns)
        else set()
    )
    outcomes = [name for name in FIG2_PCA_REWARDS if name in present]
    n_panels = max(len(outcomes), 1)
    fig_h = 2.8 * n_panels + 0.4
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(5.2, fig_h),
        constrained_layout=True,
        squeeze=False,
    )

    def _pca_colorbar(sc, ax: plt.Axes) -> None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(["0", "1"])
        cbar.ax.tick_params(labelsize=6)

    if not outcomes:
        ax = axes[0, 0]
        panel_label(ax, "A")
        ax.text(
            0.5,
            0.5,
            "No PCA-by-reward\n(death / ADL / IADL labels missing)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        clean_axis(ax, grid=None)
        return save_fig(fig, figures_dir / "fig2_health_latent.png", dpi)

    for i, outcome in enumerate(outcomes):
        ax = axes[i, 0]
        panel = (
            FIG2_PCA_PANEL_LABELS[i]
            if i < len(FIG2_PCA_PANEL_LABELS)
            else chr(ord("A") + i)
        )
        panel_label(ax, panel)
        sub = pca_rewards[pca_rewards["outcome"] == outcome]
        if sub.empty:
            ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
        else:
            sc = ax.scatter(
                sub["x"],
                sub["y"],
                c=sub["color"],
                s=6,
                alpha=0.45,
                cmap=FIG2_BINARY_CMAP,
                vmin=0.0,
                vmax=1.0,
                linewidths=0,
            )
            label = str(sub["label"].iloc[0]) if "label" in sub.columns else outcome
            ax.set_title(label.replace("Next-wave ", "")[:28], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            _pca_colorbar(sc, ax)
        clean_axis(ax, grid=None)

    return save_fig(fig, figures_dir / "fig2_health_latent.png", dpi)


def _plot_worsening_one_step_bars(
    ax: plt.Axes,
    clinical: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    panel: str,
    event_order: Sequence[str] | None = None,
    ylim: tuple[float, float] | None = None,
    ref_line: float | None = None,
) -> None:
    """Grouped bar chart for one-step clinical metrics (AUROC / balanced accuracy, etc.)."""
    if clinical.empty or metric not in clinical.columns:
        ax.text(
            0.5,
            0.5,
            f"No worsening-event {metric}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        panel_title(ax, panel, title)
        clean_axis(ax)
        return
    preferred = list(event_order) if event_order else list(PRIMARY_LEVELS)
    events = [e for e in preferred if e in set(clinical["event"])]
    events += [e for e in clinical["event"].unique() if e not in events]
    models = ["Prevalence", "Logistic", "MLP", "JEPA"]
    models = [m for m in models if m in set(clinical["model"])]
    x = np.arange(len(events))
    n_m = len(models)
    w = 0.8 / max(n_m, 1)
    colors = {
        "Prevalence": NATURE["light_grey"],
        "Logistic": GREY,
        "MLP": ORANGE,
        "JEPA": BLUE,
    }
    for i, model in enumerate(models):
        sub = clinical[clinical["model"] == model].set_index("event")
        vals = [sub.loc[e, metric] if e in sub.index else np.nan for e in events]
        ax.bar(
            x + (i - (n_m - 1) / 2) * w,
            vals,
            width=w,
            label=_legend_label(model),
            color=colors.get(model, GREY),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_SHORT_WORSENING_XTICK.get(e, EVENT_LABELS.get(e, e)[:12]) for e in events],
        rotation=0,
        ha="center",
    )
    ax.set_ylabel(ylabel)
    panel_title(ax, panel, title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ref_line is not None:
        ax.axhline(ref_line, color=GREY, linestyle=":", linewidth=1, alpha=0.8)
    ax.legend(fontsize=7, ncol=2)
    clean_axis(ax)


def _plot_level_mse_bars(
    ax: plt.Axes,
    level_metrics: pd.DataFrame,
    *,
    panel: str,
    title: str,
) -> None:
    """One-step MSE for continuous levels (Persistence / Linear / MLP / JEPA)."""
    if level_metrics.empty or "mse" not in level_metrics.columns:
        ax.text(
            0.5,
            0.5,
            "No continuous-level MSE",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        panel_title(ax, panel, title)
        clean_axis(ax)
        return
    name_col = "outcome" if "outcome" in level_metrics.columns else "event"
    outcomes = list(dict.fromkeys(level_metrics[name_col].astype(str).tolist()))
    models = ["Persistence", "Linear", "MLP", "JEPA"]
    models = [m for m in models if m in set(level_metrics["model"])]
    x = np.arange(len(outcomes))
    n_m = len(models)
    w = 0.8 / max(n_m, 1)
    colors = {
        "Persistence": NATURE["light_grey"],
        "Linear": GREY,
        "MLP": ORANGE,
        "JEPA": BLUE,
    }
    for i, model in enumerate(models):
        sub = level_metrics[level_metrics["model"] == model].set_index(name_col)
        vals = [sub.loc[o, "mse"] if o in sub.index else np.nan for o in outcomes]
        ax.bar(
            x + (i - (n_m - 1) / 2) * w,
            vals,
            width=w,
            label=_legend_label(model),
            color=colors.get(model, GREY),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [EVENT_LABELS.get(o, o)[:22] for o in outcomes],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("MSE")
    panel_title(ax, panel, title)
    ax.legend(fontsize=7, ncol=2)
    clean_axis(ax)


def _plot_openloop_risk_panel(
    ax,
    summary: pd.DataFrame,
    *,
    event: str,
    panel: str,
    outcome_label: str | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_column_title: bool = True,
) -> None:
    """Fig 3 open-loop mean P(event) by risk tertile + observed stratum."""
    label = outcome_label or EVENT_LABELS.get(event, event)
    if panel:
        panel_label(ax, panel)
    sub = (
        summary[summary["event"] == event]
        if not summary.empty and "event" in summary.columns
        else pd.DataFrame()
    )
    if sub.empty and not summary.empty and "event" not in summary.columns:
        sub = summary
    risk = (
        sub[sub["stratum_kind"] == "risk_tertile"]
        if not sub.empty and "stratum_kind" in sub.columns
        else pd.DataFrame()
    )
    obs = (
        sub[sub["stratum_kind"] == "obs_stratum"]
        if not sub.empty and "stratum_kind" in sub.columns
        else pd.DataFrame()
    )
    prob_col = (
        "event_prob"
        if not sub.empty and "event_prob" in sub.columns
        else ("death_prob" if not sub.empty and "death_prob" in sub.columns else None)
    )
    if (risk.empty and obs.empty) or prob_col is None:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=7,
        )
        if show_column_title:
            ax.set_title(label, fontsize=8.5, pad=6)
        _fig3_style_axis(ax)
        return
    if not risk.empty:
        order = ["low", "mid", "high", "all"]
        present = list(dict.fromkeys(risk["stratum"].tolist()))
        ordered = [s for s in order if s in present] + [
            s for s in present if s not in order
        ]
        for i, stratum in enumerate(ordered):
            g = risk[risk["stratum"] == stratum].sort_values("horizon")
            ax.plot(
                g["horizon"],
                g[prob_col],
                marker="o",
                color=RISK_COLORS.get(str(stratum), nature_color(i)),
                label=RISK_STRATUM_LABELS.get(str(stratum), str(stratum)),
                markersize=3.5,
                linewidth=1.2,
                zorder=3,
            )
    if not obs.empty:
        for stratum, group in obs.groupby("stratum", sort=False):
            g = group.sort_values("horizon")
            style = FIG3_OBS_STYLE.get(str(stratum), {"color": NATURE["grey"], "linestyle": "--", "marker": "x"})
            ax.plot(
                g["horizon"],
                g[prob_col],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                label=FIG3_OBS_LABELS.get(str(stratum), str(stratum)),
                markersize=4.0,
                linewidth=1.1,
                zorder=2,
            )
    _fig3_set_horizon_axis(ax, show_label=show_xlabel)
    if show_ylabel:
        ax.set_ylabel("Mean probability", fontsize=7.5)
    if show_column_title:
        ax.set_title(label, fontsize=8.5, pad=6)
    _fig3_style_axis(ax)


def _openloop_risk_event_order(
    summary: pd.DataFrame,
    preferred: Sequence[str],
) -> list[str]:
    if summary.empty or "event" not in summary.columns:
        return []
    present = set(summary["event"].tolist())
    ordered = [e for e in preferred if e in present]
    ordered += [e for e in summary["event"].tolist() if e not in ordered]
    # unique preserve order
    return list(dict.fromkeys(ordered))


def _filter_level_openloop(
    summary: pd.DataFrame | None,
    outcome: str,
) -> pd.DataFrame:
    use = summary if summary is not None else pd.DataFrame()
    if use.empty:
        return use
    if "outcome" in use.columns and outcome in set(use["outcome"].astype(str)):
        use = use[use["outcome"].astype(str) == outcome]
    return use


def _plot_level_openloop_relative_mse(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    panel: str,
    title: str,
    outcome: str = "adl_worsening",
) -> None:
    """Open-loop MSE_JEPA / MSE_Persistence vs horizon ( <1 beats copy-last)."""
    use = _filter_level_openloop(summary, outcome)
    if use.empty or "relative_mse" not in use.columns:
        ax.text(
            0.5,
            0.5,
            "No open-loop level MSE",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        panel_title(ax, panel, title)
        clean_axis(ax)
        return
    outcomes = list(dict.fromkeys(use["outcome"].astype(str).tolist()))
    plotted = False
    for i, name in enumerate(outcomes):
        sub = use[use["outcome"].astype(str) == name].sort_values("horizon")
        if sub.empty:
            continue
        plotted = True
        label = str(sub["label"].iloc[0]) if "label" in sub.columns else name
        ax.plot(
            sub["horizon"],
            sub["relative_mse"],
            marker="o",
            color=nature_color(i),
            label=label,
        )
        if "n" in sub.columns:
            ymax = float(np.nanmax(sub["relative_mse"].to_numpy(float)))
            for _, row in sub.iterrows():
                ax.annotate(
                    f"n={int(row['n'])}",
                    (row["horizon"], ymax * 1.08 if np.isfinite(ymax) else row["relative_mse"]),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=GREY,
                )
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No open-loop level MSE",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        panel_title(ax, panel, title)
        clean_axis(ax)
        return
    ax.axhline(1.0, color=GREY, linestyle="--", linewidth=1)
    ax.set_xlabel("Horizon (waves)")
    ax.set_ylabel(r"MSE$_{\mathrm{JEPA}}$ / MSE$_{\mathrm{Persistence}}$")
    panel_title(ax, panel, title)
    finite = pd.to_numeric(use["relative_mse"], errors="coerce")
    hi = float(np.nanmax(finite.to_numpy(float))) if finite.notna().any() else 1.2
    ax.set_ylim(0.0, max(hi * 1.25, 1.15))
    if len(outcomes) > 1:
        ax.legend(fontsize=7)
    clean_axis(ax)


def _plot_level_openloop_absolute_mse(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    panel: str,
    title: str,
    outcome: str = "adl_worsening",
) -> None:
    """Open-loop MSE vs horizon: solid=JEPA, dotted=Persistence (same form as 3F/3H)."""
    use = _filter_level_openloop(summary, outcome)
    need = {"horizon", "jepa_mse", "persistence_mse"}
    if use.empty or not need.issubset(use.columns):
        ax.text(
            0.5,
            0.5,
            "No open-loop level MSE",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        panel_title(ax, panel, title)
        clean_axis(ax)
        return
    outcomes = list(dict.fromkeys(use["outcome"].astype(str).tolist()))
    plotted = False
    hi = 0.0
    for i, name in enumerate(outcomes):
        sub = use[use["outcome"].astype(str) == name].sort_values("horizon")
        if sub.empty:
            continue
        plotted = True
        color = nature_color(i)
        label = str(sub["label"].iloc[0]) if "label" in sub.columns else name
        ax.plot(
            sub["horizon"],
            sub["jepa_mse"],
            marker="o",
            color=color,
            label=f"{WM_LEGEND} {label}",
        )
        ax.plot(
            sub["horizon"],
            sub["persistence_mse"],
            marker="x",
            color=color,
            linestyle=":",
            alpha=0.85,
        )
        hi = max(
            hi,
            float(np.nanmax(pd.to_numeric(sub["jepa_mse"], errors="coerce").to_numpy(float))),
            float(np.nanmax(pd.to_numeric(sub["persistence_mse"], errors="coerce").to_numpy(float))),
        )
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No open-loop level MSE",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        panel_title(ax, panel, title)
        clean_axis(ax)
        return
    ax.set_xlabel("Horizon (waves)")
    ax.set_ylabel("MSE")
    panel_title(ax, panel, title)
    ax.set_ylim(0.0, hi * 1.15 if hi > 0 else 1.0)
    ax.legend(fontsize=6, ncol=1)
    clean_axis(ax)


def _combined_dynamics_event_order(dyn: pd.DataFrame) -> list[str]:
    """Worsening events first, then acute clinical events."""
    if dyn.empty or "event" not in dyn.columns:
        return []
    present = set(dyn["event"].astype(str))
    ordered = [e for e in WORSENING_REWARDS if e in present]
    ordered += [e for e in CLINICAL_EVENT_REWARDS if e in present and e not in ordered]
    ordered += [e for e in dyn["event"].astype(str).tolist() if e not in ordered]
    return list(dict.fromkeys(ordered))


def _openloop_metric_specs() -> list[tuple[str, str, tuple[float, float]]]:
    """(metric_key, label, absolute_ylim) for Fig 3 open-loop panels."""
    specs: list[tuple[str, str, tuple[float, float]]] = [
        ("auroc", "AUROC", (0.45, 1.02)),
    ]
    skip = {"balanced_accuracy", "specificity"}
    for key, label in OPENLOOP_CLASSIFICATION_METRICS + OPENLOOP_EXTRA_CLASSIFICATION_METRICS:
        if key in skip:
            continue
        specs.append((key, label, (0.0, 1.05)))
    return specs


def _plot_openloop_outcome_combined(
    ax: plt.Axes,
    dyn: pd.DataFrame,
    event: str,
    *,
    metric_key: str,
    metric_label: str,
    panel: str,
    abs_ylim: tuple[float, float],
    outcome_label: str,
    show_column_title: bool,
    show_xlabel: bool,
    show_ylabel_metric: bool,
    show_ylabel_rel: bool,
    baseline_col_prefix: str = "mlp_h1",
    relative_col_prefix: str = "relative",
    series: dict[str, dict[str, object]] | None = None,
    relative_ylabel: str = "Relative / MLP h=1",
) -> plt.Axes | None:
    """Fig 3 metric cell: absolute WM/MLP (left) + relative (right)."""
    if panel:
        panel_label(ax, panel)
    if dyn.empty or "event" not in dyn.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        _fig3_style_axis(ax)
        return None
    sub = dyn[dyn["event"].astype(str) == event].sort_values("horizon")
    rel_col = f"{relative_col_prefix}_{metric_key}"
    jepa_col = f"jepa_{metric_key}"
    mlp_col = f"{baseline_col_prefix}_{metric_key}"
    if sub.empty or rel_col not in sub.columns or jepa_col not in sub.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        _fig3_style_axis(ax)
        return None

    plot_series = series or FIG3_SERIES
    rel_s = plot_series["relative"]
    wm_s = plot_series["wm"]
    mlp_s = plot_series["mlp"]
    ax2 = ax.twinx()
    ax.plot(
        sub["horizon"],
        sub[jepa_col],
        color=wm_s["color"],
        marker=wm_s["marker"],
        linestyle=wm_s["linestyle"],
        linewidth=wm_s["linewidth"],
        markersize=wm_s["markersize"],
        label=wm_s["label"],
        zorder=3,
    )
    if mlp_col in sub.columns:
        ax.plot(
            sub["horizon"],
            sub[mlp_col],
            color=mlp_s["color"],
            marker=mlp_s["marker"],
            linestyle=mlp_s["linestyle"],
            linewidth=mlp_s["linewidth"],
            markersize=mlp_s["markersize"],
            label=mlp_s["label"],
            zorder=2,
        )
    ax2.plot(
        sub["horizon"],
        sub[rel_col],
        color=rel_s["color"],
        marker=rel_s["marker"],
        linestyle=rel_s["linestyle"],
        linewidth=rel_s["linewidth"],
        markersize=rel_s["markersize"],
        label=rel_s["label"],
        zorder=3,
    )
    ax2.axhline(1.0, color=FIG3_REL_COLOR, linestyle="--", linewidth=0.9, alpha=0.45, zorder=1)
    rel_vals = pd.to_numeric(sub[rel_col], errors="coerce")
    if rel_vals.notna().any():
        lo, hi = float(rel_vals.min()), float(rel_vals.max())
        pad = max((hi - lo) * 0.12, 0.08)
        ax2.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.set_ylim(*abs_ylim)
    _fig3_set_horizon_axis(ax, show_label=show_xlabel)
    if show_ylabel_metric:
        ax.set_ylabel(metric_label, fontsize=7.5)
    if show_ylabel_rel:
        ax2.set_ylabel(relative_ylabel, fontsize=7.5, color=FIG3_REL_COLOR)
    if show_column_title:
        ax.set_title(outcome_label, fontsize=8.5, pad=6)
    _fig3_style_axis(ax)
    _fig3_style_twin(ax2, color=FIG3_REL_COLOR)
    return ax2


def plot_fig3(
    *,
    worsening_dynamics: pd.DataFrame,
    clinical_dynamics: pd.DataFrame,
    km: pd.DataFrame,
    worsening_risk_summary: pd.DataFrame,
    clinical_risk_summary: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    level_openloop: pd.DataFrame | None = None,
    include_cognition: bool | None = None,
) -> Path | None:
    """Fig 3: merged worsening + acute clinical open-loop dynamics, KM, and risk."""
    level_openloop = level_openloop if level_openloop is not None else pd.DataFrame()
    if include_cognition is None:
        include_cognition = not level_openloop.empty

    dyn_parts = [f for f in (worsening_dynamics, clinical_dynamics) if f is not None and not f.empty]
    dyn = pd.concat(dyn_parts, ignore_index=True) if dyn_parts else pd.DataFrame()
    events = _combined_dynamics_event_order(dyn)
    n_events = max(len(events), 1)

    risk_parts = [
        f
        for f in (worsening_risk_summary, clinical_risk_summary)
        if f is not None and not f.empty
    ]
    risk_summary = pd.concat(risk_parts, ignore_index=True) if risk_parts else pd.DataFrame()
    risk_events = _openloop_risk_event_order(risk_summary, events)

    km = km if km is not None else pd.DataFrame()
    km_events: list[str] = []
    if not km.empty and "event" in km.columns:
        have = set(km["event"].astype(str))
        km_events = [e for e in KM_PANEL_ORDER if e in have]
        km_events += [e for e in have if e not in km_events]

    metric_specs = _openloop_metric_specs()
    n_metrics = len(metric_specs)
    n_km = len(km_events)
    n_risk_rows = (
        max(int(math.ceil(len(risk_events) / n_events)), 1) if risk_events else 0
    )
    cogn_row = 1 if include_cognition else 0
    total_rows = n_metrics + cogn_row + n_risk_rows + n_km
    height_ratios: list[float] = [1.0] * n_metrics
    if include_cognition:
        height_ratios.append(0.85)
    if n_risk_rows:
        height_ratios.extend([1.0] * n_risk_rows)
    if n_km:
        height_ratios.extend([1.0] * n_km)
    row_h = 2.22
    fig_h = row_h * (n_metrics + n_risk_rows + n_km) + (2.0 if include_cognition else 0) + 0.5
    fig_w = max(2.85 * n_events, 8.5)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.10, hspace=0.22, wspace=0.14)
    gs = fig.add_gridspec(
        total_rows,
        n_events,
        height_ratios=height_ratios,
    )
    panel_idx = 0
    row = 0

    for mi, (metric_key, metric_label, abs_ylim) in enumerate(metric_specs):
        row_letter = chr(ord("A") + panel_idx)
        panel_idx += 1
        legend_axes: list[plt.Axes] = []
        legend_ax2: plt.Axes | None = None
        legend_col = n_events // 2
        for ei, event in enumerate(events):
            ax = fig.add_subplot(gs[row + mi, ei])
            outcome_label = EVENT_LABELS.get(event, event)
            ax2 = _plot_openloop_outcome_combined(
                ax,
                dyn,
                event,
                metric_key=metric_key,
                metric_label=metric_label,
                panel=row_letter if ei == 0 else "",
                abs_ylim=abs_ylim,
                outcome_label=outcome_label,
                show_column_title=True,
                show_xlabel=True,
                show_ylabel_metric=(ei == 0),
                show_ylabel_rel=(ei == n_events - 1),
            )
            legend_axes.append(ax)
            if ei == legend_col:
                legend_ax2 = ax2
        _fig3_place_subplot_legend(legend_axes, legend_ax2)
    row += n_metrics

    if include_cognition:
        gs_cog = gs[row, :n_events].subgridspec(1, 2, wspace=0.18)
        ax_rel = fig.add_subplot(gs_cog[0, 0])
        ax_abs = fig.add_subplot(gs_cog[0, 1])
        letter = chr(ord("A") + panel_idx)
        panel_idx += 1
        _plot_level_openloop_relative_mse(
            ax_rel,
            level_openloop,
            panel=letter,
            title="Open-loop relative MSE (vs Persistence)",
        )
        _plot_level_openloop_absolute_mse(
            ax_abs,
            level_openloop,
            panel="",
            title="Open-loop MSE (solid=JEPA, dotted=Persistence)",
        )
        row += 1

    if risk_events:
        n_risk_grid_rows = max(int(math.ceil(len(risk_events) / n_events)), 1)
        for r_i in range(n_risk_grid_rows):
            row_letter = chr(ord("A") + panel_idx)
            panel_idx += 1
            legend_axes: list[plt.Axes] = []
            for col in range(n_events):
                j = r_i * n_events + col
                if j >= len(risk_events):
                    break
                event = risk_events[j]
                ax = fig.add_subplot(gs[row + r_i, col])
                _plot_openloop_risk_panel(
                    ax,
                    risk_summary,
                    event=event,
                    panel=row_letter if col == 0 else "",
                    outcome_label=EVENT_LABELS.get(event, event),
                    show_column_title=True,
                    show_xlabel=True,
                    show_ylabel=(col == 0),
                )
                legend_axes.append(ax)
            _fig3_place_subplot_legend(legend_axes)
        row += n_risk_grid_rows

    if km_events:
        km_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        n_km_cols = min(n_events, 3)
        for i, event in enumerate(km_events):
            km_axes = [fig.add_subplot(gs[row + i, j]) for j in range(n_km_cols)]
            letter = km_letters[panel_idx] if panel_idx < len(km_letters) else f"K{panel_idx}"
            sub = km[km["event"].astype(str).eq(event)]
            label = EVENT_LABELS.get(event, event)
            _plot_openloop_km_row(
                km_axes,
                sub,
                panel=letter,
                event_label=label,
            )
            _fig3_place_subplot_legend(km_axes)
            panel_idx += 1

    return save_fig(fig, figures_dir / "fig3_dynamics.png", dpi, pad_inches=0.25)


def plot_figS3_dynamic(
    *,
    worsening_dynamics: pd.DataFrame,
    clinical_dynamics: pd.DataFrame,
    km: pd.DataFrame | None = None,
    worsening_risk_summary: pd.DataFrame | None = None,
    clinical_risk_summary: pd.DataFrame | None = None,
    figures_dir: Path,
    dpi: int,
) -> Path | None:
    """Fig S3: MLP-h* open-loop metrics plus Fig 3 risk (D) and KM (E/F) panels."""
    dyn_parts = [f for f in (worsening_dynamics, clinical_dynamics) if f is not None and not f.empty]
    dyn = pd.concat(dyn_parts, ignore_index=True) if dyn_parts else pd.DataFrame()
    if dyn.empty:
        return None
    events = _combined_dynamics_event_order(dyn)
    n_events = max(len(events), 1)

    risk_parts = [
        f
        for f in (worsening_risk_summary, clinical_risk_summary)
        if f is not None and not f.empty
    ]
    risk_summary = pd.concat(risk_parts, ignore_index=True) if risk_parts else pd.DataFrame()
    risk_events = _openloop_risk_event_order(risk_summary, events)

    km = km if km is not None else pd.DataFrame()
    km_events: list[str] = []
    if not km.empty and "event" in km.columns:
        have = set(km["event"].astype(str))
        km_events = [e for e in KM_PANEL_ORDER if e in have]
        km_events += [e for e in have if e not in km_events]

    metric_specs = _openloop_metric_specs()
    n_metrics = len(metric_specs)
    n_km = len(km_events)
    n_risk_rows = (
        max(int(math.ceil(len(risk_events) / n_events)), 1) if risk_events else 0
    )
    total_rows = n_metrics + n_risk_rows + n_km
    row_h = 2.22
    fig_h = row_h * (n_metrics + n_risk_rows + n_km) + 0.5
    fig_w = max(2.85 * n_events, 8.5)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.10, hspace=0.22, wspace=0.14)
    height_ratios = [1.0] * n_metrics
    if n_risk_rows:
        height_ratios.extend([1.0] * n_risk_rows)
    if n_km:
        height_ratios.extend([1.0] * n_km)
    gs = fig.add_gridspec(
        total_rows,
        n_events,
        height_ratios=height_ratios,
    )
    panel_idx = 0
    row = 0

    for mi, (metric_key, metric_label, abs_ylim) in enumerate(metric_specs):
        row_letter = chr(ord("A") + panel_idx)
        panel_idx += 1
        legend_axes: list[plt.Axes] = []
        legend_ax2: plt.Axes | None = None
        legend_col = n_events // 2
        for ei, event in enumerate(events):
            ax = fig.add_subplot(gs[row + mi, ei])
            outcome_label = EVENT_LABELS.get(event, event)
            ax2 = _plot_openloop_outcome_combined(
                ax,
                dyn,
                event,
                metric_key=metric_key,
                metric_label=metric_label,
                panel=row_letter if ei == 0 else "",
                abs_ylim=abs_ylim,
                outcome_label=outcome_label,
                show_column_title=True,
                show_xlabel=True,
                show_ylabel_metric=(ei == 0),
                show_ylabel_rel=(ei == n_events - 1),
                baseline_col_prefix="mlp_hstar",
                relative_col_prefix="relative_hstar",
                series=FIGS3_SERIES,
                relative_ylabel="Relative / MLP-h*",
            )
            legend_axes.append(ax)
            if ei == legend_col:
                legend_ax2 = ax2
        _fig3_place_subplot_legend(legend_axes, legend_ax2)
    row += n_metrics

    if risk_events:
        n_risk_grid_rows = max(int(math.ceil(len(risk_events) / n_events)), 1)
        for r_i in range(n_risk_grid_rows):
            row_letter = chr(ord("A") + panel_idx)
            panel_idx += 1
            legend_axes: list[plt.Axes] = []
            for col in range(n_events):
                j = r_i * n_events + col
                if j >= len(risk_events):
                    break
                event = risk_events[j]
                ax = fig.add_subplot(gs[row + r_i, col])
                _plot_openloop_risk_panel(
                    ax,
                    risk_summary,
                    event=event,
                    panel=row_letter if col == 0 else "",
                    outcome_label=EVENT_LABELS.get(event, event),
                    show_column_title=True,
                    show_xlabel=True,
                    show_ylabel=(col == 0),
                )
                legend_axes.append(ax)
            _fig3_place_subplot_legend(legend_axes)
        row += n_risk_grid_rows

    if km_events:
        km_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        n_km_cols = min(n_events, 3)
        for i, event in enumerate(km_events):
            km_axes = [fig.add_subplot(gs[row + i, j]) for j in range(n_km_cols)]
            letter = km_letters[panel_idx] if panel_idx < len(km_letters) else f"K{panel_idx}"
            sub = km[km["event"].astype(str).eq(event)]
            label = EVENT_LABELS.get(event, event)
            _plot_openloop_km_row(
                km_axes,
                sub,
                panel=letter,
                event_label=label,
            )
            _fig3_place_subplot_legend(km_axes)
            panel_idx += 1

    return save_fig(fig, figures_dir / "figS3_dynamic.png", dpi, pad_inches=0.25)


def _action_short_label(action: str) -> str:
    return {
        "cigarettes_per_day": "Cigarettes/day",
        "alcohol_days_per_week": "Alcohol days/wk",
        "drinks_per_drinking_day": "Drinks/drinking day",
        "vigorous_activity_frequency": "Frequency of vigorous physical activity",
        "moderate_activity_frequency": "Moderate activity",
        "light_activity_frequency": "Light activity",
        "hypertension_treatment": "Antihypertensive",
        "diabetes_oral_medication": "Diabetes oral med",
    }.get(action, action)


def _prob_col(frame: pd.DataFrame) -> str:
    if not frame.empty and "event_prob" in frame.columns:
        return "event_prob"
    return "adl_pred"


def _plot_always_on_action_panel(
    ax,
    sub: pd.DataFrame,
    *,
    panel: str,
    title: str,
    ylabel: str,
) -> None:
    panel_label(ax, panel)
    ycol = _prob_col(sub)
    if sub.empty or ycol not in sub.columns:
        ax.text(
            0.5,
            0.5,
            "No always-on sim",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title(title, fontsize=9)
        clean_axis(ax)
        return
    for regime, color in [
        ("low", ACTION_COLORS["low"]),
        ("baseline", ACTION_COLORS["baseline"]),
        ("high", ACTION_COLORS["high"]),
    ]:
        g = sub[sub["regime"] == regime].sort_values("horizon")
        if g.empty:
            continue
        ax.plot(
            g["horizon"],
            g[ycol],
            marker="o",
            color=color,
            label=regime,
            markersize=4,
        )
    ax.set_xlabel("Horizon (waves)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6.5)
    clean_axis(ax)


def _plot_delayed_switch_action_panel(
    ax,
    sub: pd.DataFrame,
    *,
    panel: str,
    title: str,
    ylabel: str,
    switch_horizon: int,
) -> None:
    panel_label(ax, panel)
    ycol = _prob_col(sub)
    if sub.empty or ycol not in sub.columns:
        ax.text(
            0.5,
            0.5,
            "No delayed switch",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title(title, fontsize=9)
        clean_axis(ax)
        return
    style = {
        "baseline": (ACTION_COLORS["baseline"], "-", "stay baseline"),
        "switch_low": (ACTION_COLORS["switch_low"], "-", f"→ low at h={switch_horizon}"),
        "switch_high": (ACTION_COLORS["switch_high"], "-", f"→ high at h={switch_horizon}"),
    }
    for regime, (color, ls, label) in style.items():
        g = sub[sub["regime"] == regime].sort_values("horizon")
        if g.empty:
            continue
        ax.plot(
            g["horizon"],
            g[ycol],
            marker="o",
            color=color,
            linestyle=ls,
            label=label,
            markersize=4,
        )
    ax.axvline(
        switch_horizon - 0.5,
        color=GREY,
        linestyle=":",
        linewidth=1,
        alpha=0.85,
    )
    ax.set_xlabel("Horizon (waves)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6)
    clean_axis(ax)


def _resolve_fig5_actions(
    sim_levels: pd.DataFrame,
    delayed_switch: pd.DataFrame,
    action_order: Sequence[str] | None,
) -> list[str]:
    if action_order:
        return filter_fig5_actions(list(dict.fromkeys(action_order)))
    actions: list[str] = []
    for frame in (sim_levels, delayed_switch):
        if not frame.empty and "action" in frame.columns:
            for a in frame["action"].tolist():
                if a not in actions:
                    actions.append(a)
    preferred = list(FIG5_SIMULATION_ACTIONS)
    present = set(actions)
    ordered = [a for a in preferred if a in present] + [
        a for a in actions if a not in preferred
    ]
    return filter_fig5_actions(ordered)


def _drop_excluded_fig5_actions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "action" not in frame.columns:
        return frame
    allowed = set(FIG5_SIMULATION_ACTIONS) - set(FIG5_EXCLUDED_ACTIONS)
    return frame[frame["action"].isin(allowed)].copy()


def plot_fig5_simulation(
    *,
    sim_levels: pd.DataFrame,
    delayed_switch: pd.DataFrame | None = None,
    figures_dir: Path,
    dpi: int,
    switch_horizon: int = 3,
    action_order: Sequence[str] | None = None,
) -> Path | None:
    """Fig 5: per action — always-on (left) + delayed switch (right); ADL."""
    delayed_switch = delayed_switch if delayed_switch is not None else pd.DataFrame()
    # Prefer ADL rows when multi-event frames are passed.
    if not sim_levels.empty and "event" in sim_levels.columns:
        sim_levels = sim_levels[sim_levels["event"].eq("adl_worsening")].copy()
    if not delayed_switch.empty and "event" in delayed_switch.columns:
        delayed_switch = delayed_switch[delayed_switch["event"].eq("adl_worsening")].copy()

    actions = _resolve_fig5_actions(sim_levels, delayed_switch, action_order)
    if not actions:
        actions = ["(none)"]

    n_rows = len(actions)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(8.8, 3.2 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for row, action in enumerate(actions):
        ax_on, ax_sw = axes[row, 0], axes[row, 1]
        short = _action_short_label(action)
        sub_on = (
            sim_levels[sim_levels["action"] == action]
            if not sim_levels.empty and "action" in sim_levels.columns
            else (sim_levels if row == 0 else pd.DataFrame())
        )
        sub_sw = (
            delayed_switch[delayed_switch["action"] == action]
            if not delayed_switch.empty and "action" in delayed_switch.columns
            else (delayed_switch if row == 0 else pd.DataFrame())
        )
        _plot_always_on_action_panel(
            ax_on,
            sub_on,
            panel=chr(ord("A") + 2 * row),
            title=f"Always-on · {short}",
            ylabel=r"$P$(ADL worsened)",
        )
        _plot_delayed_switch_action_panel(
            ax_sw,
            sub_sw,
            panel=chr(ord("A") + 2 * row + 1),
            title=f"Baseline cohort · switch {short} at h={switch_horizon}",
            ylabel=r"$P$(ADL worsened)",
            switch_horizon=switch_horizon,
        )

    return save_fig(fig, figures_dir / "fig5_simulation.png", dpi)


def _plot_observed_with_sim_overlay(
    ax,
    observed: pd.DataFrame,
    simulated: pd.DataFrame,
    *,
    obs_group_col: str,
    sim_regime_col: str,
    obs_regimes: Sequence[tuple[str, str, str]],
    sim_regimes: Sequence[tuple[str, str, str]],
    panel: str,
    title: str = "",
    ylabel: str,
    switch_horizon: int | None = None,
    sim_legend_suffix: str = "observed cigs",
    ylim_factor: float = 1.15,
    ylim_cap: float = 1.2,
    xlabel: str = "Horizon (waves)",
    integer_xticks: bool = False,
    include_sim_n: bool = False,
    title_fontsize: float = 9,
) -> None:
    """Dashed = observed rates; solid = simulated probabilities."""
    panel_label(ax, panel)
    if observed.empty and simulated.empty:
        ax.text(
            0.5,
            0.5,
            "No validation data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        if title:
            ax.set_title(title, fontsize=title_fontsize)
        clean_axis(ax)
        return
    for regime, color, label in obs_regimes:
        if observed.empty or obs_group_col not in observed.columns:
            continue
        g = observed[observed[obs_group_col] == regime].sort_values("horizon")
        if g.empty:
            continue
        n0 = (
            int(g["n_origin"].iloc[0])
            if "n_origin" in g.columns
            else int(g["n_labeled"].iloc[0])
        )
        ax.plot(
            g["horizon"],
            g["observed_rate"],
            marker="o",
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            label=f"obs {label} (n={n0})",
            markersize=4,
        )
    ycol = _prob_col(simulated)
    for regime, color, label in sim_regimes:
        if simulated.empty or sim_regime_col not in simulated.columns:
            continue
        g = simulated[simulated[sim_regime_col] == regime].sort_values("horizon")
        if g.empty or ycol not in g.columns:
            continue
        n_sim = None
        if include_sim_n:
            if "n_origin" in g.columns:
                n_orig = pd.to_numeric(g["n_origin"], errors="coerce").dropna()
                if not n_orig.empty and float(n_orig.iloc[0]) > 0:
                    n_sim = int(n_orig.iloc[0])
            if n_sim is None and "n_labeled" in g.columns:
                n_lab = pd.to_numeric(g["n_labeled"], errors="coerce").dropna()
                if not n_lab.empty:
                    n_sim = int(n_lab.iloc[0])
            if n_sim is None and "n" in g.columns:
                n_sim = int(pd.to_numeric(g["n"], errors="coerce").iloc[0])
        if n_sim is not None:
            sim_label = f"sim {label} (n={n_sim})"
        elif sim_legend_suffix:
            sim_label = f"sim {label} ({sim_legend_suffix})"
        else:
            sim_label = f"sim {label}"
        ax.plot(
            g["horizon"],
            g[ycol],
            marker=".",
            color=color,
            linestyle="-",
            linewidth=1.6,
            label=sim_label,
            markersize=3,
        )
    if switch_horizon is not None:
        ax.axvline(
            switch_horizon - 0.5,
            color=GREY,
            linestyle=":",
            linewidth=1,
            alpha=0.85,
        )
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    if integer_xticks:
        xs: list[float] = []
        for line in ax.get_lines():
            if line.get_linestyle() == ":":
                continue
            xd = np.asarray(line.get_xdata(), dtype=float)
            if xd.size and np.isfinite(xd).any():
                xs.extend(xd[np.isfinite(xd)].tolist())
        if xs:
            ax.set_xticks(list(range(int(np.floor(min(xs))), int(np.ceil(max(xs))) + 1)))
    y_max = 0.0
    for line in ax.get_lines():
        if line.get_linestyle() == ":" or line.get_label() in {"", "_nolegend_"}:
            continue
        yd = np.asarray(line.get_ydata(), dtype=float)
        if yd.size and np.isfinite(yd).any():
            y_max = max(y_max, float(np.nanmax(yd)))
    y_top = max(0.12, y_max * float(ylim_factor)) if y_max > 0 else 1.0
    ax.set_ylim(0.0, min(float(ylim_cap), y_top))
    legend = ax.legend(
        fontsize=6.5,
        loc="upper left",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor=NATURE["light_grey"],
        framealpha=0.95,
        borderpad=0.35,
        labelspacing=0.22,
        columnspacing=0.8,
        handlelength=1.4,
        handletextpad=0.35,
    )
    legend.set_in_layout(False)
    clean_axis(ax)


def _collapse_obs_validation_summary(obs: pd.DataFrame) -> pd.DataFrame:
    """Merge mid+high observed summary rows (backward-compatible replot)."""
    if obs.empty or "stratum" not in obs.columns:
        return obs
    strata = set(obs["stratum"].astype(str))
    if "mid_high" in strata or not strata.intersection({"mid", "high"}):
        return obs
    meta_cols = [
        c
        for c in obs.columns
        if c
        not in (
            "stratum",
            "observed_rate",
            "crude_rate",
            "n_labeled",
            "n",
            "n_origin",
        )
    ]
    parts: list[pd.DataFrame] = []
    for key_vals, sub in obs.groupby(meta_cols, dropna=False, sort=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        base = {col: val for col, val in zip(meta_cols, key_vals)}
        low = sub[sub["stratum"].astype(str) == "low"]
        if not low.empty:
            parts.append(low)
        mh = sub[sub["stratum"].astype(str).isin(["mid", "high"])]
        if mh.empty:
            continue
        w = mh["n_labeled"].to_numpy(dtype=float)
        wsum = float(w.sum())
        if wsum <= 0:
            continue
        parts.append(
            pd.DataFrame(
                [
                    {
                        **base,
                        "stratum": "mid_high",
                        "observed_rate": float(
                            np.average(mh["observed_rate"], weights=w)
                        ),
                        "crude_rate": float(np.average(mh["crude_rate"], weights=w)),
                        "n_labeled": int(wsum),
                        "n": int(wsum),
                        "n_origin": int(
                            mh.drop_duplicates("stratum")["n_origin"].sum()
                        ),
                    }
                ]
            )
        )
    if not parts:
        return obs
    return pd.concat(parts, ignore_index=True)


FIG4_COL_TITLE_STATIC = "Open-loop simulation (static intervention)"
FIG4_COL_TITLE_CF = "Open-loop simulation (counterfactual switch at h2)"
FIG4_STRIP_THICKNESS_IN = 0.48


def _fig4_strip_layout_ratios(
    fig_width: float,
    fig_height: float,
    n_rows: int,
    *,
    strip_in: float = FIG4_STRIP_THICKNESS_IN,
    left_margin: float = 0.03,
    right_margin: float = 0.99,
    top_margin: float = 0.96,
    bottom_margin: float = 0.07,
) -> tuple[list[float], list[float]]:
    """Return width/height ratios so top and left strips share the same thickness."""
    usable_w = fig_width * (right_margin - left_margin)
    usable_h = fig_height * (top_margin - bottom_margin)
    w_frac = min(max(strip_in / max(usable_w, strip_in + 1e-6), 0.02), 0.12)
    h_frac = min(max(strip_in / max(usable_h, strip_in + 1e-6), 0.02), 0.12)
    w_strip = w_frac / max(1.0 - w_frac, 1e-6)
    h_strip = h_frac * n_rows / max(1.0 - h_frac, 1e-6)
    return [w_strip, 1.0], [h_strip] + [1.0] * n_rows


def _fig4_blank_strip(ax, *, facecolor: str = "#D9E3EC") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(facecolor)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _fig4_row_strip_label(ax, text: str, *, fontsize: float = 10) -> None:
    _fig4_blank_strip(ax)
    ax.text(
        0.52,
        0.5,
        text,
        rotation=90,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=NATURE["navy"],
    )


def _fig4_column_header_strip(ax, text: str) -> None:
    _fig4_blank_strip(ax, facecolor="#E8EEF4")
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=NATURE["navy"],
    )


def _fig4_add_column_header_row(
    fig,
    outer_header_spec,
    *,
    left_title: str = FIG4_COL_TITLE_STATIC,
    right_title: str = FIG4_COL_TITLE_CF,
    wspace: float = 0.28,
) -> None:
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_header_spec, wspace=wspace)
    ax_left = fig.add_subplot(inner[0, 0])
    ax_right = fig.add_subplot(inner[0, 1])
    _fig4_column_header_strip(ax_left, left_title)
    _fig4_column_header_strip(ax_right, right_title)


def plot_fig4(
    *,
    observed_strata: pd.DataFrame,
    sim_observed_action: pd.DataFrame,
    observed_h1_counterfactual: pd.DataFrame,
    sim_h1_counterfactual: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    outcomes: Sequence[str] = FIG5_VALIDATION_OUTCOMES,
    action: str = FIG4_DEFAULT_ACTION,
    cf_switch_horizon: int = FIG5_H1_CF_SWITCH_HORIZON,
    primary_action: str | None = None,
    output_name: str | None = None,
) -> Path | None:
    """Fig 4: open-loop validation (persistent + h1-low counterfactual)."""
    action_spec = get_fig4_action_spec(action)
    outcome_labels = {
        name: EVENT_LABELS.get(name, name.replace("_", " ").title())
        for name in FIG5_VALIDATION_OUTCOMES
    }
    plot_outcomes = [o for o in outcomes if o in outcome_labels]
    if not plot_outcomes:
        return None
    if (
        observed_strata.empty
        and sim_observed_action.empty
        and observed_h1_counterfactual.empty
        and sim_h1_counterfactual.empty
    ):
        return None

    n_rows = len(plot_outcomes)
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig_w, fig_h = 13.2, 3.45 * n_rows + 0.65
    fig = plt.figure(figsize=(fig_w, fig_h))
    width_ratios, height_ratios = _fig4_strip_layout_ratios(
        fig_w,
        fig_h,
        n_rows,
        left_margin=0.05,
    )
    outer = GridSpec(
        n_rows + 1,
        2,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        wspace=0.08,
        hspace=0.38,
        left=0.05,
        right=0.99,
        top=0.96,
        bottom=0.07,
    )
    _fig4_add_column_header_row(fig, outer[0, 1])
    persistent_obs_regimes, persistent_sim_regimes = _fig4_persistent_regimes(
        action_spec
    )
    cf_obs_regimes, cf_sim_regimes = _fig4_cf_regimes_for_cohort(
        action_spec,
        action,
        observed_h1_counterfactual,
        sim_h1_counterfactual,
    )
    cf_sim_strata = set()
    if (
        not sim_h1_counterfactual.empty
        and {"action", "stratum"}.issubset(sim_h1_counterfactual.columns)
    ):
        cf_sim_strata = set(
            sim_h1_counterfactual.loc[
                sim_h1_counterfactual["action"].eq(action), "stratum"
            ]
            .astype(str)
            .unique()
        )
    if cf_sim_strata.intersection(
        {"stay_low", "switch_mid", "switch_high", "switch_mid_high"}
    ):
        right_sim_suffix = "observed action"
    else:
        right_sim_suffix = f"cf h{cf_switch_horizon}+"
    sim_suffix = (
        action_spec.sim_legend_suffix if action_spec is not None else "observed"
    )

    for row, event in enumerate(plot_outcomes):
        _fig4_row_strip_label(
            fig.add_subplot(outer[row + 1, 0]),
            outcome_labels[event],
        )
        inner = GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=outer[row + 1, 1],
            wspace=0.28,
        )
        ax_left = fig.add_subplot(inner[0, 0])
        ax_right = fig.add_subplot(inner[0, 1])
        sub_obs = (
            observed_strata[
                observed_strata["action"].eq(action)
                & observed_strata["event"].eq(event)
            ]
            if not observed_strata.empty
            and {"action", "event"}.issubset(observed_strata.columns)
            else pd.DataFrame()
        )
        sub_sim = (
            sim_observed_action[
                sim_observed_action["action"].eq(action)
                & sim_observed_action["event"].eq(event)
            ]
            if not sim_observed_action.empty
            and {"action", "event"}.issubset(sim_observed_action.columns)
            else pd.DataFrame()
        )
        _plot_observed_with_sim_overlay(
            ax_left,
            sub_obs,
            sub_sim,
            obs_group_col="stratum",
            sim_regime_col="stratum",
            obs_regimes=persistent_obs_regimes,
            sim_regimes=persistent_sim_regimes,
            panel=chr(ord("A") + row),
            ylabel="Age-standardized risk",
            sim_legend_suffix=sim_suffix,
        )

        cf_obs = (
            observed_h1_counterfactual[
                observed_h1_counterfactual["action"].eq(action)
                & observed_h1_counterfactual["event"].eq(event)
            ]
            if not observed_h1_counterfactual.empty
            and {"action", "event"}.issubset(observed_h1_counterfactual.columns)
            else pd.DataFrame()
        )
        cf_sim = (
            sim_h1_counterfactual[
                sim_h1_counterfactual["action"].eq(action)
                & sim_h1_counterfactual["event"].eq(event)
            ]
            if not sim_h1_counterfactual.empty
            and {"action", "event"}.issubset(sim_h1_counterfactual.columns)
            else pd.DataFrame()
        )
        _plot_observed_with_sim_overlay(
            ax_right,
            cf_obs,
            cf_sim,
            obs_group_col="stratum",
            sim_regime_col="stratum",
            obs_regimes=cf_obs_regimes,
            sim_regimes=cf_sim_regimes,
            panel=chr(ord("A") + row + len(plot_outcomes)),
            ylabel="Age-standardized risk",
            switch_horizon=cf_switch_horizon,
            sim_legend_suffix=right_sim_suffix,
        )

    if output_name is None:
        output_name = f"fig4_{action}.png"
    path = save_fig(fig, figures_dir / output_name, dpi)
    if action == FIG4_DEFAULT_ACTION and output_name != "fig4.png":
        alias = figures_dir / "fig4.png"
        alias.write_bytes(path.read_bytes())
        print(f"Wrote {alias}", flush=True)
    return path


def plot_fig4_all_actions(
    *,
    observed_strata: pd.DataFrame,
    sim_observed_action: pd.DataFrame,
    observed_h1_counterfactual: pd.DataFrame,
    sim_h1_counterfactual: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    default_action: str = FIG4_DEFAULT_ACTION,
    actions: Sequence[str] | None = None,
    all_actions: bool = False,
) -> list[Path]:
    """Write ``fig4_{action}.png`` for each action; also ``fig4.png`` for ``default_action``."""
    present: list[str] = []
    for frame in (
        observed_strata,
        sim_observed_action,
        observed_h1_counterfactual,
        sim_h1_counterfactual,
    ):
        if frame.empty or "action" not in frame.columns:
            continue
        for name in frame["action"].astype(str):
            if name not in present:
                present.append(name)
    if actions:
        ordered = [a for a in actions if a in present or a == default_action]
    else:
        ordered = [a for a in FIG4_ACTION_ORDER if a in present] + [
            a for a in present if a not in FIG4_ACTION_ORDER
        ]
    if not all_actions:
        ordered = [a for a in ordered if a == default_action]
    if not ordered:
        ordered = [default_action]
    paths: list[Path] = []
    for action in ordered:
        path = plot_fig4(
            observed_strata=observed_strata,
            sim_observed_action=sim_observed_action,
            observed_h1_counterfactual=observed_h1_counterfactual,
            sim_h1_counterfactual=sim_h1_counterfactual,
            figures_dir=figures_dir,
            dpi=dpi,
            action=action,
            primary_action=default_action,
        )
        if path is not None:
            paths.append(path)
    return paths


def _fig4_subset(frame: pd.DataFrame, action: str, event: str) -> pd.DataFrame:
    if frame.empty or not {"action", "event"}.issubset(frame.columns):
        return pd.DataFrame()
    return frame[frame["action"].eq(action) & frame["event"].eq(event)]


def _fig4_persistent_regimes(
    action_spec,
    *,
    short_labels: bool = False,
) -> tuple[
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
]:
    if action_spec is not None and action_spec.binary:
        persistent_obs = [
            ("low", ACTION_COLORS["low"], "low"),
            ("high", ACTION_COLORS["high"], "high"),
        ]
        persistent_sim = [
            ("low", ACTION_COLORS["low"], "low"),
            ("high", ACTION_COLORS["high"], "high"),
        ]
    else:
        if short_labels:
            obs_low = sim_low = "low"
            sim_mid = "mid"
            sim_high = "high"
        else:
            obs_low = action_spec.obs_low_label if action_spec else "low"
            sim_low = action_spec.sim_low_label if action_spec else "low"
            sim_mid = (
                action_spec.sim_mid_label
                if action_spec and action_spec.sim_mid_label
                else "mid"
            )
            sim_high = action_spec.sim_high_label if action_spec else "high"
        persistent_obs = [
            ("low", ACTION_COLORS["low"], obs_low),
            ("mid", ACTION_COLORS["baseline"], sim_mid),
            ("high", ACTION_COLORS["high"], sim_high),
        ]
        persistent_sim = [
            ("low", ACTION_COLORS["low"], sim_low),
            ("mid", ACTION_COLORS["baseline"], sim_mid),
            ("high", ACTION_COLORS["high"], sim_high),
        ]
    return persistent_obs, persistent_sim


def _fig4_cf_regimes_for_cohort(
    action_spec,
    action: str,
    cf_obs_all: pd.DataFrame,
    cf_sim_all: pd.DataFrame,
) -> tuple[
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
]:
    if action_spec is not None and action_spec.binary:
        cf_obs_regimes = [
            ("stay_low", ACTION_COLORS["low"], "stay low"),
            ("switch_high", ACTION_COLORS["high"], "low → high"),
        ]
        cf_sim_regimes = [
            ("low", ACTION_COLORS["low"], "stay low"),
            ("high", ACTION_COLORS["high"], "low → high"),
        ]
    else:
        cf_obs_regimes = [
            ("stay_low", ACTION_COLORS["low"], "stay low"),
            ("switch_mid", ACTION_COLORS["baseline"], "low → mid"),
            ("switch_high", ACTION_COLORS["high"], "low → high"),
        ]
        cf_sim_regimes = [
            ("low", ACTION_COLORS["low"], "stay low"),
            ("mid", ACTION_COLORS["baseline"], "low → mid"),
            ("high", ACTION_COLORS["high"], "low → high"),
        ]
    cf_strata: set[str] = set()
    if not cf_obs_all.empty and "stratum" in cf_obs_all.columns:
        sub = cf_obs_all
        if "action" in sub.columns:
            sub = sub[sub["action"].eq(action)]
        cf_strata = set(sub["stratum"].astype(str).unique())
    has_split_switch = bool(cf_strata.intersection({"switch_mid", "switch_high"}))
    if "switch_mid_high" in cf_strata and not has_split_switch:
        if action_spec is not None and action_spec.binary:
            cf_obs_regimes = [
                ("stay_low", ACTION_COLORS["low"], "stay low"),
                ("switch_mid_high", ACTION_COLORS["high"], "low → high"),
            ]
        else:
            cf_obs_regimes = [
                ("stay_low", ACTION_COLORS["low"], "stay low"),
                ("switch_mid_high", ACTION_COLORS["high"], "low → mid/high"),
            ]
    cf_sim_strata: set[str] = set()
    if not cf_sim_all.empty and "stratum" in cf_sim_all.columns:
        sim_sub = cf_sim_all
        if "action" in sim_sub.columns:
            sim_sub = sim_sub[sim_sub["action"].eq(action)]
        cf_sim_strata = set(sim_sub["stratum"].astype(str).unique())
    if cf_sim_strata.intersection(
        {"stay_low", "switch_mid", "switch_high", "switch_mid_high"}
    ):
        cf_sim_regimes = list(cf_obs_regimes)
    return cf_obs_regimes, cf_sim_regimes


def _fig4_actions_in_cohort_tables(
    cohorts: Sequence[tuple[str, dict]],
) -> list[str]:
    present: set[str] | None = None
    for _, tables in cohorts:
        actions: set[str] = set()
        for key in (
            "observed_strata",
            "sim_observed_action",
            "observed_h1_counterfactual",
            "sim_h1_counterfactual",
        ):
            frame = tables.get(key, pd.DataFrame())
            if frame is not None and not frame.empty and "action" in frame.columns:
                actions.update(frame["action"].astype(str).unique())
        present = actions if present is None else present & actions
    if not present:
        return []
    return [a for a in FIG4_ACTION_ORDER if a in present] + [
        a for a in sorted(present) if a not in FIG4_ACTION_ORDER
    ]


def plot_fig4_cohort3(
    cohorts: Sequence[tuple[str, dict]],
    figures_dir: Path,
    dpi: int,
    *,
    action: str = "moderate_activity_frequency",
    event: str = "death_event",
    output_name: str | None = None,
    cf_switch_horizon: int = FIG5_H1_CF_SWITCH_HORIZON,
) -> Path | None:
    """Death panels from Fig 4 (original C/F) for HRS, ELSA, and CHARLS.

    Rows are cohorts (left strip); columns use top strip headers for static vs
    counterfactual switch. Panels are relabeled A–F.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    action_spec = get_fig4_action_spec(action)
    if action_spec is None or not cohorts:
        return None
    if output_name is None:
        output_name = f"fig4_cohort3_{action}.png"
    persistent_obs_regimes, persistent_sim_regimes = _fig4_persistent_regimes(
        action_spec,
        short_labels=True,
    )
    n_rows = len(cohorts)
    fig_w, fig_h = 13.4, 3.55 * n_rows + 0.75
    fig = plt.figure(figsize=(fig_w, fig_h))
    width_ratios, height_ratios = _fig4_strip_layout_ratios(fig_w, fig_h, n_rows)
    outer = GridSpec(
        n_rows + 1,
        2,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        wspace=0.08,
        hspace=0.38,
        left=0.03,
        right=0.99,
        top=0.96,
        bottom=0.07,
    )
    _fig4_add_column_header_row(fig, outer[0, 1])
    letters = "ABCDEF"
    for row, (cohort_name, tables) in enumerate(cohorts):
        observed_strata = tables.get("observed_strata", pd.DataFrame())
        sim_observed = tables.get("sim_observed_action", pd.DataFrame())
        cf_obs_all = tables.get("observed_h1_counterfactual", pd.DataFrame())
        cf_sim_all = tables.get("sim_h1_counterfactual", pd.DataFrame())
        cf_obs_regimes, cf_sim_regimes = _fig4_cf_regimes_for_cohort(
            action_spec,
            action,
            cf_obs_all,
            cf_sim_all,
        )

        _fig4_row_strip_label(
            fig.add_subplot(outer[row + 1, 0]), cohort_name, fontsize=11
        )

        inner = GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=outer[row + 1, 1],
            wspace=0.28,
        )
        ax_left = fig.add_subplot(inner[0, 0])
        ax_right = fig.add_subplot(inner[0, 1])
        extra_top = 1.42 if str(cohort_name).upper() != "HRS" else 1.15
        extra_cap = 1.22 if str(cohort_name).upper() != "HRS" else 1.2
        _plot_observed_with_sim_overlay(
            ax_left,
            _fig4_subset(observed_strata, action, event),
            _fig4_subset(sim_observed, action, event),
            obs_group_col="stratum",
            sim_regime_col="stratum",
            obs_regimes=persistent_obs_regimes,
            sim_regimes=persistent_sim_regimes,
            panel=letters[2 * row],
            ylabel="Age-standardized risk",
            sim_legend_suffix="",
            ylim_factor=extra_top,
            ylim_cap=extra_cap,
            xlabel="Horizon",
            integer_xticks=True,
            include_sim_n=True,
        )
        _plot_observed_with_sim_overlay(
            ax_right,
            _fig4_subset(cf_obs_all, action, event),
            _fig4_subset(cf_sim_all, action, event),
            obs_group_col="stratum",
            sim_regime_col="stratum",
            obs_regimes=cf_obs_regimes,
            sim_regimes=cf_sim_regimes,
            panel=letters[2 * row + 1],
            ylabel="Age-standardized risk",
            switch_horizon=cf_switch_horizon,
            sim_legend_suffix="",
            ylim_factor=extra_top,
            ylim_cap=extra_cap,
            xlabel="Horizon",
            integer_xticks=True,
            include_sim_n=True,
        )
    return save_fig(fig, figures_dir / output_name, dpi)


def plot_fig4_cohort3_all_actions(
    cohorts: Sequence[tuple[str, dict]],
    figures_dir: Path,
    dpi: int,
    *,
    actions: Sequence[str] | None = None,
    event: str = "death_event",
) -> list[Path]:
    """Write ``fig4_cohort3_{action}.png`` for each intervention across cohorts."""
    ordered = list(actions) if actions else _fig4_actions_in_cohort_tables(cohorts)
    paths: list[Path] = []
    for action in ordered:
        if get_fig4_action_spec(action) is None:
            continue
        path = plot_fig4_cohort3(
            cohorts,
            figures_dir,
            dpi,
            action=action,
            event=event,
        )
        if path is not None:
            paths.append(path)
    return paths


plot_fig5_validation = plot_fig4  # backward-compatible alias


def _fig5_km_n_persons(group: pd.DataFrame) -> int | None:
    if group.empty or "n_persons" not in group.columns:
        return None
    vals = group["n_persons"].dropna()
    if vals.empty:
        return None
    return int(vals.max())


def _plot_fig5_km_panel(
    ax,
    km: pd.DataFrame,
    *,
    obs_regimes: Sequence[tuple[str, str, str]],
    sim_regimes: Sequence[tuple[str, str, str]],
    panel: str,
    title: str,
    switch_horizon: int | None = None,
    sim_legend_suffix: str = "observed cigs",
) -> None:
    """Step KM: dashed = observed, solid = simulated (Fig 3E form + Fig 4 groups)."""
    panel_label(ax, panel)
    if km.empty:
        ax.text(
            0.5,
            0.5,
            "No KM data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Wave", fontsize=8)
        ax.set_ylabel("Event-free survival", fontsize=8)
        ax.set_ylim(0.0, 1.05)
        _fig3_style_axis(ax)
        return

    kind_col = "kind" if "kind" in km.columns else "source"
    for regime, color, label in obs_regimes:
        g = km[(km[kind_col] == "obs") & (km["stratum"] == regime)].sort_values(
            "horizon"
        )
        if g.empty:
            continue
        n0 = _fig5_km_n_persons(g)
        n_txt = f" (n={n0})" if n0 is not None else ""
        ax.step(
            g["horizon"],
            g["survival"],
            where="post",
            color=color,
            linestyle="--",
            linewidth=1.3,
            alpha=0.9,
            label=f"obs {label}{n_txt}",
        )
    for regime, color, label in sim_regimes:
        g = km[(km[kind_col] == "sim") & (km["stratum"] == regime)].sort_values(
            "horizon"
        )
        if g.empty:
            continue
        ax.step(
            g["horizon"],
            g["survival"],
            where="post",
            color=color,
            linestyle="-",
            linewidth=1.6,
            label=f"sim {label} ({sim_legend_suffix})",
        )
    if switch_horizon is not None:
        ax.axvline(
            switch_horizon - 0.5,
            color=GREY,
            linestyle=":",
            linewidth=1,
            alpha=0.85,
        )
    ax.set_xlabel("Wave", fontsize=8)
    ax.set_ylabel("Event-free survival", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0.0, 1.05)
    if "horizon" in km.columns and not km.empty:
        xmax = int(km["horizon"].max())
        ax.set_xticks(list(range(0, xmax + 1)))
    ax.legend(fontsize=5.5, loc="best")
    _fig3_style_axis(ax)


def _fig5_km_actions_present(
    persistent: pd.DataFrame,
    counterfactual: pd.DataFrame,
) -> list[str]:
    present: list[str] = []
    for frame in (persistent, counterfactual):
        if frame.empty or "action" not in frame.columns:
            continue
        for name in frame["action"].astype(str):
            if name not in present:
                present.append(name)
    ordered = [a for a in FIG4_ACTION_ORDER if a in present]
    ordered += [a for a in present if a not in ordered]
    if not ordered:
        ordered = [FIG5_KM_ACTION]
    return ordered


def _fig5_km_regimes(
    action: str,
) -> tuple[
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
    str,
    str,
]:
    spec = get_fig4_action_spec(action)
    if spec is not None and spec.binary:
        persistent_obs = [
            ("low", ACTION_COLORS["low"], spec.obs_low_label),
            ("high", ACTION_COLORS["high"], spec.obs_exposed_label),
        ]
        persistent_sim = [
            ("low", ACTION_COLORS["low"], spec.sim_low_label),
            ("high", ACTION_COLORS["high"], spec.sim_high_label),
        ]
        cf_sim = [
            ("low", ACTION_COLORS["low"], "stay low"),
            ("high", ACTION_COLORS["high"], "→ high"),
        ]
    else:
        obs_low = spec.obs_low_label if spec else "low"
        sim_low = spec.sim_low_label if spec else "low"
        sim_mid = spec.sim_mid_label if spec and spec.sim_mid_label else "mid"
        sim_high = spec.sim_high_label if spec else "high"
        persistent_obs = [
            ("low", ACTION_COLORS["low"], obs_low),
            ("mid", ACTION_COLORS["baseline"], sim_mid),
            ("high", ACTION_COLORS["high"], sim_high),
        ]
        persistent_sim = [
            ("low", ACTION_COLORS["low"], sim_low),
            ("mid", ACTION_COLORS["baseline"], sim_mid),
            ("high", ACTION_COLORS["high"], sim_high),
        ]
        cf_sim = [
            ("low", ACTION_COLORS["low"], "stay low"),
            ("mid", ACTION_COLORS["baseline"], "→ mid"),
            ("high", ACTION_COLORS["high"], "→ high"),
        ]
    if spec is not None and spec.binary:
        cf_obs = [
            ("stay_low", ACTION_COLORS["low"], "stay low"),
            ("switch_high", ACTION_COLORS["high"], "→ high"),
        ]
    else:
        cf_obs = [
            ("stay_low", ACTION_COLORS["low"], "stay low"),
            ("switch_mid", ACTION_COLORS["baseline"], "→ mid"),
            ("switch_high", ACTION_COLORS["high"], "→ high"),
        ]
    title_left = spec.title if spec is not None else _action_short_label(action)
    sim_suffix = spec.sim_legend_suffix if spec is not None else "observed"
    return persistent_obs, persistent_sim, cf_obs, cf_sim, title_left, sim_suffix


def _filter_fig5_km_action(frame: pd.DataFrame, action: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "action" not in frame.columns:
        return frame if action == FIG5_KM_ACTION else pd.DataFrame()
    return frame[frame["action"].astype(str).eq(action)].copy()


def plot_fig5(
    *,
    persistent: pd.DataFrame,
    counterfactual: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
    cf_switch_horizon: int = FIG5_H1_CF_SWITCH_HORIZON,
) -> Path | None:
    """Fig 5: death KM by Fig 4 action groups (one row per intervention)."""
    if persistent.empty and counterfactual.empty:
        return None
    actions = _fig5_km_actions_present(persistent, counterfactual)
    n_rows = len(actions)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(13.2, 3.4 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    for row, action in enumerate(actions):
        (
            persistent_obs,
            persistent_sim,
            cf_obs,
            cf_sim,
            title_left,
            sim_suffix,
        ) = _fig5_km_regimes(action)
        sub_p = _filter_fig5_km_action(persistent, action)
        sub_c = _filter_fig5_km_action(counterfactual, action)
        _plot_fig5_km_panel(
            axes[row, 0],
            sub_p,
            obs_regimes=persistent_obs,
            sim_regimes=persistent_sim,
            panel=chr(ord("A") + 2 * row),
            title=f"{title_left} · Death KM",
            sim_legend_suffix=sim_suffix,
        )
        _plot_fig5_km_panel(
            axes[row, 1],
            sub_c,
            obs_regimes=cf_obs,
            sim_regimes=cf_sim,
            panel=chr(ord("A") + 2 * row + 1),
            title=f"h1 low counterfactual · {_action_short_label(action)}",
            switch_horizon=cf_switch_horizon,
            sim_legend_suffix=f"cf h{cf_switch_horizon}+",
        )
    return save_fig(fig, figures_dir / "fig5.png", dpi)


FIG6_EVENT_OUTPUT: dict[str, str] = {
    "death_event": "fig6_simulation_death.png",
}

FIG6_SCENARIO_STYLE: dict[str, dict[str, object]] = {
    "observed": {
        "color": NATURE["grey"],
        "linestyle": "-",
        "marker": "o",
        "markersize": 4.5,
        "linewidth": 1.3,
    },
    "reduce_smoking": {
        "color": NATURE["green"],
        "linestyle": "--",
        "marker": "s",
        "markersize": 5.0,
        "linewidth": 1.7,
    },
    "moderate_activity": {
        "color": NATURE["blue"],
        "linestyle": "-.",
        "marker": "D",
        "markersize": 5.0,
        "linewidth": 1.7,
    },
    "start_antihypertensive": {
        "color": NATURE["red"],
        "linestyle": ":",
        "marker": "^",
        "markersize": 5.5,
        "linewidth": 1.9,
    },
}

FIG6_SCENARIO_SHORT_LABELS: dict[str, str] = {
    "observed": "Observed",
    "reduce_smoking": "Reduce smoking",
    "moderate_activity": "Moderate activity",
    "start_antihypertensive": "Start antihypertensive",
}

FIG6_TABLE_LABELS: dict[str, str] = {
    "observed": "Observed",
    "reduce_smoking": "Reduce smoking",
    "moderate_activity": "Moderate activity",
    "start_antihypertensive": "Start antihypertensive",
}


def _fig6_scenario_plot_label(
    scenario: str,
    sub: pd.DataFrame,
    label_col: str,
    scenario_labels: dict[str, str],
) -> str:
    if scenario in FIG6_SCENARIO_SHORT_LABELS:
        return FIG6_SCENARIO_SHORT_LABELS[scenario]
    if label_col in sub.columns and not sub[label_col].empty:
        return str(sub[label_col].iloc[0])
    return scenario_labels.get(scenario, scenario)


def _fig6_end_horizon_rows(
    death: pd.DataFrame,
    *,
    scenarios: Sequence[str],
    scenario_col: str,
    label_col: str,
    scenario_labels: dict[str, str],
    horizon: int,
) -> list[tuple[str, str, str, str]]:
    """Pathway / P(death) / ΔP rows at the plotted end horizon."""
    end = death[pd.to_numeric(death["horizon"], errors="coerce").eq(horizon)].copy()
    obs = end[end[scenario_col].astype(str).eq("observed")]
    obs_p = (
        float(pd.to_numeric(obs["risk_prob"], errors="coerce").iloc[0])
        if not obs.empty
        else None
    )
    rows: list[tuple[str, str, str, str]] = []
    for scenario in scenarios:
        sub = end[end[scenario_col].astype(str).eq(scenario)]
        if sub.empty:
            continue
        p = float(pd.to_numeric(sub["risk_prob"], errors="coerce").iloc[0])
        if not np.isfinite(p):
            continue
        label = FIG6_TABLE_LABELS.get(
            scenario,
            _fig6_scenario_plot_label(scenario, sub, label_col, scenario_labels),
        )
        color = str(FIG6_SCENARIO_STYLE.get(scenario, {}).get("color", NATURE["navy"]))
        if scenario == "observed" or obs_p is None:
            delta = "—"
        else:
            delta = f"{p - obs_p:+.2f}"
        rows.append((label, f"{p:.2f}", delta, color))
    return rows


def _fig6_draw_horizon_table(
    ax: plt.Axes,
    *,
    rows: Sequence[tuple[str, str, str, str]],
    box: tuple[float, float, float, float],
) -> None:
    if not rows:
        return
    x0, y0, width, height = box
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            height,
            transform=ax.transAxes,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            facecolor="#F3F8FC",
            edgecolor="#C5D9EA",
            linewidth=0.7,
            mutation_aspect=0.8,
            zorder=5,
            clip_on=True,
        )
    )
    n_rows = len(rows)
    inner_x0 = x0 + 0.014
    inner_w = width - 0.026
    inner_top = y0 + height - 0.012
    inner_bot = y0 + 0.014
    header_h = (inner_top - inner_bot) / (n_rows + 1.55) * 1.55
    row_h = (inner_top - inner_bot - header_h) / max(n_rows, 1)
    col_x = [
        inner_x0 + 0.02 * inner_w,
        inner_x0 + 0.60 * inner_w,
        inner_x0 + 0.88 * inner_w,
    ]
    headers = ["Action\nsimulation", "Mortality\nrisk (h=5)", "ΔP"]
    ink = NATURE["navy"]
    header_face = "#D7EAF7"
    ax.add_patch(
        Rectangle(
            (inner_x0, inner_top - header_h),
            inner_w,
            header_h,
            transform=ax.transAxes,
            facecolor=header_face,
            edgecolor="none",
            zorder=5.2,
            clip_on=True,
        )
    )
    for x, header in zip(col_x, headers):
        ax.text(
            x,
            inner_top - 0.5 * header_h,
            header,
            transform=ax.transAxes,
            ha="left" if header.startswith("Action") else "center",
            va="center",
            fontsize=6.2,
            color=ink,
            fontweight="bold",
            zorder=6,
            clip_on=True,
            linespacing=1.05,
        )
    for i, (label, p_txt, d_txt, _color) in enumerate(rows):
        cy = inner_top - header_h - (i + 0.5) * row_h
        values = (label, p_txt, d_txt)
        aligns = ("left", "center", "center")
        for x, val, ha in zip(col_x, values, aligns):
            ax.text(
                x,
                cy,
                val,
                transform=ax.transAxes,
                ha=ha,
                va="center",
                fontsize=6.2,
                color=ink,
                zorder=6,
                clip_on=True,
                linespacing=1.05,
            )


def _fig6_inset_table_box(
    ax: plt.Axes,
    death: pd.DataFrame,
    max_h: int,
) -> tuple[float, float, float, float]:
    """Lower-right box in axes fraction, kept below the rightmost curves."""
    width, height, x0, y0 = 0.50, 0.32, 0.48, 0.02
    xmin, xmax = ax.get_xlim()
    x_left = xmin + x0 * (xmax - xmin)
    right = death[pd.to_numeric(death["horizon"], errors="coerce") >= max(x_left - 0.05, 3.5)]
    right_min = float(pd.to_numeric(right["risk_prob"], errors="coerce").min())
    ymin, ymax = ax.get_ylim()
    if not np.isfinite(right_min):
        return x0, y0, width, height
    frac = y0 + height
    clearance = 0.16
    new_ymin = (right_min - clearance - frac * ymax) / max(1.0 - frac, 1e-6)
    ax.set_ylim(new_ymin, ymax)
    return x0, y0, width, height


def plot_fig6_case_study(
    *,
    rollout: pd.DataFrame,
    person_meta: pd.DataFrame | dict[str, object],
    figures_dir: Path,
    dpi: int,
) -> Path | None:
    """Fig 6: delayed single-action counterfactuals (switch at h=3, plot h=1..5)."""
    if rollout.empty:
        return None
    death = rollout[rollout["event"].astype(str).eq("death_event")].copy()
    if death.empty:
        death = rollout.copy()

    scenario_col = "scenario" if "scenario" in death.columns else "regime"
    label_col = (
        "scenario_label"
        if "scenario_label" in death.columns
        else ("regime_label" if "regime_label" in death.columns else scenario_col)
    )
    scenario_labels = dict(FIG6_CASE_STUDY_SCENARIO_LABELS)
    scenario_order = list(FIG6_CASE_STUDY_SCENARIO_ORDER)
    style_map = FIG6_SCENARIO_STYLE
    max_h = int(
        pd.to_numeric(death["horizon"], errors="coerce").max()
        if "horizon" in death.columns
        else FIG6_CASE_STUDY_MAX_HORIZON
    )

    switch_h = int(
        pd.to_numeric(death.get("switch_horizon"), errors="coerce").dropna().iloc[0]
        if "switch_horizon" in death.columns and death["switch_horizon"].notna().any()
        else FIG6_CASE_STUDY_SWITCH_HORIZON
    )
    scenarios = [
        s for s in scenario_order if s in set(death[scenario_col].astype(str))
    ]
    if not scenarios:
        scenarios = sorted(death[scenario_col].astype(str).unique())
    table_rows = _fig6_end_horizon_rows(
        death,
        scenarios=scenarios,
        scenario_col=scenario_col,
        label_col=label_col,
        scenario_labels=scenario_labels,
        horizon=max_h,
    )

    fig, ax = plt.subplots(1, 1, figsize=(5.3, 3.9))
    for scenario in scenarios:
        sub = death[death[scenario_col].astype(str).eq(scenario)].sort_values("horizon")
        if sub.empty:
            continue
        style = style_map.get(scenario, {})
        label = _fig6_scenario_plot_label(scenario, sub, label_col, scenario_labels)
        ax.plot(
            sub["horizon"],
            sub["risk_prob"],
            color=style.get("color", NATURE["navy"]),
            linestyle=str(style.get("linestyle", "-")),
            marker=style.get("marker", "o"),
            markersize=style.get("markersize", 3.5),
            linewidth=style.get("linewidth", 1.2),
            label=label,
            zorder=4 if scenario == "observed" else 3,
            alpha=0.95 if scenario != "observed" else 1.0,
        )
    y_vals = pd.to_numeric(death["risk_prob"], errors="coerce").dropna()
    if not y_vals.empty:
        ymin = float(y_vals.min())
        ymax = float(y_vals.max())
        span = max(ymax - ymin, 0.04)
        pad = max(0.025, span * 0.15)
        ax.set_ylim(ymin - pad, ymax + pad * 1.35)
    else:
        ax.set_ylim(-0.05, 1.05)
    action_mark = float(switch_h) - 0.5
    ax.axvline(
        action_mark,
        color=NATURE["grey"],
        linestyle="--",
        linewidth=0.95,
        zorder=1,
    )
    ax.annotate(
        "",
        xy=(action_mark, 1.045),
        xycoords=("data", "axes fraction"),
        xytext=(action_mark, 1.0),
        textcoords=("data", "axes fraction"),
        arrowprops={
            "arrowstyle": "-|>",
            "color": NATURE["navy"],
            "lw": 0.9,
            "mutation_scale": 9,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
        zorder=6,
    )
    ax.annotate(
        "action starts",
        xy=(action_mark, 1.045),
        xycoords=("data", "axes fraction"),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=NATURE["navy"],
        annotation_clip=False,
        zorder=6,
    )
    ax.set_xlabel("Open-loop horizon")
    ax.set_ylabel("Mortality risk")
    ax.set_xticks(list(range(1, max(max_h, FIG6_CASE_STUDY_MAX_HORIZON) + 1)))
    ax.set_xlim(0.75, max(max_h, FIG6_CASE_STUDY_MAX_HORIZON) + 0.35)
    clean_axis(ax)
    ax.legend(loc="upper left", fontsize=7, frameon=False, handlelength=2.8)
    box = _fig6_inset_table_box(ax, death, max_h)
    _fig6_draw_horizon_table(ax, rows=table_rows, box=box)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return save_fig(fig, figures_dir / "fig6_case_study.png", dpi)


def _plot_fig6_clinical_event(
    *,
    sim_levels: pd.DataFrame,
    delayed_switch: pd.DataFrame,
    event: str,
    figures_dir: Path,
    dpi: int,
    switch_horizon: int,
    action_order: Sequence[str] | None,
    output_name: str,
) -> Path | None:
    """One Fig 6 panel grid: all actions for a single clinical outcome."""
    actions = _resolve_fig5_actions(sim_levels, delayed_switch, action_order)
    if not actions:
        actions = ["(none)"]

    short_e = EVENT_LABELS.get(event, event)
    ylabel = f"Mean P({short_e})"
    n_rows = len(actions)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(8.8, 2.85 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for row, action in enumerate(actions):
        ax_on, ax_sw = axes[row, 0], axes[row, 1]
        short_a = _action_short_label(action)
        sub_on = pd.DataFrame()
        sub_sw = pd.DataFrame()
        if not sim_levels.empty and "action" in sim_levels.columns:
            m = sim_levels["action"].eq(action)
            if "event" in sim_levels.columns:
                m = m & sim_levels["event"].eq(event)
            sub_on = sim_levels[m]
        if not delayed_switch.empty and "action" in delayed_switch.columns:
            m = delayed_switch["action"].eq(action)
            if "event" in delayed_switch.columns:
                m = m & delayed_switch["event"].eq(event)
            sub_sw = delayed_switch[m]

        letter_i = 2 * row
        panel_on = chr(ord("A") + letter_i) if letter_i < 26 else f"A{letter_i}"
        panel_sw = (
            chr(ord("A") + letter_i + 1) if letter_i + 1 < 26 else f"A{letter_i + 1}"
        )
        _plot_always_on_action_panel(
            ax_on,
            sub_on,
            panel=panel_on,
            title=f"Always-on · {short_a}",
            ylabel=ylabel,
        )
        _plot_delayed_switch_action_panel(
            ax_sw,
            sub_sw,
            panel=panel_sw,
            title=f"Switch {short_a} at h={switch_horizon}",
            ylabel=ylabel,
            switch_horizon=switch_horizon,
        )

    return save_fig(fig, figures_dir / output_name, dpi)


def plot_fig6_simulation_clinical(
    *,
    sim_levels: pd.DataFrame,
    delayed_switch: pd.DataFrame | None = None,
    figures_dir: Path,
    dpi: int,
    switch_horizon: int = 3,
    action_order: Sequence[str] | None = None,
    event_order: Sequence[str] | None = None,
) -> list[Path]:
    """Fig 6: action-intervention simulation for configured acute outcomes (death only)."""
    delayed_switch = delayed_switch if delayed_switch is not None else pd.DataFrame()
    allowed = set(FIG6_SIMULATION_EVENTS)

    events: list[str] = []
    if event_order:
        events = [e for e in dict.fromkeys(event_order) if e in allowed]
    else:
        for frame in (sim_levels, delayed_switch):
            if not frame.empty and "event" in frame.columns:
                for e in frame["event"].tolist():
                    if e in allowed and e not in events:
                        events.append(e)
        preferred = [e for e in FIG6_SIMULATION_EVENTS if e in allowed]
        present = set(events)
        events = [e for e in preferred if e in present] + [
            e for e in events if e not in preferred
        ]

    if not events:
        events = ["(none)"]

    paths: list[Path] = []
    for event in events:
        output_name = FIG6_EVENT_OUTPUT.get(
            event,
            f"fig6_simulation_{event}.png",
        )
        path = _plot_fig6_clinical_event(
            sim_levels=sim_levels,
            delayed_switch=delayed_switch,
            event=event,
            figures_dir=figures_dir,
            dpi=dpi,
            switch_horizon=switch_horizon,
            action_order=action_order,
            output_name=output_name,
        )
        if path is not None:
            paths.append(path)
    return paths


def plot_fig7(
    *,
    examples: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> Path | None:
    """Fig 7: individual one-step trajectories (observed 0/1 vs JEPA P(worsen))."""
    present_d = (
        set(examples["outcome"].unique())
        if not examples.empty and "outcome" in examples.columns
        else set()
    )
    outcomes_d = [o for o in PRIMARY_LEVELS if o in present_d]
    n_cols_d = 1 if len(outcomes_d) <= 1 else 2
    n_rows_d = max(int(np.ceil(len(outcomes_d) / n_cols_d)), 1)

    fig = plt.figure(figsize=(9.0, 2.2 + 1.9 * n_rows_d), constrained_layout=True)
    gs = fig.add_gridspec(n_rows_d, n_cols_d)

    if not outcomes_d:
        ax = fig.add_subplot(gs[0, :])
        panel_label(ax, "A")
        ax.text(0.5, 0.5, "No example trajectories", ha="center", va="center", transform=ax.transAxes)
        clean_axis(ax)
        return save_fig(fig, figures_dir / "fig7_trajectories.png", dpi)

    if "next_wave" in examples.columns:
        time_col = "next_wave"
    elif "wave" in examples.columns:
        time_col = "wave"
    else:
        time_col = "horizon"
    persons = list(dict.fromkeys(examples["person_id"].tolist()))
    person_color = {pid: nature_color(i) for i, pid in enumerate(persons)}
    for j, outcome in enumerate(outcomes_d):
        row, col = divmod(j, n_cols_d)
        span = gs[row, :] if n_cols_d == 1 else gs[row, col]
        ax = fig.add_subplot(span)
        if j == 0:
            panel_label(ax, "A")
        sub_all = examples[examples["outcome"] == outcome]
        label = EVENT_LABELS.get(outcome, outcome)
        for pid in persons:
            sub = sub_all[sub_all["person_id"] == pid].sort_values(time_col)
            if sub.empty:
                continue
            color = person_color[pid]
            ax.plot(
                sub[time_col],
                sub["observed_raw"],
                color=color,
                linestyle="-",
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                alpha=0.95,
            )
            ax.plot(
                sub[time_col],
                sub["jepa_raw"],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.95,
            )
        ax.set_ylim(-0.08, 1.08)
        if j == 0:
            ax.set_title("Individual trajectories (1-step)", fontsize=9)
        ax.set_ylabel(label, fontsize=7.5)
        if row == n_rows_d - 1:
            ax.set_xlabel("Wave")
        ax.tick_params(labelsize=8)
        if j == 0:
            from matplotlib.lines import Line2D

            ax.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linestyle="-",
                        marker="o",
                        markersize=3.5,
                        label="Observed 0/1",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linestyle="--",
                        label=f"{WM_LEGEND} P(worsen)",
                    ),
                ],
                loc="best",
                fontsize=7,
            )
        clean_axis(ax)

    return save_fig(fig, figures_dir / "fig7_trajectories.png", dpi)


def _plot_openloop_relative(
    ax,
    dyn: pd.DataFrame,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    panel: str,
    empty_msg: str,
) -> None:
    if dyn.empty or value_col not in dyn.columns:
        ax.text(0.5, 0.5, empty_msg, ha="center", va="center", transform=ax.transAxes)
        panel_title(ax, panel, title)
    else:
        events = list(dict.fromkeys(dyn["event"].tolist()))
        for i, event in enumerate(events):
            sub = dyn[dyn["event"] == event].sort_values("horizon")
            label = str(sub["label"].iloc[0])[:18]
            ax.plot(
                sub["horizon"],
                sub[value_col],
                marker="o",
                color=nature_color(i),
                label=label,
            )
        ax.axhline(1.0, color=GREY, linestyle="--", linewidth=1)
        ax.set_xlabel("Horizon (waves)")
        ax.set_ylabel(ylabel)
        panel_title(ax, panel, title)
        ax.legend(fontsize=6.5, ncol=2)
    clean_axis(ax)


def _plot_openloop_absolute(
    ax,
    dyn: pd.DataFrame,
    *,
    jepa_col: str,
    persist_col: str,
    ylabel: str,
    title: str,
    panel: str,
    ylim: tuple[float, float],
    empty_msg: str,
) -> None:
    if dyn.empty or jepa_col not in dyn.columns:
        ax.text(0.5, 0.5, empty_msg, ha="center", va="center", transform=ax.transAxes)
        panel_title(ax, panel, title)
    else:
        events = list(dict.fromkeys(dyn["event"].tolist()))
        for i, event in enumerate(events):
            sub = dyn[dyn["event"] == event].sort_values("horizon")
            color = nature_color(i)
            label = str(sub["label"].iloc[0])[:14]
            ax.plot(
                sub["horizon"],
                sub[jepa_col],
                marker="o",
                color=color,
                label=f"{WM_LEGEND} {label}",
            )
            if persist_col in sub.columns:
                ax.plot(
                    sub["horizon"],
                    sub[persist_col],
                    marker="x",
                    color=color,
                    linestyle=":",
                    alpha=0.85,
                )
        ax.set_xlabel("Horizon (waves)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        panel_title(ax, panel, title)
        ax.legend(fontsize=6, ncol=2)
    clean_axis(ax)


def _plot_openloop_km_row(
    axes,
    km: pd.DataFrame,
    *,
    panel: str,
    event_label: str = "",
    horizons: tuple[int, ...] = (1, 2, 3),
) -> None:
    km_ok = not km.empty and "horizon" in km.columns
    n_h = len(horizons)
    for i, (ax, h) in enumerate(zip(axes, horizons)):
        if i == 0:
            panel_label(ax, panel)
        sub = km[km["horizon"] == h] if km_ok else pd.DataFrame()
        if sub.empty:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7,
            )
        else:
            for stratum in _fig3_ordered_strata(sub["stratum"].unique()):
                group = sub[sub["stratum"] == stratum]
                ax.step(
                    group["time"],
                    group["survival"],
                    where="post",
                    label=RISK_STRATUM_LABELS.get(str(stratum), str(stratum)),
                    color=RISK_COLORS.get(str(stratum), NATURE["grey"]),
                    linewidth=1.3,
                )
        title = f"{event_label} KM (h={h})" if event_label else f"KM (h={h})"
        if not sub.empty and "n_persons" in sub.columns:
            n_total = int(sub.drop_duplicates("stratum")["n_persons"].sum())
            title = f"{title}, n={n_total}"
        ax.set_title(title, fontsize=8.5, pad=6)
        ax.set_xlabel("Wave", fontsize=7.5)
        if i == 0:
            ax.set_ylabel("Event-free survival", fontsize=7.5)
        ax.set_ylim(0.0, 1.05)
        times = sub["time"] if not sub.empty and "time" in sub.columns else None
        _fig3_set_integer_xticks(ax, times)
        _fig3_style_axis(ax)


def write_report_md(
    path: Path,
    *,
    checkpoint: Path | None,
    notices: list[str],
    figure_paths: list[Path],
    csv_paths: list[Path],
    clinical: pd.DataFrame,
    level_cmp: pd.DataFrame,
    title: str | None = None,
    extra_header: list[str] | None = None,
) -> Path:
    lines = [
        title or "# Health world model Fig 1–7 evaluation report",
        "",
        f"- Checkpoint: `{checkpoint}`" if checkpoint else "- Checkpoint: *(history-only / missing)*",
    ]
    if extra_header:
        lines.extend(extra_header)
    lines += [
        "",
        "## Figures",
        "",
    ]
    for p in figure_paths:
        lines.append(f"- `{p.name}`")
    lines += ["", "## Source tables", ""]
    for p in csv_paths:
        lines.append(f"- `{p.name}`")
    if notices:
        lines += ["", "## Skipped / notices", ""]
        for n in notices:
            lines.append(f"- {n}")
    if not level_cmp.empty:
        metric = "auprc" if "auprc" in level_cmp.columns else "mae"
        index_col = "event" if "event" in level_cmp.columns else "outcome"
        heading = (
            "Worsening one-step metrics (Fig 3A/B)"
            if metric == "auprc"
            else "One-step level MAE (primary)"
        )
        lines += ["", f"## {heading}", "", "```"]
        pivot = level_cmp.pivot_table(index=index_col, columns="model", values=metric)
        lines.append(pivot.to_string())
        lines.append("```")
    if not clinical.empty:
        lines += ["", "## Clinical AUROC", "", "```"]
        pivot = clinical.pivot_table(index="event", columns="model", values="auroc")
        lines.append(pivot.to_string())
        lines.append("```")
        if "auprc" in clinical.columns:
            lines += ["", "## Clinical AUPRC", "", "```"]
            pivot = clinical.pivot_table(index="event", columns="model", values="auprc")
            lines.append(pivot.to_string())
            lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _fig4_validation_cache_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.fig4_validation_cache_dir:
        return Path(args.fig4_validation_cache_dir).resolve()
    return (output_dir / "fig4_validation_cache").resolve()


def _fig5_km_cache_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    if getattr(args, "fig5_km_cache_dir", None):
        return Path(args.fig5_km_cache_dir).resolve()
    return (output_dir / "fig5_km_cache").resolve()


def _collect_fig5_km(
    *,
    agent,
    spec,
    test_loader,
    device: torch.device,
    preprocessing: dict,
    args: argparse.Namespace,
    max_h: int,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load or compute Fig 5 death KM for every Fig 4 intervention."""
    notices: list[str] = []
    available_actions = {a.name for a in spec.action_continuous} | {
        a.name for a in spec.action_categorical
    }
    actions = resolve_fig5_simulation_actions(available_actions)
    if not actions:
        notices.append("Fig 5 skipped: no intervention actions in ModelSpec")
        return pd.DataFrame(), pd.DataFrame(), notices
    if "death_event" not in spec.reward_binary:
        notices.append("Fig 5 skipped: death_event unavailable")
        return pd.DataFrame(), pd.DataFrame(), notices

    persistent_parts: list[pd.DataFrame] = []
    cf_parts: list[pd.DataFrame] = []
    for action in actions:
        cached = None
        if not getattr(args, "refit_fig5_cache", False):
            cached = load_fig5_km_cache(
                cache_dir,
                test_data_path=Path(args.test_data),
                seed=args.seed,
                max_horizon=max_h,
                action=action,
                preprocessing_path=Path(args.preprocessing),
            )
        if cached is not None:
            notices.append(f"Fig 5 KM loaded from cache ({action})")
            persistent_parts.append(cached[0])
            cf_parts.append(cached[1])
            continue
        print(f"Collecting Fig 5 death KM ({action})...", flush=True)
        persistent, counterfactual = collect_fig5_action_death_km(
            agent,
            test_loader,
            device,
            preprocessing,
            action_name=action,
            max_horizon=max_h,
        )
        if persistent.empty and counterfactual.empty:
            notices.append(f"Fig 5 empty: no death KM rows ({action})")
            continue
        save_fig5_km_cache(
            cache_dir,
            persistent,
            counterfactual,
            test_data_path=Path(args.test_data),
            seed=args.seed,
            max_horizon=max_h,
            action=action,
            preprocessing_path=Path(args.preprocessing),
        )
        notices.append(f"Fig 5 KM cache written ({action})")
        persistent_parts.append(persistent)
        cf_parts.append(counterfactual)
    return _concat_frames(persistent_parts), _concat_frames(cf_parts), notices


def _write_and_plot_fig5(
    *,
    persistent: pd.DataFrame,
    counterfactual: pd.DataFrame,
    source_dir: Path,
    figures_dir: Path,
    dpi: int,
    csv_paths: list[Path],
    figure_paths: list[Path],
) -> None:
    for name, frame in [
        ("fig5_km_persistent.csv", persistent),
        ("fig5_km_counterfactual.csv", counterfactual),
    ]:
        p = write_csv(frame, source_dir / name)
        if p:
            csv_paths.append(p)
    path = plot_fig5(
        persistent=persistent,
        counterfactual=counterfactual,
        figures_dir=figures_dir,
        dpi=dpi,
    )
    if path is not None:
        figure_paths.append(path)


def _fig4_selected_action(args: argparse.Namespace) -> str:
    name = getattr(args, "fig4_action", None) or FIG4_DEFAULT_ACTION
    return str(name)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [f for f in frames if f is not None and not f.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)


def _collect_fig4_validation(
    *,
    agent,
    spec,
    test_loader,
    device: torch.device,
    preprocessing: dict,
    args: argparse.Namespace,
    max_h: int,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load or compute Fig 4 open-loop validation (default: --fig4-action only)."""
    notices: list[str] = []
    available_actions = {a.name for a in spec.action_continuous} | {
        a.name for a in spec.action_categorical
    }
    selected = _fig4_selected_action(args)
    if getattr(args, "fig4_all_actions", False):
        actions = resolve_fig5_simulation_actions(available_actions)
    elif selected in available_actions:
        actions = [selected]
    else:
        actions = []
    validation_outcomes = [
        o for o in FIG5_VALIDATION_OUTCOMES if o in spec.reward_binary
    ]
    if not actions:
        if getattr(args, "fig4_all_actions", False):
            notices.append("Fig 4 skipped: no intervention actions in ModelSpec")
        else:
            notices.append(f"Fig 4 skipped: {selected} not in ModelSpec")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), notices
    if not validation_outcomes:
        notices.append("Fig 4 skipped: outcomes unavailable")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), notices

    obs_parts: list[pd.DataFrame] = []
    sim_parts: list[pd.DataFrame] = []
    cf_obs_parts: list[pd.DataFrame] = []
    cf_sim_parts: list[pd.DataFrame] = []
    for action in actions:
        action_cache = cache_dir / action
        validation_manifest = build_fig5_validation_cache_manifest(
            test_data=Path(args.test_data),
            seed=args.seed,
            action=action,
            max_horizon=max_h,
            outcomes=validation_outcomes,
            preprocessing_path=Path(args.preprocessing),
        )
        cached_validation = (
            None
            if args.refit_fig4_validation_cache
            else try_load_fig5_validation_cache(action_cache, validation_manifest)
        )
        if cached_validation is not None:
            print(f"Reusing Fig 4 cache ({action}): {action_cache}", flush=True)
            obs_parts.append(cached_validation["observed_strata"])
            sim_parts.append(cached_validation["sim_observed_action"])
            cf_obs_parts.append(cached_validation["observed_h1_counterfactual"])
            cf_sim_parts.append(cached_validation["sim_h1_counterfactual"])
            continue
        if action not in available_actions:
            notices.append(f"Fig 4 skipped {action}: not in ModelSpec")
            continue
        print(
            f"Collecting Fig 4 ({action}: persistent + h1-low counterfactual)...",
            flush=True,
        )
        observed, sim_obs = collect_persistent_action_validation(
            agent,
            test_loader,
            device,
            preprocessing,
            action_name=action,
            max_horizon=max_h,
            outcome_names=validation_outcomes,
        )
        cf_obs, cf_sim = collect_h1_low_counterfactual_validation(
            agent,
            test_loader,
            device,
            preprocessing,
            action_name=action,
            max_horizon=max_h,
            outcome_names=validation_outcomes,
            matched_obs_groups=True,
        )
        if observed.empty and sim_obs.empty and cf_obs.empty and cf_sim.empty:
            notices.append(f"Fig 4 empty: {action}")
            continue
        write_fig5_validation_cache(
            action_cache,
            manifest=validation_manifest,
            observed_strata=observed,
            sim_observed_action=sim_obs,
            observed_h1_counterfactual=cf_obs,
            sim_h1_counterfactual=cf_sim,
        )
        obs_parts.append(observed)
        sim_parts.append(sim_obs)
        cf_obs_parts.append(cf_obs)
        cf_sim_parts.append(cf_sim)

    observed_strata_all = _concat_frames(obs_parts)
    sim_observed_action_all = _concat_frames(sim_parts)
    observed_h1_cf_all = _concat_frames(cf_obs_parts)
    sim_h1_cf_all = _concat_frames(cf_sim_parts)
    if (
        observed_strata_all.empty
        and sim_observed_action_all.empty
        and observed_h1_cf_all.empty
        and sim_h1_cf_all.empty
    ):
        notices.append("Fig 4 empty: no action validation rows")
    return (
        observed_strata_all,
        sim_observed_action_all,
        observed_h1_cf_all,
        sim_h1_cf_all,
        notices,
    )


def _validate_exclusive_modes(args: argparse.Namespace) -> None:
    only_modes = [
        name
        for name, active in (
            ("--only-fig4", args.only_fig4),
            ("--only-fig5", args.only_fig5),
            ("--only-figS3", args.only_figS3),
            ("--only-fig6", args.only_fig6),
        )
        if active
    ]
    if len(only_modes) > 1:
        raise SystemExit(
            f"ERROR: {' and '.join(only_modes)} cannot be combined."
        )
    if not only_modes:
        return
    if args.history_only or args.baselines_only:
        raise SystemExit(
            f"ERROR: {only_modes[0]} cannot be combined with "
            "--history-only or --baselines-only."
        )
    if args.figS3_from_cache and not args.only_figS3:
        raise SystemExit("ERROR: --figS3-from-cache requires --only-figS3.")
    if args.refit_figS3_hstar and not args.only_figS3:
        raise SystemExit("ERROR: --refit-figS3-hstar requires --only-figS3.")
    if args.fig6_from_cache and not args.only_fig6:
        raise SystemExit("ERROR: --fig6-from-cache requires --only-fig6.")


def run_fig6_only(args: argparse.Namespace) -> int:
    """Collect/plot Fig 6 case study without the full report pipeline."""
    set_style()
    try:
        cohort = apply_external_cohort(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    if cohort is not None:
        print(
            "WARNING: Fig 6 case study is intended for HRS test; "
            f"continuing on {cohort.label}.",
            flush=True,
        )
    report_kw = report_md_kwargs(args)
    run_dir = Path(args.run_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_dir / "evaluation").resolve()
    )
    figures_dir = output_dir / "figures_report"
    source_dir = output_dir / "source_data_report"
    for p in (output_dir, figures_dir, source_dir):
        p.mkdir(parents=True, exist_ok=True)

    notices: list[str] = ["Fig 6 only (--only-fig6)"]
    if args.fig6_from_cache:
        notices.append("Fig 6 cache mode (--fig6-from-cache)")
    figure_paths: list[Path] = []
    csv_paths: list[Path] = []
    horizons = parse_horizons(args.horizons)
    max_h = max(horizons) if horizons else 5

    if args.fig6_from_cache:
        rollout = read_csv_if_exists(source_dir / "fig6_case_study_rollout.csv")
        person_meta = read_csv_if_exists(source_dir / "fig6_case_study_person.csv")
        if rollout.empty:
            print("ERROR: missing fig6_case_study_rollout.csv", flush=True)
            return 1
        fig_path = plot_fig6_case_study(
            rollout=rollout,
            person_meta=person_meta,
            figures_dir=figures_dir,
            dpi=args.dpi,
        )
        if fig_path is not None:
            figure_paths.append(fig_path)
        write_report_md(
            output_dir / "report_summary.md",
            checkpoint=run_dir / "best_final.pt",
            notices=notices,
            figure_paths=figure_paths,
            csv_paths=csv_paths,
            clinical=pd.DataFrame(),
            level_cmp=pd.DataFrame(),
            title="# Health world model Fig 6 case study only",
            **{k: v for k, v in report_kw.items() if k != "title"},
        )
        print(f"Done. Fig 6 under {figures_dir}", flush=True)
        for p in figure_paths:
            print(p, flush=True)
        return 0

    try:
        checkpoint = selected_checkpoint(run_dir, args.checkpoint)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1

    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print("Fig 6 only: case study rollout; skip Fig 1/2/3/4/5/7.", flush=True)

    agent, spec, cfg = load_agent(checkpoint, device)
    preprocessing = json.loads(Path(args.preprocessing).read_text(encoding="utf-8"))
    bs = int(args.batch_size or cfg.get("data", {}).get("batch_size", 128))
    test_loader = build_loader(
        table_path=Path(args.test_data),
        spec=spec,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    case_batch, case_pi, person_meta_df, case_meta = pick_case_study_person(
        agent,
        test_loader,
        device,
        person_id=args.fig6_person_id,
        exclude_person_id=args.fig6_exclude_person_id,
        seed=args.seed,
        preprocessing=preprocessing,
    )
    if case_batch is None or case_pi is None:
        print("ERROR: no suitable case-study person found in test data.", flush=True)
        print(
            "  Required profile: age 50–65, hypertension, no antihypertensive at baseline, "
            "smoker, light activity = never.",
            flush=True,
        )
        return 1

    origin_step = int(case_meta.get("origin_step", 0))
    rollout = collect_case_study_rollout(
        agent,
        case_batch,
        case_pi,
        device,
        max_horizon=FIG6_CASE_STUDY_MAX_HORIZON,
        origin_step=origin_step,
        preprocessing=preprocessing,
    )
    if rollout.empty:
        print("ERROR: Fig 6 rollout empty.", flush=True)
        return 1

    p_meta = write_csv(person_meta_df, source_dir / "fig6_case_study_person.csv")
    p_roll = write_csv(rollout, source_dir / "fig6_case_study_rollout.csv")
    if p_meta:
        csv_paths.append(p_meta)
    if p_roll:
        csv_paths.append(p_roll)

    fig_path = plot_fig6_case_study(
        rollout=rollout,
        person_meta=person_meta_df,
        figures_dir=figures_dir,
        dpi=args.dpi,
    )
    if fig_path is not None:
        figure_paths.append(fig_path)

    write_report_md(
        output_dir / "report_summary.md",
        checkpoint=checkpoint,
        notices=notices,
        figure_paths=figure_paths,
        csv_paths=csv_paths,
        clinical=pd.DataFrame(),
        level_cmp=pd.DataFrame(),
        title="# Health world model Fig 6 case study only",
        **{k: v for k, v in report_kw.items() if k != "title"},
    )
    print(f"Selected person: {person_meta_df.iloc[0]['person_id']}", flush=True)
    print(f"Done. Fig 6 under {figures_dir}", flush=True)
    for p in figure_paths:
        print(p, flush=True)
    return 0


def run_figS3_only(args: argparse.Namespace) -> int:
    """Collect/plot supplementary Fig S3 without the full report pipeline."""
    set_style()
    try:
        cohort = apply_external_cohort(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    report_kw = report_md_kwargs(args)
    run_dir = Path(args.run_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_dir / "evaluation").resolve()
    )
    figures_dir = output_dir / "figures_report"
    source_dir = output_dir / "source_data_report"
    baseline_dir = (
        Path(args.baseline_dir).resolve()
        if args.baseline_dir
        else (output_dir / "baseline_cache")
    )
    for p in (output_dir, figures_dir, source_dir, baseline_dir):
        p.mkdir(parents=True, exist_ok=True)

    notices: list[str] = ["Fig S3 only (--only-figS3)"]
    if args.figS3_from_cache:
        notices.append("Fig S3 cache mode (--figS3-from-cache)")
    figure_paths: list[Path] = []
    csv_paths: list[Path] = []
    horizons = parse_horizons(args.horizons)
    max_h = max(horizons) if horizons else 5

    if cohort is not None:
        notices.append(cohort.note)
        print(f"External validation: {cohort.label}", flush=True)
        print(f"Test table: {Path(args.test_data).resolve()}", flush=True)
        print(f"Output: {output_dir}", flush=True)

    spec_path = run_dir / "model_spec.json"
    spec = ModelSpec.load(spec_path) if spec_path.exists() else None
    allowed_features = (
        list(spec.state_names) + list(spec.static_names) + list(spec.action_names)
        if spec is not None
        else None
    )
    pos_max = (
        float(args.binary_pos_weight_max)
        if args.binary_pos_weight_max is not None
        else DEFAULT_BINARY_POS_WEIGHT_MAX
    )

    if args.figS3_from_cache:
        wors_dyn = read_csv_if_exists(source_dir / "figS3_worsening_openloop_dynamics.csv")
        clin_dyn = read_csv_if_exists(source_dir / "figS3_clinical_openloop_dynamics.csv")
        wors_roll = read_csv_if_exists(source_dir / "figS3_event_rollout_rows_worsening.csv")
        clin_roll = read_csv_if_exists(source_dir / "figS3_event_rollout_rows_clinical.csv")
        need_rebuild = (
            args.refit_figS3_hstar
            or wors_dyn.empty
            or clin_dyn.empty
        )
        if need_rebuild:
            if spec is None:
                print(f"ERROR: missing {spec_path} for MLP-h* rebuild.", flush=True)
                return 1
            print("Rebuilding Fig S3 MLP-h* metrics from cached rollout rows...", flush=True)
            train_table = read_table(Path(args.train_data))
            test_table = read_table(Path(args.test_data))
            (
                wors_roll,
                clin_roll,
                wors_dyn,
                clin_dyn,
            ) = _rebuild_figS3_hstar_from_rollout_csvs(
                source_dir=source_dir,
                train_table=train_table,
                test_table=test_table,
                baseline_dir=baseline_dir,
                spec=spec,
                seed=args.seed,
                max_h=max_h,
                pos_max=pos_max,
                allowed_features=allowed_features,
                refit=args.refit_figS3_hstar or args.refit_baselines,
            )
        wors_risk, clin_risk, km = _load_figS3_risk_km_from_cache(source_dir)
        if wors_dyn.empty and clin_dyn.empty:
            print("ERROR: Fig S3 dynamics unavailable from cache.", flush=True)
            return 1
        csv_paths.extend(
            _write_figS3_csvs(
                source_dir,
                worsening_roll=wors_roll,
                event_roll=clin_roll,
                worsening_dyn_hstar=wors_dyn,
                event_dyn_hstar=clin_dyn,
            )
        )
        fig_path = _plot_figS3_bundle(
            worsening_dyn_hstar=wors_dyn,
            event_dyn_hstar=clin_dyn,
            km=km,
            worsening_risk_summary=wors_risk,
            acute_risk_summary=clin_risk,
            figures_dir=figures_dir,
            dpi=args.dpi,
        )
        if fig_path is not None:
            figure_paths.append(fig_path)
        if cohort is not None:
            report_kw["title"] = f"# Health world model Fig S3 only ({cohort.label})"
        else:
            report_kw["title"] = "# Health world model Fig S3 only"
        write_report_md(
            output_dir / "report_summary.md",
            checkpoint=run_dir / "best_final.pt",
            notices=notices,
            figure_paths=figure_paths,
            csv_paths=csv_paths,
            clinical=pd.DataFrame(),
            level_cmp=pd.DataFrame(),
            **report_kw,
        )
        print(f"Done. Fig S3 under {figures_dir}", flush=True)
        for p in figure_paths:
            print(p, flush=True)
        return 0

    try:
        checkpoint = selected_checkpoint(run_dir, args.checkpoint)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1

    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print("Fig S3 only: skipping Fig 1/2/4/5/7.", flush=True)

    agent, spec, cfg = load_agent(checkpoint, device)
    preprocessing = json.loads(Path(args.preprocessing).read_text(encoding="utf-8"))
    bs = int(args.batch_size or cfg.get("data", {}).get("batch_size", 128))
    if args.binary_pos_weight_max is not None:
        pos_max = float(args.binary_pos_weight_max)
    else:
        pos_max = float(
            (cfg.get("world_loss") or {}).get(
                "binary_reward_pos_weight_max",
                DEFAULT_BINARY_POS_WEIGHT_MAX,
            )
        )
    allowed_features = (
        list(spec.state_names) + list(spec.static_names) + list(spec.action_names)
    )

    baseline, _persistence, _linear, mlp_tabular, _manifest = prepare_linear_and_persistence(
        train_data=Path(args.train_data),
        test_data=Path(args.test_data),
        preprocessing_path=Path(args.preprocessing),
        preprocessing=preprocessing,
        baseline_dir=baseline_dir,
        seed=args.seed,
        skip_linear_baseline=args.skip_linear_baseline,
        refit_baselines=args.refit_baselines or args.refit_figS3_hstar,
        event_names=list(spec.reward_binary),
        output_dir=output_dir,
        skip_mlp=args.skip_mlp,
        level_names=list(spec.reward_continuous),
        binary_pos_weight_max=pos_max,
        allowed_feature_names=allowed_features,
    )
    if args.skip_mlp:
        notices.append("MLP baselines skipped (--skip-mlp): Fig S3 needs tabular MLP-h*")

    test_loader = build_loader(
        table_path=Path(args.test_data),
        spec=spec,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_table = read_table(Path(args.train_data))
    test_table = read_table(Path(args.test_data))
    event_names_all = _figS3_event_names(spec)

    print("Collecting Fig S3 MLP-h* open-loop dynamics...", flush=True)
    (
        worsening_roll,
        event_roll,
        worsening_dyn_hstar,
        event_dyn_hstar,
        hstar_notices,
    ) = _collect_figS3_hstar_dynamics(
        agent,
        test_loader,
        device,
        spec=spec,
        baseline=baseline,
        mlp_tabular=mlp_tabular,
        train_table=train_table,
        test_table=test_table,
        baseline_dir=baseline_dir,
        seed=args.seed,
        max_h=max_h,
        pos_max=pos_max,
        allowed_features=allowed_features,
        refit_baselines=args.refit_baselines,
        refit_figS3_hstar=args.refit_figS3_hstar,
        skip_mlp=args.skip_mlp,
    )
    notices.extend(hstar_notices)

    print("Collecting Fig S3 risk / KM panels...", flush=True)
    test_loader = build_loader(
        table_path=Path(args.test_data),
        spec=spec,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    (
        _event_risk,
        _worsening_risk_rows,
        _acute_risk_rows,
        worsening_risk_summary,
        acute_risk_summary,
        km,
        risk_notices,
    ) = _collect_openloop_risk_km(
        agent,
        test_loader,
        device,
        spec,
        event_names_all=event_names_all,
        max_h=max_h,
    )
    notices.extend(risk_notices)

    csv_paths.extend(
        _write_figS3_csvs(
            source_dir,
            worsening_roll=worsening_roll,
            event_roll=event_roll,
            worsening_dyn_hstar=worsening_dyn_hstar,
            event_dyn_hstar=event_dyn_hstar,
            worsening_risk_summary=worsening_risk_summary,
            acute_risk_summary=acute_risk_summary,
            km=km,
        )
    )
    fig_path = _plot_figS3_bundle(
        worsening_dyn_hstar=worsening_dyn_hstar,
        event_dyn_hstar=event_dyn_hstar,
        km=km,
        worsening_risk_summary=worsening_risk_summary,
        acute_risk_summary=acute_risk_summary,
        figures_dir=figures_dir,
        dpi=args.dpi,
    )
    if fig_path is not None:
        figure_paths.append(fig_path)

    if cohort is not None:
        report_kw["title"] = f"# Health world model Fig S3 only ({cohort.label})"
    else:
        report_kw["title"] = "# Health world model Fig S3 only"
    write_report_md(
        output_dir / "report_summary.md",
        checkpoint=checkpoint,
        notices=notices,
        figure_paths=figure_paths,
        csv_paths=csv_paths,
        clinical=pd.DataFrame(),
        level_cmp=pd.DataFrame(),
        **report_kw,
    )
    print(f"Done. Fig S3 under {figures_dir}", flush=True)
    for p in figure_paths:
        print(p, flush=True)
    return 0


def run_fig4_only(args: argparse.Namespace) -> int:
    """Collect/plot Fig 4 without running the full Fig 1–3 pipeline."""
    set_style()
    try:
        cohort = apply_external_cohort(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    report_kw = report_md_kwargs(args)
    run_dir = Path(args.run_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_dir / "evaluation").resolve()
    )
    figures_dir = output_dir / "figures_report"
    source_dir = output_dir / "source_data_report"
    fig4_cache_dir = _fig4_validation_cache_dir(args, output_dir)
    for p in (output_dir, figures_dir, source_dir, fig4_cache_dir):
        p.mkdir(parents=True, exist_ok=True)

    notices: list[str] = ["Fig 4 only (--only-fig4)"]
    figure_paths: list[Path] = []
    csv_paths: list[Path] = []
    horizons = parse_horizons(args.horizons)
    max_h = max(horizons) if horizons else 5

    if cohort is not None:
        notices.append(cohort.note)
        print(f"External validation: {cohort.label}", flush=True)
        print(f"Test table: {Path(args.test_data).resolve()}", flush=True)
        print(f"Output: {output_dir}", flush=True)

    try:
        checkpoint = selected_checkpoint(run_dir, args.checkpoint)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1

    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print("Fig 4 only: skipping baselines and Fig 1/3/2/7.", flush=True)

    agent, spec, cfg = load_agent(checkpoint, device)
    preprocessing = json.loads(Path(args.preprocessing).read_text(encoding="utf-8"))
    bs = int(args.batch_size or cfg.get("data", {}).get("batch_size", 128))
    test_loader = build_loader(
        table_path=Path(args.test_data),
        spec=spec,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    (
        observed_strata_all,
        sim_observed_action_all,
        observed_h1_cf_all,
        sim_h1_cf_all,
        val_notices,
    ) = _collect_fig4_validation(
        agent=agent,
        spec=spec,
        test_loader=test_loader,
        device=device,
        preprocessing=preprocessing,
        args=args,
        max_h=max_h,
        cache_dir=fig4_cache_dir,
    )
    notices.extend(val_notices)

    for name, frame in [
        ("fig4_observed_action_strata.csv", observed_strata_all),
        ("fig4_sim_observed_action.csv", sim_observed_action_all),
        ("fig4_observed_h1_counterfactual.csv", observed_h1_cf_all),
        ("fig4_sim_h1_counterfactual.csv", sim_h1_cf_all),
    ]:
        p = merge_fig4_action_csv(source_dir / name, frame)
        if p:
            csv_paths.append(p)

    figure_paths.extend(
        plot_fig4_all_actions(
            observed_strata=observed_strata_all,
            sim_observed_action=sim_observed_action_all,
            observed_h1_counterfactual=observed_h1_cf_all,
            sim_h1_counterfactual=sim_h1_cf_all,
            figures_dir=figures_dir,
            dpi=args.dpi,
            default_action=_fig4_selected_action(args),
            all_actions=bool(getattr(args, "fig4_all_actions", False)),
        )
    )
    figure_paths = [p for p in figure_paths if p is not None]

    if cohort is not None:
        report_kw["title"] = f"# Health world model Fig 4 only ({cohort.label})"
    else:
        report_kw["title"] = "# Health world model Fig 4 only"
    write_report_md(
        output_dir / "report_summary.md",
        checkpoint=checkpoint,
        notices=notices,
        figure_paths=figure_paths,
        csv_paths=csv_paths,
        clinical=pd.DataFrame(),
        level_cmp=pd.DataFrame(),
        **report_kw,
    )
    print(f"Done. Fig 4 under {figures_dir}", flush=True)
    for p in figure_paths:
        print(p, flush=True)
    return 0


run_fig5_validation_only = run_fig4_only  # backward-compatible alias


def run_fig5_only(args: argparse.Namespace) -> int:
    """Collect/plot Fig 5 death KM for every Fig 4 intervention."""
    set_style()
    try:
        cohort = apply_external_cohort(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    report_kw = report_md_kwargs(args)
    run_dir = Path(args.run_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_dir / "evaluation").resolve()
    )
    figures_dir = output_dir / "figures_report"
    source_dir = output_dir / "source_data_report"
    fig5_cache_dir = _fig5_km_cache_dir(args, output_dir)
    for p in (output_dir, figures_dir, source_dir, fig5_cache_dir):
        p.mkdir(parents=True, exist_ok=True)

    notices: list[str] = ["Fig 5 only (--only-fig5)"]
    figure_paths: list[Path] = []
    csv_paths: list[Path] = []
    horizons = parse_horizons(args.horizons)
    max_h = max(horizons) if horizons else 5

    if cohort is not None:
        notices.append(cohort.note)
        print(f"External validation: {cohort.label}", flush=True)
        print(f"Test table: {Path(args.test_data).resolve()}", flush=True)
        print(f"Output: {output_dir}", flush=True)

    try:
        checkpoint = selected_checkpoint(run_dir, args.checkpoint)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1

    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print("Fig 5 only: death KM for all interventions; skip Fig 1/3/4/2/7.", flush=True)

    agent, spec, cfg = load_agent(checkpoint, device)
    preprocessing = json.loads(Path(args.preprocessing).read_text(encoding="utf-8"))
    bs = int(args.batch_size or cfg.get("data", {}).get("batch_size", 128))
    test_loader = build_loader(
        table_path=Path(args.test_data),
        spec=spec,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    persistent, counterfactual, km_notices = _collect_fig5_km(
        agent=agent,
        spec=spec,
        test_loader=test_loader,
        device=device,
        preprocessing=preprocessing,
        args=args,
        max_h=max_h,
        cache_dir=fig5_cache_dir,
    )
    notices.extend(km_notices)
    _write_and_plot_fig5(
        persistent=persistent,
        counterfactual=counterfactual,
        source_dir=source_dir,
        figures_dir=figures_dir,
        dpi=args.dpi,
        csv_paths=csv_paths,
        figure_paths=figure_paths,
    )
    figure_paths = [p for p in figure_paths if p is not None]

    if cohort is not None:
        report_kw["title"] = f"# Health world model Fig 5 only ({cohort.label})"
    else:
        report_kw["title"] = "# Health world model Fig 5 only"
    write_report_md(
        output_dir / "report_summary.md",
        checkpoint=checkpoint,
        notices=notices,
        figure_paths=figure_paths,
        csv_paths=csv_paths,
        clinical=pd.DataFrame(),
        level_cmp=pd.DataFrame(),
        **report_kw,
    )
    print(f"Done. Fig 5 under {figures_dir}", flush=True)
    for p in figure_paths:
        print(p, flush=True)
    return 0


def run_history_only(run_dir: Path, output_dir: Path, dpi: int) -> int:
    figures_dir = output_dir / "figures_report"
    source_dir = output_dir / "source_data_report"
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    history = load_training_history_curves(run_dir)
    if history.empty:
        print(f"No usable training_history.csv under {run_dir}", flush=True)
        return 1
    write_csv(fig1c_export_table(history), source_dir / "fig1c_training_history.csv")
    path = plot_fig1(
        latent_summary=pd.DataFrame(),
        history=history,
        variance=pd.DataFrame(),
        figures_dir=figures_dir,
        dpi=dpi,
    )
    write_report_md(
        output_dir / "report_summary.md",
        checkpoint=None,
        notices=["history-only mode: only Fig 1C populated"],
        figure_paths=[path] if path else [],
        csv_paths=list(source_dir.glob("*.csv")),
        clinical=pd.DataFrame(),
        level_cmp=pd.DataFrame(),
    )
    return 0


def run_report(args: argparse.Namespace) -> int:
    set_style()
    try:
        cohort = apply_external_cohort(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    report_kw = report_md_kwargs(args)
    run_dir = Path(args.run_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_dir / "evaluation").resolve()
    )
    figures_dir = output_dir / "figures_report"
    source_dir = output_dir / "source_data_report"
    baseline_dir = (
        Path(args.baseline_dir).resolve()
        if args.baseline_dir
        else (output_dir / "baseline_cache")
    )
    fig2_cache_dir = (
        Path(args.fig2_cache_dir).resolve()
        if args.fig2_cache_dir
        else (output_dir / "fig2_cache")
    )
    fig4_validation_cache_dir = _fig4_validation_cache_dir(args, output_dir)
    fig5_km_cache_dir = _fig5_km_cache_dir(args, output_dir)
    for p in (
        output_dir,
        figures_dir,
        source_dir,
        baseline_dir,
        fig2_cache_dir,
        fig4_validation_cache_dir,
        fig5_km_cache_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)

    notices: list[str] = []
    figure_paths: list[Path] = []
    csv_paths: list[Path] = []
    horizons = parse_horizons(args.horizons)
    if cohort is not None:
        notices.append(cohort.note)
        print(f"External validation: {cohort.label}", flush=True)
        print(f"Test table: {Path(args.test_data).resolve()}", flush=True)
        print(f"Output: {output_dir}", flush=True)
        meta = {
            "role": "health_world_model_external_validation",
            "cohort": cohort.name,
            "label": cohort.label,
            "test_data": str(Path(args.test_data).resolve()),
            "train_data": str(Path(args.train_data).resolve()),
            "preprocessing": str(Path(args.preprocessing).resolve()),
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "note": cohort.note,
        }
        (output_dir / "external_eval_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if args.baselines_only:
        pos_max = (
            float(args.binary_pos_weight_max)
            if args.binary_pos_weight_max is not None
            else DEFAULT_BINARY_POS_WEIGHT_MAX
        )
        allowed_features = None
        spec_path = run_dir / "model_spec.json"
        if spec_path.exists():
            baseline_spec = ModelSpec.load(spec_path)
            allowed_features = (
                list(baseline_spec.state_names)
                + list(baseline_spec.static_names)
                + list(baseline_spec.action_names)
            )
            print(
                f"Baseline features restricted by {spec_path.name} "
                f"({len(allowed_features)} names)",
                flush=True,
            )
        prepare_baselines(
            train_data=Path(args.train_data).resolve(),
            test_data=Path(args.test_data).resolve(),
            preprocessing_path=Path(args.preprocessing).resolve(),
            baseline_dir=baseline_dir,
            seed=args.seed,
            skip_linear_baseline=args.skip_linear_baseline,
            binary_pos_weight_max=pos_max,
            allowed_feature_names=allowed_features,
        )
        print(str((baseline_dir / "linear_baseline_predictions.parquet").resolve()))
        if not args.skip_mlp:
            preprocessing = json.loads(
                Path(args.preprocessing).read_text(encoding="utf-8")
            )
            train = read_table(Path(args.train_data))
            test = read_table(Path(args.test_data))
            load_or_fit_mlp_tabular(
                train=train,
                test=test,
                preprocessing=preprocessing,
                baseline_dir=baseline_dir,
                seed=args.seed,
                event_names=discover_binary_event_names(
                    train.columns, preferred=list(EVENT_LABELS)
                ),
                refit=args.refit_baselines,
                skip=False,
                binary_pos_weight_max=pos_max,
                allowed_feature_names=allowed_features,
            )
            print(str(mlp_tabular_cache_paths(baseline_dir)["predictions"].resolve()))
            horizons_fit = parse_horizons(args.horizons)
            max_h_fit = max(horizons_fit) if horizons_fit else 5
            load_or_fit_mlp_horizon_star(
                train=train,
                test=test,
                baseline_dir=baseline_dir,
                seed=args.seed,
                event_names=discover_binary_event_names(
                    train.columns, preferred=list(EVENT_LABELS)
                ),
                max_horizon=max_h_fit,
                refit=args.refit_baselines,
                skip=False,
                binary_pos_weight_max=pos_max,
                allowed_feature_names=allowed_features,
            )
            print(str(mlp_horizon_star_cache_paths(baseline_dir)["predictions"].resolve()))
        return 0

    if args.history_only:
        return run_history_only(run_dir, output_dir, args.dpi)

    # Checkpoint
    try:
        checkpoint = selected_checkpoint(run_dir, args.checkpoint)
    except FileNotFoundError as exc:
        skip_notice(str(exc))
        notices.append(str(exc))
        print("Falling back to --history-only (Fig 1C).", flush=True)
        return run_history_only(run_dir, output_dir, args.dpi)

    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)

    agent, spec, cfg = load_agent(checkpoint, device)
    preprocessing = json.loads(Path(args.preprocessing).read_text(encoding="utf-8"))
    bs = int(args.batch_size or cfg.get("data", {}).get("batch_size", 128))
    if args.binary_pos_weight_max is not None:
        pos_max = float(args.binary_pos_weight_max)
    else:
        pos_max = float(
            (cfg.get("world_loss") or {}).get(
                "binary_reward_pos_weight_max",
                DEFAULT_BINARY_POS_WEIGHT_MAX,
            )
        )

    test_loader = build_loader(
        table_path=Path(args.test_data),
        spec=spec,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    # The train-side loader is only needed by the Fig 2 probe (cache miss path);
    # it is built there so its trajectory cache is not resident for the whole run.

    # Baselines (Linear Ridge / Logistic + tabular MLP on same features)
    print(
        f"Baseline binary pos-class weight max: {pos_max} "
        "(Logistic class_weight / MLP positive-row oversample)",
        flush=True,
    )
    baseline, persistence, linear, mlp_tabular, _manifest = prepare_linear_and_persistence(
        train_data=Path(args.train_data),
        test_data=Path(args.test_data),
        preprocessing_path=Path(args.preprocessing),
        preprocessing=preprocessing,
        baseline_dir=baseline_dir,
        seed=args.seed,
        skip_linear_baseline=args.skip_linear_baseline,
        refit_baselines=args.refit_baselines,
        event_names=list(spec.reward_binary),
        output_dir=output_dir,
        skip_mlp=args.skip_mlp,
        level_names=list(spec.reward_continuous),
        binary_pos_weight_max=pos_max,
        allowed_feature_names=(
            list(spec.state_names) + list(spec.static_names) + list(spec.action_names)
        ),
    )
    if args.skip_mlp:
        notices.append("MLP baselines skipped (--skip-mlp): no tabular MLP (Fig 3/4)")

    # The world-model MLP baseline was dropped: its latent error lived in its
    # own embedding space against its own target, so it was not comparable with
    # JEPA. Cross-model comparison now happens in observable units (Fig 3A/4).
    # Tabular MLP (sklearn) remains the Fig 3A / Fig 4 outcome baseline.
    # Fig 2A/B use JEPA frozen probes only.

    # ----- Fig 1 -----
    print("Collecting JEPA latent prediction metrics...", flush=True)
    latent_all = collect_jepa_latent_prediction(
        agent, test_loader, device, horizons=horizons
    )
    latent_summary = summarize_latent_by_horizon(latent_all)
    history = load_training_history_curves(run_dir)
    if history.empty:
        notices.append(
            "training_history.csv missing or lacks train/val world_loss columns (Fig 1C empty)"
        )
    variance = collect_latent_variance(agent, test_loader, device)
    for name, frame in [
        ("fig1_latent_prediction_rows.csv", latent_all),
        ("fig1_latent_summary.csv", latent_summary),
        ("fig1c_training_history.csv", fig1c_export_table(history)),
        ("fig1d_latent_variance.csv", variance),
    ]:
        p = write_csv(frame, source_dir / name)
        if p:
            csv_paths.append(p)
    figure_paths.append(
        plot_fig1(
            latent_summary=latent_summary,
            history=history,
            variance=variance,
            figures_dir=figures_dir,
            dpi=args.dpi,
        )
    )

    # Fig 2 (health probing) is deferred to the end — heaviest / OOM-prone step.

    # ----- Fig 3 (worsening events) + shared one-step / open-loop events -----
    print("Collecting one-step event predictions...", flush=True)
    test_loader = build_loader(
        table_path=Path(args.test_data),
        spec=spec,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    levels_jepa, events_jepa = collect_one_step_predictions(
        agent, test_loader, preprocessing, device
    )
    levels_jepa, events_jepa = attach_baselines(levels_jepa, events_jepa, baseline)
    mlp_events = (
        mlp_events_long_from_wide(mlp_tabular, events_jepa)
        if not mlp_tabular.empty
        else pd.DataFrame()
    )
    clinical_all = clinical_compare_table(events_jepa, mlp_events, baseline)
    worsening_cmp = _subset_named(clinical_all, list(WORSENING_REWARDS))
    clinical = _subset_named(clinical_all, list(CLINICAL_EVENT_REWARDS))
    include_cognition = bool(spec.reward_continuous)
    level_mse = (
        level_compare_metrics(
            levels_jepa,
            persistence,
            linear,
            mlp_tabular,
        )
        if include_cognition
        else pd.DataFrame()
    )
    level_cmp = worsening_cmp
    max_h = max(horizons) if horizons else 5

    examples = pd.DataFrame()
    if not args.skip_fig7:
        print("Collecting individual trajectories (Fig 7)...", flush=True)
        test_loader = build_loader(
            table_path=Path(args.test_data),
            spec=spec,
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        traj_pool = collect_one_step_person_trajectories(
            agent, test_loader, device, preprocessing, outcome=PRIMARY_LEVELS
        )
        examples = pick_example_trajectories(
            traj_pool,
            outcome="adl_worsening",
            max_persons=args.max_persons_traj,
            seed=args.seed,
            keep_all_outcomes=True,
        )

    test_loader = build_loader(
        table_path=Path(args.test_data),
        spec=spec,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    latent_roll = collect_latent_rollout_mse(agent, test_loader, device, max_horizon=max_h)

    level_openloop = pd.DataFrame()
    if include_cognition:
        print("Collecting open-loop continuous levels...", flush=True)
        test_loader = build_loader(
            table_path=Path(args.test_data),
            spec=spec,
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        level_roll = collect_jepa_level_rollouts(
            agent, test_loader, device, preprocessing, max_horizon=max_h
        )
        level_openloop = summarize_level_openloop_mse(level_roll)
        if level_openloop.empty:
            notices.append("Open-loop continuous-level MSE unavailable")

    print("Collecting open-loop event dynamics (worsening + acute)...", flush=True)
    test_loader = build_loader(
        table_path=Path(args.test_data),
        spec=spec,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    event_names_all = _figS3_event_names(spec)
    print("Fitting MLP-h* baselines (direct t0→t+h)...", flush=True)
    allowed_features = (
        list(spec.state_names) + list(spec.static_names) + list(spec.action_names)
    )
    train_table = read_table(Path(args.train_data))
    test_table = read_table(Path(args.test_data))
    (
        worsening_roll,
        event_roll,
        worsening_dyn_hstar,
        event_dyn_hstar,
        figS3_notices,
    ) = _collect_figS3_hstar_dynamics(
        agent,
        test_loader,
        device,
        spec=spec,
        baseline=baseline,
        mlp_tabular=mlp_tabular,
        train_table=train_table,
        test_table=test_table,
        baseline_dir=baseline_dir,
        seed=args.seed,
        max_h=max_h,
        pos_max=pos_max,
        allowed_features=allowed_features,
        refit_baselines=args.refit_baselines,
        refit_figS3_hstar=args.refit_figS3_hstar,
        skip_mlp=args.skip_mlp,
    )
    notices.extend(figS3_notices)
    worsening_dyn = summarize_event_openloop_dynamics(worsening_roll)
    event_dyn = summarize_event_openloop_dynamics(event_roll)
    if worsening_dyn.empty:
        dyn_panel = "3E–H" if include_cognition else "3C–F"
        notices.append(
            f"Fig {dyn_panel} empty: open-loop worsening dynamics unavailable"
        )
    if event_dyn.empty:
        notices.append("Fig 4E/F empty: open-loop event dynamics unavailable")
    if worsening_dyn_hstar.empty and event_dyn_hstar.empty:
        notices.append("Fig S3 empty: MLP-h* open-loop dynamics unavailable")

    print("Collecting open-loop event risk (Fig 3/4/5)...", flush=True)
    test_loader = build_loader(
        table_path=Path(args.test_data),
        spec=spec,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    (
        event_risk,
        worsening_risk_rows,
        acute_risk_rows,
        worsening_risk_summary,
        acute_risk_summary,
        km,
        risk_km_notices,
    ) = _collect_openloop_risk_km(
        agent,
        test_loader,
        device,
        spec,
        event_names_all=event_names_all,
        max_h=max_h,
    )
    notices.extend(risk_km_notices)

    fig3_csvs: list[tuple[str, pd.DataFrame]] = [
        ("fig3_worsening_metrics.csv", worsening_cmp),
        ("fig3_latent_rollout_mse.csv", latent_roll),
        ("fig3_event_rollout_rows.csv", worsening_roll),
        ("fig3_openloop_dynamics.csv", worsening_dyn),
        ("fig3_openloop_risk_rows.csv", worsening_risk_rows),
        ("fig3_openloop_risk_summary.csv", worsening_risk_summary),
        ("fig3_kaplan_meier.csv", km),
        ("fig3_event_openloop_dynamics.csv", event_dyn),
        ("fig3_clinical_openloop_risk_summary.csv", acute_risk_summary),
        ("figS3_worsening_openloop_dynamics.csv", worsening_dyn_hstar),
        ("figS3_clinical_openloop_dynamics.csv", event_dyn_hstar),
        ("figS3_event_rollout_rows_worsening.csv", worsening_roll),
        ("figS3_event_rollout_rows_clinical.csv", event_roll),
    ]
    if not args.skip_fig7:
        fig3_csvs.append(("fig7_example_trajectories.csv", examples))
    if include_cognition:
        fig3_csvs.insert(1, ("fig3_level_mse.csv", level_mse))
        fig3_csvs.insert(2, ("fig3_level_openloop_mse.csv", level_openloop))
    for name, frame in fig3_csvs:
        p = write_csv(frame, source_dir / name)
        if p:
            csv_paths.append(p)

    figure_paths.append(
        plot_fig3(
            worsening_dynamics=worsening_dyn,
            clinical_dynamics=event_dyn,
            km=km,
            worsening_risk_summary=worsening_risk_summary,
            clinical_risk_summary=acute_risk_summary,
            level_openloop=level_openloop,
            include_cognition=include_cognition,
            figures_dir=figures_dir,
            dpi=args.dpi,
        )
    )
    figure_paths.append(
        plot_figS3_dynamic(
            worsening_dynamics=worsening_dyn_hstar,
            clinical_dynamics=event_dyn_hstar,
            km=km,
            worsening_risk_summary=worsening_risk_summary,
            clinical_risk_summary=acute_risk_summary,
            figures_dir=figures_dir,
            dpi=args.dpi,
        )
    )

    # ----- Fig 4: smoking open-loop validation -----
    observed_strata_all = pd.DataFrame()
    sim_observed_action_all = pd.DataFrame()
    observed_h1_cf_all = pd.DataFrame()
    sim_h1_cf_all = pd.DataFrame()
    (
        observed_strata_all,
        sim_observed_action_all,
        observed_h1_cf_all,
        sim_h1_cf_all,
        fig4_notices,
    ) = _collect_fig4_validation(
        agent=agent,
        spec=spec,
        test_loader=test_loader,
        device=device,
        preprocessing=preprocessing,
        args=args,
        max_h=max_h,
        cache_dir=fig4_validation_cache_dir,
    )
    notices.extend(fig4_notices)

    for name, frame in [
        ("fig4_observed_action_strata.csv", observed_strata_all),
        ("fig4_sim_observed_action.csv", sim_observed_action_all),
        ("fig4_observed_h1_counterfactual.csv", observed_h1_cf_all),
        ("fig4_sim_h1_counterfactual.csv", sim_h1_cf_all),
    ]:
        p = write_csv(frame, source_dir / name)
        if p:
            csv_paths.append(p)

    figure_paths.extend(
        plot_fig4_all_actions(
            observed_strata=observed_strata_all,
            sim_observed_action=sim_observed_action_all,
            observed_h1_counterfactual=observed_h1_cf_all,
            sim_h1_counterfactual=sim_h1_cf_all,
            figures_dir=figures_dir,
            dpi=args.dpi,
            default_action=_fig4_selected_action(args),
            all_actions=bool(getattr(args, "fig4_all_actions", False)),
        )
    )

    # ----- Fig 5: smoking-group death KM -----
    if args.skip_fig5:
        notices.append("Fig 5 skipped (--skip-fig5)")
    else:
        persistent_km, cf_km, fig5_notices = _collect_fig5_km(
            agent=agent,
            spec=spec,
            test_loader=test_loader,
            device=device,
            preprocessing=preprocessing,
            args=args,
            max_h=max_h,
            cache_dir=fig5_km_cache_dir,
        )
        notices.extend(fig5_notices)
        _write_and_plot_fig5(
            persistent=persistent_km,
            counterfactual=cf_km,
            source_dir=source_dir,
            figures_dir=figures_dir,
            dpi=args.dpi,
            csv_paths=csv_paths,
            figure_paths=figure_paths,
        )

    # ----- Fig 7: individual worsening trajectories (optional) -----
    if not args.skip_fig7:
        figure_paths.append(
            plot_fig7(
                examples=examples,
                figures_dir=figures_dir,
                dpi=args.dpi,
            )
        )
    else:
        notices.append("Fig 7 skipped (--skip-fig7; use --no-skip-fig7 to enable)")

    # ----- Fig 6: single-person case study (world-model application) -----
    print("Collecting Fig 6 case study...", flush=True)
    test_loader = build_loader(
        table_path=Path(args.test_data),
        spec=spec,
        batch_size=bs,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    case_batch, case_pi, case_person_df, case_meta = pick_case_study_person(
        agent,
        test_loader,
        device,
        person_id=args.fig6_person_id,
        exclude_person_id=args.fig6_exclude_person_id,
        seed=args.seed,
        preprocessing=preprocessing,
    )
    if case_batch is None or case_pi is None:
        notices.append("Fig 6 empty: no suitable case-study person in HRS test")
    else:
        origin_step = int(case_meta.get("origin_step", 0))
        case_rollout = collect_case_study_rollout(
            agent,
            case_batch,
            case_pi,
            device,
            max_horizon=FIG6_CASE_STUDY_MAX_HORIZON,
            origin_step=origin_step,
            preprocessing=preprocessing,
        )
        p = write_csv(case_person_df, source_dir / "fig6_case_study_person.csv")
        if p:
            csv_paths.append(p)
        p = write_csv(case_rollout, source_dir / "fig6_case_study_rollout.csv")
        if p:
            csv_paths.append(p)
        figure_paths.append(
            plot_fig6_case_study(
                rollout=case_rollout,
                person_meta=case_person_df,
                figures_dir=figures_dir,
                dpi=args.dpi,
            )
        )
        if case_rollout.empty:
            notices.append("Fig 6 empty: case-study rollout unavailable")

    # Checkpoint Fig 1/3/4 before the heavy Fig 2 block.
    figure_paths = [p for p in figure_paths if p is not None]
    if args.skip_fig2:
        notices.append("Fig 2 skipped (--skip-fig2; use --no-skip-fig2 to enable)")
        write_report_md(
            output_dir / "report_summary.md",
            checkpoint=checkpoint,
            notices=notices,
            figure_paths=figure_paths,
            csv_paths=csv_paths,
            clinical=clinical_all,
            level_cmp=level_cmp,
            **report_kw,
        )
        print(f"Wrote Fig 1/3/4/5 under {figures_dir}; Fig 2 skipped.", flush=True)
    else:
        write_report_md(
            output_dir / "report_summary.md",
            checkpoint=checkpoint,
            notices=notices + ["Fig 2 pending (runs last)"],
            figure_paths=figure_paths,
            csv_paths=csv_paths,
            clinical=clinical_all,
            level_cmp=level_cmp,
            **report_kw,
        )
        print(
            f"Wrote Fig 1/3/4/5 under {figures_dir}; starting Fig 2 (health probing)...",
            flush=True,
        )

        # ----- Fig 2 (last: slow / memory-heavy; cached under fig2_cache/) -----
        # One-column PCA colored by death / ADL / IADL (probe CSVs still written).
        print("Collecting latent geometry for Fig 2...", flush=True)
        try:
            fig2_manifest = build_fig2_cache_manifest(
                checkpoint=checkpoint,
                train_data=Path(args.train_data),
                test_data=Path(args.test_data),
                preprocessing_path=Path(args.preprocessing),
                seed=args.seed,
                train_max_rows=FIG2_TRAIN_MAX_ROWS,
                test_max_rows=FIG2_TEST_MAX_ROWS,
                skip_mlp=True,
                mlp_weights=None,
            )
            cached_fig2 = (
                None
                if args.refit_fig2_cache
                else try_load_fig2_cache(fig2_cache_dir, fig2_manifest)
            )

            if cached_fig2 is not None:
                probe = cached_fig2["probe"]
                binary_probe = cached_fig2["binary_probe"]
                if not probe.empty and "model" in probe.columns:
                    probe = probe[probe["model"] == "JEPA"].reset_index(drop=True)
                if not binary_probe.empty and "model" in binary_probe.columns:
                    binary_probe = binary_probe[
                        binary_probe["model"] == "JEPA"
                    ].reset_index(drop=True)
                latent_2d = cached_fig2["latent_2d"]
                method_2d = cached_fig2["method_2d"]
                pca_rewards = cached_fig2["pca_rewards"]
                change = cached_fig2["change"]
                if pca_rewards.empty or not any(
                    name in set(pca_rewards.get("outcome", pd.Series(dtype=object)))
                    for name in FIG2_PCA_REWARDS
                ):
                    notices.append(
                        "Fig 2 PCA empty: death / ADL / IADL labels missing"
                    )
            else:
                test_loader = build_loader(
                    table_path=Path(args.test_data),
                    spec=spec,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=args.num_workers,
                    seed=args.seed,
                )
                jepa_health = collect_latents_and_health(
                    agent,
                    test_loader,
                    device,
                    preprocessing,
                    max_rows=FIG2_TEST_MAX_ROWS,
                )
                train_loader = build_loader(
                    table_path=Path(args.train_data),
                    spec=spec,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=args.num_workers,
                    seed=args.seed,
                )
                train_health = collect_latents_and_health(
                    agent,
                    train_loader,
                    device,
                    preprocessing,
                    max_rows=FIG2_TRAIN_MAX_ROWS,
                )
                # Train trajectory cache is the peak allocation here; release it now.
                del train_loader
                gc.collect()
                probe = pd.DataFrame()
                binary_probe = fit_binary_probe_metrics(
                    train_health,
                    jepa_health,
                    "JEPA",
                    outcome_names=list(spec.reward_binary),
                )
                latent_2d, method_2d = compute_latent_2d(
                    jepa_health,
                    color_col="adl_total_score",
                    seed=args.seed,
                    force_pca=True,
                )
                pca_rewards = compute_latent_pca_by_rewards(
                    jepa_health,
                    reward_names=list(FIG2_PCA_REWARDS),
                    seed=args.seed,
                )
                if pca_rewards.empty:
                    notices.append(
                        "Fig 2 PCA empty: death / ADL / IADL labels missing"
                    )
                test_loader = build_loader(
                    table_path=Path(args.test_data),
                    spec=spec,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=args.num_workers,
                    seed=args.seed,
                )
                change = collect_latent_change_vs_decline(
                    agent, test_loader, device, preprocessing
                )
                write_fig2_cache(
                    fig2_cache_dir,
                    manifest=fig2_manifest,
                    jepa_train=train_health,
                    jepa_test=jepa_health,
                    mlp_train=pd.DataFrame(),
                    mlp_test=pd.DataFrame(),
                    probe=probe,
                    binary_probe=binary_probe,
                    change=change,
                    latent_2d=latent_2d,
                    pca_rewards=pca_rewards,
                    method_2d=method_2d,
                )

            if not probe.empty and "model" in probe.columns:
                probe = probe[probe["model"] == "JEPA"].reset_index(drop=True)
            if not binary_probe.empty and "model" in binary_probe.columns:
                binary_probe = binary_probe[binary_probe["model"] == "JEPA"].reset_index(
                    drop=True
                )
            for name, frame in [
                ("fig2_linear_probe.csv", probe),
                ("fig2_binary_probe.csv", binary_probe),
                ("fig2_latent_2d.csv", latent_2d.drop(columns=[], errors="ignore")),
                ("fig2_latent_pca_by_reward.csv", pca_rewards),
                ("fig2_latent_change.csv", change),
            ]:
                p = write_csv(frame, source_dir / name)
                if p:
                    csv_paths.append(p)
            fig2_path = plot_fig2(
                probe=probe,
                binary_probe=binary_probe,
                latent_2d=latent_2d,
                method_2d=method_2d,
                change=change,
                latent_pca_rewards=pca_rewards,
                figures_dir=figures_dir,
                dpi=args.dpi,
            )
            if fig2_path is not None:
                figure_paths.append(fig2_path)
        except Exception as exc:  # noqa: BLE001 — keep Fig 1/3/4 if Fig 2 dies
            notices.append(f"Fig 2 failed: {exc}")
            print(f"WARNING: Fig 2 failed ({exc}); keeping Fig 1/3/4/5.", flush=True)

    figure_paths = [p for p in figure_paths if p is not None]
    write_report_md(
        output_dir / "report_summary.md",
        checkpoint=checkpoint,
        notices=notices,
        figure_paths=figure_paths,
        csv_paths=csv_paths,
        clinical=clinical_all,
        level_cmp=level_cmp,
        **report_kw,
    )
    print(f"Done. Figures in {figures_dir}", flush=True)
    for p in figure_paths:
        print(p)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Ensure package root is importable when run as a script.
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    args = parse_args(argv)
    _validate_exclusive_modes(args)
    if args.only_fig4:
        return run_fig4_only(args)
    if args.only_fig5:
        return run_fig5_only(args)
    if args.only_figS3:
        return run_figS3_only(args)
    if args.only_fig6:
        return run_fig6_only(args)
    return run_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
