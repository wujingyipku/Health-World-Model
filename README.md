# Analysis code for "A world model for longitudinal health-trajectory prediction in older adults across multinational cohorts"

The health world model was developed based on the Joint-Embedding Predictive Architecture framework, comprising four functional components: a state encoder, an action- and context-conditioned predictor, a stop-gradient target encoder, and clinical outcome heads. The model was trained using data from the US Health and Retirement Study (HRS) and externally validated in two additional national aging cohorts: the English Longitudinal Study of Ageing (ELSA) and the China Health and Retirement Longitudinal Study (CHARLS). Clinical outcomes of mortality and functional decline, including activities of daily living (ADL) worsening and instrumental activities of daily living (IADL) worsening, were evaluated. Model performance was assessed across two complementary dimensions in an open-loop setting: discrimination and calibration in characterizing clinically meaningful risk, and validity in simulating plausible longitudinal trajectories under specified actions.

## Overview

This repository contains the **model code and the checkpoint**
(`weights/best_final.pt`), plus the aggregate tables and plotting code for
manuscript Figures 2–5. Training entry points and the full cohort tables are
not part of this release.

## Data availability

Person-level data are **not included** in this repository (restricted cohort
microdata). The de-identified, training-set–normalized example used to test the
checkpoint is available under **controlled access** — see the Data Availability
statement in the published article and [`data/README.md`](data/README.md) for
the request and placement procedure.

Cohort-level tables for Figures 2–5 are included under `data/figures/`. They
hold AUROC, sensitivity, risk-tertile means, Kaplan–Meier curves, and
age-standardized rates. They do not contain respondent identifiers.

To run the evaluation script, obtain that excerpt and place it at:

```
data/cleaned/demo_transitions.parquet
```

## Requirements and how to run

- Python 3.10+ with the packages in `requirements.txt`
  (numpy, pandas, pyarrow, torch, matplotlib, scikit-learn). Evaluation uses torch. Redrawing Figures 2–5 imports the evaluation module, so it also needs torch and scikit-learn, plus matplotlib to write the PNGs.

```bash
pip install -r requirements.txt
# place the de-identified excerpt at data/cleaned/demo_transitions.parquet
python run_demo.py --device cuda
```

`run_demo.py` loads `weights/best_final.pt`, encodes each example trajectory,
and writes an open-loop table to `results/demo_predictions.csv`. Two scenarios
are evaluated for horizons 1–3:

- `observed` — actions, static context, and the inter-wave interval stay at the
  first transition.
- `light_activity_high` — the same inputs, except from horizon 2
  `light_activity_frequency` is set to 4, the high code in Figure 5.
  Other actions stay at the first transition.

Printed and saved values are probabilities of `death_event`, `adl_worsening`,
and `iadl_worsening`. Horizon 1 matches across the two scenarios because the
light-activity change starts at horizon 2, the same switch as Figure 5. If the
excerpt is missing, the script exits and points to `data/README.md`.

Redraw Figures 2–5 from the shipped tables (no checkpoint, no microdata):

```bash
python plot_manuscript_figures.py
```

This writes four files under `results/`:

| Output | Manuscript | Cohort | Source tables |
|--|--|--|--|
| `results/figure2_hrs_openloop.png` | Figure 2 | HRS | `data/figures/fig2_hrs/` |
| `results/figure3_elsa_openloop.png` | Figure 3 | ELSA | `data/figures/fig3_elsa/` |
| `results/figure4_charls_openloop.png` | Figure 4 | CHARLS | `data/figures/fig4_charls/` |
| `results/figure5_light_activity.png` | Figure 5 | HRS, ELSA, CHARLS | `data/figures/fig5_light_activity/` |

Figure 2 presents open-loop prediction of clinical outcomes by the world model on the HRS test set. Panels (A-C) show discrimination performance for ADL worsening, IADL worsening, and death across open-loop horizons h = 1-5 waves. Each row reports one metric: area under the receiver operating characteristic curve (AUROC; panel A), sensitivity (panel B), and sensitivity at specificity ≥0.8 (panel C). In each subplot, the left axis shows absolute performance of the world model (dark blue) and an MLP-h* baseline fit at the same horizon (light blue), and the right axis shows relative performance (grey dashed line), defined as the ratio of the world model at horizon h to MLP-h*; the horizontal dashed line marks parity (ratio = 1). Binary classifications use a probability threshold of 0.5. Panel (D) shows the mean open-loop predicted event probability by horizon, stratified by world model risk tertile (low, mid, high) and by whether the outcome was observed (dark grey dashed) or not observed (light grey dotted) on the realized trajectory. Panel (E) shows Kaplan-Meier estimates of event-free survival for death, stratified by open-loop risk tertile assigned at horizons h = 1, 2 and 3 waves; n denotes the number of individuals in each horizon-specific cohort, and time is measured in HRS survey waves. 

Figure 3 shows the same panels for ELSA, with open-loop horizons h = 1-5. Figure 4 shows them for CHARLS, with open-loop horizons h = 1-4.

Figure 5 shows open-loop validation of simulated age-standardized mortality risk under static and counterfactual light-activity-frequency exposure in HRS, ELSA and CHARLS. The world model was trained on HRS and evaluated by open-loop simulation on held-out HRS individuals (A, B) and, without refitting, on the external ELSA (C, D) and CHARLS (E, F) cohorts. In the left column (A, C, E), individuals are stratified into three static, baseline-fixed light-activity-frequency regimes, defined from activity-frequency codes recorded at both of the first two waves; observed age-standardized risk (dashed lines, circles) is compared with world model-simulated risk (solid lines, dots) within each regime. In the right column (B, D, F), only individuals in the low regime at baseline are retained and further split into three groups by their empirically realized activity status at horizon 2: remained in the low regime, transitioned to the mid regime, and transitioned to the high regime; the world model rollout is counterfactually forced to switch to the corresponding regime at horizon 2, so that simulated and observed risk are compared under the same realized transition. 

## File map

| File | Computes | Reproduces |
|--|--|--|
| `run_demo.py` | Open-loop death / ADL / IADL risks under observed actions and a single-lever increase in light activity | `results/demo_predictions.csv` |
| `plot_manuscript_figures.py` | Reads the aggregate tables and calls the figure functions | `results/figure2_hrs_openloop.png`, `results/figure3_elsa_openloop.png`, `results/figure4_charls_openloop.png`, `results/figure5_light_activity.png` |
| `evaluate_and_plot_hrs_jepa.py` |Model performance evaluation: discrimination and calibration in characterizing clinically meaningful risk, and validity in simulating plausible longitudinal trajectories under specified actions| The panels inside those four PNGs |
| `code/specs.py` | Channel layout, categorical codes, and Δt standardization | `weights/model_spec.json` |
| `code/nn.py` | State, static, and action encoders; temporal transformer; conditioned predictor; EMA target; clinical heads | The forward pass of the health world model |
| `code/data.py` | Reads a transition table and pads person-level trajectories | The tensors consumed by `run_demo.py` |
| `code/trainer.py` | Training-step helpers imported by the figure module | No separate output |
| `code/evaluation.py` | Metric helpers imported by the figure module | No separate output |
| `code/fig_report.py` | Action-stratum definitions used by Figure 5 (light activity; switch at horizon 2) | The stratum labels and colours in Figure 5 |
| `code/external_cohorts.py` | ELSA and CHARLS cohort names. Full tables are not in this release | No separate output |
| `weights/resolved_config.json` | Architecture widths, depth, and dropout | The network built around the checkpoint |
| `weights/best_final.pt` | Manuscript checkpoint (`agent` state dict) | The model used for the reported results |
| `data/figures/fig2_hrs/` | HRS open-loop metrics, risk tertiles, and death KM | Figure 2 |
| `data/figures/fig3_elsa/` | ELSA open-loop metrics, risk tertiles, and death KM | Figure 3 |
| `data/figures/fig4_charls/` | CHARLS open-loop metrics, risk tertiles, and death KM | Figure 4 |
| `data/figures/fig5_light_activity/{hrs,elsa,charls}/` | Observed and simulated age-standardized death risk by light-activity stratum | Figure 5 |

`code/__init__.py` marks the package. It is imported by the scripts above and produces no output of its own.

## License

MIT — see [LICENSE](LICENSE). The license covers this code. It does not cover
HRS, ELSA, or CHARLS microdata, and it does not grant the right to re-identify
respondents in the controlled-access excerpt.
