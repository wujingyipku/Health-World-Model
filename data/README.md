# Data availability

Person-level data are **not included** in this repository. HRS microdata
are available only under a data-use agreement from the Health and Retirement
Study (https://hrs.isr.umich.edu/). ELSA (available at https://www.elsa-project.ac.uk/) and CHARLS (available at https://charls.pku.edu.cn/) files used for external
validation are likewise restricted and are not shipped.

## Source files for cleaning

The cleaners read these paths. Person-level data and the variable dictionary
are not included in this repository. Place `data/hrs/variable_specs.json`
locally before cleaning.

| Cohort | Default path |
|---|---|
| HRS variable specs | `data/hrs/variable_specs.json` |
| RAND HRS | `data/hrs/randhrs1992_2022v1.dta` |
| Harmonized HRS | `data/hrs/H_HRS_d.dta` |
| ELSA Waves 1–10 | `data/elsa/` |
| Harmonized CHARLS Waves 1–5 | `data/charls/` |

Place these files in `data/elsa/` when cleaning ELSA: `h_elsa_g3.tab`, `wave_10_elsa_data_eul_v4.tab`, `wave_10_ifs_derived_variables.tab`, `wave_10_financial_derived_variables.tab`, and `h_elsa_eol_a2.tab`.

Place these files in `data/charls/` when cleaning CHARLS: `H_CHARLS_D_Data_w5.csv`, `H_CHARLS_EOL_a_w5.csv`, and `H_CHARLS_LH_a_w5.csv`.

## Cleaned cohort files

The cleaners write these files, and `evaluate_and_plot_hrs_jepa.py` reads the same paths. Person-level data are not included in this repository.

| File | Path |
|---|---|
| HRS train | `data/hrs/HRS_train.parquet` |
| HRS validation | `data/hrs/HRS_validation.parquet` |
| HRS test | `data/hrs/HRS_test.parquet` |
| HRS preprocessing | `data/hrs/HRS_preprocessing.json` |
| HRS model config | `data/hrs/HRS_model_config.json` |
| ELSA external | `data/elsa/ELSA_external.parquet` |
| CHARLS external | `data/charls/CHARLS_external.parquet` |

`--external elsa` and `--external charls` replace the default HRS test file with the ELSA or CHARLS file above. ELSA and CHARLS cleaning read the HRS preprocessing and model config from `data/hrs/`.

## Example data

A de-identified, training-set–normalized excerpt is included so the released
checkpoint can be evaluated without the full cohort:

```
data/cleaned/demo_transitions.parquet
```

`data/cleaned/demo_manifest.json` records how the excerpt was drawn.

The excerpt contains 24 anonymous respondents (`demo_001`–`demo_024`) and 118
person–wave transitions from the HRS held-out test split. Original HRS
identifiers are removed, and the respondent shown in the manuscript case study
is not included. Continuous state and action channels are z-scores from the HRS
training-set mean and standard deviation. Categorical channels are integer codes
with a paired mask (`1` = observed, `0` = missing). `delta_t_years` is the
inter-wave interval in years. Outcome columns present for inspection are
`reward__death_event`, `reward__adl_worsening`, and `reward__iadl_worsening`.

This file is for executing `run_demo.py` only. Do not attempt to re-identify
respondents, and do not treat the excerpt as the analysis cohort.

## Cohort cleaning

`data_clean/` contains the HRS, ELSA, and CHARLS cleaning scripts. Person-level data are not included in this repository.

## Figure tables

`data/figures/` is part of the repository. Those CSVs are cohort-level
summaries used to draw manuscript Figures 2–5 (AUROC and related metrics,
risk-tertile means, Kaplan–Meier curves, and age-standardized rates). They
contain no respondent identifiers. `plot_manuscript_figures.py` reads them
directly. They are not a substitute for the restricted microdata.

## Full cohort access

After publication, access to the restricted HRS, ELSA, and CHARLS microdata
matches the Data Availability statement in the article. `run_demo.py` uses the
excerpt in `data/cleaned/`.
