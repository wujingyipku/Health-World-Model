# Data availability

The person-level HRS cohort is **not included** in this repository. HRS microdata
are available only under a data-use agreement from the Health and Retirement
Study (https://hrs.isr.umich.edu/). ELSA (available at https://www.elsa-project.ac.uk/) and CHARLS (available at https://charls.pku.edu.cn/) files used for external
validation are likewise restricted and are not shipped.

## Example data

A de-identified, training-set–normalized excerpt is provided for peer review so
the released checkpoint can be evaluated without the full cohort. It is **not**
committed to git. After approval, place it at:

```
data/cleaned/demo_transitions.parquet
```

`data/cleaned/demo_manifest.json` may sit next to that file. It records how the
excerpt was drawn.

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

## Figure tables

`data/figures/` is part of the repository. Those CSVs are cohort-level
summaries used to draw manuscript Figures 2–5 (AUROC and related metrics,
risk-tertile means, Kaplan–Meier curves, and age-standardized rates). They
contain no respondent identifiers. `plot_manuscript_figures.py` reads them
directly. They are not a substitute for the restricted microdata.

## Request
After publication, the access route will match the Data
Availability statement in the article. The model weights in `weights/` do not
require this file in order to be inspected, but evaluation does.
