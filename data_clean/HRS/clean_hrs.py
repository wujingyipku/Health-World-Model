#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRS Waves 8-14 cleaning and transition builder using only RAND and Harmonized HRS files.

Inputs
------
1. RAND HRS Longitudinal File 2022 V1 Stata file.
2. Harmonized HRS Version D Stata file.
3. The finalized HRS medical-world-model data dictionary.

Outputs
-------
- hrs_world_model_long_raw.csv (live person-wave long table before nearby-wave fill; not transitions)
- hrs_world_model_transitions.parquet
- feature_metadata.json
- source_audit.csv
- missingness_summary.csv
- variable_descriptive_summary.csv
- variable_descriptive_by_wave.csv
- variable_descriptive_transitions.csv
- HRS_train/validation/test.parquet (model-ready, imputed/scaled)
- HRS_train/validation/test.csv (audit copies of the model-ready tables)
- HRS_model_config.json
- HRS_preprocessing.json
- split_summary.json
- optional CSV copies

Important missing-value rule
----------------------------
All Stata extended missing codes (including .x/.q/.s) remain NA with
observed=0. There is no global structural-zero fill (.x/.q/.s → 0).

For static/dynamic state and action variables, remaining missing values are
first filled within person from the nearest earlier wave (LOCF) then later
wave (NOCB), using only rows with observed==1 as donors. Observation masks
stay unchanged after this carry step. Any values still missing are imputed
with training-set median/mode statistics only when the HRS model
parquet bundle is written.

This script creates person-level training, validation, and test sets.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue
from sklearn.model_selection import GroupShuffleSplit

LOGGER = logging.getLogger("hrs_cleaning")

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
DATA_HRS = PACKAGE_ROOT / "data" / "hrs"
DEFAULT_WAVES = [8, 9, 10, 11, 12, 13, 14]
BUILTIN_SPECS_PATH = DATA_HRS / "variable_specs.json"
DEFAULT_RAND_FILE = DATA_HRS / "randhrs1992_2022v1.dta"
DEFAULT_HARMONIZED_FILE = DATA_HRS / "H_HRS_d.dta"
DEFAULT_OUTPUT_DIR = DATA_HRS
WAVE_YEAR = {8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018}
# Legacy label for .x/.q/.s; these codes are now always treated as missing.
STRUCTURAL_ZERO_CODES = {".x", ".q", ".s"}
OTHER_STATA_MISSING_RE = re.compile(r"^\.[a-z]$", flags=re.IGNORECASE)

# Dropped from cleaning extraction and model tables (high missingness / unused).
EXCLUDE_STATE_VARIABLES = frozenset(
    {
        "current_industry",
        "proxy_memory_change",
        "proxy_memory_rating",
        "age_at_us_arrival",
    }
)
# Backward-compatible alias used by model-bundle helpers.
EXCLUDE_MODEL_STATE_VARIABLES = EXCLUDE_STATE_VARIABLES

# Rewards still derived into transitions for analysis, but omitted from the
# Model reward head / composite / model_config.reward_columns.
EXCLUDE_MODEL_REWARD_VARIABLES = frozenset({"cancer_incident"})

# Force ordinal model heads even when the dictionary Data Type is "categorical".
# current_occupation is coarsened to physical-demand ranks 1<2<3.
FORCE_MODEL_ORDINAL_VARIABLES = frozenset({"current_occupation"})

# Route these variables to model ``static_context_columns`` (context encoder only;
# not reconstructed by the state decoder). Extraction stays wave-local via
# SOURCE_CANDIDATES / derives, except person-level statics already handled by
# STATIC_CANDIDATES (living_siblings, urban_rural_residence).
# Order here is the preferred order inside static_context_columns after true
# demography statics.
FORCE_STATIC_CONTEXT_VARIABLES: tuple[str, ...] = (
    "age_years",
    "marital_status",
    "household_size",
    "living_children",
    "living_siblings",
    "living_alone",
    "nursing_home_residence",
    "urban_rural_residence",
    "home_ownership",
    "housing_type",
    "total_household_income",
    "total_household_wealth",
    "poverty_threshold",
    "income_to_poverty_ratio",
    "labor_force_status",
    "currently_working",
    "self_employed",
    "hours_worked_per_week",
    "weeks_worked_per_year",
    "current_occupation",
    "job_tenure",
    "health_limits_work",
    "health_insurance_any",
    "medicare_coverage",
    "medicaid_coverage",
    "va_coverage",
    "long_term_care_insurance",
    "measured_height",
)
# Already correctly extracted once-per-person via STATIC_CANDIDATES.
PERSON_LEVEL_STATIC_CONTEXT_VARIABLES = frozenset(
    {
        "living_siblings",
        "urban_rural_residence",
    }
)

# Physical-demand coarsening of RAND major occupation codes.
# 1=sedentary, 2=light physical, 3=heavy physical.
OCCUPATION_PHYSICAL_DEMAND_SEDENTARY = 1.0
OCCUPATION_PHYSICAL_DEMAND_LIGHT = 2.0
OCCUPATION_PHYSICAL_DEMAND_HEAVY = 3.0

# OCCUPB / JCOCCB — 2000 Census major groups (W8–W9; also W10–W14 fallback).
OCCUPB_TO_PHYSICAL_DEMAND = {
    1: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # management
    2: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # business operations specialists
    3: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # financial specialists
    4: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # computer + math
    5: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # architecture + engineering
    6: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # life / physical / social science
    7: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # community + social services
    8: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # legal
    9: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # education / training / library
    10: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # arts / design / entertainment
    11: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # healthcare practitioners / technical
    12: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # healthcare support
    13: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # protective service
    14: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # food prep + serving
    15: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # building / grounds / cleaning / maint
    16: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # personal care + service
    17: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # sales
    18: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # office + admin support
    19: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # farm / fish / forestry
    20: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # construction trades
    21: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # extraction
    22: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # install / maint / repair
    23: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # production
    24: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # transport / material moving
    25: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # military specific
}

# OCCUPC / JCOCCC — 2010 Census major groups (preferred W10–W14).
OCCUPC_TO_PHYSICAL_DEMAND = {
    1: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # management
    2: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # business + financial operations
    3: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # computer + mathematical
    4: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # architecture + engineering
    5: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # life / physical / social science
    6: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # community + social service
    7: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # legal
    8: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # education / training / library
    9: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # arts / design / entertainment / media
    10: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # healthcare practitioners / technical
    11: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # healthcare support
    12: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # protective service
    13: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # food prep + serving
    14: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # building / grounds cleaning / maint
    15: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # personal care + service
    16: OCCUPATION_PHYSICAL_DEMAND_LIGHT,  # sales
    17: OCCUPATION_PHYSICAL_DEMAND_SEDENTARY,  # office + administrative support
    18: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # farming / fishing / forestry
    19: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # construction + extraction
    20: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # installation / maintenance / repair
    21: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # production
    22: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # transportation + material moving
    23: OCCUPATION_PHYSICAL_DEMAND_HEAVY,  # military specific
}

# Annual CPI-U values. Used only to convert medical cost to 2018 dollars.
CPI_U = {
    2006: 201.6,
    2008: 215.303,
    2010: 218.056,
    2012: 229.594,
    2014: 236.736,
    2016: 240.007,
    2018: 251.107,
}

SOURCE_CANDIDATES = {
  "person_id": {
    "8": [
      "HHIDPN"
    ],
    "9": [
      "HHIDPN"
    ],
    "10": [
      "HHIDPN"
    ],
    "11": [
      "HHIDPN"
    ],
    "12": [
      "HHIDPN"
    ],
    "13": [
      "HHIDPN"
    ],
    "14": [
      "HHIDPN"
    ]
  },
  "household_id": {
    "8": [
      "HHID"
    ],
    "9": [
      "HHID"
    ],
    "10": [
      "HHID"
    ],
    "11": [
      "HHID"
    ],
    "12": [
      "HHID"
    ],
    "13": [
      "HHID"
    ],
    "14": [
      "HHID"
    ]
  },
  "person_number": {
    "8": [
      "PN"
    ],
    "9": [
      "PN"
    ],
    "10": [
      "PN"
    ],
    "11": [
      "PN"
    ],
    "12": [
      "PN"
    ],
    "13": [
      "PN"
    ],
    "14": [
      "PN"
    ]
  },
  "wave": {
    "8": [],
    "9": [],
    "10": [],
    "11": [],
    "12": [],
    "13": [],
    "14": []
  },
  "interview_year": {
    "8": [
      "R8IWBEG",
      "R8IWEND",
      "R8IWY"
    ],
    "9": [
      "R9IWBEG",
      "R9IWEND",
      "R9IWY"
    ],
    "10": [
      "R10IWBEG",
      "R10IWEND",
      "R10IWY"
    ],
    "11": [
      "R11IWBEG",
      "R11IWEND",
      "R11IWY"
    ],
    "12": [
      "R12IWBEG",
      "R12IWEND",
      "R12IWY"
    ],
    "13": [
      "R13IWBEG",
      "R13IWEND",
      "R13IWY"
    ],
    "14": [
      "R14IWBEG",
      "R14IWEND",
      "R14IWY"
    ]
  },
  "interview_month": {
    "8": [
      "R8IWBEG",
      "R8IWEND",
      "R8IWM"
    ],
    "9": [
      "R9IWBEG",
      "R9IWEND",
      "R9IWM"
    ],
    "10": [
      "R10IWBEG",
      "R10IWEND",
      "R10IWM"
    ],
    "11": [
      "R11IWBEG",
      "R11IWEND",
      "R11IWM"
    ],
    "12": [
      "R12IWBEG",
      "R12IWEND",
      "R12IWM"
    ],
    "13": [
      "R13IWBEG",
      "R13IWEND",
      "R13IWM"
    ],
    "14": [
      "R14IWBEG",
      "R14IWEND",
      "R14IWM"
    ]
  },
  "interview_date": {
    "8": [
      "R8IWEND",
      "R8IWBEG"
    ],
    "9": [
      "R9IWEND",
      "R9IWBEG"
    ],
    "10": [
      "R10IWEND",
      "R10IWBEG"
    ],
    "11": [
      "R11IWEND",
      "R11IWBEG"
    ],
    "12": [
      "R12IWEND",
      "R12IWBEG"
    ],
    "13": [
      "R13IWEND",
      "R13IWBEG"
    ],
    "14": [
      "R14IWEND",
      "R14IWBEG"
    ]
  },
  "delta_time_years": {
    "8": [],
    "9": [],
    "10": [],
    "11": [],
    "12": [],
    "13": [],
    "14": []
  },
  "respondent_weight": {
    "8": [
      "R8WTRESP"
    ],
    "9": [
      "R9WTRESP"
    ],
    "10": [
      "R10WTRESP"
    ],
    "11": [
      "R11WTRESP"
    ],
    "12": [
      "R12WTRESP"
    ],
    "13": [
      "R13WTRESP"
    ],
    "14": [
      "R14WTRESP"
    ]
  },
  "interview_status": {
    "8": [
      "R8IWSTAT"
    ],
    "9": [
      "R9IWSTAT"
    ],
    "10": [
      "R10IWSTAT"
    ],
    "11": [
      "R11IWSTAT"
    ],
    "12": [
      "R12IWSTAT"
    ],
    "13": [
      "R13IWSTAT"
    ],
    "14": [
      "R14IWSTAT"
    ]
  },
  "sex": {
    "8": [
      "RAGENDER"
    ],
    "9": [
      "RAGENDER"
    ],
    "10": [
      "RAGENDER"
    ],
    "11": [
      "RAGENDER"
    ],
    "12": [
      "RAGENDER"
    ],
    "13": [
      "RAGENDER"
    ],
    "14": [
      "RAGENDER"
    ]
  },
  "race_ethnicity": {
    "8": [
      "RARACEM",
      "RAHISPAN"
    ],
    "9": [
      "RARACEM",
      "RAHISPAN"
    ],
    "10": [
      "RARACEM",
      "RAHISPAN"
    ],
    "11": [
      "RARACEM",
      "RAHISPAN"
    ],
    "12": [
      "RARACEM",
      "RAHISPAN"
    ],
    "13": [
      "RARACEM",
      "RAHISPAN"
    ],
    "14": [
      "RARACEM",
      "RAHISPAN"
    ]
  },
  "birth_year": {
    "8": [
      "RABYEAR"
    ],
    "9": [
      "RABYEAR"
    ],
    "10": [
      "RABYEAR"
    ],
    "11": [
      "RABYEAR"
    ],
    "12": [
      "RABYEAR"
    ],
    "13": [
      "RABYEAR"
    ],
    "14": [
      "RABYEAR"
    ]
  },
  "birth_month": {
    "8": [
      "RABMONTH"
    ],
    "9": [
      "RABMONTH"
    ],
    "10": [
      "RABMONTH"
    ],
    "11": [
      "RABMONTH"
    ],
    "12": [
      "RABMONTH"
    ],
    "13": [
      "RABMONTH"
    ],
    "14": [
      "RABMONTH"
    ]
  },
  "education_years": {
    "8": [
      "RAEDYRS"
    ],
    "9": [
      "RAEDYRS"
    ],
    "10": [
      "RAEDYRS"
    ],
    "11": [
      "RAEDYRS"
    ],
    "12": [
      "RAEDYRS"
    ],
    "13": [
      "RAEDYRS"
    ],
    "14": [
      "RAEDYRS"
    ]
  },
  "education_level": {
    "8": [
      "RAEDUCL"
    ],
    "9": [
      "RAEDUCL"
    ],
    "10": [
      "RAEDUCL"
    ],
    "11": [
      "RAEDUCL"
    ],
    "12": [
      "RAEDUCL"
    ],
    "13": [
      "RAEDUCL"
    ],
    "14": [
      "RAEDUCL"
    ]
  },
  "foreign_born": {
    "8": [
      "RABPLACE"
    ],
    "9": [
      "RABPLACE"
    ],
    "10": [
      "RABPLACE"
    ],
    "11": [
      "RABPLACE"
    ],
    "12": [
      "RABPLACE"
    ],
    "13": [
      "RABPLACE"
    ],
    "14": [
      "RABPLACE"
    ]
  },
  "age_at_us_arrival": {
    "8": [
      "RAARRIAGE"
    ],
    "9": [
      "RAARRIAGE"
    ],
    "10": [
      "RAARRIAGE"
    ],
    "11": [
      "RAARRIAGE"
    ],
    "12": [
      "RAARRIAGE"
    ],
    "13": [
      "RAARRIAGE"
    ],
    "14": [
      "RAARRIAGE"
    ]
  },
  "childhood_health": {
    "8": [
      "RACHSHLT"
    ],
    "9": [
      "RACHSHLT"
    ],
    "10": [
      "RACHSHLT"
    ],
    "11": [
      "RACHSHLT"
    ],
    "12": [
      "RACHSHLT"
    ],
    "13": [
      "RACHSHLT"
    ],
    "14": [
      "RACHSHLT"
    ]
  },
  "childhood_ses": {
    "8": [],
    "9": [],
    "10": [],
    "11": [],
    "12": [],
    "13": [],
    "14": []
  },
  "mother_education": {
    "8": [
      "RAMEDUC"
    ],
    "9": [
      "RAMEDUC"
    ],
    "10": [
      "RAMEDUC"
    ],
    "11": [
      "RAMEDUC"
    ],
    "12": [
      "RAMEDUC"
    ],
    "13": [
      "RAMEDUC"
    ],
    "14": [
      "RAMEDUC"
    ]
  },
  "father_education": {
    "8": [
      "RAFEDUC"
    ],
    "9": [
      "RAFEDUC"
    ],
    "10": [
      "RAFEDUC"
    ],
    "11": [
      "RAFEDUC"
    ],
    "12": [
      "RAFEDUC"
    ],
    "13": [
      "RAFEDUC"
    ],
    "14": [
      "RAFEDUC"
    ]
  },
  "longest_job_occupation": {
    "8": [
      "RALJOC"
    ],
    "9": [
      "RALJOC"
    ],
    "10": [
      "RALJOC"
    ],
    "11": [
      "RALJOC"
    ],
    "12": [
      "RALJOC"
    ],
    "13": [
      "RALJOC"
    ],
    "14": [
      "RALJOC"
    ]
  },
  "longest_job_tenure": {
    "8": [
      "RALJTEN"
    ],
    "9": [
      "RALJTEN"
    ],
    "10": [
      "RALJTEN"
    ],
    "11": [
      "RALJTEN"
    ],
    "12": [
      "RALJTEN"
    ],
    "13": [
      "RALJTEN"
    ],
    "14": [
      "RALJTEN"
    ]
  },
  "age_years": {
    "8": [
      "R8AGEY_E"
    ],
    "9": [
      "R9AGEY_E"
    ],
    "10": [
      "R10AGEY_E"
    ],
    "11": [
      "R11AGEY_E"
    ],
    "12": [
      "R12AGEY_E"
    ],
    "13": [
      "R13AGEY_E"
    ],
    "14": [
      "R14AGEY_E"
    ]
  },
  "marital_status": {
    "8": [
      "R8MSTAT"
    ],
    "9": [
      "R9MSTAT"
    ],
    "10": [
      "R10MSTAT"
    ],
    "11": [
      "R11MSTAT"
    ],
    "12": [
      "R12MSTAT"
    ],
    "13": [
      "R13MSTAT"
    ],
    "14": [
      "R14MSTAT"
    ]
  },
  "household_size": {
    "8": [
      "H8HHRES"
    ],
    "9": [
      "H9HHRES"
    ],
    "10": [
      "H10HHRES"
    ],
    "11": [
      "H11HHRES"
    ],
    "12": [
      "H12HHRES"
    ],
    "13": [
      "H13HHRES"
    ],
    "14": [
      "H14HHRES"
    ]
  },
  "living_children": {
    "8": [
      "H8CHILD"
    ],
    "9": [
      "H9CHILD"
    ],
    "10": [
      "H10CHILD"
    ],
    "11": [
      "H11CHILD"
    ],
    "12": [
      "H12CHILD"
    ],
    "13": [
      "H13CHILD"
    ],
    "14": [
      "H14CHILD"
    ]
  },
  "living_siblings": {
    "8": [
      "R8LIVSIB"
    ],
    "9": [
      "R9LIVSIB"
    ],
    "10": [
      "R10LIVSIB"
    ],
    "11": [
      "R11LIVSIB"
    ],
    "12": [
      "R12LIVSIB"
    ],
    "13": [
      "R13LIVSIB"
    ],
    "14": [
      "R14LIVSIB"
    ]
  },
  "living_alone": {
    "8": [
      "H8HHRES"
    ],
    "9": [
      "H9HHRES"
    ],
    "10": [
      "H10HHRES"
    ],
    "11": [
      "H11HHRES"
    ],
    "12": [
      "H12HHRES"
    ],
    "13": [
      "H13HHRES"
    ],
    "14": [
      "H14HHRES"
    ]
  },
  "nursing_home_residence": {
    "8": [
      "R8NHMLIV"
    ],
    "9": [
      "R9NHMLIV"
    ],
    "10": [
      "R10NHMLIV"
    ],
    "11": [
      "R11NHMLIV"
    ],
    "12": [
      "R12NHMLIV"
    ],
    "13": [
      "R13NHMLIV"
    ],
    "14": [
      "R14NHMLIV"
    ]
  },
  "urban_rural_residence": {
    "8": [
      "H8RURAL"
    ],
    "9": [
      "H9RURAL"
    ],
    "10": [
      "H10RURAL"
    ],
    "11": [
      "H11RURAL"
    ],
    "12": [
      "H12RURAL"
    ],
    "13": [
      "H13RURAL"
    ],
    "14": [
      "H14RURAL"
    ]
  },
  "home_ownership": {
    "8": [
      "H8AHOUS"
    ],
    "9": [
      "H9AHOUS"
    ],
    "10": [
      "H10AHOUS"
    ],
    "11": [
      "H11AHOUS"
    ],
    "12": [
      "H12AHOUS"
    ],
    "13": [
      "H13AHOUS"
    ],
    "14": [
      "H14AHOUS"
    ]
  },
  "housing_type": {
    "8": [
      "R8HOMETYP"
    ],
    "9": [
      "R9HOMETYP"
    ],
    "10": [
      "R10HOMETYP"
    ],
    "11": [
      "R11HOMETYP"
    ],
    "12": [
      "R12HOMETYP"
    ],
    "13": [
      "R13HOMETYP"
    ],
    "14": [
      "R14HOMETYP"
    ]
  },
  "total_household_income": {
    "8": [
      "H8ITOT"
    ],
    "9": [
      "H9ITOT"
    ],
    "10": [
      "H10ITOT"
    ],
    "11": [
      "H11ITOT"
    ],
    "12": [
      "H12ITOT"
    ],
    "13": [
      "H13ITOT"
    ],
    "14": [
      "H14ITOT"
    ]
  },
  "total_household_wealth": {
    "8": [
      "H8ATOTB"
    ],
    "9": [
      "H9ATOTB"
    ],
    "10": [
      "H10ATOTB"
    ],
    "11": [
      "H11ATOTB"
    ],
    "12": [
      "H12ATOTB"
    ],
    "13": [
      "H13ATOTB"
    ],
    "14": [
      "H14ATOTB"
    ]
  },
  "poverty_threshold": {
    "8": [
      "H8POVTHR"
    ],
    "9": [
      "H9POVTHR"
    ],
    "10": [
      "H10POVTHR"
    ],
    "11": [
      "H11POVTHR"
    ],
    "12": [
      "H12POVTHR"
    ],
    "13": [
      "H13POVTHR"
    ],
    "14": [
      "H14POVTHR"
    ]
  },
  "income_to_poverty_ratio": {
    "8": [
      "H8ITOT",
      "H8POVTHR"
    ],
    "9": [
      "H9ITOT",
      "H9POVTHR"
    ],
    "10": [
      "H10ITOT",
      "H10POVTHR"
    ],
    "11": [
      "H11ITOT",
      "H11POVTHR"
    ],
    "12": [
      "H12ITOT",
      "H12POVTHR"
    ],
    "13": [
      "H13ITOT",
      "H13POVTHR"
    ],
    "14": [
      "H14ITOT",
      "H14POVTHR"
    ]
  },
  "labor_force_status": {
    "8": [
      "R8LBRF"
    ],
    "9": [
      "R9LBRF"
    ],
    "10": [
      "R10LBRF"
    ],
    "11": [
      "R11LBRF"
    ],
    "12": [
      "R12LBRF"
    ],
    "13": [
      "R13LBRF"
    ],
    "14": [
      "R14LBRF"
    ]
  },
  "currently_working": {
    "8": [
      "R8WORK",
      "R8LBRF"
    ],
    "9": [
      "R9WORK",
      "R9LBRF"
    ],
    "10": [
      "R10WORK",
      "R10LBRF"
    ],
    "11": [
      "R11WORK",
      "R11LBRF"
    ],
    "12": [
      "R12WORK",
      "R12LBRF"
    ],
    "13": [
      "R13WORK",
      "R13LBRF"
    ],
    "14": [
      "R14WORK",
      "R14LBRF"
    ]
  },
  "self_employed": {
    "8": [
      "R8SLFEMP"
    ],
    "9": [
      "R9SLFEMP"
    ],
    "10": [
      "R10SLFEMP"
    ],
    "11": [
      "R11SLFEMP"
    ],
    "12": [
      "R12SLFEMP"
    ],
    "13": [
      "R13SLFEMP"
    ],
    "14": [
      "R14SLFEMP"
    ]
  },
  "hours_worked_per_week": {
    "8": [
      "R8JHOURS"
    ],
    "9": [
      "R9JHOURS"
    ],
    "10": [
      "R10JHOURS"
    ],
    "11": [
      "R11JHOURS"
    ],
    "12": [
      "R12JHOURS"
    ],
    "13": [
      "R13JHOURS"
    ],
    "14": [
      "R14JHOURS"
    ]
  },
  "weeks_worked_per_year": {
    "8": [
      "R8JWEEKS"
    ],
    "9": [
      "R9JWEEKS"
    ],
    "10": [
      "R10JWEEKS"
    ],
    "11": [
      "R11JWEEKS"
    ],
    "12": [
      "R12JWEEKS"
    ],
    "13": [
      "R13JWEEKS"
    ],
    "14": [
      "R14JWEEKS"
    ]
  },
  "current_occupation": {
    "8": [
      "R8JCOCCB"
    ],
    "9": [
      "R9JCOCCB"
    ],
    "10": [
      "R10JCOCCC",
      "R10JCOCCB"
    ],
    "11": [
      "R11JCOCCC",
      "R11JCOCCB"
    ],
    "12": [
      "R12JCOCCC",
      "R12JCOCCB"
    ],
    "13": [
      "R13JCOCCC",
      "R13JCOCCB"
    ],
    "14": [
      "R14JCOCCC",
      "R14JCOCCB"
    ]
  },
  "job_tenure": {
    "8": [
      "R8JCTEN"
    ],
    "9": [
      "R9JCTEN"
    ],
    "10": [
      "R10JCTEN"
    ],
    "11": [
      "R11JCTEN"
    ],
    "12": [
      "R12JCTEN"
    ],
    "13": [
      "R13JCTEN"
    ],
    "14": [
      "R14JCTEN"
    ]
  },
  "health_limits_work": {
    "8": [
      "R8HLTHLM"
    ],
    "9": [
      "R9HLTHLM"
    ],
    "10": [
      "R10HLTHLM"
    ],
    "11": [
      "R11HLTHLM"
    ],
    "12": [
      "R12HLTHLM"
    ],
    "13": [
      "R13HLTHLM"
    ],
    "14": [
      "R14HLTHLM"
    ]
  },
  "health_insurance_any": {
    "8": [
      "R8HIGOV",
      "R8COVR",
      "R8COVS",
      "R8HIOTHP"
    ],
    "9": [
      "R9HIGOV",
      "R9COVR",
      "R9COVS",
      "R9HIOTHP"
    ],
    "10": [
      "R10HIGOV",
      "R10COVR",
      "R10COVS",
      "R10HIOTHP"
    ],
    "11": [
      "R11HIGOV",
      "R11COVR",
      "R11COVS",
      "R11HIOTHP"
    ],
    "12": [
      "R12HIGOV",
      "R12COVR",
      "R12COVS",
      "R12HIOTHP"
    ],
    "13": [
      "R13HIGOV",
      "R13COVR",
      "R13COVS",
      "R13HIOTHP"
    ],
    "14": [
      "R14HIGOV",
      "R14COVR",
      "R14COVS",
      "R14HIOTHP"
    ]
  },
  "medicare_coverage": {
    "8": [
      "R8GOVMR"
    ],
    "9": [
      "R9GOVMR"
    ],
    "10": [
      "R10GOVMR"
    ],
    "11": [
      "R11GOVMR"
    ],
    "12": [
      "R12GOVMR"
    ],
    "13": [
      "R13GOVMR"
    ],
    "14": [
      "R14GOVMR"
    ]
  },
  "medicaid_coverage": {
    "8": [
      "R8GOVMD"
    ],
    "9": [
      "R9GOVMD"
    ],
    "10": [
      "R10GOVMD"
    ],
    "11": [
      "R11GOVMD"
    ],
    "12": [
      "R12GOVMD"
    ],
    "13": [
      "R13GOVMD"
    ],
    "14": [
      "R14GOVMD"
    ]
  },
  "va_coverage": {
    "8": [
      "R8GOVVA"
    ],
    "9": [
      "R9GOVVA"
    ],
    "10": [
      "R10GOVVA"
    ],
    "11": [
      "R11GOVVA"
    ],
    "12": [
      "R12GOVVA"
    ],
    "13": [
      "R13GOVVA"
    ],
    "14": [
      "R14GOVVA"
    ]
  },
  "long_term_care_insurance": {
    "8": [
      "R8HILTC"
    ],
    "9": [
      "R9HILTC"
    ],
    "10": [
      "R10HILTC"
    ],
    "11": [
      "R11HILTC"
    ],
    "12": [
      "R12HILTC"
    ],
    "13": [
      "R13HILTC"
    ],
    "14": [
      "R14HILTC"
    ]
  },
  "self_rated_health": {
    "8": [
      "R8SHLT"
    ],
    "9": [
      "R9SHLT"
    ],
    "10": [
      "R10SHLT"
    ],
    "11": [
      "R11SHLT"
    ],
    "12": [
      "R12SHLT"
    ],
    "13": [
      "R13SHLT"
    ],
    "14": [
      "R14SHLT"
    ]
  },
  "hypertension_dx": {
    "8": [
      "R8HIBPE"
    ],
    "9": [
      "R9HIBPE"
    ],
    "10": [
      "R10HIBPE"
    ],
    "11": [
      "R11HIBPE"
    ],
    "12": [
      "R12HIBPE"
    ],
    "13": [
      "R13HIBPE"
    ],
    "14": [
      "R14HIBPE"
    ]
  },
  "diabetes_dx": {
    "8": [
      "R8DIABE"
    ],
    "9": [
      "R9DIABE"
    ],
    "10": [
      "R10DIABE"
    ],
    "11": [
      "R11DIABE"
    ],
    "12": [
      "R12DIABE"
    ],
    "13": [
      "R13DIABE"
    ],
    "14": [
      "R14DIABE"
    ]
  },
  "cancer_dx": {
    "8": [
      "R8CANCRE"
    ],
    "9": [
      "R9CANCRE"
    ],
    "10": [
      "R10CANCRE"
    ],
    "11": [
      "R11CANCRE"
    ],
    "12": [
      "R12CANCRE"
    ],
    "13": [
      "R13CANCRE"
    ],
    "14": [
      "R14CANCRE"
    ]
  },
  "lung_disease_dx": {
    "8": [
      "R8LUNGE"
    ],
    "9": [
      "R9LUNGE"
    ],
    "10": [
      "R10LUNGE"
    ],
    "11": [
      "R11LUNGE"
    ],
    "12": [
      "R12LUNGE"
    ],
    "13": [
      "R13LUNGE"
    ],
    "14": [
      "R14LUNGE"
    ]
  },
  "heart_disease_dx": {
    "8": [
      "R8HEARTE"
    ],
    "9": [
      "R9HEARTE"
    ],
    "10": [
      "R10HEARTE"
    ],
    "11": [
      "R11HEARTE"
    ],
    "12": [
      "R12HEARTE"
    ],
    "13": [
      "R13HEARTE"
    ],
    "14": [
      "R14HEARTE"
    ]
  },
  "stroke_dx": {
    "8": [
      "R8STROKE"
    ],
    "9": [
      "R9STROKE"
    ],
    "10": [
      "R10STROKE"
    ],
    "11": [
      "R11STROKE"
    ],
    "12": [
      "R12STROKE"
    ],
    "13": [
      "R13STROKE"
    ],
    "14": [
      "R14STROKE"
    ]
  },
  "psychiatric_dx": {
    "8": [
      "R8PSYCHE"
    ],
    "9": [
      "R9PSYCHE"
    ],
    "10": [
      "R10PSYCHE"
    ],
    "11": [
      "R11PSYCHE"
    ],
    "12": [
      "R12PSYCHE"
    ],
    "13": [
      "R13PSYCHE"
    ],
    "14": [
      "R14PSYCHE"
    ]
  },
  "arthritis_dx": {
    "8": [
      "R8ARTHRE2"
    ],
    "9": [
      "R9ARTHRE2"
    ],
    "10": [
      "R10ARTHRE2"
    ],
    "11": [
      "R11ARTHRE2"
    ],
    "12": [
      "R12ARTHRE2"
    ],
    "13": [
      "R13ARTHRE2"
    ],
    "14": [
      "R14ARTHRE2"
    ]
  },
  "memory_disease_dx": {
    # W8-W9: broad memory-related disease; W10+: OR of Alzheimer and dementia ever-had.
    "8": [
      "R8MEMRYE"
    ],
    "9": [
      "R9MEMRYE"
    ],
    "10": [
      "R10ALZHEE",
      "R10DEMENE"
    ],
    "11": [
      "R11ALZHEE",
      "R11DEMENE"
    ],
    "12": [
      "R12ALZHEE",
      "R12DEMENE"
    ],
    "13": [
      "R13ALZHEE",
      "R13DEMENE"
    ],
    "14": [
      "R14ALZHEE",
      "R14DEMENE"
    ]
  },
  "multimorbidity_count": {
    "8": [
      "R8CONDE"
    ],
    "9": [
      "R9CONDE"
    ],
    "10": [
      "R10CONDE"
    ],
    "11": [
      "R11CONDE"
    ],
    "12": [
      "R12CONDE"
    ],
    "13": [
      "R13CONDE"
    ],
    "14": [
      "R14CONDE"
    ]
  },
  "eyesight": {
    "8": [
      "R8EYERT",
      "R8SIGHT"
    ],
    "9": [
      "R9EYERT",
      "R9SIGHT"
    ],
    "10": [
      "R10EYERT",
      "R10SIGHT"
    ],
    "11": [
      "R11EYERT",
      "R11SIGHT"
    ],
    "12": [
      "R12EYERT",
      "R12SIGHT"
    ],
    "13": [
      "R13EYERT",
      "R13SIGHT"
    ],
    "14": [
      "R14EYERT",
      "R14SIGHT"
    ]
  },
  "near_vision": {
    "8": [
      "R8EYENR"
    ],
    "9": [
      "R9EYENR"
    ],
    "10": [
      "R10EYENR"
    ],
    "11": [
      "R11EYENR"
    ],
    "12": [
      "R12EYENR"
    ],
    "13": [
      "R13EYENR"
    ],
    "14": [
      "R14EYENR"
    ]
  },
  "distance_vision": {
    "8": [
      "R8EYEFR"
    ],
    "9": [
      "R9EYEFR"
    ],
    "10": [
      "R10EYEFR"
    ],
    "11": [
      "R11EYEFR"
    ],
    "12": [
      "R12EYEFR"
    ],
    "13": [
      "R13EYEFR"
    ],
    "14": [
      "R14EYEFR"
    ]
  },
  "hearing": {
    "8": [
      "R8EARRT"
    ],
    "9": [
      "R9EARRT"
    ],
    "10": [
      "R10EARRT"
    ],
    "11": [
      "R11EARRT"
    ],
    "12": [
      "R12EARRT"
    ],
    "13": [
      "R13EARRT"
    ],
    "14": [
      "R14EARRT"
    ]
  },
  "hearing_aid_use": {
    "8": [
      "R8EARAID"
    ],
    "9": [
      "R9EARAID"
    ],
    "10": [
      "R10EARAID"
    ],
    "11": [
      "R11EARAID"
    ],
    "12": [
      "R12EARAID"
    ],
    "13": [
      "R13EARAID"
    ],
    "14": [
      "R14EARAID"
    ]
  },
  "pain_presence": {
    "8": [
      "R8PAINFR"
    ],
    "9": [
      "R9PAINFR"
    ],
    "10": [
      "R10PAINFR"
    ],
    "11": [
      "R11PAINFR"
    ],
    "12": [
      "R12PAINFR"
    ],
    "13": [
      "R13PAINFR"
    ],
    "14": [
      "R14PAINFR"
    ]
  },
  "pain_severity": {
    "8": [
      "R8PAINLV"
    ],
    "9": [
      "R9PAINLV"
    ],
    "10": [
      "R10PAINLV"
    ],
    "11": [
      "R11PAINLV"
    ],
    "12": [
      "R12PAINLV"
    ],
    "13": [
      "R13PAINLV"
    ],
    "14": [
      "R14PAINLV"
    ]
  },
  "falls_any": {
    "8": [
      "R8FALL"
    ],
    "9": [
      "R9FALL"
    ],
    "10": [
      "R10FALL"
    ],
    "11": [
      "R11FALL"
    ],
    "12": [
      "R12FALL"
    ],
    "13": [
      "R13FALL"
    ],
    "14": [
      "R14FALL"
    ]
  },
  "falls_count": {
    "8": [
      "R8FALLNUM"
    ],
    "9": [
      "R9FALLNUM"
    ],
    "10": [
      "R10FALLNUM"
    ],
    "11": [
      "R11FALLNUM"
    ],
    "12": [
      "R12FALLNUM"
    ],
    "13": [
      "R13FALLNUM"
    ],
    "14": [
      "R14FALLNUM"
    ]
  },
  "fall_injury": {
    "8": [
      "R8FALLINJ"
    ],
    "9": [
      "R9FALLINJ"
    ],
    "10": [
      "R10FALLINJ"
    ],
    "11": [
      "R11FALLINJ"
    ],
    "12": [
      "R12FALLINJ"
    ],
    "13": [
      "R13FALLINJ"
    ],
    "14": [
      "R14FALLINJ"
    ]
  },
  "urinary_incontinence": {
    "8": [
      "R8URINAI"
    ],
    "9": [
      "R9URINAI"
    ],
    "10": [
      "R10URINAI"
    ],
    "11": [
      "R11URINAI"
    ],
    "12": [
      "R12URINAI"
    ],
    "13": [
      "R13URINAI"
    ],
    "14": [
      "R14URINAI"
    ]
  },
  "back_problem": {
    "8": [
      "R8BACK",
      "R8BACKP"
    ],
    "9": [
      "R9BACK",
      "R9BACKP"
    ],
    "10": [
      "R10BACK",
      "R10BACKP"
    ],
    "11": [
      "R11BACK",
      "R11BACKP"
    ],
    "12": [
      "R12BACK",
      "R12BACKP"
    ],
    "13": [
      "R13BACK",
      "R13BACKP"
    ],
    "14": [
      "R14BACK",
      "R14BACKP"
    ]
  },
  "sleep_falling_problem": {
    "8": [
      "R8SLEEPFAL"
    ],
    "9": [
      "R9SLEEPFAL"
    ],
    "10": [
      "R10SLEEPFAL"
    ],
    "11": [
      "R11SLEEPFAL"
    ],
    "12": [
      "R12SLEEPFAL"
    ],
    "13": [
      "R13SLEEPFAL"
    ],
    "14": [
      "R14SLEEPFAL"
    ]
  },
  "sleep_waking_problem": {
    "8": [
      "R8SLEEPWKN"
    ],
    "9": [
      "R9SLEEPWKN"
    ],
    "10": [
      "R10SLEEPWKN"
    ],
    "11": [
      "R11SLEEPWKN"
    ],
    "12": [
      "R12SLEEPWKN"
    ],
    "13": [
      "R13SLEEPWKN"
    ],
    "14": [
      "R14SLEEPWKN"
    ]
  },
  "sleep_early_waking": {
    "8": [
      "R8SLEEPWKE"
    ],
    "9": [
      "R9SLEEPWKE"
    ],
    "10": [
      "R10SLEEPWKE"
    ],
    "11": [
      "R11SLEEPWKE"
    ],
    "12": [
      "R12SLEEPWKE"
    ],
    "13": [
      "R13SLEEPWKE"
    ],
    "14": [
      "R14SLEEPWKE"
    ]
  },
  "rested_in_morning": {
    "8": [
      "R8SLEEPRT"
    ],
    "9": [
      "R9SLEEPRT"
    ],
    "10": [
      "R10SLEEPRT"
    ],
    "11": [
      "R11SLEEPRT"
    ],
    "12": [
      "R12SLEEPRT"
    ],
    "13": [
      "R13SLEEPRT"
    ],
    "14": [
      "R14SLEEPRT"
    ]
  },
  "shortness_of_breath": {
    "8": [
      "R8BREATH"
    ],
    "9": [
      "R9BREATH"
    ],
    "10": [
      "R10BREATH"
    ],
    "11": [
      "R11BREATH"
    ],
    "12": [
      "R12BREATH"
    ],
    "13": [
      "R13BREATH"
    ],
    "14": [
      "R14BREATH"
    ]
  },
  "dizziness": {
    "8": [
      "R8DIZZY"
    ],
    "9": [
      "R9DIZZY"
    ],
    "10": [
      "R10DIZZY"
    ],
    "11": [
      "R11DIZZY"
    ],
    "12": [
      "R12DIZZY"
    ],
    "13": [
      "R13DIZZY"
    ],
    "14": [
      "R14DIZZY"
    ]
  },
  "fatigue": {
    "8": [
      "R8FATIGUE"
    ],
    "9": [
      "R9FATIGUE"
    ],
    "10": [
      "R10FATIGUE"
    ],
    "11": [
      "R11FATIGUE"
    ],
    "12": [
      "R12FATIGUE"
    ],
    "13": [
      "R13FATIGUE"
    ],
    "14": [
      "R14FATIGUE"
    ]
  },
  "adl_walk_room": {
    "8": [
      "R8WALKRA"
    ],
    "9": [
      "R9WALKRA"
    ],
    "10": [
      "R10WALKRA"
    ],
    "11": [
      "R11WALKRA"
    ],
    "12": [
      "R12WALKRA"
    ],
    "13": [
      "R13WALKRA"
    ],
    "14": [
      "R14WALKRA"
    ]
  },
  "adl_dress": {
    "8": [
      "R8DRESSA"
    ],
    "9": [
      "R9DRESSA"
    ],
    "10": [
      "R10DRESSA"
    ],
    "11": [
      "R11DRESSA"
    ],
    "12": [
      "R12DRESSA"
    ],
    "13": [
      "R13DRESSA"
    ],
    "14": [
      "R14DRESSA"
    ]
  },
  "adl_bath": {
    "8": [
      "R8BATHA"
    ],
    "9": [
      "R9BATHA"
    ],
    "10": [
      "R10BATHA"
    ],
    "11": [
      "R11BATHA"
    ],
    "12": [
      "R12BATHA"
    ],
    "13": [
      "R13BATHA"
    ],
    "14": [
      "R14BATHA"
    ]
  },
  "adl_eat": {
    "8": [
      "R8EATA"
    ],
    "9": [
      "R9EATA"
    ],
    "10": [
      "R10EATA"
    ],
    "11": [
      "R11EATA"
    ],
    "12": [
      "R12EATA"
    ],
    "13": [
      "R13EATA"
    ],
    "14": [
      "R14EATA"
    ]
  },
  "adl_bed_transfer": {
    "8": [
      "R8BEDA"
    ],
    "9": [
      "R9BEDA"
    ],
    "10": [
      "R10BEDA"
    ],
    "11": [
      "R11BEDA"
    ],
    "12": [
      "R12BEDA"
    ],
    "13": [
      "R13BEDA"
    ],
    "14": [
      "R14BEDA"
    ]
  },
  "adl_toilet": {
    "8": [
      "R8TOILTA"
    ],
    "9": [
      "R9TOILTA"
    ],
    "10": [
      "R10TOILTA"
    ],
    "11": [
      "R11TOILTA"
    ],
    "12": [
      "R12TOILTA"
    ],
    "13": [
      "R13TOILTA"
    ],
    "14": [
      "R14TOILTA"
    ]
  },
  "iadl_phone": {
    "8": [
      "R8PHONEA"
    ],
    "9": [
      "R9PHONEA"
    ],
    "10": [
      "R10PHONEA"
    ],
    "11": [
      "R11PHONEA"
    ],
    "12": [
      "R12PHONEA"
    ],
    "13": [
      "R13PHONEA"
    ],
    "14": [
      "R14PHONEA"
    ]
  },
  "iadl_money": {
    "8": [
      "R8MONEYA"
    ],
    "9": [
      "R9MONEYA"
    ],
    "10": [
      "R10MONEYA"
    ],
    "11": [
      "R11MONEYA"
    ],
    "12": [
      "R12MONEYA"
    ],
    "13": [
      "R13MONEYA"
    ],
    "14": [
      "R14MONEYA"
    ]
  },
  "iadl_medication": {
    "8": [
      "R8MEDSA"
    ],
    "9": [
      "R9MEDSA"
    ],
    "10": [
      "R10MEDSA"
    ],
    "11": [
      "R11MEDSA"
    ],
    "12": [
      "R12MEDSA"
    ],
    "13": [
      "R13MEDSA"
    ],
    "14": [
      "R14MEDSA"
    ]
  },
  "iadl_shopping": {
    "8": [
      "R8SHOPA"
    ],
    "9": [
      "R9SHOPA"
    ],
    "10": [
      "R10SHOPA"
    ],
    "11": [
      "R11SHOPA"
    ],
    "12": [
      "R12SHOPA"
    ],
    "13": [
      "R13SHOPA"
    ],
    "14": [
      "R14SHOPA"
    ]
  },
  "iadl_meals": {
    "8": [
      "R8MEALSA"
    ],
    "9": [
      "R9MEALSA"
    ],
    "10": [
      "R10MEALSA"
    ],
    "11": [
      "R11MEALSA"
    ],
    "12": [
      "R12MEALSA"
    ],
    "13": [
      "R13MEALSA"
    ],
    "14": [
      "R14MEALSA"
    ]
  },
  "difficulty_walk_several_blocks": {
    "8": [
      "R8WALKSA"
    ],
    "9": [
      "R9WALKSA"
    ],
    "10": [
      "R10WALKSA"
    ],
    "11": [
      "R11WALKSA"
    ],
    "12": [
      "R12WALKSA"
    ],
    "13": [
      "R13WALKSA"
    ],
    "14": [
      "R14WALKSA"
    ]
  },
  "difficulty_walk_one_block": {
    "8": [
      "R8WALK1A"
    ],
    "9": [
      "R9WALK1A"
    ],
    "10": [
      "R10WALK1A"
    ],
    "11": [
      "R11WALK1A"
    ],
    "12": [
      "R12WALK1A"
    ],
    "13": [
      "R13WALK1A"
    ],
    "14": [
      "R14WALK1A"
    ]
  },
  "difficulty_sit_two_hours": {
    "8": [
      "R8SITA"
    ],
    "9": [
      "R9SITA"
    ],
    "10": [
      "R10SITA"
    ],
    "11": [
      "R11SITA"
    ],
    "12": [
      "R12SITA"
    ],
    "13": [
      "R13SITA"
    ],
    "14": [
      "R14SITA"
    ]
  },
  "difficulty_rise_chair": {
    "8": [
      "R8CHAIRA"
    ],
    "9": [
      "R9CHAIRA"
    ],
    "10": [
      "R10CHAIRA"
    ],
    "11": [
      "R11CHAIRA"
    ],
    "12": [
      "R12CHAIRA"
    ],
    "13": [
      "R13CHAIRA"
    ],
    "14": [
      "R14CHAIRA"
    ]
  },
  "difficulty_climb_several_flights": {
    "8": [
      "R8CLIMSA"
    ],
    "9": [
      "R9CLIMSA"
    ],
    "10": [
      "R10CLIMSA"
    ],
    "11": [
      "R11CLIMSA"
    ],
    "12": [
      "R12CLIMSA"
    ],
    "13": [
      "R13CLIMSA"
    ],
    "14": [
      "R14CLIMSA"
    ]
  },
  "difficulty_climb_one_flight": {
    "8": [
      "R8CLIM1A"
    ],
    "9": [
      "R9CLIM1A"
    ],
    "10": [
      "R10CLIM1A"
    ],
    "11": [
      "R11CLIM1A"
    ],
    "12": [
      "R12CLIM1A"
    ],
    "13": [
      "R13CLIM1A"
    ],
    "14": [
      "R14CLIM1A"
    ]
  },
  "difficulty_stoop": {
    "8": [
      "R8STOOPA"
    ],
    "9": [
      "R9STOOPA"
    ],
    "10": [
      "R10STOOPA"
    ],
    "11": [
      "R11STOOPA"
    ],
    "12": [
      "R12STOOPA"
    ],
    "13": [
      "R13STOOPA"
    ],
    "14": [
      "R14STOOPA"
    ]
  },
  "difficulty_lift_10lb": {
    "8": [
      "R8LIFTA"
    ],
    "9": [
      "R9LIFTA"
    ],
    "10": [
      "R10LIFTA"
    ],
    "11": [
      "R11LIFTA"
    ],
    "12": [
      "R12LIFTA"
    ],
    "13": [
      "R13LIFTA"
    ],
    "14": [
      "R14LIFTA"
    ]
  },
  "difficulty_pick_dime": {
    "8": [
      "R8DIMEA"
    ],
    "9": [
      "R9DIMEA"
    ],
    "10": [
      "R10DIMEA"
    ],
    "11": [
      "R11DIMEA"
    ],
    "12": [
      "R12DIMEA"
    ],
    "13": [
      "R13DIMEA"
    ],
    "14": [
      "R14DIMEA"
    ]
  },
  "difficulty_reach_arms": {
    "8": [
      "R8ARMSA"
    ],
    "9": [
      "R9ARMSA"
    ],
    "10": [
      "R10ARMSA"
    ],
    "11": [
      "R11ARMSA"
    ],
    "12": [
      "R12ARMSA"
    ],
    "13": [
      "R13ARMSA"
    ],
    "14": [
      "R14ARMSA"
    ]
  },
  "difficulty_push_large_object": {
    "8": [
      "R8PUSHA"
    ],
    "9": [
      "R9PUSHA"
    ],
    "10": [
      "R10PUSHA"
    ],
    "11": [
      "R11PUSHA"
    ],
    "12": [
      "R12PUSHA"
    ],
    "13": [
      "R13PUSHA"
    ],
    "14": [
      "R14PUSHA"
    ]
  },
  "adl_total_score": {
    "8": [
      "R8ADL6A"
    ],
    "9": [
      "R9ADL6A"
    ],
    "10": [
      "R10ADL6A"
    ],
    "11": [
      "R11ADL6A"
    ],
    "12": [
      "R12ADL6A"
    ],
    "13": [
      "R13ADL6A"
    ],
    "14": [
      "R14ADL6A"
    ]
  },
  "iadl_total_score": {
    "8": [
      "R8IADL5A"
    ],
    "9": [
      "R9IADL5A"
    ],
    "10": [
      "R10IADL5A"
    ],
    "11": [
      "R11IADL5A"
    ],
    "12": [
      "R12IADL5A"
    ],
    "13": [
      "R13IADL5A"
    ],
    "14": [
      "R14IADL5A"
    ]
  },
  "mobility_total_score": {
    "8": [
      "R8MOBILA"
    ],
    "9": [
      "R9MOBILA"
    ],
    "10": [
      "R10MOBILA"
    ],
    "11": [
      "R11MOBILA"
    ],
    "12": [
      "R12MOBILA"
    ],
    "13": [
      "R13MOBILA"
    ],
    "14": [
      "R14MOBILA"
    ]
  },
  "large_muscle_total_score": {
    "8": [
      "R8LGMUSA"
    ],
    "9": [
      "R9LGMUSA"
    ],
    "10": [
      "R10LGMUSA"
    ],
    "11": [
      "R11LGMUSA"
    ],
    "12": [
      "R12LGMUSA"
    ],
    "13": [
      "R13LGMUSA"
    ],
    "14": [
      "R14LGMUSA"
    ]
  },
  "fine_motor_total_score": {
    "8": [
      "R8FINEA"
    ],
    "9": [
      "R9FINEA"
    ],
    "10": [
      "R10FINEA"
    ],
    "11": [
      "R11FINEA"
    ],
    "12": [
      "R12FINEA"
    ],
    "13": [
      "R13FINEA"
    ],
    "14": [
      "R14FINEA"
    ]
  },
  "receives_adl_help": {
    "8": [
      "R8ADL6H"
    ],
    "9": [
      "R9ADL6H"
    ],
    "10": [
      "R10ADL6H"
    ],
    "11": [
      "R11ADL6H"
    ],
    "12": [
      "R12ADL6H"
    ],
    "13": [
      "R13ADL6H"
    ],
    "14": [
      "R14ADL6H"
    ]
  },
  "receives_iadl_help": {
    "8": [
      "R8IADL5H"
    ],
    "9": [
      "R9IADL5H"
    ],
    "10": [
      "R10IADL5H"
    ],
    "11": [
      "R11IADL5H"
    ],
    "12": [
      "R12IADL5H"
    ],
    "13": [
      "R13IADL5H"
    ],
    "14": [
      "R14IADL5H"
    ]
  },
  "self_rated_memory": {
    "8": [
      "R8SLFMEM"
    ],
    "9": [
      "R9SLFMEM"
    ],
    "10": [
      "R10SLFMEM"
    ],
    "11": [
      "R11SLFMEM"
    ],
    "12": [
      "R12SLFMEM"
    ],
    "13": [
      "R13SLFMEM"
    ],
    "14": [
      "R14SLFMEM"
    ]
  },
  "immediate_word_recall": {
    "8": [
      "R8IMRC"
    ],
    "9": [
      "R9IMRC"
    ],
    "10": [
      "R10IMRC"
    ],
    "11": [
      "R11IMRC"
    ],
    "12": [
      "R12IMRC"
    ],
    "13": [
      "R13IMRC"
    ],
    "14": [
      "R14IMRCP",
      "R14IMRCW",
      "R14IMRC"
    ]
  },
  "delayed_word_recall": {
    "8": [
      "R8DLRC"
    ],
    "9": [
      "R9DLRC"
    ],
    "10": [
      "R10DLRC"
    ],
    "11": [
      "R11DLRC"
    ],
    "12": [
      "R12DLRC"
    ],
    "13": [
      "R13DLRC"
    ],
    "14": [
      "R14DLRCP",
      "R14DLRCW",
      "R14DLRC"
    ]
  },
  "serial_sevens": {
    "8": [
      "R8SER7"
    ],
    "9": [
      "R9SER7"
    ],
    "10": [
      "R10SER7"
    ],
    "11": [
      "R11SER7"
    ],
    "12": [
      "R12SER7"
    ],
    "13": [
      "R13SER7"
    ],
    "14": [
      "R14SER7P",
      "R14SER7W",
      "R14SER7"
    ]
  },
  "backward_counting": {
    "8": [
      "R8BWC20"
    ],
    "9": [
      "R9BWC20"
    ],
    "10": [
      "R10BWC20"
    ],
    "11": [
      "R11BWC20"
    ],
    "12": [
      "R12BWC20"
    ],
    "13": [
      "R13BWC20"
    ],
    "14": [
      "R14BWC20P",
      "R14BWC20W",
      "R14BWC20"
    ]
  },
  "total_word_recall": {
    "8": [
      "R8TR20"
    ],
    "9": [
      "R9TR20"
    ],
    "10": [
      "R10TR20"
    ],
    "11": [
      "R11TR20"
    ],
    "12": [
      "R12TR20"
    ],
    "13": [
      "R13TR20"
    ],
    "14": [
      "R14TR20P",
      "R14TR20W",
      "R14TR20"
    ]
  },
  "mental_status_score": {
    "8": [
      "R8MSTOT"
    ],
    "9": [
      "R9MSTOT"
    ],
    "10": [
      "R10MSTOT"
    ],
    "11": [
      "R11MSTOT"
    ],
    "12": [
      "R12MSTOT"
    ],
    "13": [
      "R13MSTOT"
    ],
    "14": [
      "R14MSTOTP",
      "R14MSTOT"
    ]
  },
  "total_cognition_score": {
    "8": [
      "R8COGTOT"
    ],
    "9": [
      "R9COGTOT"
    ],
    "10": [
      "R10COGTOT"
    ],
    "11": [
      "R11COGTOT"
    ],
    "12": [
      "R12COGTOT"
    ],
    "13": [
      "R13COGTOT"
    ],
    "14": [
      "R14COGTOTP",
      "R14COGTOT"
    ]
  },
  "cognition_27_score": {
    "8": [
      "R8COG27"
    ],
    "9": [
      "R9COG27"
    ],
    "10": [
      "R10COG27"
    ],
    "11": [
      "R11COG27"
    ],
    "12": [
      "R12COG27"
    ],
    "13": [
      "R13COG27"
    ],
    "14": [
      "R14COG27"
    ]
  },
  "proxy_memory_rating": {
    "8": [
      "R8PRMEM"
    ],
    "9": [
      "R9PRMEM"
    ],
    "10": [
      "R10PRMEM"
    ],
    "11": [
      "R11PRMEM"
    ],
    "12": [
      "R12PRMEM"
    ],
    "13": [
      "R13PRMEM"
    ],
    "14": [
      "R14PRMEM"
    ]
  },
  "proxy_memory_change": {
    "8": [
      "R8PRCHMEM"
    ],
    "9": [
      "R9PRCHMEM"
    ],
    "10": [
      "R10PRCHMEM"
    ],
    "11": [
      "R11PRCHMEM"
    ],
    "12": [
      "R12PRCHMEM"
    ],
    "13": [
      "R13PRCHMEM"
    ],
    "14": [
      "R14PRCHMEM"
    ]
  },
  "cesd_depressed": {
    "8": [
      "R8DEPRES"
    ],
    "9": [
      "R9DEPRES"
    ],
    "10": [
      "R10DEPRES"
    ],
    "11": [
      "R11DEPRES"
    ],
    "12": [
      "R12DEPRES"
    ],
    "13": [
      "R13DEPRES"
    ],
    "14": [
      "R14DEPRES"
    ]
  },
  "cesd_effort": {
    "8": [
      "R8EFFORT"
    ],
    "9": [
      "R9EFFORT"
    ],
    "10": [
      "R10EFFORT"
    ],
    "11": [
      "R11EFFORT"
    ],
    "12": [
      "R12EFFORT"
    ],
    "13": [
      "R13EFFORT"
    ],
    "14": [
      "R14EFFORT"
    ]
  },
  "cesd_restless_sleep": {
    "8": [
      "R8SLEEPR"
    ],
    "9": [
      "R9SLEEPR"
    ],
    "10": [
      "R10SLEEPR"
    ],
    "11": [
      "R11SLEEPR"
    ],
    "12": [
      "R12SLEEPR"
    ],
    "13": [
      "R13SLEEPR"
    ],
    "14": [
      "R14SLEEPR"
    ]
  },
  "cesd_happy": {
    "8": [
      "R8WHAPPY"
    ],
    "9": [
      "R9WHAPPY"
    ],
    "10": [
      "R10WHAPPY"
    ],
    "11": [
      "R11WHAPPY"
    ],
    "12": [
      "R12WHAPPY"
    ],
    "13": [
      "R13WHAPPY"
    ],
    "14": [
      "R14WHAPPY"
    ]
  },
  "cesd_lonely": {
    "8": [
      "R8FLONE"
    ],
    "9": [
      "R9FLONE"
    ],
    "10": [
      "R10FLONE"
    ],
    "11": [
      "R11FLONE"
    ],
    "12": [
      "R12FLONE"
    ],
    "13": [
      "R13FLONE"
    ],
    "14": [
      "R14FLONE"
    ]
  },
  "cesd_sad": {
    "8": [
      "R8FSAD"
    ],
    "9": [
      "R9FSAD"
    ],
    "10": [
      "R10FSAD"
    ],
    "11": [
      "R11FSAD"
    ],
    "12": [
      "R12FSAD"
    ],
    "13": [
      "R13FSAD"
    ],
    "14": [
      "R14FSAD"
    ]
  },
  "cesd_could_not_get_going": {
    "8": [
      "R8GOING"
    ],
    "9": [
      "R9GOING"
    ],
    "10": [
      "R10GOING"
    ],
    "11": [
      "R11GOING"
    ],
    "12": [
      "R12GOING"
    ],
    "13": [
      "R13GOING"
    ],
    "14": [
      "R14GOING"
    ]
  },
  "cesd_enjoyed_life": {
    "8": [
      "R8ENLIFE"
    ],
    "9": [
      "R9ENLIFE"
    ],
    "10": [
      "R10ENLIFE"
    ],
    "11": [
      "R11ENLIFE"
    ],
    "12": [
      "R12ENLIFE"
    ],
    "13": [
      "R13ENLIFE"
    ],
    "14": [
      "R14ENLIFE"
    ]
  },
  "cesd_score": {
    "8": [
      "R8CESD"
    ],
    "9": [
      "R9CESD"
    ],
    "10": [
      "R10CESD"
    ],
    "11": [
      "R11CESD"
    ],
    "12": [
      "R12CESD"
    ],
    "13": [
      "R13CESD"
    ],
    "14": [
      "R14CESD"
    ]
  },
  "life_satisfaction": {
    "8": [
      "R8LBSATWLF"
    ],
    "9": [
      "R9LBSATWLF"
    ],
    "10": [
      "R10LBSATWLF"
    ],
    "11": [
      "R11LBSATWLF"
    ],
    "12": [
      "R12LBSATWLF"
    ],
    "13": [
      "R13LBSATWLF"
    ],
    "14": [
      "R14LBSATWLF"
    ]
  },
  "loneliness_score": {
    "8": [
      "R8LBLONELY3",
      "R8LNLYS3"
    ],
    "9": [
      "R9LBLONELY3",
      "R9LNLYS3"
    ],
    "10": [
      "R10LBLONELY3",
      "R10LNLYS3"
    ],
    "11": [
      "R11LBLONELY3",
      "R11LNLYS3"
    ],
    "12": [
      "R12LBLONELY3",
      "R12LNLYS3"
    ],
    "13": [
      "R13LBLONELY3",
      "R13LNLYS3"
    ],
    "14": [
      "R14LBLONELY3",
      "R14LNLYS3"
    ]
  },
  "positive_affect_score": {
    "8": [
      "R8LBPOSAFFECT6",
      "R8LBPOSAFFECT"
    ],
    "9": [
      "R9LBPOSAFFECT",
      "R9LBPOSAFFECT6"
    ],
    "10": [
      "R10LBPOSAFFECT",
      "R10LBPOSAFFECT6"
    ],
    "11": [
      "R11LBPOSAFFECT",
      "R11LBPOSAFFECT6"
    ],
    "12": [
      "R12LBPOSAFFECT",
      "R12LBPOSAFFECT6"
    ],
    "13": [
      "R13LBPOSAFFECT",
      "R13LBPOSAFFECT6"
    ],
    "14": [
      "R14LBPOSAFFECT",
      "R14LBPOSAFFECT6"
    ]
  },
  "negative_affect_score": {
    "8": [
      "R8LBNEGAFFECT6",
      "R8LBNEGAFFECT"
    ],
    "9": [
      "R9LBNEGAFFECT",
      "R9LBNEGAFFECT6"
    ],
    "10": [
      "R10LBNEGAFFECT",
      "R10LBNEGAFFECT6"
    ],
    "11": [
      "R11LBNEGAFFECT",
      "R11LBNEGAFFECT6"
    ],
    "12": [
      "R12LBNEGAFFECT",
      "R12LBNEGAFFECT6"
    ],
    "13": [
      "R13LBNEGAFFECT",
      "R13LBNEGAFFECT6"
    ],
    "14": [
      "R14LBNEGAFFECT",
      "R14LBNEGAFFECT6"
    ]
  },
  "chronic_stress_count": {
    "8": [
      "R8LBONCHRSTR"
    ],
    # Leave-behind chronic stress scale was not fielded in wave 9.
    "9": [],
    "10": [
      "R10LBONCHRSTR"
    ],
    "11": [
      "R11LBONCHRSTR"
    ],
    "12": [
      "R12LBONCHRSTR"
    ],
    "13": [
      "R13LBONCHRSTR"
    ],
    "14": [
      "R14LBONCHRSTR"
    ]
  },
  "social_support_spouse": {
    "8": [
      "R8SSUPPORT"
    ],
    "9": [
      "R9SSUPPORT"
    ],
    "10": [
      "R10SSUPPORT"
    ],
    "11": [
      "R11SSUPPORT"
    ],
    "12": [
      "R12SSUPPORT"
    ],
    "13": [
      "R13SSUPPORT"
    ],
    "14": [
      "R14SSUPPORT"
    ]
  },
  "social_contact_children": {
    "8": [
      "R8KCNT"
    ],
    "9": [
      "R9KCNT"
    ],
    "10": [
      "R10KCNT"
    ],
    "11": [
      "R11KCNT"
    ],
    "12": [
      "R12KCNT"
    ],
    "13": [
      "R13KCNT"
    ],
    "14": [
      "R14KCNT"
    ]
  },
  "social_contact_friends": {
    "8": [
      "R8RFCNT"
    ],
    "9": [
      "R9RFCNT"
    ],
    "10": [
      "R10RFCNT"
    ],
    "11": [
      "R11RFCNT"
    ],
    "12": [
      "R12RFCNT"
    ],
    "13": [
      "R13RFCNT"
    ],
    "14": [
      "R14RFCNT"
    ]
  },
  "neighborhood_disorder": {
    "8": [
      "R8NPDISUM"
    ],
    "9": [
      "R9NPDISUM"
    ],
    "10": [
      "R10NPDISUM"
    ],
    "11": [
      "R11NPDISUM"
    ],
    "12": [
      "R12NPDISUM"
    ],
    "13": [
      "R13NPDISUM"
    ],
    "14": [
      "R14NPDISUM"
    ]
  },
  "neighborhood_cohesion": {
    "8": [
      "R8NSOCOSUM"
    ],
    "9": [
      "R9NSOCOSUM"
    ],
    "10": [
      "R10NSOCOSUM"
    ],
    "11": [
      "R11NSOCOSUM"
    ],
    "12": [
      "R12NSOCOSUM"
    ],
    "13": [
      "R13NSOCOSUM"
    ],
    "14": [
      "R14NSOCOSUM"
    ]
  },
  "everyday_discrimination": {
    "8": [
      "R8DSCRIM5",
      "R8DSCRIM"
    ],
    "9": [
      "R9DSCRIM5",
      "R9DSCRIM"
    ],
    "10": [
      "R10DSCRIM5",
      "R10DSCRIM"
    ],
    "11": [
      "R11DSCRIM5",
      "R11DSCRIM"
    ],
    "12": [
      "R12DSCRIM5",
      "R12DSCRIM"
    ],
    "13": [
      "R13DSCRIM5",
      "R13DSCRIM"
    ],
    "14": [
      "R14DSCRIM5",
      "R14DSCRIM"
    ]
  },
  "systolic_bp": {
    "8": [
      "R8SYSTO"
    ],
    "9": [
      "R9SYSTO"
    ],
    "10": [
      "R10SYSTO"
    ],
    "11": [
      "R11SYSTO"
    ],
    "12": [
      "R12SYSTO"
    ],
    "13": [
      "R13SYSTO"
    ],
    "14": [
      "R14SYSTO"
    ]
  },
  "diastolic_bp": {
    "8": [
      "R8DIASTO"
    ],
    "9": [
      "R9DIASTO"
    ],
    "10": [
      "R10DIASTO"
    ],
    "11": [
      "R11DIASTO"
    ],
    "12": [
      "R12DIASTO"
    ],
    "13": [
      "R13DIASTO"
    ],
    "14": [
      "R14DIASTO"
    ]
  },
  "resting_pulse": {
    "8": [
      "R8PULSE"
    ],
    "9": [
      "R9PULSE"
    ],
    "10": [
      "R10PULSE"
    ],
    "11": [
      "R11PULSE"
    ],
    "12": [
      "R12PULSE"
    ],
    "13": [
      "R13PULSE"
    ],
    "14": [
      "R14PULSE"
    ]
  },
  "peak_expiratory_flow": {
    "8": [
      "R8PUFF"
    ],
    "9": [
      "R9PUFF"
    ],
    "10": [
      "R10PUFF"
    ],
    "11": [
      "R11PUFF"
    ],
    "12": [
      "R12PUFF"
    ],
    "13": [
      "R13PUFF"
    ],
    "14": [
      "R14PUFF"
    ]
  },
  "grip_strength": {
    "8": [
      "R8GRIPSUM"
    ],
    "9": [
      "R9GRIPSUM"
    ],
    "10": [
      "R10GRIPSUM"
    ],
    "11": [
      "R11GRIPSUM"
    ],
    "12": [
      "R12GRIPSUM"
    ],
    "13": [
      "R13GRIPSUM"
    ],
    "14": [
      "R14GRIPSUM"
    ]
  },
  "left_grip_strength": {
    "8": [
      "R8LGRIP"
    ],
    "9": [
      "R9LGRIP"
    ],
    "10": [
      "R10LGRIP"
    ],
    "11": [
      "R11LGRIP"
    ],
    "12": [
      "R12LGRIP"
    ],
    "13": [
      "R13LGRIP"
    ],
    "14": [
      "R14LGRIP"
    ]
  },
  "right_grip_strength": {
    "8": [
      "R8RGRIP"
    ],
    "9": [
      "R9RGRIP"
    ],
    "10": [
      "R10RGRIP"
    ],
    "11": [
      "R11RGRIP"
    ],
    "12": [
      "R12RGRIP"
    ],
    "13": [
      "R13RGRIP"
    ],
    "14": [
      "R14RGRIP"
    ]
  },
  "balance_score": {
    "8": [
      "R8BALANCE"
    ],
    "9": [
      "R9BALANCE"
    ],
    "10": [
      "R10BALANCE"
    ],
    "11": [
      "R11BALANCE"
    ],
    "12": [
      "R12BALANCE"
    ],
    "13": [
      "R13BALANCE"
    ],
    "14": [
      "R14BALANCE"
    ]
  },
  "walking_speed_time": {
    "8": [
      "R8WSPEED"
    ],
    "9": [
      "R9WSPEED"
    ],
    "10": [
      "R10WSPEED"
    ],
    "11": [
      "R11WSPEED"
    ],
    "12": [
      "R12WSPEED"
    ],
    "13": [
      "R13WSPEED"
    ],
    "14": [
      "R14WSPEED"
    ]
  },
  "measured_height": {
    "8": [
      "R8MHEIGHT"
    ],
    "9": [
      "R9MHEIGHT"
    ],
    "10": [
      "R10MHEIGHT"
    ],
    "11": [
      "R11MHEIGHT"
    ],
    "12": [
      "R12MHEIGHT"
    ],
    "13": [
      "R13MHEIGHT"
    ],
    "14": [
      "R14MHEIGHT"
    ]
  },
  "measured_weight": {
    "8": [
      "R8MWEIGHT"
    ],
    "9": [
      "R9MWEIGHT"
    ],
    "10": [
      "R10MWEIGHT"
    ],
    "11": [
      "R11MWEIGHT"
    ],
    "12": [
      "R12MWEIGHT"
    ],
    "13": [
      "R13MWEIGHT"
    ],
    "14": [
      "R14MWEIGHT"
    ]
  },
  "waist_circumference": {
    "8": [
      "R8MWAIST"
    ],
    "9": [
      "R9MWAIST"
    ],
    "10": [
      "R10MWAIST"
    ],
    "11": [
      "R11MWAIST"
    ],
    "12": [
      "R12MWAIST"
    ],
    "13": [
      "R13MWAIST"
    ],
    "14": [
      "R14MWAIST"
    ]
  },
  "measured_bmi": {
    "8": [
      "R8MBMI"
    ],
    "9": [
      "R9MBMI"
    ],
    "10": [
      "R10MBMI"
    ],
    "11": [
      "R11MBMI"
    ],
    "12": [
      "R12MBMI"
    ],
    "13": [
      "R13MBMI"
    ],
    "14": [
      "R14MBMI"
    ]
  },
  "hospitalization": {
    "8": [
      "R8HOSP"
    ],
    "9": [
      "R9HOSP"
    ],
    "10": [
      "R10HOSP"
    ],
    "11": [
      "R11HOSP"
    ],
    "12": [
      "R12HOSP"
    ],
    "13": [
      "R13HOSP"
    ],
    "14": [
      "R14HOSP"
    ]
  },
  "hospital_stays_count": {
    "8": [
      "R8HSPTIM"
    ],
    "9": [
      "R9HSPTIM"
    ],
    "10": [
      "R10HSPTIM"
    ],
    "11": [
      "R11HSPTIM"
    ],
    "12": [
      "R12HSPTIM"
    ],
    "13": [
      "R13HSPTIM"
    ],
    "14": [
      "R14HSPTIM"
    ]
  },
  "hospital_nights": {
    "8": [
      "R8HSPNIT"
    ],
    "9": [
      "R9HSPNIT"
    ],
    "10": [
      "R10HSPNIT"
    ],
    "11": [
      "R11HSPNIT"
    ],
    "12": [
      "R12HSPNIT"
    ],
    "13": [
      "R13HSPNIT"
    ],
    "14": [
      "R14HSPNIT"
    ]
  },
  "nursing_home_use": {
    "8": [
      "R8NRSHOM"
    ],
    "9": [
      "R9NRSHOM"
    ],
    "10": [
      "R10NRSHOM"
    ],
    "11": [
      "R11NRSHOM"
    ],
    "12": [
      "R12NRSHOM"
    ],
    "13": [
      "R13NRSHOM"
    ],
    "14": [
      "R14NRSHOM"
    ]
  },
  "nursing_home_stays_count": {
    "8": [
      "R8NRSTIM"
    ],
    "9": [
      "R9NRSTIM"
    ],
    "10": [
      "R10NRSTIM"
    ],
    "11": [
      "R11NRSTIM"
    ],
    "12": [
      "R12NRSTIM"
    ],
    "13": [
      "R13NRSTIM"
    ],
    "14": [
      "R14NRSTIM"
    ]
  },
  "doctor_visits_any": {
    "8": [
      "R8DOCTOR"
    ],
    "9": [
      "R9DOCTOR"
    ],
    "10": [
      "R10DOCTOR"
    ],
    "11": [
      "R11DOCTOR"
    ],
    "12": [
      "R12DOCTOR"
    ],
    "13": [
      "R13DOCTOR"
    ],
    "14": [
      "R14DOCTOR"
    ]
  },
  "doctor_visits_count": {
    "8": [
      "R8DOCTIM"
    ],
    "9": [
      "R9DOCTIM"
    ],
    "10": [
      "R10DOCTIM"
    ],
    "11": [
      "R11DOCTIM"
    ],
    "12": [
      "R12DOCTIM"
    ],
    "13": [
      "R13DOCTIM"
    ],
    "14": [
      "R14DOCTIM"
    ]
  },
  "home_health_care": {
    "8": [
      "R8HOMCAR"
    ],
    "9": [
      "R9HOMCAR"
    ],
    "10": [
      "R10HOMCAR"
    ],
    "11": [
      "R11HOMCAR"
    ],
    "12": [
      "R12HOMCAR"
    ],
    "13": [
      "R13HOMCAR"
    ],
    "14": [
      "R14HOMCAR"
    ]
  },
  "outpatient_surgery": {
    "8": [
      "R8OUTPT"
    ],
    "9": [
      "R9OUTPT"
    ],
    "10": [
      "R10OUTPT"
    ],
    "11": [
      "R11OUTPT"
    ],
    "12": [
      "R12OUTPT"
    ],
    "13": [
      "R13OUTPT"
    ],
    "14": [
      "R14OUTPT"
    ]
  },
  "dental_visit": {
    "8": [
      "R8DENTST"
    ],
    "9": [
      "R9DENTST"
    ],
    "10": [
      "R10DENTST"
    ],
    "11": [
      "R11DENTST"
    ],
    "12": [
      "R12DENTST"
    ],
    "13": [
      "R13DENTST"
    ],
    "14": [
      "R14DENTST"
    ]
  },
  "prescription_drug_use": {
    "8": [
      "R8DRUGS"
    ],
    "9": [
      "R9DRUGS"
    ],
    "10": [
      "R10DRUGS"
    ],
    "11": [
      "R11DRUGS"
    ],
    "12": [
      "R12DRUGS"
    ],
    "13": [
      "R13DRUGS"
    ],
    "14": [
      "R14DRUGS"
    ]
  },
  "out_of_pocket_medical_cost": {
    "8": [
      "R8OOPMD"
    ],
    "9": [
      "R9OOPMD"
    ],
    "10": [
      "R10OOPMD"
    ],
    "11": [
      "R11OOPMD"
    ],
    "12": [
      "R12OOPMD"
    ],
    "13": [
      "R13OOPMD"
    ],
    "14": [
      "R14OOPMD"
    ]
  },
  "out_of_pocket_medical_cost_extended": {
    "8": [],
    "9": [],
    "10": [
      "R10OOPMDO"
    ],
    "11": [
      "R11OOPMDO"
    ],
    "12": [
      "R12OOPMDO"
    ],
    "13": [
      "R13OOPMDO"
    ],
    "14": [
      "R14OOPMDO"
    ]
  },
  "smoking_status": {
    "8": [
      "R8SMOKEV",
      "R8SMOKEN"
    ],
    "9": [
      "R9SMOKEV",
      "R9SMOKEN"
    ],
    "10": [
      "R10SMOKEV",
      "R10SMOKEN"
    ],
    "11": [
      "R11SMOKEV",
      "R11SMOKEN"
    ],
    "12": [
      "R12SMOKEV",
      "R12SMOKEN"
    ],
    "13": [
      "R13SMOKEV",
      "R13SMOKEN"
    ],
    "14": [
      "R14SMOKEV",
      "R14SMOKEN"
    ]
  },
  "cigarettes_per_day": {
    "8": [
      "R8SMOKEF"
    ],
    "9": [
      "R9SMOKEF"
    ],
    "10": [
      "R10SMOKEF"
    ],
    "11": [
      "R11SMOKEF"
    ],
    "12": [
      "R12SMOKEF"
    ],
    "13": [
      "R13SMOKEF"
    ],
    "14": [
      "R14SMOKEF"
    ]
  },
  "alcohol_use": {
    "8": [
      "R8DRINK"
    ],
    "9": [
      "R9DRINK"
    ],
    "10": [
      "R10DRINK"
    ],
    "11": [
      "R11DRINK"
    ],
    "12": [
      "R12DRINK"
    ],
    "13": [
      "R13DRINK"
    ],
    "14": [
      "R14DRINK"
    ]
  },
  "alcohol_days_per_week": {
    "8": [
      "R8DRINKD"
    ],
    "9": [
      "R9DRINKD"
    ],
    "10": [
      "R10DRINKD"
    ],
    "11": [
      "R11DRINKD"
    ],
    "12": [
      "R12DRINKD"
    ],
    "13": [
      "R13DRINKD"
    ],
    "14": [
      "R14DRINKD"
    ]
  },
  "drinks_per_drinking_day": {
    "8": [
      "R8DRINKN"
    ],
    "9": [
      "R9DRINKN"
    ],
    "10": [
      "R10DRINKN"
    ],
    "11": [
      "R11DRINKN"
    ],
    "12": [
      "R12DRINKN"
    ],
    "13": [
      "R13DRINKN"
    ],
    "14": [
      "R14DRINKN"
    ]
  },
  "binge_drinking": {
    "8": [
      "R8DRINKB"
    ],
    "9": [
      "R9DRINKB"
    ],
    "10": [
      "R10DRINKB"
    ],
    "11": [
      "R11DRINKB"
    ],
    "12": [
      "R12DRINKB"
    ],
    "13": [
      "R13DRINKB"
    ],
    "14": [
      "R14DRINKB"
    ]
  },
  "vigorous_activity_frequency": {
    "8": [
      "R8VGACTX"
    ],
    "9": [
      "R9VGACTX"
    ],
    "10": [
      "R10VGACTX"
    ],
    "11": [
      "R11VGACTX"
    ],
    "12": [
      "R12VGACTX"
    ],
    "13": [
      "R13VGACTX"
    ],
    "14": [
      "R14VGACTX"
    ]
  },
  "moderate_activity_frequency": {
    "8": [
      "R8MDACTX"
    ],
    "9": [
      "R9MDACTX"
    ],
    "10": [
      "R10MDACTX"
    ],
    "11": [
      "R11MDACTX"
    ],
    "12": [
      "R12MDACTX"
    ],
    "13": [
      "R13MDACTX"
    ],
    "14": [
      "R14MDACTX"
    ]
  },
  "light_activity_frequency": {
    "8": [
      "R8LTACTX"
    ],
    "9": [
      "R9LTACTX"
    ],
    "10": [
      "R10LTACTX"
    ],
    "11": [
      "R11LTACTX"
    ],
    "12": [
      "R12LTACTX"
    ],
    "13": [
      "R13LTACTX"
    ],
    "14": [
      "R14LTACTX"
    ]
  },
  "hypertension_treatment": {
    "8": [
      "R8RXHIBP"
    ],
    "9": [
      "R9RXHIBP"
    ],
    "10": [
      "R10RXHIBP"
    ],
    "11": [
      "R11RXHIBP"
    ],
    "12": [
      "R12RXHIBP"
    ],
    "13": [
      "R13RXHIBP"
    ],
    "14": [
      "R14RXHIBP"
    ]
  },
  "diabetes_oral_medication": {
    "8": [
      "R8RXDIABO",
      "R8DBORLMED"
    ],
    "9": [
      "R9RXDIABO",
      "R9DBORLMED"
    ],
    "10": [
      "R10RXDIABO",
      "R10DBORLMED"
    ],
    "11": [
      "R11RXDIABO",
      "R11DBORLMED"
    ],
    "12": [
      "R12RXDIABO",
      "R12DBORLMED"
    ],
    "13": [
      "R13RXDIABO",
      "R13DBORLMED"
    ],
    "14": [
      "R14RXDIABO",
      "R14DBORLMED"
    ]
  }
}
STATIC_CANDIDATES = {
  "person_id": [
    "HHIDPN"
  ],
  "household_id": [
    "HHID"
  ],
  "person_number": [
    "PN"
  ],
  "wave": [],
  "interview_year": [
    "R8IWEND",
    "R8IWBEG"
  ],
  "interview_month": [
    "R8IWEND",
    "R8IWBEG"
  ],
  "interview_date": [
    "R8IWEND",
    "R8IWBEG"
  ],
  "delta_time_years": [],
  "respondent_weight": [
    "R8WTRESP"
  ],
  "interview_status": [
    "R8IWSTAT"
  ],
  "sex": [
    "RAGENDER"
  ],
  "race_ethnicity": [
    "RARACEM",
    "RAHISPAN"
  ],
  "birth_year": [
    "RABYEAR"
  ],
  "birth_month": [
    "RABMONTH"
  ],
  "education_years": [
    "RAEDYRS"
  ],
  "education_level": [
    "RAEDUCL"
  ],
  "foreign_born": [
    "RABPLACE"
  ],
  "age_at_us_arrival": [
    "RAARRIAGE"
  ],
  "childhood_health": [
    "RACHSHLT"
  ],
  "childhood_ses": [
    "RAMEDUC",
    "RAFEDUC"
  ],
  "mother_education": [
    "RAMEDUC"
  ],
  "father_education": [
    "RAFEDUC"
  ],
  "longest_job_occupation": [
    "RALJOC",
    "R8JLOCC",
    "R9JLOCC",
    "R10JLOCC",
    "R11JLOCC",
    "R12JLOCC",
    "R13JLOCC",
    "R14JLOCC"
  ],
  "longest_job_tenure": [
    "RALJTEN",
    "R8JLTEN",
    "R9JLTEN",
    "R10JLTEN",
    "R11JLTEN",
    "R12JLTEN",
    "R13JLTEN",
    "R14JLTEN"
  ],
  "age_years": [
    "R8AGEY_E"
  ],
  "marital_status": [
    "R8MSTAT"
  ],
  "household_size": [
    "H8HHRES"
  ],
  "living_children": [
    "H8CHILD"
  ],
  "living_siblings": [
    "R8LIVSIB",
    "R9LIVSIB",
    "R10LIVSIB",
    "R11LIVSIB",
    "R12LIVSIB",
    "R13LIVSIB",
    "R14LIVSIB",
  ],
  "living_alone": [
    "H8HHRES"
  ],
  "nursing_home_residence": [
    "R8NHMLIV"
  ],
  "urban_rural_residence": [
    "H8RURAL",
    "H9RURAL",
    "H10RURAL",
    "H11RURAL",
    "H12RURAL",
    "H13RURAL",
    "H14RURAL",
  ],
  "home_ownership": [
    "H8AHOUS"
  ],
  "housing_type": [
    "R8HOMETYP"
  ],
  "total_household_income": [
    "H8ITOT"
  ],
  "total_household_wealth": [
    "H8ATOTB"
  ],
  "poverty_threshold": [
    "H8POVTHR"
  ],
  "income_to_poverty_ratio": [
    "H8ITOT",
    "H8POVTHR"
  ],
  "labor_force_status": [
    "R8LBRF"
  ],
  "currently_working": [
    "R8WORK",
    "R8LBRF"
  ],
  "self_employed": [
    "R8SLFEMP"
  ],
  "hours_worked_per_week": [
    "R8JHOURS"
  ],
  "weeks_worked_per_year": [
    "R8JWEEKS"
  ],
  "current_occupation": [
    "R8JCOCCB"
  ],
  "job_tenure": [
    "R8JCTEN"
  ],
  "health_limits_work": [
    "R8HLTHLM"
  ],
  "health_insurance_any": [
    "R8HIGOV",
    "R8COVR",
    "R8COVS",
    "R8HIOTHP"
  ],
  "medicare_coverage": [
    "R8GOVMR"
  ],
  "medicaid_coverage": [
    "R8GOVMD"
  ],
  "va_coverage": [
    "R8GOVVA"
  ],
  "long_term_care_insurance": [
    "R8HILTC"
  ],
  "self_rated_health": [
    "R8SHLT"
  ],
  "hypertension_dx": [
    "R8HIBPE"
  ],
  "diabetes_dx": [
    "R8DIABE"
  ],
  "cancer_dx": [
    "R8CANCRE"
  ],
  "lung_disease_dx": [
    "R8LUNGE"
  ],
  "heart_disease_dx": [
    "R8HEARTE"
  ],
  "stroke_dx": [
    "R8STROKE"
  ],
  "psychiatric_dx": [
    "R8PSYCHE"
  ],
  "arthritis_dx": [
    "R8ARTHRE2"
  ],
  "memory_disease_dx": [
    "R8MEMRYE",
    "R10ALZHEE",
    "R10DEMENE"
  ],
  "multimorbidity_count": [
    "R8CONDE"
  ],
  "eyesight": [
    "R8EYERT",
    "R8SIGHT"
  ],
  "near_vision": [
    "R8EYENR"
  ],
  "distance_vision": [
    "R8EYEFR"
  ],
  "hearing": [
    "R8EARRT"
  ],
  "hearing_aid_use": [
    "R8EARAID"
  ],
  "pain_presence": [
    "R8PAINFR"
  ],
  "pain_severity": [
    "R8PAINLV"
  ],
  "falls_any": [
    "R8FALL"
  ],
  "falls_count": [
    "R8FALLNUM"
  ],
  "fall_injury": [
    "R8FALLINJ"
  ],
  "urinary_incontinence": [
    "R8URINAI"
  ],
  "back_problem": [
    "R8BACK",
    "R8BACKP"
  ],
  "sleep_falling_problem": [
    "R8SLEEPFAL"
  ],
  "sleep_waking_problem": [
    "R8SLEEPWKN"
  ],
  "sleep_early_waking": [
    "R8SLEEPWKE"
  ],
  "rested_in_morning": [
    "R8SLEEPRT"
  ],
  "shortness_of_breath": [
    "R8BREATH"
  ],
  "dizziness": [
    "R8DIZZY"
  ],
  "fatigue": [
    "R8FATIGUE"
  ],
  "adl_walk_room": [
    "R8WALKRA"
  ],
  "adl_dress": [
    "R8DRESSA"
  ],
  "adl_bath": [
    "R8BATHA"
  ],
  "adl_eat": [
    "R8EATA"
  ],
  "adl_bed_transfer": [
    "R8BEDA"
  ],
  "adl_toilet": [
    "R8TOILTA"
  ],
  "iadl_phone": [
    "R8PHONEA"
  ],
  "iadl_money": [
    "R8MONEYA"
  ],
  "iadl_medication": [
    "R8MEDSA"
  ],
  "iadl_shopping": [
    "R8SHOPA"
  ],
  "iadl_meals": [
    "R8MEALSA"
  ],
  "difficulty_walk_several_blocks": [
    "R8WALKSA"
  ],
  "difficulty_walk_one_block": [
    "R8WALK1A"
  ],
  "difficulty_sit_two_hours": [
    "R8SITA"
  ],
  "difficulty_rise_chair": [
    "R8CHAIRA"
  ],
  "difficulty_climb_several_flights": [
    "R8CLIMSA"
  ],
  "difficulty_climb_one_flight": [
    "R8CLIM1A"
  ],
  "difficulty_stoop": [
    "R8STOOPA"
  ],
  "difficulty_lift_10lb": [
    "R8LIFTA"
  ],
  "difficulty_pick_dime": [
    "R8DIMEA"
  ],
  "difficulty_reach_arms": [
    "R8ARMSA"
  ],
  "difficulty_push_large_object": [
    "R8PUSHA"
  ],
  "adl_total_score": [
    "R8ADL6A"
  ],
  "iadl_total_score": [
    "R8IADL5A"
  ],
  "mobility_total_score": [
    "R8MOBILA"
  ],
  "large_muscle_total_score": [
    "R8LGMUSA"
  ],
  "fine_motor_total_score": [
    "R8FINEA"
  ],
  "receives_adl_help": [
    "R8ADL6H"
  ],
  "receives_iadl_help": [
    "R8IADL5H"
  ],
  "self_rated_memory": [
    "R8SLFMEM"
  ],
  "immediate_word_recall": [
    "R8IMRC"
  ],
  "delayed_word_recall": [
    "R8DLRC"
  ],
  "serial_sevens": [
    "R8SER7"
  ],
  "backward_counting": [
    "R8BWC20"
  ],
  "total_word_recall": [
    "R8TR20"
  ],
  "mental_status_score": [
    "R8MSTOT"
  ],
  "total_cognition_score": [
    "R8COGTOT"
  ],
  "cognition_27_score": [
    "R8COG27"
  ],
  "proxy_memory_rating": [
    "R8PRMEM"
  ],
  "proxy_memory_change": [
    "R8PRCHMEM"
  ],
  "cesd_depressed": [
    "R8DEPRES"
  ],
  "cesd_effort": [
    "R8EFFORT"
  ],
  "cesd_restless_sleep": [
    "R8SLEEPR"
  ],
  "cesd_happy": [
    "R8WHAPPY"
  ],
  "cesd_lonely": [
    "R8FLONE"
  ],
  "cesd_sad": [
    "R8FSAD"
  ],
  "cesd_could_not_get_going": [
    "R8GOING"
  ],
  "cesd_enjoyed_life": [
    "R8ENLIFE"
  ],
  "cesd_score": [
    "R8CESD"
  ],
  "life_satisfaction": [
    "R8LBSATWLF"
  ],
  "loneliness_score": [
    "R8LBLONELY3"
  ],
  "positive_affect_score": [
    "R8LBPOSAFFECT6"
  ],
  "negative_affect_score": [
    "R8LBNEGAFFECT6"
  ],
  "chronic_stress_count": [
    "R8LBONCHRSTR"
  ],
  "social_support_spouse": [
    "R8SSUPPORT"
  ],
  "social_contact_children": [
    "R8KCNT"
  ],
  "social_contact_friends": [
    "R8RFCNT"
  ],
  "neighborhood_disorder": [
    "R8NPDISUM"
  ],
  "neighborhood_cohesion": [
    "R8NSOCOSUM"
  ],
  "everyday_discrimination": [
    "R8DSCRIM5"
  ],
  "systolic_bp": [
    "R8SYSTO"
  ],
  "diastolic_bp": [
    "R8DIASTO"
  ],
  "resting_pulse": [
    "R8PULSE"
  ],
  "peak_expiratory_flow": [
    "R8PUFF"
  ],
  "grip_strength": [
    "R8GRIPSUM"
  ],
  "left_grip_strength": [
    "R8LGRIP"
  ],
  "right_grip_strength": [
    "R8RGRIP"
  ],
  "balance_score": [
    "R8BALANCE"
  ],
  "walking_speed_time": [
    "R8WSPEED"
  ],
  "measured_height": [
    "R8MHEIGHT"
  ],
  "measured_weight": [
    "R8MWEIGHT"
  ],
  "waist_circumference": [
    "R8MWAIST"
  ],
  "measured_bmi": [
    "R8MBMI"
  ],
  "hospitalization": [
    "R8HOSP"
  ],
  "hospital_stays_count": [
    "R8HSPTIM"
  ],
  "hospital_nights": [
    "R8HSPNIT"
  ],
  "nursing_home_use": [
    "R8NRSHOM"
  ],
  "nursing_home_stays_count": [
    "R8NRSTIM"
  ],
  "doctor_visits_any": [
    "R8DOCTOR"
  ],
  "doctor_visits_count": [
    "R8DOCTIM"
  ],
  "home_health_care": [
    "R8HOMCAR"
  ],
  "outpatient_surgery": [
    "R8OUTPT"
  ],
  "dental_visit": [
    "R8DENTST"
  ],
  "prescription_drug_use": [
    "R8DRUGS"
  ],
  "out_of_pocket_medical_cost": [
    "R8OOPMD"
  ],
  "out_of_pocket_medical_cost_extended": [],
  "smoking_status": [
    "R8SMOKEN",
    "R8SMOKEV"
  ],
  "cigarettes_per_day": [
    "R8SMOKEF"
  ],
  "alcohol_use": [
    "R8DRINK"
  ],
  "alcohol_days_per_week": [
    "R8DRINKD"
  ],
  "drinks_per_drinking_day": [
    "R8DRINKN"
  ],
  "binge_drinking": [
    "R8DRINKB"
  ],
  "vigorous_activity_frequency": [
    "R8VGACTX"
  ],
  "moderate_activity_frequency": [
    "R8MDACTX"
  ],
  "light_activity_frequency": [
    "R8LTACTX"
  ],
  "hypertension_treatment": [
    "R8RXHIBP"
  ],
  "diabetes_oral_medication": [
    "R8RXDIABO",
    "R8DBORLMED"
  ]
}


DIRECT_DERIVED_VARIABLES = {
    "wave",
    "interview_year",
    "interview_month",
    "interview_date",
    "delta_time_years",
    "race_ethnicity",
    "smoking_status",
    "living_alone",
    "income_to_poverty_ratio",
    "currently_working",
    "health_insurance_any",
    "multimorbidity_count",
    "adl_total_score",
    "iadl_total_score",
    "mobility_total_score",
    "large_muscle_total_score",
    "fine_motor_total_score",
    "total_word_recall",
    "measured_bmi",
}

# Transition-table column naming (aligned with the CHARLS and HRS model loaders).
STATE_PREFIX = "state__"
STATE_MASK_PREFIX = "state_mask__"
NEXT_STATE_PREFIX = "next_state__"
NEXT_STATE_MASK_PREFIX = "next_state_mask__"
ACTION_PREFIX = "action__"
ACTION_MASK_PREFIX = "action_mask__"
ACTION_ELIGIBLE_PREFIX = "action_eligible__"
REWARD_PREFIX = "reward__"
REWARD_MASK_PREFIX = "reward_mask__"

# Clinical rewards are binary "is next_wave severe?":
# 1 = severe at t+1, 0 = observed and non-severe at t+1, NA = next wave unobserved.
# Each tuple is (state_name, reward_name, severe_op, severe_threshold) where
# severe_op is ">=" when a higher score is worse and "<=" when higher is better.
# Thresholds follow each score's coding range:
#   adl_total_score 0-6 (6 ADL items); iadl_total_score 0-5 (5 IADL items);
#   mobility_total_score 0-4 (4 mobility items); cesd_score 0-8 (8-item CES-D);
#   self_rated_health 1=excellent..5=poor.
# cognition_decline is *not* in this list: it copies next-wave cognition_27_score.
WORSENING_REWARD_FROM_STATE: tuple[tuple[str, str, str, float], ...] = (
    ("adl_total_score", "adl_worsening", ">=", 4.0),
    ("iadl_total_score", "iadl_worsening", ">=", 4.0),
    ("mobility_total_score", "mobility_worsening", ">=", 3.0),
    ("cesd_score", "cesd_worsening", ">=", 4.0),
    ("self_rated_health", "self_rated_health_worsening", ">=", 4.0),
)

# Historical name retained; value is the next-wave 27-item cognition score
# (0-27, higher=better), not a binary decline/severe flag.
CONTINUOUS_LEVEL_REWARD_FROM_STATE: tuple[tuple[str, str], ...] = (
    ("cognition_27_score", "cognition_decline"),
)

# Load-time-compatible CVD pool: OR of heart/stroke among people free of both.
CVD_INCIDENT_REWARD = "cvd_incident"
CVD_INCIDENT_SOURCES: tuple[str, ...] = (
    "heart_disease_incident",
    "stroke_incident",
)

# Cost reward: 1 when inflation-adjusted out-of-pocket spending rises by at
# least 20% between the interval ending at wave t and the one ending at t+1.
# A relative threshold keeps small dollar wiggles at the low end from counting
# as a cost shock. The name keeps its historical *_next_interval suffix for
# dictionary compatibility.
OOP_INCREASE_REWARD = "out_of_pocket_medical_expenditure_next_interval"
OOP_INCREASE_FROM_STATE = "out_of_pocket_medical_cost"
OOP_INCREASE_THRESHOLD = 0.20

# Every binary reward derived by ``derive_worsening_rewards`` (Bernoulli 0/1).
# cognition_decline is continuous and is not in this set.
BINARY_WORSENING_REWARDS: frozenset[str] = frozenset(
    [name for _, name, _, _ in WORSENING_REWARD_FROM_STATE] + [OOP_INCREASE_REWARD]
)

# Composite is -(Σ weight × standardized reward). Positive weight = adverse
# binary event. cognition_decline is the next-wave score (higher=better), so
# its weight is negative. cvd_incident is written to the table but omitted
# here so composite does not double-count heart + stroke.
DEFAULT_REWARD_WEIGHTS = {
    "death_event": 10.0,
    "hospitalization_event": 1.0,
    "heart_disease_incident": 2.0,
    "stroke_incident": 2.0,
    "adl_worsening": 0.50,
    "iadl_worsening": 0.50,
    "mobility_worsening": 0.50,
    "cesd_worsening": 0.25,
    "cognition_decline": -0.25,
    "self_rated_health_worsening": 0.25,
    "out_of_pocket_medical_expenditure_next_interval": 0.10,
}


@dataclass(frozen=True)
class VariableSpec:
    name: str
    role: str
    data_type: str
    source_dataset: str
    coding: str
    standard_rule: str
    priority: str


@dataclass
class ExtractionResult:
    values: pd.Series
    observed: pd.Series
    structural_zero: pd.Series
    source_name: str | None
    source_column: str | None


class SourceResolver:
    """Case-insensitive column resolver over multiple person-indexed sources."""

    def __init__(self, sources: Sequence[tuple[str, pd.DataFrame]]) -> None:
        self.sources: list[tuple[str, pd.DataFrame, dict[str, str]]] = []
        for name, frame in sources:
            lookup = {str(c).upper(): str(c) for c in frame.columns}
            self.sources.append((name, frame, lookup))

    def extract(
        self,
        candidates: Sequence[str],
        master_index: pd.Index,
        preferred: str,
    ) -> ExtractionResult:
        ordered = self._ordered_sources(preferred)
        for candidate in candidates:
            key = candidate.upper()
            for source_name, frame, lookup in ordered:
                actual = lookup.get(key)
                if actual is None:
                    continue
                series = frame[actual].reindex(master_index)
                cleaned, observed, structural = clean_stata_missing(series)
                return ExtractionResult(
                    values=cleaned,
                    observed=observed,
                    structural_zero=structural,
                    source_name=source_name,
                    source_column=actual,
                )
        empty = pd.Series(np.nan, index=master_index, dtype="float64")
        zero = pd.Series(0, index=master_index, dtype="int8")
        return ExtractionResult(empty, zero, zero, None, None)

    def _ordered_sources(
        self, preferred: str
    ) -> list[tuple[str, pd.DataFrame, dict[str, str]]]:
        preferred_l = preferred.lower()
        if "rand" in preferred_l:
            keys = ("rand", "harmonized")
        elif "harmonized" in preferred_l:
            keys = ("harmonized", "rand")
        else:
            keys = ("harmonized", "rand")
        ordered: list[tuple[str, pd.DataFrame, dict[str, str]]] = []
        for key in keys:
            ordered.extend([item for item in self.sources if key in item[0].lower()])
        ordered.extend([item for item in self.sources if item not in ordered])
        return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean HRS Waves 8-14 and create train/validation/test transitions."
    )
    parser.add_argument(
        "--rand-file",
        default=str(DEFAULT_RAND_FILE),
        help="RAND HRS longitudinal Stata file.",
    )
    parser.add_argument(
        "--harmonized-file",
        default=str(DEFAULT_HARMONIZED_FILE),
        help="Harmonized HRS Stata file.",
    )
    parser.add_argument(
        "--dictionary",
        default="",
        help="Optional variable workbook. Omitted runs use data/hrs/variable_specs.json, which is not included in this repository.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    parser.add_argument("--waves", nargs="+", type=int, default=DEFAULT_WAVES)
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.15,
        help="Target proportion of unique respondents assigned to validation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Target proportion of unique respondents assigned to test.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--csv-copy", action="store_true")
    parser.add_argument(
        "--min-live-state-features",
        type=int,
        default=1,
        help="Minimum number of observed static/dynamic features required for a live state row.",
    )
    parser.add_argument(
        "--max-transition-years",
        type=float,
        default=6.5,
        help="Maximum live-to-live interval retained for training.",
    )
    parser.add_argument(
        "--reward-weights-json",
        default=None,
        help="Optional JSON file overriding composite reward weights.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--prepare-model-bundle",
        action="store_true",
        help=(
            "Skip full cleaning; rebuild HRS_*.parquet/config/preprocessing "
            "from existing hrs_world_model_transitions in --output-dir."
        ),
    )
    return parser.parse_args()


def load_transitions_table(output_dir: Path) -> pd.DataFrame:
    """Load cleaned transitions from parquet (preferred) or CSV."""
    parquet = output_dir / "hrs_world_model_transitions.parquet"
    csv_path = output_dir / "hrs_world_model_transitions.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv_path.exists():
        return pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    raise FileNotFoundError(
        f"Missing transitions table in {output_dir} "
        "(expected hrs_world_model_transitions.parquet/csv)"
    )


def prepare_model_bundle_from_output(
    output_dir: Path,
    dictionary_path: Path | None = None,
) -> dict[str, Any]:
    """Rebuild model-ready parquet/config from an existing cleaning output directory.

    Split ratios/seed are taken from feature_metadata.json when present so the
    model_config matches the existing person-level ``split`` column.
    """
    output_dir = Path(output_dir)
    transitions = load_transitions_table(output_dir)
    specs = load_specs(dictionary_path)
    meta_path = output_dir / "feature_metadata.json"
    metadata: dict[str, Any] = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    # Clinical/cost rewards: binary severe flags, continuous cognition, CVD pool.
    transitions = derive_transition_rewards(transitions)
    # Refresh composite using current DEFAULT_REWARD_WEIGHTS (excludes model-excluded
    # rewards such as cancer_incident) so modeling composite_reward stays consistent.
    reward_weights = {
        name: float(weight)
        for name, weight in DEFAULT_REWARD_WEIGHTS.items()
        if name not in EXCLUDE_MODEL_REWARD_VARIABLES
    }
    transitions, reward_scales = add_composite_reward(transitions, reward_weights)
    clinical = f"{REWARD_PREFIX}composite_clinical_reward"
    modeling = f"{REWARD_PREFIX}composite_reward"
    transitions[modeling] = transitions[clinical]
    transitions[f"{REWARD_MASK_PREFIX}composite_reward"] = transitions[
        f"{REWARD_MASK_PREFIX}composite_clinical_reward"
    ]
    # Persist refreshed raw transitions so re-opens see the binary rewards.
    transitions_path = output_dir / "hrs_world_model_transitions.parquet"
    transitions.to_parquet(transitions_path, index=False)
    csv_path = output_dir / "hrs_world_model_transitions.csv"
    if csv_path.exists():
        transitions.to_csv(csv_path, index=False, encoding="utf-8-sig")
    LOGGER.info(
        "Wrote refreshed transitions (continuous cognition + cvd_incident): %s",
        transitions_path,
    )
    # Always refresh feature lists from the current dictionary roles.
    split_meta = metadata.get("split", {})
    feature_lists = select_model_features(transitions, specs)
    metadata = {
        **metadata,
        **feature_lists,
        "split": split_meta,
        "reward_weights": reward_weights,
        "reward_scales": reward_scales,
        "reward_representation": "binary_severe_plus_continuous_cognition",
        "worsening_reward_from_state": {
            reward_name: {
                "state": state_name,
                "severe_when": f"{severe_op} {severe_threshold:g}",
                "event": "severe at next wave",
            }
            for state_name, reward_name, severe_op, severe_threshold in WORSENING_REWARD_FROM_STATE
        }
        | {
            OOP_INCREASE_REWARD: {
                "state": OOP_INCREASE_FROM_STATE,
                "worse_when": "increases (2018 USD)",
            }
        },
        "continuous_level_reward_from_state": {
            reward_name: state_name
            for state_name, reward_name in CONTINUOUS_LEVEL_REWARD_FROM_STATE
        },
        "cvd_incident_sources": list(CVD_INCIDENT_SOURCES),
    }
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    split = split_meta
    model_config = write_hrs_model_bundle(
        transitions=transitions,
        specs=specs,
        metadata=metadata,
        output_dir=output_dir,
        dictionary_path=dictionary_path,
        validation_size=float(split.get("validation_size_target", 0.15)),
        test_size=float(split.get("test_size_target", 0.15)),
        seed=int(split.get("seed", 2026)),
    )
    LOGGER.info(
        "HRS model bundle ready in %s (%d state / %d static / %d action / %d reward)",
        output_dir,
        len(model_config["state_columns_model"]),
        len(model_config["static_context_columns"]),
        len(model_config["action_columns_model"]),
        len(model_config["reward_columns"]),
    )
    return model_config


def builtin_specs() -> dict[str, VariableSpec]:
    """Variable roles read from ``data/hrs/variable_specs.json``.

    That file is not included in this repository.
    """
    if not BUILTIN_SPECS_PATH.is_file():
        raise FileNotFoundError(
            f"Variable dictionary is not included in this repository. "
            f"Place it at {BUILTIN_SPECS_PATH}. See data/README.md."
        )
    payload = json.loads(BUILTIN_SPECS_PATH.read_text(encoding="utf-8"))
    specs: dict[str, VariableSpec] = {}
    for row in payload:
        name = str(row["name"]).strip()
        specs[name] = VariableSpec(
            name=name,
            role=str(row["role"]),
            data_type=str(row["data_type"]),
            source_dataset=str(row["source_dataset"]),
            coding="",
            standard_rule="",
            priority=str(row["priority"]),
        )
    if not specs:
        raise ValueError(f"No variable specs in {BUILTIN_SPECS_PATH}")
    return specs


def load_specs(path: Path | None) -> dict[str, VariableSpec]:
    """Read an optional workbook, otherwise the built-in variable specs."""
    if path is not None and str(path) not in {"", "."} and Path(path).exists():
        _, specs = read_dictionary(Path(path))
        return specs
    LOGGER.info("Using built-in variable specs from %s", BUILTIN_SPECS_PATH.name)
    return builtin_specs()


def read_dictionary(path: Path) -> tuple[pd.DataFrame, dict[str, VariableSpec]]:
    dictionary = pd.read_excel(path, sheet_name="Model_variables")
    required = {
        "Variable Name",
        "World-model Role",
        "Data Type",
        "Source Dataset",
        "Coding",
        "HRS → Standard Rule",
        "Model Priority",
    }
    missing = required.difference(dictionary.columns)
    if missing:
        raise ValueError(f"Dictionary is missing required columns: {sorted(missing)}")
    dictionary = dictionary.copy()
    dictionary["Variable Name"] = dictionary["Variable Name"].astype(str).str.strip()
    if dictionary["Variable Name"].duplicated().any():
        dup = dictionary.loc[dictionary["Variable Name"].duplicated(), "Variable Name"].tolist()
        raise ValueError(f"Duplicate standardized variables in dictionary: {dup}")
    specs = {
        row["Variable Name"]: VariableSpec(
            name=row["Variable Name"],
            role=str(row["World-model Role"]),
            data_type=str(row["Data Type"]),
            source_dataset=str(row["Source Dataset"]),
            coding=str(row["Coding"]),
            standard_rule=str(row["HRS → Standard Rule"]),
            priority=str(row["Model Priority"]),
        )
        for _, row in dictionary.iterrows()
    }
    return dictionary, specs


def dta_columns(path: Path) -> list[str]:
    try:
        import pyreadstat

        _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
        return list(meta.column_names)
    except Exception as exc:
        LOGGER.warning("pyreadstat metadata read failed for %s: %s", path, exc)
        reader = pd.read_stata(path, iterator=True, convert_categoricals=False)
        try:
            return list(reader.varlist)
        finally:
            reader.close()


def read_dta_selected(path: Path, requested_upper: set[str]) -> pd.DataFrame:
    available = dta_columns(path)
    lookup = {c.upper(): c for c in available}
    selected = [lookup[c] for c in requested_upper if c in lookup]
    for identifier in ("HHIDPN", "HHID", "PN"):
        if identifier in lookup and lookup[identifier] not in selected:
            selected.append(lookup[identifier])
    if not selected:
        raise ValueError(f"No requested columns were found in {path}")
    LOGGER.info("Reading %s: %d/%d columns", path.name, len(selected), len(available))
    frame = pd.read_stata(
        path,
        columns=selected,
        convert_categoricals=False,
        convert_missing=True,
        preserve_dtypes=True,
    )
    return prepare_person_index(frame, source=str(path))


def prepare_person_index(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    lookup = {c.upper(): c for c in frame.columns}
    if "HHIDPN" in lookup:
        person = frame[lookup["HHIDPN"]].map(normalize_identifier)
    elif "HHID" in lookup and "PN" in lookup:
        hh = frame[lookup["HHID"]].map(normalize_identifier)
        pn = frame[lookup["PN"]].map(normalize_identifier)
        person = hh.str.zfill(6) + pn.str.zfill(3)
    else:
        raise ValueError(f"Cannot construct person ID for {source}; HHIDPN or HHID+PN is required.")
    frame = frame.copy()
    frame.index = pd.Index(person, name="person_id")
    frame = frame.loc[frame.index.notna()]
    if frame.index.duplicated().any():
        duplicate_count = int(frame.index.duplicated().sum())
        LOGGER.warning("%s contains %d duplicate person IDs; keeping first nonmissing values.", source, duplicate_count)
        frame = frame.groupby(level=0, sort=False).first()
    return frame


def normalize_identifier(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return str(int(value))
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def stata_missing_code(value: Any) -> str | None:
    if isinstance(value, StataMissingValue):
        return str(value).strip().lower()
    if isinstance(value, str):
        text = value.strip().lower()
        if OTHER_STATA_MISSING_RE.fullmatch(text):
            return text
    return None


def clean_stata_missing(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Map all Stata special missings (including .x/.q/.s) to NA with observed=0."""
    cleaned: list[Any] = []
    observed: list[int] = []
    structural: list[int] = []
    for value in series.tolist():
        code = stata_missing_code(value)
        if code is not None:
            cleaned.append(np.nan)
            observed.append(0)
            # Keep a legacy structural flag for former .x/.q/.s codes (audit only).
            structural.append(1 if code in STRUCTURAL_ZERO_CODES else 0)
        elif value is None or pd.isna(value):
            cleaned.append(np.nan)
            observed.append(0)
            structural.append(0)
        else:
            cleaned.append(value)
            observed.append(1)
            structural.append(0)
    return (
        pd.Series(cleaned, index=series.index),
        pd.Series(observed, index=series.index, dtype="int8"),
        pd.Series(structural, index=series.index, dtype="int8"),
    )


def collect_requested_columns(specs: Mapping[str, VariableSpec], waves: Sequence[int]) -> set[str]:
    requested = {"HHIDPN", "HHID", "PN", "RARACEM", "RAHISPAN"}
    for name in specs:
        if name in EXCLUDE_STATE_VARIABLES:
            continue
        for candidate in STATIC_CANDIDATES.get(name, []):
            requested.add(candidate.upper())
        for wave in waves:
            for candidate in SOURCE_CANDIDATES.get(name, {}).get(str(wave), []):
                requested.add(candidate.upper())
        # Death/interview-status columns are needed even when reward fields are derived.
        for wave in waves:
            requested.add(f"R{wave}IWSTAT")
    return requested



def preferred_source(spec: VariableSpec) -> str:
    return spec.source_dataset


def extract_static_coalesce(
    resolver: "SourceResolver",
    candidates: Sequence[str],
    master_index: pd.Index,
    preferred: str,
) -> ExtractionResult:
    """Fill person-level static values by first non-missing candidate column."""
    if not candidates:
        empty = pd.Series(np.nan, index=master_index, dtype="float64")
        zero = pd.Series(0, index=master_index, dtype="int8")
        return ExtractionResult(empty, zero, zero, None, None)
    if len(candidates) == 1:
        return resolver.extract(candidates, master_index, preferred)

    values = pd.Series(np.nan, index=master_index, dtype="float64")
    observed = pd.Series(0, index=master_index, dtype="int8")
    structural = pd.Series(0, index=master_index, dtype="int8")
    source_name: str | None = None
    source_columns: list[str] = []
    for candidate in candidates:
        result = resolver.extract([candidate], master_index, preferred)
        if result.source_column is None:
            continue
        fill = values.isna() & result.values.notna()
        if not fill.any():
            continue
        values = values.mask(fill, result.values)
        observed = observed.mask(fill, result.observed)
        structural = structural.mask(fill, result.structural_zero)
        source_columns.append(str(result.source_column))
        if source_name is None:
            source_name = result.source_name
    return ExtractionResult(
        values=values,
        observed=observed,
        structural_zero=structural,
        source_name=source_name,
        source_column=";".join(source_columns) if source_columns else None,
    )


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def to_stata_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    as_date = pd.to_datetime(numeric, unit="D", origin="1960-01-01", errors="coerce")
    text_date = pd.to_datetime(series, errors="coerce")
    return as_date.fillna(text_date)


def recode_binary(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    valid = x.dropna()
    if valid.empty:
        return x
    values = set(valid.unique().tolist())
    if values.issubset({0, 1}):
        return x.where(x.isin([0, 1]))
    if values.issubset({1, 2}):
        return x.map({1: 1.0, 2: 0.0})
    # Prefer explicit 0/1 and common yes/no 1/2; other values remain missing.
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    out.loc[x.eq(1)] = 1.0
    out.loc[x.isin([0, 2])] = 0.0
    return out


def recode_home_ownership(series: pd.Series) -> pd.Series:
    """Map RAND HxAHOUS primary-residence asset value ($) to ownership.

    HxAHOUS is a continuous housing asset amount, not a 0/1 flag. Ownership is
    1 when the reported/imputed value is positive and 0 when it is zero; NA
    stays missing.
    """
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    observed = x.notna()
    out.loc[observed & x.gt(0)] = 1.0
    out.loc[observed & x.le(0)] = 0.0
    return out


def recode_sex(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.map({1: 1.0, 2: 0.0, 0: 0.0})


def recode_foreign_born(series: pd.Series) -> pd.Series:
    """Map RAND RABPLACE (census-region birthplace) to foreign-born indicator.

    RAND codes 1-10/12 are US / US territory; 11 and 13 are not US.
    """
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    out.loc[x.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])] = 0.0
    out.loc[x.isin([11, 13])] = 1.0
    return out


def recode_activity_frequency(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    valid = x.dropna()
    if not valid.empty and valid.min() >= 1 and valid.max() <= 5 and not x.eq(0).any():
        return x.map({1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: 0.0})
    return x.where(x.between(0, 4))


def recode_education_level(series: pd.Series) -> pd.Series:
    """Keep harmonized education level RAEDUCL codes 1-3.

    Coding: 1=less than upper secondary; 2=upper secondary/vocational;
    3=tertiary. Do not fall back to RAND RAEDUC (1-5), whose category-3
    (HS graduate) is not the same as RAEDUCL-3 (tertiary).
    """
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x.isin([1, 2, 3]))


def recode_parental_education_years(series: pd.Series) -> pd.Series:
    """Keep RAND parental education years/categories on 0-17 (incl. 7.5/8.5)."""
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x.between(0, 17))


def recode_marital_status(series: pd.Series) -> pd.Series:
    """Keep RAND RwMSTAT codes 1-8; do not collapse to a coarser scheme."""
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x.isin([1, 2, 3, 4, 5, 6, 7, 8]))


def recode_sleep_likert3(series: pd.Series) -> pd.Series:
    """Keep RAND sleep frequency codes: 1=most, 2=sometimes, 3=rarely/never.

    Direction is left as in RAND (higher = less frequent endorsement).
    """
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x.isin([1, 2, 3]))


def recode_life_satisfaction(series: pd.Series) -> pd.Series:
    """Keep RAND leave-behind life-satisfaction mean (RwLBSATWLF).

    Wave 8 uses a 1-6 item scale; waves 9-14 use 1-7. Values outside 1-7 → NA.
    """
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x.between(1, 7))


def recode_weekly_contact(series: pd.Series) -> pd.Series:
    """Map any-weekly-contact indicators to 0/1; other codes stay NA."""
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x.isin([0, 1]))


def recode_balance_score(series: pd.Series) -> pd.Series:
    """Keep harmonized balance summary codes 1-4."""
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x.isin([1, 2, 3, 4]))


def occupation_census_scheme(source_column: str | None) -> str:
    """Return 'occupc' (2010) or 'occupb' (2000) from the RAND source column name."""
    text = str(source_column or "").upper()
    if "JCOCCC" in text:
        return "occupc"
    return "occupb"


def recode_current_occupation_physical_demand(
    series: pd.Series,
    *,
    scheme: str,
) -> pd.Series:
    """Coarsen major occupation codes to physical-demand ranks.

    Coding after recode:
      1 = sedentary
      2 = light physical
      3 = heavy physical

    ``scheme`` must be ``occupb`` (2000 Census / JCOCCB) or ``occupc``
    (2010 Census / JCOCCC). Unmapped codes → NA.
    """
    key = str(scheme).strip().lower()
    if key == "occupc":
        mapping = OCCUPC_TO_PHYSICAL_DEMAND
    elif key == "occupb":
        mapping = OCCUPB_TO_PHYSICAL_DEMAND
    else:
        raise ValueError(f"Unknown occupation census scheme: {scheme!r}")
    x = pd.to_numeric(series, errors="coerce")
    # Map via rounded integer codes; fractional/invalid → NA.
    codes = x.round()
    out = codes.map(mapping)
    return out.astype("float64")


def standardize_extracted(name: str, spec: VariableSpec, series: pd.Series) -> pd.Series:
    if name == "sex":
        return recode_sex(series)
    if name == "foreign_born":
        return recode_foreign_born(series)
    if name == "home_ownership":
        return recode_home_ownership(series)
    if name == "education_level":
        return recode_education_level(series)
    if name in {"mother_education", "father_education", "childhood_ses"}:
        return recode_parental_education_years(series)
    if name == "marital_status":
        return recode_marital_status(series)
    if name in {
        "sleep_falling_problem",
        "sleep_waking_problem",
        "sleep_early_waking",
        "rested_in_morning",
    }:
        return recode_sleep_likert3(series)
    if name == "life_satisfaction":
        return recode_life_satisfaction(series)
    if name in {"social_contact_children", "social_contact_friends"}:
        return recode_weekly_contact(series)
    if name == "balance_score":
        return recode_balance_score(series)
    if name in {
        "vigorous_activity_frequency",
        "moderate_activity_frequency",
        "light_activity_frequency",
    }:
        return recode_activity_frequency(series)
    if "binary" in spec.data_type.lower():
        return recode_binary(series)
    if "date" in spec.data_type.lower():
        return to_stata_date(series)
    return to_numeric(series)


def derive_childhood_ses(
    resolver: "SourceResolver",
    master_index: pd.Index,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build childhood SES as the mean of available parental education years.

    Revised dictionary coding is 0-17 (incl. 7.5/8.5). Prior coalesce incorrectly
    preferred childhood health (RACHSHLT, 1-5) because RACHFIN/RACHSES are absent.
    """
    mother = resolver.extract(["RAMEDUC"], master_index, "rand")
    father = resolver.extract(["RAFEDUC"], master_index, "rand")
    mom = recode_parental_education_years(mother.values)
    dad = recode_parental_education_years(father.values)
    value = pd.concat([mom, dad], axis=1).mean(axis=1, skipna=True)
    observed = value.notna().astype("int8")
    structural = (
        (mother.structural_zero.fillna(0).astype("int8") > 0)
        | (father.structural_zero.fillna(0).astype("int8") > 0)
    ).astype("int8")
    return value, observed, structural


def combine_identifiers(resolver: SourceResolver, master_index: pd.Index) -> tuple[pd.Series, pd.Series]:
    hh = resolver.extract(["HHID"], master_index, "rand").values.map(normalize_identifier)
    pn = resolver.extract(["PN"], master_index, "rand").values.map(normalize_identifier)
    return hh, pn


def derive_race_ethnicity(
    resolver: SourceResolver, master_index: pd.Index
) -> tuple[pd.Series, pd.Series, pd.Series]:
    race = resolver.extract(["RARACEM"], master_index, "rand")
    hisp = resolver.extract(["RAHISPAN"], master_index, "rand")
    r = pd.to_numeric(race.values, errors="coerce")
    h = pd.to_numeric(hisp.values, errors="coerce")
    out = pd.Series(np.nan, index=master_index, dtype="float64")
    out.loc[h.eq(1)] = 3.0
    non_hisp = h.isin([0, 2]) | h.isna()
    out.loc[non_hisp & r.eq(1)] = 1.0
    out.loc[non_hisp & r.eq(2)] = 2.0
    out.loc[non_hisp & r.notna() & ~r.isin([1, 2])] = 4.0
    observed = ((race.observed.eq(1)) | (hisp.observed.eq(1))).astype("int8")
    structural = ((race.structural_zero.eq(1)) | (hisp.structural_zero.eq(1))).astype("int8")
    return out, observed, structural


def derive_smoking_status(
    resolver: SourceResolver, master_index: pd.Index, wave: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ever = resolver.extract([f"R{wave}SMOKEV"], master_index, "harmonized")
    now = resolver.extract([f"R{wave}SMOKEN"], master_index, "harmonized")
    ever_x = recode_binary(ever.values)
    now_x = recode_binary(now.values)
    out = pd.Series(np.nan, index=master_index, dtype="float64")
    out.loc[ever_x.eq(0)] = 0.0
    out.loc[ever_x.eq(1) & now_x.eq(0)] = 1.0
    out.loc[now_x.eq(1)] = 2.0
    observed = out.notna().astype("int8")
    structural = ((ever.structural_zero.eq(1)) | (now.structural_zero.eq(1))).astype("int8")
    return out, observed, structural


def derive_health_insurance(
    resolver: SourceResolver, master_index: pd.Index, wave: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    candidates = [f"R{wave}HIGOV", f"R{wave}COVR", f"R{wave}COVS", f"R{wave}HIOTHP"]
    parts = [resolver.extract([c], master_index, "rand") for c in candidates]
    values = pd.concat([recode_binary(p.values) for p in parts], axis=1)
    out = pd.Series(np.nan, index=master_index, dtype="float64")
    any_obs = values.notna().any(axis=1)
    out.loc[any_obs] = values.loc[any_obs].fillna(0).max(axis=1)
    observed = any_obs.astype("int8")
    structural = pd.concat([p.structural_zero for p in parts], axis=1).max(axis=1).astype("int8")
    return out, observed, structural


def derive_memory_disease_dx(
    resolver: SourceResolver, master_index: pd.Index, wave: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Unified memory-related diagnosis across the HRS questionnaire change.

    Waves 8-9 use RxMEMRYE. From wave 10 onward RAND replaced that item with
    separate Alzheimer and dementia ever-had indicators; those are OR-combined.
    """
    if int(wave) <= 9:
        result = resolver.extract([f"R{wave}MEMRYE"], master_index, "rand")
        value = recode_binary(result.values)
        observed = result.observed.where(value.notna(), 0).astype("int8")
        return value, observed, result.structural_zero.astype("int8")

    parts = [
        resolver.extract([f"R{wave}ALZHEE"], master_index, "rand"),
        resolver.extract([f"R{wave}DEMENE"], master_index, "rand"),
    ]
    values = pd.concat([recode_binary(p.values) for p in parts], axis=1)
    out = pd.Series(np.nan, index=master_index, dtype="float64")
    any_obs = values.notna().any(axis=1)
    out.loc[any_obs] = values.loc[any_obs].fillna(0).max(axis=1)
    observed = any_obs.astype("int8")
    structural = pd.concat([p.structural_zero for p in parts], axis=1).max(axis=1).astype("int8")
    return out, observed, structural


NEARBY_WAVE_FILL_EXCLUDE = {
    "person_id",
    "household_id",
    "person_number",
    "wave",
    "interview_status",
    "interview_date",
    "interview_year",
    "interview_month",
    "delta_time_years",
    "respondent_weight",
}


def fill_missing_from_nearby_waves(
    long_df: pd.DataFrame,
    variable_names: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill NA values within person from earlier then later observed waves.

    Donors are restricted to rows with ``{name}__observed == 1``. Filled cells
    keep ``observed == 0`` so tensors still expose that the wave itself was
    unobserved; remaining NAs are handled later by training-set imputers.
    """
    if long_df.empty:
        return long_df, {}

    out = long_df.sort_values(["person_id", "wave"]).copy()
    person_ids = out["person_id"]
    fill_counts: dict[str, int] = {}

    for name in variable_names:
        if name in NEARBY_WAVE_FILL_EXCLUDE or name not in out.columns:
            continue
        if name.endswith(("__observed", "__eligible", "__structural_zero")):
            continue

        raw = out[name]
        need = raw.isna()
        if not need.any():
            fill_counts[name] = 0
            continue

        obs_col = f"{name}__observed"
        if obs_col in out.columns:
            observed = pd.to_numeric(out[obs_col], errors="coerce").fillna(0).eq(1)
            seed = raw.where(observed)
        else:
            seed = raw

        filled = seed.groupby(person_ids, sort=False).ffill()
        filled = filled.groupby(person_ids, sort=False).bfill()
        n_filled = int((need & filled.notna()).sum())
        if n_filled:
            out[name] = raw.fillna(filled)
        fill_counts[name] = n_filled

    return out, fill_counts


def backfill_oop_extended_from_wave10(
    long_df: pd.DataFrame,
    target_waves: Sequence[int] = (8, 9),
    donor_wave: int = 10,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Fill W8-W9 OOP extended costs with each person's wave-10 R10OOPMDO value.

    RAND only provides the extended OOP item from wave 10 onward. Person-level
    carry-back keeps early-wave states usable when the same respondent is later
    observed at wave 10.
    """
    name = "out_of_pocket_medical_cost_extended"
    obs_name = f"{name}__observed"
    fill_counts = {int(w): 0 for w in target_waves}
    if name not in long_df.columns or not long_df["wave"].eq(donor_wave).any():
        return long_df, fill_counts

    out = long_df.copy()
    donor = (
        out.loc[out["wave"].eq(donor_wave), ["person_id", name]]
        .assign(**{name: lambda d: pd.to_numeric(d[name], errors="coerce")})
        .dropna(subset=[name])
        .drop_duplicates("person_id", keep="last")
        .set_index("person_id")[name]
    )
    if donor.empty:
        return out, fill_counts

    for wave in target_waves:
        wave = int(wave)
        mask = out["wave"].eq(wave) & pd.to_numeric(out[name], errors="coerce").isna()
        if not mask.any():
            continue
        filled = out.loc[mask, "person_id"].map(donor)
        valid = filled.notna()
        if not valid.any():
            continue
        idx = out.loc[mask].index[valid.to_numpy()]
        out.loc[idx, name] = filled.loc[valid].to_numpy()
        if obs_name in out.columns:
            out.loc[idx, obs_name] = 1
        fill_counts[wave] = int(valid.sum())
    return out, fill_counts


def build_long_data(
    specs: Mapping[str, VariableSpec],
    resolver: SourceResolver,
    master_index: pd.Index,
    waves: Sequence[int],
    min_live_state_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build live long table, interview-status grid, source audit, and pre-fill long raw.

    Returns
    -------
    long_live
        Live person-wave rows after nearby-wave LOCF/NOCB fill (feeds transitions).
    status_grid
        Interview status for every person-wave.
    audit
        Source-column audit rows.
    long_live_raw
        Same live rows as ``long_live`` but *before* nearby-wave fill (CSV export).
    """
    role_names = {
        role: [
            name
            for name, spec in specs.items()
            if spec.role == role and name not in EXCLUDE_STATE_VARIABLES
        ]
        for role in {
            "Auxiliary/metadata",
            "Static state variable",
            "Dynamic state variable",
            "Action variable",
            "Reward variable",
        }
    }
    # Wave-local SES/housing/work vars may be labeled Static in the dictionary
    # (model context only) but must still be extracted per wave.
    wave_local_context = [
        name
        for name in FORCE_STATIC_CONTEXT_VARIABLES
        if name in specs
        and name not in EXCLUDE_STATE_VARIABLES
        and name not in PERSON_LEVEL_STATIC_CONTEXT_VARIABLES
    ]
    static_extract_names = [
        name
        for name in role_names["Static state variable"] + ["birth_year", "birth_month"]
        if name not in wave_local_context
    ]
    dynamic_extract_names = list(
        dict.fromkeys(role_names["Dynamic state variable"] + wave_local_context)
    )
    source_audit: list[dict[str, Any]] = []

    static_values: dict[str, pd.Series] = {}
    static_observed: dict[str, pd.Series] = {}
    static_structural: dict[str, pd.Series] = {}

    for name in static_extract_names:
        if name not in specs or name in EXCLUDE_STATE_VARIABLES:
            continue
        spec = specs[name]
        if name == "race_ethnicity":
            value, obs, structural = derive_race_ethnicity(resolver, master_index)
            static_values[name], static_observed[name], static_structural[name] = value, obs, structural
            source_audit.append({
                "variable": name, "wave": "static", "candidates": "RARACEM; RAHISPAN",
                "source": "derived", "source_column": "RARACEM+RAHISPAN",
                "nonmissing": int(value.notna().sum()), "structural_zero": int(structural.sum()),
            })
            continue
        if name == "childhood_ses":
            value, obs, structural = derive_childhood_ses(resolver, master_index)
            static_values[name], static_observed[name], static_structural[name] = value, obs, structural
            source_audit.append({
                "variable": name,
                "wave": "static",
                "candidates": "RAMEDUC; RAFEDUC",
                "source": "derived",
                "source_column": "mean(RAMEDUC, RAFEDUC)",
                "nonmissing": int(value.notna().sum()),
                "structural_zero": int(structural.sum()),
            })
            continue
        candidates = STATIC_CANDIDATES.get(name, [])
        result = extract_static_coalesce(
            resolver,
            candidates,
            master_index,
            preferred_source(spec),
        )
        value = standardize_extracted(name, spec, result.values)
        static_values[name] = value
        static_observed[name] = result.observed.where(value.notna(), 0).astype("int8")
        static_structural[name] = result.structural_zero.astype("int8")
        source_audit.append({
            "variable": name, "wave": "static", "candidates": "; ".join(candidates),
            "source": result.source_name, "source_column": result.source_column,
            "nonmissing": int(value.notna().sum()), "structural_zero": int(result.structural_zero.sum()),
        })

    hh, pn = combine_identifiers(resolver, master_index)
    all_wave_frames: list[pd.DataFrame] = []
    status_rows: list[pd.DataFrame] = []

    for wave in waves:
        # Collect columns first, then build once — avoids fragmented frame.insert warnings.
        wave_columns: dict[str, Any] = {
            "person_id": master_index.astype(str),
            "household_id": hh,
            "person_number": pn,
            "wave": int(wave),
        }

        status_result = resolver.extract([f"R{wave}IWSTAT"], master_index, "rand")
        status = pd.to_numeric(status_result.values, errors="coerce")
        wave_columns["interview_status"] = status
        wave_columns["interview_status__observed"] = status_result.observed
        status_rows.append(pd.DataFrame({
            "person_id": master_index.astype(str),
            "wave": wave,
            "interview_status": status.values,
        }))

        for name, value in static_values.items():
            wave_columns[name] = value.values
            wave_columns[f"{name}__observed"] = static_observed[name].values

        for role, names in [
            ("Auxiliary/metadata", role_names["Auxiliary/metadata"]),
            ("Dynamic state variable", dynamic_extract_names),
            ("Action variable", role_names["Action variable"]),
        ]:
            for name in names:
                if name in {
                    "person_id", "household_id", "person_number", "wave",
                    "interview_status", "birth_year", "birth_month",
                    "delta_time_years",
                }:
                    continue
                # Already written from person-level static extraction.
                if name in static_values:
                    continue
                spec = specs[name]

                if name == "race_ethnicity":
                    continue
                if name == "smoking_status":
                    value, obs, structural = derive_smoking_status(resolver, master_index, wave)
                    source_col = f"R{wave}SMOKEV+R{wave}SMOKEN"
                    source_name = "derived"
                elif name == "health_insurance_any":
                    value, obs, structural = derive_health_insurance(resolver, master_index, wave)
                    source_col = "+".join(SOURCE_CANDIDATES[name][str(wave)])
                    source_name = "derived"
                elif name == "memory_disease_dx":
                    value, obs, structural = derive_memory_disease_dx(resolver, master_index, wave)
                    source_col = "+".join(SOURCE_CANDIDATES[name][str(wave)])
                    source_name = "derived"
                elif name in {"interview_date", "interview_year", "interview_month"}:
                    date_result = resolver.extract(
                        SOURCE_CANDIDATES["interview_date"][str(wave)],
                        master_index,
                        preferred_source(spec),
                    )
                    dates = to_stata_date(date_result.values)
                    if name == "interview_date":
                        value = dates
                    elif name == "interview_year":
                        value = dates.dt.year.astype("float64")
                        fallback = resolver.extract([f"R{wave}IWY"], master_index, "rand")
                        value = value.fillna(pd.to_numeric(fallback.values, errors="coerce"))
                        value = value.fillna(WAVE_YEAR.get(wave))
                    else:
                        value = dates.dt.month.astype("float64")
                        fallback = resolver.extract([f"R{wave}IWM"], master_index, "rand")
                        value = value.fillna(pd.to_numeric(fallback.values, errors="coerce"))
                    obs = value.notna().astype("int8")
                    structural = date_result.structural_zero
                    source_col = date_result.source_column
                    source_name = date_result.source_name
                elif name in {"living_alone", "income_to_poverty_ratio", "currently_working"}:
                    value = pd.Series(np.nan, index=master_index, dtype="float64")
                    obs = pd.Series(0, index=master_index, dtype="int8")
                    structural = pd.Series(0, index=master_index, dtype="int8")
                    source_col = None
                    source_name = "derived-later"
                else:
                    candidates = SOURCE_CANDIDATES.get(name, {}).get(str(wave), [])
                    result = resolver.extract(
                        candidates,
                        master_index,
                        preferred_source(spec),
                    )
                    if name == "current_occupation":
                        # W8–W9 use OCCUPB; W10–W14 prefer OCCUPC with OCCUPB fallback.
                        # Same numeric codes mean different jobs across schemes.
                        scheme = occupation_census_scheme(result.source_column)
                        value = recode_current_occupation_physical_demand(
                            result.values,
                            scheme=scheme,
                        )
                    else:
                        value = standardize_extracted(name, spec, result.values)
                    obs = result.observed.where(value.notna(), 0).astype("int8")
                    structural = result.structural_zero.astype("int8")
                    source_col = result.source_column
                    source_name = result.source_name

                wave_columns[name] = getattr(value, "values", value)
                wave_columns[f"{name}__observed"] = getattr(obs, "values", obs)
                source_audit.append({
                    "variable": name,
                    "wave": wave,
                    "candidates": "; ".join(SOURCE_CANDIDATES.get(name, {}).get(str(wave), [])),
                    "source": source_name,
                    "source_column": source_col,
                    "nonmissing": int(pd.Series(value).notna().sum()),
                    "structural_zero": int(pd.Series(structural).sum()),
                })

        frame = pd.DataFrame(wave_columns, index=master_index)
        frame = apply_wave_derivations(frame, specs)
        # Refresh audit for variables materialized only after wave-level derivation.
        derived_source_cols = {
            "living_alone": f"H{wave}HHRES==1",
            "income_to_poverty_ratio": f"H{wave}ITOT/H{wave}POVTHR",
            "currently_working": f"R{wave}LBRF in {{1,2}}",
        }
        for row in source_audit:
            if row.get("wave") != wave or row.get("source") != "derived-later":
                continue
            name = row["variable"]
            if name not in frame:
                continue
            values = pd.to_numeric(frame[name], errors="coerce")
            row["source"] = "derived"
            row["source_column"] = derived_source_cols.get(name)
            row["nonmissing"] = int(values.notna().sum())
            row["structural_zero"] = 0
        all_wave_frames.append(frame.reset_index(drop=True))

    long_all = pd.concat(all_wave_frames, ignore_index=True)
    long_all = long_all.sort_values(["person_id", "wave"]).reset_index(drop=True)
    long_all, oop_extended_fill_counts = backfill_oop_extended_from_wave10(long_all)
    for wave, n_filled in oop_extended_fill_counts.items():
        for row in source_audit:
            if row.get("variable") != "out_of_pocket_medical_cost_extended" or row.get("wave") != wave:
                continue
            row["candidates"] = "R10OOPMDO (carry-back)"
            row["source"] = "derived"
            row["source_column"] = "person-level R10OOPMDO"
            row["nonmissing"] = int(
                pd.to_numeric(
                    long_all.loc[long_all["wave"].eq(wave), "out_of_pocket_medical_cost_extended"],
                    errors="coerce",
                ).notna().sum()
            )
            row["structural_zero"] = 0
            LOGGER.info(
                "Filled out_of_pocket_medical_cost_extended for wave %s from wave 10: %s rows",
                wave,
                n_filled,
            )

    # A state is live only when interview status is 1 and sufficient state data are observed.
    state_names = role_names["Static state variable"] + role_names["Dynamic state variable"]
    state_present = long_all[[c for c in state_names if c in long_all]].notna().sum(axis=1)
    live_mask = long_all["interview_status"].eq(1) & state_present.ge(min_live_state_features)
    long_live = long_all.loc[live_mask].copy()

    dates = pd.to_datetime(long_live.get("interview_date"), errors="coerce")
    long_live["delta_time_years"] = (
        dates.groupby(long_live["person_id"]).diff().dt.days / 365.25
    )
    fallback_delta = long_live.groupby("person_id")["interview_year"].diff()
    long_live["delta_time_years"] = long_live["delta_time_years"].fillna(fallback_delta)
    for wave in waves:
        wave_delta = long_live.loc[long_live["wave"].eq(wave), "delta_time_years"]
        source_audit.append({
            "variable": "delta_time_years",
            "wave": wave,
            "candidates": "interview_date.diff; interview_year.diff",
            "source": "derived",
            "source_column": "person-wave interview spacing",
            "nonmissing": int(pd.to_numeric(wave_delta, errors="coerce").notna().sum()),
            "structural_zero": 0,
        })

    nearby_fill_vars = (
        role_names["Static state variable"]
        + role_names["Dynamic state variable"]
        + role_names["Action variable"]
    )
    # Snapshot before LOCF/NOCB so analysts can inspect truly wave-local missings.
    long_live_raw = long_live.copy()
    long_live, nearby_fill_counts = fill_missing_from_nearby_waves(
        long_live, nearby_fill_vars
    )
    nearby_filled_cells = int(sum(nearby_fill_counts.values()))
    nearby_filled_vars = int(sum(1 for n in nearby_fill_counts.values() if n > 0))
    LOGGER.info(
        "Nearby-wave fill: %d missing cells across %d variables (LOCF then NOCB; masks unchanged)",
        nearby_filled_cells,
        nearby_filled_vars,
    )

    status_grid = pd.concat(status_rows, ignore_index=True)
    audit = pd.DataFrame(source_audit)
    structural_marker_columns = [
        c for c in long_live.columns if c.endswith("__structural_zero")
    ]
    if structural_marker_columns:
        long_live = long_live.drop(columns=structural_marker_columns)
        long_live_raw = long_live_raw.drop(
            columns=[c for c in structural_marker_columns if c in long_live_raw.columns]
        )
    return (
        long_live.reset_index(drop=True),
        status_grid,
        audit,
        long_live_raw.reset_index(drop=True),
    )


def apply_wave_derivations(frame: pd.DataFrame, specs: Mapping[str, VariableSpec]) -> pd.DataFrame:
    updates: dict[str, pd.Series] = {}

    def series_of(name: str, default: float | int = np.nan) -> pd.Series:
        if name in updates:
            return updates[name]
        if name in frame:
            return frame[name]
        return pd.Series(default, index=frame.index)

    def set_derived(name: str, value: pd.Series, components: Sequence[str]) -> None:
        if name not in specs:
            return
        existing = pd.to_numeric(series_of(name), errors="coerce")
        value = pd.to_numeric(value, errors="coerce")
        final = existing.fillna(value)
        updates[name] = final
        component_obs = [series_of(f"{c}__observed", 0) for c in components]
        derived_obs = pd.concat(component_obs, axis=1).max(axis=1).astype("int8")
        updates[f"{name}__observed"] = series_of(f"{name}__observed", 0).where(
            existing.notna(), derived_obs
        ).astype("int8")

    if "household_size" in frame:
        set_derived("living_alone", pd.to_numeric(frame["household_size"], errors="coerce").eq(1).astype(float), ["household_size"])
    if "total_household_income" in frame and "poverty_threshold" in frame:
        income = pd.to_numeric(frame["total_household_income"], errors="coerce")
        poverty = pd.to_numeric(frame["poverty_threshold"], errors="coerce")
        ratio = income.div(poverty.where(poverty.gt(0)))
        set_derived("income_to_poverty_ratio", ratio, ["total_household_income", "poverty_threshold"])
    if "labor_force_status" in frame:
        lbrf = pd.to_numeric(frame["labor_force_status"], errors="coerce")
        # RAND/Harmonized codes 1/2 commonly identify working full/part time.
        set_derived("currently_working", lbrf.isin([1, 2]).astype(float).where(lbrf.notna()), ["labor_force_status"])

    disease_names = [
        "hypertension_dx", "diabetes_dx", "cancer_dx", "lung_disease_dx",
        "heart_disease_dx", "stroke_dx", "psychiatric_dx", "arthritis_dx",
        "memory_disease_dx",
    ]
    disease_available = [c for c in disease_names if c in frame]
    if disease_available:
        dx = frame[disease_available].apply(pd.to_numeric, errors="coerce")
        count = dx.fillna(0).sum(axis=1).where(dx.notna().any(axis=1))
        set_derived("multimorbidity_count", count, disease_available)

    adl_items = ["adl_walk_room", "adl_dress", "adl_bath", "adl_eat", "adl_bed_transfer", "adl_toilet"]
    iadl_items = ["iadl_phone", "iadl_money", "iadl_medication", "iadl_shopping", "iadl_meals"]
    mobility_items = [
        "difficulty_walk_several_blocks", "difficulty_walk_one_block",
        "difficulty_climb_several_flights", "difficulty_climb_one_flight",
    ]
    large_muscle_items = [
        "difficulty_sit_two_hours", "difficulty_rise_chair", "difficulty_stoop",
        "difficulty_lift_10lb", "difficulty_push_large_object",
    ]
    fine_items = ["difficulty_pick_dime", "difficulty_reach_arms"]

    for target, items in [
        ("adl_total_score", adl_items),
        ("iadl_total_score", iadl_items),
        ("mobility_total_score", mobility_items),
        ("large_muscle_total_score", large_muscle_items),
        ("fine_motor_total_score", fine_items),
    ]:
        present = [c for c in items if c in frame]
        if present:
            values = frame[present].apply(pd.to_numeric, errors="coerce")
            score = values.fillna(0).sum(axis=1).where(values.notna().any(axis=1))
            set_derived(target, score, present)

    if "immediate_word_recall" in frame and "delayed_word_recall" in frame:
        total = pd.to_numeric(frame["immediate_word_recall"], errors="coerce") + pd.to_numeric(
            frame["delayed_word_recall"], errors="coerce"
        )
        set_derived("total_word_recall", total, ["immediate_word_recall", "delayed_word_recall"])

    if "measured_weight" in frame and "measured_height" in frame:
        weight = pd.to_numeric(frame["measured_weight"], errors="coerce")
        height = pd.to_numeric(frame["measured_height"], errors="coerce")
        # Infer common HRS units: pounds and inches when medians exceed metric ranges.
        positive_weight = weight.where(weight > 0)
        positive_height = height.where(height > 0)
        fit_weight = positive_weight.dropna()
        fit_height = positive_height.dropna()
        if len(fit_weight) and fit_weight.median() > 100:
            weight_kg = positive_weight * 0.45359237
        else:
            weight_kg = positive_weight
        if len(fit_height) and fit_height.median() > 3:
            height_m = positive_height * 0.0254
        else:
            height_m = positive_height
        bmi = weight_kg / height_m.pow(2)
        bmi = bmi.where(
            weight_kg.notna()
            & height_m.notna()
            & weight_kg.gt(0)
            & height_m.gt(0)
            & bmi.between(10, 80)
        )
        set_derived("measured_bmi", bmi, ["measured_weight", "measured_height"])

    # Eligibility-aware action handling.
    if "cigarettes_per_day" in frame and "smoking_status" in frame:
        status = pd.to_numeric(series_of("smoking_status"), errors="coerce")
        cigs = pd.to_numeric(series_of("cigarettes_per_day"), errors="coerce").copy()
        cigs_obs = series_of("cigarettes_per_day__observed", 0).astype("int8").copy()
        mask = status.isin([0, 1])
        cigs.loc[mask] = 0.0
        cigs_obs.loc[mask] = 1
        updates["cigarettes_per_day"] = cigs
        updates["cigarettes_per_day__observed"] = cigs_obs
    if "alcohol_use" in frame:
        use = pd.to_numeric(series_of("alcohol_use"), errors="coerce")
        for action in ["alcohol_days_per_week", "drinks_per_drinking_day", "binge_drinking"]:
            if action in frame or action in updates:
                vals = pd.to_numeric(series_of(action), errors="coerce").copy()
                obs = series_of(f"{action}__observed", 0).astype("int8").copy()
                vals.loc[use.eq(0)] = 0.0
                obs.loc[use.eq(0)] = 1
                updates[action] = vals
                updates[f"{action}__observed"] = obs
    for action, diagnosis in [
        ("hypertension_treatment", "hypertension_dx"),
        ("diabetes_oral_medication", "diabetes_dx"),
    ]:
        if (action in frame or action in updates) and (diagnosis in frame or diagnosis in updates):
            dx = pd.to_numeric(series_of(diagnosis), errors="coerce")
            vals = pd.to_numeric(series_of(action), errors="coerce").copy()
            obs = series_of(f"{action}__observed", 0).astype("int8").copy()
            vals.loc[dx.eq(0)] = 0.0
            obs.loc[dx.eq(0)] = 0
            # Action is zero-filled for tensors, but not eligible for policy optimization.
            updates[action] = vals
            updates[f"{action}__observed"] = obs
            updates[f"{action}__eligible"] = dx.eq(1).astype("int8")
        elif action in frame or action in updates:
            updates[f"{action}__eligible"] = series_of(f"{action}__observed", 0).astype("int8")

    if not updates:
        return frame
    update_df = pd.DataFrame(updates, index=frame.index)
    overlap = [c for c in update_df.columns if c in frame.columns]
    base = frame.drop(columns=overlap) if overlap else frame
    return pd.concat([base, update_df], axis=1)


def build_transitions(
    long_live: pd.DataFrame,
    status_grid: pd.DataFrame,
    specs: Mapping[str, VariableSpec],
    waves: Sequence[int],
    max_transition_years: float,
) -> pd.DataFrame:
    role_names = {
        role: [
            name
            for name, spec in specs.items()
            if spec.role == role and name not in EXCLUDE_STATE_VARIABLES
        ]
        for role in {"Static state variable", "Dynamic state variable", "Action variable", "Reward variable"}
    }
    state_names = role_names["Static state variable"] + role_names["Dynamic state variable"]
    action_names = role_names["Action variable"]

    live_lookup = {
        (str(row.person_id), int(row.wave)): row
        for row in long_live.itertuples(index=False)
    }
    status_lookup = {
        (str(row.person_id), int(row.wave)): row.interview_status
        for row in status_grid.itertuples(index=False)
    }
    by_person = long_live.groupby("person_id", sort=False)
    records: list[dict[str, Any]] = []

    for person_id, group in by_person:
        observed_waves = set(group["wave"].astype(int).tolist())
        for current_wave in sorted(observed_waves):
            current = live_lookup[(str(person_id), current_wave)]
            next_live_wave: int | None = None
            death_wave: int | None = None

            for future_wave in [w for w in waves if w > current_wave]:
                status = status_lookup.get((str(person_id), int(future_wave)))
                if status == 5:
                    death_wave = int(future_wave)
                    break
                if status == 1 and future_wave in observed_waves:
                    next_live_wave = int(future_wave)
                    break
                if status == 6:
                    # Death is known, but the exact interval is not identified here.
                    break

            if next_live_wave is None and death_wave is None:
                continue

            rec: dict[str, Any] = {
                "person_id": str(person_id),
                "wave": current_wave,
                "next_wave": next_live_wave if next_live_wave is not None else death_wave,
                "terminal_transition": int(death_wave is not None),
            }

            current_year = float(getattr(current, "interview_year", WAVE_YEAR[current_wave]))
            if next_live_wave is not None:
                nxt = live_lookup[(str(person_id), next_live_wave)]
                next_year = float(getattr(nxt, "interview_year", WAVE_YEAR[next_live_wave]))
                delta = next_year - current_year
                if not math.isfinite(delta) or delta <= 0:
                    delta = float(WAVE_YEAR[next_live_wave] - WAVE_YEAR[current_wave])
                if delta > max_transition_years:
                    continue
                rec["delta_t_years"] = delta
            else:
                nxt = None
                delta = float(WAVE_YEAR[death_wave] - current_year)
                rec["delta_t_years"] = delta if delta > 0 else 2.0

            for name in state_names:
                rec[f"{STATE_PREFIX}{name}"] = getattr(current, name, np.nan)
                rec[f"{STATE_MASK_PREFIX}{name}"] = getattr(current, f"{name}__observed", 0)
                if nxt is not None:
                    rec[f"{NEXT_STATE_PREFIX}{name}"] = getattr(nxt, name, np.nan)
                    rec[f"{NEXT_STATE_MASK_PREFIX}{name}"] = getattr(nxt, f"{name}__observed", 0)
                else:
                    rec[f"{NEXT_STATE_PREFIX}{name}"] = np.nan
                    rec[f"{NEXT_STATE_MASK_PREFIX}{name}"] = 0

            for name in action_names:
                rec[f"{ACTION_PREFIX}{name}"] = getattr(current, name, np.nan)
                rec[f"{ACTION_MASK_PREFIX}{name}"] = getattr(current, f"{name}__observed", 0)
                rec[f"{ACTION_ELIGIBLE_PREFIX}{name}"] = getattr(
                    current, f"{name}__eligible", getattr(current, f"{name}__observed", 0)
                )

            rec["continuation"] = 0.0 if death_wave is not None else 1.0
            rec[f"{REWARD_PREFIX}death_event"] = 1.0 if death_wave is not None else 0.0

            if nxt is not None:
                rec[f"{REWARD_PREFIX}hospitalization_event"] = numeric_attr(nxt, "hospitalization")
                for disease, reward_name in [
                    ("cancer_dx", "cancer_incident"),
                    ("heart_disease_dx", "heart_disease_incident"),
                    ("stroke_dx", "stroke_incident"),
                ]:
                    current_dx = numeric_attr(current, disease)
                    next_dx = numeric_attr(nxt, disease)
                    if pd.isna(current_dx) or pd.isna(next_dx) or current_dx == 1:
                        rec[f"{REWARD_PREFIX}{reward_name}"] = np.nan
                    else:
                        rec[f"{REWARD_PREFIX}{reward_name}"] = float(
                            current_dx == 0 and next_dx == 1
                        )

            else:
                for reward_name in [
                    "hospitalization_event", "cancer_incident",
                    "heart_disease_incident", "stroke_incident",
                ]:
                    rec[f"{REWARD_PREFIX}{reward_name}"] = np.nan

            records.append(rec)

    transitions = pd.DataFrame(records)
    if transitions.empty:
        raise ValueError("No valid transitions were constructed. Check source columns and interview-status coding.")
    # Worsening/cost rewards read the state pair, so they are derived in one
    # pass here rather than row by row above.
    transitions = derive_transition_rewards(transitions)
    reward_cols = [c for c in transitions.columns if c.startswith(REWARD_PREFIX)]
    for col in reward_cols:
        name = col[len(REWARD_PREFIX) :]
        transitions[f"{REWARD_MASK_PREFIX}{name}"] = transitions[col].notna().astype("int8")
    return transitions.sort_values(["person_id", "wave"]).reset_index(drop=True)


def numeric_attr(row: Any, name: str) -> float:
    value = getattr(row, name, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def split_by_person(
    transitions: pd.DataFrame,
    validation_size: float,
    test_size: float,
    seed: int,
) -> pd.DataFrame:
    """Split by respondent so no person appears in more than one subset.

    ``validation_size`` and ``test_size`` are target fractions of all unique
    respondents. The second split uses a relative validation fraction within
    the non-test respondents so the final proportions approximate the requested
    overall fractions.
    """
    if not 0 < validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1.")
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if validation_size + test_size >= 1:
        raise ValueError("--validation-size + --test-size must be less than 1.")

    persons = transitions[["person_id"]].drop_duplicates().reset_index(drop=True)
    if len(persons) < 3:
        raise ValueError(
            "At least three respondents are required for train/validation/test splitting."
        )

    test_splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_val_idx, test_idx = next(
        test_splitter.split(persons, groups=persons["person_id"])
    )
    train_val_people_frame = persons.iloc[train_val_idx].reset_index(drop=True)
    test_people = set(persons.iloc[test_idx]["person_id"])

    if len(train_val_people_frame) < 2:
        raise ValueError(
            "Too few non-test respondents remain to create both training and validation sets."
        )

    relative_validation_size = validation_size / (1.0 - test_size)
    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_validation_size,
        random_state=seed + 1,
    )
    train_idx, validation_idx = next(
        validation_splitter.split(
            train_val_people_frame,
            groups=train_val_people_frame["person_id"],
        )
    )

    train_people = set(train_val_people_frame.iloc[train_idx]["person_id"])
    validation_people = set(
        train_val_people_frame.iloc[validation_idx]["person_id"]
    )

    overlaps = {
        "train_validation": train_people & validation_people,
        "train_test": train_people & test_people,
        "validation_test": validation_people & test_people,
    }
    if any(overlaps.values()):
        raise AssertionError(f"Person-level split leakage detected: {overlaps}")
    if not train_people or not validation_people or not test_people:
        raise ValueError(
            "One or more subsets are empty. Reduce validation/test fractions "
            "or use a larger respondent sample."
        )

    result = transitions.copy()
    result["split"] = np.select(
        [
            result["person_id"].isin(validation_people),
            result["person_id"].isin(test_people),
        ],
        ["validation", "test"],
        default="train",
    )
    return result


def load_reward_weights(path: Path | None) -> dict[str, float]:
    weights = dict(DEFAULT_REWARD_WEIGHTS)
    if path is not None:
        with path.open("r", encoding="utf-8") as f:
            supplied = json.load(f)
        for key, value in supplied.items():
            weights[str(key)] = float(value)
    return weights


def _masked_state_pair(
    frame: pd.DataFrame,
    name: str,
) -> tuple[pd.Series, pd.Series]:
    """Current and next-wave values of one state variable, NA where unobserved."""
    current = pd.to_numeric(frame[f"{STATE_PREFIX}{name}"], errors="coerce")
    following = pd.to_numeric(frame[f"{NEXT_STATE_PREFIX}{name}"], errors="coerce")
    current_mask = frame.get(f"{STATE_MASK_PREFIX}{name}")
    following_mask = frame.get(f"{NEXT_STATE_MASK_PREFIX}{name}")
    if current_mask is not None:
        current = current.where(pd.to_numeric(current_mask, errors="coerce").eq(1))
    if following_mask is not None:
        following = following.where(
            pd.to_numeric(following_mask, errors="coerce").eq(1)
        )
    return current, following


def _is_severe(score: pd.Series, op: str, threshold: float) -> pd.Series:
    """True where ``score`` meets the severe-impairment rule for one state variable."""
    if op == ">=":
        return score >= threshold
    if op == "<=":
        return score <= threshold
    raise ValueError(f"Unsupported severe comparison operator: {op!r}")


def _cpi_factor_to_2018(waves: pd.Series) -> pd.Series:
    """Multiplier converting nominal dollars at a wave into 2018 USD."""
    years = pd.to_numeric(waves, errors="coerce").astype("Int64").map(WAVE_YEAR)
    base = CPI_U[2018]
    return years.map(
        lambda year: base / CPI_U[int(year)]
        if pd.notna(year) and int(year) in CPI_U
        else np.nan
    ).astype("float64")


def derive_worsening_rewards(transitions: pd.DataFrame) -> pd.DataFrame:
    """Rewrite clinical/cost rewards as binary next-wave severe / OOP-increase flags.

    For every listed clinical transition ``t -> t+1``::

        1  severe at t+1 (thresholds in ``WORSENING_REWARD_FROM_STATE``)
        0  observed and non-severe at t+1
        NA next wave unobserved

    ``cognition_decline`` is *not* rewritten here; see
    ``derive_continuous_level_rewards``. Out-of-pocket cost compares the two
    per-interval totals after deflating each to 2018 USD, and scores 1 only on
    a rise of at least ``OOP_INCREASE_THRESHOLD``.

    Only reads ``state__*`` / ``next_state__*``, so it is safe to call on an
    existing transitions table (e.g. during ``--prepare-model-bundle``).
    """
    result = transitions.copy()

    def _write(reward_name: str, worse: pd.Series) -> None:
        result[f"{REWARD_PREFIX}{reward_name}"] = worse
        result[f"{REWARD_MASK_PREFIX}{reward_name}"] = worse.notna().astype("int8")

    for state_name, reward_name, severe_op, severe_threshold in WORSENING_REWARD_FROM_STATE:
        if f"{NEXT_STATE_PREFIX}{state_name}" not in result.columns:
            LOGGER.warning(
                "Cannot derive %s: missing %s%s",
                reward_name,
                NEXT_STATE_PREFIX,
                state_name,
            )
            continue
        _current, following = _masked_state_pair(result, state_name)
        observed = following.notna()
        next_severe = _is_severe(following, severe_op, severe_threshold)
        _write(reward_name, next_severe.astype("float64").where(observed))

    if f"{NEXT_STATE_PREFIX}{OOP_INCREASE_FROM_STATE}" in result.columns:
        current, following = _masked_state_pair(result, OOP_INCREASE_FROM_STATE)
        current_real = current.where(current >= 0) * _cpi_factor_to_2018(result["wave"])
        following_real = following.where(following >= 0) * _cpi_factor_to_2018(
            result["next_wave"]
        )
        rise = (following_real >= current_real * (1.0 + OOP_INCREASE_THRESHOLD)).where(
            current_real > 0,
            following_real > 0,
        )
        observed = current_real.notna() & following_real.notna()
        _write(
            OOP_INCREASE_REWARD,
            rise.astype("float64").where(observed),
        )
    else:
        LOGGER.warning(
            "Cannot derive %s: missing %s%s",
            OOP_INCREASE_REWARD,
            NEXT_STATE_PREFIX,
            OOP_INCREASE_FROM_STATE,
        )
    return result


def derive_continuous_level_rewards(transitions: pd.DataFrame) -> pd.DataFrame:
    """Copy next-wave state levels into named continuous rewards (masked)."""
    result = transitions.copy()
    for state_name, reward_name in CONTINUOUS_LEVEL_REWARD_FROM_STATE:
        value_col = f"{NEXT_STATE_PREFIX}{state_name}"
        mask_col = f"{NEXT_STATE_MASK_PREFIX}{state_name}"
        if value_col not in result.columns:
            LOGGER.warning("Cannot derive %s: missing %s", reward_name, value_col)
            continue
        values = pd.to_numeric(result[value_col], errors="coerce")
        if mask_col in result.columns:
            values = values.where(
                pd.to_numeric(result[mask_col], errors="coerce").eq(1)
            )
        result[f"{REWARD_PREFIX}{reward_name}"] = values
        result[f"{REWARD_MASK_PREFIX}{reward_name}"] = values.notna().astype("int8")
    return result


def derive_cvd_incident(transitions: pd.DataFrame) -> pd.DataFrame:
    """OR of heart/stroke incidence among people currently free of both.

    At-risk (mask=1): every source ``reward_mask`` is 1 (current dx=0 and next
    dx observed for both). Event=1 if any observed source is 1. Matches JEPA
    ``pooled_binary_arrays(how='any', mask_how='all')``. Not-at-risk rows are
    NA with mask 0, same convention as the source incident columns.
    """
    result = transitions.copy()
    missing = [
        src
        for src in CVD_INCIDENT_SOURCES
        if f"{REWARD_PREFIX}{src}" not in result.columns
    ]
    if missing:
        LOGGER.warning(
            "Cannot derive %s: missing source columns %s",
            CVD_INCIDENT_REWARD,
            ", ".join(missing),
        )
        return result

    observed_parts: list[pd.Series] = []
    event_parts: list[pd.Series] = []
    for src in CVD_INCIDENT_SOURCES:
        values = pd.to_numeric(result[f"{REWARD_PREFIX}{src}"], errors="coerce")
        mask_col = f"{REWARD_MASK_PREFIX}{src}"
        if mask_col in result.columns:
            mask = pd.to_numeric(result[mask_col], errors="coerce").fillna(0)
        else:
            # build_transitions writes reward_mask__* after this function.
            mask = values.notna().astype("int8")
        observed = values.notna() & mask.eq(1)
        observed_parts.append(observed)
        event_parts.append(values.gt(0.5) & observed)
    at_risk = observed_parts[0]
    event = event_parts[0]
    for obs, ev in zip(observed_parts[1:], event_parts[1:]):
        at_risk = at_risk & obs
        event = event | ev
    event = event & at_risk
    result[f"{REWARD_PREFIX}{CVD_INCIDENT_REWARD}"] = event.astype("float64").where(
        at_risk
    )
    result[f"{REWARD_MASK_PREFIX}{CVD_INCIDENT_REWARD}"] = at_risk.astype("int8")
    return result


def derive_transition_rewards(transitions: pd.DataFrame) -> pd.DataFrame:
    """Binary severe flags + continuous cognition + pooled CVD incident."""
    out = derive_worsening_rewards(transitions)
    out = derive_continuous_level_rewards(out)
    out = derive_cvd_incident(out)
    return out


def add_composite_reward(
    transitions: pd.DataFrame,
    reward_weights: Mapping[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = transitions.copy()
    train = result["split"].eq("train")
    scales: dict[str, float] = {}
    total = pd.Series(0.0, index=result.index)
    observed_any = pd.Series(False, index=result.index)

    for reward_name, weight in reward_weights.items():
        col = f"{REWARD_PREFIX}{reward_name}"
        if col not in result:
            continue
        values = pd.to_numeric(result[col], errors="coerce")
        train_values = values.loc[train].dropna()
        scale = float(train_values.std(ddof=0)) if len(train_values) else 1.0
        if not math.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        scales[reward_name] = scale
        normalized = values / scale
        total = total.add(normalized.fillna(0) * float(weight), fill_value=0)
        observed_any |= values.notna()

    composite_col = f"{REWARD_PREFIX}composite_clinical_reward"
    result[composite_col] = (-total).where(observed_any)
    result[f"{REWARD_MASK_PREFIX}composite_clinical_reward"] = result[composite_col].notna().astype(
        "int8"
    )
    return result, scales


def select_model_features(
    transitions: pd.DataFrame,
    specs: Mapping[str, VariableSpec],
) -> dict[str, Any]:
    """Select state/action/reward features with at least one training observation."""
    candidate_state_names = [
        name for name, spec in specs.items()
        if spec.role in {"Static state variable", "Dynamic state variable"}
        and name not in EXCLUDE_MODEL_STATE_VARIABLES
    ]
    candidate_action_names = [
        name for name, spec in specs.items() if spec.role == "Action variable"
    ]
    candidate_reward_names = [
        name for name, spec in specs.items()
        if spec.role == "Reward variable"
        and name not in {"continuation_target", "composite_clinical_reward"}
        and name not in EXCLUDE_MODEL_REWARD_VARIABLES
    ]

    train = transitions.loc[transitions["split"].eq("train")].copy()
    state_names = [
        name for name in candidate_state_names
        if f"{STATE_PREFIX}{name}" in train and train[f"{STATE_PREFIX}{name}"].notna().any()
    ]
    action_names = [
        name for name in candidate_action_names
        if f"{ACTION_PREFIX}{name}" in train and (
            train[f"{ACTION_PREFIX}{name}"].notna().any()
            or (
                f"{ACTION_ELIGIBLE_PREFIX}{name}" in train
                and train[f"{ACTION_ELIGIBLE_PREFIX}{name}"].fillna(0).gt(0).any()
            )
        )
    ]
    reward_names = [
        name for name in candidate_reward_names
        if f"{REWARD_PREFIX}{name}" in train and train[f"{REWARD_PREFIX}{name}"].notna().any()
    ]
    dropped_state = sorted(set(candidate_state_names) - set(state_names))
    dropped_action = sorted(set(candidate_action_names) - set(action_names))
    dropped_reward = sorted(set(candidate_reward_names) - set(reward_names))
    if dropped_state:
        LOGGER.warning(
            "State features exclude all-missing training variables: %s",
            ", ".join(dropped_state),
        )
    if dropped_action:
        LOGGER.warning(
            "Action features exclude all-missing training variables: %s",
            ", ".join(dropped_action),
        )
    if dropped_reward:
        LOGGER.warning(
            "Reward features exclude all-missing training targets: %s",
            ", ".join(dropped_reward),
        )
    if not state_names:
        raise ValueError("No state variables have training observations.")
    if not action_names:
        raise ValueError("No action variables have training observations.")

    categorical_state = [
        name for name in state_names
        if "categorical" in specs[name].data_type.lower()
    ]
    numeric_state = [name for name in state_names if name not in categorical_state]
    return {
        "state_variables": state_names,
        "numeric_state_variables": numeric_state,
        "categorical_state_variables": categorical_state,
        "action_variables": action_names,
        "reward_vector_variables": reward_names,
        "dropped_all_missing_state_variables": dropped_state,
        "dropped_all_missing_action_variables": dropped_action,
        "dropped_all_missing_reward_variables": dropped_reward,
    }


def missingness_summary(long_live: pd.DataFrame, specs: Mapping[str, VariableSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, spec in specs.items():
        if name not in long_live:
            continue
        obs_col = f"{name}__observed"
        rows.append({
            "variable": name,
            "role": spec.role,
            "n_rows": len(long_live),
            "n_nonmissing": int(long_live[name].notna().sum()),
            "missing_rate": float(long_live[name].isna().mean()),
            "n_observed_mask": int(long_live.get(obs_col, pd.Series(0, index=long_live.index)).sum()),
        })
    return pd.DataFrame(rows).sort_values(["role", "variable"])


def _series_descriptive_stats(
    series: pd.Series,
    *,
    n_rows: int | None = None,
    max_top_values: int = 10,
) -> dict[str, Any]:
    """Compute missingness and distribution stats for one series."""
    n_rows = int(len(series) if n_rows is None else n_rows)
    nonmissing = series.dropna()
    n_nonmissing = int(len(nonmissing))
    stats: dict[str, Any] = {
        "n_rows": n_rows,
        "n_nonmissing": n_nonmissing,
        "missing_rate": float(1.0 - (n_nonmissing / n_rows)) if n_rows else np.nan,
        "n_unique": int(nonmissing.nunique(dropna=True)),
        "mean": np.nan,
        "std": np.nan,
        "min": np.nan,
        "p25": np.nan,
        "median": np.nan,
        "p75": np.nan,
        "max": np.nan,
        "mode": np.nan,
        "mode_count": 0,
        "top_values": "",
    }
    if n_nonmissing == 0:
        return stats

    numeric = pd.to_numeric(nonmissing, errors="coerce")
    numeric_ok = int(numeric.notna().sum())
    treat_as_numeric = numeric_ok >= max(1, int(0.9 * n_nonmissing))
    if treat_as_numeric:
        num = numeric.dropna()
        quantiles = num.quantile([0.25, 0.5, 0.75])
        stats.update({
            "mean": float(num.mean()),
            "std": float(num.std(ddof=0)),
            "min": float(num.min()),
            "p25": float(quantiles.loc[0.25]),
            "median": float(quantiles.loc[0.5]),
            "p75": float(quantiles.loc[0.75]),
            "max": float(num.max()),
        })

    value_counts = nonmissing.astype(str).value_counts(dropna=True)
    if not value_counts.empty:
        stats["mode"] = value_counts.index[0]
        stats["mode_count"] = int(value_counts.iloc[0])
        if stats["n_unique"] <= 30 or not treat_as_numeric:
            top = value_counts.head(max_top_values)
            stats["top_values"] = "; ".join(f"{idx}:{int(cnt)}" for idx, cnt in top.items())
    return stats


def descriptive_variable_summary(
    long_live: pd.DataFrame,
    specs: Mapping[str, VariableSpec],
) -> pd.DataFrame:
    """Person-wave descriptive statistics for each dictionary variable."""
    rows: list[dict[str, Any]] = []
    n_people = int(long_live["person_id"].nunique()) if "person_id" in long_live else np.nan
    for name, spec in specs.items():
        if name not in long_live.columns:
            continue
        obs_col = f"{name}__observed"
        base = {
            "table": "long_live",
            "variable": name,
            "role": spec.role,
            "data_type": spec.data_type,
            "priority": spec.priority,
            "n_people": n_people,
            "n_observed_mask": int(
                long_live.get(obs_col, pd.Series(0, index=long_live.index)).sum()
            ),
        }
        base.update(_series_descriptive_stats(long_live[name]))
        rows.append(base)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["role", "variable"]).reset_index(drop=True)


def descriptive_variable_by_wave(
    long_live: pd.DataFrame,
    specs: Mapping[str, VariableSpec],
) -> pd.DataFrame:
    """Wave-stratified descriptive statistics for long-table variables."""
    if "wave" not in long_live.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for wave, group in long_live.groupby("wave", sort=True):
        n_people = int(group["person_id"].nunique()) if "person_id" in group else np.nan
        for name, spec in specs.items():
            if name not in group.columns:
                continue
            obs_col = f"{name}__observed"
            base = {
                "table": "long_live",
                "variable": name,
                "role": spec.role,
                "data_type": spec.data_type,
                "wave": int(wave),
                "n_people": n_people,
                "n_observed_mask": int(
                    group.get(obs_col, pd.Series(0, index=group.index)).sum()
                ),
            }
            base.update(_series_descriptive_stats(group[name]))
            rows.append(base)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["variable", "wave"]).reset_index(drop=True)


def descriptive_transition_summary(
    transitions: pd.DataFrame,
    specs: Mapping[str, VariableSpec],
) -> pd.DataFrame:
    """Transition-table descriptive stats for state/action/reward columns."""
    rows: list[dict[str, Any]] = []

    candidates: list[tuple[str, str, str, str]] = []
    for name, spec in specs.items():
        if spec.role in {"Static state variable", "Dynamic state variable"}:
            for prefix, label in (
                (STATE_PREFIX, "state_t"),
                (NEXT_STATE_PREFIX, "state_t1"),
            ):
                col = f"{prefix}{name}"
                if col in transitions.columns:
                    candidates.append((col, name, spec.role, label))
        elif spec.role == "Action variable":
            col = f"{ACTION_PREFIX}{name}"
            if col in transitions.columns:
                candidates.append((col, name, spec.role, "action_t"))
        elif spec.role == "Reward variable":
            col = f"{REWARD_PREFIX}{name}"
            if col in transitions.columns:
                candidates.append((col, name, spec.role, "reward"))

    composite_col = f"{REWARD_PREFIX}composite_clinical_reward"
    if (
        composite_col in transitions.columns
        and not any(col == composite_col for col, *_ in candidates)
    ):
        candidates.append(
            (composite_col, "composite_clinical_reward", "Reward variable", "reward")
        )

    split_values: list[str | None]
    if "split" in transitions.columns:
        split_values = [None] + sorted(
            transitions["split"].dropna().astype(str).unique().tolist()
        )
    else:
        split_values = [None]

    for col, name, role, side in candidates:
        data_type = specs[name].data_type if name in specs else "continuous derived"
        priority = specs[name].priority if name in specs else "Derived"
        for split_name in split_values:
            frame = (
                transitions
                if split_name is None
                else transitions.loc[transitions["split"].astype(str).eq(split_name)]
            )
            if frame.empty:
                continue
            base = {
                "table": "transitions",
                "variable": name,
                "column": col,
                "side": side,
                "role": role,
                "data_type": data_type,
                "priority": priority,
                "split": "all" if split_name is None else split_name,
                "n_people": int(frame["person_id"].nunique()) if "person_id" in frame else np.nan,
            }
            base.update(_series_descriptive_stats(frame[col]))
            rows.append(base)

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["role", "variable", "side", "split"])
        .reset_index(drop=True)
    )


def _is_model_continuous(data_type: str) -> bool:
    text = str(data_type).lower()
    if "categorical" in text or "binary" in text:
        return False
    return any(token in text for token in ("continuous", "count", "numeric", "score"))


def _is_model_ordinal(data_type: str) -> bool:
    """True for ordered categorical states (pure ordinal), not continuous hybrids."""
    text = str(data_type).lower()
    if "ordinal" not in text:
        return False
    # "continuous/ordinal" or "ordinal/continuous" stay on the continuous path.
    if _is_model_continuous(text):
        return False
    return True


# Align with code.specs reward binary/continuous routing.
_BINARY_REWARD_PREFIXES = (
    "death_event",
    "hospitalization_event",
    "incident_",
)
_BINARY_REWARD_SUFFIXES = (
    "_incident",
    "_event",
)
# Empty since out-of-pocket cost became a binary increase flag. Kept so any
# future skewed continuous reward can opt into log1p before standardization.
_LOG1P_REWARD_NAMES: frozenset[str] = frozenset()


def _is_model_continuous_reward(name: str, values: pd.Series) -> bool:
    """True for two-hot continuous rewards; False for Bernoulli binary rewards."""
    if name in BINARY_WORSENING_REWARDS:
        return False
    if name.startswith(_BINARY_REWARD_PREFIXES) or name.endswith(_BINARY_REWARD_SUFFIXES):
        return False
    unique = {float(x) for x in pd.to_numeric(values, errors="coerce").dropna().unique()}
    if unique and unique.issubset({0.0, 1.0}) and not name.startswith("change_"):
        return False
    return True


def _transform_reward_values(name: str, values: pd.Series) -> pd.Series:
    """Optional pre-standardization transform (e.g. log1p for OOP dollars)."""
    numeric = pd.to_numeric(values, errors="coerce")
    if name in _LOG1P_REWARD_NAMES:
        return np.log1p(numeric.clip(lower=0))
    return numeric


def normalize_transition_column_names(transitions: pd.DataFrame) -> pd.DataFrame:
    """Map legacy HRS prefixes to model column names."""
    frame = transitions.copy()
    rename_map: dict[str, str] = {}
    for col in frame.columns:
        if col.startswith("sobs__"):
            rename_map[col] = STATE_MASK_PREFIX + col[len("sobs__") :]
        elif col.startswith("spobs__"):
            rename_map[col] = NEXT_STATE_MASK_PREFIX + col[len("spobs__") :]
        elif col.startswith("s__"):
            rename_map[col] = STATE_PREFIX + col[len("s__") :]
        elif col.startswith("sp__"):
            rename_map[col] = NEXT_STATE_PREFIX + col[len("sp__") :]
        elif col.startswith("aobs__"):
            rename_map[col] = ACTION_MASK_PREFIX + col[len("aobs__") :]
        elif col.startswith("aeligible__"):
            rename_map[col] = ACTION_ELIGIBLE_PREFIX + col[len("aeligible__") :]
        elif col.startswith("a__"):
            rename_map[col] = ACTION_PREFIX + col[len("a__") :]
        elif col.startswith("robs__"):
            rename_map[col] = REWARD_MASK_PREFIX + col[len("robs__") :]
        elif col.startswith("r__"):
            rename_map[col] = REWARD_PREFIX + col[len("r__") :]
    if rename_map:
        collisions = sorted(
            dest for dest in rename_map.values() if dest in frame.columns
        )
        if collisions:
            raise ValueError(
                "Legacy and canonical transition columns both present for: "
                + ", ".join(collisions)
            )
        frame = frame.rename(columns=rename_map)
        if frame.columns.duplicated().any():
            dupes = frame.columns[frame.columns.duplicated()].unique().tolist()
            raise ValueError(f"Duplicate columns after legacy rename: {dupes}")
    if "delta_time_years" in frame.columns and "delta_t_years" not in frame.columns:
        frame = frame.rename(columns={"delta_time_years": "delta_t_years"})
    if f"{REWARD_PREFIX}continuation_target" in frame.columns and "continuation" not in frame.columns:
        frame = frame.rename(
            columns={f"{REWARD_PREFIX}continuation_target": "continuation"}
        )
    reward_cols = [c for c in frame.columns if c.startswith(REWARD_PREFIX)]
    for col in reward_cols:
        name = col[len(REWARD_PREFIX) :]
        mask_col = f"{REWARD_MASK_PREFIX}{name}"
        if mask_col not in frame.columns:
            frame[mask_col] = frame[col].notna().astype("int8")
    return frame


def write_hrs_model_bundle(
    transitions: pd.DataFrame,
    specs: Mapping[str, VariableSpec],
    metadata: Mapping[str, Any],
    output_dir: Path,
    *,
    dictionary_path: Path | None = None,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 2026,
) -> dict[str, Any]:
    """Write HRS model parquet tables, preprocessing.json, and model_config.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = normalize_transition_column_names(transitions)
    if "split" not in frame.columns:
        raise ValueError("transitions must contain a person-level split column")
    if "delta_t_years" not in frame.columns:
        raise ValueError("transitions must contain delta_t_years")
    if "continuation" not in frame.columns:
        raise ValueError("transitions must contain continuation")
    if "next_wave" not in frame.columns:
        frame["next_wave"] = np.nan

    frame = frame.sort_values(["person_id", "wave"], kind="stable").copy()
    frame["transition_index"] = frame.groupby("person_id", sort=False).cumcount().astype(int)

    # Imagination head expects this name.
    composite_src = f"{REWARD_PREFIX}composite_clinical_reward"
    composite_dst = f"{REWARD_PREFIX}composite_reward"
    if composite_src in frame.columns and composite_dst not in frame.columns:
        frame[composite_dst] = frame[composite_src]
        frame[f"{REWARD_MASK_PREFIX}composite_reward"] = frame[
            f"{REWARD_MASK_PREFIX}composite_clinical_reward"
        ]

    state_names = [
        name
        for name, spec in specs.items()
        if spec.role == "Dynamic state variable"
        and name not in EXCLUDE_MODEL_STATE_VARIABLES
        and f"{STATE_PREFIX}{name}" in frame.columns
    ]
    static_names = [
        name
        for name, spec in specs.items()
        if spec.role == "Static state variable"
        and name not in EXCLUDE_MODEL_STATE_VARIABLES
        and f"{STATE_PREFIX}{name}" in frame.columns
    ]
    # SES/housing/work block: context-only even if still Dynamic in older dicts.
    forced_static = [
        name
        for name in FORCE_STATIC_CONTEXT_VARIABLES
        if name not in EXCLUDE_MODEL_STATE_VARIABLES
        and f"{STATE_PREFIX}{name}" in frame.columns
    ]
    static_names = list(dict.fromkeys([*static_names, *forced_static]))
    # Keep static out of dynamic decoder targets while still providing context.
    state_names = [name for name in state_names if name not in set(static_names)]
    action_names = [
        name
        for name, spec in specs.items()
        if spec.role == "Action variable" and f"{ACTION_PREFIX}{name}" in frame.columns
    ]
    reward_names = [
        name
        for name, spec in specs.items()
        if spec.role == "Reward variable"
        and name not in {"continuation_target", "composite_clinical_reward"}
        and name not in EXCLUDE_MODEL_REWARD_VARIABLES
        and f"{REWARD_PREFIX}{name}" in frame.columns
    ]
    if f"{REWARD_PREFIX}composite_reward" in frame.columns:
        reward_names = list(reward_names) + ["composite_reward"]

    # Prefer metadata lists when present; otherwise drop all-missing train features here.
    meta_state = [str(x) for x in metadata.get("state_variables", [])]
    meta_action = [str(x) for x in metadata.get("action_variables", [])]
    meta_reward = [str(x) for x in metadata.get("reward_vector_variables", [])]
    train = frame.loc[frame["split"].eq("train")].copy()
    if meta_state:
        keep = set(meta_state)
        state_names = [n for n in state_names if n in keep]
        static_names = [n for n in static_names if n in keep]
    else:
        state_names = [
            n for n in state_names if train[f"{STATE_PREFIX}{n}"].notna().any()
        ]
        static_names = [
            n for n in static_names if train[f"{STATE_PREFIX}{n}"].notna().any()
        ]
    if meta_action:
        keep = set(meta_action)
        action_names = [n for n in action_names if n in keep]
    else:
        action_names = [
            n for n in action_names
            if train[f"{ACTION_PREFIX}{n}"].notna().any()
            or (
                f"{ACTION_ELIGIBLE_PREFIX}{n}" in train
                and train[f"{ACTION_ELIGIBLE_PREFIX}{n}"].fillna(0).gt(0).any()
            )
        ]
    if meta_reward:
        keep = set(meta_reward) | {"composite_reward"}
        reward_names = [n for n in reward_names if n in keep]
    else:
        reward_names = [
            n for n in reward_names
            if n == "composite_reward" or train[f"{REWARD_PREFIX}{n}"].notna().any()
        ]

    continuous_stats: dict[str, dict[str, float]] = {}
    candidate_continuous = []
    for name in state_names + static_names + action_names:
        dtype = specs[name].data_type if name in specs else "continuous"
        if _is_model_continuous(dtype):
            candidate_continuous.append(name)

    def _obs_mask(split_frame: pd.DataFrame, mask_col: str, values: pd.Series) -> pd.Series:
        if mask_col in split_frame.columns:
            return pd.to_numeric(split_frame[mask_col], errors="coerce").fillna(0).astype("int8")
        return values.notna().astype("int8")

    for name in candidate_continuous:
        parts: list[pd.Series] = []
        for value_col, mask_col in (
            (f"{STATE_PREFIX}{name}", f"{STATE_MASK_PREFIX}{name}"),
            (f"{NEXT_STATE_PREFIX}{name}", f"{NEXT_STATE_MASK_PREFIX}{name}"),
            (f"{ACTION_PREFIX}{name}", f"{ACTION_MASK_PREFIX}{name}"),
        ):
            if value_col not in train.columns:
                continue
            values = pd.to_numeric(train[value_col], errors="coerce")
            mask = _obs_mask(train, mask_col, values)
            if mask_col.startswith(ACTION_MASK_PREFIX):
                eligible_col = f"{ACTION_ELIGIBLE_PREFIX}{name}"
                if eligible_col in train.columns:
                    eligible = pd.to_numeric(
                        train[eligible_col], errors="coerce"
                    ).fillna(0).astype("int8")
                    mask = (mask.astype("int8") * eligible).astype("int8")
            parts.append(values.where(mask.eq(1)))
        if not parts:
            continue
        fit = pd.concat(parts, ignore_index=True).dropna()
        if fit.empty:
            continue
        std = float(fit.std(ddof=0))
        if not math.isfinite(std) or std < 1e-8:
            std = 1.0
        continuous_stats[name] = {
            "median": float(fit.median()),
            "mean": float(fit.mean()),
            "std": std,
        }

    # Continuous rewards use a separate stats dict (avoids colliding with state names).
    reward_continuous_stats: dict[str, dict[str, float | str]] = {}
    for name in reward_names:
        value_col = f"{REWARD_PREFIX}{name}"
        if value_col not in train.columns:
            continue
        values = pd.to_numeric(train[value_col], errors="coerce")
        mask = _obs_mask(train, f"{REWARD_MASK_PREFIX}{name}", values)
        observed = values.where(mask.eq(1))
        if not _is_model_continuous_reward(name, observed):
            continue
        transformed = _transform_reward_values(name, observed).dropna()
        if transformed.empty:
            continue
        std = float(transformed.std(ddof=0))
        if not math.isfinite(std) or std < 1e-8:
            std = 1.0
        reward_continuous_stats[name] = {
            "median": float(transformed.median()),
            "mean": float(transformed.mean()),
            "std": std,
            "transform": "log1p" if name in _LOG1P_REWARD_NAMES else "identity",
        }

    def convert(split_frame: pd.DataFrame) -> pd.DataFrame:
        blocks: dict[str, Any] = {
            "person_id": split_frame["person_id"].to_numpy(),
            "transition_index": split_frame["transition_index"].to_numpy(),
            "wave": split_frame["wave"].to_numpy(),
            "next_wave": split_frame["next_wave"].to_numpy(),
            "delta_t_years": split_frame["delta_t_years"].to_numpy(),
            "continuation": split_frame["continuation"].to_numpy(),
            "split": split_frame["split"].to_numpy(),
        }
        for name in state_names + static_names:
            current = pd.to_numeric(split_frame[f"{STATE_PREFIX}{name}"], errors="coerce")
            next_col = f"{NEXT_STATE_PREFIX}{name}"
            if next_col in split_frame.columns:
                future = pd.to_numeric(split_frame[next_col], errors="coerce")
            else:
                future = pd.Series(np.nan, index=split_frame.index, dtype="float64")
            # Preserve observation masks (nearby-wave fill must not flip them to 1).
            current_mask = _obs_mask(
                split_frame, f"{STATE_MASK_PREFIX}{name}", current
            )
            future_mask = _obs_mask(
                split_frame, f"{NEXT_STATE_MASK_PREFIX}{name}", future
            )
            blocks[f"{STATE_MASK_PREFIX}{name}"] = current_mask.to_numpy()
            blocks[f"{NEXT_STATE_MASK_PREFIX}{name}"] = future_mask.to_numpy()
            if name in continuous_stats:
                p = continuous_stats[name]
                blocks[f"{STATE_PREFIX}{name}"] = (
                    (current.fillna(p["median"]) - p["mean"]) / p["std"]
                ).to_numpy(dtype=np.float32)
                blocks[f"{NEXT_STATE_PREFIX}{name}"] = (
                    (future.fillna(p["median"]) - p["mean"]) / p["std"]
                ).to_numpy(dtype=np.float32)
            else:
                blocks[f"{STATE_PREFIX}{name}"] = current.fillna(-1).astype("float32").to_numpy()
                blocks[f"{NEXT_STATE_PREFIX}{name}"] = future.fillna(-1).astype("float32").to_numpy()
        for name in action_names:
            value = pd.to_numeric(split_frame[f"{ACTION_PREFIX}{name}"], errors="coerce")
            obs_mask = _obs_mask(split_frame, f"{ACTION_MASK_PREFIX}{name}", value)
            eligible_col = f"{ACTION_ELIGIBLE_PREFIX}{name}"
            if eligible_col in split_frame.columns:
                eligible = pd.to_numeric(
                    split_frame[eligible_col], errors="coerce"
                ).fillna(0).astype("int8")
            else:
                eligible = pd.Series(1, index=split_frame.index, dtype="int8")
            # Collapse eligibility into the model action mask.
            action_mask = (obs_mask.astype("int8") * eligible.astype("int8")).astype("int8")
            blocks[f"{ACTION_MASK_PREFIX}{name}"] = action_mask.to_numpy()
            if name in continuous_stats:
                p = continuous_stats[name]
                filled = value.where(action_mask.eq(1), np.nan).fillna(p["median"])
                blocks[f"{ACTION_PREFIX}{name}"] = (
                    (filled - p["mean"]) / p["std"]
                ).to_numpy(dtype=np.float32)
            else:
                blocks[f"{ACTION_PREFIX}{name}"] = (
                    value.where(action_mask.eq(1), np.nan).fillna(-1).astype("float32").to_numpy()
                )
        for name in reward_names:
            value = pd.to_numeric(split_frame[f"{REWARD_PREFIX}{name}"], errors="coerce")
            mask = _obs_mask(split_frame, f"{REWARD_MASK_PREFIX}{name}", value)
            blocks[f"{REWARD_MASK_PREFIX}{name}"] = mask.to_numpy()
            if name in reward_continuous_stats:
                p = reward_continuous_stats[name]
                transformed = _transform_reward_values(name, value)
                blocks[f"{REWARD_PREFIX}{name}"] = (
                    (transformed.fillna(float(p["median"])) - float(p["mean"]))
                    / float(p["std"])
                ).to_numpy(dtype=np.float32)
            else:
                # Binary Bernoulli rewards stay on the raw 0/1 scale.
                blocks[f"{REWARD_PREFIX}{name}"] = (
                    value.fillna(0).astype("float32").to_numpy()
                )
        return pd.DataFrame(blocks, index=split_frame.index)

    train_out = convert(train)
    validation_out = convert(frame.loc[frame["split"].eq("validation")].copy())
    test_out = convert(frame.loc[frame["split"].eq("test")].copy())

    train_path = output_dir / "HRS_train.parquet"
    validation_path = output_dir / "HRS_validation.parquet"
    test_path = output_dir / "HRS_test.parquet"
    train_csv_path = output_dir / "HRS_train.csv"
    validation_csv_path = output_dir / "HRS_validation.csv"
    test_csv_path = output_dir / "HRS_test.csv"
    preprocessing_path = output_dir / "HRS_preprocessing.json"
    model_config_path = output_dir / "HRS_model_config.json"

    for split_frame in (train_out, validation_out, test_out):
        if "person_id" in split_frame.columns:
            split_frame["person_id"] = split_frame["person_id"].astype("string")
    train_out.to_parquet(train_path, index=False)
    validation_out.to_parquet(validation_path, index=False)
    test_out.to_parquet(test_path, index=False)
    # CSV audit copies of the model-ready tables (same content as parquet).
    train_out.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
    validation_out.to_csv(validation_csv_path, index=False, encoding="utf-8-sig")
    test_out.to_csv(test_csv_path, index=False, encoding="utf-8-sig")

    ordinal_names = [
        name
        for name in state_names + static_names
        if name not in continuous_stats
        and (
            name in FORCE_MODEL_ORDINAL_VARIABLES
            or _is_model_ordinal(specs[name].data_type if name in specs else "")
        )
    ]
    preprocessing = {
        "version": "hrs-model-bundle-v3",
        "continuous": continuous_stats,
        "reward_continuous": reward_continuous_stats,
        "ordinal": ordinal_names,
        "categorical_missing_code": -1,
    }
    with preprocessing_path.open("w", encoding="utf-8") as f:
        json.dump(preprocessing, f, ensure_ascii=False, indent=2)

    model_config = {
        "version": "hrs-model-bundle-v3",
        "dictionary": str(dictionary_path) if dictionary_path else None,
        "transition_unit": "(s_t, a_t, delta_t) -> (s_t+1, reward, continuation)",
        "split": {
            "method": "person-level",
            "train_ratio": 1.0 - validation_size - test_size,
            "validation_ratio": validation_size,
            "test_ratio": test_size,
            "validation_set": True,
            "seed": seed,
        },
        "state_columns_dictionary": [
            name for name, spec in specs.items()
            if spec.role in {"Static state variable", "Dynamic state variable"}
        ],
        "static_context_columns": static_names,
        "action_columns_requested": [
            name for name, spec in specs.items() if spec.role == "Action variable"
        ],
        "state_columns_model": state_names,
        "action_columns_model": action_names,
        "reward_columns": reward_names,
        "ordinal_state_columns": ordinal_names,
        "delta_t_column": "delta_t_years",
        "continuation_column": "continuation",
        "reward_weights": metadata.get("reward_weights", DEFAULT_REWARD_WEIGHTS),
        "outputs": {
            "train": train_path.name,
            "validation": validation_path.name,
            "test": test_path.name,
            "train_csv": train_csv_path.name,
            "validation_csv": validation_csv_path.name,
            "test_csv": test_csv_path.name,
            "preprocessing": preprocessing_path.name,
            "transitions_raw": "hrs_world_model_transitions.parquet",
        },
    }
    with model_config_path.open("w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    LOGGER.info(
        "Wrote HRS model bundle: %d state / %d static / %d action / %d reward "
        "(%d continuous rewards standardized; %d ordinal states)",
        len(state_names),
        len(static_names),
        len(action_names),
        len(reward_names),
        len(reward_continuous_stats),
        len(ordinal_names),
    )
    return model_config


def write_outputs(
    transitions: pd.DataFrame,
    audit: pd.DataFrame,
    missing_summary: pd.DataFrame,
    descriptive_summary: pd.DataFrame,
    descriptive_by_wave: pd.DataFrame,
    descriptive_transitions: pd.DataFrame,
    metadata: Mapping[str, Any],
    output_dir: Path,
    csv_copy: bool,
    long_live_raw: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    transitions.to_parquet(output_dir / "hrs_world_model_transitions.parquet", index=False)
    if long_live_raw is not None:
        long_raw_path = output_dir / "hrs_world_model_long_raw.csv"
        long_live_raw.to_csv(long_raw_path, index=False, encoding="utf-8-sig")
        LOGGER.info(
            "Wrote pre-nearby-fill live long CSV: %s (%d rows, %d cols)",
            long_raw_path.name,
            len(long_live_raw),
            long_live_raw.shape[1],
        )
    audit.to_csv(output_dir / "source_audit.csv", index=False, encoding="utf-8-sig")
    missing_summary.to_csv(
        output_dir / "missingness_summary.csv", index=False, encoding="utf-8-sig"
    )
    descriptive_summary.to_csv(
        output_dir / "variable_descriptive_summary.csv", index=False, encoding="utf-8-sig"
    )
    descriptive_by_wave.to_csv(
        output_dir / "variable_descriptive_by_wave.csv", index=False, encoding="utf-8-sig"
    )
    descriptive_transitions.to_csv(
        output_dir / "variable_descriptive_transitions.csv", index=False, encoding="utf-8-sig"
    )
    with (output_dir / "feature_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    if csv_copy:
        transitions.to_csv(
            output_dir / "hrs_world_model_transitions.csv", index=False, encoding="utf-8-sig"
        )


def run_self_test() -> int:
    series = pd.Series([1, ".x", ".Q", ".s", ".d", np.nan, 0])
    cleaned, observed, structural = clean_stata_missing(series)
    assert pd.isna(cleaned.iloc[1]) and pd.isna(cleaned.iloc[2]) and pd.isna(cleaned.iloc[3])
    assert observed.iloc[1:4].tolist() == [0, 0, 0]
    assert structural.iloc[1:4].tolist() == [1, 1, 1]
    assert pd.isna(cleaned.iloc[4]) and observed.iloc[4] == 0
    assert pd.isna(cleaned.iloc[5]) and observed.iloc[5] == 0
    assert cleaned.iloc[0] == 1 and cleaned.iloc[6] == 0
    assert "proxy_memory_change" in EXCLUDE_STATE_VARIABLES
    assert "age_at_us_arrival" in EXCLUDE_STATE_VARIABLES

    ownership = recode_home_ownership(
        pd.Series([np.nan, 0.0, 1.0, 125000.0, -1.0])
    )
    assert pd.isna(ownership.iloc[0])
    assert ownership.iloc[1:5].tolist() == [0.0, 1.0, 1.0, 0.0]

    edu = recode_education_level(pd.Series([1, 2, 3, 4, 5, np.nan]))
    assert edu.tolist()[:3] == [1.0, 2.0, 3.0]
    assert pd.isna(edu.iloc[3]) and pd.isna(edu.iloc[4]) and pd.isna(edu.iloc[5])

    sleep = recode_sleep_likert3(pd.Series([1, 2, 3, 4, 0]))
    assert sleep.tolist()[:3] == [1.0, 2.0, 3.0]
    assert pd.isna(sleep.iloc[3]) and pd.isna(sleep.iloc[4])

    mstat = recode_marital_status(pd.Series([1, 6, 8, 9]))
    assert mstat.tolist()[:3] == [1.0, 6.0, 8.0]
    assert pd.isna(mstat.iloc[3])

    contact = recode_weekly_contact(pd.Series([0, 1, 2, np.nan]))
    assert contact.tolist()[:2] == [0.0, 1.0]
    assert pd.isna(contact.iloc[2]) and pd.isna(contact.iloc[3])

    bal = recode_balance_score(pd.Series([1, 4, 5]))
    assert bal.tolist()[:2] == [1.0, 4.0]
    assert pd.isna(bal.iloc[2])

    assert occupation_census_scheme("R10JCOCCC") == "occupc"
    assert occupation_census_scheme("R8JCOCCB") == "occupb"
    # OCCUPB: 1 management→sedentary, 17 sales→light, 20 construction→heavy
    occ_b = recode_current_occupation_physical_demand(
        pd.Series([1, 17, 20, 99, np.nan]),
        scheme="occupb",
    )
    assert occ_b.tolist()[:3] == [1.0, 2.0, 3.0]
    assert pd.isna(occ_b.iloc[3]) and pd.isna(occ_b.iloc[4])
    # OCCUPC: code 17 is office/admin (sedentary), not sales.
    occ_c = recode_current_occupation_physical_demand(
        pd.Series([1, 16, 17, 19]),
        scheme="occupc",
    )
    assert occ_c.tolist() == [1.0, 2.0, 1.0, 3.0]
    assert "current_occupation" in FORCE_MODEL_ORDINAL_VARIABLES
    assert "cancer_incident" in EXCLUDE_MODEL_REWARD_VARIABLES
    assert "cancer_incident" not in DEFAULT_REWARD_WEIGHTS
    # cognition_decline is the next-wave 27-item score (higher=better).
    assert DEFAULT_REWARD_WEIGHTS["cognition_decline"] < 0
    assert WORSENING_REWARD_FROM_STATE[0] == ("adl_total_score", "adl_worsening", ">=", 4.0)
    assert "cognition_decline" not in {name for _, name, _, _ in WORSENING_REWARD_FROM_STATE}
    assert CONTINUOUS_LEVEL_REWARD_FROM_STATE[0] == ("cognition_27_score", "cognition_decline")
    assert BINARY_WORSENING_REWARDS == set(DEFAULT_REWARD_WEIGHTS) - {
        "death_event",
        "hospitalization_event",
        "heart_disease_incident",
        "stroke_incident",
        "cognition_decline",
    }
    for _name in BINARY_WORSENING_REWARDS:
        assert not _is_model_continuous_reward(_name, pd.Series([0.0, 1.0]))
    assert _is_model_continuous_reward(
        "cognition_decline", pd.Series([11.0, 15.0, 20.0])
    )

    # Same wave on both sides so the CPI factors cancel and the 20% threshold
    # can be checked on the raw numbers.
    worsening_demo = pd.DataFrame(
        {
            "wave": [8, 8, 8, 8, 8, 8],
            "next_wave": [8, 8, 8, 8, 8, 8],
            f"{STATE_PREFIX}adl_total_score": [2.0, 3.0, 4.0, 2.0, 1.0, 1.0],
            f"{STATE_MASK_PREFIX}adl_total_score": [1, 1, 1, 0, 1, 1],
            f"{NEXT_STATE_PREFIX}adl_total_score": [4.0, 4.0, 5.0, 4.0, 3.0, 2.0],
            f"{NEXT_STATE_MASK_PREFIX}adl_total_score": [1, 1, 1, 1, 1, 1],
            f"{STATE_PREFIX}cognition_27_score": [15.0, 12.0, 10.0, 15.0, 15.0, 15.0],
            f"{STATE_MASK_PREFIX}cognition_27_score": [1, 1, 1, 0, 1, 1],
            f"{NEXT_STATE_PREFIX}cognition_27_score": [11.0, 11.0, 9.0, 11.0, 12.0, 15.0],
            f"{NEXT_STATE_MASK_PREFIX}cognition_27_score": [1, 1, 1, 1, 1, 1],
            f"{STATE_PREFIX}out_of_pocket_medical_cost": [
                1000.0, 1000.0, 1000.0, -1.0, 0.0, 0.0,
            ],
            f"{STATE_MASK_PREFIX}out_of_pocket_medical_cost": [1, 1, 1, 1, 1, 1],
            f"{NEXT_STATE_PREFIX}out_of_pocket_medical_cost": [
                1200.0, 1199.0, 900.0, 500.0, 50.0, 0.0,
            ],
            f"{NEXT_STATE_MASK_PREFIX}out_of_pocket_medical_cost": [1, 1, 1, 1, 1, 1],
            f"{REWARD_PREFIX}heart_disease_incident": [0.0] * 6,
            f"{REWARD_MASK_PREFIX}heart_disease_incident": [1] * 6,
            f"{REWARD_PREFIX}stroke_incident": [0.0] * 6,
            f"{REWARD_MASK_PREFIX}stroke_incident": [1] * 6,
        }
    )
    derived = derive_transition_rewards(worsening_demo)
    adl = derived[f"{REWARD_PREFIX}adl_worsening"]
    # severe / severe / severe / severe even if current unobserved / non-severe
    assert adl.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert derived[f"{REWARD_MASK_PREFIX}adl_worsening"].tolist() == [1, 1, 1, 1, 1, 1]
    cog = derived[f"{REWARD_PREFIX}cognition_decline"]
    # Next-wave cognition_27_score (current-wave mask is not used).
    assert cog.tolist() == [11.0, 11.0, 9.0, 11.0, 12.0, 15.0]
    assert derived[f"{REWARD_MASK_PREFIX}cognition_decline"].tolist() == [1, 1, 1, 1, 1, 1]
    # +20.0% clears the bar, +19.9% does not, a drop does not; negative dollars
    # are a coding error (NA); any spending from a zero baseline counts.
    oop = derived[f"{REWARD_PREFIX}{OOP_INCREASE_REWARD}"]
    assert oop.tolist()[:3] == [1.0, 0.0, 0.0]
    assert pd.isna(oop.iloc[3])
    assert oop.tolist()[4:] == [1.0, 0.0]
    cvd_demo = pd.DataFrame(
        {
            f"{REWARD_PREFIX}heart_disease_incident": [0.0, 1.0, 0.0, np.nan],
            f"{REWARD_MASK_PREFIX}heart_disease_incident": [1, 1, 1, 0],
            f"{REWARD_PREFIX}stroke_incident": [0.0, 0.0, 1.0, 0.0],
            f"{REWARD_MASK_PREFIX}stroke_incident": [1, 1, 1, 1],
        }
    )
    cvd = derive_cvd_incident(cvd_demo)[f"{REWARD_PREFIX}{CVD_INCIDENT_REWARD}"]
    # free of both / heart only / stroke only / missing heart mask (not at risk)
    assert cvd.tolist()[:3] == [0.0, 1.0, 1.0]
    assert pd.isna(cvd.iloc[3])
    assert derive_cvd_incident(cvd_demo)[
        f"{REWARD_MASK_PREFIX}{CVD_INCIDENT_REWARD}"
    ].tolist() == [1, 1, 1, 0]
    assert "age_years" in FORCE_STATIC_CONTEXT_VARIABLES
    assert "measured_height" in FORCE_STATIC_CONTEXT_VARIABLES
    assert "living_siblings" in PERSON_LEVEL_STATIC_CONTEXT_VARIABLES

    nearby = pd.DataFrame({
        "person_id": ["a", "a", "a", "b", "b"],
        "wave": [8, 9, 10, 8, 10],
        "bmi": [np.nan, 25.0, np.nan, np.nan, 30.0],
        "bmi__observed": [0, 1, 0, 0, 1],
        "smoke": [1.0, np.nan, np.nan, np.nan, np.nan],
        "smoke__observed": [1, 0, 0, 0, 0],
    })
    filled, counts = fill_missing_from_nearby_waves(nearby, ["bmi", "smoke"])
    assert filled.loc[0, "bmi"] == 25.0 and filled.loc[2, "bmi"] == 25.0
    assert filled.loc[3, "bmi"] == 30.0
    assert filled.loc[1, "smoke"] == 1.0 and filled.loc[2, "smoke"] == 1.0
    assert filled["bmi__observed"].tolist() == [0, 1, 0, 0, 1]
    assert counts["bmi"] == 3 and counts["smoke"] == 2

    demo = pd.DataFrame({"x": [1.0, 2.0, 2.0, np.nan]})
    desc = _series_descriptive_stats(demo["x"])
    assert desc["n_nonmissing"] == 3 and desc["median"] == 2.0 and desc["n_unique"] == 2

    print(
        "Self-test passed: all Stata special missings (incl. .x/.q/.s) stay NA; "
        "high-missing state vars excluded; nearby-wave LOCF/NOCB fills values "
        "without changing observation masks; dictionary coding revisions "
        "(education_level 1-3, sleep 1-3, marital 1-8, weekly contact 0/1, "
        "balance 1-4, current_occupation physical-demand 1-3) validated; "
        "descriptive stats helper works."
    )
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.self_test:
        return run_self_test()

    dictionary_path = Path(args.dictionary) if str(args.dictionary).strip() else None
    output_dir = Path(args.output_dir)

    if args.prepare_model_bundle:
        prepare_model_bundle_from_output(
            output_dir=output_dir,
            dictionary_path=dictionary_path,
        )
        return 0

    rand_path = Path(args.rand_file)
    harmonized_path = Path(args.harmonized_file)
    waves = sorted(set(args.waves))

    for path in [rand_path, harmonized_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    if dictionary_path is not None and not dictionary_path.exists():
        raise FileNotFoundError(dictionary_path)
    if any(w not in DEFAULT_WAVES for w in waves):
        raise ValueError(f"Only Waves 8-14 are supported; received {waves}.")
    if not 0 < args.validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.validation_size + args.test_size >= 1:
        raise ValueError("--validation-size + --test-size must be less than 1.")

    specs = load_specs(dictionary_path)
    requested = collect_requested_columns(specs, waves)
    rand = read_dta_selected(rand_path, requested)
    harmonized = read_dta_selected(harmonized_path, requested)
    sources = [("harmonized", harmonized), ("rand", rand)]
    resolver = SourceResolver(sources)
    master_index = harmonized.index.union(rand.index)
    master_index = pd.Index(master_index.astype(str), name="person_id")

    long_live, status_grid, audit, long_live_raw = build_long_data(
        specs=specs,
        resolver=resolver,
        master_index=master_index,
        waves=waves,
        min_live_state_features=args.min_live_state_features,
    )
    LOGGER.info(
        "Live person-wave rows: %d (raw pre-nearby-fill CSV rows: %d)",
        len(long_live),
        len(long_live_raw),
    )

    transitions = build_transitions(
        long_live=long_live,
        status_grid=status_grid,
        specs=specs,
        waves=waves,
        max_transition_years=args.max_transition_years,
    )
    transitions = split_by_person(
        transitions,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    reward_weights = load_reward_weights(
        Path(args.reward_weights_json) if args.reward_weights_json else None
    )
    transitions, reward_scales = add_composite_reward(transitions, reward_weights)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = select_model_features(transitions, specs)
    metadata.update({
        "waves": waves,
        "wave_year": WAVE_YEAR,
        "structural_zero_codes": sorted(STRUCTURAL_ZERO_CODES),
        "exclude_state_variables": sorted(EXCLUDE_STATE_VARIABLES),
        "structural_zero_policy": (
            "All Stata special missings including .x/.q/.s stay NA (observed=0); "
            "no global .x/.q/.s → 0 fill."
        ),
        "other_missing_policy": (
            "State/action missings are filled within person from earlier then later "
            "observed waves (LOCF/NOCB; observation masks unchanged). "
            "Any remaining missings are imputed with training-set median/mode only in model tensors."
        ),
        "long_raw_export": {
            "file": "hrs_world_model_long_raw.csv",
            "description": (
                "Live person-wave long table after extraction/derivations and "
                "wave-10 OOP extended backfill, but before nearby-wave LOCF/NOCB "
                "and before transition construction."
            ),
        },
        "dictionary_coding_revisions": {
            "education_level": "RAEDUCL only; ordinal 1-3",
            "childhood_ses": "mean(RAMEDUC, RAFEDUC) on 0-17; not childhood_health",
            "mother_education": "RAMEDUC years 0-17 (incl. 7.5/8.5)",
            "father_education": "RAFEDUC years 0-17 (incl. 7.5/8.5)",
            "marital_status": "RAND RwMSTAT codes 1-8 retained",
            "sleep_* / rested_in_morning": "RAND 1-3 frequency retained (not reversed)",
            "life_satisfaction": "RwLBSATWLF mean only (W8:1-6, W9-14:1-7); no SATLIFE_H",
            "social_contact_*": "weekly contact binary 0/1",
            "balance_score": "harmonized summary ordinal 1-4",
            "current_occupation": (
                "Coarsened to physical demand 1=sedentary / 2=light / 3=heavy; "
                "OCCUPB (JCOCCB) map for W8–W9 and JCOCCB fallback; "
                "OCCUPC (JCOCCC) map when that source column is used (W10–W14). "
                "Modeled as ordinal. longest_job_occupation (1980 OCCUP) unchanged."
            ),
            "interview_year": "observed calendar year 2006-2019",
            "interview_status": "long_raw keeps completed live interviews (status=1)",
            "adl_worsening / iadl_worsening / mobility_worsening / "
            "cesd_worsening / self_rated_health_worsening": (
                "Binary 1=severe at next wave, 0=observed and non-severe at next wave, "
                "NA=next wave unobserved. Thresholds: ADL>=4, IADL>=4, mobility>=3, "
                "CES-D>=4, self-rated health>=4. See WORSENING_REWARD_FROM_STATE."
            ),
            "cognition_decline": (
                "Continuous next-wave cognition_27_score (0-27, higher=better). "
                "Historical name retained; not a binary decline/severe flag. "
                "See CONTINUOUS_LEVEL_REWARD_FROM_STATE."
            ),
            "cvd_incident": (
                "Binary 1=heart_disease_incident OR stroke_incident among people "
                "currently free of both; 0=stayed free of both; NA=not at risk. "
                "See CVD_INCIDENT_SOURCES."
            ),
            "out_of_pocket_medical_expenditure_next_interval": (
                "Binary 1=higher out-of-pocket spending than the previous "
                "interval after deflating both to 2018 USD, 0=same or lower. "
                "Name keeps the *_next_interval suffix for dictionary "
                "compatibility; it is no longer a dollar amount."
            ),
        },
        "split": {
            "type": "person-level train/validation/test",
            "training_size_target": 1.0 - args.validation_size - args.test_size,
            "validation_size_target": args.validation_size,
            "test_size_target": args.test_size,
            "seed": args.seed,
            "preprocessing_fitted_on": "train only",
        },
        "reward_weights": reward_weights,
        "reward_scales_fitted_on_train": reward_scales,
        "reward_representation": "binary_severe_plus_continuous_cognition",
        "continuous_level_reward_from_state": {
            reward_name: state_name
            for state_name, reward_name in CONTINUOUS_LEVEL_REWARD_FROM_STATE
        },
        "cvd_incident_sources": list(CVD_INCIDENT_SOURCES),
        "source_files": {
            "rand": str(rand_path),
            "harmonized": str(harmonized_path),
            "dictionary": str(dictionary_path) if dictionary_path else str(BUILTIN_SPECS_PATH),
        },
        "counts": {
            "live_person_wave_rows": int(len(long_live)),
            "live_person_wave_rows_raw_pre_nearby_fill": int(len(long_live_raw)),
            "transitions": int(len(transitions)),
            "train_transitions": int(transitions["split"].eq("train").sum()),
            "validation_transitions": int(
                transitions["split"].eq("validation").sum()
            ),
            "test_transitions": int(transitions["split"].eq("test").sum()),
            "unique_people": int(transitions["person_id"].nunique()),
        },
    })

    summary = missingness_summary(long_live, specs)
    descriptive_summary = descriptive_variable_summary(long_live, specs)
    descriptive_by_wave = descriptive_variable_by_wave(long_live, specs)
    descriptive_transitions = descriptive_transition_summary(transitions, specs)
    LOGGER.info(
        "Descriptive summaries: %d long variables, %d wave rows, %d transition rows",
        len(descriptive_summary),
        len(descriptive_by_wave),
        len(descriptive_transitions),
    )
    write_outputs(
        transitions=transitions,
        audit=audit,
        missing_summary=summary,
        descriptive_summary=descriptive_summary,
        descriptive_by_wave=descriptive_by_wave,
        descriptive_transitions=descriptive_transitions,
        metadata=metadata,
        output_dir=output_dir,
        csv_copy=args.csv_copy,
        long_live_raw=long_live_raw,
    )
    model_config = write_hrs_model_bundle(
        transitions=transitions,
        specs=specs,
        metadata=metadata,
        output_dir=output_dir,
        dictionary_path=dictionary_path,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    LOGGER.info(
        "Model bundle: %s / %s / %s",
        model_config["outputs"]["train"],
        model_config["outputs"]["validation"],
        model_config["outputs"]["test"],
    )

    train_people = set(
        transitions.loc[transitions["split"].eq("train"), "person_id"]
    )
    validation_people = set(
        transitions.loc[transitions["split"].eq("validation"), "person_id"]
    )
    test_people = set(
        transitions.loc[transitions["split"].eq("test"), "person_id"]
    )
    all_split_people = train_people | validation_people | test_people
    split_summary = {
        "target_proportions": {
            "train": 1.0 - args.validation_size - args.test_size,
            "validation": args.validation_size,
            "test": args.test_size,
        },
        "train_people": int(len(train_people)),
        "validation_people": int(len(validation_people)),
        "test_people": int(len(test_people)),
        "train_transitions": int(transitions["split"].eq("train").sum()),
        "validation_transitions": int(
            transitions["split"].eq("validation").sum()
        ),
        "test_transitions": int(transitions["split"].eq("test").sum()),
        "train_validation_overlap": int(len(train_people & validation_people)),
        "train_test_overlap": int(len(train_people & test_people)),
        "validation_test_overlap": int(len(validation_people & test_people)),
        "people_covered": int(len(all_split_people)),
        "unique_people_in_transitions": int(transitions["person_id"].nunique()),
    }
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(split_summary, f, ensure_ascii=False, indent=2)

    all_missing_core = audit.groupby("variable")["nonmissing"].sum()
    problematic = [
        name for name, spec in specs.items()
        if (
            spec.role != "Reward variable"
            and name in all_missing_core.index
            and spec.priority in {"Required", "Core"}
            and all_missing_core.get(name, 0) == 0
        )
    ]
    if problematic:
        LOGGER.warning(
            "Required/Core variables with no mapped source values: %s. "
            "Review source_audit.csv and verify the RAND/Harmonized source-field mappings.",
            ", ".join(problematic),
        )

    LOGGER.info("Transitions: %d", len(transitions))
    LOGGER.info(
        "Train/validation/test transitions: %d/%d/%d",
        split_summary["train_transitions"],
        split_summary["validation_transitions"],
        split_summary["test_transitions"],
    )
    LOGGER.info("Output directory: %s", output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        LOGGER.exception("HRS cleaning failed.")
        raise
