#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HRS-aligned Harmonized ELSA G3 + Wave 10 map for external validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

WAVE_YEAR = {
    1: 2002,
    2: 2004,
    3: 2006,
    4: 2008,
    5: 2010,
    6: 2012,
    7: 2014,
    8: 2016,
    9: 2018,
    10: 2021,
}

# World Bank / WDI UK CPI, 2010=100. OOP reward (unused in G3) uses local GBP.
UK_CPI = {
    2002: 82.46,
    2004: 85.60,
    2006: 90.66,
    2008: 97.09,
    2010: 100.00,
    2012: 107.43,
    2014: 112.97,
    2016: 115.61,
    2018: 121.74,
    2021: 128.85,
    2022: 139.83,
    2023: 149.70,
}

# FRED AEXUSUK: annual average USD per 1 GBP (Fed G.5A noon buying rates).
# Applied to income/wealth before the frozen HRS USD scaler. Not PPP.
GBP_USD = {
    2002: 1.5025,
    2003: 1.6347,
    2004: 1.8330,
    2005: 1.8204,
    2006: 1.8434,
    2007: 2.0020,
    2008: 1.8545,
    2009: 1.5661,
    2010: 1.5452,
    2011: 1.6043,
    2012: 1.5853,
    2013: 1.5642,
    2014: 1.6484,
    2015: 1.5284,
    2016: 1.3555,
    2017: 1.2890,
    2018: 1.3363,
    2019: 1.2768,
    2020: 1.2829,
    2021: 1.3764,
    2022: 1.2371,
    2023: 1.2440,
    2024: 1.2781,
    2025: 1.3192,
}

MONEY_GBP_COLUMNS = (
    "total_household_income",
    "total_household_wealth",
)


def usd_per_gbp(year: float | int | None) -> float | None:
    """USD per 1 GBP for an interview year; nearest AEXUSUK year if needed."""
    if year is None or year != year:
        return None
    y = int(year)
    if y in GBP_USD:
        return float(GBP_USD[y])
    years = sorted(GBP_USD)
    if not years:
        return None
    nearest = min(years, key=lambda t: abs(t - y))
    return float(GBP_USD[nearest])


def annualise_w10_weekly_income(weekly: pd.Series) -> pd.Series:
    """IFS totinc_bu_s is weekly GBP; Harmonized itot is already annual."""
    return pd.to_numeric(weekly, errors="coerce") * W10_INCOME_WEEKS_PER_YEAR


def recode_w10_hehavedi(diab: pd.Series) -> tuple[pd.Series, pd.Series]:
    """W10 HEHaveDI: 1=have, no Rx; 2=have, on treatment; 3=no longer have; -1=never."""
    values = pd.to_numeric(diab, errors="coerce")
    dx = pd.Series(
        np.where(values.isna(), np.nan, np.where(values.isin([1.0, 2.0, 3.0]), 1.0, np.where(values.eq(-1.0), 0.0, np.nan))),
        index=values.index,
        dtype="float64",
    )
    oral = pd.Series(
        np.where(values.eq(2.0), 1.0, np.where(values.isin([1.0, 3.0, -1.0]), 0.0, np.nan)),
        index=values.index,
        dtype="float64",
    )
    return dx, oral

MISSING_SENTINELS = set(float(x) for x in range(-30, 0)) | {-99.0}

# Harmonized ELSA vgactx_e: 1=every day … 5=hardly ever/never (HRS VGACTX).
ACT5_TO_HRS = {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: 0.0}
# Native W10 heacta/b/c: 1=>once/week, 2=once/week, 3=1-3/month, 4=hardly ever.
ACT4_TO_HRS = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0}

LBRF_TO_HRS = {1: 1.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 3.0, 6: 6.0, 7: 6.0}
YESNO12 = {1: 1.0, 2: 0.0}
# IFS W10 marstat (derived): 1 married/civil, 2 cohabiting, 3 never-married single,
# 5 divorced, 6 widowed (couple=0). Mapped onto RAND RwMSTAT 1-8.
IFS_MARSTAT_TO_HRS = {1: 1.0, 2: 3.0, 3: 8.0, 5: 5.0, 6: 7.0}
# EUL DIMARR (collapsed DiMar): 1 never, 2 married, 3 civil partner, 4 remarried,
# 5 separated/widowed-collapsed, 6 divorced; 7-11 if a fuller file is used.
DIMARR_TO_HRS = {
    1: 8.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 7.0,
    6: 5.0,
    7: 7.0,
    8: 4.0,
    9: 5.0,
    10: 7.0,
    11: 1.0,
}
# IFS totinc_bu_s is current weekly GBP; Harmonized h{w}itot is annual.
W10_INCOME_WEEKS_PER_YEAR = 52.0
# Same item sets HRS derive uses when overwriting RAND summary scores.
HRS_MOBILITY_SCORE_ITEMS = (
    "difficulty_walk_several_blocks",
    "difficulty_walk_one_block",
    "difficulty_climb_several_flights",
    "difficulty_climb_one_flight",
)
HRS_LARGE_MUSCLE_ITEMS = (
    "difficulty_sit_two_hours",
    "difficulty_rise_chair",
    "difficulty_stoop",
    "difficulty_lift_10lb",
    "difficulty_push_large_object",
)
HRS_FINE_MOTOR_ITEMS = (
    "difficulty_pick_dime",
    "difficulty_reach_arms",
)

ABSORBING_DISEASES = (
    "hypertension_dx",
    "diabetes_dx",
    "cancer_dx",
    "lung_disease_dx",
    "heart_disease_dx",
    "stroke_dx",
    "psychiatric_dx",
    "arthritis_dx",
    "memory_disease_dx",
)

CESD_ITEMS_G3 = (
    ("cesd_depressed", "depres", "negative"),
    ("cesd_effort", "effort", "negative"),
    ("cesd_restless_sleep", "sleepr", "negative"),
    ("cesd_happy", "whappy", "positive_raw"),
    ("cesd_lonely", "flone", "negative"),
    ("cesd_enjoyed_life", "enlife", "positive_raw"),
    ("cesd_sad", "fsad", "negative"),
    ("cesd_could_not_get_going", "going", "negative"),
)
CESD_ITEMS_W10 = (
    ("cesd_depressed", "psceda", "negative"),
    ("cesd_effort", "pscedb", "negative"),
    ("cesd_restless_sleep", "pscedc", "negative"),
    ("cesd_happy", "pscedd", "positive_raw"),
    ("cesd_lonely", "pscede", "negative"),
    ("cesd_enjoyed_life", "pscedf", "positive_raw"),
    ("cesd_sad", "pscedg", "negative"),
    ("cesd_could_not_get_going", "pscedh", "negative"),
)

ADL_ITEMS = {
    "adl_walk_room": "walkra",
    "adl_dress": "dressa",
    "adl_bath": "batha",
    "adl_eat": "eata",
    "adl_bed_transfer": "beda",
    "adl_toilet": "toilta",
}
ADL_ITEMS_W10 = {
    "adl_walk_room": "headlwa",
    "adl_dress": "headldr",
    "adl_bath": "headlba",
    "adl_eat": "headlea",
    "adl_bed_transfer": "headlbe",
    "adl_toilet": "headlwc",
}
IADL_ITEMS = {
    "iadl_phone": "phonea",
    "iadl_money": "moneya",
    "iadl_medication": "medsa",
    "iadl_shopping": "shopa",
    "iadl_meals": "mealsa",
}
IADL_ITEMS_W10 = {
    "iadl_phone": "headlph",
    "iadl_money": "headlmo",
    "iadl_medication": "headlme",
    "iadl_shopping": "headlsh",
    "iadl_meals": "headlpr",
}
MOBILITY_ITEMS = {
    "difficulty_walk_one_block": "walk100a",
    "difficulty_sit_two_hours": "sita",
    "difficulty_rise_chair": "chaira",
    "difficulty_climb_several_flights": "climsa",
    "difficulty_climb_one_flight": "clim1a",
    "difficulty_stoop": "stoopa",
    "difficulty_lift_10lb": "lifta",
    "difficulty_pick_dime": "dimea",
    "difficulty_reach_arms": "armsa",
    "difficulty_push_large_object": "pusha",
}
DISEASE_STEMS = {
    "hypertension_dx": "hibpe",
    "diabetes_dx": "diabe",
    "cancer_dx": "cancre",
    "lung_disease_dx": "lunge",
    "heart_disease_dx": "hearte",
    "stroke_dx": "stroke",
    "psychiatric_dx": "psyche",
    "arthritis_dx": "arthre",
    "memory_disease_dx": "memrye",
}
SERIAL7_TARGETS = (93.0, 86.0, 79.0, 72.0, 65.0)


def _c(coverage: str, compatibility: str, sources: str, rule: str) -> dict[str, str]:
    return {"coverage": coverage, "compatibility": compatibility, "sources": sources, "rule": rule}


VAR_COVERAGE: dict[str, dict[str, str]] = {
    "person_id": _c("available", "High", "idauniq", "Stable ELSA person ID; store as string."),
    "household_id": _c("available", "High", "hh{w}hhid / idahhw10", "Wave-specific household ID."),
    "person_number": _c("partial", "Low", "pn / perid", "ELSA person number, not HRS PN."),
    "wave": _c("derived", "High", "file wave 1-10", "ELSA 1-10 = 2002-2021. Not comparable to HRS 8-14; use interview_year. W10 is native, not in G3."),
    "interview_year": _c("available", "High", "r{w}iwindy / iintdaty", "Valid year; missing → wave calendar year."),
    "interview_month": _c("available", "High", "r{w}iwindm / iintdatm", "1-12."),
    "interview_date": _c("derived", "High", "year/month", "Construct YYYY-MM-15 when day is absent."),
    "delta_time_years": _c("derived", "High", "interview_date", "Same HRS rule."),
    "respondent_weight": _c("available", "High", "r{w}cwtresp / IFS wgt", "Cross-sectional weight; nonpositive → NA."),
    "interview_status": _c("derived", "High", "inw{w} + iwstat + EOL", "1=core (inw=1 or W10 indout 11/13). 5=died (iwstat=5, EOL raxt, radyear). iwstat 4=NR, 6=prior death. G3 does not flag new deaths in W7-W9."),
    "sex": _c("available", "High", "ragender / IFS sex", "1=male→1; 2=female→0."),
    "race_ethnicity": _c("us_only", "None", "raracem / fqethnmr / IFS nonwhite", "G3 raracem is mostly 1=White vs 4=other; W10 fqethnmr 1/2. Not HRS census White/Black/Hispanic. Leave NA."),
    "birth_year": _c("available", "High", "rabyear", "Valid year retained."),
    "birth_month": _c("available", "High", "rabmonth", "1-12 if present."),
    "education_years": _c("unavailable", "None", "raedyrs_e / raeduc_e", "Both are UK qualification groups (1-6 / 1-5), not RAEDYRS years. Do not map category codes onto the HRS years channel."),
    "education_level": _c("available", "High", "raeducl", "ISCED 1-3, same as HRS RAEDUCL."),
    "foreign_born": _c("partial", "Moderate", "rabplace", "1=UK-born→0; other positive codes→1."),
    "age_at_us_arrival": _c("us_only", "None", "", "Leave NA."),
    "childhood_health": _c("available", "High", "rachshlt", "1-5 childhood SRH when observed; 6 folded to 5."),
    "childhood_ses": _c("unavailable", "None", "", "Leave NA."),
    "mother_education": _c("unavailable", "None", "ramomeduage", "School-leaving-age groups 1-7, not HRS RAMEDUC years 0-17. Leave NA."),
    "father_education": _c("unavailable", "None", "radadeduage", "School-leaving-age groups 1-7, not years. Leave NA."),
    "longest_job_occupation": _c("unavailable", "None", "ramaoccup is mother's job; nssec is current", "No respondent longest-job Census code. Leave NA."),
    "longest_job_tenure": _c("unavailable", "None", "", "Leave NA."),
    "age_years": _c("available", "High", "r{w}agey / IFS age", "Age in years. Prefer IFS age on W10 (indager is top-coded)."),
    "marital_status": _c("available", "High", "r{w}mstat / IFS marstat / dimarr", "G3 RAND mstat. W10: IFS marstat 1 married/civil→1, 2 cohabiting→3, 3 never-married single→8, 5 divorced→5, 6 widowed→7; fallback EUL dimarr."),
    "household_size": _c("available", "High", "h{w}hhres / hhtot", "Household resident count."),
    "living_children": _c("available", "High", "h{w}child / r{w}child", "Number of living children."),
    "living_siblings": _c("available", "High", "r{w}livsib", "Number of living siblings. W10 often missing."),
    "living_alone": _c("derived", "High", "household_size", "size==1 → 1."),
    "nursing_home_residence": _c("partial", "Moderate", "r{w}nhmliv / IFS inst", "G3 W3+; W10 IFS inst."),
    "urban_rural_residence": _c("unavailable", "None", "gor / IMD in native", "G3 EUL has no urban/rural. Native gor/IMD are geography, not HRS urban/rural 0/1. Leave NA."),
    "home_ownership": _c("available", "High", "r{w}hownrnt", "1 own → 1; 2/3 observed → 0."),
    "housing_type": _c("unavailable", "None", "IFS tenure", "IFS tenure is own/rent (already in home_ownership), not HRS housing structure type. Leave NA."),
    "total_household_income": _c("available", "Moderate", "h{w}itot / totinc_bu_s", "G3 h{w}itot is annual GBP. W10 totinc_bu_s is weekly benefit-unit GBP × 52, then both × FRED AEXUSUK (USD per GBP, interview-year average) before the HRS USD scaler. Not PPP."),
    "total_household_wealth": _c("available", "Moderate", "h{w}atotb / nettotw_bu_s", "Nominal GBP net wealth × same AEXUSUK rate. Not PPP."),
    "poverty_threshold": _c("unavailable", "None", "", "No US poverty line."),
    "income_to_poverty_ratio": _c("unavailable", "None", "", "Leave NA."),
    "labor_force_status": _c("partial", "Moderate", "r{w}lbrf_e", "1-2→1 working; 3→2 unemp; 4-5→3 retired; 6-7→6 NILF."),
    "currently_working": _c("available", "High", "r{w}work / wpemp", "0/1."),
    "self_employed": _c("partial", "Moderate", "r{w}lbrf_e", "lbrf_e==2 → 1."),
    "hours_worked_per_week": _c("available", "High", "r{w}jhours", "Clip >168."),
    "weeks_worked_per_year": _c("available", "High", "r{w}jweeks_e", "1-52 weeks/year among workers; same scale as HRS. Non-workers typically NA."),
    "current_occupation": _c("unavailable", "None", "r{w}nssec8 / w10nssec8", "UK NS-SEC class, not US Census occupation / physical-demand 1-3. Leave NA."),
    "job_tenure": _c("unavailable", "None", "", "Leave NA."),
    "health_limits_work": _c("available", "High", "r{w}hlthlm", "0/1. W1 often missing."),
    "health_insurance_any": _c("partial", "Moderate", "NHS + r{w}hipriv", "NHS is universal: live interviews → 1. Not Medicare."),
    "medicare_coverage": _c("us_only", "None", "", "NHS is not Medicare. Leave NA."),
    "medicaid_coverage": _c("us_only", "None", "", "Leave NA."),
    "va_coverage": _c("us_only", "None", "", "Leave NA."),
    "long_term_care_insurance": _c("unavailable", "None", "exltcev / r{w}hipriv", "exltcev is 0-100 chance of needing care, not insurance. hipriv is private medical insurance, not LTC. Leave NA."),
    "self_rated_health": _c("available", "High", "r{w}shlt / shlta / hehelf", "1=excellent … 5=poor."),
    "hypertension_dx": _c("available", "High", "r{w}hibpe / heeverbp", "Ever-had; absorb."),
    "diabetes_dx": _c("available", "High", "r{w}diabe / hehavedi", "Ever-had. W10 HEHaveDI: 1 still have no Rx, 2 still have on treatment, 3 no longer have → ever=1; -1 never."),
    "cancer_dx": _c("available", "High", "r{w}cancre / heeverca", "Ever-had; absorb."),
    "lung_disease_dx": _c("available", "High", "r{w}lunge / heevercl|as", "G3 lung ever-had. W10 COPD or asthma."),
    "heart_disease_dx": _c("available", "High", "r{w}hearte / heever heart flags", "Ever-had; absorb."),
    "stroke_dx": _c("available", "High", "r{w}stroke / heeverst", "Ever-had; absorb."),
    "psychiatric_dx": _c("available", "High", "r{w}psyche / heeverps", "Ever-had; absorb."),
    "arthritis_dx": _c("available", "High", "r{w}arthre / heeverar", "Ever-had; absorb."),
    "memory_disease_dx": _c("available", "High", "r{w}memrye / heeverad|dm", "Ever-had; absorb."),
    "multimorbidity_count": _c("derived", "High", "eight disease flags", "Sum of overlapping ever-hads."),
    "eyesight": _c("available", "High", "r{w}sight / heeye", "1-5; 6=blind folded to 5."),
    "near_vision": _c("available", "High", "r{w}nsight", "1-5."),
    "distance_vision": _c("available", "High", "r{w}dsight", "1-5."),
    "hearing": _c("available", "High", "r{w}hearing / hehear", "1-5."),
    "hearing_aid_use": _c("unavailable", "None", "hehear mentions aid", "G3 has no hearing-aid flag. Native heaid* are mobility aids (cane/walker), not hearing aids. Leave NA."),
    "pain_presence": _c("available", "High", "r{w}painfr / hepain", "0/1. W10 1=yes 2=no."),
    "pain_severity": _c("partial", "Moderate", "r{w}painlv", "0-3 ELSA levels."),
    "falls_any": _c("available", "High", "r{w}fall", "0/1 when asked. W10 not in core."),
    "falls_count": _c("available", "High", "r{w}fallnum", "Count; 0 if falls_any=0."),
    "fall_injury": _c("available", "High", "r{w}fallinj", "0/1."),
    "urinary_incontinence": _c("partial", "Moderate", "r{w}urinai", "0/1."),
    "back_problem": _c("unavailable", "None", "", "Leave NA."),
    "sleep_falling_problem": _c("unavailable", "None", "", "Leave NA."),
    "sleep_waking_problem": _c("unavailable", "None", "", "Leave NA."),
    "sleep_early_waking": _c("unavailable", "None", "", "Leave NA."),
    "rested_in_morning": _c("unavailable", "None", "", "Leave NA."),
    "shortness_of_breath": _c("partial", "High", "r{w}breath_e", "0/1. Harmonized only W1-W5. Later waves NA."),
    "dizziness": _c("partial", "Low", "hediz W10", "W10 native: 1-4 problem → 1; 5=never → 0. Not in G3. Not HRS binary wording."),
    "fatigue": _c("unavailable", "None", "", "Leave NA."),
    "adl_walk_room": _c("available", "High", "r{w}walkra / headlwa", "0/1. Present unlike CHARLS."),
    "adl_dress": _c("available", "High", "r{w}dressa / headldr", "0/1."),
    "adl_bath": _c("available", "High", "r{w}batha / headlba", "0/1."),
    "adl_eat": _c("available", "High", "r{w}eata / headlea", "0/1."),
    "adl_bed_transfer": _c("available", "High", "r{w}beda / headlbe", "0/1."),
    "adl_toilet": _c("available", "High", "r{w}toilta / headlwc", "0/1."),
    "iadl_phone": _c("available", "High", "r{w}phonea / headlph", "0/1."),
    "iadl_money": _c("available", "High", "r{w}moneya / headlmo", "0/1."),
    "iadl_medication": _c("available", "High", "r{w}medsa / headlme", "0/1."),
    "iadl_shopping": _c("available", "High", "r{w}shopa / headlsh", "0/1."),
    "iadl_meals": _c("available", "High", "r{w}mealsa / headlpr", "0/1."),
    "difficulty_walk_several_blocks": _c("unavailable", "None", "", "No several-blocks item."),
    "difficulty_walk_one_block": _c("available", "High", "r{w}walk100a", "0/1."),
    "difficulty_sit_two_hours": _c("available", "High", "r{w}sita", "0/1."),
    "difficulty_rise_chair": _c("available", "High", "r{w}chaira", "0/1."),
    "difficulty_climb_several_flights": _c("available", "High", "r{w}climsa", "0/1."),
    "difficulty_climb_one_flight": _c("available", "High", "r{w}clim1a", "0/1."),
    "difficulty_stoop": _c("available", "High", "r{w}stoopa", "0/1."),
    "difficulty_lift_10lb": _c("partial", "High", "r{w}lifta", "ELSA lift ≈10 lb."),
    "difficulty_pick_dime": _c("available", "High", "r{w}dimea", "0/1."),
    "difficulty_reach_arms": _c("available", "High", "r{w}armsa", "0/1."),
    "difficulty_push_large_object": _c("available", "High", "r{w}pusha", "0/1."),
    "adl_total_score": _c("available", "High", "sum of 6 ADL items", "Walk+dress+bath+eat+bed+toilet 0-6. Do not use r{w}adla (max 5)."),
    "iadl_total_score": _c("available", "High", "r{w}iadla / iadlza / item sum", "Prefer iadla/iadlza."),
    "mobility_total_score": _c("partial", "Moderate", "walk100a+climsa+clim1a", "Same 4-item HRS derive (several-blocks + one-block + climb several + climb one). ELSA has no several-blocks, so 0-3. Do not use r{w}mobilsev (0-7)."),
    "large_muscle_total_score": _c("available", "High", "sita+chaira+stoopa+lifta+pusha", "Same 5-item HRS derive, 0-5. Do not use r{w}lgmusa (0-4, no lift)."),
    "fine_motor_total_score": _c("derived", "High", "r{w}dimea + r{w}armsa", "Same two items HRS derive uses (pick dime, reach arms), 0-2. Do not use r{w}finea (dime+eat+dress)."),
    "receives_adl_help": _c("derived", "Moderate", "ADL items", "Any mapped ADL=1."),
    "receives_iadl_help": _c("derived", "Moderate", "IADL items", "Any mapped IADL=1."),
    "self_rated_memory": _c("available", "High", "r{w}slfmem", "1-5."),
    "immediate_word_recall": _c("available", "High", "r{w}imrc / cflisen", "0-10."),
    "delayed_word_recall": _c("available", "High", "r{w}dlrc / cflisd", "0-10."),
    "serial_sevens": _c("partial", "Moderate", "r{w}ser7 W7-9 / W10 cfsv*", "G3 ser7 only W7-9. W10 = count matching 93/86/79/72/65."),
    "backward_counting": _c("partial", "High", "r{w}bwc20", "Same RAND 0-2 as HRS RwBWC20 (not 0-20). G3 only W7-W9."),
    "total_word_recall": _c("available", "High", "r{w}tr20", "imrc+dlrc 0-20."),
    "mental_status_score": _c("proxy", "Moderate", "ser7 + orient", "No TICS mental status."),
    "total_cognition_score": _c("proxy", "Moderate", "tr20 + ser7 + orient", "Not TICS-27."),
    "cognition_27_score": _c("proxy", "Moderate", "imrc+dlrc+ser7", "With ser7: (0-25)×27/25. Else (imrc+dlrc)×27/20. Not TICS-27."),
    "proxy_memory_rating": _c("unavailable", "None", "", "Leave NA."),
    "proxy_memory_change": _c("unavailable", "None", "", "Leave NA."),
    "cesd_depressed": _c("available", "High", "r{w}depres / psceda", "G3 already 0/1. W10 1=yes 2=no."),
    "cesd_effort": _c("available", "High", "r{w}effort / pscedb", "Same."),
    "cesd_restless_sleep": _c("available", "High", "r{w}sleepr / pscedc", "Same."),
    "cesd_happy": _c("available", "High", "r{w}whappy / pscedd", "1=felt happy (RAND polarity)."),
    "cesd_lonely": _c("available", "High", "r{w}flone / pscede", "0/1 symptom."),
    "cesd_sad": _c("available", "High", "r{w}fsad / pscedg", "0/1 symptom."),
    "cesd_could_not_get_going": _c("available", "High", "r{w}going / pscedh", "0/1 symptom."),
    "cesd_enjoyed_life": _c("available", "High", "r{w}enlife / pscedf", "G3 already 0/1 RAND polarity (1=enjoyed). W10 1=yes 2=no."),
    "cesd_score": _c("available", "High", "r{w}cesd / W10 8-item sum", "Native 0-8."),
    "life_satisfaction": _c("available", "High", "r{w}satlife_e / sclifea", "Already 1-7."),
    "loneliness_score": _c("unavailable", "None", "r{w}lonela", "Single 1-7 lonely item on W1/W3/W7. HRS is UCLA-3 mean ~1-3. Do not map."),
    "positive_affect_score": _c("unavailable", "None", "r5panasp13 / casp19", "PANAS-13 mean only W5; CASP is QoL not PANAS. Leave NA."),
    "negative_affect_score": _c("unavailable", "None", "scghq native", "GHQ-12 in native cores is not HRS leave-behind negative affect. Leave NA."),
    "chronic_stress_count": _c("unavailable", "None", "", "Leave NA."),
    "social_support_spouse": _c("unavailable", "None", "", "Leave NA."),
    "social_contact_children": _c("partial", "Moderate", "r{w}kcntf", "Weekly in-person child contact 0/1."),
    "social_contact_friends": _c("partial", "Moderate", "r{w}unfriend", "Meet-friends frequency 1-7; 1-2 → weekly=1, 3-7 → 0. Only W1/W3/W7. Native scfrd* is support quality, not contact."),
    "neighborhood_disorder": _c("unavailable", "None", "", "Leave NA."),
    "neighborhood_cohesion": _c("unavailable", "None", "", "Leave NA."),
    "everyday_discrimination": _c("unavailable", "None", "", "Leave NA."),
    "systolic_bp": _c("available", "High", "r{w}systo1-3", "Nurse waves 2/4/6/8."),
    "diastolic_bp": _c("available", "High", "r{w}diasto1-3", "Nurse waves."),
    "resting_pulse": _c("available", "High", "r{w}pulse1-3", "Nurse waves."),
    "peak_expiratory_flow": _c("available", "High", "r{w}puff", "Nurse waves if present."),
    "grip_strength": _c("available", "High", "max(lgrip,rgrip)", "kg. Nurse waves 2/4/6/8."),
    "left_grip_strength": _c("available", "High", "r{w}lgrip", "kg."),
    "right_grip_strength": _c("available", "High", "r{w}rgrip", "kg."),
    "balance_score": _c("partial", "High", "r{w}balance_e", "Nurse tandem/side-by-side summary 1-4, same as HRS. Only W2/W4/W6. W10 hebal is self-reported imbalance, not this score."),
    "walking_speed_time": _c("available", "High", "r{w}wspeed1 / wspeed2", "Seconds on nurse waves."),
    "measured_height": _c("available", "High", "r{w}mheight", "Meters."),
    "measured_weight": _c("available", "High", "r{w}mweight", "kg."),
    "waist_circumference": _c("available", "High", "r{w}mwaist", "cm when present."),
    "measured_bmi": _c("available", "High", "r{w}mbmi", "kg/m²."),
    "hospitalization": _c("unavailable", "None", "native hehpa; W10 hecvhosp", "Not in G3 EUL. Native W1 hehpa is inpatient yes/no (dropped by later waves / W10). W10 hehps is HRT, not hospital. hecvhosp is COVID-specific. Leave NA."),
    "hospital_stays_count": _c("unavailable", "None", "", "Leave NA."),
    "hospital_nights": _c("unavailable", "None", "", "Leave NA."),
    "nursing_home_use": _c("unavailable", "None", "r{w}pnhm5y / cahnhm", "pnhm5y is 0-100 expected chance of NH in 5 years, not use. cahnhm almost unused. Leave NA."),
    "nursing_home_stays_count": _c("unavailable", "None", "", "Leave NA."),
    "doctor_visits_any": _c("unavailable", "None", "native hegpoft", "Not in G3. Native/W10 hegpoft is GP contact in previous 4 weeks, not HRS ~2-year any-visit. Do not put in the HRS channel."),
    "doctor_visits_count": _c("unavailable", "None", "", "No visit count comparable to HRS."),
    "home_health_care": _c("partial", "Low", "r{w}rfaany_e", "Any formal care 0/1."),
    "outpatient_surgery": _c("unavailable", "None", "", "Leave NA."),
    "dental_visit": _c("unavailable", "None", "r{w}dentalh", "G3 dentalh is 1-5 dental health, not a visit flag. Native hedent is wave-sparse. Leave NA."),
    "prescription_drug_use": _c("unavailable", "None", "", "Leave NA."),
    "out_of_pocket_medical_cost": _c("unavailable", "None", "", "NHS; no HRS-like OOP total in G3."),
    "out_of_pocket_medical_cost_extended": _c("unavailable", "None", "", "Leave NA."),
    "smoking_status": _c("available", "High", "r{w}smokev / smoken / heska", "0 never, 1 former, 2 current."),
    "cigarettes_per_day": _c("available", "High", "r{w}smokef / heskb", "Eligible if current smoker."),
    "alcohol_use": _c("available", "High", "r{w}drink", "0/1."),
    "alcohol_days_per_week": _c("available", "High", "r{w}drinkd_e", "0-7. W1 missing; non-drinkers → 0."),
    "drinks_per_drinking_day": _c("partial", "Moderate", "r{w}drinkn_e", "Eligible if drinker."),
    "binge_drinking": _c("unavailable", "None", "", "Leave NA."),
    "vigorous_activity_frequency": _c("available", "High", "r{w}vgactx_e / heacta", "G3 1-5 inverted to 0-4. W10 heacta is 1-4 (no daily)."),
    "moderate_activity_frequency": _c("available", "High", "r{w}mdactx_e / heactb", "Same."),
    "light_activity_frequency": _c("available", "High", "r{w}ltactx_e / heactc", "Same."),
    "hypertension_treatment": _c("available", "High", "r{w}rxhibp", "Eligible if hypertension_dx=1."),
    "diabetes_oral_medication": _c("partial", "Moderate", "r{w}rxdiab / hehavedi", "G3 any diabetes Rx. W10 HEHaveDI code 2 = currently on treatment (includes insulin, not oral-only); 1 and 3 = not on treatment."),
    "continuation_target": _c("derived", "High", "transition continuation", "Stored as continuation."),
    "death_event": _c("derived", "High", "EOL + iwstat=5 + radyear", "Under-ascertained after W6: G3 has no new iwstat=5 in W7-W9; EOL-A2 is waves 2/3/4/6."),
    "adl_worsening": _c("derived", "High", "next adl_total_score", "HRS rule next ADL≥4."),
    "iadl_worsening": _c("derived", "High", "next iadl_total_score", "HRS rule next IADL≥4."),
    "mobility_worsening": _c("derived", "High", "next mobility_total_score", "If mobility observed."),
    "cesd_worsening": _c("derived", "High", "next cesd_score", "HRS rule CES-D≥4 on 0-8."),
    "cognition_decline": _c("proxy", "Moderate", "next cognition_27_score", "ELSA proxy, not TICS-27."),
    "self_rated_health_worsening": _c("derived", "High", "next self_rated_health", "HRS rule SRH≥4."),
    "hospitalization_event": _c("unavailable", "None", "", "No G3 hospitalization."),
    "heart_disease_incident": _c("derived", "High", "heart_disease_dx t,t+1", "0→1 among at-risk."),
    "stroke_incident": _c("derived", "High", "stroke_dx t,t+1", "Same."),
    "cvd_incident": _c("derived", "High", "heart ∪ stroke", "HRS pooling."),
    "cancer_incident": _c("derived", "High", "cancer_dx t,t+1", "Same."),
    "out_of_pocket_medical_expenditure_next_interval": _c("unavailable", "None", "", "No OOP total."),
}


def coverage_of(name: str) -> str:
    info = VAR_COVERAGE.get(name)
    return str(info["coverage"]) if info else "unavailable"
