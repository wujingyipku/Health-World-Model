#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HRS-aligned Harmonized CHARLS D map for external validation.

Variable names match the HRS cleaner in ``../HRS``.
Sources are Harmonized CHARLS Version D (Gateway/RAND naming), plus EOL-A and Life History-A.
"""
from __future__ import annotations

from typing import Any

WAVE_YEAR = {1: 2011, 2: 2013, 3: 2015, 4: 2018, 5: 2020}

# World Bank / WDI CPI, 2010=100. Used only to deflate CNY medical costs
# before the 20% OOP-increase reward (threshold matches HRS; local currency).
CHINA_CPI = {
    2011: 105.55,
    2013: 111.16,
    2015: 114.92,
    2018: 121.56,
    2020: 128.11,
}

MISSING_SENTINELS = {-99.0, -9.0, -8.0, -1.0}

# Harmonized CHARLS CES-D 10 is 1-4 Likert. HRS CES-D 8 is binary "much of the time".
# Negative items: 3-4 → 1 (symptom). Positive items are stored RAND/HRS polarity
# (1 = felt happy / hopeful) then inverted only when summing cesd_score.
CESD_NEGATIVE_TO_BINARY = {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0}
CESD_POSITIVE_RAW_TO_BINARY = {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0}

# CHARLS raeduc_c (1-10) → HRS RAEDUCL 1-3 and approximate years.
EDUC_C_TO_LEVEL = {
    1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
    6: 2.0, 7: 2.0,
    8: 3.0, 9: 3.0, 10: 3.0,
}
EDUC_C_TO_YEARS = {
    1: 0.0, 2: 3.0, 3: 4.0, 4: 6.0, 5: 9.0,
    6: 12.0, 7: 12.0, 8: 15.0, 9: 16.0, 10: 19.0, 11: 19.0,
}

# CHARLS drinkn_c: 0=none; 1-7 treated as days/week; 8-9 ≈ daily.
DRINKN_TO_DAYS = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 6.0, 7: 7.0, 8: 7.0, 9: 7.0}
# drinkr_c 0-4 amount category → approximate drinks per drinking day.
DRINKR_TO_DRINKS = {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0, 4: 8.0}

# CHARLS lbrf_c → coarsened HRS-like labor force (1 work, 2 unemp, 3 retired, 6 NILF).
LBRF_TO_HRS = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 2.0, 6: 3.0, 7: 6.0, 8: 6.0}

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

# (hrs_name, charls_stem, direction)
CESD_ITEMS: tuple[tuple[str, str, str], ...] = (
    ("cesd_depressed", "depresl", "negative"),
    ("cesd_effort", "effortl", "negative"),
    ("cesd_restless_sleep", "sleeprl", "negative"),
    ("cesd_happy", "whappyl", "positive_raw"),
    ("cesd_lonely", "flonel", "negative"),
    ("cesd_sad", "botherl", "negative"),  # proxy: bothered
    ("cesd_could_not_get_going", "goingl", "negative"),
    ("cesd_enjoyed_life", "fhopel", "positive_raw"),  # proxy: hopeful
)

ADL_ITEMS = {
    "adl_dress": "dressa",
    "adl_bath": "batha",
    "adl_eat": "eata",
    "adl_bed_transfer": "beda",
    "adl_toilet": "toilta",
}
IADL_ITEMS = {
    "iadl_phone": "phonea",
    "iadl_money": "moneya",
    "iadl_medication": "medsa",
    "iadl_shopping": "shopa",
    "iadl_meals": "mealsa",
}
MOBILITY_ITEMS = {
    "difficulty_walk_several_blocks": "walk1kma",
    "difficulty_walk_one_block": "walk100a",
    "difficulty_rise_chair": "chaira",
    "difficulty_climb_several_flights": "climsa",
    "difficulty_stoop": "stoopa",
    "difficulty_lift_10lb": "lifta",
    "difficulty_pick_dime": "dimea",
    "difficulty_reach_arms": "armsa",
}
# Same 4-item HRS derive (several-blocks + one-block + climb several + climb one).
# CHARLS has no clim1a, so observed range is 0-3. Do not use r{w}mobilsev (0-7).
HRS_MOBILITY_SCORE_ITEMS = (
    "difficulty_walk_several_blocks",
    "difficulty_walk_one_block",
    "difficulty_climb_several_flights",
    "difficulty_climb_one_flight",
)
HRS_FINE_MOTOR_ITEMS = (
    "difficulty_pick_dime",
    "difficulty_reach_arms",
)

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


def exercise_days_to_hrs_frequency(days: float) -> float | None:
    if days != days:
        return None
    if days <= 0:
        return 0.0
    if days < 1:
        return 1.0
    if days < 2:
        return 2.0
    if days < 7:
        return 3.0
    return 4.0


def _c(coverage: str, compatibility: str, sources: str, rule: str) -> dict[str, str]:
    return {
        "coverage": coverage,
        "compatibility": compatibility,
        "sources": sources,
        "rule": rule,
    }


VAR_COVERAGE: dict[str, dict[str, str]] = {
    "person_id": _c("available", "High", "ID", "Harmonized person ID; store as string."),
    "household_id": _c("available", "High", "householdID / hhid", "Household ID; retain original formatting."),
    "person_number": _c("partial", "Low", "pn / pnc", "CHARLS person number within household, not HRS PN."),
    "wave": _c("derived", "High", "file wave 1-5", "CHARLS waves 1-5 (2011-2020). Not numerically comparable to HRS 8-14; use interview_year."),
    "interview_year": _c("available", "High", "r{w}iwy", "Valid year; missing → wave calendar year."),
    "interview_month": _c("available", "High", "r{w}iwm", "1-12; negative/-99 → NA."),
    "interview_date": _c("derived", "High", "r{w}iwy / r{w}iwm", "Construct YYYY-MM-15 when day is absent."),
    "delta_time_years": _c("derived", "High", "interview_date", "Same HRS rule: years between observed interviews."),
    "respondent_weight": _c("available", "High", "r{w}wtresp / r{w}wtrespb", "Cross-sectional respondent weight; nonpositive → NA."),
    "interview_status": _c("derived", "High", "inw{w} + r{w}iwstat + EOL", "1=core interview (inw=1); 5=died this interval (iwstat=5, inwxt=1, or EOL raxt). iwstat 4 is nonresponse, not death; 6 is already deceased in a prior wave."),
    "sex": _c("available", "High", "ragender", "1=male→1; 2=female→0."),
    "race_ethnicity": _c("us_only", "None", "", "Not collected. Leave NA."),
    "birth_year": _c("available", "High", "rabyear / LH rabyear", "Valid year retained."),
    "birth_month": _c("available", "High", "rabmonth / LH rabmonth", "1-12."),
    "education_years": _c("partial", "Moderate", "raeduc_c", "CHARLS 1-10 credential → 0/3/4/6/9/12/15/16/19 years. Not US years."),
    "education_level": _c("available", "High", "raeducl / raeduc_c", "Prefer raeducl (ISCED 1-3, same as HRS RAEDUCL). Else map raeduc_c 1-5→1, 6-7→2, 8-10→3."),
    "foreign_born": _c("unavailable", "None", "", "No nativity item comparable to HRS."),
    "age_at_us_arrival": _c("us_only", "None", "", "US immigration item; leave NA."),
    "childhood_health": _c("proxy", "Moderate", "rahltcom / LH rachchlt", "1-5 health vs other children before 16 (1=much healthier … 5=much less). Direction matches HRS RACHSHLT (higher=worse) but is relative, not absolute childhood SRH. Illness flags ramischlth/rachbedhlth used only if the 1-5 item is missing."),
    "childhood_ses": _c("proxy", "Moderate", "rameduc_c / rafeduc_c", "HRS is mean(RAMEDUC, RAFEDUC) years 0-17. CHARLS parents are credential 1-11 → same year map as education_years, then mean, clipped 0-17. rafinacom is a 1-5 childhood-finance ladder — do not map (wrong scale). ramomoccup_c is farm/non-farm guardian job, not SES."),
    "mother_education": _c("partial", "Moderate", "ramomeducl", "Same 1-3 ISCED as raeducl, not years."),
    "father_education": _c("partial", "Moderate", "radadeducl", "Same 1-3 ISCED as raeducl, not years."),
    "longest_job_occupation": _c("unavailable", "None", "ramomoccup_c / radadoccup_c", "Those are guardian occupation before 17 (farm vs non-farm, 1-2), not the respondent's longest job. No r{w}jocc in D."),
    "longest_job_tenure": _c("unavailable", "None", "", "No respondent job-tenure years in Harmonized D."),
    "age_years": _c("available", "High", "r{w}agey", "Age in years at interview."),
    "marital_status": _c("available", "High", "r{w}mstat", "Already RAND/HRS codes: 1 married, 3 partnered, 4 separated, 5 divorced, 7 widowed, 8 never. Keep."),
    "household_size": _c("available", "High", "h{w}hhres", "Household resident count."),
    "living_children": _c("available", "High", "h{w}child", "Number of living children."),
    "living_siblings": _c("available", "High", "r{w}livsib", "Number of living siblings."),
    "living_alone": _c("derived", "High", "household_size", "hhres==1 → 1; >1 → 0."),
    "nursing_home_residence": _c("partial", "Moderate", "r{w}nhmliv", "Available W2+; W1 NA."),
    "urban_rural_residence": _c("available", "High", "h{w}rural / hh{w}rural / r{w}urban", "Household residence: 1=rural, 0=urban (same as HRS H{w}RURAL). Prefer h{w}rural / hh{w}rural. If missing (W5 has no h5rural), invert r{w}urban (1=urban → 0=rural). Do not use r{w}rural2 — it is not the inverse of r{w}urban and is a different construct than h{w}rural."),
    "home_ownership": _c("available", "High", "hh{w}ahrto / h{w}ahrto", "1 own → 1; else 0 if observed."),
    "housing_type": _c("unavailable", "None", "r{w}housewka", "housewka is housework difficulty, not housing type. Leave NA."),
    "total_household_income": _c("available", "Moderate", "hh{w}itot / h{w}itot", "CNY. Do not convert to USD."),
    "total_household_wealth": _c("available", "Moderate", "hh{w}atotb / h{w}atotb", "CNY net household assets; may be negative."),
    "poverty_threshold": _c("unavailable", "None", "", "No RAND-style US poverty line."),
    "income_to_poverty_ratio": _c("unavailable", "None", "", "Leave NA."),
    "labor_force_status": _c("partial", "Moderate", "r{w}lbrf_c", "CHARLS 1-8 coarsened: 1-4→1 working; 5→2 unemployed; 6→3 retired; 7-8→6 NILF."),
    "currently_working": _c("available", "High", "r{w}work", "0/1."),
    "self_employed": _c("partial", "Moderate", "r{w}lbrf_c", "lbrf_c==3 → 1; other observed labor codes → 0."),
    "hours_worked_per_week": _c("available", "High", "r{w}jhourtot", "Hours; implausible >168 → NA."),
    "weeks_worked_per_year": _c("available", "High", "r{w}jweeks_c", "Weeks/year on main job (months×4.345). Same RAND construct as HRS JWEEKS; clip 0-52. Non-workers typically NA (.w)."),
    "current_occupation": _c("unavailable", "None", "", "No r{w}jocc / Census occupation in Harmonized D."),
    "job_tenure": _c("unavailable", "None", "", "No r{w}jyears / tenure in Harmonized D."),
    "health_limits_work": _c("available", "High", "r{w}hlthlm_c", "0/1."),
    "health_insurance_any": _c("available", "Moderate", "r{w}higov / hipriv / hiothp", "Any of public/private/other = 1."),
    "medicare_coverage": _c("us_only", "None", "", "Leave NA."),
    "medicaid_coverage": _c("us_only", "None", "", "UEBMI/NRCMS is not Medicaid. Leave NA rather than a false analogue."),
    "va_coverage": _c("us_only", "None", "", "Leave NA."),
    "long_term_care_insurance": _c("unavailable", "None", "rahltcom", "rahltcom is childhood health vs peers, not LTC insurance. Leave NA."),
    "self_rated_health": _c("available", "High", "r{w}shlt / r{w}shlta", "1=excellent … 5=poor. W4-W5 often shlta only."),
    "hypertension_dx": _c("available", "High", "r{w}hibpe", "Harmonized ever-had; absorb 0 after first 1."),
    "diabetes_dx": _c("available", "High", "r{w}diabe", "Ever-had; absorb."),
    "cancer_dx": _c("available", "High", "r{w}cancre", "Ever-had; absorb."),
    "lung_disease_dx": _c("available", "High", "r{w}lunge", "Ever-had; absorb."),
    "heart_disease_dx": _c("available", "High", "r{w}hearte", "Ever-had; absorb."),
    "stroke_dx": _c("available", "High", "r{w}stroke", "Ever-had; absorb."),
    "psychiatric_dx": _c("available", "High", "r{w}psyche", "Ever-had; absorb."),
    "arthritis_dx": _c("available", "High", "r{w}arthre", "Ever-had; absorb."),
    "memory_disease_dx": _c("available", "High", "r{w}memrye", "Ever-had; absorb."),
    "multimorbidity_count": _c("derived", "High", "eight disease flags", "Sum of HRS-overlapping disease ever-hads."),
    "eyesight": _c("unavailable", "None", "native DA041", "CHARLS questionnaire has DA041; Harmonized D never built r{w}sight. Native Health_Status files are not in this OpenHealth tree."),
    "near_vision": _c("unavailable", "None", "", "Not in Harmonized CHARLS D."),
    "distance_vision": _c("unavailable", "None", "", "Not in Harmonized CHARLS D."),
    "hearing": _c("unavailable", "None", "native DA049", "Questionnaire has DA049; D has no r{w}hearing (r{w}hearte is heart disease)."),
    "hearing_aid_use": _c("unavailable", "None", "", "Not in Harmonized CHARLS D."),
    "pain_presence": _c("unavailable", "None", "native DA032", "Questionnaire has pain; D did not construct r{w}painfr."),
    "pain_severity": _c("unavailable", "None", "", "Not in Harmonized CHARLS D."),
    "falls_any": _c("unavailable", "None", "", "Do-file / D have no r{w}fall. Leave NA."),
    "falls_count": _c("unavailable", "None", "", "Leave NA."),
    "fall_injury": _c("unavailable", "None", "", "Leave NA."),
    "urinary_incontinence": _c("partial", "Moderate", "r{w}urina", "CHARLS ADL continence item (0/1 difficulty), not a separate UI battery."),
    "back_problem": _c("unavailable", "None", "", "Leave NA."),
    "sleep_falling_problem": _c("unavailable", "None", "r{w}sleeprl / r5sleep", "sleeprl is CES-D restless sleep; r5sleep is hours slept. Neither is the HRS falling-asleep item. Leave NA."),
    "sleep_waking_problem": _c("unavailable", "None", "", "Leave NA."),
    "sleep_early_waking": _c("unavailable", "None", "", "Leave NA."),
    "rested_in_morning": _c("unavailable", "None", "", "Leave NA."),
    "shortness_of_breath": _c("unavailable", "None", "", "Not in Harmonized CHARLS D."),
    "dizziness": _c("unavailable", "None", "", "Not in Harmonized CHARLS D."),
    "fatigue": _c("unavailable", "None", "", "Not in Harmonized CHARLS D."),
    "adl_walk_room": _c("unavailable", "None", "", "CHARLS 6-ADL has no walk-across-room item (uses continence instead). Do-file: no walkra."),
    "adl_dress": _c("available", "High", "r{w}dressa", "0/1 any difficulty."),
    "adl_bath": _c("available", "High", "r{w}batha", "0/1."),
    "adl_eat": _c("available", "High", "r{w}eata", "0/1."),
    "adl_bed_transfer": _c("available", "High", "r{w}beda", "0/1."),
    "adl_toilet": _c("available", "High", "r{w}toilta", "0/1."),
    "iadl_phone": _c("available", "High", "r{w}phonea", "0/1. Often missing in W1."),
    "iadl_money": _c("available", "High", "r{w}moneya", "0/1."),
    "iadl_medication": _c("available", "High", "r{w}medsa", "0/1."),
    "iadl_shopping": _c("available", "High", "r{w}shopa", "0/1."),
    "iadl_meals": _c("available", "High", "r{w}mealsa", "0/1."),
    "difficulty_walk_several_blocks": _c("partial", "Moderate", "r{w}walk1kma", "1 km walk, not several US blocks."),
    "difficulty_walk_one_block": _c("partial", "Moderate", "r{w}walk100a", "100 m walk."),
    "difficulty_sit_two_hours": _c("unavailable", "None", "", "Do-file: ***no sita***. Leave NA."),
    "difficulty_rise_chair": _c("available", "High", "r{w}chaira", "0/1."),
    "difficulty_climb_several_flights": _c("partial", "Moderate", "r{w}climsa", "CHARLS several stairs, not several flights."),
    "difficulty_climb_one_flight": _c("unavailable", "None", "", "Do-file: ***no clim1a***. climsa (several stairs) is already mapped to several-flights."),
    "difficulty_stoop": _c("available", "High", "r{w}stoopa", "0/1."),
    "difficulty_lift_10lb": _c("partial", "High", "r{w}lifta", "5 kg ≈ 11 lb."),
    "difficulty_pick_dime": _c("available", "High", "r{w}dimea", "0/1."),
    "difficulty_reach_arms": _c("available", "High", "r{w}armsa", "0/1."),
    "difficulty_push_large_object": _c("unavailable", "None", "", "Do-file: ***no pusha***. Leave NA."),
    "adl_total_score": _c("partial", "High", "r{w}adlab_c", "0-6 including continence. HRS 6-ADL uses walk-room instead of urine. Severe threshold ≥4 still used."),
    "iadl_total_score": _c("available", "High", "r{w}iadla / iadlza / item sum", "Prefer iadla/iadlza; else sum of 5 HRS IADL items."),
    "mobility_total_score": _c("partial", "Moderate", "walk1kma+walk100a+climsa", "Same 4-item HRS derive (several-blocks + one-block + climb several + climb one). CHARLS has no clim1a, so 0-3. Do not use r{w}mobilsev (0-7). W5 has none of these items."),
    "large_muscle_total_score": _c("unavailable", "None", "chaira/stoopa/lifta", "HRS sums sit+chair+stoop+lift+push (0-5). CHARLS lacks sita and pusha; a 3-item sum would not match the trained scale. Leave NA."),
    "fine_motor_total_score": _c("derived", "High", "r{w}dimea + r{w}armsa", "Same two items HRS derive uses (pick dime, reach arms). 0-2. No RAND FINEA column in D."),
    "receives_adl_help": _c("derived", "Moderate", "ADL items", "Any of mapped ADLs = 1."),
    "receives_iadl_help": _c("derived", "Moderate", "IADL items", "Any of mapped IADLs = 1."),
    "self_rated_memory": _c("available", "High", "r{w}slfmem", "1-5."),
    "immediate_word_recall": _c("available", "High", "r{w}imrc", "0-10 CHARLS word list."),
    "delayed_word_recall": _c("available", "High", "r{w}dlrc", "0-10."),
    "serial_sevens": _c("available", "High", "r{w}ser7", "0-5."),
    "backward_counting": _c("unavailable", "None", "", "No r{w}bwc20 in Harmonized CHARLS D."),
    "total_word_recall": _c("available", "High", "r{w}tr20", "imrc+dlrc, 0-20."),
    "mental_status_score": _c("proxy", "Moderate", "r{w}ser7 + orient + draw", "CHARLS orientation 0-4 + drawing 0-1 + serial 7; not HRS TICS mental status."),
    "total_cognition_score": _c("proxy", "Moderate", "tr20 + ser7 + orient + draw", "CHARLS battery total, not HRS TICS-27 raw."),
    "cognition_27_score": _c("proxy", "Moderate", "imrc+dlrc+ser7", "(0-25) × 27/25. No backward counting. Not TICS-27."),
    "proxy_memory_rating": _c("unavailable", "None", "", "Excluded from HRS model; leave NA."),
    "proxy_memory_change": _c("unavailable", "None", "", "Excluded from HRS model; leave NA."),
    "cesd_depressed": _c("available", "High", "r{w}depresl", "Likert 1-4 → HRS binary (3-4=1)."),
    "cesd_effort": _c("available", "High", "r{w}effortl", "Same binarization."),
    "cesd_restless_sleep": _c("available", "High", "r{w}sleeprl", "Same."),
    "cesd_happy": _c("available", "High", "r{w}whappyl", "1=felt happy (RAND polarity): Likert 3-4 → 1. Invert only in cesd_score."),
    "cesd_lonely": _c("available", "High", "r{w}flonel", "Same as depressed."),
    "cesd_sad": _c("proxy", "Moderate", "r{w}botherl", "CHARLS has no 'sad'; use bothered."),
    "cesd_could_not_get_going": _c("available", "High", "r{w}goingl", "Same."),
    "cesd_enjoyed_life": _c("proxy", "Moderate", "r{w}fhopel", "CHARLS has no 'enjoyed life'; use hopeful. Store 1=felt hopeful (Likert 3-4); invert only in cesd_score."),
    "cesd_score": _c("partial", "Moderate", "8 mapped items", "0-8 symptom count: negatives + (1-happy) + (1-hopeful). Two items are proxies. Do not use native cesd10 0-30."),
    "life_satisfaction": _c("partial", "Moderate", "r{w}satlife", "CHARLS 1-5; rescale 1+(x-1)*6/4 toward HRS 1-7."),
    "loneliness_score": _c("unavailable", "None", "r{w}flonel", "flonel is the CES-D lonely item (already in cesd_lonely), not UCLA-3. Do not map."),
    "positive_affect_score": _c("unavailable", "None", "", "Leave NA."),
    "negative_affect_score": _c("unavailable", "None", "", "Leave NA."),
    "chronic_stress_score": _c("unavailable", "None", "", "Leave NA."),
    "chronic_stress_count": _c("unavailable", "None", "", "Leave NA."),
    "social_support_spouse": _c("unavailable", "None", "", "Leave NA."),
    "social_contact_children": _c("partial", "Moderate", "h{w}kcntf", "Weekly in-person child contact 0/1, not HRS frequency."),
    "social_contact_friends": _c("unavailable", "None", "r{w}socwk", "socwk is any social activity from DA056 (mahjong/cards/etc.), not weekly friend contact. Do not map."),
    "neighborhood_disorder": _c("unavailable", "None", "", "Leave NA."),
    "neighborhood_cohesion": _c("unavailable", "None", "", "Leave NA."),
    "everyday_discrimination": _c("unavailable", "None", "", "Leave NA."),
    "systolic_bp": _c("available", "High", "r{w}systo1-3", "Mean of valid readings; -99 excluded. Sparse in W5."),
    "diastolic_bp": _c("available", "High", "r{w}diasto1-3", "Mean of valid readings."),
    "resting_pulse": _c("available", "High", "r{w}pulse1-3 / r{w}pulse", "Mean of valid readings."),
    "peak_expiratory_flow": _c("available", "High", "r{w}puff / puff1", "L/min; W1-W3 mainly."),
    "grip_strength": _c("available", "High", "r{w}gripsum / max(lgrip,rgrip)", "kg. W5 often missing."),
    "left_grip_strength": _c("available", "High", "r{w}lgrip", "kg."),
    "right_grip_strength": _c("available", "High", "r{w}rgrip", "kg."),
    "balance_score": _c("available", "High", "r{w}balance", "Same RAND 1-4 tandem summary (semi-tandem / side-by-side / full tandem). W1-W3 only."),
    "walking_speed_time": _c("available", "High", "r{w}wspeed1 / wspeed2", "Seconds; mean of two trials."),
    "measured_height": _c("available", "High", "r{w}mheight", "Meters (same as RAND MHEIGHT)."),
    "measured_weight": _c("available", "High", "r{w}mweight", "kg."),
    "waist_circumference": _c("available", "High", "r{w}mwaist", "cm."),
    "measured_bmi": _c("available", "High", "r{w}mbmi", "kg/m²; else weight/height²."),
    "hospitalization": _c("available", "High", "r{w}hosp1y", "Past-year any admission 0/1 (not since previous interview)."),
    "hospital_stays_count": _c("available", "High", "r{w}hsptim1y", "Past-year count."),
    "hospital_nights": _c("partial", "Moderate", "r{w}hspnite", "Nights of last stay, not total nights."),
    "nursing_home_use": _c("unavailable", "None", "r{w}nhmliv", "nhmliv is current NH residence (already mapped). No interval stay/count in D."),
    "nursing_home_stays_count": _c("unavailable", "None", "", "Leave NA."),
    "doctor_visits_any": _c("partial", "Moderate", "r{w}doctor1m", "Past-month outpatient, not HRS 2-year doctor visits."),
    "doctor_visits_count": _c("partial", "Moderate", "r{w}doctim1m", "Past-month count."),
    "home_health_care": _c("partial", "Low", "r{w}rfaany", "Any formal care, not HRS home-health visits."),
    "outpatient_surgery": _c("unavailable", "None", "", "Leave NA."),
    "dental_visit": _c("partial", "Moderate", "r{w}dentst1y / dentim1y", "W2-W3 mainly; count>0 or yes→1."),
    "prescription_drug_use": _c("unavailable", "None", "r{w}rxhibp / rxheart / …", "Disease-specific meds only; no r{w}rxany. OR of listed Rx would miss other drugs. Leave NA."),
    "out_of_pocket_medical_cost": _c("partial", "Moderate", "oophos1y + oopdoc1m + oopden1y", "CNY; mixed 1-year inpatient/dental and 1-month outpatient. Do not mix with USD."),
    "out_of_pocket_medical_cost_extended": _c("unavailable", "None", "", "Leave NA."),
    "smoking_status": _c("available", "High", "r{w}smokev / smoken", "0 never, 1 former, 2 current. Auxiliary in HRS model but extracted."),
    "cigarettes_per_day": _c("available", "High", "r{w}smokef", "Eligible only if current smoker; else NA with eligible=0."),
    "alcohol_use": _c("available", "High", "r{w}drinkl / drink", "0/1 current drinker."),
    "alcohol_days_per_week": _c("partial", "Moderate", "r{w}drinkn_c", "0-7 days; 8-9 folded to 7. Non-drinkers → 0."),
    "drinks_per_drinking_day": _c("proxy", "Low", "r{w}drinkr_c", "Category 1-4 → 1/2/4/8 drinks. Eligible if drinker."),
    "binge_drinking": _c("unavailable", "None", "r{w}drinkn_c / drinkr_c", "Days/week and amount category are already alcohol_days / drinks_per_day. 8-9 on drinkn is more-than-daily, not 4+/5+ drinks in one sitting. Leave NA."),
    "vigorous_activity_frequency": _c("partial", "Moderate", "r{w}vgact_c / vgactx_c", "Days/week → HRS 0-4 frequency. Indicator=no → 0."),
    "moderate_activity_frequency": _c("partial", "Moderate", "r{w}mdact_c / mdactx_c", "Same mapping."),
    "light_activity_frequency": _c("partial", "Moderate", "r{w}ltact_c / ltactx_c", "Same mapping."),
    "hypertension_treatment": _c("available", "High", "r{w}rxhibp_c / rxhibp", "Eligible if hypertension_dx=1."),
    "diabetes_oral_medication": _c("partial", "Moderate", "r{w}rxdiab_c", "Any diabetes treatment, not oral-only (insulin is rxdiabi)."),
    "continuation_target": _c("derived", "High", "transition continuation", "Not a reward head; stored as continuation."),
    "death_event": _c("derived", "High", "EOL raxt / inw{w}xt / iwstat=5", "1 if the next record is death in the interval. CHARLS iwstat 6 is prior-wave death and is not a new event."),
    "adl_worsening": _c("derived", "High", "next adl_total_score", "HRS rule: next-wave ADL ≥ 4."),
    "iadl_worsening": _c("derived", "High", "next iadl_total_score", "HRS rule: next-wave IADL ≥ 4."),
    "mobility_worsening": _c("derived", "Moderate", "next mobility_total_score", "HRS rule next mobility ≥ 3 on the 4-item (CHARLS observed 0-3) score."),
    "cesd_worsening": _c("derived", "Moderate", "next cesd_score", "HRS rule CES-D ≥ 4 on the 8-item 0-8 score."),
    "cognition_decline": _c("proxy", "Moderate", "next cognition_27_score", "Copies next-wave CHARLS proxy 0-27, not TICS-27."),
    "self_rated_health_worsening": _c("derived", "High", "next self_rated_health", "HRS rule SRH ≥ 4."),
    "hospitalization_event": _c("partial", "Moderate", "next hosp1y", "Next-wave past-year hospitalization, not since previous interview."),
    "heart_disease_incident": _c("derived", "High", "heart_disease_dx t and t+1", "At-risk if current dx=0 and next observed; 1 if 0→1."),
    "stroke_incident": _c("derived", "High", "stroke_dx t and t+1", "Same incidence rule."),
    "cvd_incident": _c("derived", "High", "heart ∪ stroke incident", "HRS pooling among people free of both."),
    "cancer_incident": _c("derived", "High", "cancer_dx t and t+1", "Same incidence rule; excluded from the HRS model reward head."),
    "out_of_pocket_medical_expenditure_next_interval": _c(
        "partial",
        "Moderate",
        "out_of_pocket_medical_cost",
        "Same 20% real-increase rule, using China CPI to 2018 CNY rather than US CPI-U dollars.",
    ),
}


def coverage_of(name: str) -> str:
    info = VAR_COVERAGE.get(name)
    return str(info["coverage"]) if info else "unavailable"


def is_extractable(name: str) -> bool:
    return coverage_of(name) in {"available", "partial", "proxy", "derived"}
