#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Clean Harmonized CHARLS D into HRS-compatible external-validation tables."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
HRS_DIR = SCRIPT_DIR.parent / "HRS"
sys.path.insert(0, str(HRS_DIR))
import clean_hrs as hrs  # noqa: E402

from charls_hrs_mappings import (  # noqa: E402
    ABSORBING_DISEASES,
    ADL_ITEMS,
    CESD_ITEMS,
    CESD_NEGATIVE_TO_BINARY,
    CESD_POSITIVE_RAW_TO_BINARY,
    CHINA_CPI,
    DISEASE_STEMS,
    DRINKN_TO_DAYS,
    DRINKR_TO_DRINKS,
    EDUC_C_TO_LEVEL,
    EDUC_C_TO_YEARS,
    HRS_FINE_MOTOR_ITEMS,
    HRS_MOBILITY_SCORE_ITEMS,
    IADL_ITEMS,
    LBRF_TO_HRS,
    MISSING_SENTINELS,
    MOBILITY_ITEMS,
    VAR_COVERAGE,
    WAVE_YEAR,
    coverage_of,
    exercise_days_to_hrs_frequency,
)

LOGGER = logging.getLogger("charls_hrs_cleaning")
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "data" / "charls"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parents[1] / "data" / "charls"
DEFAULT_WAVES = [1, 2, 3, 4, 5]
DEFAULT_HRS_PREPROCESS = SCRIPT_DIR.parents[1] / "data" / "hrs" / "HRS_preprocessing.json"
DEFAULT_HRS_CONFIG = SCRIPT_DIR.parents[1] / "data" / "hrs" / "HRS_model_config.json"
CORE_FILE = "H_CHARLS_D_Data_w5.csv"
EOL_FILE = "H_CHARLS_EOL_a_w5.csv"
LH_FILE = "H_CHARLS_LH_a_w5.csv"
SKIP_NEARBY_FILL: set[str] = set()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean CHARLS into HRS external-validation tables.")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--dictionary", default="")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--waves", nargs="+", type=int, default=DEFAULT_WAVES)
    p.add_argument("--max-transition-years", type=float, default=5.5)
    p.add_argument("--hrs-preprocessing", default=str(DEFAULT_HRS_PREPROCESS))
    p.add_argument("--hrs-model-config", default=str(DEFAULT_HRS_CONFIG))
    p.add_argument("--no-hrs-scaler", action="store_true")
    p.add_argument("--csv-copy", action="store_true")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def _lookup(columns: Sequence[str]) -> dict[str, str]:
    return {str(c).lower(): str(c) for c in columns}


def normalize_identifier(series: pd.Series, width: int) -> pd.Series:
    text = series.astype("string").str.strip()
    mask = text.notna() & text.ne("") & text.ne("<NA>")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out.loc[mask] = text.loc[mask].str.replace(r"\.0$", "", regex=True).str.zfill(width)
    return out


def to_num(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values.isin(list(MISSING_SENTINELS)))


def recode_map(series: pd.Series, mapping: dict[Any, float]) -> pd.Series:
    return to_num(series).map(mapping)


def pick(frame: pd.DataFrame, wave: int, stems: Sequence[str]) -> pd.Series | None:
    lookup = _lookup(frame.columns)
    for stem in stems:
        stem_l = str(stem).lower().replace("{w}", str(wave))
        candidates = [
            stem_l,
            f"r{wave}{stem_l}",
            f"h{wave}{stem_l}",
            f"hh{wave}{stem_l}",
        ]
        for cand in candidates:
            hit = lookup.get(cand)
            if hit is not None:
                return to_num(frame[hit])
    return None


def pick_cols(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series | None:
    lookup = _lookup(frame.columns)
    for name in names:
        hit = lookup.get(str(name).lower())
        if hit is not None:
            return to_num(frame[hit])
    return None


def sum_named(out: pd.DataFrame, names: Sequence[str], min_count: int) -> pd.Series:
    parts = [out[n] for n in names if isinstance(out.get(n), pd.Series)]
    if not parts:
        return pd.Series(np.nan, index=out.index)
    mat = pd.concat(parts, axis=1)
    return mat.sum(axis=1, min_count=min_count)


def mean_loop(frame: pd.DataFrame, wave: int, prefixes: Sequence[str], low: float, high: float) -> pd.Series:
    parts = []
    for prefix in prefixes:
        raw = pick(frame, wave, (prefix,))
        if raw is None:
            continue
        parts.append(raw.where(raw.between(low, high)))
    if not parts:
        return pd.Series(np.nan, index=frame.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def bin01(series: pd.Series | None) -> pd.Series | float:
    if series is None:
        return np.nan
    return series.where(series.isin([0.0, 1.0]))


def extract_wave(core: pd.DataFrame, wave: int, lh: pd.DataFrame | None) -> pd.DataFrame:
    out = pd.DataFrame(index=core.index)
    pid = core["id"] if "id" in core.columns else core["ID"]
    out["person_id"] = pid.astype("string")
    hh = core["householdid"] if "householdid" in core.columns else core.get("householdID")
    out["household_id"] = hh.astype("string") if hh is not None else pd.Series(pd.NA, index=core.index, dtype="string")
    out["wave"] = wave

    inw = pick(core, wave, (f"inw{wave}",))
    iwstat = pick(core, wave, ("iwstat",))
    live = (inw.eq(1) if inw is not None else False) | (iwstat.eq(1) if iwstat is not None else False)
    if not isinstance(live, pd.Series):
        live = pd.Series(False, index=core.index)
    out["_live"] = live.fillna(False)
    out["interview_status"] = np.where(out["_live"], 1, np.nan)

    year = pick(core, wave, ("iwy",))
    month = pick(core, wave, ("iwm",))
    if year is None:
        year = pd.Series(np.nan, index=core.index)
    if month is None:
        month = pd.Series(np.nan, index=core.index)
    out["interview_year"] = year.fillna(WAVE_YEAR[wave])
    out["interview_month"] = month
    out["interview_date"] = pd.to_datetime(
        {"year": out["interview_year"], "month": month.fillna(7).clip(1, 12), "day": 15},
        errors="coerce",
    )

    sex = pick(core, wave, ("ragender", "gender"))
    out["sex"] = recode_map(sex, {1: 1.0, 2: 0.0}) if sex is not None else np.nan
    out["age_years"] = pick(core, wave, ("agey",))
    byear = pick(core, wave, ("rabyear",))
    bmonth = pick(core, wave, ("rabmonth",))
    if lh is not None and "person_id" in lh.columns:
        lh_map_y = lh.drop_duplicates("person_id").set_index("person_id")
        if byear is None and "birth_year" in lh_map_y.columns:
            byear = out["person_id"].map(lh_map_y["birth_year"])
        if bmonth is None and "birth_month" in lh_map_y.columns:
            bmonth = out["person_id"].map(lh_map_y["birth_month"])
    out["birth_year"] = byear
    out["birth_month"] = bmonth

    educl = pick(core, wave, ("raeducl", "educl"))
    educ = pick(core, wave, ("raeduc_c", "educ_c"))
    if educl is not None and educl.notna().any():
        out["education_level"] = educl.where(educl.isin([1, 2, 3]))
    elif educ is not None:
        out["education_level"] = recode_map(educ, EDUC_C_TO_LEVEL)
    else:
        out["education_level"] = np.nan
    out["education_years"] = recode_map(educ, EDUC_C_TO_YEARS) if educ is not None else np.nan

    out["marital_status"] = pick(core, wave, ("mstat",))
    hhsize = pick(core, wave, ("hhres",))
    out["household_size"] = hhsize
    out["living_alone"] = np.where(hhsize.isna(), np.nan, (hhsize == 1).astype(float)) if hhsize is not None else np.nan
    out["living_children"] = pick(core, wave, ("child",))
    out["living_siblings"] = pick(core, wave, ("livsib",))
    rural = pick_cols(core, (f"h{wave}rural", f"hh{wave}rural"))
    urban = pick(core, wave, ("urban",))
    residence = rural.where(rural.isin([0, 1])) if rural is not None else pd.Series(np.nan, index=core.index)
    if urban is not None:
        inverted = recode_map(urban, {0: 1.0, 1: 0.0})
        residence = residence.where(residence.notna(), inverted)
    out["urban_rural_residence"] = residence
    own = pick(core, wave, ("ahrto",))
    out["home_ownership"] = np.where(own.isna(), np.nan, (own == 1).astype(float)) if own is not None else np.nan
    out["nursing_home_residence"] = bin01(pick(core, wave, ("nhmliv",)))
    work = bin01(pick(core, wave, ("work",)))
    out["currently_working"] = work
    lbrf = pick(core, wave, ("lbrf_c",))
    out["labor_force_status"] = recode_map(lbrf, LBRF_TO_HRS) if lbrf is not None else np.nan
    if lbrf is not None:
        out["self_employed"] = np.where(lbrf.isna(), np.nan, (lbrf == 3).astype(float))
    else:
        out["self_employed"] = np.nan
    hours = pick(core, wave, ("jhourtot", "jhours_c"))
    out["hours_worked_per_week"] = hours.where(hours.between(0, 168)) if hours is not None else np.nan
    weeks = pick(core, wave, ("jweeks_c",))
    out["weeks_worked_per_year"] = weeks.clip(0, 52) if weeks is not None else np.nan
    out["health_limits_work"] = bin01(pick(core, wave, ("hlthlm_c", "hlthlm")))
    gov = bin01(pick(core, wave, ("higov",)))
    priv = bin01(pick(core, wave, ("hipriv",)))
    oth = bin01(pick(core, wave, ("hiothp", "hioth")))
    ins_parts = [s for s in (gov, priv, oth) if isinstance(s, pd.Series)]
    if ins_parts:
        imat = pd.concat(ins_parts, axis=1)
        out["health_insurance_any"] = imat.max(axis=1, skipna=True).where(imat.notna().any(axis=1))
    else:
        out["health_insurance_any"] = np.nan

    shlt = pick(core, wave, ("shlt", "shlta"))
    out["self_rated_health"] = shlt.where(shlt.between(1, 5)) if shlt is not None else np.nan
    for name, stem in DISEASE_STEMS.items():
        out[name] = bin01(pick(core, wave, (stem,)))
    for name, stem in ADL_ITEMS.items():
        out[name] = bin01(pick(core, wave, (stem,)))
    adl_total = pick(core, wave, ("adlab_c", "adla_c"))
    if adl_total is not None:
        out["adl_total_score"] = adl_total.where(adl_total.between(0, 6))
    else:
        parts = [out[n] for n in ADL_ITEMS if n in out.columns and isinstance(out[n], pd.Series)]
        out["adl_total_score"] = pd.concat(parts, axis=1).sum(axis=1, min_count=3) if parts else np.nan
    adl_mat_cols = [out[n] for n in ADL_ITEMS if isinstance(out.get(n), pd.Series)]
    if adl_mat_cols:
        amat = pd.concat(adl_mat_cols, axis=1)
        out["receives_adl_help"] = amat.eq(1).any(axis=1).astype(float).where(amat.notna().any(axis=1))
    else:
        out["receives_adl_help"] = np.nan
    out["urinary_incontinence"] = bin01(pick(core, wave, ("urina",)))
    for name, stem in IADL_ITEMS.items():
        out[name] = bin01(pick(core, wave, (stem,)))
    iadl_total = pick(core, wave, ("iadlza", "iadla", "iadl6_c"))
    iadl_parts = [out[n] for n in IADL_ITEMS if isinstance(out.get(n), pd.Series)]
    if iadl_total is not None:
        out["iadl_total_score"] = iadl_total.where(iadl_total.between(0, 8))
    elif iadl_parts:
        out["iadl_total_score"] = pd.concat(iadl_parts, axis=1).sum(axis=1, min_count=3)
    else:
        out["iadl_total_score"] = np.nan
    if iadl_parts:
        imat = pd.concat(iadl_parts, axis=1)
        out["receives_iadl_help"] = imat.eq(1).any(axis=1).astype(float).where(imat.notna().any(axis=1))
    else:
        out["receives_iadl_help"] = np.nan
    for name, stem in MOBILITY_ITEMS.items():
        out[name] = bin01(pick(core, wave, (stem,)))
    out["mobility_total_score"] = sum_named(out, HRS_MOBILITY_SCORE_ITEMS, min_count=2)
    out["fine_motor_total_score"] = sum_named(out, HRS_FINE_MOTOR_ITEMS, min_count=1)

    cesd_cols = []
    for hrs_name, stem, direction in CESD_ITEMS:
        raw = pick(core, wave, (stem,))
        mapping = CESD_POSITIVE_RAW_TO_BINARY if direction == "positive_raw" else CESD_NEGATIVE_TO_BINARY
        rec = recode_map(raw, mapping) if raw is not None else pd.Series(np.nan, index=core.index)
        out[hrs_name] = rec
        if direction == "positive_raw":
            cesd_cols.append((1.0 - rec).where(rec.notna()))
        else:
            cesd_cols.append(rec)
    out["cesd_score"] = pd.concat(cesd_cols, axis=1).sum(axis=1, min_count=6)

    imrc = pick(core, wave, ("imrc",))
    dlrc = pick(core, wave, ("dlrc",))
    ser7 = pick(core, wave, ("ser7",))
    orient = pick(core, wave, ("orient",))
    draw = bin01(pick(core, wave, ("draw",)))
    tr20 = pick(core, wave, ("tr20",))
    out["immediate_word_recall"] = imrc
    out["delayed_word_recall"] = dlrc
    out["serial_sevens"] = ser7
    out["total_word_recall"] = tr20
    if tr20 is None and imrc is not None and dlrc is not None:
        out["total_word_recall"] = imrc + dlrc
    cog_parts = [s for s in (imrc, dlrc, ser7) if s is not None]
    if len(cog_parts) == 3:
        raw25 = pd.concat(cog_parts, axis=1).sum(axis=1, min_count=3)
        out["cognition_27_score"] = raw25 * (27.0 / 25.0)
    else:
        out["cognition_27_score"] = np.nan
    ms_parts = [s for s in (ser7, orient, draw if isinstance(draw, pd.Series) else None) if s is not None]
    out["mental_status_score"] = pd.concat(ms_parts, axis=1).sum(axis=1, min_count=2) if ms_parts else np.nan
    tot_parts = [
        s
        for s in (
            out.get("total_word_recall"),
            ser7,
            orient,
            draw if isinstance(draw, pd.Series) else None,
        )
        if isinstance(s, pd.Series)
    ]
    out["total_cognition_score"] = pd.concat(tot_parts, axis=1).sum(axis=1, min_count=2) if tot_parts else np.nan
    out["self_rated_memory"] = pick(core, wave, ("slfmem",))
    sat = pick(core, wave, ("satlife",))
    out["life_satisfaction"] = (1.0 + (sat - 1.0) * 6.0 / 4.0).where(sat.between(1, 5)) if sat is not None else np.nan

    mom = pick(core, wave, ("ramomeducl", "momeducl"))
    dad = pick(core, wave, ("radadeducl", "dadeducl"))
    out["mother_education"] = mom.where(mom.isin([1, 2, 3])) if mom is not None else np.nan
    out["father_education"] = dad.where(dad.isin([1, 2, 3])) if dad is not None else np.nan
    mom_cred = pick(core, wave, ("rameduc_c", "meduc_c"))
    dad_cred = pick(core, wave, ("rafeduc_c", "feduc_c"))
    parent_years = []
    if mom_cred is not None:
        parent_years.append(recode_map(mom_cred, EDUC_C_TO_YEARS).clip(0, 17))
    if dad_cred is not None:
        parent_years.append(recode_map(dad_cred, EDUC_C_TO_YEARS).clip(0, 17))
    out["childhood_ses"] = (
        pd.concat(parent_years, axis=1).mean(axis=1, skipna=True) if parent_years else np.nan
    )
    ch = pick(core, wave, ("rahltcom", "rachchlt"))
    if ch is None and lh is not None and "childhood_health_lh" in lh.columns:
        lh_map = lh.drop_duplicates("person_id").set_index("person_id")
        ch = out["person_id"].map(lh_map["childhood_health_lh"])
        ch = to_num(ch)
    childhood = ch.where(ch.between(1, 5)) if ch is not None else pd.Series(np.nan, index=core.index)
    misch = bin01(pick(core, wave, ("ramischlth", "mischlth")))
    bed = bin01(pick(core, wave, ("rachbedhlth", "chbedhlth")))
    flag_parts = [s for s in (misch, bed) if isinstance(s, pd.Series)]
    if flag_parts:
        mat = pd.concat(flag_parts, axis=1)
        observed = mat.notna().any(axis=1)
        any_ill = mat.fillna(0).eq(1).any(axis=1)
        childhood = childhood.mask(childhood.isna() & observed & any_ill, 4.0)
        childhood = childhood.mask(childhood.isna() & observed & ~any_ill, 2.0)
    out["childhood_health"] = childhood
    kcnt = bin01(pick(core, wave, ("kcntf",)))
    out["social_contact_children"] = kcnt

    out["systolic_bp"] = mean_loop(core, wave, ("systo1", "systo2", "systo3", "systo"), 50, 300)
    out["diastolic_bp"] = mean_loop(core, wave, ("diasto1", "diasto2", "diasto3", "diasto"), 30, 200)
    out["resting_pulse"] = mean_loop(core, wave, ("pulse1", "pulse2", "pulse3", "pulse"), 20, 250)
    puff = pick(core, wave, ("puff1", "puff"))
    out["peak_expiratory_flow"] = puff.where(puff.between(0, 1200)) if puff is not None else np.nan
    left = pick(core, wave, ("lgrip",))
    right = pick(core, wave, ("rgrip",))
    grip = pick(core, wave, ("gripsum",))
    out["left_grip_strength"] = left.where(left.between(0, 120)) if left is not None else np.nan
    out["right_grip_strength"] = right.where(right.between(0, 120)) if right is not None else np.nan
    if grip is not None:
        out["grip_strength"] = grip.where(grip.between(0, 120))
    else:
        hands = [s for s in (out["left_grip_strength"], out["right_grip_strength"]) if isinstance(s, pd.Series)]
        out["grip_strength"] = pd.concat(hands, axis=1).max(axis=1, skipna=True) if hands else np.nan
    w1 = pick(core, wave, ("wspeed1",))
    w2 = pick(core, wave, ("wspeed2",))
    speeds = [s.where(s.between(0.5, 180)) for s in (w1, w2) if s is not None]
    out["walking_speed_time"] = pd.concat(speeds, axis=1).mean(axis=1, skipna=True) if speeds else np.nan
    ht = pick(core, wave, ("mheight",))
    wt = pick(core, wave, ("mweight",))
    out["measured_height"] = ht.where(ht.between(1.0, 2.5)) if ht is not None else np.nan
    out["measured_weight"] = wt.where(wt.between(20, 300)) if wt is not None else np.nan
    bmi = pick(core, wave, ("mbmi",))
    if bmi is None or not isinstance(bmi, pd.Series) or not bmi.notna().any():
        if isinstance(out["measured_height"], pd.Series) and isinstance(out["measured_weight"], pd.Series):
            bmi = out["measured_weight"] / (out["measured_height"] ** 2)
    out["measured_bmi"] = bmi.where(bmi.between(10, 80)) if isinstance(bmi, pd.Series) else np.nan
    waist = pick(core, wave, ("mwaist",))
    out["waist_circumference"] = waist.where(waist.between(30, 220)) if waist is not None else np.nan
    bal = pick(core, wave, ("balance",))
    out["balance_score"] = bal.where(bal.between(1, 4)) if bal is not None else np.nan

    hosp = bin01(pick(core, wave, ("hosp1y",)))
    stays = pick(core, wave, ("hsptim1y",))
    out["hospital_stays_count"] = stays
    if hosp is not None:
        out["hospitalization"] = hosp
    elif stays is not None:
        out["hospitalization"] = np.where(stays.isna(), np.nan, (stays > 0).astype(float))
    else:
        out["hospitalization"] = np.nan
    out["hospital_nights"] = pick(core, wave, ("hspnite",))
    doc_any = bin01(pick(core, wave, ("doctor1m",)))
    doc_n = pick(core, wave, ("doctim1m",))
    out["doctor_visits_count"] = doc_n
    if doc_any is not None:
        out["doctor_visits_any"] = doc_any
    elif doc_n is not None:
        out["doctor_visits_any"] = np.where(doc_n.isna(), np.nan, (doc_n > 0).astype(float))
    else:
        out["doctor_visits_any"] = np.nan
    dent = bin01(pick(core, wave, ("dentst1y",)))
    dent_n = pick(core, wave, ("dentim1y",))
    if dent is not None:
        out["dental_visit"] = dent
    elif dent_n is not None:
        out["dental_visit"] = np.where(dent_n.isna(), np.nan, (dent_n > 0).astype(float))
    else:
        out["dental_visit"] = np.nan
    out["home_health_care"] = bin01(pick(core, wave, ("rfaany",)))
    oop_parts = [pick(core, wave, (st,)) for st in ("oophos1y", "oopdoc1m", "oopden1y")]
    oop_cols = [p for p in oop_parts if p is not None]
    if oop_cols:
        omat = pd.concat(oop_cols, axis=1)
        out["out_of_pocket_medical_cost"] = omat.fillna(0).sum(axis=1).where(omat.notna().any(axis=1))
    else:
        out["out_of_pocket_medical_cost"] = np.nan
    out["total_household_income"] = pick(core, wave, ("itot",))
    out["total_household_wealth"] = pick(core, wave, ("atotb",))
    wgt = pick(core, wave, ("wtrespb", "wtresp", "wtrespa"))
    out["respondent_weight"] = wgt.where(wgt > 0) if wgt is not None else np.nan

    ever = bin01(pick(core, wave, ("smokev",)))
    current = bin01(pick(core, wave, ("smoken",)))
    out["_ever_smoked_wave"] = ever
    out["_current_smoke"] = current
    out["cigarettes_per_day"] = pick(core, wave, ("smokef",))
    drink = bin01(pick(core, wave, ("drinkl", "drink")))
    out["alcohol_use"] = drink
    drinkn = pick(core, wave, ("drinkn_c",))
    out["alcohol_days_per_week"] = recode_map(drinkn, DRINKN_TO_DAYS) if drinkn is not None else np.nan
    drinkr = pick(core, wave, ("drinkr_c",))
    out["drinks_per_drinking_day"] = recode_map(drinkr, DRINKR_TO_DRINKS) if drinkr is not None else np.nan
    if isinstance(out["alcohol_use"], pd.Series):
        out.loc[out["alcohol_use"].eq(0), "alcohol_days_per_week"] = 0.0
        out.loc[out["alcohol_use"].eq(0), "drinks_per_drinking_day"] = np.nan

    for hrs_name, flag, days in (
        ("vigorous_activity_frequency", "vgact_c", "vgactx_c"),
        ("moderate_activity_frequency", "mdact_c", "mdactx_c"),
        ("light_activity_frequency", "ltact_c", "ltactx_c"),
    ):
        ind = bin01(pick(core, wave, (flag,)))
        day = pick(core, wave, (days,))
        mod = pd.Series(np.nan, index=core.index)
        if isinstance(ind, pd.Series):
            mod = mod.mask(ind.eq(0), 0.0)
            if day is not None:
                mapped = day.map(lambda x: exercise_days_to_hrs_frequency(float(x)) if pd.notna(x) else np.nan)
                mod = mod.mask(ind.eq(1), mapped)
        elif day is not None:
            mod = day.map(lambda x: exercise_days_to_hrs_frequency(float(x)) if pd.notna(x) else np.nan)
        out[hrs_name] = mod
    out["hypertension_treatment"] = bin01(pick(core, wave, ("rxhibp_c", "rxhibp")))
    out["diabetes_oral_medication"] = bin01(pick(core, wave, ("rxdiab_c", "rxdiab")))

    keep = out["_live"].fillna(False)
    result = out.loc[keep].drop(columns=["_live"]).reset_index(drop=True)
    LOGGER.info("Wave %s extracted n=%s pids=%s", wave, len(result), result["person_id"].nunique())
    return result


def apply_observed_masks(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    masks = {}
    for name in names:
        if name not in frame.columns:
            continue
        masks[f"{name}__observed"] = pd.to_numeric(frame[name], errors="coerce").notna().astype("int8")
    if not masks:
        return frame
    drop = [c for c in masks if c in frame.columns]
    base = frame.drop(columns=drop) if drop else frame
    return pd.concat([base, pd.DataFrame(masks, index=frame.index)], axis=1)


def absorb_binary(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    out = frame.sort_values(["person_id", "wave"]).copy()
    for name in names:
        if name not in out.columns:
            continue
        values = pd.to_numeric(out[name], errors="coerce")
        ever = values.eq(1).groupby(out["person_id"], sort=False).cummax()
        out[name] = np.where(ever, 1.0, values)
    return out


def finalize_smoking(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["person_id", "wave"]).copy()
    ever = pd.to_numeric(out.get("_ever_smoked_wave"), errors="coerce")
    current = pd.to_numeric(out.get("_current_smoke"), errors="coerce")
    ever = ever.mask(current.eq(1), 1.0)
    ever = ever.groupby(out["person_id"], sort=False).ffill()
    status = pd.Series(np.nan, index=out.index)
    status = status.mask(ever.eq(0) & current.fillna(0).eq(0), 0.0)
    status = status.mask(ever.eq(0) & current.isna(), 0.0)
    status = status.mask(current.eq(1), 2.0)
    status = status.mask(ever.eq(1) & current.eq(0), 1.0)
    status = status.mask(ever.eq(1) & current.isna(), 1.0)
    out["smoking_status"] = status
    cig = pd.to_numeric(out.get("cigarettes_per_day"), errors="coerce")
    eligible = current.eq(1)
    out["cigarettes_per_day"] = cig.where(eligible)
    out["cigarettes_per_day__eligible"] = eligible.fillna(False).astype("int8")
    out["cigarettes_per_day__observed"] = cig.where(eligible).notna().astype("int8")
    drink = pd.to_numeric(out.get("alcohol_use"), errors="coerce")
    for name in ("alcohol_days_per_week", "drinks_per_drinking_day"):
        if name not in out.columns:
            continue
        values = pd.to_numeric(out[name], errors="coerce")
        if name == "alcohol_days_per_week":
            values = values.mask(drink.eq(0), 0.0)
            out[f"{name}__eligible"] = drink.notna().astype("int8")
        else:
            values = values.where(drink.eq(1))
            out[f"{name}__eligible"] = drink.eq(1).fillna(False).astype("int8")
        out[name] = values
        out[f"{name}__observed"] = values.notna().astype("int8")
    for name, dx in (
        ("hypertension_treatment", "hypertension_dx"),
        ("diabetes_oral_medication", "diabetes_dx"),
    ):
        if name not in out.columns:
            continue
        diagnosed = pd.to_numeric(out.get(dx), errors="coerce").eq(1)
        values = pd.to_numeric(out[name], errors="coerce").where(diagnosed)
        out[name] = values
        out[f"{name}__eligible"] = diagnosed.astype("int8")
        out[f"{name}__observed"] = values.notna().astype("int8")
    out["multimorbidity_count"] = out[
        [c for c in ABSORBING_DISEASES if c != "memory_disease_dx" and c in out.columns]
    ].sum(axis=1, min_count=1)
    return out


def load_core(path: Path) -> pd.DataFrame:
    LOGGER.info("Reading %s", path)
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "id" not in frame.columns and "id_w1" in frame.columns:
        frame["id"] = frame["id_w1"]
    frame["id"] = normalize_identifier(frame["id"], 12)
    if "householdid" not in frame.columns and "householdid_w1" in frame.columns:
        frame["householdid"] = frame["householdid_w1"]
    if "householdid" in frame.columns:
        frame["householdid"] = normalize_identifier(frame["householdid"], 10)
    LOGGER.info("CHARLS D n=%s ncols=%s", f"{len(frame):,}", len(frame.columns))
    return frame


def load_eol(path: Path, waves: Sequence[int]) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Missing EOL file %s", path)
        return pd.DataFrame(columns=["person_id", "wave", "interview_status", "death_year", "death_month"])
    eol = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    eol.columns = [str(c).strip().lower() for c in eol.columns]
    pid = normalize_identifier(eol["id"] if "id" in eol.columns else eol["id_w1"], 12)
    death_wave = pd.to_numeric(eol.get("raxt"), errors="coerce")
    recs = []
    wave_set = set(int(w) for w in waves)
    for i in range(len(eol)):
        wave = death_wave.iloc[i]
        if pd.isna(wave):
            for w in waves:
                flag = eol.get(f"inw{w}xt")
                if flag is not None and pd.to_numeric(flag.iloc[i], errors="coerce") == 1:
                    wave = w
                    break
        if pd.isna(wave) or int(wave) not in wave_set:
            continue
        year = pd.to_numeric(eol.get("radyear", pd.Series(np.nan, index=eol.index)).iloc[i], errors="coerce")
        if pd.isna(year):
            year = pd.to_numeric(eol.get("raxtiwy", pd.Series(np.nan, index=eol.index)).iloc[i], errors="coerce")
        month = pd.to_numeric(eol.get("raxtiwm", pd.Series(np.nan, index=eol.index)).iloc[i], errors="coerce")
        recs.append(
            {
                "person_id": str(pid.iloc[i]),
                "wave": int(wave),
                "interview_status": 5,
                "death_year": year,
                "death_month": month,
            }
        )
    out = (
        pd.DataFrame(recs).drop_duplicates(["person_id", "wave"])
        if recs
        else pd.DataFrame(columns=["person_id", "wave", "interview_status", "death_year", "death_month"])
    )
    LOGGER.info("EOL death rows %s", len(out))
    return out


def load_lh(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["person_id"])
    lh = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    lh.columns = [str(c).strip().lower() for c in lh.columns]
    pid = lh["id"] if "id" in lh.columns else lh.get("id_w1")
    out = pd.DataFrame({"person_id": normalize_identifier(pid, 12)})
    if "rabyear" in lh.columns:
        out["birth_year"] = to_num(lh["rabyear"])
    if "rabmonth" in lh.columns:
        out["birth_month"] = to_num(lh["rabmonth"])
    if "rachchlt" in lh.columns:
        out["childhood_health_lh"] = to_num(lh["rachchlt"])
    LOGGER.info("Life History n=%s", len(out))
    return out.drop_duplicates("person_id")


def specs_from_dictionary(path: Path | str) -> dict[str, hrs.VariableSpec]:
    text = str(path).strip()
    workbook = Path(text) if text and text != "." else None
    if workbook is not None and workbook.is_file():
        try:
            _df, specs = hrs.read_dictionary(workbook)
            return specs
        except Exception as exc:
            LOGGER.warning("CHARLS dictionary reader failed (%s); using built-in specs.", exc)
    return hrs.builtin_specs()


def write_long_and_audit(long_raw: pd.DataFrame, specs: dict[str, hrs.VariableSpec], output_dir: Path) -> None:
    long_raw.to_csv(output_dir / "charls_world_model_long_raw.csv", index=False, encoding="utf-8-sig")
    rows = []
    for name, spec in specs.items():
        info = VAR_COVERAGE.get(name, {})
        series = pd.to_numeric(long_raw[name], errors="coerce") if name in long_raw.columns else pd.Series(dtype=float)
        rows.append(
            {
                "variable": name,
                "role": spec.role,
                "priority": spec.priority,
                "charls_coverage": info.get("coverage", "unavailable"),
                "compatibility": info.get("compatibility", ""),
                "n_nonmissing": int(series.notna().sum()) if len(series) else 0,
                "missing_rate": float(series.isna().mean()) if len(series) else 1.0,
                "n_person_wave": int(len(long_raw)),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "source_audit.csv", index=False, encoding="utf-8-sig")
    miss = []
    for wave, grp in long_raw.groupby("wave"):
        for name, spec in specs.items():
            if spec.role not in {"Static state variable", "Dynamic state variable", "Action variable"}:
                continue
            if name not in grp.columns:
                continue
            series = pd.to_numeric(grp[name], errors="coerce")
            miss.append(
                {
                    "wave": int(wave),
                    "variable": name,
                    "n": int(len(grp)),
                    "n_observed": int(series.notna().sum()),
                    "missing_rate": float(series.isna().mean()),
                }
            )
    pd.DataFrame(miss).to_csv(output_dir / "missingness_summary.csv", index=False, encoding="utf-8-sig")
    LOGGER.info("Wrote long_raw (%s rows) and audit CSVs", f"{len(long_raw):,}")


def apply_hrs_preprocessing(
    transitions: pd.DataFrame,
    preprocess_path: Path,
    config_path: Path,
    output_dir: Path,
    csv_copy: bool,
) -> None:
    preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    continuous_stats = preprocess.get("continuous") or preprocess.get("continuous_stats") or {}
    reward_stats = preprocess.get("reward_continuous") or preprocess.get("reward_continuous_stats") or {}
    state_names = list(config.get("state_columns_model", []))
    static_names = list(config.get("static_context_columns", []))
    action_names = list(config.get("action_columns_model", []))
    reward_names = list(config.get("reward_columns", []))
    frame = hrs.normalize_transition_column_names(transitions).copy()
    frame["split"] = "external"
    frame["transition_index"] = frame.groupby("person_id", sort=False).cumcount().astype(int)
    if "next_wave" not in frame.columns:
        frame["next_wave"] = np.nan

    def obs_mask(split_frame: pd.DataFrame, mask_col: str, values: pd.Series) -> pd.Series:
        if mask_col in split_frame.columns:
            return pd.to_numeric(split_frame[mask_col], errors="coerce").fillna(0).astype("int8")
        return values.notna().astype("int8")

    blocks: dict[str, Any] = {
        "person_id": frame["person_id"].astype("string").to_numpy(),
        "transition_index": frame["transition_index"].to_numpy(),
        "wave": frame["wave"].to_numpy(),
        "next_wave": frame["next_wave"].to_numpy(),
        "delta_t_years": frame["delta_t_years"].to_numpy(),
        "continuation": frame["continuation"].to_numpy(),
        "split": frame["split"].to_numpy(),
    }
    for name in state_names + static_names:
        current = pd.to_numeric(frame.get(f"{hrs.STATE_PREFIX}{name}"), errors="coerce")
        if current is None or isinstance(current, float):
            current = pd.Series(np.nan, index=frame.index)
        future = pd.to_numeric(frame.get(f"{hrs.NEXT_STATE_PREFIX}{name}"), errors="coerce")
        if future is None or isinstance(future, float):
            future = pd.Series(np.nan, index=frame.index)
        current_mask = obs_mask(frame, f"{hrs.STATE_MASK_PREFIX}{name}", current)
        future_mask = obs_mask(frame, f"{hrs.NEXT_STATE_MASK_PREFIX}{name}", future)
        blocks[f"{hrs.STATE_MASK_PREFIX}{name}"] = current_mask.to_numpy()
        blocks[f"{hrs.NEXT_STATE_MASK_PREFIX}{name}"] = future_mask.to_numpy()
        stats = continuous_stats.get(name)
        if stats:
            med, mean, std = float(stats["median"]), float(stats["mean"]), float(stats["std"] or 1.0)
            if std < 1e-8:
                std = 1.0
            blocks[f"{hrs.STATE_PREFIX}{name}"] = ((current.fillna(med) - mean) / std).astype("float32").to_numpy()
            blocks[f"{hrs.NEXT_STATE_PREFIX}{name}"] = ((future.fillna(med) - mean) / std).astype("float32").to_numpy()
        else:
            blocks[f"{hrs.STATE_PREFIX}{name}"] = current.fillna(-1).astype("float32").to_numpy()
            blocks[f"{hrs.NEXT_STATE_PREFIX}{name}"] = future.fillna(-1).astype("float32").to_numpy()
    for name in action_names:
        value = pd.to_numeric(frame.get(f"{hrs.ACTION_PREFIX}{name}"), errors="coerce")
        if value is None or isinstance(value, float):
            value = pd.Series(np.nan, index=frame.index)
        obs = obs_mask(frame, f"{hrs.ACTION_MASK_PREFIX}{name}", value)
        eligible_col = f"{hrs.ACTION_ELIGIBLE_PREFIX}{name}"
        eligible = (
            pd.to_numeric(frame[eligible_col], errors="coerce").fillna(0).astype("int8")
            if eligible_col in frame.columns
            else pd.Series(1, index=frame.index, dtype="int8")
        )
        action_mask = (obs.astype("int8") * eligible.astype("int8")).astype("int8")
        blocks[f"{hrs.ACTION_MASK_PREFIX}{name}"] = action_mask.to_numpy()
        stats = continuous_stats.get(name)
        if stats:
            med, mean, std = float(stats["median"]), float(stats["mean"]), float(stats["std"] or 1.0)
            if std < 1e-8:
                std = 1.0
            filled = value.where(action_mask.eq(1), np.nan).fillna(med)
            blocks[f"{hrs.ACTION_PREFIX}{name}"] = ((filled - mean) / std).astype("float32").to_numpy()
        else:
            blocks[f"{hrs.ACTION_PREFIX}{name}"] = value.where(action_mask.eq(1), np.nan).fillna(-1).astype("float32").to_numpy()
    for name in reward_names:
        value = pd.to_numeric(frame.get(f"{hrs.REWARD_PREFIX}{name}"), errors="coerce")
        if value is None or isinstance(value, float):
            value = pd.Series(np.nan, index=frame.index)
        mask = obs_mask(frame, f"{hrs.REWARD_MASK_PREFIX}{name}", value)
        blocks[f"{hrs.REWARD_MASK_PREFIX}{name}"] = mask.to_numpy()
        stats = reward_stats.get(name)
        if stats:
            transformed = hrs._transform_reward_values(name, value)
            med, mean, std = float(stats["median"]), float(stats["mean"]), float(stats["std"] or 1.0)
            if std < 1e-8:
                std = 1.0
            blocks[f"{hrs.REWARD_PREFIX}{name}"] = ((transformed.fillna(med) - mean) / std).astype("float32").to_numpy()
        else:
            blocks[f"{hrs.REWARD_PREFIX}{name}"] = value.fillna(0).astype("float32").to_numpy()
    out = pd.DataFrame(blocks)
    parquet_path = output_dir / "CHARLS_external.parquet"
    out.to_parquet(parquet_path, index=False)
    if csv_copy:
        out.to_csv(output_dir / "CHARLS_external.csv", index=False, encoding="utf-8-sig")
    meta = {
        "role": "hrs_external_validation",
        "n_transitions": int(len(out)),
        "n_persons": int(out["person_id"].nunique()),
        "hrs_preprocessing": str(preprocess_path),
        "hrs_model_config": str(config_path),
        "note": "Values are standardized with HRS training-set mean/std/median. Masks flag CHARLS observation.",
    }
    (output_dir / "CHARLS_external_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info("Wrote external bundle %s (%s rows)", parquet_path, f"{len(out):,}")


def run_self_test() -> int:
    assert recode_map(pd.Series([1, 2, -99]), {1: 1.0, 2: 0.0}).tolist()[0] == 1.0
    assert recode_map(pd.Series([1, 4, -99]), CESD_NEGATIVE_TO_BINARY).tolist()[1] == 1.0
    happy = recode_map(pd.Series([1, 4]), CESD_POSITIVE_RAW_TO_BINARY)
    assert happy.tolist() == [0.0, 1.0]
    assert coverage_of("mobility_total_score") == "partial"
    assert exercise_days_to_hrs_frequency(7) == 4.0
    assert coverage_of("sex") == "available"
    assert coverage_of("medicare_coverage") == "us_only"
    assert coverage_of("weeks_worked_per_year") == "available"
    assert coverage_of("fine_motor_total_score") == "derived"
    assert coverage_of("childhood_ses") == "proxy"
    assert coverage_of("balance_score") == "available"
    assert coverage_of("social_contact_friends") == "unavailable"
    if hrs.BUILTIN_SPECS_PATH.is_file():
        specs = hrs.builtin_specs()
        assert "hypertension_dx" in specs
        assert specs["hypertension_dx"].role == "Dynamic state variable"
    LOGGER.info("CHARLS HRS self-test passed")
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.self_test:
        return run_self_test()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    waves = sorted(set(args.waves))

    hrs.WAVE_YEAR.clear()
    hrs.WAVE_YEAR.update(WAVE_YEAR)
    hrs.CPI_U.clear()
    hrs.CPI_U.update(CHINA_CPI)

    specs = specs_from_dictionary(Path(args.dictionary))
    core = load_core(data_dir / CORE_FILE)
    lh = load_lh(data_dir / LH_FILE)
    live_parts = [extract_wave(core, wave, lh) for wave in waves]
    long_live = pd.concat(live_parts, ignore_index=True, sort=False)
    long_live = absorb_binary(long_live, ABSORBING_DISEASES)
    long_live = finalize_smoking(long_live)

    state_action_names = [
        name
        for name, spec in specs.items()
        if spec.role in {"Static state variable", "Dynamic state variable", "Action variable"}
        and name not in hrs.EXCLUDE_STATE_VARIABLES
    ]
    extra = [
        "cesd_score",
        "adl_total_score",
        "iadl_total_score",
        "multimorbidity_count",
        "hospitalization",
        "smoking_status",
        "cognition_27_score",
        "living_alone",
        "out_of_pocket_medical_cost",
        "moderate_activity_frequency",
        "vigorous_activity_frequency",
        "light_activity_frequency",
    ]
    for name in extra:
        if name not in state_action_names and name in long_live.columns:
            state_action_names.append(name)
    long_live = apply_observed_masks(long_live, state_action_names)

    exits = load_eol(data_dir / EOL_FILE, waves)
    death_extra = []
    radyear = to_num(core["radyear"]) if "radyear" in core.columns else None
    radmonth = to_num(core["radmonth"]) if "radmonth" in core.columns else None
    for wave in waves:
        iwstat = pick(core, wave, ("iwstat",))
        inw = pick(core, wave, (f"inw{wave}",))
        inwxt = pick(core, wave, (f"inw{wave}xt",))
        dead = pd.Series(False, index=core.index)
        if iwstat is not None:
            dead = dead | iwstat.eq(5)
        if inwxt is not None:
            dead = dead | inwxt.eq(1)
        if inw is not None:
            dead = dead & ~inw.eq(1)
        mask = dead.fillna(False)
        if not mask.any():
            continue
        rec = pd.DataFrame(
            {
                "person_id": core.loc[mask, "id"].astype(str).to_numpy(),
                "wave": int(wave),
                "interview_status": 5,
                "death_year": radyear.loc[mask].to_numpy() if radyear is not None else np.nan,
                "death_month": radmonth.loc[mask].to_numpy() if radmonth is not None else np.nan,
            }
        )
        death_extra.append(rec)
    if death_extra:
        exits = pd.concat([exits, *death_extra], ignore_index=True).drop_duplicates(["person_id", "wave"])

    persons = pd.Index(sorted(set(long_live["person_id"]).union(exits["person_id"])))
    grid = pd.MultiIndex.from_product([persons, waves], names=["person_id", "wave"]).to_frame(index=False)
    live_status = long_live[["person_id", "wave", "interview_status"]].drop_duplicates()
    status_grid = grid.merge(live_status, on=["person_id", "wave"], how="left")
    status_grid = status_grid.merge(
        exits[["person_id", "wave", "interview_status"]].rename(columns={"interview_status": "exit_status"}),
        on=["person_id", "wave"],
        how="left",
    )
    status_grid["interview_status"] = status_grid["interview_status"].fillna(status_grid["exit_status"])
    status_grid = status_grid.drop(columns=["exit_status"])

    long_raw = long_live.copy()
    fill_names = [
        n
        for n in state_action_names
        if n in long_live.columns
        and coverage_of(n) not in {"unavailable", "us_only"}
        and n not in SKIP_NEARBY_FILL
    ]
    long_filled, fill_counts = hrs.fill_missing_from_nearby_waves(long_live, fill_names)
    LOGGER.info(
        "Nearby-wave fills: %s variables, %s cells",
        sum(v > 0 for v in fill_counts.values()),
        sum(fill_counts.values()),
    )
    write_long_and_audit(long_raw, specs, output_dir)

    transitions = hrs.build_transitions(
        long_live=long_filled,
        status_grid=status_grid,
        specs=specs,
        waves=waves,
        max_transition_years=args.max_transition_years,
    )
    transitions = transitions.copy()
    transitions["split"] = "external"
    trans_path = output_dir / "charls_world_model_transitions.parquet"
    transitions.to_parquet(trans_path, index=False)
    if args.csv_copy:
        transitions.to_csv(output_dir / "charls_world_model_transitions.csv", index=False, encoding="utf-8-sig")
    LOGGER.info(
        "Wrote %s (%s rows, %s persons)",
        trans_path,
        f"{len(transitions):,}",
        f"{transitions['person_id'].nunique():,}",
    )

    meta = {
        "dataset": "CHARLS",
        "role": "hrs_external_validation",
        "waves": waves,
        "wave_year": WAVE_YEAR,
        "n_person_wave_live": int(len(long_raw)),
        "n_transitions": int(len(transitions)),
        "n_persons": int(transitions["person_id"].nunique()),
        "nearby_wave_fill_cells": {k: int(v) for k, v in fill_counts.items() if v},
        "currency": "CNY",
        "oop_cpi": "China CPI 2010=100, deflated to 2018",
    }
    (output_dir / "feature_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not args.no_hrs_scaler:
        pre = Path(args.hrs_preprocessing)
        cfg = Path(args.hrs_model_config)
        if pre.exists() and cfg.exists():
            apply_hrs_preprocessing(transitions, pre, cfg, output_dir, args.csv_copy)
        else:
            LOGGER.warning("HRS preprocessing/config not found (%s, %s); skip external bundle.", pre, cfg)

    print()
    print("CHARLS HRS external-validation cleaning completed successfully.")
    print(f"Output folder: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
