#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Clean Harmonized ELSA G3 + Wave 10 into HRS-compatible tables."""
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

from elsa_hrs_mappings import (  # noqa: E402
    ABSORBING_DISEASES,
    ACT4_TO_HRS,
    ACT5_TO_HRS,
    ADL_ITEMS,
    ADL_ITEMS_W10,
    CESD_ITEMS_G3,
    CESD_ITEMS_W10,
    DIMARR_TO_HRS,
    DISEASE_STEMS,
    HRS_FINE_MOTOR_ITEMS,
    HRS_LARGE_MUSCLE_ITEMS,
    HRS_MOBILITY_SCORE_ITEMS,
    IADL_ITEMS,
    IADL_ITEMS_W10,
    IFS_MARSTAT_TO_HRS,
    LBRF_TO_HRS,
    MISSING_SENTINELS,
    MOBILITY_ITEMS,
    MONEY_GBP_COLUMNS,
    SERIAL7_TARGETS,
    UK_CPI,
    VAR_COVERAGE,
    WAVE_YEAR,
    YESNO12,
    annualise_w10_weekly_income,
    coverage_of,
    recode_w10_hehavedi,
    usd_per_gbp,
)

LOGGER = logging.getLogger("elsa_hrs_cleaning")
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "data" / "elsa"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parents[1] / "data" / "elsa"
DEFAULT_WAVES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DEFAULT_HRS_PREPROCESS = SCRIPT_DIR.parents[1] / "data" / "hrs" / "HRS_preprocessing.json"
DEFAULT_HRS_CONFIG = SCRIPT_DIR.parents[1] / "data" / "hrs" / "HRS_model_config.json"
G3_FILE = "h_elsa_g3.tab"
W10_CORE_FILE = "wave_10_elsa_data_eul_v4.tab"
W10_IFS_FILE = "wave_10_ifs_derived_variables.tab"
W10_FIN_FILE = "wave_10_financial_derived_variables.tab"
EOL_FILE = "h_elsa_eol_a2.tab"
SKIP_NEARBY_FILL: set[str] = set()
W10_LIVE_INDOUT = {11.0, 13.0}

G3_STATIC = [
    "idauniq",
    "pn",
    "perid",
    "ragender",
    "raeducl",
    "raedyrs_e",
    "rabyear",
    "rabmonth",
    "rabplace",
    "rachshlt",
    "radyear",
    "radmonth",
    "raracem",
]
G3_R_STEMS = [
    "iwstat",
    "iwindy",
    "iwindm",
    "agey",
    "mstat",
    "hownrnt",
    "nhmliv",
    "work",
    "lbrf_e",
    "jhours",
    "hlthlm",
    "hipriv",
    "shlt",
    "shlta",
    "hibpe",
    "diabe",
    "cancre",
    "lunge",
    "hearte",
    "stroke",
    "psyche",
    "arthre",
    "memrye",
    "walkra",
    "dressa",
    "batha",
    "eata",
    "beda",
    "toilta",
    "phonea",
    "moneya",
    "medsa",
    "shopa",
    "mealsa",
    "adla",
    "iadla",
    "iadlza",
    "walk100a",
    "sita",
    "chaira",
    "climsa",
    "clim1a",
    "stoopa",
    "lifta",
    "dimea",
    "armsa",
    "pusha",
    "mobilsev",
    "lgmusa",
    "finea",
    "cesd",
    "depres",
    "effort",
    "sleepr",
    "whappy",
    "flone",
    "enlife",
    "fsad",
    "going",
    "imrc",
    "dlrc",
    "ser7",
    "tr20",
    "orient",
    "draw",
    "slfmem",
    "satlife_e",
    "satlife",
    "sight",
    "nsight",
    "dsight",
    "hearing",
    "painfr",
    "painlv",
    "fall",
    "fallnum",
    "fallinj",
    "urinai",
    "kcntf",
    "cwtresp",
    "smokev",
    "smoken",
    "smokef",
    "drink",
    "drinkd_e",
    "drinkn_e",
    "vgactx_e",
    "mdactx_e",
    "ltactx_e",
    "rxhibp",
    "rxdiab",
    "rfaany_e",
    "child",
    "livsib",
    "systo1",
    "systo2",
    "systo3",
    "systo",
    "diasto1",
    "diasto2",
    "diasto3",
    "diasto",
    "pulse1",
    "pulse2",
    "pulse3",
    "pulse",
    "puff",
    "puff1",
    "lgrip",
    "rgrip",
    "gripsum",
    "wspeed1",
    "wspeed2",
    "mheight",
    "mweight",
    "mbmi",
    "mwaist",
    "bwc20",
    "balance_e",
    "breath_e",
    "jweeks_e",
    "unfriend",
]
G3_H_STEMS = ["hhres", "child", "itot", "atotb"]
G3_HH_STEMS = ["hhid"]
W10_CORE_WANTED = [
    "idauniq",
    "idahhw10",
    "w10indout",
    "iintdaty",
    "iintdatm",
    "indager",
    "dhsex",
    "dimar",
    "dimarr",
    "hhtot",
    "hehelf",
    "hepain",
    "heeye",
    "hehear",
    "hediz",
    "headlwa",
    "headldr",
    "headlba",
    "headlea",
    "headlbe",
    "headlwc",
    "headlph",
    "headlmo",
    "headlme",
    "headlsh",
    "headlpr",
    "psceda",
    "pscedb",
    "pscedc",
    "pscedd",
    "pscede",
    "pscedf",
    "pscedg",
    "pscedh",
    "heeverbp",
    "heeverst",
    "heeverca",
    "heeverps",
    "heeverar",
    "heeverad",
    "heeverdm",
    "heevercl",
    "heeveras",
    "heeveran",
    "heevermi",
    "heeverhf",
    "heeverah",
    "hehavedi",
    "heacta",
    "heactb",
    "heactc",
    "heska",
    "heskb",
    "hesmk",
    "heskd",
    "cflisen",
    "cflisd",
    "cfsva",
    "cfsvb",
    "cfsvc",
    "cfsvd",
    "cfsve",
    "wpemp",
    "sclifea",
    "hefunc",
]
W10_IFS_WANTED = ["idauniq", "age", "sex", "inst", "wgt", "intdaty", "intdatm", "elsa", "marstat"]
W10_FIN_WANTED = ["idauniq", "totinc_bu_s", "nettotw_bu_s"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean ELSA into HRS external-validation tables.")
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


def tab_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        line = f.readline()
    sep = "\t" if "\t" in line else ","
    return [str(c).strip() for c in line.rstrip("\n").split(sep)]


def read_tab(path: Path, wanted: Sequence[str] | None = None) -> pd.DataFrame:
    header = tab_header(path)
    lower_map = {str(c).strip().lower(): c for c in header}
    if wanted is None:
        cols = header
    else:
        cols = [lower_map[w] for w in wanted if w in lower_map]
        if "idauniq" in lower_map and lower_map["idauniq"] not in cols:
            cols.insert(0, lower_map["idauniq"])
        if not cols:
            raise ValueError(f"None of the requested columns were found in {path}")
    sep = "\t" if any("\t" in c for c in header) or True else ","
    LOGGER.info("Reading %s (%s columns)", path.name, len(cols))
    frame = pd.read_csv(path, sep="\t", usecols=cols, low_memory=False)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    return frame


def normalize_identifier(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    mask = text.notna() & text.ne("") & text.ne("<NA>")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out.loc[mask] = text.loc[mask].str.replace(r"\.0$", "", regex=True)
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


def yesno12(series: pd.Series | None) -> pd.Series | float:
    if series is None:
        return np.nan
    return recode_map(series, YESNO12)


def binary_flag(series: pd.Series | None) -> pd.Series | float:
    if series is None:
        return np.nan
    values = to_num(series)
    return values.map({0.0: 0.0, 1.0: 1.0, 2.0: 0.0}).where(values.isin([0.0, 1.0, 2.0]))


def raw_num(frame: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in frame.columns:
        return None
    return pd.to_numeric(frame[name], errors="coerce")


def fold_15(series: pd.Series | None, high: float = 5.0) -> pd.Series | float:
    if series is None:
        return np.nan
    values = to_num(series)
    values = values.mask(values.eq(6.0), high)
    return values.where(values.between(1.0, high))


def sum_named(out: pd.DataFrame, names: Sequence[str], min_count: int) -> pd.Series:
    parts = [out[n] for n in names if isinstance(out.get(n), pd.Series)]
    if not parts:
        return pd.Series(np.nan, index=out.index)
    mat = pd.concat(parts, axis=1)
    return mat.sum(axis=1, min_count=min_count)


def any_yes(parts: Sequence[pd.Series]) -> pd.Series:
    if not parts:
        return pd.Series(dtype=float)
    mat = pd.concat(parts, axis=1)
    return mat.eq(1).any(axis=1).astype(float).where(mat.notna().any(axis=1))


def expand_cpi(cpi: dict[int, float]) -> dict[int, float]:
    years = list(range(min(cpi), max(cpi) + 1))
    series = pd.Series(cpi, dtype=float).reindex(years).interpolate(method="linear")
    return {int(k): float(v) for k, v in series.items() if pd.notna(v)}


def convert_gbp_money_to_usd(frame: pd.DataFrame) -> pd.DataFrame:
    """Multiply income/wealth by interview-year USD/GBP (FRED AEXUSUK)."""
    out = frame.copy()
    years = pd.to_numeric(out["interview_year"], errors="coerce")
    rate = years.map(usd_per_gbp).astype("float64")
    converted = 0
    for col in MONEY_GBP_COLUMNS:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        n = int(values.notna().sum())
        out[col] = values * rate
        converted += n
    LOGGER.info(
        "Converted income/wealth GBP→USD with AEXUSUK (n_nonmissing_cells=%s)",
        converted,
    )
    return out


def g3_wanted_columns(waves: Sequence[int]) -> list[str]:
    cols = set(G3_STATIC)
    for wave in waves:
        if int(wave) >= 10:
            continue
        cols.add(f"inw{wave}")
        cols.add(f"inw{wave}xt")
        for stem in G3_R_STEMS:
            cols.add(f"r{wave}{stem}")
        for stem in G3_H_STEMS:
            cols.add(f"h{wave}{stem}")
        for stem in G3_HH_STEMS:
            cols.add(f"hh{wave}{stem}")
    return sorted(cols)


def cognition_from_parts(
    imrc: pd.Series | None,
    dlrc: pd.Series | None,
    ser7: pd.Series | None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    index = imrc.index if isinstance(imrc, pd.Series) else (dlrc.index if isinstance(dlrc, pd.Series) else ser7.index)
    empty = pd.Series(np.nan, index=index)
    imrc = imrc if isinstance(imrc, pd.Series) else empty
    dlrc = dlrc if isinstance(dlrc, pd.Series) else empty
    ser7 = ser7 if isinstance(ser7, pd.Series) else empty
    tr20 = (imrc + dlrc).where(imrc.notna() & dlrc.notna())
    raw25 = (imrc + dlrc + ser7).where(imrc.notna() & dlrc.notna() & ser7.notna())
    raw20 = (imrc + dlrc).where(imrc.notna() & dlrc.notna())
    cog = raw25 * (27.0 / 25.0)
    cog = cog.fillna(raw20 * (27.0 / 20.0))
    return imrc, dlrc, tr20, cog


def serial7_from_remainders(frame: pd.DataFrame) -> pd.Series:
    hits = []
    for col, target in zip(("cfsva", "cfsvb", "cfsvc", "cfsvd", "cfsve"), SERIAL7_TARGETS):
        if col not in frame.columns:
            continue
        raw = to_num(frame[col])
        hits.append(raw.eq(target).astype(float).where(raw.notna()))
    if not hits:
        return pd.Series(np.nan, index=frame.index)
    return pd.concat(hits, axis=1).sum(axis=1, min_count=1)


def extract_wave_g3(core: pd.DataFrame, wave: int) -> pd.DataFrame:
    out = pd.DataFrame(index=core.index)
    out["person_id"] = normalize_identifier(core["idauniq"])
    hh = pick(core, wave, ("hhid",))
    out["household_id"] = normalize_identifier(hh) if hh is not None else pd.Series(pd.NA, index=core.index, dtype="string")
    pn = pick(core, wave, ("pn", "perid"))
    if pn is None and "pn" in core.columns:
        pn = to_num(core["pn"])
    out["person_number"] = pn
    out["wave"] = wave

    inw = pick(core, wave, (f"inw{wave}",))
    live = inw.eq(1) if inw is not None else pd.Series(False, index=core.index)
    out["_live"] = live.fillna(False)
    out["interview_status"] = np.where(out["_live"], 1, np.nan)

    year = pick(core, wave, ("iwindy",))
    month = pick(core, wave, ("iwindm",))
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

    sex = pick(core, wave, ("ragender",))
    out["sex"] = recode_map(sex, {1: 1.0, 2: 0.0}) if sex is not None else np.nan
    out["age_years"] = pick(core, wave, ("agey",))
    out["birth_year"] = pick(core, wave, ("rabyear",))
    out["birth_month"] = pick(core, wave, ("rabmonth",))
    educl = pick(core, wave, ("raeducl",))
    out["education_level"] = educl.where(educl.isin([1, 2, 3])) if educl is not None else np.nan
    out["education_years"] = np.nan
    born = pick(core, wave, ("rabplace",))
    if born is not None:
        out["foreign_born"] = np.where(born.isna(), np.nan, np.where(born.eq(1), 0.0, np.where(born.gt(0), 1.0, np.nan)))
    else:
        out["foreign_born"] = np.nan
    ch = pick(core, wave, ("rachshlt",))
    out["childhood_health"] = fold_15(ch) if ch is not None else np.nan

    out["marital_status"] = pick(core, wave, ("mstat",))
    hhsize = pick(core, wave, ("hhres",))
    out["household_size"] = hhsize
    out["living_alone"] = np.where(hhsize.isna(), np.nan, (hhsize == 1).astype(float)) if hhsize is not None else np.nan
    out["living_children"] = pick(core, wave, ("child",))
    out["living_siblings"] = pick(core, wave, ("livsib",))
    own = pick(core, wave, ("hownrnt",))
    out["home_ownership"] = np.where(own.isna(), np.nan, (own == 1).astype(float)) if own is not None else np.nan
    out["nursing_home_residence"] = bin01(pick(core, wave, ("nhmliv",)))
    work = bin01(pick(core, wave, ("work",)))
    out["currently_working"] = work
    lbrf = pick(core, wave, ("lbrf_e",))
    out["labor_force_status"] = recode_map(lbrf, LBRF_TO_HRS) if lbrf is not None else np.nan
    out["self_employed"] = np.where(lbrf.isna(), np.nan, (lbrf == 2).astype(float)) if lbrf is not None else np.nan
    hours = pick(core, wave, ("jhours",))
    out["hours_worked_per_week"] = hours.where(hours.between(0, 168)) if hours is not None else np.nan
    weeks = pick(core, wave, ("jweeks_e",))
    out["weeks_worked_per_year"] = weeks.where(weeks.between(0, 52)) if weeks is not None else np.nan
    out["health_limits_work"] = bin01(pick(core, wave, ("hlthlm",)))
    out["health_insurance_any"] = np.where(out["_live"], 1.0, np.nan)

    shlt = pick(core, wave, ("shlt", "shlta"))
    out["self_rated_health"] = shlt.where(shlt.between(1, 5)) if shlt is not None else np.nan
    for name, stem in DISEASE_STEMS.items():
        out[name] = bin01(pick(core, wave, (stem,)))
    for name, stem in ADL_ITEMS.items():
        out[name] = bin01(pick(core, wave, (stem,)))
    adl_parts = [out[n] for n in ADL_ITEMS if isinstance(out.get(n), pd.Series)]
    if adl_parts:
        amat = pd.concat(adl_parts, axis=1)
        out["adl_total_score"] = amat.sum(axis=1, min_count=3)
        out["receives_adl_help"] = amat.eq(1).any(axis=1).astype(float).where(amat.notna().any(axis=1))
    else:
        out["adl_total_score"] = np.nan
        out["receives_adl_help"] = np.nan
    for name, stem in IADL_ITEMS.items():
        out[name] = bin01(pick(core, wave, (stem,)))
    iadl_total = pick(core, wave, ("iadlza", "iadla"))
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
    out["large_muscle_total_score"] = sum_named(out, HRS_LARGE_MUSCLE_ITEMS, min_count=3)
    out["fine_motor_total_score"] = sum_named(out, HRS_FINE_MOTOR_ITEMS, min_count=1)
    out["urinary_incontinence"] = bin01(pick(core, wave, ("urinai",)))

    cesd_cols = []
    for hrs_name, stem, direction in CESD_ITEMS_G3:
        rec = bin01(pick(core, wave, (stem,)))
        if not isinstance(rec, pd.Series):
            rec = pd.Series(np.nan, index=core.index)
        out[hrs_name] = rec
        if direction == "positive_raw":
            cesd_cols.append((1.0 - rec).where(rec.notna()))
        else:
            cesd_cols.append(rec)
    native_cesd = pick(core, wave, ("cesd",))
    item_score = pd.concat(cesd_cols, axis=1).sum(axis=1, min_count=6) if cesd_cols else pd.Series(np.nan, index=core.index)
    if native_cesd is not None:
        out["cesd_score"] = native_cesd.where(native_cesd.between(0, 8)).fillna(item_score)
    else:
        out["cesd_score"] = item_score

    imrc, dlrc, tr20, cog = cognition_from_parts(
        pick(core, wave, ("imrc",)),
        pick(core, wave, ("dlrc",)),
        pick(core, wave, ("ser7",)),
    )
    out["immediate_word_recall"] = imrc
    out["delayed_word_recall"] = dlrc
    out["serial_sevens"] = pick(core, wave, ("ser7",))
    native_tr = pick(core, wave, ("tr20",))
    out["total_word_recall"] = native_tr if native_tr is not None else tr20
    out["cognition_27_score"] = cog
    orient = pick(core, wave, ("orient",))
    draw = bin01(pick(core, wave, ("draw",)))
    ser7 = out["serial_sevens"] if isinstance(out["serial_sevens"], pd.Series) else None
    ms_parts = [s for s in (ser7, orient, draw if isinstance(draw, pd.Series) else None) if isinstance(s, pd.Series)]
    out["mental_status_score"] = pd.concat(ms_parts, axis=1).sum(axis=1, min_count=2) if ms_parts else np.nan
    tot_parts = [s for s in (out.get("total_word_recall"), ser7, orient, draw if isinstance(draw, pd.Series) else None) if isinstance(s, pd.Series)]
    out["total_cognition_score"] = pd.concat(tot_parts, axis=1).sum(axis=1, min_count=2) if tot_parts else np.nan
    out["self_rated_memory"] = pick(core, wave, ("slfmem",))
    sat = pick(core, wave, ("satlife_e", "satlife"))
    out["life_satisfaction"] = sat.where(sat.between(1, 7)) if sat is not None else np.nan
    bwc = pick(core, wave, ("bwc20",))
    out["backward_counting"] = bwc.where(bwc.isin([0.0, 1.0, 2.0])) if bwc is not None else np.nan
    out["social_contact_children"] = bin01(pick(core, wave, ("kcntf",)))
    meet_fr = pick(core, wave, ("unfriend",))
    if meet_fr is not None:
        out["social_contact_friends"] = np.where(
            meet_fr.isna(),
            np.nan,
            np.where(meet_fr.isin([1.0, 2.0]), 1.0, np.where(meet_fr.between(3, 7), 0.0, np.nan)),
        )
    else:
        out["social_contact_friends"] = np.nan
    out["shortness_of_breath"] = bin01(pick(core, wave, ("breath_e",)))
    bal = pick(core, wave, ("balance_e",))
    out["balance_score"] = bal.where(bal.isin([1.0, 2.0, 3.0, 4.0])) if bal is not None else np.nan

    out["eyesight"] = fold_15(pick(core, wave, ("sight",)))
    out["near_vision"] = fold_15(pick(core, wave, ("nsight",)))
    out["distance_vision"] = fold_15(pick(core, wave, ("dsight",)))
    out["hearing"] = fold_15(pick(core, wave, ("hearing",)))
    out["pain_presence"] = bin01(pick(core, wave, ("painfr",)))
    painlv = pick(core, wave, ("painlv",))
    out["pain_severity"] = painlv.where(painlv.between(0, 3)) if painlv is not None else np.nan
    out["falls_any"] = bin01(pick(core, wave, ("fall",)))
    falls_n = pick(core, wave, ("fallnum",))
    out["falls_count"] = falls_n
    if isinstance(out["falls_any"], pd.Series) and isinstance(out["falls_count"], pd.Series):
        out.loc[out["falls_any"].eq(0), "falls_count"] = 0.0
    out["fall_injury"] = bin01(pick(core, wave, ("fallinj",)))

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
    if bmi is None or (isinstance(bmi, pd.Series) and not bmi.notna().any()):
        if isinstance(out["measured_height"], pd.Series) and isinstance(out["measured_weight"], pd.Series):
            bmi = out["measured_weight"] / (out["measured_height"] ** 2)
    out["measured_bmi"] = bmi.where(bmi.between(10, 80)) if isinstance(bmi, pd.Series) else np.nan
    waist = pick(core, wave, ("mwaist",))
    out["waist_circumference"] = waist.where(waist.between(30, 220)) if waist is not None else np.nan

    out["home_health_care"] = bin01(pick(core, wave, ("rfaany_e",)))
    out["total_household_income"] = pick(core, wave, ("itot",))
    out["total_household_wealth"] = pick(core, wave, ("atotb",))
    wgt = pick(core, wave, ("cwtresp",))
    out["respondent_weight"] = wgt.where(wgt > 0) if wgt is not None else np.nan

    ever = bin01(pick(core, wave, ("smokev",)))
    current = bin01(pick(core, wave, ("smoken",)))
    out["_ever_smoked_wave"] = ever
    out["_current_smoke"] = current
    out["cigarettes_per_day"] = pick(core, wave, ("smokef",))
    drink = bin01(pick(core, wave, ("drink",)))
    out["alcohol_use"] = drink
    days = pick(core, wave, ("drinkd_e",))
    out["alcohol_days_per_week"] = days.where(days.between(0, 7)) if days is not None else np.nan
    drinks = pick(core, wave, ("drinkn_e",))
    out["drinks_per_drinking_day"] = drinks.where(drinks.between(0, 30)) if drinks is not None else np.nan
    if isinstance(out["alcohol_use"], pd.Series):
        out.loc[out["alcohol_use"].eq(0), "alcohol_days_per_week"] = 0.0
        out.loc[out["alcohol_use"].eq(0), "drinks_per_drinking_day"] = np.nan
    for hrs_name, stem in (
        ("vigorous_activity_frequency", "vgactx_e"),
        ("moderate_activity_frequency", "mdactx_e"),
        ("light_activity_frequency", "ltactx_e"),
    ):
        raw = pick(core, wave, (stem,))
        out[hrs_name] = recode_map(raw, ACT5_TO_HRS) if raw is not None else np.nan
    out["hypertension_treatment"] = bin01(pick(core, wave, ("rxhibp",)))
    out["diabetes_oral_medication"] = bin01(pick(core, wave, ("rxdiab",)))

    keep = out["_live"].fillna(False)
    result = out.loc[keep].drop(columns=["_live"]).reset_index(drop=True)
    LOGGER.info("G3 wave %s extracted n=%s pids=%s", wave, len(result), result["person_id"].nunique())
    return result


def extract_wave_w10(core: pd.DataFrame, static: pd.DataFrame | None) -> pd.DataFrame:
    out = pd.DataFrame(index=core.index)
    out["person_id"] = normalize_identifier(core["idauniq"])
    hh = core["idahhw10"] if "idahhw10" in core.columns else None
    out["household_id"] = normalize_identifier(hh) if hh is not None else pd.Series(pd.NA, index=core.index, dtype="string")
    out["wave"] = 10
    indout = to_num(core["w10indout"]) if "w10indout" in core.columns else pd.Series(np.nan, index=core.index)
    live = indout.isin(W10_LIVE_INDOUT)
    out["_live"] = live.fillna(False)
    out["interview_status"] = np.where(out["_live"], 1, np.nan)

    year = to_num(core["iintdaty"]) if "iintdaty" in core.columns else pd.Series(np.nan, index=core.index)
    if year.isna().all() and "intdaty" in core.columns:
        year = to_num(core["intdaty"])
    month = to_num(core["iintdatm"]) if "iintdatm" in core.columns else pd.Series(np.nan, index=core.index)
    if month.isna().all() and "intdatm" in core.columns:
        month = to_num(core["intdatm"])
    out["interview_year"] = year.fillna(WAVE_YEAR[10])
    out["interview_month"] = month
    out["interview_date"] = pd.to_datetime(
        {"year": out["interview_year"], "month": month.fillna(7).clip(1, 12), "day": 15},
        errors="coerce",
    )

    sex = to_num(core["sex"]) if "sex" in core.columns else None
    if sex is None or not sex.notna().any():
        sex = to_num(core["dhsex"]) if "dhsex" in core.columns else None
    out["sex"] = recode_map(sex, {1: 1.0, 2: 0.0}) if sex is not None else np.nan
    age = to_num(core["age"]) if "age" in core.columns else None
    out["age_years"] = age if age is not None else np.nan
    out["nursing_home_residence"] = bin01(to_num(core["inst"]) if "inst" in core.columns else None)
    wgt = to_num(core["wgt"]) if "wgt" in core.columns else None
    out["respondent_weight"] = wgt.where(wgt > 0) if wgt is not None else np.nan
    hhsize = to_num(core["hhtot"]) if "hhtot" in core.columns else None
    out["household_size"] = hhsize
    out["living_alone"] = np.where(hhsize.isna(), np.nan, (hhsize == 1).astype(float)) if hhsize is not None else np.nan
    out["health_insurance_any"] = np.where(out["_live"], 1.0, np.nan)
    work = yesno12(to_num(core["wpemp"]) if "wpemp" in core.columns else None)
    out["currently_working"] = work
    marital = recode_map(to_num(core["marstat"]), IFS_MARSTAT_TO_HRS) if "marstat" in core.columns else pd.Series(np.nan, index=core.index)
    if "dimarr" in core.columns:
        marital = marital.fillna(recode_map(to_num(core["dimarr"]), DIMARR_TO_HRS))
    elif "dimar" in core.columns:
        marital = marital.fillna(recode_map(to_num(core["dimar"]), DIMARR_TO_HRS))
    out["marital_status"] = marital
    out["self_rated_health"] = fold_15(to_num(core["hehelf"]) if "hehelf" in core.columns else None)
    sat = to_num(core["sclifea"]) if "sclifea" in core.columns else None
    out["life_satisfaction"] = sat.where(sat.between(1, 7)) if sat is not None else np.nan

    st = static.set_index("person_id") if static is not None and not static.empty else None
    for col in ("sex", "birth_year", "birth_month", "education_level", "foreign_born", "childhood_health"):
        donor = out["person_id"].map(st[col]) if st is not None and col in st.columns else pd.Series(np.nan, index=out.index)
        current = out[col] if col in out.columns and isinstance(out[col], pd.Series) else pd.Series(np.nan, index=out.index)
        out[col] = pd.to_numeric(current, errors="coerce").fillna(pd.to_numeric(donor, errors="coerce"))
    out["education_years"] = np.nan

    def w10_flag(*names: str) -> pd.Series:
        parts = [binary_flag(core[n]) for n in names if n in core.columns]
        parts = [p for p in parts if isinstance(p, pd.Series)]
        if not parts:
            return pd.Series(np.nan, index=core.index)
        if len(parts) == 1:
            return parts[0]
        return any_yes(parts)

    out["hypertension_dx"] = w10_flag("heeverbp")
    diab = raw_num(core, "hehavedi")
    if diab is not None:
        diab = diab.mask(diab.lt(-1))
        dx, oral = recode_w10_hehavedi(diab)
        out["diabetes_dx"] = dx
        out["diabetes_oral_medication"] = oral
    else:
        out["diabetes_dx"] = np.nan
        out["diabetes_oral_medication"] = np.nan
    out["cancer_dx"] = w10_flag("heeverca")
    out["lung_disease_dx"] = w10_flag("heevercl", "heeveras")
    out["heart_disease_dx"] = w10_flag("heeveran", "heevermi", "heeverhf", "heeverah")
    out["stroke_dx"] = w10_flag("heeverst")
    out["psychiatric_dx"] = w10_flag("heeverps")
    out["arthritis_dx"] = w10_flag("heeverar")
    out["memory_disease_dx"] = w10_flag("heeverad", "heeverdm")

    for name, col in ADL_ITEMS_W10.items():
        raw = binary_flag(core[col]) if col in core.columns else np.nan
        out[name] = raw if isinstance(raw, pd.Series) else bin01(to_num(core[col]) if col in core.columns else None)
    adl_parts = [out[n] for n in ADL_ITEMS_W10 if isinstance(out.get(n), pd.Series)]
    if adl_parts:
        amat = pd.concat(adl_parts, axis=1)
        out["adl_total_score"] = amat.sum(axis=1, min_count=3)
        out["receives_adl_help"] = amat.eq(1).any(axis=1).astype(float).where(amat.notna().any(axis=1))
    else:
        out["adl_total_score"] = np.nan
        out["receives_adl_help"] = np.nan
    for name, col in IADL_ITEMS_W10.items():
        raw = binary_flag(core[col]) if col in core.columns else np.nan
        out[name] = raw if isinstance(raw, pd.Series) else np.nan
    iadl_parts = [out[n] for n in IADL_ITEMS_W10 if isinstance(out.get(n), pd.Series)]
    if iadl_parts:
        imat = pd.concat(iadl_parts, axis=1)
        out["iadl_total_score"] = imat.sum(axis=1, min_count=3)
        out["receives_iadl_help"] = imat.eq(1).any(axis=1).astype(float).where(imat.notna().any(axis=1))
    else:
        out["iadl_total_score"] = np.nan
        out["receives_iadl_help"] = np.nan

    cesd_cols = []
    for hrs_name, col, direction in CESD_ITEMS_W10:
        rec = yesno12(to_num(core[col]) if col in core.columns else None)
        if not isinstance(rec, pd.Series):
            rec = pd.Series(np.nan, index=core.index)
        out[hrs_name] = rec
        cesd_cols.append((1.0 - rec).where(rec.notna()) if direction == "positive_raw" else rec)
    out["cesd_score"] = pd.concat(cesd_cols, axis=1).sum(axis=1, min_count=6) if cesd_cols else np.nan

    imrc = to_num(core["cflisen"]) if "cflisen" in core.columns else None
    dlrc = to_num(core["cflisd"]) if "cflisd" in core.columns else None
    ser7 = serial7_from_remainders(core)
    imrc, dlrc, tr20, cog = cognition_from_parts(imrc, dlrc, ser7)
    out["immediate_word_recall"] = imrc
    out["delayed_word_recall"] = dlrc
    out["serial_sevens"] = ser7
    out["total_word_recall"] = tr20
    out["cognition_27_score"] = cog
    out["eyesight"] = fold_15(to_num(core["heeye"]) if "heeye" in core.columns else None)
    out["hearing"] = fold_15(to_num(core["hehear"]) if "hehear" in core.columns else None)
    diz = to_num(core["hediz"]) if "hediz" in core.columns else None
    if diz is not None:
        out["dizziness"] = np.where(diz.isin([1.0, 2.0, 3.0, 4.0]), 1.0, np.where(diz.eq(5.0), 0.0, np.nan))
    else:
        out["dizziness"] = np.nan
    out["pain_presence"] = yesno12(to_num(core["hepain"]) if "hepain" in core.columns else None)

    heska = raw_num(core, "heska")
    if heska is not None:
        heska = heska.mask(heska.lt(-1))
    ever = binary_flag(core["hesmk"] if "hesmk" in core.columns else (core["heskd"] if "heskd" in core.columns else None))
    current = pd.Series(np.nan, index=core.index)
    if heska is not None:
        current = pd.Series(np.where(heska.eq(1), 1.0, np.where(heska.eq(2), 0.0, np.nan)), index=core.index)
        if not isinstance(ever, pd.Series) or not ever.notna().any():
            ever = pd.Series(np.where(heska.eq(-1), 0.0, np.where(heska.isin([1.0, 2.0]), 1.0, np.nan)), index=core.index)
    out["_ever_smoked_wave"] = ever
    out["_current_smoke"] = current
    cig = to_num(core["heskb"]) if "heskb" in core.columns else None
    out["cigarettes_per_day"] = cig.where(cig.between(0, 80)) if cig is not None else np.nan
    for hrs_name, col in (
        ("vigorous_activity_frequency", "heacta"),
        ("moderate_activity_frequency", "heactb"),
        ("light_activity_frequency", "heactc"),
    ):
        raw = to_num(core[col]) if col in core.columns else None
        out[hrs_name] = recode_map(raw, ACT4_TO_HRS) if raw is not None else np.nan

    inc = to_num(core["totinc_bu_s"]) if "totinc_bu_s" in core.columns else None
    wealth = to_num(core["nettotw_bu_s"]) if "nettotw_bu_s" in core.columns else None
    out["total_household_income"] = annualise_w10_weekly_income(inc) if inc is not None else np.nan
    out["total_household_wealth"] = wealth

    keep = out["_live"].fillna(False)
    result = out.loc[keep].drop(columns=["_live"]).reset_index(drop=True)
    LOGGER.info("W10 extracted n=%s pids=%s", len(result), result["person_id"].nunique())
    return result


def g3_static_table(core: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"person_id": normalize_identifier(core["idauniq"])})
    sex = to_num(core["ragender"]) if "ragender" in core.columns else None
    out["sex"] = recode_map(sex, {1: 1.0, 2: 0.0}) if sex is not None else np.nan
    out["birth_year"] = to_num(core["rabyear"]) if "rabyear" in core.columns else np.nan
    out["birth_month"] = to_num(core["rabmonth"]) if "rabmonth" in core.columns else np.nan
    educl = to_num(core["raeducl"]) if "raeducl" in core.columns else None
    out["education_level"] = educl.where(educl.isin([1, 2, 3])) if educl is not None else np.nan
    born = to_num(core["rabplace"]) if "rabplace" in core.columns else None
    if born is not None:
        out["foreign_born"] = np.where(born.isna(), np.nan, np.where(born.eq(1), 0.0, np.where(born.gt(0), 1.0, np.nan)))
    else:
        out["foreign_born"] = np.nan
    out["childhood_health"] = fold_15(to_num(core["rachshlt"]) if "rachshlt" in core.columns else None)
    return out.drop_duplicates("person_id")


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


def load_g3(path: Path, waves: Sequence[int]) -> pd.DataFrame:
    wanted = g3_wanted_columns(waves)
    frame = read_tab(path, wanted)
    if "idauniq" not in frame.columns:
        raise ValueError(f"{path} has no idauniq")
    LOGGER.info("ELSA G3 n=%s ncols=%s", f"{len(frame):,}", len(frame.columns))
    return frame


def load_w10(data_dir: Path) -> pd.DataFrame:
    core = read_tab(data_dir / W10_CORE_FILE, W10_CORE_WANTED)
    ifs_path = data_dir / W10_IFS_FILE
    fin_path = data_dir / W10_FIN_FILE
    if ifs_path.exists():
        ifs = read_tab(ifs_path, W10_IFS_WANTED)
        overlap = [c for c in ifs.columns if c != "idauniq" and c in core.columns]
        ifs = ifs.drop(columns=overlap)
        core = core.merge(ifs, on="idauniq", how="left")
    if fin_path.exists():
        fin = read_tab(fin_path, W10_FIN_WANTED)
        overlap = [c for c in fin.columns if c != "idauniq" and c in core.columns]
        fin = fin.drop(columns=overlap)
        core = core.merge(fin, on="idauniq", how="left")
    LOGGER.info("ELSA W10 merged n=%s ncols=%s", f"{len(core):,}", len(core.columns))
    return core


def load_eol(path: Path, waves: Sequence[int]) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["person_id", "wave", "interview_status", "death_year", "death_month"])
    if not path.exists():
        LOGGER.warning("Missing EOL file %s", path)
        return empty
    header = [c.lower() for c in tab_header(path)]
    wanted = [c for c in ("idauniq", "raxt", "radyear", "radmonth", "raxtiwy", "raxtiwm", "raxyear") if c in header]
    eol = read_tab(path, wanted)
    pid = normalize_identifier(eol["idauniq"])
    death_wave = to_num(eol["raxt"]) if "raxt" in eol.columns else pd.Series(np.nan, index=eol.index)
    year = to_num(eol["radyear"]) if "radyear" in eol.columns else pd.Series(np.nan, index=eol.index)
    if year.isna().all() and "raxyear" in eol.columns:
        year = to_num(eol["raxyear"])
    if year.isna().all() and "raxtiwy" in eol.columns:
        year = to_num(eol["raxtiwy"])
    month = to_num(eol["radmonth"]) if "radmonth" in eol.columns else pd.Series(np.nan, index=eol.index)
    if month.isna().all() and "raxtiwm" in eol.columns:
        month = to_num(eol["raxtiwm"])
    wave_set = set(int(w) for w in waves)
    recs = []
    for i in range(len(eol)):
        wave = death_wave.iloc[i]
        if pd.isna(wave) or int(wave) not in wave_set:
            continue
        recs.append(
            {
                "person_id": str(pid.iloc[i]),
                "wave": int(wave),
                "interview_status": 5,
                "death_year": year.iloc[i],
                "death_month": month.iloc[i],
            }
        )
    out = pd.DataFrame(recs).drop_duplicates(["person_id", "wave"]) if recs else empty
    LOGGER.info("EOL death rows %s", len(out))
    return out


def deaths_from_g3(core: pd.DataFrame, waves: Sequence[int], live_keys: set[tuple[str, int]]) -> pd.DataFrame:
    recs: list[dict[str, Any]] = []
    radyear = to_num(core["radyear"]) if "radyear" in core.columns else None
    radmonth = to_num(core["radmonth"]) if "radmonth" in core.columns else None
    pid = normalize_identifier(core["idauniq"])
    g3_waves = [w for w in waves if int(w) < 10]
    for wave in g3_waves:
        iwstat = pick(core, wave, ("iwstat",))
        inw = pick(core, wave, (f"inw{wave}",))
        dead = pd.Series(False, index=core.index)
        if iwstat is not None:
            dead = dead | iwstat.eq(5)
        if inw is not None:
            dead = dead & ~inw.eq(1)
        mask = dead.fillna(False)
        if not mask.any():
            continue
        years = radyear.loc[mask].to_numpy() if radyear is not None else np.full(int(mask.sum()), np.nan)
        months = radmonth.loc[mask].to_numpy() if radmonth is not None else np.full(int(mask.sum()), np.nan)
        for person, year, month in zip(pid.loc[mask].astype(str).to_numpy(), years, months):
            recs.append(
                {
                    "person_id": str(person),
                    "wave": int(wave),
                    "interview_status": 5,
                    "death_year": year,
                    "death_month": month,
                }
            )
    if radyear is not None:
        pid_values = pid.astype(str).to_numpy()
        years = radyear.to_numpy()
        months = radmonth.to_numpy() if radmonth is not None else np.full(len(core), np.nan)
        for person, year, month in zip(pid_values, years, months):
            if pd.isna(year):
                continue
            assigned = None
            for wave in waves:
                if WAVE_YEAR[int(wave)] >= float(year) and (person, int(wave)) not in live_keys:
                    assigned = int(wave)
                    break
            if assigned is None:
                last = int(waves[-1])
                if (person, last) not in live_keys:
                    assigned = last
            if assigned is None:
                continue
            recs.append(
                {
                    "person_id": str(person),
                    "wave": assigned,
                    "interview_status": 5,
                    "death_year": year,
                    "death_month": month,
                }
            )
    if not recs:
        return pd.DataFrame(columns=["person_id", "wave", "interview_status", "death_year", "death_month"])
    out = pd.DataFrame(recs).drop_duplicates(["person_id", "wave"])
    LOGGER.info("G3/radyear death rows %s", len(out))
    return out


def specs_from_dictionary(path: Path | str) -> dict[str, hrs.VariableSpec]:
    text = str(path).strip()
    workbook = Path(text) if text and text != "." else None
    if workbook is not None and workbook.is_file():
        try:
            _df, specs = hrs.read_dictionary(workbook)
            return specs
        except Exception as exc:
            LOGGER.warning("ELSA dictionary reader failed (%s); using built-in specs.", exc)
    return hrs.builtin_specs()


def write_long_and_audit(long_raw: pd.DataFrame, specs: dict[str, hrs.VariableSpec], output_dir: Path) -> None:
    long_raw.to_csv(output_dir / "elsa_world_model_long_raw.csv", index=False, encoding="utf-8-sig")
    rows = []
    for name, spec in specs.items():
        info = VAR_COVERAGE.get(name, {})
        series = pd.to_numeric(long_raw[name], errors="coerce") if name in long_raw.columns else pd.Series(dtype=float)
        rows.append(
            {
                "variable": name,
                "role": spec.role,
                "priority": spec.priority,
                "elsa_coverage": info.get("coverage", "unavailable"),
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
    parquet_path = output_dir / "ELSA_external.parquet"
    out.to_parquet(parquet_path, index=False)
    if csv_copy:
        out.to_csv(output_dir / "ELSA_external.csv", index=False, encoding="utf-8-sig")
    meta = {
        "role": "hrs_external_validation",
        "n_transitions": int(len(out)),
        "n_persons": int(out["person_id"].nunique()),
        "hrs_preprocessing": str(preprocess_path),
        "hrs_model_config": str(config_path),
        "note": "W10 income annualised (weekly x 52) then income/wealth GBP->USD with FRED AEXUSUK (interview year), then z-scored with HRS training continuous stats. Masks flag ELSA observation.",
    }
    (output_dir / "ELSA_external_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info("Wrote external bundle %s (%s rows)", parquet_path, f"{len(out):,}")


def run_self_test() -> int:
    assert recode_map(pd.Series([1, 5, -8]), ACT5_TO_HRS).tolist() == [4.0, 0.0, np.nan] or (
        recode_map(pd.Series([1, 5, -8]), ACT5_TO_HRS).iloc[0] == 4.0
        and recode_map(pd.Series([1, 5, -8]), ACT5_TO_HRS).iloc[1] == 0.0
        and pd.isna(recode_map(pd.Series([1, 5, -8]), ACT5_TO_HRS).iloc[2])
    )
    mapped = recode_map(pd.Series([1, 4, -1]), ACT4_TO_HRS)
    assert mapped.iloc[0] == 3.0 and mapped.iloc[1] == 0.0 and pd.isna(mapped.iloc[2])
    yn = recode_map(pd.Series([1, 2, -9]), YESNO12)
    assert yn.iloc[0] == 1.0 and yn.iloc[1] == 0.0 and pd.isna(yn.iloc[2])
    dummy = pd.DataFrame({"cfsva": [93, 90], "cfsvb": [86, 86], "cfsvc": [79, 79], "cfsvd": [72, 70], "cfsve": [65, 65]})
    ser = serial7_from_remainders(dummy)
    assert ser.iloc[0] == 5.0
    assert ser.iloc[1] == 3.0
    assert coverage_of("sex") == "available"
    assert coverage_of("medicare_coverage") == "us_only"
    assert coverage_of("weeks_worked_per_year") == "available"
    assert coverage_of("backward_counting") == "partial"
    assert coverage_of("balance_score") == "partial"
    assert coverage_of("shortness_of_breath") == "partial"
    assert coverage_of("education_years") == "unavailable"
    assert coverage_of("hospitalization") == "unavailable"
    assert coverage_of("cesd_enjoyed_life") == "available"
    assert coverage_of("fine_motor_total_score") == "derived"
    dx, oral = recode_w10_hehavedi(pd.Series([1.0, 2.0, 3.0, -1.0, np.nan]))
    assert dx.tolist()[:4] == [1.0, 1.0, 1.0, 0.0]
    assert oral.tolist()[:4] == [0.0, 1.0, 0.0, 0.0]
    assert pd.isna(dx.iloc[4]) and pd.isna(oral.iloc[4])
    weekly = annualise_w10_weekly_income(pd.Series([10.0, np.nan]))
    assert abs(float(weekly.iloc[0]) - 520.0) < 1e-9
    assert pd.isna(weekly.iloc[1])
    assert recode_map(pd.Series([1, 2, 3, 5, 6]), IFS_MARSTAT_TO_HRS).tolist() == [1.0, 3.0, 8.0, 5.0, 7.0]
    assert abs(float(usd_per_gbp(2018)) - 1.3363) < 1e-9
    fx = convert_gbp_money_to_usd(
        pd.DataFrame(
            {
                "interview_year": [2002, 2018],
                "total_household_income": [1000.0, 1000.0],
                "total_household_wealth": [1000.0, np.nan],
            }
        )
    )
    assert abs(float(fx.loc[0, "total_household_income"]) - 1502.5) < 1e-6
    assert abs(float(fx.loc[1, "total_household_income"]) - 1336.3) < 1e-6
    assert pd.isna(fx.loc[1, "total_household_wealth"])
    if hrs.BUILTIN_SPECS_PATH.is_file():
        specs = hrs.builtin_specs()
        assert "hypertension_dx" in specs
        assert specs["hypertension_dx"].role == "Dynamic state variable"
    LOGGER.info("ELSA HRS self-test passed")
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
    waves = sorted(set(int(w) for w in args.waves))

    hrs.WAVE_YEAR.clear()
    hrs.WAVE_YEAR.update(WAVE_YEAR)
    hrs.CPI_U.clear()
    hrs.CPI_U.update(expand_cpi(UK_CPI))

    specs = specs_from_dictionary(Path(args.dictionary))
    g3_path = data_dir / G3_FILE
    g3 = load_g3(g3_path, waves) if g3_path.exists() and any(w < 10 for w in waves) else None
    if g3 is None and any(w < 10 for w in waves):
        raise FileNotFoundError(g3_path)
    w10 = load_w10(data_dir) if 10 in waves else None
    static = g3_static_table(g3) if g3 is not None else None

    live_parts = []
    for wave in waves:
        if wave < 10:
            live_parts.append(extract_wave_g3(g3, wave))
        elif w10 is not None:
            live_parts.append(extract_wave_w10(w10, static))
    long_live = pd.concat(live_parts, ignore_index=True, sort=False)
    long_live = convert_gbp_money_to_usd(long_live)
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
        if name not in state_action_names:
            state_action_names.append(name)
    for name in state_action_names:
        if name not in long_live.columns:
            long_live[name] = np.nan
    long_live = apply_observed_masks(long_live, state_action_names)

    live_keys = set(zip(long_live["person_id"].astype(str), long_live["wave"].astype(int)))
    exits = load_eol(data_dir / EOL_FILE, waves)
    if g3 is not None:
        exits = pd.concat([exits, deaths_from_g3(g3, waves, live_keys)], ignore_index=True)
        exits = exits.drop_duplicates(["person_id", "wave"])

    persons = pd.Index(sorted(set(long_live["person_id"].astype(str)).union(exits["person_id"].astype(str))))
    grid = pd.MultiIndex.from_product([persons, waves], names=["person_id", "wave"]).to_frame(index=False)
    live_status = long_live[["person_id", "wave", "interview_status"]].drop_duplicates()
    live_status["person_id"] = live_status["person_id"].astype(str)
    status_grid = grid.merge(live_status, on=["person_id", "wave"], how="left")
    exit_status = exits[["person_id", "wave", "interview_status"]].rename(columns={"interview_status": "exit_status"})
    exit_status["person_id"] = exit_status["person_id"].astype(str)
    status_grid = status_grid.merge(exit_status, on=["person_id", "wave"], how="left")
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
    trans_path = output_dir / "elsa_world_model_transitions.parquet"
    transitions.to_parquet(trans_path, index=False)
    if args.csv_copy:
        transitions.to_csv(output_dir / "elsa_world_model_transitions.csv", index=False, encoding="utf-8-sig")
    LOGGER.info(
        "Wrote %s (%s rows, %s persons)",
        trans_path,
        f"{len(transitions):,}",
        f"{transitions['person_id'].nunique():,}",
    )

    meta = {
        "dataset": "ELSA",
        "role": "hrs_external_validation",
        "waves": waves,
        "wave_year": WAVE_YEAR,
        "n_person_wave_live": int(len(long_raw)),
        "n_transitions": int(len(transitions)),
        "n_persons": int(transitions["person_id"].nunique()),
        "nearby_wave_fill_cells": {k: int(v) for k, v in fill_counts.items() if v},
        "currency": "USD (income: G3 annual GBP, W10 weekly totinc_bu_s x 52; then GBP x FRED AEXUSUK interview-year rate; not PPP)",
        "fx": "FRED AEXUSUK annual average, USD per 1 GBP",
        "oop_cpi": "UK CPI 2010=100, deflated to 2018; OOP not in G3 EUL",
        "death_note": "iwstat=5 + EOL-A2 + radyear. G3 has no new iwstat=5 in W7-W9.",
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
    print("ELSA HRS external-validation cleaning completed successfully.")
    print(f"Output folder: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
