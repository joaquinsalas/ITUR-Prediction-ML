#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute ITUR and export regression dataset:
  - normalized predictors
  - signed difference: RESUL_ITUR - ITUR_calc

Filters:
  - no missing raw values
  - raw values >= 0
  - TAM_POB > 0
  - DILOCCON50 > 0
  - no NaN in normalized predictors

Output:
  ../data/itur_difference_regression_dataset.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


# ----------------------------
# Modelo
# ----------------------------
ALPHA_1, ALPHA_2 = 0.56957, 0.43043
BETA_1, BETA_2 = 0.49964, 0.50036

GAMMA_1 = 0.08577
GAMMA_2 = 0.30994
GAMMA_3 = 0.12458
GAMMA_4 = 0.11860
GAMMA_5 = 0.17644
GAMMA_6 = 0.18467


def _log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------
# Normalizaciones
# ----------------------------
def L(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    out = pd.Series(np.zeros(len(x)), index=x.index)
    m = x > 0
    out.loc[m] = np.log(x.loc[m])
    return out


def n_minmax(x: pd.Series, xmin: float, xmax: float) -> pd.Series:
    denom = xmax - xmin
    if denom == 0 or not np.isfinite(denom):
        return pd.Series(np.nan, index=x.index)
    return (x - xmin) / denom


def eta_minmax_inv(x: pd.Series, xmin: float, xmax: float) -> pd.Series:
    denom = xmax - xmin
    if denom == 0 or not np.isfinite(denom):
        return pd.Series(np.nan, index=x.index)
    return (xmax - x) / denom


def col_minmax_params(s: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce")
    return float(np.nanmin(s)), float(np.nanmax(s))


# ----------------------------
# Compute ITUR
# ----------------------------
def compute_itur(df: pd.DataFrame, verbose=True):

    df = df.copy()

    if "P_USOSUECON" not in df.columns and "USO_SUECON" in df.columns:
        df = df.rename(columns={"USO_SUECON": "P_USOSUECON"})

    raw_vars = [
        "P_USOSUEPV", "COND_ACCE", "P_USOSUECON", "EQUIP_URB",
        "DILOCCON50", "TAM_POB", "DEN_POB", "CAR_SER_VI", "RESUL_ITUR"
    ]

    for c in raw_vars:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # strict filtering
    mask_valid = df[raw_vars].notna().all(axis=1)
    mask_nonneg = (df[raw_vars] >= 0).all(axis=1)
    mask_tam_positive = df["TAM_POB"] > 0
    mask_diloc_positive = df["DILOCCON50"] > 0

    df = df.loc[
        mask_valid
        & mask_nonneg
        & mask_tam_positive
        & mask_diloc_positive
    ].copy()

    if verbose:
        _log(f"Rows after strict filtering: {len(df):,}")

    # logs
    D_log = L(df["DILOCCON50"])
    T_log = L(df["TAM_POB"])
    P_log = L(df["DEN_POB"])

    # min/max
    pu_xmin, pu_xmax = col_minmax_params(df["P_USOSUEPV"])
    ca_xmin, ca_xmax = col_minmax_params(df["COND_ACCE"])
    ps_xmin, ps_xmax = col_minmax_params(df["P_USOSUECON"])
    eu_xmin, eu_xmax = col_minmax_params(df["EQUIP_URB"])
    D_xmin, D_xmax = col_minmax_params(D_log)
    T_xmin, T_xmax = col_minmax_params(T_log)
    P_xmin, P_xmax = col_minmax_params(P_log)

    # hats
    df["hat_P_USOSUEPV"]  = n_minmax(df["P_USOSUEPV"], pu_xmin, pu_xmax)
    df["hat_COND_ACCE"]   = n_minmax(df["COND_ACCE"], ca_xmin, ca_xmax)
    df["hat_P_USOSUECON"] = eta_minmax_inv(df["P_USOSUECON"], ps_xmin, ps_xmax)
    df["hat_EQUIP_URB"]   = eta_minmax_inv(df["EQUIP_URB"], eu_xmin, eu_xmax)
    df["hat_DILOCCON50"]  = n_minmax(D_log, D_xmin, D_xmax)
    df["hat_TAM_POB"]     = eta_minmax_inv(T_log, T_xmin, T_xmax)
    df["hat_DEN_POB"]     = eta_minmax_inv(P_log, P_xmin, P_xmax)
    df["hat_CAR_SER_VI"]  = df["CAR_SER_VI"]

    # ITUR components
    df["demografica"] = (BETA_1 * df["hat_TAM_POB"]) + (BETA_2 * df["hat_DEN_POB"])
    df["territorial"] = (
        GAMMA_1 * df["hat_EQUIP_URB"]
        + GAMMA_2 * df["hat_COND_ACCE"]
        + GAMMA_3 * df["hat_P_USOSUECON"]
        + GAMMA_4 * df["hat_P_USOSUEPV"]
        + GAMMA_5 * df["hat_DILOCCON50"]
        + GAMMA_6 * df["hat_CAR_SER_VI"]
    )

    df["ITUR_calc"] = (ALPHA_1 * df["territorial"]) + (ALPHA_2 * df["demografica"])

    return df


# ----------------------------
# Export regression dataset
# ----------------------------
def export_regression_dataset(in_csv, out_csv):

    df = pd.read_csv(in_csv)
    df2 = compute_itur(df)

    feature_cols = [
        "hat_P_USOSUEPV", "hat_COND_ACCE", "hat_P_USOSUECON",
        "hat_EQUIP_URB", "hat_DILOCCON50",
        "hat_TAM_POB", "hat_DEN_POB", "hat_CAR_SER_VI"
    ]

    # remove rows with NaN in hats
    df2 = df2.loc[~df2[feature_cols + ["ITUR_calc", "RESUL_ITUR"]].isna().any(axis=1)].copy()

    # signed difference
    df2["diff"] = df2["RESUL_ITUR"] - df2["ITUR_calc"]

    out = df2[feature_cols + ["diff"]].copy()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(f"Wrote regression dataset: {out_csv}")


if __name__ == "__main__":
    IN_CSV = "/mnt/data-r1/data/ITUR/2026.02.11_ITUR_resultados_nacional_v1_shapefile.csv"
    OUT_CSV = "../data/itur_difference_regression_dataset.csv"

    export_regression_dataset(IN_CSV, OUT_CSV)