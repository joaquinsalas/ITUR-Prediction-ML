#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verifica si RESUL_ITUR coincide con el ITUR calculado a partir del modelo
y MUESTRA AVANCE del cálculo (lectura, normalización, cómputo, verificación y escritura).

Entrada:
  /mnt/data-r1/data/ITUR/2026.02.11_ITUR_resultados_nacional_v1_shapefile.csv

Salida:
  ../data/cumple.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scipy.optimize as opt

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


# ----------------------------
# Utilidades de progreso
# ----------------------------
def _pct(i: int, n: int) -> str:
    if n <= 0:
        return "0.0%"
    return f"{100.0 * i / n:5.1f}%"


def _log(msg: str) -> None:
    print(msg, flush=True)


def log_stage(stage: str) -> None:
    _log(f"\n=== {stage} ===")


# ----------------------------
# Normalizaciones
# ----------------------------
def L(x: pd.Series) -> pd.Series:
    """L(A)=ln(A) si A>0; 0 de lo contrario."""
    x = pd.to_numeric(x, errors="coerce")
    out = pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    m = x > 0
    out.loc[m] = np.log(x.loc[m].astype(float))
    return out


def n_minmax(x: pd.Series, xmin: float, xmax: float) -> pd.Series:
    """min-max estándar."""
    denom = (xmax - xmin)
    if not np.isfinite(denom) or denom == 0:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - xmin) / denom


def eta_minmax_inv(x: pd.Series, xmin: float, xmax: float) -> pd.Series:
    """min-max invertida."""
    denom = (xmax - xmin)
    if not np.isfinite(denom) or denom == 0:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (xmax - x) / denom


def col_minmax_params(s: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce")
    xmin = np.nanmin(s.to_numpy(dtype=float))
    xmax = np.nanmax(s.to_numpy(dtype=float))
    return float(xmin), float(xmax)


# ----------------------------
# Cálculo ITUR
# ----------------------------
def compute_itur(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        log_stage("Cálculo ITUR: preparación")

    # Mapear nombre: en ecuaciones es P_USOSUECON; en CSV viene como USO_SUECON
    if "P_USOSUECON" not in df.columns and "USO_SUECON" in df.columns:
        df = df.rename(columns={"USO_SUECON": "P_USOSUECON"}).copy()
        if verbose:
            _log("Renombre: USO_SUECON -> P_USOSUECON")
    else:
        df = df.copy()

    # Convertir numéricas
    num_cols = [
        "TAM_POB", "DEN_POB", "DILOCCON50", "CAR_SER_VI",
        "P_USOSUEPV", "P_USOSUECON", "COND_ACCE", "EQUIP_URB", "RESUL_ITUR"
    ]
    if verbose:
        _log("Convirtiendo columnas a numérico...")
    for i, c in enumerate(num_cols, 1):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if verbose:
            _log(f"  [{_pct(i, len(num_cols))}] {c}")
    # ----------------------------
    # Remove rows with negative values
    # ----------------------------
    if verbose:
        log_stage("Eliminando filas con valores negativos")

    vars_no_negative = [
        "TAM_POB",
        "DEN_POB",
        "DILOCCON50",
        "CAR_SER_VI",
        "P_USOSUEPV",
        "P_USOSUECON",
        "COND_ACCE",
        "EQUIP_URB",
    ]

    initial_rows = len(df)

    mask_negative = (df[vars_no_negative] < 0).any(axis=1)
    n_removed = int(mask_negative.sum())

    df = df.loc[~mask_negative].copy()

    if verbose:
        _log(f"Filas iniciales: {initial_rows:,}")
        _log(f"Filas eliminadas (alguna variable < 0): {n_removed:,}")
        _log(f"Filas restantes: {len(df):,}")

    # Logs
    if verbose:
        log_stage("Transformaciones log + parámetros min/max")
        _log("Calculando logs: DILOCCON50, TAM_POB, DEN_POB")
    D_log = L(df["DILOCCON50"])
    T_log = L(df["TAM_POB"])
    P_log = L(df["DEN_POB"])

    if verbose:
        _log("Calculando min/max (ignorando NaN) ...")

    pu_xmin, pu_xmax = col_minmax_params(df["P_USOSUEPV"])
    ca_xmin, ca_xmax = col_minmax_params(df["COND_ACCE"])
    ps_xmin, ps_xmax = col_minmax_params(df["P_USOSUECON"])
    eu_xmin, eu_xmax = col_minmax_params(df["EQUIP_URB"])
    D_xmin, D_xmax = col_minmax_params(D_log)
    T_xmin, T_xmax = col_minmax_params(T_log)
    P_xmin, P_xmax = col_minmax_params(P_log)

    if verbose:
        _log("Min/max usados:")
        _log(f"  P_USOSUEPV:   min={pu_xmin:.6g}  max={pu_xmax:.6g}")
        _log(f"  COND_ACCE:    min={ca_xmin:.6g}  max={ca_xmax:.6g}")
        _log(f"  P_USOSUECON:  min={ps_xmin:.6g}  max={ps_xmax:.6g}")
        _log(f"  EQUIP_URB:    min={eu_xmin:.6g}  max={eu_xmax:.6g}")
        _log(f"  log(DILOCCON50): min={D_xmin:.6g} max={D_xmax:.6g}")
        _log(f"  log(TAM_POB):    min={T_xmin:.6g} max={T_xmax:.6g}")
        _log(f"  log(DEN_POB):    min={P_xmin:.6g} max={P_xmax:.6g}")

    # Normalizadas
    if verbose:
        log_stage("Normalización (sombreros)")
        _log("Aplicando n() / eta() ...")

    df["hat_P_USOSUEPV"] = n_minmax(df["P_USOSUEPV"], pu_xmin, pu_xmax)
    if verbose: _log("  [ 16.7%] hat_P_USOSUEPV")

    df["hat_COND_ACCE"] = n_minmax(df["COND_ACCE"], ca_xmin, ca_xmax)
    if verbose: _log("  [ 33.3%] hat_COND_ACCE")

    df["hat_P_USOSUECON"] = eta_minmax_inv(df["P_USOSUECON"], ps_xmin, ps_xmax)
    if verbose: _log("  [ 50.0%] hat_P_USOSUECON")

    df["hat_EQUIP_URB"] = eta_minmax_inv(df["EQUIP_URB"], eu_xmin, eu_xmax)
    if verbose: _log("  [ 66.7%] hat_EQUIP_URB")

    df["hat_DILOCCON50"] = n_minmax(D_log, D_xmin, D_xmax)
    if verbose: _log("  [ 83.3%] hat_DILOCCON50")

    df["hat_TAM_POB"] = eta_minmax_inv(T_log, T_xmin, T_xmax)
    df["hat_DEN_POB"] = eta_minmax_inv(P_log, P_xmin, P_xmax)

    if verbose:
        _log("  [100.0%] hat_TAM_POB, hat_DEN_POB")

    # CAR_SER_VI ya en [0,1]
    df["hat_CAR_SER_VI"] = df["CAR_SER_VI"]

    if verbose:
        _log("  [OK] hat_CAR_SER_VI (sin normalización, ya en [0,1])")
        _log(f"        min={df['hat_CAR_SER_VI'].min():.6g}  max={df['hat_CAR_SER_VI'].max():.6g}")

    df["demografica"] = (BETA_1 * df["hat_TAM_POB"]) + (BETA_2 * df["hat_DEN_POB"])
    df["territorial"] = (
            GAMMA_1 * df["hat_EQUIP_URB"]
            + GAMMA_2 * df["hat_COND_ACCE"]
            + GAMMA_3 * df["hat_P_USOSUECON"]
            + GAMMA_4 * df["hat_P_USOSUEPV"]
            + GAMMA_5 * df["hat_DILOCCON50"]
            + GAMMA_6 * df["hat_CAR_SER_VI"]
    )

    if verbose:
        _log("  Territorial incluye:")
        _log("    γ6 * hat_CAR_SER_VI  ✔")

    df["ITUR_calc"] = (ALPHA_1 * df["territorial"]) + (ALPHA_2 * df["demografica"])

    if verbose:
        _log("OK: ITUR_calc calculado.")

    return df

import scipy.optimize as opt


# ----------------------------
# Clean dataset: keep only >=0 and no missing
# ----------------------------
def clean_dataset(df: pd.DataFrame, verbose=True) -> pd.DataFrame:

    df = df.copy()

    # --- Harmonize column name ---
    if "P_USOSUECON" not in df.columns and "USO_SUECON" in df.columns:
        df = df.rename(columns={"USO_SUECON": "P_USOSUECON"})
        if verbose:
            print("Renamed USO_SUECON -> P_USOSUECON")

    vars_model = [
        "P_USOSUEPV",
        "COND_ACCE",
        "P_USOSUECON",
        "EQUIP_URB",
        "DILOCCON50",
        "TAM_POB",
        "DEN_POB",
        "CAR_SER_VI",
        "RESUL_ITUR",
    ]

    # Check missing columns explicitly
    missing = [c for c in vars_model if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial = len(df)

    for c in vars_model:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mask_valid = df[vars_model].notna().all(axis=1)
    mask_nonneg = (df[vars_model] >= 0).all(axis=1)

    df = df.loc[mask_valid & mask_nonneg].copy()

    if verbose:
        print("\nDataset cleaned:")
        print(f"Initial rows: {initial:,}")
        print(f"Remaining rows: {len(df):,}")

    return df



def fit_all_variables(df, variables):

    df = df.copy()

    def objective(params):

        df_tmp = df.copy()

        # params = [a1,b1,a2,b2,...]
        for i, var in enumerate(variables):
            a = params[2*i]
            b = params[2*i + 1]
            df_tmp[var] = a + b * df_tmp[var]

        df_tmp = compute_itur(df_tmp, verbose=False)

        err = df_tmp["ITUR_calc"] - df_tmp["RESUL_ITUR"]
        return np.mean(err**2)

    # initial guess: a=0, b=1 for each variable
    x0 = []
    for _ in variables:
        x0.extend([0.0, 1.0])

    result = opt.minimize(objective, x0=x0)

    params_opt = result.x

    results = []
    for i, var in enumerate(variables):
        results.append({
            "variable": var,
            "a": params_opt[2*i],
            "b": params_opt[2*i+1]
        })

    return pd.DataFrame(results)


# ----------------------------
# Fit affine transform a + b x
# ----------------------------
def fit_affine_parameter(df, variable_name):

    df = df.copy()

    def objective(params):
        a, b = params
        df_tmp = df.copy()
        df_tmp[variable_name] = a + b * df_tmp[variable_name]
        df_tmp = compute_itur(df_tmp, verbose=False)

        err = df_tmp["ITUR_calc"] - df_tmp["RESUL_ITUR"]
        return np.mean(err**2)

    result = opt.minimize(objective, x0=[0.0, 1.0])

    a_opt, b_opt = result.x

    print(f"\nOptimal parameters for {variable_name}:")
    print(f"a = {a_opt}")
    print(f"b = {b_opt}")

    return a_opt, b_opt


# ----------------------------
# Extract close estimates
# ----------------------------
def extract_close(df, tol=1e-4):

    diff = np.abs(df["ITUR_calc"] - df["RESUL_ITUR"])
    mask = diff < tol

    df_close = df.loc[mask].copy()

    print(f"\nClose estimates (< {tol}): {len(df_close):,}")

    return df_close


def verify_and_write(in_csv, out_csv, verbose=True):

    df_original = pd.read_csv(in_csv)

    # Clean dataset
    df_original = clean_dataset(df_original, verbose=verbose)

    variables = [
        "P_USOSUEPV",
        "COND_ACCE",
        "P_USOSUECON",
        "EQUIP_URB",
        "DILOCCON50",
        "TAM_POB",
        "DEN_POB",
        "CAR_SER_VI",
    ]

    results = []
    all_close = []

    param_df = fit_all_variables(df_original, variables)
    param_df.to_csv("../data/fit_parameters.csv", index=False)

    # Apply optimal transformation
    df = df_original.copy()
    for _, row in param_df.iterrows():
        df[row["variable"]] = row["a"] + row["b"] * df[row["variable"]]

    df = compute_itur(df, verbose=False)

    df_close = extract_close(df, tol=1e-2)
    df_close.to_csv("../data/close_estimates.csv", index=False)



    print("\nFiles written:")
    print("  ../data/fit_parameters.csv")
    print("  ../data/close_estimates.csv")

if __name__ == "__main__":
    IN_CSV = "/mnt/data-r1/data/ITUR/2026.02.11_ITUR_resultados_nacional_v1_shapefile.csv"
    OUT_CSV = "../data/cumple.csv"

    verify_and_write(
        in_csv=IN_CSV,
        out_csv=OUT_CSV,
        verbose=True,
    )