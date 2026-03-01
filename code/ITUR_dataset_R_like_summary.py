#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R-like summary() for:
 /mnt/data-r1/data/ITUR/2026.02.11_ITUR_resultados_nacional_v1_shapefile.csv

For numeric columns:
  Min, 1st Qu., Median, Mean, 3rd Qu., Max, NA's

For non-numeric columns:
  Counts of unique values (top 10) + NA's
"""

import pandas as pd
import numpy as np


INPUT_CSV = "/mnt/data-r1/data/ITUR/2026.02.11_ITUR_resultados_nacional_v1_shapefile.csv"


def numeric_summary(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    return pd.Series({
        "Min.":     np.nanmin(s_num),
        "1st Qu.":  np.nanpercentile(s_num, 25),
        "Median":   np.nanmedian(s_num),
        "Mean":     np.nanmean(s_num),
        "3rd Qu.":  np.nanpercentile(s_num, 75),
        "Max.":     np.nanmax(s_num),
        "NA's":     s_num.isna().sum()
    })


def categorical_summary(s: pd.Series, top_n: int = 10) -> pd.Series:
    counts = s.value_counts(dropna=True)
    top_counts = counts.head(top_n)
    summary_dict = {f"{idx}": val for idx, val in top_counts.items()}
    summary_dict["Unique"] = counts.size
    summary_dict["NA's"] = s.isna().sum()
    return pd.Series(summary_dict)


def r_like_summary(df: pd.DataFrame) -> None:
    print(f"\nRows: {len(df):,} | Columns: {df.shape[1]}\n")

    for col in df.columns:
        print(f"--- {col} ---")

        if pd.api.types.is_numeric_dtype(df[col]):
            summ = numeric_summary(df[col])
        else:
            # Try numeric coercion to mimic R behavior
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() > 0:
                summ = numeric_summary(df[col])
            else:
                summ = categorical_summary(df[col])

        print(summ)
        print()


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    r_like_summary(df)