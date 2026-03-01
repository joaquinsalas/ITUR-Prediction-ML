import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# 1. Load the data
# -------------------------------------------------------------------
file_path = Path("/mnt/data-r1/data/ITUR/ITUR_resultados_nacional_v1.csv")
df = pd.read_csv(file_path)

# -------------------------------------------------------------------
# 2. Basic shape information
# -------------------------------------------------------------------
n_rows, n_cols = df.shape
print(f"\nRows: {n_rows:,d}")
print(f"Columns ({n_cols}): {list(df.columns)}")

# -------------------------------------------------------------------
# 3. Helper that classifies each column and computes the right stats
# -------------------------------------------------------------------
def summarize_column(col_series: pd.Series) -> dict:
    name = col_series.name
    missing = col_series.isna().sum()

    if pd.api.types.is_numeric_dtype(col_series):
        col_type = "continuous"
        out = {
            "variable" : name,
            "type"     : col_type,
            "missing"  : missing,
            "min"      : col_series.min(),
            "max"      : col_series.max(),
            "mean"     : col_series.mean(),
        }
    else:
        col_type = "discrete"
        value_counts = col_series.value_counts(dropna=False)
        out = {
            "variable"   : name,
            "type"       : col_type,
            "missing"    : missing,
            "n_unique"   : value_counts.shape[0],
            "value_counts" : value_counts.to_dict(),
        }
    return out

# -------------------------------------------------------------------
# 4. Build the summary for every column
# -------------------------------------------------------------------
summary = [summarize_column(df[col]) for col in df.columns]

# nice tabular view for a quick glance
summary_df = pd.DataFrame(summary)
print("\n===== SUMMARY =====")
print(summary_df[["variable", "type", "missing", "n_unique" if "n_unique" in summary_df else "min"]])

# -------------------------------------------------------------------
# 5. (Optional) export the full summary—including the full value-counts
#    dicts—to a JSON file for later inspection.
# -------------------------------------------------------------------
summary_df.to_json("ITUR_basic_summary.json", orient="records", indent=2)
print("\nA detailed JSON file has been written to ITUR_basic_summary.json")
