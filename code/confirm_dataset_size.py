from pathlib import Path
import pandas as pd

image_dir = Path("/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A")
csv_file = "/mnt/data-r1/data/ITUR/ITUR_resultados_nacional_v1.csv"
cod_col = "COD_CELDA"
itur_col = "RESUL_ITUR"   # or whatever column you actually use

# 1) How many TIFFs?
tifs = {p.stem for p in list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff"))}
print("TIFF files:", len(tifs))

# 2) From CSV: codes present, numeric & non-NaN ITUR
df = pd.read_csv(csv_file)
print("CSV rows:", len(df))
df_num = df[pd.to_numeric(df[itur_col], errors="coerce").notna()].copy()
df_num[itur_col] = pd.to_numeric(df_num[itur_col], errors="coerce")
print("CSV rows with numeric ITUR:", len(df_num))

# 3) Duplicates on COD_CELDA (these will be skipped)
dup_mask = df_num.duplicated(subset=[cod_col], keep=False)
print("Duplicate COD_CELDA rows (will be dropped):", dup_mask.sum())

# 4) Unique, single-match codes with numeric ITUR
single = df_num[~dup_mask]
codes_csv = set(single[cod_col].astype(str))
print("Unique single-match CSV codes with numeric ITUR:", len(codes_csv))

# 5) Final usable = intersection of TIFF stems and CSV codes
usable = tifs & codes_csv
print("Usable matched samples:", len(usable))
