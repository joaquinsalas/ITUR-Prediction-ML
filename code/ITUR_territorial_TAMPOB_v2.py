#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib_inline as plt

# ============================================================
# 1. LEER EL ARCHIVO
# ============================================================
archivo = "/mnt/data-r1/data/ITUR/ITUR_resultados_nacional_v1_area.csv"

df = pd.read_csv(archivo)

print("Archivo leído:", archivo)
print("Columnas del archivo:")
print(df.columns.tolist())

# Valores que representan "sin dato" (no se usan en min/máx)
valores_sin_dato = [-99.0, -9999.0]


# In[22]:


# ============================================================
# 1.b CONVERSIÓN PREVIA DE TAM_POB A FLOTANTE
#     - Crea TAM_POB_float
#     - Errores -> NaN -> se cambian a -9999
# ============================================================
col_origen = "TAM_POB"

df["TAM_POB_float"] = pd.to_numeric(df[col_origen], errors="coerce").fillna(-9999)

# Conteos de control
num_nan_float_after       = df["TAM_POB_float"].isna().sum()
num_menos99_float_after   = (df["TAM_POB_float"] == -99.0).sum()
num_menos9999_float_after = (df["TAM_POB_float"] == -9999.0).sum()

print("\n=== Conteos en TAM_POB_float DESPUÉS de fillna ===")
print(f"NaN:        {num_nan_float_after}")
print(f"-99.0:      {num_menos99_float_after}")
print(f"-9999.0:    {num_menos9999_float_after}")

print("\nVista rápida de TAM_POB y TAM_POB_float:")
print(df[["TAM_POB", "TAM_POB_float"]].head())


# In[23]:


# ============================================================
# 2. DILOCCON50  →  DILOCCON50_norm_log2  (min–max + log2)
# ============================================================
col = "DILOCCON50"
col_nueva = "DILOCCON50_norm"

print("\n------------------------------")
print("Procesando columna:", col)

df[col] = pd.to_numeric(df[col], errors="coerce")
mask_valid = df[col].notna() & ~df[col].isin(valores_sin_dato)

serie = df.loc[mask_valid, col]
X_min = serie.min()
X_max = serie.max()
rango = X_max - X_min

print("Mínimo válido:", X_min)
print("Máximo válido:", X_max)

df[col_nueva] = np.nan

if rango != 0:
    norm = (serie - X_min) / rango
    df.loc[mask_valid, col_nueva] = norm
else:
    print("Rango = 0, no se puede normalizar", col)

print(df[[col, col_nueva]].head())


# In[24]:


# ============================================================
# 3. P_USOSUEPV  →  P_USOSUEPV_norm  (min–max directo)
# ============================================================
col = "P_USOSUEPV"
col_nueva = "P_USOSUEPV_norm"

print("\n------------------------------")
print("Procesando columna:", col)

df[col] = pd.to_numeric(df[col], errors="coerce")
mask_valid = df[col].notna() & ~df[col].isin(valores_sin_dato)

serie = df.loc[mask_valid, col]
X_min = serie.min()
X_max = serie.max()
rango = X_max - X_min

print("Mínimo válido:", X_min)
print("Máximo válido:", X_max)

df[col_nueva] = np.nan

if rango != 0:
    df.loc[mask_valid, col_nueva] = (df.loc[mask_valid, col] - X_min) / rango
else:
    print("⚠ Rango = 0, no se puede normalizar", col)

print(df[[col, col_nueva]].head())


# In[25]:


# ============================================================
# 4. USO_SUECON  →  USO_SUECON_norm_inv  (min–max invertido)
#      mínimo → 1, máximo → 0
# ============================================================
col = "USO_SUECON"
col_nueva = "USO_SUECON_norm_inv"

print("\n------------------------------")
print("Procesando columna:", col)

df[col] = pd.to_numeric(df[col], errors="coerce")
mask_valid = df[col].notna() & ~df[col].isin(valores_sin_dato)

serie = df.loc[mask_valid, col]
X_min = serie.min()
X_max = serie.max()
rango = X_max - X_min

print("Mínimo válido:", X_min)
print("Máximo válido:", X_max)

df[col_nueva] = np.nan

if rango != 0:
    df.loc[mask_valid, col_nueva] = (X_max - df.loc[mask_valid, col]) / rango
else:
    print("Rango = 0, no se puede normalizar", col)

print(df[[col, col_nueva]].head())


# In[26]:


# ============================================================
# 5. COND_ACCE  →  COND_ACCE_norm  (min–max directo)
# ============================================================
col = "COND_ACCE"
col_nueva = "COND_ACCE_norm"

print("\n------------------------------")
print("Procesando columna:", col)

df[col] = pd.to_numeric(df[col], errors="coerce")
mask_valid = df[col].notna() & ~df[col].isin(valores_sin_dato)

serie = df.loc[mask_valid, col]
X_min = serie.min()
X_max = serie.max()
rango = X_max - X_min

print("Mínimo válido:", X_min)
print("Máximo válido:", X_max)

df[col_nueva] = np.nan

if rango != 0:
    df.loc[mask_valid, col_nueva] = (df.loc[mask_valid, col] - X_min) / rango
else:
    print("Rango = 0, no se puede normalizar", col)

print(df[[col, col_nueva]].head())


# In[27]:


# ============================================================
# 6. EQUIP_URB  →  EQUIP_URB_norm_inv  (min–max invertido)
# ============================================================
col = "EQUIP_URB"
col_nueva = "EQUIP_URB_norm_inv"

print("\n------------------------------")
print("Procesando columna:", col)

df[col] = pd.to_numeric(df[col], errors="coerce")
mask_valid = df[col].notna() & ~df[col].isin(valores_sin_dato)

serie = df.loc[mask_valid, col]
X_min = serie.min()
X_max = serie.max()
rango = X_max - X_min

print("Mínimo válido:", X_min)
print("Máximo válido:", X_max)

df[col_nueva] = np.nan

if rango != 0:
    df.loc[mask_valid, col_nueva] = (X_max - df.loc[mask_valid, col]) / rango
else:
    print("Rango = 0, no se puede normalizar", col)

print(df[[col, col_nueva]].head())


# In[28]:


# ============================================================
# 8. TAM_POB_float  →  TAM_POB_norm_inv  y  TAM_POB_norm_inv_log2
#      - min–max invertido
#      - luego log2(1 + x_norm_inv)
# ============================================================
col = "TAM_POB_float"          # usamos la versión flotante
col_norm_inv = "TAM_POB_norm_inv"
col_log2 = "TAM_POB_norm_inv_log2"

print("\n------------------------------")
print("Procesando columna:", col)

df[col] = pd.to_numeric(df[col], errors="coerce")
mask_valid = df[col].notna() & ~df[col].isin(valores_sin_dato)

serie = df.loc[mask_valid, col]
X_min = serie.min()
X_max = serie.max()
rango = X_max - X_min

print("Mínimo válido:", X_min)
print("Máximo válido:", X_max)

df[col_norm_inv] = np.nan
df[col_log2] = np.nan

if rango != 0:
    # Normalización min–max invertida
    df.loc[mask_valid, col_norm_inv] = (X_max - df.loc[mask_valid, col]) / rango
    # log2(1 + x_norm_inv)
    df.loc[mask_valid, col_log2] = np.log2(df.loc[mask_valid, col_norm_inv] + 1.0)
else:
    print("Rango = 0, no se puede normalizar", col)

print(df[["TAM_POB", "TAM_POB_float", col_norm_inv, col_log2]].head())

# (Opcional) ver algunos ejemplos con datos válidos de TAM_POB_float
ejemplo_validos = df[df["TAM_POB_float"].isin(valores_sin_dato) == False][
    ["TAM_POB", "TAM_POB_float", col_norm_inv, col_log2]
].head(10)

print("\n Ejemplos de filas con datos válidos de TAM_POB:")
print(ejemplo_validos)


# In[29]:


# ============================================================
# 9. CÁLCULO DE LA DIMENSIÓN TERRITORIAL  (MISMO df / MISMO ARCHIVO)
# ============================================================

# Dimensión territorial (según coeficientes que indicaste)
df["territorial"] = (
    0.08577 * df["EQUIP_URB_norm_inv"] +
    0.30994 * df["COND_ACCE_norm"] +
    0.12458 * df["USO_SUECON_norm_inv"] +
    0.11860 * df["P_USOSUEPV"] +
    0.17644 * df["DILOCCON50_norm"] +
    0.18467 * df["CAR_SER_VI"]      # CAR_SER_VI original, como en tu código
)

print("\nDescripción de la columna 'territorial':")
print(df["territorial"].describe())


# In[30]:


# ============================================================
# 10. HISTOGRAMA DE LA DIMENSIÓN TERRITORIAL (OPCIONAL)
# ============================================================
import matplotlib.pyplot as plt

print("\n📊 Histograma de la dimensión territorial")

col = "territorial"
serie = df[col].dropna()

print(f"\nDescripción estadística de {col} (solo valores válidos):")
print(serie.describe())

valor_min = serie.min()
valor_max = serie.max()
print(f"\nMínimo válido de {col}: {valor_min}")
print(f"Máximo válido de {col}: {valor_max}")

plt.figure(figsize=(8, 6))
plt.hist(serie, bins=100, edgecolor="black")
plt.title(f"Histograma de {col} (sin NaN)")
plt.xlabel(col)
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()


# In[32]:


# ============================================================
# 11. CÁLCULO DE TAM_POB_calc Y TAM_POB_calc_clip  (MISMO df)
# ============================================================

# Asegurar tipos numéricos
df["RESUL_ITUR"]  = pd.to_numeric(df["RESUL_ITUR"], errors="coerce")
df["territorial"] = pd.to_numeric(df["territorial"], errors="coerce")
df["area_km2"]    = pd.to_numeric(df["area_km2"], errors="coerce")

# Evitar división entre cero en el área
df["area_safe"] = df["area_km2"].replace(0, 1e-6)

mask_valid = (
    df["RESUL_ITUR"].notna() &
    df["territorial"].notna() &
    df["area_safe"].notna()
)

print("\nTotal registros:", len(df))
print("Registros válidos para calcular TAM_POB:", mask_valid.sum())

# Fórmula de TAM_POB del documento:
# TAM_POB = (ITUR - territorial * 0.56957) / (0.2151 + 0.2154/area)
df["TAM_POB_calc"] = np.nan
df.loc[mask_valid, "TAM_POB_calc"] = (
    (df.loc[mask_valid, "RESUL_ITUR"] - 0.56957 * df.loc[mask_valid, "territorial"]) /
    (0.2151 + 0.2154 / df.loc[mask_valid, "area_safe"])
)

# Recorte al rango [0, 1]
df["TAM_POB_calc_clip"] = df["TAM_POB_calc"].clip(lower=0, upper=1)

print("\nDescripción de TAM_POB_calc (sin clip):")
print(df["TAM_POB_calc"].dropna().describe())

print("\nDescripción de TAM_POB_calc_clip (recortada a [0,1]):")
print(df["TAM_POB_calc_clip"].dropna().describe())

print("\nPrimeros 10 registros (ITUR, territorial, área, calc, clip):")
print(df[["RESUL_ITUR", "territorial", "area_km2",
          "TAM_POB_calc", "TAM_POB_calc_clip","TAM_POB_norm_inv_log2"]].head(10))

# Quitar columna auxiliar
df.drop(columns=["area_safe"], inplace=True)


# In[33]:


# ============================================================
# 12. GUARDAR TODO EN EL MISMO ARCHIVO ORIGINAL
# ============================================================

print("\nGuardando archivo actualizado con TODAS las columnas...")
df.to_csv(archivo, index=False)
print(f"✅ Archivo sobrescrito con 'territorial', 'TAM_POB_calc' y 'TAM_POB_calc_clip': {archivo}")


# In[ ]:





# In[34]:


archivo = "ITUR_resultados_nacional_v1_area.csv"

df = pd.read_csv(archivo)


# In[35]:


df


# In[36]:


# Suponiendo que ya tienes el DataFrame df cargado
col = "TAM_POB_norm_inv_log2"

# Máscara: True donde la columna NO es NaN
mask_no_nan = df[col].notna()

# DataFrame filtrado
df_sin_nan = df[mask_no_nan]

print(df_sin_nan[[col]].head())
print("Número de filas sin NaN en", col, ":", len(df_sin_nan))


# In[37]:


df_sin_nan.to_csv("ITUR_TAM_POB_norm_inv_log2_sin_nan.csv", index=False)
print("✅ Archivo guardado: ITUR_TAM_POB_norm_inv_log2_sin_nan.csv")


# In[ ]:




