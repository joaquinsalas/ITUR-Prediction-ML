#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# -------------------------------------------------------------------
# 1. Archivos de entrada / salida
# -------------------------------------------------------------------
archivo_entrada = "/mnt/data-r1/data/ITUR/ITUR_resultados_nacional_v1_area.csv"
archivo_salida  = "../data/ITUR_resultados_nacional_v1_con_TAM_POB_calc_ITUR.csv"


# In[3]:


# -------------------------------------------------------------------
# 2. Cargar datos
# -------------------------------------------------------------------
df = pd.read_csv(archivo_entrada)

# Columnas necesarias
cols_req = [
    "DILOCCON50",
    "USO_SUECON",
    "COND_ACCE",
    "EQUIP_URB",
    "P_USOSUEPV",
    "CAR_SER_VI",
    "RESUL_ITUR",
    "area_km2",
]

faltantes = [c for c in cols_req if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas en el CSV: {faltantes}")


# Remove rows where DILOCCON50 is negative
df = df[df["DILOCCON50"] >= 0].copy()



# In[4]:


# -------------------------------------------------------------------
# 3. DILOCCON50
# -------------------------------------------------------------------
df["DILOCCON50_safe"] = df["DILOCCON50"].replace(0, 1)  # evitar log(0)
df["DILOCCON50_log"] = np.log(df["DILOCCON50_safe"])

xmin = df["DILOCCON50_log"].min()
xmax = df["DILOCCON50_log"].max()
df["DILOCCON50_norm"] = (df["DILOCCON50_log"] - xmin) / (xmax - xmin)

df.drop(columns=["DILOCCON50_safe"], inplace=True)


# In[5]:


# -------------------------------------------------------------------
# 4. USO_SUECON (uso de suelo construido)
# -------------------------------------------------------------------
xmin = df["USO_SUECON"].min()
xmax = df["USO_SUECON"].max()
df["USO_SUECON_norm"] = (xmax - df["USO_SUECON"]) / (xmax - xmin)


# In[6]:


# -------------------------------------------------------------------
# 5. COND_ACCE (condiciones de accesibilidad):
# -------------------------------------------------------------------
xmin = df["COND_ACCE"].min()
xmax = df["COND_ACCE"].max()
df["COND_ACCE_norm"] = (df["COND_ACCE"] - xmin) / (xmax - xmin)


# In[7]:


# -------------------------------------------------------------------
# 6. EQUIP_URB (equipamiento urbano):
# -------------------------------------------------------------------
xmin = df["EQUIP_URB"].min()
xmax = df["EQUIP_URB"].max()
df["EQUIP_URB_norm"] = (xmax - df["EQUIP_URB"]) / (xmax - xmin)


# In[8]:


# -------------------------------------------------------------------
# 7. Cálculo de la dimensión territorial
# -------------------------------------------------------------------
df["territorial"] = (
    0.08577 * df["EQUIP_URB_norm"] +
    0.30994 * df["COND_ACCE_norm"] +
    0.12458 * df["USO_SUECON_norm"] +
    0.11860 * df["P_USOSUEPV"] +        
    0.17644 * df["DILOCCON50_norm"] +
    0.18467 * df["CAR_SER_VI"]          
)       


# In[9]:


# -------------------------------------------------------------------
# 8. Cálculo de TAM_POB_calc con la fórmula invertida
# -------------------------------------------------------------------
df["area_safe"] = df["area_km2"].replace(0, 1e-6)

df["TAM_POB_calc"] = (
    (df["RESUL_ITUR"] - 0.56957 * df["territorial"]) /
    (0.2151 + 0.2154 / df["area_safe"])
)

df.drop(columns=["area_safe"], inplace=True)


# In[10]:


# -------------------------------------------------------------------
# 9. Guardar resultado
# -------------------------------------------------------------------
df.to_csv(archivo_salida, index=False)
print(f"Archivo guardado como: {archivo_salida}")

