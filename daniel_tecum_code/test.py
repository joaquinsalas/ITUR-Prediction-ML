 # satellite_xgb_target.py ------------------------------------------------
import os, platform, pickle, warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"          # usa la GPU 2

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import rasterio
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from tqdm.auto import tqdm        # ya no hace falta pin_memory=True aquí
import argparse

# ------------------- Argument parser for mode selection -------------------
# parser = argparse.ArgumentParser(description="Train XGBoost on Sentinel-2/ITUR data.")
# parser.add_argument('--mode', choices=['legacy', 'prematched'], default='legacy',
#                     help='Choose data loading mode: legacy (COD_CELDA/ITUR_resultados_nacional_v1.csv) or prematched (image_path/itur_value CSV)')
# parser.add_argument('--csv', type=str, default=None, help='Path to input CSV file')
# parser.add_argument('--img_dir', type=str, default=None, help='Path to image directory (legacy mode only)')
# parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to use')
# parser.add_argument('--split', type=str, default='0.6,0.2,0.2', help='Train/val/test split proportions, comma-separated (e.g., 0.6,0.2,0.2)')
# args = parser.parse_args()

# # ------------------- rutas -------------------------------------------
# if args.mode == 'legacy':
#     if platform.system() == "Windows":
#         IMG_DIR  = r"E:\sentinel_images\BaseDatos_Sentinel2A"
#         CSV_IN   = r"E:\ITUR_resultados_nacional_v1.csv"
#         MODEL_DIR= r"E:\xgb_models"
#     else:
#         IMG_DIR  = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
#         CSV_IN   = "ITUR_resultados_nacional_v1.csv"
#         MODEL_DIR= "../outputs/models"
#     if args.csv is not None:
#         CSV_IN = args.csv
#     if args.img_dir is not None:
#         IMG_DIR = args.img_dir
# else:
#     # Pre-matched mode
#     CSV_IN = args.csv if args.csv is not None else "train_matched_data600000.csv"
#     MODEL_DIR = "../outputs/models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# # ------------------- 1. Cargar meta-datos y objetivo -----------------
# if args.mode == 'legacy':
#     df = pd.read_csv(CSV_IN)
#     # Ensure only rows with available images are used
#     df = df[df['COD_CELDA'].apply(lambda x: os.path.exists(f"{IMG_DIR}/{x}.tif"))].reset_index(drop=True)
#     # Normalize ITUR value to [0,1] if not already
#     itur_min, itur_max = df['RESUL_ITUR'].min(), df['RESUL_ITUR'].max()
#     df['target'] = (df['RESUL_ITUR'] - itur_min) / (itur_max - itur_min)
# else:
#     df = pd.read_csv(CSV_IN)
#     # If 'itur_value' is not in [0,1], normalize it
#     if df['itur_value'].max() > 1.0 or df['itur_value'].min() < 0.0:
#         itur_min, itur_max = df['itur_value'].min(), df['itur_value'].max()
#         df['target'] = (df['itur_value'] - itur_min) / (itur_max - itur_min)
#     else:
#         df['target'] = df['itur_value']

# # --------- Limit number of images if requested ---------
# if args.max_images is not None and len(df) > args.max_images:
#     df = df.sample(n=args.max_images, random_state=42).reset_index(drop=True)

# # --------- Split into train/val/test ---------
# split_props = [float(x) for x in args.split.split(',')]
# if not np.isclose(sum(split_props), 1.0):
#     raise ValueError(f"Split proportions must sum to 1.0, got {split_props}")
# train_prop, val_prop, test_prop = split_props

# df_trainval, df_test = train_test_split(df, test_size=test_prop, random_state=42)
# val_relative = val_prop / (train_prop + val_prop)
# df_train, df_val = train_test_split(df_trainval, test_size=val_relative, random_state=42)

# print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# ------------------- 2. Dataset de Sentinel-2 ------------------------
def load_image(path):
    with rasterio.open(path) as src:
        return src.read()                          # (bands,H,W)

class SatelliteDatasetLegacy(Dataset):
    def __init__(self, meta, img_dir):
        self.meta, self.img_dir = meta, img_dir
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = load_image(f"{self.img_dir}/{row['COD_CELDA']}.tif") * 5e-5
        img = img[:, 24:72:2, 24:72:2]
        blue, green, red, nir, sw1, sw2 = img[:6]
        eps=1e-8
        idx_stack = np.stack([
            (nir-red)/(nir+red+eps),
            2.5*(nir-red)/(nir+6*red-7.5*blue+1+eps),
            (green-nir)/(green+nir+eps),
            (sw1-nir)/(sw1+nir+eps),
            1.5*(nir-red)/(nir+red+0.5+eps),
            (nir-sw2)/(nir+sw2+eps),
            2.5*(nir-red)/(nir+2.4*red+1+eps),
            (2*nir+1-np.sqrt((2*nir+1)**2-8*(nir-red)))/2,
            (nir-(sw1-sw2))/(nir+(sw1-sw2)+eps),
            (red-blue)/(red+blue+eps),
            (blue+green+red)/3
        ])
        feat = np.concatenate([img, idx_stack], axis=0).astype(np.float32)
        return feat.ravel(), row["target"]

class SatelliteDatasetPrematched(Dataset):
    def __init__(self, meta):
        self.meta = meta
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = load_image(row['image_path']) * 5e-5
        img = img[:, 24:72:2, 24:72:2]
        blue, green, red, nir, sw1, sw2 = img[:6]
        eps=1e-8
        idx_stack = np.stack([
            (nir-red)/(nir+red+eps),
            2.5*(nir-red)/(nir+6*red-7.5*blue+1+eps),
            (green-nir)/(green+nir+eps),
            (sw1-nir)/(sw1+nir+eps),
            1.5*(nir-red)/(nir+red+0.5+eps),
            (nir-sw2)/(nir+sw2+eps),
            2.5*(nir-red)/(nir+2.4*red+1+eps),
            (2*nir+1-np.sqrt((2*nir+1)**2-8*(nir-red)))/2,
            (nir-(sw1-sw2))/(nir+(sw1-sw2)+eps),
            (red-blue)/(red+blue+eps),
            (blue+green+red)/3
        ])
        feat = np.concatenate([img, idx_stack], axis=0).astype(np.float32)
        return feat.ravel(), row["target"]

# All code outside of function/class definitions and the main block that references undefined variables has been removed. Only imports, class/function definitions, and the main block remain. All logic is inside run_xgboost_pipeline or other functions, and only called from the main block.


def run_xgboost_pipeline(
    mode='legacy',
    csv=None,
    img_dir=None,
    max_images=None,
    split='0.6,0.2,0.2',
    batch_size=256,
    n_iter=20,
    n_jobs=-1,
    random_state=42,
    model_dir=None
):
    """
    Run the full XGBoost pipeline: load data, sample, split, extract features, train, evaluate, and save.

    Parameters:
    - mode: 'legacy' or 'prematched'.
    - csv: Path to input CSV file.
    - img_dir: Path to image directory (legacy mode only).
    - max_images: Maximum number of images to use (None for all).
    - split: Comma-separated proportions for train, val, and test sets. Must sum to 1.0.
        Example values:
            '0.6,0.2,0.2'   # 60% train, 20% val, 20% test (default)
            '0.7,0.15,0.15' # 70% train, 15% val, 15% test
            '0.8,0.1,0.1'   # 80% train, 10% val, 10% test
        The script first splits off the test set, then splits the remainder into train and val.
    - batch_size: Batch size for feature extraction.
    - n_iter: Number of parameter settings sampled in RandomizedSearchCV.
    - n_jobs: Number of parallel jobs for RandomizedSearchCV.
    - random_state: Random seed for reproducibility.
    - model_dir: Directory to save the model and scaler.

    Returns:
        Dictionary with R2, best_params, itur_min, itur_max.
    """
    # Set up paths
    if mode == 'legacy':
        if platform.system() == "Windows":
            IMG_DIR  = r"E:\sentinel_images\BaseDatos_Sentinel2A"
            CSV_IN   = r"E:\ITUR_resultados_nacional_v1.csv"
            MODEL_DIR= r"E:\xgb_models"
        else:
            IMG_DIR  = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A/"
            CSV_IN   = "ITUR_resultados_nacional_v1.csv"
            MODEL_DIR= "../outputs/models"
        if csv is not None:
            CSV_IN = csv
        if img_dir is not None:
            IMG_DIR = img_dir
    else:
        CSV_IN = csv if csv is not None else "train_matched_data600000.csv"
        MODEL_DIR = "../outputs/models"
    if model_dir is not None:
        MODEL_DIR = model_dir
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    if mode == 'legacy':
        df = pd.read_csv(CSV_IN)
        df = df[df['COD_CELDA'].apply(lambda x: os.path.exists(f"{IMG_DIR}/{x}.tif"))].reset_index(drop=True)
        itur_min, itur_max = df['RESUL_ITUR'].min(), df['RESUL_ITUR'].max()
        df['target'] = (df['RESUL_ITUR'] - itur_min) / (itur_max - itur_min)
    else:
        df = pd.read_csv(CSV_IN)
        if df['itur_value'].max() > 1.0 or df['itur_value'].min() < 0.0:
            itur_min, itur_max = df['itur_value'].min(), df['itur_value'].max()
            df['target'] = (df['itur_value'] - itur_min) / (itur_max - itur_min)
        else:
            df['target'] = df['itur_value']

    # Limit number of images
    if max_images is not None and len(df) > max_images:
        df = df.sample(n=max_images, random_state=random_state).reset_index(drop=True)

    # Split into train/val/test
    split_props = [float(x) for x in split.split(',')]
    if not np.isclose(sum(split_props), 1.0):
        raise ValueError(f"Split proportions must sum to 1.0, got {split_props}")
    train_prop, val_prop, test_prop = split_props
    from sklearn.model_selection import train_test_split
    df_trainval, df_test = train_test_split(df, test_size=test_prop, random_state=random_state)
    val_relative = val_prop / (train_prop + val_prop)
    df_train, df_val = train_test_split(df_trainval, test_size=val_relative, random_state=random_state)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Feature extraction
    def extract_features(df_split):
        if mode == 'legacy':
            dataset = SatelliteDatasetLegacy(df_split, IMG_DIR)
        else:
            dataset = SatelliteDatasetPrematched(df_split)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
        X_list, y_list = [], []
        for feats, target in tqdm(loader, desc="Extrayendo características", total=len(loader)):
            X_list.append(feats)
            y_list.append(target)
        X = torch.cat(X_list).numpy()
        y = torch.cat(y_list).numpy()
        return X, y

    X_train, y_train = extract_features(df_train)
    X_val, y_val = extract_features(df_val)
    X_test, y_test = extract_features(df_test)
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    print("Matriz X (train+val):", X_trainval.shape, "  vector y:", y_trainval.shape)
    print("Matriz X (test):", X_test.shape, "  vector y:", y_test.shape)

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_trainval)
    train_X = scaler.transform(X_trainval)
    test_X  = scaler.transform(X_test)
    train_y = y_trainval
    test_y = y_test

    # XGBoost hyperparameters
    param_dist = {
        "eta":              np.linspace(0.01, 0.3, 30),
        "max_depth":        np.arange(3, 11),
        "subsample":        np.linspace(0.5, 1.0, 20),
        "colsample_bytree": np.linspace(0.5, 1.0, 20),
        "gamma":            np.linspace(0, 1, 20),
        "min_child_weight": np.arange(1, 7)
    }
    from xgboost import XGBRegressor
    from sklearn.model_selection import RandomizedSearchCV
    base_model = XGBRegressor(
        n_estimators=100,
        objective="reg:squarederror",
        tree_method="gpu_hist", predictor="gpu_predictor"
    )
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        verbose=1,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        random_state=random_state)
    search.fit(train_X, train_y)
    best = search.best_estimator_
    best.fit(train_X, train_y)
    pred = best.predict(test_X)
    from sklearn.metrics import r2_score
    r2 = r2_score(test_y, pred)
    print(f"\n► Test R² = {r2:.4f}")
    import pickle
    with open(os.path.join(MODEL_DIR, "xgb_target.pkl"), "wb") as f:
        pickle.dump(best, f)
    with open(os.path.join(MODEL_DIR, "scaler_target.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    pd.DataFrame([{
        "r2_test": r2,
        **search.best_params_,
        "itur_min": itur_min,
        "itur_max": itur_max
    }]).to_csv(os.path.join('../outputs/', "xgb_spectral_target_summary.csv"), index=False)
    print("\nModel & summary saved in", MODEL_DIR)
    return {
        'r2': r2,
        'best_params': search.best_params_,
        'itur_min': itur_min,
        'itur_max': itur_max,
        'model_path': os.path.join(MODEL_DIR, "xgb_target.pkl"),
        'scaler_path': os.path.join(MODEL_DIR, "scaler_target.pkl")
    }

if __name__ == "__main__":

    IMAGE_DIR = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A"
    CSV_FILE = "ITUR_resultados_nacional_v1.csv"

    run_xgboost_pipeline(
        mode='legacy',
        csv=CSV_FILE,
        img_dir=IMAGE_DIR,
        max_images=10000,
        split='0.6,0.2,0.2',
        batch_size=256,
        n_iter=10,
        n_jobs=1
    )
