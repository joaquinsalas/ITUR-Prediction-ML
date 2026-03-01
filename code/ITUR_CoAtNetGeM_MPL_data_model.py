#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # pick your GPU

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.backends import cudnn
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
import rasterio
import timm
from torch.optim.lr_scheduler import OneCycleLR

# ----------------------------
# Config
# ----------------------------
TIF_DIR           = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/AlphaEarth_Aguascalientes"
LABEL_CSV         = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/AlphaEarth_Aguascalientes_ITUR.csv"
OUT_DIR           = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/results_regression_from_tifs"
BEST_MODEL_PATH   = os.path.join(OUT_DIR, "best_coatnet1_ITUR_from_vars.pth")

SEED         = 42
BATCH_SIZE   = 64
EPOCHS       = 50          # use early stopping
LR           = 1e-4
MAX_LR       = 1e-3
WEIGHT_DECAY = 1e-2
PRINT_EVERY  = 1

PROPORTION   = 1.0
NUM_WORKERS  = 0
PIN_MEMORY   = True
PERSISTENT   = NUM_WORKERS > 0

# Model / input
IN_CHANS      = 64             # AlphaEarth has 64 bands
IMG_SIZE      = 224            # CoAtNet rw_224 variants use 224
BACKBONE_NAME = "coatnet_1_rw_224"  # timm CoAtNet variant

# Early stopping
PATIENCE   = 100
MIN_DELTA  = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True

# ----------------------------
# ITUR decomposition constants
# ----------------------------
# Variable order in the CSV:
VAR_COLS = [
    "TAM_POB",
    "DEN_POB",
    "EQUIP_URB",
    "COND_ACCE",
    "USO_SUECON",
    "P_USOSUEPV",
    "DILOCCON50",
    "CAR_SER_VI",
]
N_VARS = len(VAR_COLS)

# ITUR = α1 * territorial + α2 * demográfica
ALPHA_1 = 0.56957
ALPHA_2 = 0.43043

# demográfica = β1 * TAM_POB + β2 * DEN_POB
BETA_1 = 0.49964
BETA_2 = 0.50036

# territorial = Σ γ_i * corresponding variable
GAMMA_1 = 0.08577   # EQUIP_URB
GAMMA_2 = 0.30994   # COND_ACCE
GAMMA_3 = 0.12458   # USO_SUECON (P_USOSUECON)
GAMMA_4 = 0.11860   # P_USOSUEPV
GAMMA_5 = 0.17644   # DILOCCON50
GAMMA_6 = 0.18467   # CAR_SER_VI


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subsample_df(df, proportion, seed=SEED):
    if not (0 < proportion <= 1):
        raise ValueError("PROPORTION must be in (0,1].")
    if proportion >= 1.0:
        return df.reset_index(drop=True)
    n = max(1, int(len(df) * proportion))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def per_channel_minmax(x, eps=1e-6):
    # x: (C,H,W) np.float32
    cmins = x.reshape(x.shape[0], -1).min(axis=1, keepdims=True)
    cmaxs = x.reshape(x.shape[0], -1).max(axis=1, keepdims=True)
    rng = np.maximum(cmaxs - cmins, eps)
    x2 = (x.reshape(x.shape[0], -1) - cmins) / rng
    return x2.reshape(x.shape).astype(np.float32)


# ----------------------------
# ITUR helpers (torch + numpy)
# ----------------------------
def compute_demografica_from_vars(vars_tensor):
    """
    vars_tensor: (B, N_VARS), order given by VAR_COLS
    Uses TAM_POB (idx 0) and DEN_POB (idx 1)
    """
    tam = vars_tensor[:, 0]
    den = vars_tensor[:, 1]
    dem = BETA_1 * tam + BETA_2 * den
    return dem


def compute_territorial_from_vars(vars_tensor):
    """
    vars_tensor: (B, N_VARS)
    Uses indices:
      EQUIP_URB   -> 2
      COND_ACCE   -> 3
      USO_SUECON  -> 4
      P_USOSUEPV  -> 5
      DILOCCON50  -> 6
      CAR_SER_VI  -> 7
    """
    equip   = vars_tensor[:, 2]
    cond    = vars_tensor[:, 3]
    pusecon = vars_tensor[:, 4]
    pusepv  = vars_tensor[:, 5]
    diloc   = vars_tensor[:, 6]
    carser  = vars_tensor[:, 7]

    territorial = (
        GAMMA_1 * equip +
        GAMMA_2 * cond +
        GAMMA_3 * pusecon +
        GAMMA_4 * pusepv +
        GAMMA_5 * diloc +
        GAMMA_6 * carser
    )
    return territorial


def compute_itur_from_vars_tensor(vars_tensor):
    """
    Torch version.
    vars_tensor: (B, N_VARS) in [0,1]
    Returns itur_h (B,)
    """
    dem = compute_demografica_from_vars(vars_tensor)
    terr = compute_territorial_from_vars(vars_tensor)
    itur_h = ALPHA_1 * terr + ALPHA_2 * dem
    return itur_h


def compute_itur_h_np(var_array):
    """
    NumPy version for dataset construction/debug.
    var_array: (N_VARS,) or (N_VARS,1)
    """
    tam = var_array[0]
    den = var_array[1]
    dem = BETA_1 * tam + BETA_2 * den

    equip   = var_array[2]
    cond    = var_array[3]
    pusecon = var_array[4]
    pusepv  = var_array[5]
    diloc   = var_array[6]
    carser  = var_array[7]

    terr = (
        GAMMA_1 * equip +
        GAMMA_2 * cond +
        GAMMA_3 * pusecon +
        GAMMA_4 * pusepv +
        GAMMA_5 * diloc +
        GAMMA_6 * carser
    )

    itur_h = ALPHA_1 * terr + ALPHA_2 * dem
    return float(itur_h)


# ----------------------------
# GeM pooling
# ----------------------------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # If already pooled (B,C), just return
        if x.dim() == 2:
            return x
        # Expect (B, C, H, W)
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=(2, 3)).pow(1.0 / self.p)
        return x


# ----------------------------
# Single CoAtNet: predicts 8 variables
# ----------------------------
class CoAtNetVarsRegressor(nn.Module):
    def __init__(self, backbone_name=BACKBONE_NAME, in_chans=IN_CHANS, hidden_dim=512, n_vars=N_VARS):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            in_chans=in_chans,
            pretrained=True,
            num_classes=0,
            global_pool=""
        )
        feat_dim = self.backbone.num_features
        self.gem = GeM()
        self.head_vars = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_vars)
            # sigmoid applied in forward to get [0,1]
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x)  # (B,C,H,W)
        pooled = self.gem(feats)                  # (B,C)
        raw_vars = self.head_vars(pooled)         # (B,N_VARS)
        vars_hat = torch.sigmoid(raw_vars)        # (B,N_VARS) in [0,1]
        return vars_hat


# ----------------------------
# Dataset
# ----------------------------
class AlphaEarthTIFITURDataset(Dataset):
    """
    Returns:
      img_t:      (C, H, W)
      itur_ref:   scalar (RESUL_ITUR)
      vars_true:  (N_VARS,)  [only for debug/analysis; not used in loss]
    """
    def __init__(self, df_subset, tif_dir=TIF_DIR, scan_check=True, resize=IMG_SIZE):
        self.df = df_subset.reset_index(drop=True).copy()
        self.tif_dir = tif_dir
        self.resize = resize

        if scan_check:
            missing = wrong_bands = unreadable = 0
            keep_mask = np.ones(len(self.df), dtype=bool)
            for i, row in enumerate(
                tqdm(self.df.itertuples(), total=len(self.df),
                     desc="Scanning TIFs", leave=False)
            ):
                code = str(getattr(row, "COD_CELDA"))
                tif_path = os.path.join(self.tif_dir, f"AlphaEarth_{code}.tif")
                if not os.path.exists(tif_path):
                    keep_mask[i] = False
                    missing += 1
                    continue
                try:
                    with rasterio.open(tif_path) as src:
                        if src.count != IN_CHANS:
                            keep_mask[i] = False
                            wrong_bands += 1
                            continue
                except Exception:
                    keep_mask[i] = False
                    unreadable += 1
            if not keep_mask.all():
                self.df = self.df[keep_mask].reset_index(drop=True)
            kept = len(self.df)
            if kept == 0:
                raise RuntimeError(
                    f"No valid samples. missing={missing}, "
                    f"wrong_bands={wrong_bands}, unreadable={unreadable}"
                )
            print(
                f"TIF scan kept={kept} | missing={missing} | "
                f"wrong_bands={wrong_bands} | unreadable={unreadable}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        code = str(row["COD_CELDA"])
        itur_ref = float(row["RESUL_ITUR"])

        vars_true = row[VAR_COLS].values.astype(np.float32)  # (N_VARS,)

        tif_path = os.path.join(self.tif_dir, f"AlphaEarth_{code}.tif")
        with rasterio.open(tif_path) as src:
            img = src.read()  # (C,H,W)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # per-image, per-channel min-max normalization
        img = per_channel_minmax(img)

        # resize to 224 for CoAtNet
        img_t = torch.from_numpy(img)  # (C,H,W)
        img_t = F.interpolate(
            img_t.unsqueeze(0), size=(self.resize, self.resize),
            mode='bilinear', align_corners=False
        ).squeeze(0)  # (C,224,224)

        return (
            img_t,
            torch.tensor(itur_ref, dtype=torch.float32),
            torch.from_numpy(vars_true),
        )


# ----------------------------
# Eval on ITUR (single model)
# ----------------------------
@torch.no_grad()
def eval_itur_regressor(model_vars, loader, device, criterion):
    model_vars.eval()
    preds_itur, trues_itur = [], []
    val_loss = 0.0

    for xb, itur_ref, vars_true in loader:
        xb       = xb.to(device, non_blocking=True)
        itur_ref = itur_ref.to(device, non_blocking=True)

        vars_hat = model_vars(xb)  # (B, N_VARS) in [0,1]

        # Compute ITUR from predicted vars via analytic formula
        itur_h_hat = compute_itur_from_vars_tensor(vars_hat)  # (B,)
        itur_pred  = torch.sigmoid(itur_h_hat)                # enforce [0,1]

        loss = criterion(itur_pred, itur_ref)
        val_loss += loss.item()

        preds_itur.append(itur_pred.detach().cpu().numpy())
        trues_itur.append(itur_ref.detach().cpu().numpy())

    preds_itur = np.concatenate(preds_itur) if preds_itur else np.array([])
    trues_itur = np.concatenate(trues_itur) if trues_itur else np.array([])

    # Filter non-finite
    mask_preds = np.isfinite(preds_itur)
    mask_trues = np.isfinite(trues_itur)
    mask = mask_preds & mask_trues
    if mask.sum() != len(preds_itur):
        print("Non-finite in eval arrays:")
        print("  bad preds:", (~mask_preds).sum(),
              "bad trues:", (~mask_trues).sum())

    preds_itur = preds_itur[mask]
    trues_itur = trues_itur[mask]

    if len(trues_itur) == 0:
        r2 = float("nan")
    else:
        r2 = r2_score(trues_itur, preds_itur)

    val_mse = val_loss / max(1, len(loader))
    return val_mse, r2, preds_itur, trues_itur


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(LABEL_CSV):
        raise FileNotFoundError(f"LABEL_CSV not found: {LABEL_CSV}")
    df = pd.read_csv(LABEL_CSV)

    # Remove missing ITUR or missing ITUR variable rows
    cols_required = ["RESUL_ITUR"]
    df = df.dropna(subset=cols_required).reset_index(drop=True)

    needed = {"COD_CELDA", "RESUL_ITUR"} | set(VAR_COLS)
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns {needed}, has {set(df.columns)}")

    df = subsample_df(df, PROPORTION, seed=SEED)

    full_ds = AlphaEarthTIFITURDataset(df, tif_dir=TIF_DIR, scan_check=True, resize=IMG_SIZE)
    n_total = len(full_ds)
    n_train = int(0.50 * n_total)
    n_val   = int(0.20 * n_total)
    n_test  = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=gen)

    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    if NUM_WORKERS > 0:
        dl_kwargs["persistent_workers"] = PERSISTENT
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    # Single CoAtNet model
    model_vars = CoAtNetVarsRegressor(backbone_name=BACKBONE_NAME, in_chans=IN_CHANS)
    model_vars = model_vars.to(DEVICE)

    optimizer = optim.AdamW(
        model_vars.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    mse_loss = nn.MSELoss()

    steps_per_epoch = max(1, len(train_loader))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        anneal_strategy='linear'
    )

    best_val_loss = float("inf")
    best_state_vars = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model_vars.train()
        total_loss = 0.0

        for xb, itur_ref, vars_true in tqdm(
            train_loader, desc=f"Train {epoch}/{EPOCHS}", leave=False
        ):
            xb       = xb.to(DEVICE, non_blocking=True)
            itur_ref = itur_ref.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward
            vars_hat = model_vars(xb)                   # (B,N_VARS) in [0,1]
            itur_h_hat = compute_itur_from_vars_tensor(vars_hat)  # (B,)
            itur_pred  = torch.sigmoid(itur_h_hat)      # (B,) in [0,1]

            loss_itur = mse_loss(itur_pred, itur_ref)

            if not torch.isfinite(loss_itur):
                print("Non-finite loss detected.")
                print("  loss_itur:", loss_itur.item())
                print("  vars_hat stats:",
                      vars_hat.min().item(), vars_hat.max().item())
                raise RuntimeError("Aborting due to non-finite loss")

            loss_itur.backward()
            torch.nn.utils.clip_grad_norm_(
                model_vars.parameters(),
                max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss_itur.item()

        tr_loss = total_loss / max(1, len(train_loader))
        val_mse, val_r2, _, _ = eval_itur_regressor(
            model_vars, val_loader, DEVICE, mse_loss
        )

        if epoch % PRINT_EVERY == 0:
            print(
                f"Epoch {epoch:04d} | Train loss {tr_loss:.6f} | "
                f"Val MSE (ITUR) {val_mse:.6f} | Val R2 (ITUR) {val_r2:.4f} | "
                f"LR {scheduler.get_last_lr()[0]:.6f}"
            )

        # Early stopping on validation ITUR MSE
        if val_mse < best_val_loss - MIN_DELTA:
            best_val_loss = val_mse
            best_state_vars = {
                k: v.detach().cpu().clone() for k, v in model_vars.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val-loss improvement for {PATIENCE} epochs). "
                    f"Best Val MSE (ITUR): {best_val_loss:.6f}"
                )
                break

    if best_state_vars is not None:
        model_vars.load_state_dict(best_state_vars)

    # Test
    test_mse, test_r2, test_itur_pred, test_itur_true = eval_itur_regressor(
        model_vars, test_loader, DEVICE, mse_loss
    )
    print(f"\nTest R² (ITUR): {test_r2:.4f} | Test MSE (ITUR): {test_mse:.6f}")

    # Save metrics
    pd.DataFrame(
        [
            {"metric": "test_R2_ITUR", "value": float(test_r2)},
            {"metric": "test_MSE_ITUR", "value": float(test_mse)},
        ]
    ).to_csv(
        os.path.join(OUT_DIR, "regression_test_metrics_coatnet1_ITUR_singleCoAtNet.csv"),
        index=False
    )

    # Save predictions (ITUR only)
    pd.DataFrame(
        {
            "ITUR_true": test_itur_true,
            "ITUR_pred": test_itur_pred,
        }
    ).to_csv(
        os.path.join(OUT_DIR, "regression_test_predictions_coatnet1_ITUR_singleCoAtNet.csv"),
        index=False
    )

    torch.save(model_vars.state_dict(), BEST_MODEL_PATH)

    print(f"Saved test metrics and predictions to {OUT_DIR}")
    print(f"Best CoAtNet1 ITUR-from-vars regressor weights saved to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
