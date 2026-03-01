#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # pick your GPU

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
TIF_DIR         = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/AlphaEarth_Aguascalientes"
LABEL_CSV       = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/AlphaEarth_Aguascalientes_ITUR.csv"
OUT_DIR         = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/results_regression_from_tifs"
BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_resnet50_regressor_full.pth")

SEED         = 42
BATCH_SIZE   = 256
EPOCHS       = 10000           # use early stopping
LR           = 1e-4
MAX_LR       = 1e-3
WEIGHT_DECAY = 1e-2
PRINT_EVERY  = 1

PROPORTION   = 1.0
NUM_WORKERS  = 0
PIN_MEMORY   = True
PERSISTENT   = NUM_WORKERS > 0

# Model / input
IN_CHANS     = 64             # AlphaEarth has 64 bands
IMG_SIZE     = 224            # ResNet50 in timm works fine with 224
RESNET_NAME  = "resnet50"

# Early stopping
PATIENCE   = 10
MIN_DELTA  = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed)
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
# Dataset
# ----------------------------
class AlphaEarthTIFRegDataset(Dataset):
    def __init__(self, df_subset, tif_dir=TIF_DIR, scan_check=True, resize=IMG_SIZE):
        self.df = df_subset.reset_index(drop=True).copy()
        self.tif_dir = tif_dir
        self.resize = resize

        if scan_check:
            missing = wrong_bands = unreadable = 0
            keep_mask = np.ones(len(self.df), dtype=bool)
            for i, row in enumerate(tqdm(self.df.itertuples(), total=len(self.df), desc="Scanning TIFs", leave=False)):
                code = str(getattr(row, "COD_CELDA"))
                tif_path = os.path.join(self.tif_dir, f"AlphaEarth_{code}.tif")
                if not os.path.exists(tif_path):
                    keep_mask[i] = False; missing += 1; continue
                try:
                    with rasterio.open(tif_path) as src:
                        if src.count != IN_CHANS:
                            keep_mask[i] = False; wrong_bands += 1; continue
                except Exception:
                    keep_mask[i] = False; unreadable += 1
            if not keep_mask.all():
                self.df = self.df[keep_mask].reset_index(drop=True)
            kept = len(self.df)
            if kept == 0:
                raise RuntimeError(
                    f"No valid samples. missing={missing}, wrong_bands={wrong_bands}, unreadable={unreadable}"
                )
            print(f"TIF scan kept={kept} | missing={missing} | wrong_bands={wrong_bands} | unreadable={unreadable}")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        code = str(row["COD_CELDA"])
        y = float(row["RESUL_ITUR"])

        tif_path = os.path.join(self.tif_dir, f"AlphaEarth_{code}.tif")
        with rasterio.open(tif_path) as src:
            img = src.read()  # (C,H,W)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # per-image, per-channel min-max normalization
        img = per_channel_minmax(img)

        # resize to 224 for ResNet
        img_t = torch.from_numpy(img)  # (C,H,W)
        img_t = F.interpolate(img_t.unsqueeze(0), size=(self.resize, self.resize),
                              mode='bilinear', align_corners=False).squeeze(0)  # (C,224,224)

        return img_t, torch.tensor(y, dtype=torch.float32)

# ----------------------------
# Eval
# ----------------------------
@torch.no_grad()
def eval_regressor(model, loader, device, criterion):
    model.eval()
    preds, trues = [], []
    val_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        yhat = model(xb).squeeze(-1)
        loss = criterion(yhat, yb)
        val_loss += loss.item()
        preds.append(yhat.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    trues = np.concatenate(trues) if trues else np.array([])
    r2 = r2_score(trues, preds) if len(trues) > 0 else float("nan")
    val_mse = val_loss / max(1, len(loader))
    return val_mse, r2, preds, trues

# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(LABEL_CSV):
        raise FileNotFoundError(f"LABEL_CSV not found: {LABEL_CSV}")
    df = pd.read_csv(LABEL_CSV)
    needed = {"COD_CELDA", "RESUL_ITUR"}
    if not needed.issubset(df.columns) if hasattr(set, "issubset") else not needed.issubset(set(df.columns)):
        # backward safe check
        if not needed.issubset(df.columns):
            raise ValueError(f"CSV must contain columns {needed}, has {set(df.columns)}")

    df = subsample_df(df, PROPORTION, seed=SEED)

    full_ds = AlphaEarthTIFRegDataset(df, tif_dir=TIF_DIR, scan_check=True, resize=IMG_SIZE)
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

    # ResNet50 regression head
    model = timm.create_model(RESNET_NAME, in_chans=IN_CHANS, pretrained=True)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    else:
        # fallback for unusual variants
        model.reset_classifier(num_classes=1)
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    steps_per_epoch = max(1, len(train_loader))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        anneal_strategy='linear'
    )

    best_val_loss = float("inf")
    best_state    = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train {epoch}/{EPOCHS}", leave=False):
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        tr_mse = total_loss / max(1, len(train_loader))
        val_mse, val_r2, _, _ = eval_regressor(model, val_loader, DEVICE, criterion)

        if epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch:04d} | Train MSE {tr_mse:.6f} | Val MSE {val_mse:.6f} | Val R2 {val_r2:.4f} | LR {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping on validation loss
        if val_mse < best_val_loss - MIN_DELTA:
            best_val_loss = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val-loss improvement for {PATIENCE} epochs). Best Val MSE: {best_val_loss:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    test_mse, test_r2, test_preds, test_trues = eval_regressor(model, test_loader, DEVICE, criterion)
    print(f"\nTest R²: {test_r2:.4f} | Test MSE: {test_mse:.6f}")

    # Save
    pd.DataFrame(
        [{"metric": "test_R2", "value": float(test_r2)},
         {"metric":"test_MSE","value": float(test_mse)}]
    ).to_csv(os.path.join(OUT_DIR, "regression_test_metrics_resnet50_full.csv"), index=False)

    pd.DataFrame({"y_true": test_trues, "y_pred": test_preds}).to_csv(
        os.path.join(OUT_DIR, "regression_test_predictions_resnet50_full.csv"), index=False
    )

    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f"Saved test metrics and predictions to {OUT_DIR}")
    print(f"Best ResNet50 regressor weights saved to {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()
