#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # pick your GPU

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.backends import cudnn
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
import torch.multiprocessing as mp
import rasterio

# ----------------------------
# Config
# ----------------------------
TIF_DIR         = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/AlphaEarth_Aguascalientes"
LABEL_CSV       = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/AlphaEarth_Aguascalientes_ITUR.csv"
OUT_DIR         = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/results_regression_from_tifs_lt_tau"
BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_s5_regressor_lt_tau.pt")

TAU            = 0.878389   # use rows with RESUL_ITUR < TAU

SEED         = 42
BATCH_SIZE   = 128
EPOCHS       = 2000
LR           = 1e-4
WEIGHT_DECAY = 1e-2
PRINT_EVERY  = 1

# Keep this knob to run on a subset before splitting (0<PROPORTION<=1)
PROPORTION   = 1

NUM_WORKERS  = 0
PIN_MEMORY   = True
PERSISTENT   = NUM_WORKERS > 0

L_SEQ     = 64
D_INPUT   = 1
D_MODEL   = 1048
N_LAYERS  = 3
DROPOUT   = 0.10
PRENORM   = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

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
        raise ValueError("proportion must be in (0,1].")
    if proportion >= 1.0:
        return df.reset_index(drop=True)
    n = max(1, int(len(df) * proportion))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

# ----------------------------
# Dataset
# ----------------------------
class AlphaEarthTIFRegDataset(Dataset):
    def __init__(self, df_subset, tif_dir=TIF_DIR, scan_check=True):
        self.df = df_subset.reset_index(drop=True).copy()
        self.tif_dir = tif_dir

        if scan_check:
            missing = wrong_bands = unreadable = 0
            keep_mask = np.ones(len(self.df), dtype=bool)
            for i, row in enumerate(tqdm(self.df.itertuples(),
                                         total=len(self.df),
                                         desc="Scanning TIFs", leave=False)):
                code = str(getattr(row, "COD_CELDA"))
                tif_path = os.path.join(self.tif_dir, f"AlphaEarth_{code}.tif")
                if not os.path.exists(tif_path):
                    keep_mask[i] = False; missing += 1; continue
                try:
                    with rasterio.open(tif_path) as src:
                        if src.count != 64:
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
            arr = src.read()  # (64, H, W)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[0] != 64:
            raise ValueError(f"{tif_path} has {arr.shape[0]} bands; expected 64.")
        chan_means = arr.reshape(64, -1).mean(axis=1).astype(np.float32)
        seq = chan_means.reshape(L_SEQ, 1)
        return torch.from_numpy(seq), torch.tensor(y, dtype=torch.float32)

# ----------------------------
# Model
# ----------------------------
from s5 import S5Block

class S5Regressor(nn.Module):
    def __init__(self, d_in=1, d_model=D_MODEL, n_layers=N_LAYERS,
                 dropout=DROPOUT, prenorm=PRENORM):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_in, d_model)
        self.s5_layers = nn.ModuleList([
            S5Block(dim=d_model, state_dim=d_model, bidir=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 1)

        nn.init.kaiming_uniform_(self.encoder.weight, a=np.sqrt(5))
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x):
        x = self.encoder(x)
        for block, norm, drop in zip(self.s5_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z)
            z = block(z)
            z = drop(z)
            x = x + z
            if not self.prenorm:
                x = norm(x)
        x = x.mean(dim=1)
        y = self.head(x).squeeze(-1)
        return y

# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def eval_regressor(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        yhat = model(xb)
        preds.append(yhat.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    trues = np.concatenate(trues) if trues else np.array([])
    r2 = r2_score(trues, preds) if len(trues) > 0 else float("nan")
    return r2, preds, trues

# ----------------------------
# Train/Early stop and loaders
# ----------------------------
def make_loaders(dataset, batch_size=BATCH_SIZE):
    n_total = len(dataset)
    n_train = int(0.50 * n_total)
    n_val   = int(0.20 * n_total)
    n_test  = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    dl_kwargs = dict(batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    if NUM_WORKERS > 0:
        dl_kwargs["persistent_workers"] = PERSISTENT
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)
    return train_loader, val_loader, test_loader

def train_with_early_stopping(model, train_loader, val_loader, device,
                              epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                              patience=10, ckpt_path=None):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_r2 = -np.inf
    best_state  = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        tr_mse = total_loss / max(1, len(train_loader))

        val_r2, _, _ = eval_regressor(model, val_loader, device)
        if epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch:04d} | Train MSE {tr_mse:.6f} | Val R2 {val_r2:.4f}")

        improved = val_r2 > best_val_r2
        if improved:
            best_val_r2 = val_r2
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            if ckpt_path:
                torch.save(best_state, ckpt_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val R2={best_val_r2:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_r2, best_state

# ----------------------------
# Main (Single stage)
# ----------------------------
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load CSV and filter by TAU (< TAU) ---
    if not os.path.exists(LABEL_CSV):
        raise FileNotFoundError(f"LABEL_CSV not found: {LABEL_CSV}")
    df = pd.read_csv(LABEL_CSV)
    needed = {"COD_CELDA", "RESUL_ITUR"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {needed}, has {set(df.columns)}")

    df = df.loc[df["RESUL_ITUR"].astype(float) < TAU, ["COD_CELDA", "RESUL_ITUR"]].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No rows satisfy RESUL_ITUR < TAU.")

    # Apply PROPORTION knob before splitting
    df = subsample_df(df, PROPORTION, seed=SEED)
    print(f"Rows after TAU filter and PROPORTION={PROPORTION}: {len(df)}")

    # --- Dataset & loaders ---
    dataset = AlphaEarthTIFRegDataset(df, tif_dir=TIF_DIR, scan_check=True)
    train_loader, val_loader, test_loader = make_loaders(dataset, batch_size=BATCH_SIZE)

    # --- Model / Train ---
    model = S5Regressor().to(DEVICE)
    print("\n========== Single Stage: training on selected data ==========")
    best_val_r2, best_state = train_with_early_stopping(
        model, train_loader, val_loader, device=DEVICE,
        epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
        patience=10, ckpt_path=BEST_MODEL_PATH
    )

    # --- Test & save ---
    test_r2, test_preds, test_trues = eval_regressor(model, test_loader, DEVICE)
    print(f"\nTest R² (RESUL_ITUR < {TAU}): {test_r2:.4f}")

    pd.DataFrame([{"metric": "test_R2_lt_tau", "value": float(test_r2)}]).to_csv(
        os.path.join(OUT_DIR, "regression_lt_tau_test_r2.csv"), index=False
    )
    pd.DataFrame({"y_true": test_trues, "y_pred": test_preds}).to_csv(
        os.path.join(OUT_DIR, "regression_lt_tau_test_predictions.csv"), index=False
    )
    if best_state is not None:
        torch.save(best_state, BEST_MODEL_PATH)

    print(f"\nSaved metrics/predictions to {OUT_DIR}")
    print(f"Best weights: {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()

