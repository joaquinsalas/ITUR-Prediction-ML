#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # pick your GPU

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.backends import cudnn
from tqdm.auto import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    accuracy_score, confusion_matrix, precision_recall_curve, roc_curve
)
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

import rasterio
from s5 import S5Block  # pip/your local s5 module
from torch.optim.lr_scheduler import OneCycleLR

# ----------------------------
# Config
# ----------------------------
TIF_DIR        = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/AlphaEarth_Aguascalientes"
LABEL_CSV      = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/AlphaEarth_Aguascalientes_ITUR.csv"  # must contain COD_CELDA, RESUL_ITUR
PARTITION_DIR  = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/partition"
OUT_DIR        = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/results_binary_s5_from_tifs"
BEST_MODEL_PATH= os.path.join(OUT_DIR, "best_s5_classifier.pt")
FIG_DIR        = "../figures"
TAU            = 0.878389

SEED         = 42
BATCH_SIZE   = 128
EPOCHS       = 2000
LR           = 1e-4           # base LR (optimizer); OneCycle will peak at MAX_LR
MAX_LR       = 3e-3           # OneCycleLR max learning rate
WEIGHT_DECAY = 1e-2
PATIENCE     = 10             # early stop patience (epochs with no Val LOSS improvement)
PRINT_EVERY  = 1

# Quickly test with a subset of each split (0<PROPORTION<=1)
PROPORTION   = 1.0           # start small; set to 1.0 for full run

# Debug-friendly data loader settings
NUM_WORKERS  = 0              # set to 4+ when running outside the debugger
PIN_MEMORY   = True
PERSISTENT   = NUM_WORKERS > 0

# Sequence model dims
L_SEQ     = 64
D_INPUT   = 1
D_MODEL   = 2048
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

def _load_npy(basepath):
    if os.path.exists(basepath):
        return np.load(basepath, allow_pickle=False)
    if os.path.exists(basepath + ".npy"):
        return np.load(basepath + ".npy", allow_pickle=False)
    raise FileNotFoundError(basepath + "(.npy) not found")

def save_curve_csv(path, x, y, x_name, y_name):
    df = pd.DataFrame({x_name: x, y_name: y})
    df.to_csv(path, index=False)

def subsample_df(df, proportion, seed=SEED):
    if not (0 < proportion <= 1):
        raise ValueError("PROPORTION must be in (0,1].")
    if proportion >= 1.0:
        return df.reset_index(drop=True)
    n = max(1, int(len(df) * proportion))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

def add_binary_label(df, tau=TAU):
    df = df.copy()
    df["y_bin"] = (df["RESUL_ITUR"].astype(float) >= tau).astype(int)
    return df

def undersample_majority(df, label_col="y_bin", seed=SEED):
    counts = df[label_col].value_counts()
    if len(counts) < 2:
        return df.copy()
    minority_class = counts.idxmin()
    majority_class = counts.idxmax()
    n_min = counts.min()
    df_min = df[df[label_col] == minority_class]
    df_maj = df[df[label_col] == majority_class].sample(n=n_min, random_state=seed)
    return (pd.concat([df_min, df_maj], axis=0)
              .sample(frac=1.0, random_state=seed)
              .reset_index(drop=True))

# ---------- Plotting helpers ----------
def plot_pr_curve(recall, precision, ap, out_path, title_suffix=""):
    plt.figure(figsize=(6, 5))
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    ttl = f"Precision–Recall (AP={ap:.3f})"
    if title_suffix: ttl += f" — {title_suffix}"
    plt.title(ttl); plt.xlim(0,1); plt.ylim(0,1.05)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_roc_curve(fpr, tpr, auc, out_path, title_suffix=""):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=1.5)
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1.0)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    ttl = f"ROC (AUC={auc:.3f})"
    if title_suffix: ttl += f" — {title_suffix}"
    plt.title(ttl); plt.xlim(0,1); plt.ylim(0,1.05)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

# ----------------------------
# Dataset that reads TIFFs on-the-fly
# ----------------------------
class AlphaEarthTIFDataset(Dataset):
    def __init__(self, df_subset, tif_dir=TIF_DIR, tau=TAU, scan_check=True):
        self.df = df_subset.reset_index(drop=False)
        self.tif_dir = tif_dir
        self.tau = tau

        if scan_check:
            self.missing = self.wrong_bands = self.unreadable = 0
            keep_mask = np.ones(len(self.df), dtype=bool)
            for i, row in enumerate(tqdm(self.df.itertuples(index=False),
                                         total=len(self.df),
                                         desc="Scanning TIFs", leave=False)):
                code = str(getattr(row, "COD_CELDA"))
                tif_path = os.path.join(self.tif_dir, f"AlphaEarth_{code}.tif")
                if not os.path.exists(tif_path):
                    keep_mask[i] = False; self.missing += 1; continue
                try:
                    with rasterio.open(tif_path) as src:
                        if src.count != 64:
                            keep_mask[i] = False; self.wrong_bands += 1; continue
                except Exception:
                    keep_mask[i] = False; self.unreadable += 1
            if not keep_mask.all():
                self.df = self.df[keep_mask].reset_index(drop=True)
            kept = len(self.df)
            if kept == 0:
                raise RuntimeError(
                    f"No valid samples. Missing={self.missing}, "
                    f"wrong_bands={self.wrong_bands}, unreadable={self.unreadable}"
                )
            print(f"TIF scan kept={kept} | missing={self.missing} | "
                  f"wrong_bands={self.wrong_bands} | unreadable={self.unreadable}")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        code = str(row["COD_CELDA"])
        y_real = float(row["RESUL_ITUR"])
        y_bin = 1.0 if y_real >= self.tau else 0.0

        tif_path = os.path.join(self.tif_dir, f"AlphaEarth_{code}.tif")
        with rasterio.open(tif_path) as src:
            arr = src.read()  # (64,H,W)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[0] != 64:
            raise ValueError(f"{tif_path} has {arr.shape[0]} bands; expected 64.")

        chan_means = arr.reshape(64, -1).mean(axis=1).astype(np.float32)  # (64,)
        seq = chan_means.reshape(L_SEQ, 1)  # (64,1)

        return torch.from_numpy(seq), torch.tensor(y_bin, dtype=torch.float32), torch.tensor(int(row["index"]), dtype=torch.long)

# ----------------------------
# Model (S5 classifier)
# ----------------------------
class S5Classifier(nn.Module):
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
        self.head = nn.Linear(d_model, 1)  # logits

    def forward(self, x):
        x = self.encoder(x)  # (B, 64, D_MODEL)
        for block, norm, drop in zip(self.s5_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm: z = norm(z)
            z = block(z); z = drop(z)
            x = x + z
            if not self.prenorm: x = norm(x)
        x = x.mean(dim=1)
        logits = self.head(x).squeeze(-1)
        return logits

# ----------------------------
# Eval helpers
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device, split_name="VAL"):
    model.eval()
    logits_all, y_all, idx_all = [], [], []
    for xb, yb, ib in loader:
        xb = xb.to(device)
        logits = model(xb)
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(yb.numpy())
        idx_all.append(ib.numpy())

    logits = np.concatenate(logits_all)
    y_true = np.concatenate(y_all).astype(np.float32)
    idxs   = np.concatenate(idx_all)

    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(np.int32)

    try:   roc_auc = roc_auc_score(y_true, probs)
    except ValueError: roc_auc = float("nan")
    try:   pr_auc  = average_precision_score(y_true, probs)
    except ValueError: pr_auc  = float("nan")

    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    metrics = {
        "ROC_AUC": roc_auc, "PR_AUC": pr_auc, "ACC": acc, "PREC": prec,
        "REC": rec, "F1": f1,
        "TN": int(cm[0,0]) if cm.shape==(2,2) else None,
        "FP": int(cm[0,1]) if cm.shape==(2,2) else None,
        "FN": int(cm[1,0]) if cm.shape==(2,2) else None,
        "TP": int(cm[1,1]) if cm.shape==(2,2) else None,
    }

    try:
        pr_P, pr_R, pr_T = precision_recall_curve(y_true, probs)
        fpr, tpr, roc_T  = roc_curve(y_true, probs)
    except Exception:
        pr_P = pr_R = pr_T = fpr = tpr = roc_T = None

    return metrics, probs, y_pred, y_true, idxs, (pr_P, pr_R, pr_T, fpr, tpr, roc_T)

def print_metrics(tag, m):
    print(f"[{tag}] ROC_AUC={m['ROC_AUC']:.4f}  PR_AUC={m['PR_AUC']:.4f}  "
          f"ACC={m['ACC']:.4f}  P={m['PREC']:.4f}  R={m['REC']:.4f}  F1={m['F1']:.4f}  "
          f"CM=[TN={m['TN']}, FP={m['FP']}, FN={m['FN']}, TP={m['TP']}]")

def save_predictions(split, outdir, idxs, y_true, probs, y_pred):
    df = pd.DataFrame({
        "index": idxs,
        "y_true": y_true.astype(int),
        "prob_1": probs,
        "y_pred": y_pred.astype(int),
    })
    df.to_csv(os.path.join(outdir, f"predictions_{split}.csv"), index=False)

@torch.no_grad()
def compute_val_loss(model, loader, device, criterion):
    """Average BCE-with-logits loss on a loader."""
    model.eval()
    total, n = 0.0, 0
    for xb, yb, _ in loader:
        xb = xb.to(device); yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total += loss.item()
        n += 1
    return total / max(1, n)

# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- Load CSV ---
    if not os.path.exists(LABEL_CSV):
        raise FileNotFoundError(f"LABEL_CSV not found: {LABEL_CSV}")
    df = pd.read_csv(LABEL_CSV)
    needed = {"COD_CELDA", "RESUL_ITUR"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {needed}, has {set(df.columns)}")
    df = df.reset_index(drop=True)

    # --- Split indices ---
    ind_train = _load_npy(os.path.join(PARTITION_DIR, "ind_train")).astype(int).ravel()
    ind_val   = _load_npy(os.path.join(PARTITION_DIR, "ind_val")).astype(int).ravel()
    ind_test  = _load_npy(os.path.join(PARTITION_DIR, "ind_test")).astype(int).ravel()

    nrows = len(df)
    def clamp(idxs): return idxs[(idxs >= 0) & (idxs < nrows)]
    ind_train, ind_val, ind_test = clamp(ind_train), clamp(ind_val), clamp(ind_test)

    # --- DataFrames + labels ---
    df_train = add_binary_label(df.iloc[ind_train].copy(), TAU)
    df_val   = add_binary_label(df.iloc[ind_val].copy(), TAU)
    df_test  = add_binary_label(df.iloc[ind_test].copy(), TAU)

    # --- Subsample for quick run ---
    df_train = subsample_df(df_train, PROPORTION, seed=SEED+1)
    df_val   = subsample_df(df_val,   PROPORTION, seed=SEED+2)
    df_test  = subsample_df(df_test,  PROPORTION, seed=SEED+3)

    # --- Undersample majority on TRAIN only ---
    df_train = undersample_majority(df_train, label_col="y_bin", seed=SEED+4)

    # --- Datasets / loaders ---
    ds_train = AlphaEarthTIFDataset(df_train)
    ds_val   = AlphaEarthTIFDataset(df_val)
    ds_test  = AlphaEarthTIFDataset(df_test)

    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    if NUM_WORKERS > 0:
        dl_kwargs["persistent_workers"] = PERSISTENT

    train_loader = DataLoader(ds_train, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(ds_val,   shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(ds_test,  shuffle=False, **dl_kwargs)

    # --- Model / optim / OneCycleLR ---
    model = S5Classifier().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        anneal_strategy='linear'
    )
    criterion = nn.BCEWithLogitsLoss()  # no class weighting; using undersampling

    # --- Train with early stopping (on Validation LOSS), save best model ---
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb, _ in tqdm(train_loader, desc=f"Train {epoch}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        tr_loss = total_loss / max(1, len(train_loader))
        val_loss = compute_val_loss(model, val_loader, DEVICE, criterion)
        val_metrics, *_ = evaluate(model, val_loader, DEVICE, split_name="VAL")

        if epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch:03d} | Train loss {tr_loss:.4f} | Val loss {val_loss:.4f}")
            print_metrics("VAL", val_metrics)

        if val_loss < best_val_loss - 1e-7:  # tiny tolerance
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no Val LOSS improvement for {PATIENCE}).")
            break

    # Load best model before final eval
    if os.path.isfile(BEST_MODEL_PATH):
        state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)

    # --- Evaluate & save predictions/curves + generate figures ---
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        metrics, probs, y_pred, y_true, idxs, curves = evaluate(model, loader, DEVICE, split_name=split.upper())
        print_metrics(split.upper(), metrics)
        save_predictions(split, OUT_DIR, idxs, y_true, probs, y_pred)

        pr_P, pr_R, pr_T, fpr, tpr, roc_T = curves

        # CSVs
        if pr_P is not None:
            save_curve_csv(os.path.join(OUT_DIR, f"pr_curve_{split}.csv"), pr_R, pr_P, "recall", "precision")
        if fpr is not None:
            save_curve_csv(os.path.join(OUT_DIR, f"roc_curve_{split}.csv"), fpr, tpr, "fpr", "tpr")

        # Figures
        if (pr_P is not None) and np.isfinite(metrics["PR_AUC"]):
            pr_png = os.path.join(FIG_DIR, f"pr_curve_{split}.png")
            plot_pr_curve(pr_R, pr_P, metrics["PR_AUC"], pr_png, title_suffix=split.upper())
        if (fpr is not None) and np.isfinite(metrics["ROC_AUC"]):
            roc_png = os.path.join(FIG_DIR, f"roc_curve_{split}.png")
            plot_roc_curve(fpr, tpr, metrics["ROC_AUC"], roc_png, title_suffix=split.upper())

    # Save metric summary
    def get_metrics(loader):
        m, *_ = evaluate(model, loader, DEVICE, split_name="X")
        return m
    summary = {"train": get_metrics(train_loader), "val": get_metrics(val_loader), "test": get_metrics(test_loader)}
    pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"))
    print(f"\nSaved best model to: {BEST_MODEL_PATH}")
    print(f"CSV outputs: {OUT_DIR}")
    print(f"Figures saved to: {FIG_DIR}")

if __name__ == "__main__":
    main()

