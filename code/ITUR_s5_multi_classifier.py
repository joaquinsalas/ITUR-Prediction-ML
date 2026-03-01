#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # pick your GPU

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.backends import cudnn
from tqdm.auto import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import rasterio
from s5 import S5Block  # your S5 module
from torch.optim.lr_scheduler import OneCycleLR

# ----------------------------
# Config
# ----------------------------
TIF_DIR        = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/AlphaEarth_Aguascalientes"
LABEL_CSV      = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/AlphaEarth_Aguascalientes_ITUR.csv"  # needs COD_CELDA, RESUL_ITUR
OUT_DIR        = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/results_multiclass_s5_from_tifs"
BEST_MODEL_PATH= os.path.join(OUT_DIR, "best_s5_multiclass.pt")
FIG_DIR        = "../figures_multiclass_s5"

SEED         = 42
BATCH_SIZE   = 128
EPOCHS       = 200
LR           = 1e-4            # base LR (optimizer); OneCycle will peak at MAX_LR
MAX_LR       = 3e-3            # OneCycleLR max learning rate
WEIGHT_DECAY = 1e-2
PATIENCE     = 10              # early stop on Val CE loss (epochs without improvement)
PRINT_EVERY  = 1

# Use a subset of data if desired (0 < PROPORTION <= 1)
PROPORTION   = 1.0

# DataLoader params
NUM_WORKERS  = 0               # increase when not debugging
PIN_MEMORY   = True
PERSISTENT   = NUM_WORKERS > 0

# Sequence/model dims
L_SEQ     = 64
D_INPUT   = 1
D_MODEL   = 256
N_LAYERS  = 3
DROPOUT   = 0.10
PRENORM   = True

N_CLASSES = 5
CLASS_BINS = np.array([0.0, 0.268546, 0.464280, 0.667118, 0.878389, 1.0])  # edges
CLASS_LABELS = [
    "[0.000000, 0.268546)",
    "[0.268546, 0.464280)",
    "[0.464280, 0.667118)",
    "[0.667118, 0.878389)",
    "[0.878389, 1.000000]"
]

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

def bin_to_class(t):
    """
    Map scalar t in [0,1] to class id 0..4 using CLASS_BINS:
      [e0, e1), [e1, e2), [e2, e3), [e3, e4), [e4, e5] (inclusive on last).
    """
    # np.digitize returns indices 1..len(bins)-1; we adjust to 0..4.
    # We want last bin inclusive; clip just in case of exact 1.0.
    t = min(max(float(t), CLASS_BINS[0]), CLASS_BINS[-1])
    # Make upper bound of last bin inclusive by nudging
    if np.isclose(t, CLASS_BINS[-1]):
        t = np.nextafter(t, -np.inf)
    idx = np.digitize(t, CLASS_BINS, right=False) - 1
    return int(np.clip(idx, 0, N_CLASSES-1))

def balance_by_undersampling(df, label_col="y_cls", seed=SEED):
    """
    Undersample each class to the size of the minority class.
    """
    counts = df[label_col].value_counts()
    min_n = counts.min()
    parts = []
    for c in range(N_CLASSES):
        part = df[df[label_col] == c].sample(
            n=min_n, replace=False, random_state=seed
        )
        parts.append(part)
    df_bal = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_bal

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------
# Dataset
# ----------------------------
class AlphaEarthTIFMultiDataset(Dataset):
    """
    For each row (COD_CELDA, RESUL_ITUR):
      - load AlphaEarth_{COD_CELDA}.tif (64,H,W)
      - per-band mean to (64,)
      - return (seq= (64,1), y_cls in {0..4})
    """
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
        t = float(row["RESUL_ITUR"])
        y_cls = bin_to_class(t)

        tif_path = os.path.join(self.tif_dir, f"AlphaEarth_{code}.tif")
        with rasterio.open(tif_path) as src:
            arr = src.read()  # (64, H, W)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[0] != 64:
            raise ValueError(f"{tif_path} has {arr.shape[0]} bands; expected 64.")
        chan_means = arr.reshape(64, -1).mean(axis=1).astype(np.float32)
        seq = chan_means.reshape(L_SEQ, 1)
        return torch.from_numpy(seq), torch.tensor(y_cls, dtype=torch.long)

# ----------------------------
# Model (S5 multiclass)
# ----------------------------
class S5Classifier(nn.Module):
    def __init__(self, d_in=1, d_model=D_MODEL, n_layers=N_LAYERS,
                 dropout=DROPOUT, prenorm=PRENORM, n_classes=N_CLASSES):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_in, d_model)
        self.s5_layers = nn.ModuleList([
            S5Block(dim=d_model, state_dim=d_model, bidir=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.encoder(x)  # (B, 64, D_MODEL)
        for block, norm, drop in zip(self.s5_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm: z = norm(z)
            z = block(z); z = drop(z)
            x = x + z
            if not self.prenorm: x = norm(x)
        x = x.mean(dim=1)            # (B, D_MODEL)
        logits = self.head(x)        # (B, C)
        return logits

# ----------------------------
# Splits & Loaders
# ----------------------------
def make_loaders(df, balance_train=True):
    ds_full = AlphaEarthTIFMultiDataset(df, tif_dir=TIF_DIR, scan_check=True)
    n_total = len(ds_full)
    n_train = int(0.50 * n_total)
    n_val   = int(0.20 * n_total)
    n_test  = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(ds_full, [n_train, n_val, n_test], generator=gen)

    # Balance the TRAIN split by undersampling in dataframe space
    if balance_train:
        # recover indices for train subset to access original df rows
        train_idx = train_ds.indices if hasattr(train_ds, "indices") else train_ds._indices
        df_train = df.iloc[train_idx].copy()
        df_train["y_cls"] = df_train["RESUL_ITUR"].map(bin_to_class)
        df_train_bal = balance_by_undersampling(df_train, label_col="y_cls", seed=SEED+1)
        # rebuild a dataset from the balanced df
        train_ds = AlphaEarthTIFMultiDataset(df_train_bal[["COD_CELDA", "RESUL_ITUR"]], tif_dir=TIF_DIR, scan_check=False)

    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    if NUM_WORKERS > 0:
        dl_kwargs["persistent_workers"] = PERSISTENT

    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)
    return train_loader, val_loader, test_loader

# ----------------------------
# Metrics (OVR curves on TEST)
# ----------------------------
@torch.no_grad()
def eval_ovr_curves(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)                  # (B, C)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(yb.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_targets, axis=0)  # (N,)

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()  # (N, C)

    # Per-class OVR curves
    ovr = {}
    for k in range(N_CLASSES):
        y_bin = (y_true == k).astype(np.int32)
        scores = probs[:, k]
        try:
            auc_roc = roc_auc_score(y_bin, scores)
        except ValueError:
            auc_roc = float("nan")
        try:
            ap = average_precision_score(y_bin, scores)
        except ValueError:
            ap = float("nan")

        fpr, tpr, _ = roc_curve(y_bin, scores)
        prec, rec, _ = precision_recall_curve(y_bin, scores)
        ovr[k] = {"fpr": fpr, "tpr": tpr, "auc": auc_roc,
                  "prec": prec, "rec": rec, "ap": ap}
    return ovr, probs, y_true

def plot_ovr_roc(ovr, out_png):
    plt.figure(figsize=(7, 6))
    for k in range(N_CLASSES):
        fpr = ovr[k]["fpr"]; tpr = ovr[k]["tpr"]; auc = ovr[k]["auc"]
        plt.plot(fpr, tpr, label=f"{k}: {CLASS_LABELS[k]} (AUC={auc:.3f})", linewidth=1.5)
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1.0)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC — Test")
    plt.xlim(0,1); plt.ylim(0,1.05); plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

def plot_ovr_pr(ovr, out_png):
    plt.figure(figsize=(7, 6))
    for k in range(N_CLASSES):
        prec = ovr[k]["prec"]; rec = ovr[k]["rec"]; ap = ovr[k]["ap"]
        plt.step(rec, prec, where="post", label=f"{k}: {CLASS_LABELS[k]} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("One-vs-Rest Precision–Recall — Test")
    plt.xlim(0,1); plt.ylim(0,1.05); plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

# ----------------------------
# Training (CE + OneCycle + EarlyStopping on Val CE)
# ----------------------------
def train(model, train_loader, val_loader, device):
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        anneal_strategy='linear'
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        running = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train {epoch}/{EPOCHS}", leave=False):
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
        tr_loss = running / max(1, len(train_loader))

        # ---- val loss ----
        with torch.no_grad():
            model.eval()
            total, n = 0.0, 0
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total += loss.item(); n += 1
            val_loss = total / max(1, n)

        if epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch:03d} | Train CE {tr_loss:.5f} | Val CE {val_loss:.5f} | LR {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, BEST_MODEL_PATH)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no Val CE improvement for {PATIENCE}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)
    ensure_dirs()

    # --- Load CSV ---
    if not os.path.exists(LABEL_CSV):
        raise FileNotFoundError(f"LABEL_CSV not found: {LABEL_CSV}")
    df = pd.read_csv(LABEL_CSV)
    needed = {"COD_CELDA", "RESUL_ITUR"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {needed}, has {set(df.columns)}")

    # Keep only needed columns and (optionally) subsample whole dataset
    df = df[["COD_CELDA", "RESUL_ITUR"]].copy().reset_index(drop=True)
    df = subsample_df(df, PROPORTION, seed=SEED)
    print(f"Total rows after PROPORTION: {len(df)}")

    # --- Build loaders (50/20/30, balance train via undersampling) ---
    train_loader, val_loader, test_loader = make_loaders(df, balance_train=True)

    # --- Model ---
    model = S5Classifier().to(DEVICE)

    # --- Train ---
    train(model, train_loader, val_loader, DEVICE)

    # --- Test OVR curves ---
    ovr, probs, y_true = eval_ovr_curves(model, test_loader, DEVICE)

    # Save per-sample predictions (test)
    test_df = pd.DataFrame({
        "y_true": y_true,
        **{f"p_class{k}": probs[:, k] for k in range(N_CLASSES)}
    })
    test_df.to_csv(os.path.join(OUT_DIR, "test_probs_ovr.csv"), index=False)

    # Plot combined ROC and PR
    roc_png = os.path.join(FIG_DIR, "ovr_roc_test.png")
    pr_png  = os.path.join(FIG_DIR, "ovr_pr_test.png")
    plot_ovr_roc(ovr, roc_png)
    plot_ovr_pr(ovr, pr_png)

    # Save per-class summary AUC/AP
    rows = []
    for k in range(N_CLASSES):
        rows.append({"class": k, "label": CLASS_LABELS[k],
                     "ROC_AUC": float(ovr[k]["auc"]),
                     "PR_AP": float(ovr[k]["ap"])})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "ovr_summary_test.csv"), index=False)

    print(f"\nSaved best model to: {BEST_MODEL_PATH}")
    print(f"Probabilities CSV: {os.path.join(OUT_DIR, 'test_probs_ovr.csv')}")
    print(f"Per-class summary: {os.path.join(OUT_DIR, 'ovr_summary_test.csv')}")
    print(f"Figures: {roc_png} and {pr_png}")

if __name__ == "__main__":
    main()
