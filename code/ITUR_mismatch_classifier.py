#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train an MLP classifier to predict "no_cumple" from normalized predictors.
Split: 50/20/30 (train/val/test)
Balance train split by undersampling (default: majority undersampling).

Reports:
  - ROC-AUC (test)
  - PR-AUC  (test)
  - SHAP feature importance ranking

Input:
  ../data/itur_mismatch_ml_dataset.csv

Outputs:
  ../data/nn_test_metrics.csv
  ../data/shap_feature_importance.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

import shap


FEATURES = [
    "hat_P_USOSUEPV", "hat_COND_ACCE", "hat_P_USOSUECON", "hat_EQUIP_URB",
    "hat_DILOCCON50", "hat_TAM_POB", "hat_DEN_POB", "hat_CAR_SER_VI"
]
TARGET = "no_cumple"  # 1 = significant difference


# ----------------------------
# Balancing (undersampling)
# ----------------------------
def undersample_to_balance(X: np.ndarray, y: np.ndarray, undersample_majority: bool = True, seed: int = 42):
    """
    If undersample_majority=True (recommended): reduce larger class to match smaller.
    If False: reduces minority (unusual; kept as an option).
    """
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n0, n1 = len(idx0), len(idx1)
    if n0 == 0 or n1 == 0:
        return X, y

    if undersample_majority:
        if n0 > n1:
            idx0 = rng.choice(idx0, size=n1, replace=False)
        else:
            idx1 = rng.choice(idx1, size=n0, replace=False)
    else:
        # undersample minority (as literally requested)
        if n0 < n1:
            idx0 = rng.choice(idx0, size=n1, replace=False)  # will error if n1>n0; so guard
        else:
            idx1 = rng.choice(idx1, size=n0, replace=False)

    idx = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    return X[idx], y[idx]


# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # logits


def train_model(model, train_loader, val_loader, device="cpu", epochs=50, lr=1e-3, patience=8):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = np.inf
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                va_losses.append(loss.item())

        val_loss = float(np.mean(va_losses)) if va_losses else float("inf")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_proba(model, X: np.ndarray, device="cpu", batch_size=8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            out.append(prob)
    return np.concatenate(out)


def main():
    in_csv = "../data/itur_mismatch_ml_dataset.csv"
    os.makedirs("../data", exist_ok=True)

    df = pd.read_csv(in_csv)
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.int64)

    # 50/20/30 split: do 50 train, 50 temp, then temp -> 40 val, 60 test (i.e., 20/30 overall)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.50, random_state=42, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.60, random_state=42, stratify=y_tmp)

    # balance training by undersampling majority (recommended)
    X_trb, y_trb = undersample_to_balance(X_tr, y_tr, undersample_majority=True, seed=42)

    # dataloaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = TensorDataset(torch.from_numpy(X_trb).float(), torch.from_numpy(y_trb).float())
    val_ds   = TensorDataset(torch.from_numpy(X_va).float(),  torch.from_numpy(y_va).float())

    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=4096, shuffle=False, drop_last=False)

    model = MLP(d_in=X.shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, device=device, epochs=60, lr=1e-3, patience=10)

    # test metrics
    p_te = predict_proba(model, X_te, device=device)
    roc = roc_auc_score(y_te, p_te)
    pr  = average_precision_score(y_te, p_te)

    metrics = pd.DataFrame([{"roc_auc": roc, "pr_auc": pr, "n_test": len(y_te)}])
    metrics.to_csv("../data/nn_test_metrics.csv", index=False)
    print(metrics.to_string(index=False))

    # ----------------------------
    # SHAP (KernelExplainer; sample for tractability)
    # ----------------------------
    # background + explained set (subsample)
    rng = np.random.default_rng(42)
    bg_n = min(1000, len(X_tr))
    ex_n = min(2000, len(X_te))

    bg_idx = rng.choice(len(X_tr), size=bg_n, replace=False)
    ex_idx = rng.choice(len(X_te), size=ex_n, replace=False)

    X_bg = X_tr[bg_idx]
    X_ex = X_te[ex_idx]

    def f_predict(x_np):
        # shap passes float64; cast back
        x_np = np.asarray(x_np, dtype=np.float32)
        return predict_proba(model, x_np, device=device).reshape(-1, 1)

    explainer = shap.KernelExplainer(f_predict, X_bg)
    shap_vals = explainer.shap_values(X_ex, nsamples=200)  # increase nsamples for better accuracy

    # shap_vals shape: (n_samples, n_features) or list; normalize
    # shap_vals shape handling
    # shap_vals shape handling
    if isinstance(shap_vals, list):
        shap_arr = np.asarray(shap_vals[0])
    else:
        shap_arr = np.asarray(shap_vals)

    # Expected shapes:
    # (n_samples, n_features)
    # (n_samples, n_features, 1)
    # (1, n_samples, n_features)

    shap_arr = np.squeeze(shap_arr)

    # If still 3D (rare but possible)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 0]

    # Now must be 2D
    if shap_arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {shap_arr.shape}")

    # Aggregate importance
    imp = np.mean(np.abs(shap_arr), axis=0)

    # Ensure strict 1D
    imp = np.asarray(imp).reshape(-1)



    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "mean_abs_shap": imp
    }).sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv("../data/shap_feature_importance.csv", index=False)
    print("\nTop SHAP features:")
    print(imp_df.head(10).to_string(index=False))

    print("\nWrote:")
    print("  ../data/nn_test_metrics.csv")
    print("  ../data/shap_feature_importance.csv")


if __name__ == "__main__":
    main()