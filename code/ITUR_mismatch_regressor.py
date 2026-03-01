#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train an MLP regressor to predict ITUR difference (RESUL_ITUR - ITUR_calc)
Split: 50/20/30 (train/val/test)

Reports:
  - R² (test)
  - RMSE (test)
  - SHAP feature importance ranking

Input:
  ../data/itur_difference_regression_dataset.csv

Outputs:
  ../data/nn_regression_metrics.csv
  ../data/shap_feature_importance.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import shap
from tqdm.auto import tqdm


FEATURES = [
    "hat_P_USOSUEPV", "hat_COND_ACCE", "hat_P_USOSUECON", "hat_EQUIP_URB",
    "hat_DILOCCON50", "hat_TAM_POB", "hat_DEN_POB", "hat_CAR_SER_VI"
]

TARGET = "diff"   # continuous difference


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
        return self.net(x).squeeze(1)


def train_model(model, train_loader, val_loader, device="cpu", epochs=80, lr=1e-3, patience=10):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    best_state = None
    bad = 0

    epoch_bar = tqdm(range(epochs), desc="Training epochs")

    for ep in epoch_bar:
        model.train()
        tr_losses = []

        for xb, yb in tqdm(train_loader, leave=False, desc=f"Epoch {ep+1} [train]"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, leave=False, desc=f"Epoch {ep+1} [val]"):
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        epoch_bar.set_postfix(val_loss=f"{val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model

def predict(model, X: np.ndarray, device="cpu", batch_size=8192):
    model.eval()
    out = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            preds = model(xb).cpu().numpy()
            out.append(preds)

    return np.concatenate(out)


def main():

    in_csv = "../data/itur_difference_regression_dataset.csv"
    os.makedirs("../data", exist_ok=True)

    df = pd.read_csv(in_csv)

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)

    # 50/20/30 split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.50, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.60, random_state=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))

    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4096)

    model = MLP(d_in=X.shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, device=device)

    # ----------------------------
    # Test metrics
    # ----------------------------
    y_pred = predict(model, X_te, device=device)

    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))

    metrics = pd.DataFrame([{
        "r2_test": r2,
        "rmse_test": rmse,
        "n_test": len(y_te)
    }])

    metrics.to_csv("../data/nn_regression_metrics.csv", index=False)
    print(metrics.to_string(index=False))

    # ----------------------------
    # SHAP (regression)
    # ----------------------------
    rng = np.random.default_rng(42)

    bg_n = min(1000, len(X_tr))
    ex_n = min(2000, len(X_te))

    X_bg = X_tr[rng.choice(len(X_tr), size=bg_n, replace=False)]
    X_ex = X_te[rng.choice(len(X_te), size=ex_n, replace=False)]

    def f_predict(x_np):
        x_np = np.asarray(x_np, dtype=np.float32)
        return predict(model, x_np, device=device).reshape(-1, 1)

    print("\nComputing SHAP values...")
    explainer = shap.KernelExplainer(f_predict, X_bg)
    shap_vals = explainer.shap_values(X_ex, nsamples=200)

    shap_arr = np.asarray(shap_vals)
    shap_arr = np.squeeze(shap_arr)

    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 0]

    imp = np.mean(np.abs(shap_arr), axis=0)
    imp = np.asarray(imp).reshape(-1)

    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "mean_abs_shap": imp
    }).sort_values("mean_abs_shap", ascending=False)

    imp_df.to_csv("../data/shap_feature_importance.csv", index=False)

    print("\nTop SHAP features:")
    print(imp_df.head(10).to_string(index=False))

    print("\nWrote:")
    print("  ../data/nn_regression_metrics.csv")
    print("  ../data/shap_feature_importance.csv")


if __name__ == "__main__":
    main()