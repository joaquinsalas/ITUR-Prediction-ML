#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # pick your GPU

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
LABEL_CSV       = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/AlphaEarth_Aguascalientes_ITUR_con_norm_2.csv"
OUT_DIR         = "/mnt/data-r1/data/AlphaEarth_ITUR_Aguas/results_regression_from_tifs"

# Pretrained teacher (67ch) and student output paths
TEACHER_MODEL_PATH = os.path.join(OUT_DIR, "best_coatnet1_gemmlp_regressor_full_67ch.pth")
STUDENT_MODEL_PATH = os.path.join(OUT_DIR, "best_coatnet1_gemmlp_student_64ch_distilled.pth")

SEED         = 42
BATCH_SIZE   = 128
EPOCHS       = 100          # use early stopping
LR           = 1e-4
MAX_LR       = 1e-3
WEIGHT_DECAY = 1e-2
PRINT_EVERY  = 1

PROPORTION   = 1.0
NUM_WORKERS  = 0
PIN_MEMORY   = True
PERSISTENT   = NUM_WORKERS > 0

# Distillation weight lambda
DISTILL_LAMBDA = 0.4  # between 0 and 1

# Model / input
TIF_BANDS      = 64             # bands present in AlphaEarth TIFs
IN_CHANS_TEACH = 67             # teacher: 64 AlphaEarth + 3 extra variables
IN_CHANS_STUD  = 64             # student: only 64 AlphaEarth bands
IMG_SIZE       = 224            # CoAtNet rw_224 variants use 224
BACKBONE_NAME  = "coatnet_1_rw_224"  # timm CoAtNet variant

# Early stopping (align more with Keras script)
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
# CoAtNet regressor with GeM + 2-layer MLP head
# ----------------------------
class CoAtNetRegressor(nn.Module):
    def __init__(self, backbone_name=BACKBONE_NAME, in_chans=IN_CHANS_TEACH, hidden_dim=512):
        super().__init__()
        # num_classes=0 and global_pool="" so we get raw features
        self.backbone = timm.create_model(
            backbone_name,
            in_chans=in_chans,
            pretrained=True,
            num_classes=0,
            global_pool=""
        )
        feat_dim = self.backbone.num_features
        self.gem = GeM()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)  # raw scalar regression output
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x)  # (B,C,H,W)
        pooled = self.gem(feats)                  # (B,C)
        out = self.head(pooled).squeeze(-1)       # (B,)
        return out

# ----------------------------
# Dataset
# ----------------------------
class AlphaEarthTIFRegDistillDataset(Dataset):
    """
    Each sample:
      - Load 64-band AlphaEarth TIF for the COD_CELDA
      - Per-channel min-max normalization on those 64 bands
      - Build:
          * student input: 64 bands only
          * teacher input: 64 bands + 3 extra variables as spatial channels
      - Resize both to (IMG_SIZE, IMG_SIZE)
      - Return (img_student, img_teacher, y)
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
                    keep_mask[i] = False; missing += 1; continue
                try:
                    with rasterio.open(tif_path) as src:
                        if src.count != TIF_BANDS:
                            keep_mask[i] = False; wrong_bands += 1; continue
                except Exception:
                    keep_mask[i] = False; unreadable += 1
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
        y = float(row["RESUL_ITUR"])

        # ---- 64-band AlphaEarth image ----
        tif_path = os.path.join(self.tif_dir, f"AlphaEarth_{code}.tif")
        with rasterio.open(tif_path) as src:
            img = src.read()  # (C,H,W), C = 64
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Per-image, per-channel min-max normalization on the 64 AlphaEarth bands
        img_64 = per_channel_minmax(img)  # (64,H,W)

        # ---- 3 extra variables as constant channels for TEACHER ----
        car_ser_vi = float(row["CAR_SER_VI"])
        diloc50    = float(row["DILOCCON50_log_norm"])
        p_usosuepv = float(row["P_USOSUEPV_norm"])
        extras = np.array([car_ser_vi, diloc50, p_usosuepv], dtype=np.float32)  # (3,)

        c, h, w = img_64.shape
        extras_layer = np.repeat(extras[:, None, None], h, axis=1)
        extras_layer = np.repeat(extras_layer, w, axis=2)
        img_67 = np.concatenate([img_64, extras_layer], axis=0)  # (67,H,W)

        # Resize both to 224x224 for CoAtNet
        img_64_t = torch.from_numpy(img_64)  # (64,H,W)
        img_64_t = F.interpolate(
            img_64_t.unsqueeze(0), size=(self.resize, self.resize),
            mode='bilinear', align_corners=False
        ).squeeze(0)  # (64,224,224)

        img_67_t = torch.from_numpy(img_67)  # (67,H,W)
        img_67_t = F.interpolate(
            img_67_t.unsqueeze(0), size=(self.resize, self.resize),
            mode='bilinear', align_corners=False
        ).squeeze(0)  # (67,224,224)

        return img_64_t, img_67_t, torch.tensor(y, dtype=torch.float32)

# ----------------------------
# Eval (student only)
# ----------------------------
@torch.no_grad()
def eval_student(model, loader, device, criterion):
    model.eval()
    preds, trues = [], []
    val_loss = 0.0
    for batch in loader:
        # batch can be (img_student, img_teacher, y) in distillation dataset
        if len(batch) == 3:
            xb_student, _, yb = batch
        else:
            xb_student, yb = batch
        xb_student = xb_student.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        yhat = model(xb_student).squeeze(-1)
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
# Main (distillation training)
# ----------------------------
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(LABEL_CSV):
        raise FileNotFoundError(f"LABEL_CSV not found: {LABEL_CSV}")
    df = pd.read_csv(LABEL_CSV)

    needed = {"COD_CELDA", "RESUL_ITUR",
              "CAR_SER_VI", "DILOCCON50_log_norm", "P_USOSUEPV_norm"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {needed}, has {set(df.columns)}")

    df = subsample_df(df, PROPORTION, seed=SEED)

    full_ds = AlphaEarthTIFRegDistillDataset(df, tif_dir=TIF_DIR,
                                             scan_check=True, resize=IMG_SIZE)
    n_total = len(full_ds)
    n_train = int(0.50 * n_total)
    n_val   = int(0.20 * n_total)
    n_test  = n_total - n_train - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test],
                                             generator=gen)

    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                     pin_memory=PIN_MEMORY)
    if NUM_WORKERS > 0:
        dl_kwargs["persistent_workers"] = PERSISTENT
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    # ------------------------
    # Teacher: 67ch, frozen
    # ------------------------
    teacher = CoAtNetRegressor(backbone_name=BACKBONE_NAME,
                               in_chans=IN_CHANS_TEACH)
    if not os.path.exists(TEACHER_MODEL_PATH):
        raise FileNotFoundError(f"Teacher model not found: {TEACHER_MODEL_PATH}")
    teacher_state = torch.load(TEACHER_MODEL_PATH, map_location="cpu")
    teacher.load_state_dict(teacher_state)
    teacher = teacher.to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ------------------------
    # Student: 64ch, trainable
    # ------------------------
    student = CoAtNetRegressor(backbone_name=BACKBONE_NAME,
                               in_chans=IN_CHANS_STUD)
    student = student.to(DEVICE)

    optimizer = optim.AdamW(student.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()

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

    history = []

    for epoch in range(1, EPOCHS + 1):
        student.train()
        total_loss = 0.0

        for xb_student, xb_teacher, yb in tqdm(
            train_loader, desc=f"Train {epoch}/{EPOCHS}", leave=False
        ):
            xb_student = xb_student.to(DEVICE, non_blocking=True)
            xb_teacher = xb_teacher.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Teacher prediction (no grad)
            with torch.no_grad():
                teacher_pred = teacher(xb_teacher)

            # Student prediction
            student_pred = student(xb_student)

            # Distillation + ground-truth loss
            loss_true = mse(student_pred, yb)
            loss_distill = mse(student_pred, teacher_pred)
            loss = (1.0 - DISTILL_LAMBDA) * loss_true + DISTILL_LAMBDA * loss_distill

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        tr_mse = total_loss / max(1, len(train_loader))
        val_mse, val_r2, _, _ = eval_student(student, val_loader, DEVICE, mse)
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch": epoch,
            "train_mse": float(tr_mse),
            "val_mse": float(val_mse),
            "val_r2": float(val_r2),
            "lr": float(current_lr),
        })

        if epoch % PRINT_EVERY == 0:
            print(
                f"Epoch {epoch:04d} | Train MSE {tr_mse:.6f} | "
                f"Val MSE {val_mse:.6f} | Val R2 {val_r2:.4f} | "
                f"LR {current_lr:.6f}"
            )

        # Early stopping on validation loss (student)
        if val_mse < best_val_loss - MIN_DELTA:
            best_val_loss = val_mse
            best_state = {k: v.detach().cpu().clone()
                          for k, v in student.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val-loss improvement for {PATIENCE} epochs). "
                    f"Best Val MSE: {best_val_loss:.6f}"
                )
                break

    # Save training history CSV
    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(OUT_DIR,
                             "training_history_coatnet1_student_64ch_distill.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"Training history saved to {hist_path}")

    if best_state is not None:
        student.load_state_dict(best_state)

    # Test student
    test_mse, test_r2, test_preds, test_trues = eval_student(
        student, test_loader, DEVICE, mse
    )
    print(f"\nStudent Test R²: {test_r2:.4f} | Test MSE: {test_mse:.6f}")

    # Save metrics
    pd.DataFrame(
        [{"metric": "test_R2", "value": float(test_r2)},
         {"metric": "test_MSE", "value": float(test_mse)}]
    ).to_csv(
        os.path.join(OUT_DIR,
                     "regression_test_metrics_coatnet1_student_64ch_distill.csv"),
        index=False
    )

    # Save predictions
    pd.DataFrame({"y_true": test_trues, "y_pred": test_preds}).to_csv(
        os.path.join(OUT_DIR,
                     "regression_test_predictions_coatnet1_student_64ch_distill.csv"),
        index=False
    )

    torch.save(student.state_dict(), STUDENT_MODEL_PATH)
    print(f"Student weights saved to {STUDENT_MODEL_PATH}")

if __name__ == "__main__":
    main()
