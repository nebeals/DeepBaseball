"""
train.py
--------
Training loop, evaluation, calibration, and inference script
for the MLB win probability model.

Covers:
  - MLBDataset     : PyTorch Dataset wrapping the feature matrix CSV
  - Trainer        : training loop with early stopping, LR scheduling,
                     gradient clipping, and checkpoint saving
  - evaluate()     : log loss, Brier score, accuracy, and calibration curve
  - calibrate()    : Platt scaling (isotonic available) post-hoc calibration
  - predict()      : run inference on new games

Usage (CLI):
    # Train with defaults (MLP, window=10, val=2022, test=2023)
    python src/train.py

    # Train a ResNet with a 15-game window
    python src/train.py --arch resnet --features data/features/features_w15.csv

    # Evaluate a saved checkpoint
    python src/train.py --eval_only --checkpoint checkpoints/best_mlp.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from model import WinProbMLP, WinProbResNet, build_model, count_parameters

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent
FEATURES_DIR = ROOT / "data" / "features"
CKPT_DIR     = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ── Feature columns (must match features.py exactly) ─────────────────────────

META_COLS   = ["game_date", "home_team", "away_team", "season"]
TARGET_COL  = "home_win"

FEATURE_COLS = [
    "h_run_rate", "h_ra_rate", "h_run_diff_rate", "h_win_rate",
    "h_scoring_var", "h_last3_win_rate", "h_quality_start_rt", "h_ops", "h_streak",
    "a_run_rate", "a_ra_rate", "a_run_diff_rate", "a_win_rate",
    "a_scoring_var", "a_last3_win_rate", "a_quality_start_rt", "a_ops", "a_streak",
    "diff_run_rate", "diff_ra_rate", "diff_run_diff_rate",
    "diff_win_rate", "diff_ops", "diff_streak", "diff_last3_win_rate",
    "home_advantage", "is_covid_season",
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
    "month_4", "month_5", "month_6", "month_7", "month_8", "month_9", "month_10",
]


# ── Dataset ───────────────────────────────────────────────────────────────────

class MLBDataset(Dataset):
    """
    PyTorch Dataset for the MLB feature matrix.

    Loads the CSV, filters to a subset of seasons, normalizes features
    using statistics computed from the training split only (no leakage),
    and serves (features, label) pairs.

    Parameters
    ----------
    df          : feature matrix DataFrame (output of features.py)
    seasons     : which seasons to include in this split
    scaler_mean : pre-computed mean for normalization (None = compute from data)
    scaler_std  : pre-computed std  for normalization (None = compute from data)
    """
    def __init__(
        self,
        df:          pd.DataFrame,
        seasons:     list[int],
        scaler_mean: Optional[np.ndarray] = None,
        scaler_std:  Optional[np.ndarray] = None,
    ) -> None:
        split = df[df["season"].isin(seasons)].copy()
        split = split.sort_values("game_date").reset_index(drop=True)

        X = split[FEATURE_COLS].values.astype(np.float32)
        y = split[TARGET_COL].values.astype(np.float32)

        # Normalize continuous features; leave binary/one-hot columns as-is
        # Binary columns: home_advantage, is_covid_season, dow_*, month_*
        self._binary_start = FEATURE_COLS.index("home_advantage")

        if scaler_mean is None:
            cont = X[:, :self._binary_start]
            self.mean = cont.mean(axis=0)
            self.std  = cont.std(axis=0).clip(min=1e-6)
        else:
            self.mean = scaler_mean
            self.std  = scaler_std

        X_norm = X.copy()
        X_norm[:, :self._binary_start] = (
            (X[:, :self._binary_start] - self.mean) / self.std
        )

        self.X    = torch.tensor(X_norm, dtype=torch.float32)
        self.y    = torch.tensor(y,      dtype=torch.float32).unsqueeze(1)
        self.meta = split[META_COLS].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.X[idx], self.y[idx]


# ── Metrics ───────────────────────────────────────────────────────────────────

def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob >= 0.5).astype(int) == y_true.astype(int)))


def calibration_stats(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration statistics (Expected Calibration Error + bin data).
    A perfectly calibrated model has ECE = 0: if it says 70%, it wins 70%.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        avg_conf  = y_prob[mask].mean()
        avg_acc   = y_true[mask].mean()
        bin_count = mask.sum()
        ece      += (bin_count / len(y_true)) * abs(avg_conf - avg_acc)
        bin_data.append({
            "bin_mid":  (lo + hi) / 2,
            "avg_conf": avg_conf,
            "avg_acc":  avg_acc,
            "count":    int(bin_count),
        })

    return {"ece": ece, "bins": bin_data}


def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    label:      str = "Eval",
    calibrator: object | None = None,
) -> dict:
    """
    Run full evaluation: log loss, Brier, accuracy, ECE.

    Parameters
    ----------
    calibrator : if provided, applies post-hoc calibration to raw probs
                 (must have a .predict_proba method like sklearn calibrators)
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs  = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.append(probs)
            all_labels.append(yb.numpy().ravel())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    if calibrator is not None:
        y_prob = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]

    ll  = log_loss(y_true, y_prob)
    bs  = brier_score(y_true, y_prob)
    acc = accuracy(y_true, y_prob)
    cal = calibration_stats(y_true, y_prob)

    print(f"  [{label}]  "
          f"log_loss={ll:.4f}  brier={bs:.4f}  "
          f"acc={acc:.4f}  ECE={cal['ece']:.4f}")

    return {
        "log_loss": ll,
        "brier":    bs,
        "accuracy": acc,
        "ece":      cal["ece"],
        "cal_bins": cal["bins"],
        "y_prob":   y_prob,
        "y_true":   y_true,
    }


# ── Calibration ───────────────────────────────────────────────────────────────

class PlattWrapper:
    """Picklable wrapper around a fitted LogisticRegression Platt scaler."""
    def __init__(self, lr): self.lr = lr
    def predict_proba(self, X): return self.lr.predict_proba(X)


class IsotonicWrapper:
    """Picklable wrapper around a fitted IsotonicRegression calibrator."""
    def __init__(self, ir): self.ir = ir
    def predict_proba(self, X):
        p = self.ir.predict(X.ravel())
        return np.stack([1 - p, p], axis=1)


def calibrate(
    model:      nn.Module,
    val_loader: DataLoader,
    device:     torch.device,
    method:     str = "platt",
) -> object:
    """
    Fit a post-hoc calibrator on the validation set.

    Parameters
    ----------
    method : "platt" (logistic regression on logits — fast, reliable)
             "isotonic" (non-parametric, more flexible but can overfit)

    Returns
    -------
    Fitted sklearn calibrator with a .predict_proba(X) method.
    The calibrator expects raw probabilities as input (not logits).
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    model.eval()
    raw_probs, labels = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy().ravel()
            raw_probs.append(probs)
            labels.append(yb.numpy().ravel())

    raw_probs = np.concatenate(raw_probs).reshape(-1, 1)
    labels    = np.concatenate(labels)

    if method == "platt":
        # Logistic regression on raw probability (1 feature) — Platt scaling
        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        cal.fit(raw_probs, labels)
        fitted = PlattWrapper(cal)

    elif method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs.ravel(), labels)
        fitted = IsotonicWrapper(iso)

    else:
        raise ValueError(f"Unknown calibration method: {method}")

    print(f"  Calibrator fitted ({method}) on {len(labels):,} val samples.")
    return fitted


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full training lifecycle for a single model.

    Features:
      - BCEWithLogitsLoss (numerically stable)
      - AdamW optimizer with weight decay
      - OneCycleLR scheduler (warms up then decays — great for tabular data)
      - Early stopping on validation log loss
      - Gradient clipping (prevents occasional exploding gradients)
      - Checkpoint saving (best val loss)
      - Training history logged to JSON

    Parameters
    ----------
    model          : the nn.Module to train
    train_loader   : DataLoader for training split
    val_loader     : DataLoader for validation split
    device         : torch.device
    lr             : peak learning rate (OneCycleLR)
    weight_decay   : L2 regularization strength
    max_epochs     : hard upper bound on training epochs
    patience       : early stopping patience (epochs without improvement)
    grad_clip      : max gradient norm (0 = disabled)
    checkpoint_path: where to save the best model weights
    """
    def __init__(
        self,
        model:           nn.Module,
        train_loader:    DataLoader,
        val_loader:      DataLoader,
        device:          torch.device,
        lr:              float = 1e-3,
        weight_decay:    float = 1e-4,
        max_epochs:      int   = 100,
        patience:        int   = 10,
        grad_clip:       float = 1.0,
        checkpoint_path: Path  = CKPT_DIR / "best_model.pt",
    ) -> None:
        self.model       = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device      = device
        self.max_epochs  = max_epochs
        self.patience    = patience
        self.grad_clip   = grad_clip
        self.ckpt_path   = Path(checkpoint_path)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # OneCycleLR: warms up for ~30% of steps, then decays
        total_steps = max_epochs * len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
        )

        self.history: list[dict] = []
        self.best_val_loss  = math.inf
        self.epochs_no_impr = 0
        # These are injected by main() after dataset construction
        self.arch         = "mlp"
        self.input_dim    = 41
        self.model_kwargs = {}
        self.scaler_mean  = None
        self.scaler_std   = None
        self.calibrator   = None

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for xb, yb in self.train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(xb), yb)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item() * len(xb)
        return total_loss / len(self.train_loader.dataset)

    def _val_loss(self) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                total += self.criterion(self.model(xb), yb).item() * len(xb)
        return total / len(self.val_loader.dataset)

    def fit(self) -> list[dict]:
        """
        Run the training loop. Returns the per-epoch history list.
        """
        print(f"\n{'═'*60}")
        print(f"  Training {self.model.__class__.__name__}")
        print(f"  Parameters: {count_parameters(self.model):,}")
        print(f"  Device:     {self.device}")
        print(f"  Max epochs: {self.max_epochs}  |  Patience: {self.patience}")
        print(f"{'═'*60}")

        t0 = time.time()
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            val_loss   = self._val_loss()
            lr_now     = self.scheduler.get_last_lr()[0]

            improved = val_loss < self.best_val_loss - 1e-5
            if improved:
                self.best_val_loss  = val_loss
                self.epochs_no_impr = 0
                torch.save({
                    "epoch":        epoch,
                    "state_dict":   self.model.state_dict(),
                    "val_loss":     val_loss,
                    "train_loss":   train_loss,
                    # ── everything needed to reload without extra files ──
                    "arch":         self.arch,
                    "input_dim":    self.input_dim,
                    "model_kwargs": self.model_kwargs,
                    "scaler_mean":  self.scaler_mean,
                    "scaler_std":   self.scaler_std,
                    "feature_cols": FEATURE_COLS,
                    "calibrator":   self.calibrator,   # None until set
                }, self.ckpt_path)
            else:
                self.epochs_no_impr += 1

            marker = " ✓" if improved else ""
            print(f"  Epoch {epoch:3d}/{self.max_epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={lr_now:.2e}{marker}")

            self.history.append({
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "lr":         lr_now,
            })

            if self.epochs_no_impr >= self.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs).")
                break

        elapsed = time.time() - t0
        print(f"\n  Training complete in {elapsed:.1f}s")
        print(f"  Best val loss: {self.best_val_loss:.4f}  "
              f"(saved → {self.ckpt_path})")

        # Save history
        hist_path = self.ckpt_path.with_suffix(".history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  History    → {hist_path}")

        return self.history


# ── Inference ─────────────────────────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path: str | Path,
    device:          torch.device | None = None,
    # Legacy kwargs kept for backward compat — ignored if arch is in the checkpoint
    arch:            str = "mlp",
    input_dim:       int = 41,
    **model_kwargs,
) -> tuple[nn.Module, dict]:
    """
    Load a saved model from a self-contained checkpoint.

    Returns
    -------
    (model, meta)  where meta contains:
        arch, input_dim, scaler_mean, scaler_std,
        feature_cols, calibrator, epoch, val_loss
    """
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Support both old (no arch key) and new self-contained format
    _arch      = ckpt.get("arch",         arch)
    _input_dim = ckpt.get("input_dim",    input_dim)
    _kwargs    = ckpt.get("model_kwargs", model_kwargs)

    model = build_model(_arch, input_dim=_input_dim, **_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    print(f"Loaded '{_arch}' checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f})")

    meta = {
        "arch":         _arch,
        "input_dim":    _input_dim,
        "model_kwargs": _kwargs,
        "scaler_mean":  ckpt.get("scaler_mean"),
        "scaler_std":   ckpt.get("scaler_std"),
        "feature_cols": ckpt.get("feature_cols", FEATURE_COLS),
        "calibrator":   ckpt.get("calibrator"),
        "epoch":        ckpt["epoch"],
        "val_loss":     ckpt["val_loss"],
    }
    return model, meta


def predict(
    model:       nn.Module,
    features_df: pd.DataFrame,
    scaler_mean: np.ndarray,
    scaler_std:  np.ndarray,
    device:      torch.device,
    calibrator:  object | None = None,
    batch_size:  int = 512,
) -> np.ndarray:
    """
    Run inference on a DataFrame of games.

    Parameters
    ----------
    features_df  : DataFrame with the same FEATURE_COLS used during training
    scaler_mean  : training-set normalization mean (saved alongside checkpoint)
    scaler_std   : training-set normalization std
    calibrator   : optional post-hoc calibrator from calibrate()

    Returns
    -------
    np.ndarray of shape (n_games,) — home team win probability for each game
    """
    X = features_df[FEATURE_COLS].values.astype(np.float32)
    binary_start = FEATURE_COLS.index("home_advantage")
    X[:, :binary_start] = (X[:, :binary_start] - scaler_mean) / scaler_std

    model.eval()
    probs_list = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch  = torch.tensor(X[start:start + batch_size]).to(device)
            logits = model(batch)
            probs_list.append(torch.sigmoid(logits).cpu().numpy().ravel())

    probs = np.concatenate(probs_list)

    if calibrator is not None:
        probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

    return probs


# ── Main training script ──────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    )
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    features_path = Path(args.features)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}\n"
            f"Run: python src/features.py first."
        )

    print(f"\nLoading features from {features_path}...")
    df = pd.read_csv(features_path, parse_dates=["game_date"])
    print(f"  {len(df):,} games  |  {len(FEATURE_COLS)} features")

    all_seasons  = sorted(df["season"].unique().tolist())
    test_season  = args.test_season
    val_season   = args.val_season
    train_seasons = [s for s in all_seasons if s not in (val_season, test_season)]

    print(f"  Train seasons: {train_seasons}")
    print(f"  Val season:    {val_season}")
    print(f"  Test season:   {test_season}")

    # ── Build datasets ────────────────────────────────────────────────────────
    train_ds = MLBDataset(df, train_seasons)
    val_ds   = MLBDataset(df, [val_season],  train_ds.mean, train_ds.std)
    test_ds  = MLBDataset(df, [test_season], train_ds.mean, train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)

    # ── Build model ───────────────────────────────────────────────────────────
    input_dim = len(FEATURE_COLS)
    model = build_model(args.arch, input_dim=input_dim)

    ckpt_path = CKPT_DIR / f"best_{args.arch}.pt"

    if args.eval_only:
        # ── Evaluation-only mode ──────────────────────────────────────────────
        if args.checkpoint:
            model, meta = load_checkpoint(args.checkpoint, device=device)
        else:
            raise ValueError("--eval_only requires --checkpoint path.")

        print("\nEvaluating on validation set:")
        evaluate(model, val_loader, device, label="Val")
        print("\nEvaluating on test set:")
        evaluate(model, test_loader, device, label="Test")
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        device           = device,
        lr               = args.lr,
        weight_decay     = args.weight_decay,
        max_epochs       = args.epochs,
        patience         = args.patience,
        grad_clip        = args.grad_clip,
        checkpoint_path  = ckpt_path,
    )
    # Inject metadata so the checkpoint is self-contained
    trainer.arch         = args.arch
    trainer.input_dim    = input_dim
    trainer.scaler_mean  = train_ds.mean
    trainer.scaler_std   = train_ds.std
    trainer.fit()

    # ── Reload best weights ───────────────────────────────────────────────────
    print("\nReloading best checkpoint for evaluation...")
    model, meta = load_checkpoint(ckpt_path, device=device)

    # ── Evaluate pre-calibration ──────────────────────────────────────────────
    print("\n── Pre-calibration metrics ──────────────────────────────")
    print("  Validation:")
    val_results = evaluate(model, val_loader, device, label="Val")
    print("  Test:")
    test_results = evaluate(model, test_loader, device, label="Test")

    # ── Calibrate ─────────────────────────────────────────────────────────────
    calibrator = None
    if args.calibrate:
        print(f"\n── Calibration ({args.cal_method}) ──────────────────────────────")
        calibrator = calibrate(model, val_loader, device, method=args.cal_method)

        print("\n── Post-calibration metrics ─────────────────────────────")
        print("  Validation:")
        evaluate(model, val_loader, device, label="Val (cal)", calibrator=calibrator)
        print("  Test:")
        evaluate(model, test_loader, device, label="Test (cal)", calibrator=calibrator)

        # Re-save checkpoint with calibrator bundled in
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt["calibrator"] = calibrator
        torch.save(ckpt, ckpt_path)
        print(f"\n  Calibrator bundled into checkpoint → {ckpt_path}")

    # ── Save normalization stats (needed for inference later) ─────────────────
    norm_path = ckpt_path.with_suffix(".norm.npz")
    np.savez(norm_path, mean=train_ds.mean, std=train_ds.std)
    print(f"  Normalization stats → {norm_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    baseline_ll = log_loss(
        test_results["y_true"],
        np.full_like(test_results["y_true"], test_results["y_true"].mean())
    )
    print(f"\n{'═'*60}")
    print(f"  Final Results ({args.arch})")
    print(f"{'═'*60}")
    print(f"  Test log loss:   {test_results['log_loss']:.4f}  "
          f"(baseline={baseline_ll:.4f}, "
          f"improvement={baseline_ll - test_results['log_loss']:.4f})")
    print(f"  Test Brier:      {test_results['brier']:.4f}")
    print(f"  Test accuracy:   {test_results['accuracy']:.4f}")
    print(f"  Test ECE:        {test_results['ece']:.4f}")
    print(f"{'═'*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MLB win probability model.")

    # Data
    p.add_argument("--features",    default=str(FEATURES_DIR / "features_w10.csv"),
                   help="Path to feature matrix CSV (from features.py)")
    p.add_argument("--val_season",  type=int, default=2022)
    p.add_argument("--test_season", type=int, default=2023)

    # Model
    p.add_argument("--arch", default="mlp",
                   choices=["mlp", "resnet", "ensemble_mlp", "ensemble_resnet"],
                   help="Model architecture")

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--patience",     type=int,   default=12)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--cpu",          action="store_true", help="Force CPU training")

    # Calibration
    p.add_argument("--calibrate",   action="store_true", default=True,
                   help="Apply post-hoc calibration (default: on)")
    p.add_argument("--no_calibrate", dest="calibrate", action="store_false")
    p.add_argument("--cal_method",  default="platt", choices=["platt", "isotonic"])

    # Eval-only mode
    p.add_argument("--eval_only",   action="store_true")
    p.add_argument("--checkpoint",  default=None, help="Path to .pt checkpoint")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args)