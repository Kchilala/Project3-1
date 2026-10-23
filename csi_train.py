"""
csi_train.py
============

Training / evaluation helpers shared by all experiments.

* :func:`train_torch_model` - Adam + class-weighted cross-entropy, model
  selection on the validation macro-F1 (early stopping), deterministic seeds.
* :func:`evaluate` - predictions + metrics (accuracy, macro precision /
  recall / F1, confusion matrix).
* :func:`get_device` - Intel GPU (xpu) > NVIDIA (cuda) > CPU.
* :func:`cached_run` - stores every experiment's metrics as JSON in
  ``results/`` (and weights in ``models/``) so the notebook can be re-run
  interactively without re-training; pass ``force=True`` to retrain.
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader, Subset

RESULTS_DIR = "results"
MODELS_DIR = "models"


def get_device() -> torch.device:
    """Pick the accelerator: Intel GPU (``xpu``, e.g. Arc A370M), NVIDIA
    (``cuda``), else CPU.  Override with the environment variable
    ``CSI_DEVICE`` (e.g. ``CSI_DEVICE=cpu``)."""
    forced = os.environ.get("CSI_DEVICE")
    if forced:
        return torch.device(forced)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(y_true, y_pred, n_classes: int) -> dict:
    labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


@torch.no_grad()
def predict(model: nn.Module, ds, batch_size: int = 64) -> np.ndarray:
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for x, _ in DataLoader(ds, batch_size=batch_size, shuffle=False):
        out.append(model(x.to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=int)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights (normalised to mean 1) against imbalance
    (walking = 39 % of the HAR data, get_up / get_down ~ 3-4 %)."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = counts.sum() / np.maximum(counts, 1) / n_classes
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def train_torch_model(model: nn.Module, train_ds, val_ds, n_classes: int, *,
                      epochs: int = 30, batch_size: int = 32, lr: float = 1e-3,
                      weight_decay: float = 0.0, patience: int = 8,
                      max_train_samples: Optional[int] = None, grad_clip: float = 5.0,
                      lr_decay: Optional[float] = None, seed: int = 0,
                      weighted_loss: bool = True, verbose: bool = True,
                      time_budget_s: Optional[float] = None,
                      device: Optional[torch.device] = None) -> dict:
    """Train with Adam, keep the weights with the best validation macro-F1.

    ``max_train_samples`` draws a fresh random subset of the training windows
    every epoch (keeps the epoch time bounded on a CPU while still seeing all
    windows over the epochs).  ``lr_decay`` = multiplicative decay per epoch
    ([Quy25]: "decay rate = 0.9").
    """
    set_seed(seed)
    dev = device or DEVICE
    model.to(dev)
    y_tr = train_ds.y
    crit = nn.CrossEntropyLoss(weight=class_weights(y_tr, n_classes).to(dev) if weighted_loss else None)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay) if lr_decay else None
    rng = np.random.default_rng(seed)
    best = {"f1": -1, "state": None, "epoch": -1}
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        idx = np.arange(len(train_ds))
        if max_train_samples and len(idx) > max_train_samples:
            idx = rng.choice(idx, max_train_samples, replace=False)
        loader = DataLoader(Subset(train_ds, idx), batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed + ep))
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        if sched:
            sched.step()
        yp = predict(model, val_ds)
        m = metrics(val_ds.y, yp, n_classes)
        hist.append({"epoch": ep, "loss": tot / max(n, 1), "val_acc": m["accuracy"],
                     "val_f1": m["f1_macro"], "time_s": time.time() - t0})
        if verbose:
            print(f"  ep {ep:3d}  loss {tot / max(n, 1):.4f}  val acc {m['accuracy']:.3f}"
                  f"  val F1 {m['f1_macro']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if m["f1_macro"] > best["f1"]:
            best = {"f1": m["f1_macro"], "state": copy.deepcopy(model.state_dict()), "epoch": ep}
        if ep - best["epoch"] >= patience:
            break
        if time_budget_s and time.time() - t0 > time_budget_s:
            if verbose:
                print("  time budget reached")
            break
    model.load_state_dict(best["state"])
    return {"history": hist, "best_epoch": best["epoch"], "train_time_s": time.time() - t0,
            "device": str(dev)}


def cached_run(name: str, fn: Callable[[], dict], force: bool = False) -> dict:
    """Run ``fn`` (which returns a JSON-serialisable dict) once and cache it
    in ``results/<name>.json``."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)
    res = fn()
    res["name"] = name
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    return res


def save_model(model: nn.Module, name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(MODELS_DIR, f"{name}.pt"))
