"""
csi_experiments.py
==================

One function, :func:`run_har`, runs one (model, task, protocol, seed)
experiment on the HAR dataset end-to-end and caches the result in
``results/``.  The notebook calls it cell by cell; ``run_all.py`` calls it
for the whole experiment grid in parallel processes.  Both produce the same
JSON files, so a notebook re-run only loads them (set ``force=True`` to
retrain).

All hyper-parameters live in :data:`MODEL_SPECS` - each entry documents
where the value comes from (paper) or why it deviates (CPU budget).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from csi_data import (TASKS, FeatureScaler, WindowDataset, build_features, encode_labels,
                      load_har_dataset, split_targets, target_grid, window_statistics)
from csi_models import (PACSI, InceptionTime, ShangLSTM, ShangLSTMCNN, ZhuravchakBiLSTM,
                        count_parameters)
from csi_train import cached_run, metrics, predict, save_model, train_torch_model

HERE = os.path.dirname(os.path.abspath(__file__))
# data/wifi_csi_har_dataset (from download_data.py) or the original "(3)" folder name
DATA_ROOT = next((os.path.join(HERE, "data", d) for d in ("wifi_csi_har_dataset", "wifi_csi_har_dataset (3)")
                  if os.path.isdir(os.path.join(HERE, "data", d))),
                 os.path.join(HERE, "data", "wifi_csi_har_dataset"))
CACHE = os.path.join(HERE, "cache")
GRID_STEP = 16          # one target packet every 16 packets
HISTORY = 1024          # longest window -> first usable target per recording

# every 4th subcarrier of every antenna pair (29 per pair), used by PA-CSI
SUB4 = np.concatenate([np.arange(p * 114, (p + 1) * 114, 4) for p in range(4)])
PACSI_COLS = np.concatenate([SUB4, 456 + SUB4])          # amplitude | phase

MODEL_SPECS = {
    # ---------------- [Shang21] ----------------
    "lstm_cnn": dict(  # core model: 4-channel input, Sec. 3.2 (4ch > 2ch)
        recipe="shang4", window=400, kind="torch", batch_size=10, lr=1e-3,
        epochs=60, patience=12, max_train_samples=3000,
        build=lambda F, C: ShangLSTMCNN(F, C)),
    "lstm_cnn_2ch": dict(  # [Shang21] 2-channel input variant
        recipe="shang2", window=400, kind="torch", batch_size=10, lr=1e-3,
        epochs=60, patience=12, max_train_samples=3000,
        build=lambda F, C: ShangLSTMCNN(F, C)),
    "lstm": dict(  # [Shang21] LSTM baseline (same LSTM, no CNN)
        recipe="shang4", window=400, kind="torch", batch_size=10, lr=1e-3,
        epochs=60, patience=12, max_train_samples=3000,
        build=lambda F, C: ShangLSTM(F, C)),
    "knn": dict(recipe="shang4", window=400, kind="sklearn", n_groups=8,
                build=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5))),
    "svm": dict(recipe="shang4", window=400, kind="sklearn", n_groups=8,
                build=lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale",
                                                                  class_weight="balanced"))),
    # ---------------- [Zhuravchak22] ----------------
    "bilstm": dict(  # 1024-packet window, 16/8 chunks, BiLSTM-256, dense 512/256/128
        # deviation: linear projection 14592 -> 256 per chunk (faithful = 0.7 s/sample on CPU)
        recipe="zhuravchak", window=1024, kind="torch", batch_size=16, lr=1e-3,
        epochs=40, patience=10, max_train_samples=2000, weight_decay=1e-4,
        build=lambda F, C: ZhuravchakBiLSTM(F, C, input_proj=256, dropout=0.3)),
    "inceptiontime": dict(  # 1024-packet window
        # deviation: time-aligned to 256 steps (4-packet averages), 1 network instead of 5
        recipe="zhuravchak", window=1024, align_to=256, kind="torch", batch_size=16,
        lr=1e-3, epochs=40, patience=10, max_train_samples=2000,
        build=lambda F, C: InceptionTime(F, C)),
    # ---------------- [Quy25] PA-CSI ----------------
    "pacsi": dict(  # window 1000 (Table 3, own dataset), K=10 Gaussians, 4 MCAT layers,
        # GRN 256, dropout 0.1, Adam 1e-3 with decay.  deviations: time-aligned to
        # 100 steps, every 4th subcarrier (232 inputs, paper: 90), batch 16, 40 epochs
        recipe="pacsi", window=1000, align_to=100, cols=PACSI_COLS, center_phase=True,
        kind="torch", batch_size=16, lr=1e-3, lr_decay=0.95, epochs=40, patience=10,
        max_train_samples=2000,
        build=lambda F, C, **kw: PACSI(100, F // 2, F // 2, C, **kw)),
}
# PA-CSI ablations ([Quy25] claims: phase helps, GRE and GRN help)
for _name, _kw in {"pacsi_amp_only": dict(use_phase=False),
                   "pacsi_phase_only": dict(use_amp=False),
                   "pacsi_no_gre": dict(use_gre=False),
                   "pacsi_no_grn": dict(use_grn=False)}.items():
    MODEL_SPECS[_name] = {**MODEL_SPECS["pacsi"], "model_kwargs": _kw}


@lru_cache(maxsize=1)
def get_recordings():
    return load_har_dataset(DATA_ROOT, CACHE)


@lru_cache(maxsize=8)
def get_features(recipe: str):
    return build_features(get_recordings(), recipe, CACHE)


def prepare(model: str, task: str, protocol: str, seed: int = 0):
    """Build scaled features, labels and the train/val/test window datasets."""
    spec = MODEL_SPECS[model]
    recs = get_recordings()
    feats = get_features(spec["recipe"])
    if spec.get("cols") is not None:
        feats = [f[:, spec["cols"]] for f in feats]
    labels = [encode_labels(r.labels, task) for r in recs]
    split = split_targets(target_grid(recs, GRID_STEP, HISTORY), recs, protocol, seed)
    train_recs = sorted(set(split["train"][:, 0].tolist()))
    n_feat = feats[0].shape[1]
    if spec["recipe"] == "pacsi":
        # [Quy25] Eq. 6: min-max on the amplitude; phase is centred per window
        sc = FeatureScaler("minmax", slice(0, n_feat // 2)).fit([feats[i] for i in train_recs])
    else:
        sc = FeatureScaler("standard").fit([feats[i] for i in train_recs])
    feats = sc.transform(feats)
    center = slice(n_feat // 2, n_feat) if spec.get("center_phase") else None
    ds = {k: WindowDataset(feats, labels, v, spec["window"], spec.get("align_to"), center)
          for k, v in split.items()}
    return spec, feats, labels, ds, n_feat


def run_har(model: str, task: str, protocol: str, seed: int = 0, force: bool = False,
            verbose: bool = True, threads: int | None = None, **overrides) -> dict:
    """Train + evaluate one model; cached in ``results/har__{model}__{task}__{protocol}__s{seed}.json``."""
    name = f"har__{model}__{task}__{protocol}__s{seed}"

    def _run():
        if threads:
            torch.set_num_threads(threads)
        spec, feats, labels, ds, n_feat = prepare(model, task, protocol, seed)
        spec = {**spec, **overrides}
        classes = TASKS[task]["classes"]
        C = len(classes)
        out = {"model": model, "task": task, "protocol": protocol, "seed": seed,
               "classes": classes, "sizes": {k: len(v) for k, v in ds.items()}}
        if spec["kind"] == "sklearn":
            Xs = {k: window_statistics(feats, v.targets, spec["window"], spec["n_groups"])
                  for k, v in ds.items()}
            clf = spec["build"]().fit(Xs["train"], ds["train"].y)
            for k in ("val", "test"):
                out[k] = metrics(ds[k].y, clf.predict(Xs[k]), C)
            out["test_pred"] = clf.predict(Xs["test"]).tolist()
        else:
            torch.manual_seed(seed)
            net = spec["build"](n_feat, C, **spec.get("model_kwargs", {}))
            out["n_params"] = count_parameters(net)
            tr = train_torch_model(
                net, ds["train"], ds["val"], C, epochs=spec["epochs"],
                batch_size=spec["batch_size"], lr=spec["lr"], patience=spec["patience"],
                max_train_samples=spec.get("max_train_samples"),
                weight_decay=spec.get("weight_decay", 0.0), lr_decay=spec.get("lr_decay"),
                seed=seed, verbose=verbose, time_budget_s=spec.get("time_budget_s", 3600))
            out.update(tr)
            yp = predict(net, ds["test"])
            out["val"] = metrics(ds["val"].y, predict(net, ds["val"]), C)
            out["test"] = metrics(ds["test"].y, yp, C)
            out["test_pred"] = yp.tolist()
            save_model(net, name)
        out["test_true"] = ds["test"].y.tolist()
        out["test_targets"] = ds["test"].targets.tolist()
        return out

    res = cached_run(name, _run, force=force)
    if verbose:
        t = res["test"]
        print(f"{name}: test acc {t['accuracy']:.3f}  macro-F1 {t['f1_macro']:.3f}")
    return res
