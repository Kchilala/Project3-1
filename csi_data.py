"""
csi_data.py
===========

Loading, labelling, per-paper feature recipes, leak-free splitting and lazy
windowing for the Wi-Fi CSI HAR dataset of [Zhuravchak22] and for the
ESP32 recordings in ``data/self_extracted_dataset``.

Key design decisions (see the notebook for the reasoning):

* **Recordings stay continuous.**  Filters run on full recordings; windows are
  cut afterwards and never cross a recording boundary.
* **Common target grid.**  Every model predicts the label of the *last*
  packet of its input window ([Zhuravchak22] Sec. 6.2).  All models are
  evaluated on the same set of target packets, each looking back over its own
  window length, so their scores are directly comparable.
* **Lazy windows.**  A window of 1024 packets x 912 features is 3.7 MB; the
  datasets slice windows on the fly from the in-memory recordings instead of
  materialising ~30 GB of overlapping copies.
* **Scalers are fitted on training recordings only.**
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from csi_preprocessing import (AmplitudeProduct, ButterworthFilter, HampelFilter,
                               KalmanSmoother, PhaseDifference, PhaseSanitizer,
                               PhaseUnwrapper, WaveletDenoiser, ht40_subcarrier_indices)

N_SUB = 114          # subcarriers per antenna pair (Atheros HT40)
N_PAIRS = 4          # 2 Tx x 2 Rx

# ---------------------------------------------------------------------------
# Tasks: how the 7 original activity labels map to class indices.
# -1 = "not used in this task".
# ---------------------------------------------------------------------------
ACTIVITIES = ["no_person", "standing", "sitting", "lying", "walking", "get_down", "get_up"]

TASKS = {
    # the original 7-class problem of [Zhuravchak22]
    "activity7": {"classes": ACTIVITIES,
                  "map": {a: i for i, a in enumerate(ACTIVITIES)}},
    # the end goal: empty room / person present but still / person moving
    "presence_motion3": {"classes": ["empty", "static", "moving"],
                         "map": {"no_person": 0, "standing": 1, "sitting": 1, "lying": 1,
                                 "walking": 2, "get_down": 2, "get_up": 2}},
    # binary presence detection
    "presence2": {"classes": ["empty", "occupied"],
                  "map": {"no_person": 0, **{a: 1 for a in ACTIVITIES[1:]}}},
    # [Shang21] binary task: static vs dynamic activities (empty room excluded)
    "static_dynamic2": {"classes": ["static", "dynamic"],
                        "map": {"no_person": -1, "standing": 0, "sitting": 0, "lying": 0,
                                "walking": 1, "get_down": 1, "get_up": 1}},
}


@dataclass
class Recording:
    name: str               # e.g. "room_1/3"
    room: str               # e.g. "room_1"
    amp: np.ndarray         # (T, 456) amplitude, 4 antenna pairs x 114 subcarriers
    phase: np.ndarray       # (T, 456) raw phase in [-pi, pi]
    labels: np.ndarray      # (T,) activity strings

    def __len__(self):
        return len(self.labels)


def load_har_dataset(root: str, cache_dir: Optional[str] = None) -> list[Recording]:
    """Load all ``room_*/<n>/data.csv`` + ``label.csv`` pairs.

    ``data.csv`` has no header and 1026 columns: 114 columns that only contain
    the subcarrier index 0..113 (dropped), 456 amplitudes, 456 phases (4
    antenna pairs x 114 subcarriers, pair-major).  ``data.csv`` has one row
    more than ``label.csv``; labels are aligned from the first row and the
    surplus last row is dropped.  Parsed arrays are cached as ``.npz``.
    """
    recs = []
    for d in sorted(glob.glob(os.path.join(root, "room_*", "*"))):
        if not os.path.isdir(d):
            continue
        room, sess = d.rstrip("/").split(os.sep)[-2:]
        name = f"{room}/{sess}"
        cache = os.path.join(cache_dir, f"{room}_{sess}.npz") if cache_dir else None
        if cache and os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            amp, phase, labels = z["amp"], z["phase"], z["labels"]
        else:
            x = pd.read_csv(os.path.join(d, "data.csv"), header=None).to_numpy(np.float32)
            y = pd.read_csv(os.path.join(d, "label.csv"), header=None)[1].to_numpy()
            assert np.all(x[:, :N_SUB] == np.arange(N_SUB)), "unexpected column layout"
            x = x[: len(y)]
            amp = x[:, N_SUB:N_SUB + N_PAIRS * N_SUB]
            phase = x[:, N_SUB + N_PAIRS * N_SUB:]
            labels = y.astype(str)
            if cache:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez(cache, amp=amp, phase=phase, labels=labels)
        recs.append(Recording(name, room, amp, phase, labels))
    return recs


def encode_labels(labels: np.ndarray, task: str) -> np.ndarray:
    m = TASKS[task]["map"]
    return np.array([m[l] for l in labels], dtype=np.int64)


# ---------------------------------------------------------------------------
# Per-paper feature recipes  (run once per recording, before windowing)
# ---------------------------------------------------------------------------
def recipe_zhuravchak(rec: Recording) -> np.ndarray:
    """[Zhuravchak22] Sec. 6.1: phase sanitisation -> Hampel -> DWT de-noising,
    amplitude: Hampel -> DWT.  Output (T, 912) = [amplitude | phase]."""
    amp = WaveletDenoiser().transform(HampelFilter().transform(rec.amp))
    ph = PhaseSanitizer(N_SUB, ht40_subcarrier_indices()).transform(rec.phase)
    ph = WaveletDenoiser().transform(HampelFilter().transform(ph))
    return np.concatenate([amp, ph], 1).astype(np.float32)


def recipe_shang(rec: Recording, n_channels: int = 4, cutoff: float = 0.1) -> np.ndarray:
    """[Shang21] Sec. 2.1: Butterworth low-pass de-noising; inputs per
    antenna group (pairs 0/1 and 2/3, i.e. the two Rx antennas of each Tx
    antenna) and subcarrier:

    * 2 channels: amplitude product |H_a||H_b|, phase difference
    * 4 channels: + |H_a|, |H_b|

    Output (T, n_channels * 2 groups * 114), channel-major.
    ``cutoff`` is a fraction of Nyquist (the dataset has no timestamps).
    """
    lp = ButterworthFilter(cutoff=cutoff, btype="low", order=4)
    amp = lp.transform(rec.amp)
    pdiff = PhaseDifference(((0, 1), (2, 3)), N_SUB, unwrap_time=True).transform(rec.phase)
    pdiff = lp.transform(pdiff)
    A = amp.reshape(len(amp), N_PAIRS, N_SUB)
    prod = AmplitudeProduct(((0, 1), (2, 3)), N_SUB).transform(amp)
    chans = [prod, pdiff]
    if n_channels == 4:
        chans = [A[:, [0, 2]].reshape(len(amp), -1), A[:, [1, 3]].reshape(len(amp), -1)] + chans
    return np.concatenate(chans, 1).astype(np.float32)


def recipe_pacsi(rec: Recording) -> np.ndarray:
    """[Quy25] Sec. 3.3: Kalman filter on the amplitude; phase unwrapped along
    time.  Deviation: the phase is first sanitised across subcarriers
    ([Sen12]); the Atheros raw phase jumps by up to +-pi between packets
    because of the CFO, which makes a pure time-unwrap meaningless (see the
    notebook plot).  Per-window mean removal of the phase happens in the
    dataset (``center_cols``).  Output (T, 912) = [amplitude | phase]."""
    amp = KalmanSmoother(process_ratio=0.05).transform(rec.amp)
    ph = PhaseSanitizer(N_SUB, ht40_subcarrier_indices()).transform(rec.phase)
    ph = PhaseUnwrapper(axis="time", remove_mean=False).transform(ph)
    return np.concatenate([amp, ph], 1).astype(np.float32)


RECIPES: dict[str, Callable[[Recording], np.ndarray]] = {
    "zhuravchak": recipe_zhuravchak,
    "shang4": lambda r: recipe_shang(r, 4),
    "shang2": lambda r: recipe_shang(r, 2),
    "pacsi": recipe_pacsi,
}


def build_features(recs: list[Recording], recipe: str, cache_dir: Optional[str] = None):
    """Apply a recipe to every recording (cached as ``.npy``)."""
    out = []
    for r in recs:
        path = os.path.join(cache_dir, f"{recipe}_{r.name.replace('/', '_')}.npy") if cache_dir else None
        if path and os.path.exists(path):
            out.append(np.load(path))
            continue
        f = RECIPES[recipe](r)
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(path, f)
        out.append(f)
    return out


class FeatureScaler:
    """Per-column standardisation (``"standard"``) or min-max scaling
    (``"minmax"``) fitted on the frames of the *training* recordings only.
    ``cols`` restricts the scaler to a subset of columns."""

    def __init__(self, kind: str = "standard", cols: Optional[slice] = None):
        self.kind, self.cols = kind, cols

    def fit(self, feats: list[np.ndarray]):
        X = np.concatenate(feats, 0)[:, self.cols or slice(None)].astype(np.float64)
        if self.kind == "standard":
            self.a, self.b = X.mean(0), X.std(0) + 1e-6
        else:
            self.a, self.b = X.min(0), X.max(0) - X.min(0) + 1e-6
        return self

    def transform(self, feats: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for f in feats:
            f = f.copy()
            c = self.cols or slice(None)
            f[:, c] = ((f[:, c] - self.a) / self.b).astype(np.float32)
            out.append(f)
        return out


# ---------------------------------------------------------------------------
# Target grid and splits
# ---------------------------------------------------------------------------
def target_grid(recs: list[Recording], step: int = 16, history: int = 1024) -> np.ndarray:
    """All (recording_id, end_index) pairs with at least ``history`` packets
    of context, every ``step`` packets.  Shared by all models."""
    rows = [(i, e) for i, r in enumerate(recs) for e in range(history - 1, len(r), step)]
    return np.array(rows, dtype=np.int64)


SPLITS = {
    # Held-out sessions: one recording per multi-session room for validation,
    # one for testing.  Every room is still seen in training.
    "cross_session": {"val": ["room_1/3", "room_3/4"], "test": ["room_1/4", "room_3/5"]},
    # Held-out room: room_2 is never seen during training.
    "cross_room": {"val": ["room_1/4", "room_3/5"], "test": ["room_2/1"]},
}


def split_targets(targets: np.ndarray, recs: list[Recording], protocol: str,
                  seed: int = 0, frac=(0.7, 0.15, 0.15)):
    """Return dict(train=..., val=..., test=...) of target rows.

    * ``"random"`` - the protocol used in the papers: targets of *all*
      recordings shuffled and split 70/15/15.  Neighbouring windows overlap by
      up to 99 %, so near-copies of test windows are in the training set: the
      score measures memorisation as much as generalisation (leaky, kept only
      to compare with the published numbers).
    * ``"cross_session"`` / ``"cross_room"`` - whole recordings are held out
      (see ``SPLITS``); no overlap between splits is possible.
    """
    if protocol == "random":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(targets))
        n_tr, n_va = int(frac[0] * len(idx)), int(frac[1] * len(idx))
        return {"train": targets[idx[:n_tr]], "val": targets[idx[n_tr:n_tr + n_va]],
                "test": targets[idx[n_tr + n_va:]]}
    names = np.array([r.name for r in recs])
    rec_of = names[targets[:, 0]]
    va, te = SPLITS[protocol]["val"], SPLITS[protocol]["test"]
    return {"train": targets[~np.isin(rec_of, va + te)],
            "val": targets[np.isin(rec_of, va)],
            "test": targets[np.isin(rec_of, te)]}


def train_recording_ids(split: dict, n_recs: int) -> list[int]:
    """Recordings that contribute training targets (for fitting scalers)."""
    return sorted(set(split["train"][:, 0].tolist()))


# ---------------------------------------------------------------------------
# Lazy window dataset
# ---------------------------------------------------------------------------
class WindowDataset(Dataset):
    """Windows ``feats[r][e-window+1 : e+1]`` with label ``labels[r][e]``.

    Parameters
    ----------
    feats : list of (T_r, F) float32 arrays (one per recording)
    labels : list of (T_r,) int arrays (task-encoded, -1 = ignore)
    targets : (N, 2) array of (recording id, end index)
    window : int
        Window length in packets.
    align_to : int or None
        Average-pool the window to this many steps ([Quy25] time alignment).
    center_cols : slice or None
        Columns whose per-window mean is subtracted (unwrapped phase).
    """

    def __init__(self, feats, labels, targets, window: int,
                 align_to: Optional[int] = None, center_cols: Optional[slice] = None):
        keep = np.array([labels[r][e] >= 0 for r, e in targets], dtype=bool)
        self.targets = targets[keep]
        self.feats = [torch.from_numpy(np.ascontiguousarray(f)) for f in feats]
        self.labels = labels
        self.window, self.align_to, self.center_cols = window, align_to, center_cols

    def __len__(self):
        return len(self.targets)

    @property
    def y(self) -> np.ndarray:
        return np.array([self.labels[r][e] for r, e in self.targets])

    def __getitem__(self, i):
        r, e = self.targets[i]
        x = self.feats[r][e - self.window + 1: e + 1]
        if self.center_cols is not None:
            x = x.clone()
            x[:, self.center_cols] -= x[:, self.center_cols].mean(0, keepdim=True)
        if self.align_to is not None and self.align_to < self.window:
            x = torch.nn.functional.adaptive_avg_pool1d(x.T[None], self.align_to)[0].T
        return x, int(self.labels[r][e])


# ---------------------------------------------------------------------------
# Hand-crafted window statistics for the classical baselines (KNN / SVM)
# ---------------------------------------------------------------------------
def window_statistics(feats, targets, window: int, n_groups: int) -> np.ndarray:
    """Per window: average the subcarriers of each channel group to one
    series, then take 8 statistics (mean, std, min, max, range, mean |diff|,
    skewness, kurtosis) plus the mean per-subcarrier temporal std.
    ``n_groups`` = number of channel groups in the feature matrix."""
    out = []
    for r, e in targets:
        x = feats[r][e - window + 1: e + 1].astype(np.float64)
        G = x.reshape(len(x), n_groups, -1)
        s = G.mean(-1)                                   # (window, n_groups)
        mu, sd = s.mean(0), s.std(0) + 1e-9
        z = (s - mu) / sd
        stats = [mu, sd, s.min(0), s.max(0), s.max(0) - s.min(0),
                 np.abs(np.diff(s, axis=0)).mean(0), (z ** 3).mean(0), (z ** 4).mean(0),
                 G.std(0).mean(-1)]
        out.append(np.concatenate(stats))
    return np.array(out)
