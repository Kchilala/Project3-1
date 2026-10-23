"""
csi_esp32.py
============

Pipeline for the CSI recorded with our own ESP32 setup
(``data/self_extracted_dataset/*.csv``) and for future live data.

Steps (each one a documented class from ``csi_preprocessing``):

1. :class:`ESP32CSIParser`   - keep one transmitter (MAC), 64-subcarrier
   packets, drop the 12 null subcarriers -> 52 amplitudes per packet.
2. :class:`UniformResampler` - irregular packet arrival (~3-5 Hz, gaps up to
   2 s) -> uniform ``fs`` Hz grid ([Chang23] Sec. III-A).
3. :class:`MinMaxNormalizer` (``per_sample``) - removes the per-packet gain
   (ESP32 automatic gain control) and the absolute signal level
   ([Chang23] Eq. 3).  This prevents the classifier from simply learning the
   received power, which differs strongly between our recordings.
4. Windows of ``window`` samples ending at every grid point.

Evaluation: with one recording per class, a random split would put
near-identical overlapping windows in train and test.  We therefore use
*blocked* K-fold cross-validation over time: each recording is cut into K
contiguous blocks; block k of every recording is the test set of fold k, and
``window`` samples on both sides of it are removed from training (buffer).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from csi_preprocessing import ESP32CSIParser, MinMaxNormalizer, UniformResampler

ESP32_CLASSES = ["empty", "static", "moving"]
ESP32_FILES = {"empty": "empty_room.csv", "static": "standing_still.csv", "moving": "moving.csv"}


@dataclass
class ESP32Recording:
    label: int
    t: np.ndarray        # (T,) uniform time grid [s]
    amp: np.ndarray      # (T, 52) raw amplitude
    rssi: np.ndarray     # (T,)  RSSI [dBm]
    gap: np.ndarray      # (T,)  True where the grid point lies in a >max_gap hole
    n_packets: int       # packets of the selected MAC before resampling


def load_esp32(folder: str, fs: float = 5.0, max_gap: float = 1.0, mac=None):
    """Load the three labelled ESP32 recordings."""
    parser = ESP32CSIParser(mac=mac)
    recs = []
    for c, name in enumerate(ESP32_CLASSES):
        df = parser.parse(os.path.join(folder, ESP32_FILES[name]))
        t, amp, _, rssi = parser.to_arrays(df)
        rs = UniformResampler(fs=fs, max_gap=max_gap)
        grid, X = rs.transform_with_time(t, np.c_[amp, rssi])
        recs.append(ESP32Recording(c, grid, X[:, :-1], X[:, -1], rs.gap_mask_, len(df)))
    return recs


def normalise(amp: np.ndarray, mode: str = "per_packet") -> np.ndarray:
    """``raw`` | ``per_packet`` (min-max per packet over subcarriers)."""
    if mode == "raw":
        return amp.astype(np.float32)
    if mode == "per_packet":
        return MinMaxNormalizer("per_sample", amp.shape[1]).transform(amp).astype(np.float32)
    raise ValueError(mode)


def blocked_folds(recs, window: int, k: int = 5):
    """Yield (fold, train_targets, test_targets); targets = (rec, end_index).

    Test windows lie completely inside block ``fold``; training windows lie
    completely outside block ``fold`` and do not overlap it.
    """
    for fold in range(k):
        tr, te = [], []
        for r, rec in enumerate(recs):
            T = len(rec.t)
            edges = np.linspace(0, T, k + 1).astype(int)
            lo, hi = edges[fold], edges[fold + 1]
            for e in range(window - 1, T):
                s = e - window + 1
                if s >= lo and e < hi:
                    te.append((r, e))
                elif e < lo or s >= hi:
                    tr.append((r, e))
        yield fold, np.array(tr), np.array(te)


class ESP32WindowDataset(Dataset):
    def __init__(self, feats, recs, targets, window):
        self.feats = [torch.from_numpy(f) for f in feats]
        self.targets, self.window = targets, window
        self.labels = [r.label for r in recs]

    def __len__(self):
        return len(self.targets)

    @property
    def y(self):
        return np.array([self.labels[r] for r, _ in self.targets])

    def __getitem__(self, i):
        r, e = self.targets[i]
        return self.feats[r][e - self.window + 1: e + 1], self.labels[r]


def window_features(feats, targets, window):
    """Hand-crafted statistics per window (for SVM / KNN): mean and temporal
    std per subcarrier averaged, mean |first difference|, std / range of the
    subcarrier-averaged series, and the mean of the 3 largest per-subcarrier
    stds."""
    out = []
    for r, e in targets:
        A = feats[r][e - window + 1: e + 1].astype(np.float64)
        s = A.mean(1)
        sd = A.std(0)
        out.append([A.mean(), sd.mean(), np.sort(sd)[-3:].mean(),
                    np.abs(np.diff(A, axis=0)).mean(), s.std(), s.max() - s.min()])
    return np.array(out)


def motion_energy(feats, targets, window, fs, f_cut=0.6):
    """[Korany21]-style motion statistic: first principal component of the
    window (PCA de-noising, Sec. V-A) and its spectral energy above
    ``f_cut`` Hz (Eq. 11; 0.6 Hz = twice the maximal breathing rate)."""
    out = []
    for r, e in targets:
        A = feats[r][e - window + 1: e + 1].astype(np.float64)
        A = A - A.mean(0)
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        p = u[:, 0] * s[0]
        spec = np.abs(np.fft.rfft(p * np.hanning(len(p)))) ** 2
        f = np.fft.rfftfreq(len(p), 1 / fs)
        out.append(spec[f > f_cut].sum() / len(p))
    return np.array(out)
