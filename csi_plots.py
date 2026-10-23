"""
csi_plots.py - small matplotlib helpers used by the notebook.

Colours follow a validated colour-blind-safe categorical order (blue, orange,
aqua, yellow, magenta, green, violet, red), always assigned in that fixed
order; confusion matrices use a single-hue sequential ramp.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.prop_cycle": plt.cycler(color=PALETTE), "lines.linewidth": 1.6,
        "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
        "legend.frameon": False,
    })


def before_after(x_before, x_after, title, labels=("before", "after"), t=None, ax=None,
                 xlabel="packet index"):
    """Overlay one stream before and after a pre-processing step."""
    ax = ax or plt.subplots(figsize=(9, 2.6))[1]
    t = np.arange(len(x_before)) if t is None else t
    ax.plot(t, x_before, color=PALETTE[0], lw=1.0, alpha=0.55, label=labels[0])
    ax.plot(t, x_after, color=PALETTE[1], lw=1.6, label=labels[1])
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.legend(loc="upper right", ncol=2)
    return ax


def confusion(cm, classes, title="", ax=None, normalize=True):
    """Row-normalised confusion matrix (rows = true class, % recall per cell)."""
    cm = np.asarray(cm, dtype=float)
    ax = ax or plt.subplots(figsize=(0.55 * len(classes) + 2.2, 0.5 * len(classes) + 1.8))[1]
    m = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = m[i, j]
            ax.text(j, i, f"{100 * v:.0f}" if normalize else int(v), ha="center", va="center",
                    fontsize=8, color="white" if v > 0.6 else TEXT)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left")
    return ax


def grouped_bars(df, value, group, series, title="", ax=None, ylim=(0, 1)):
    """Grouped horizontal-free bar chart: ``group`` on the x axis, one bar per
    ``series`` value (fixed colour order), value labels on top."""
    groups = list(dict.fromkeys(df[group]))
    sers = list(dict.fromkeys(df[series]))
    ax = ax or plt.subplots(figsize=(max(6, 1.1 * len(groups) * len(sers) / 2 + 2), 3.2))[1]
    w = 0.8 / len(sers)
    for k, s in enumerate(sers):
        sub = df[df[series] == s].set_index(group)
        xs = [g for g in groups if g in sub.index]
        pos = np.array([groups.index(g) for g in xs]) + (k - (len(sers) - 1) / 2) * w
        vals = sub.loc[xs, value].to_numpy(float)
        ax.bar(pos, vals, width=w * 0.92, color=PALETTE[k % len(PALETTE)], label=str(s))
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(*ylim)
    ax.set_ylabel(value)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(sers), loc="upper left", bbox_to_anchor=(0, 1.0))
    return ax


def label_timeline(y_true, y_pred, classes, title="", ax=None):
    """Prediction vs ground truth over time for one test recording."""
    ax = ax or plt.subplots(figsize=(10, 2.4))[1]
    t = np.arange(len(y_true))
    ax.step(t, y_true, where="post", color=PALETTE[0], lw=2.0, label="true")
    ax.plot(t, y_pred, "o", ms=2.5, color=PALETTE[1], label="predicted")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("target index (every 16 packets)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right", ncol=2)
    return ax
