"""
csi_preprocessing.py
====================

Pre-processing building blocks for Wi-Fi Channel State Information (CSI),
reproduced from the papers in ``../papers``.

Every step is a small class that follows the scikit-learn transformer API
(``fit`` / ``transform`` / ``fit_transform``), so steps can be chained in a
:class:`sklearn.pipeline.Pipeline` and the *same* fitted object can later be
applied to live data coming out of the ESP32 pipeline.

Conventions
-----------
* A CSI recording is a 2-D array ``X`` of shape ``(n_packets, n_streams)``:
  time runs along axis 0, every column is one "stream" (e.g. the amplitude of
  one subcarrier on one antenna pair).
* Where a step needs the antenna/subcarrier structure (phase sanitisation,
  phase difference, ...) the array is reshaped internally to
  ``(n_packets, n_antenna_pairs, n_subcarriers)``; the class documents this.
* Filters are applied to a *whole continuous recording* before windowing.
  Zero-phase filters (``filtfilt``) look into the future, which is fine for
  offline experiments; the classes expose a ``causal`` option where a
  real-time variant makes sense.
* Stateless steps have a no-op ``fit``.  Steps that learn something from data
  (min-max ranges, PCA axes, standardisation statistics) must be fitted on the
  *training split only* to avoid information leakage into the test set.

Paper references used in the docstrings
---------------------------------------
[Zhuravchak22]  A. Zhuravchak, O. Kapshii, E. Pournaras, "Human Activity
                Recognition based on Wi-Fi CSI Data - A Deep Neural Network
                Approach", Procedia Computer Science 198, 2022.
[Shang21]       S. Shang et al., "LSTM-CNN network for human activity
                recognition using WiFi CSI data", J. Phys.: Conf. Ser. 1883, 2021.
[Quy25]         T. D. Quy, C.-Y. Lin, T. K. Shih, "Enhanced Human Activity
                Recognition Using Wi-Fi Sensing: Leveraging Phase and Amplitude
                with Attention Mechanisms" (PA-CSI), Sensors 25(4) 1038, 2025.
[Korany21]      B. Korany, Y. Mostofi, "Nocturnal Seizure Detection Using
                Off-the-Shelf WiFi", arXiv:2103.13556, 2021.
[Chang23]       C.-C. Chang et al., "Attention-based Learning for Sleep Apnea
                and Limb Movement Detection using Wi-Fi CSI Signals" (ALESAL),
                arXiv:2304.06474, 2023.
[Sen12]         S. Sen et al., "You are facing the Mona Lisa: spot localization
                using PHY layer information", MobiSys 2012 (linear phase
                sanitisation, cited by [Zhuravchak22]).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pywt
from scipy import ndimage, signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

__all__ = [
    "HampelFilter",
    "WaveletDenoiser",
    "ButterworthFilter",
    "KalmanSmoother",
    "PhaseSanitizer",
    "PhaseUnwrapper",
    "PhaseDifference",
    "AmplitudeProduct",
    "TimeAligner",
    "MinMaxNormalizer",
    "Standardizer",
    "PCADenoiser",
    "SubcarrierSelector",
    "ESP32CSIParser",
    "UniformResampler",
    "ht40_subcarrier_indices",
    "esp32_ht20_subcarrier_indices",
]


def _as_2d(X: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Return ``X`` as float array of shape (T, N) plus its original shape."""
    X = np.asarray(X, dtype=np.float64)
    shape = X.shape
    if X.ndim == 1:
        X = X[:, None]
    elif X.ndim > 2:
        X = X.reshape(shape[0], -1)
    return X, shape


# ---------------------------------------------------------------------------
# Outlier removal and de-noising
# ---------------------------------------------------------------------------
class HampelFilter(BaseEstimator, TransformerMixin):
    """Hampel identifier: replace outliers by the local median.

    Used for outlier removal in [Zhuravchak22] (Sec. 6.1) and [Korany21]
    (Sec. V-A).  For every sample ``x[t]`` a window of ``2*window+1`` samples
    centred on ``t`` is taken; with ``med`` the window median and
    ``MAD`` the median absolute deviation, the sample is an outlier when

        |x[t] - med| > n_sigma * 1.4826 * MAD

    (1.4826 makes the MAD a consistent estimator of the standard deviation for
    Gaussian noise).  Outliers are replaced by ``med``.

    Implementation: :func:`scipy.ndimage.median_filter` along the time axis,
    i.e. fully vectorised over all streams.

    Parameters
    ----------
    window : int
        Half window length in samples (total window ``2*window+1``).
    n_sigma : float
        Threshold in robust standard deviations (3 = the classical choice).
    """

    def __init__(self, window: int = 5, n_sigma: float = 3.0):
        self.window = window
        self.n_sigma = n_sigma

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X2, shape = _as_2d(X)
        size = (2 * self.window + 1, 1)
        med = ndimage.median_filter(X2, size=size, mode="nearest")
        mad = ndimage.median_filter(np.abs(X2 - med), size=size, mode="nearest")
        outlier = np.abs(X2 - med) > self.n_sigma * 1.4826 * mad
        out = np.where(outlier, med, X2)
        self.outlier_fraction_ = float(outlier.mean())
        return out.reshape(shape)


class WaveletDenoiser(BaseEstimator, TransformerMixin):
    """Discrete-Wavelet-Transform shrinkage de-noising.

    [Zhuravchak22] Sec. 6.1 ("Noise reduction based on Discrete Wavelet
    Transform"): decompose the signal, shrink the small (noise) coefficients,
    reconstruct with the inverse DWT.  We use the standard VisuShrink rule
    (Donoho & Johnstone): the noise level is estimated from the finest detail
    coefficients, ``sigma = median(|d_1|) / 0.6745`` and the universal
    threshold ``sigma * sqrt(2 * log(n))`` is applied to every detail level.

    Implementation: :mod:`pywt` (``wavedec`` / ``threshold`` / ``waverec``)
    along axis 0, vectorised over all streams.

    Parameters
    ----------
    wavelet : str
        Any wavelet known to PyWavelets ("db4" is common for CSI).
    level : int or None
        Decomposition depth; ``None`` = maximum useful depth.
    mode : {"soft", "hard"}
        Thresholding mode.
    threshold_scale : float
        Multiplier on the universal threshold (<1 keeps more detail).
    """

    def __init__(self, wavelet: str = "db4", level: Optional[int] = 4,
                 mode: str = "soft", threshold_scale: float = 1.0):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.threshold_scale = threshold_scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X2, shape = _as_2d(X)
        n = X2.shape[0]
        coeffs = pywt.wavedec(X2, self.wavelet, level=self.level, axis=0)
        sigma = np.median(np.abs(coeffs[-1]), axis=0) / 0.6745       # per stream
        thr = self.threshold_scale * sigma * np.sqrt(2.0 * np.log(n))
        coeffs = [coeffs[0]] + [pywt.threshold(c, thr[None, :], mode=self.mode)
                                for c in coeffs[1:]]
        out = pywt.waverec(coeffs, self.wavelet, axis=0)[:n]
        return out.reshape(shape)


class ButterworthFilter(BaseEstimator, TransformerMixin):
    """Butterworth IIR filter (low-, high- or band-pass) along time.

    * [Shang21] Sec. 2.1: low-pass filter to remove high-frequency noise from
      the amplitude and phase difference.
    * [Chang23] Eq. (4): band-pass ``[0.1, 2] Hz`` for breathing / limb motion.

    Implementation: :func:`scipy.signal.butter` in second-order-sections form,
    applied with :func:`scipy.signal.sosfiltfilt` (zero phase, offline) or
    :func:`scipy.signal.sosfilt` (``causal=True``, usable in real time).

    Parameters
    ----------
    cutoff : float or (float, float)
        Cut-off frequency in Hz if ``fs`` is given, otherwise as a fraction of
        the Nyquist frequency (0 < cutoff < 1).  The HAR dataset has no
        timestamps, so there we use normalised cut-offs.
    btype : {"low", "high", "band"}
    order : int
    fs : float or None
        Sampling rate in Hz.
    causal : bool
        Use a one-directional filter (adds phase delay but needs no future).
    """

    def __init__(self, cutoff=0.1, btype: str = "low", order: int = 4,
                 fs: Optional[float] = None, causal: bool = False):
        self.cutoff = cutoff
        self.btype = btype
        self.order = order
        self.fs = fs
        self.causal = causal

    def fit(self, X, y=None):
        return self

    def _sos(self):
        return signal.butter(self.order, self.cutoff, btype=self.btype,
                             fs=self.fs, output="sos")

    def transform(self, X):
        X2, shape = _as_2d(X)
        sos = self._sos()
        if self.causal:
            out = signal.sosfilt(sos, X2, axis=0)
        else:
            out = signal.sosfiltfilt(sos, X2, axis=0)
        return out.reshape(shape)


class KalmanSmoother(BaseEstimator, TransformerMixin):
    """Scalar random-walk Kalman filter applied independently to every stream.

    [Quy25] Sec. 3.3.1 uses a Kalman filter as the first pre-processing step to
    suppress the measurement noise ``eta`` in ``y = Hx + eta``.  The paper
    does not give the state model, so we use the simplest model that fits a
    slowly-varying CSI amplitude:

        state:        s[t] = s[t-1] + w,   w ~ N(0, q)
        measurement:  x[t] = s[t]   + v,   v ~ N(0, r)

    ``q / r`` controls the smoothing: smaller = smoother.  By default ``r`` is
    estimated per stream from the data (variance of first differences / 2,
    which equals the white-noise variance for a constant signal) and
    ``q = process_ratio * r``.

    The forward pass is causal (real-time capable).  Set ``rts=True`` to add a
    Rauch-Tung-Striebel backward pass (offline smoothing).

    Implementation: own vectorised NumPy code (loop over time, vectorised
    over streams).  ``filterpy`` / ``pykalman`` would also work but are
    unmaintained and add a dependency for a 20-line filter.
    """

    def __init__(self, process_ratio: float = 0.05,
                 measurement_var: Optional[float] = None, rts: bool = False):
        self.process_ratio = process_ratio
        self.measurement_var = measurement_var
        self.rts = rts

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X2, shape = _as_2d(X)
        T, N = X2.shape
        if self.measurement_var is None:
            r = np.var(np.diff(X2, axis=0), axis=0) / 2.0 + 1e-12
        else:
            r = np.full(N, float(self.measurement_var))
        q = self.process_ratio * r
        s = np.empty_like(X2)
        p_post = np.empty_like(X2)
        p_prior = np.empty_like(X2)
        s_t, p_t = X2[0].copy(), r.copy()
        for t in range(T):
            p_pred = p_t + q if t > 0 else p_t
            k = p_pred / (p_pred + r)
            s_t = s_t + k * (X2[t] - s_t)
            p_t = (1.0 - k) * p_pred
            s[t], p_post[t], p_prior[t] = s_t, p_t, p_pred
        if self.rts:
            for t in range(T - 2, -1, -1):
                c = p_post[t] / p_prior[t + 1]
                s[t] = s[t] + c * (s[t + 1] - s[t])
        return s.reshape(shape)


# ---------------------------------------------------------------------------
# Phase processing
# ---------------------------------------------------------------------------
def ht40_subcarrier_indices() -> np.ndarray:
    """The 114 data+pilot subcarrier indices of an 802.11n HT40 channel
    (-58..-2, 2..58), as reported by the Atheros CSI tool used in
    [Zhuravchak22]."""
    return np.r_[np.arange(-58, -1), np.arange(2, 59)].astype(float)


def esp32_ht20_subcarrier_indices() -> np.ndarray:
    """Subcarrier indices of the 64-entry L-LTF CSI reported by an ESP32 for a
    20 MHz channel: entries 0..31 are subcarriers 0..31, entries 32..63 are
    -32..-1 (ESP-IDF ``wifi_csi_info_t`` documentation)."""
    return np.r_[np.arange(0, 32), np.arange(-32, 0)].astype(float)


class PhaseSanitizer(BaseEstimator, TransformerMixin):
    """Linear phase sanitisation across subcarriers ([Sen12], used in
    [Zhuravchak22] Sec. 6.1, Fig. 5).

    The measured phase of subcarrier ``k`` is corrupted by the carrier
    frequency offset (CFO, a constant ``b``) and the sampling frequency
    offset / timing delay (a term linear in the subcarrier index ``m_k``):

        phi_meas[k] = phi[k] + a * m_k + b + noise

    For every packet and antenna pair the phase is first unwrapped across
    subcarriers, then

        a = (phi[-1] - phi[0]) / (m[-1] - m[0]),   b = mean(phi)
        phi_sanitised[k] = phi[k] - a * m_k - b

    which removes both offsets while keeping the relative (human-induced)
    phase structure.

    Parameters
    ----------
    n_subcarriers : int
        Number of subcarriers per antenna pair; the input columns are
        reshaped to ``(T, n_pairs, n_subcarriers)``.
    subcarrier_indices : array-like or None
        Physical subcarrier indices ``m_k``; default 0..n-1.  Use
        :func:`ht40_subcarrier_indices` for the Atheros HAR dataset.
    fit_least_squares : bool
        If True, estimate ``a, b`` by a least-squares line fit instead of the
        two-endpoint rule (more robust to noisy edge subcarriers).
    """

    def __init__(self, n_subcarriers: int = 114, subcarrier_indices=None,
                 fit_least_squares: bool = False):
        self.n_subcarriers = n_subcarriers
        self.subcarrier_indices = subcarrier_indices
        self.fit_least_squares = fit_least_squares

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        T = X.shape[0]
        P = X.reshape(T, -1, self.n_subcarriers)
        m = (np.arange(self.n_subcarriers, dtype=float)
             if self.subcarrier_indices is None
             else np.asarray(self.subcarrier_indices, dtype=float))
        phi = np.unwrap(P, axis=-1)
        if self.fit_least_squares:
            mc = m - m.mean()
            a = (phi * mc).sum(-1, keepdims=True) / (mc ** 2).sum()
        else:
            a = (phi[..., -1:] - phi[..., :1]) / (m[-1] - m[0])
        b = phi.mean(-1, keepdims=True) - a * m.mean()
        out = phi - a * m - b
        return out.reshape(X.shape)


class PhaseUnwrapper(BaseEstimator, TransformerMixin):
    """Phase unwrapping, [Quy25] Sec. 3.3.6, Eq. (7).

    Adds multiples of ``2*pi`` so that consecutive samples never jump by more
    than ``pi``.  [Quy25] unwraps along time (Fig. 7: "phase average" vs
    "packet index"); ``axis="subcarrier"`` unwraps along frequency instead.

    Because the CFO makes the unwrapped phase drift without bound over a long
    recording, ``remove_mean=True`` subtracts the per-stream mean of the input
    block.  Apply it to each *window* (not the whole recording) so the
    numbers stay bounded - see ``csi_data.py``.

    Implementation: :func:`numpy.unwrap`.
    """

    def __init__(self, axis: str = "time", n_subcarriers: int = 114,
                 remove_mean: bool = True):
        self.axis = axis
        self.n_subcarriers = n_subcarriers
        self.remove_mean = remove_mean

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.axis == "time":
            out = np.unwrap(X, axis=0)
        elif self.axis == "subcarrier":
            T = X.shape[0]
            out = np.unwrap(X.reshape(T, -1, self.n_subcarriers), axis=-1).reshape(X.shape)
        else:
            raise ValueError("axis must be 'time' or 'subcarrier'")
        if self.remove_mean:
            out = out - out.mean(axis=0, keepdims=True)
        return out


class PhaseDifference(BaseEstimator, TransformerMixin):
    """Phase difference between two receive antennas.

    [Shang21] Sec. 2.1 and [Korany21] Sec. V-A: the CFO/SFO errors are
    (almost) identical on all antennas of the same receiver, so the
    difference ``angle(H_a) - angle(H_b)`` cancels them and leaves a phase
    that reacts to motion.  Computed as ``angle(exp(1j*(phi_a - phi_b)))`` so
    the result is wrapped to ``(-pi, pi]``.

    Parameters
    ----------
    pairs : sequence of (int, int)
        Antenna-pair indices ``(a, b)`` to subtract.
    n_subcarriers : int
        Input columns are reshaped to ``(T, n_pairs, n_subcarriers)``.
    unwrap_time : bool
        Additionally unwrap the difference along time (it is usually far from
        the ``+-pi`` boundary, so this mostly matters for noisy links).
    """

    def __init__(self, pairs: Sequence[tuple[int, int]] = ((0, 1), (2, 3)),
                 n_subcarriers: int = 114, unwrap_time: bool = False):
        self.pairs = pairs
        self.n_subcarriers = n_subcarriers
        self.unwrap_time = unwrap_time

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        P = X.reshape(X.shape[0], -1, self.n_subcarriers)
        d = np.stack([np.angle(np.exp(1j * (P[:, a] - P[:, b]))) for a, b in self.pairs], 1)
        if self.unwrap_time:
            d = np.unwrap(d, axis=0)
        return d.reshape(X.shape[0], -1)


class AmplitudeProduct(BaseEstimator, TransformerMixin):
    """Product of the amplitudes of two antennas, ``|H_a| * |H_b|``.

    [Shang21] Sec. 2.1 builds a "2-channel" input from the amplitude product
    and the phase difference of two antennas, and a "4-channel" input by
    adding the two individual amplitudes.  This class produces the product
    channel; see :class:`PhaseDifference` for the other.
    """

    def __init__(self, pairs: Sequence[tuple[int, int]] = ((0, 1), (2, 3)),
                 n_subcarriers: int = 114):
        self.pairs = pairs
        self.n_subcarriers = n_subcarriers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        A = X.reshape(X.shape[0], -1, self.n_subcarriers)
        prod = np.stack([A[:, a] * A[:, b] for a, b in self.pairs], 1)
        return prod.reshape(X.shape[0], -1)


# ---------------------------------------------------------------------------
# Re-sampling / alignment
# ---------------------------------------------------------------------------
class TimeAligner(BaseEstimator, TransformerMixin):
    """Time alignment by block averaging, [Quy25] Sec. 3.3.3, Fig. 6.

    Recordings (or windows) of slightly different lengths are mapped to a fixed
    length ``target_length`` by splitting the time axis into
    ``target_length`` (almost) equal bins and averaging each bin
    ("avg_0 ... avg_1999" in Fig. 6).  Besides equalising lengths this acts as
    a low-pass filter + down-sampling, which we exploit to make the attention
    model affordable on a CPU.

    Works on arrays of shape ``(T, ...)``; for a batch of windows of shape
    ``(B, T, F)`` use ``axis=1``.
    """

    def __init__(self, target_length: int = 250, axis: int = 0):
        self.target_length = target_length
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.moveaxis(X, self.axis, 0)
        T = X.shape[0]
        edges = np.linspace(0, T, self.target_length + 1).round().astype(int)
        csum = np.concatenate([np.zeros((1,) + X.shape[1:]), np.cumsum(X, axis=0)])
        counts = np.maximum(np.diff(edges), 1).reshape((-1,) + (1,) * (X.ndim - 1))
        out = (csum[edges[1:]] - csum[edges[:-1]]) / counts
        return np.moveaxis(out, 0, self.axis)


class UniformResampler(BaseEstimator, TransformerMixin):
    """Resample an irregularly-sampled CSI stream onto a uniform time grid.

    [Chang23] Sec. III-A: "Suffering from packet loss and transmission delay,
    the receiver is unable to receive every packet with the same frequency
    ... the raw CSI data is resampled".  This is essential for the ESP32
    recordings, whose packets arrive irregularly (gaps of >1 s).

    Linear interpolation (:func:`numpy.interp`) per stream.  Grid points that
    fall inside a gap longer than ``max_gap`` seconds are flagged in
    ``self.gap_mask_`` so they can be excluded later.

    Use :meth:`transform_with_time` (needs timestamps); ``transform`` is not
    meaningful without them.
    """

    def __init__(self, fs: float = 5.0, max_gap: float = 1.0):
        self.fs = fs
        self.max_gap = max_gap

    def fit(self, X, y=None):
        return self

    def transform_with_time(self, t, X):
        t = np.asarray(t, dtype=float)
        X2, _ = _as_2d(X)
        order = np.argsort(t)
        t, X2 = t[order], X2[order]
        grid = np.arange(t[0], t[-1], 1.0 / self.fs)
        out = np.stack([np.interp(grid, t, X2[:, j]) for j in range(X2.shape[1])], 1)
        idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        self.gap_mask_ = (t[idx] - t[idx - 1]) > self.max_gap
        return grid, out

    def transform(self, X):
        raise NotImplementedError("use transform_with_time(t, X)")


# ---------------------------------------------------------------------------
# Scaling and dimensionality reduction (stateful: fit on training data!)
# ---------------------------------------------------------------------------
class MinMaxNormalizer(BaseEstimator, TransformerMixin):
    """Min-max normalisation to ``[0, 1]``.

    * ``mode="global"`` - [Quy25] Eq. (6): one ``min``/``max`` per stream,
      learned on the training data (wraps :class:`sklearn.preprocessing.MinMaxScaler`).
    * ``mode="per_sample"`` - [Chang23] Eq. (3): for every packet and antenna
      pair, scale the amplitudes of all its subcarriers with that packet's own
      min and max.  This removes per-packet gain changes (automatic gain
      control of commodity NICs / ESP32) and needs no fitting.

    Parameters
    ----------
    mode : {"global", "per_sample"}
    n_subcarriers : int
        Group size for ``per_sample`` (subcarriers of one antenna pair).
    clip : bool
        Clip test data to ``[0, 1]`` in global mode.
    """

    def __init__(self, mode: str = "global", n_subcarriers: int = 114, clip: bool = False):
        self.mode = mode
        self.n_subcarriers = n_subcarriers
        self.clip = clip

    def fit(self, X, y=None):
        if self.mode == "global":
            self.scaler_ = MinMaxScaler(clip=self.clip).fit(np.asarray(X).reshape(len(X), -1))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.mode == "global":
            return self.scaler_.transform(X.reshape(len(X), -1)).reshape(X.shape)
        if self.mode == "per_sample":
            G = X.reshape(X.shape[0], -1, self.n_subcarriers)
            lo, hi = G.min(-1, keepdims=True), G.max(-1, keepdims=True)
            return ((G - lo) / np.maximum(hi - lo, 1e-12)).reshape(X.shape)
        raise ValueError("mode must be 'global' or 'per_sample'")


class Standardizer(BaseEstimator, TransformerMixin):
    """Z-score per stream (mean 0, std 1) learned on the training data.

    Not taken from a specific paper; it is the standard input scaling for
    neural networks and is used for the LSTM-type models.  Thin wrapper
    around :class:`sklearn.preprocessing.StandardScaler` that accepts
    ``(T, F)`` recordings.
    """

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(np.asarray(X).reshape(len(X), -1))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.scaler_.transform(X.reshape(len(X), -1)).reshape(X.shape)


class PCADenoiser(BaseEstimator, TransformerMixin):
    """Project the streams on their first principal component(s).

    [Korany21] Sec. V-A ("PCA denoising") and [Chang23] Eq. (5): the
    human-motion component is common to all subcarriers while noise is
    spread over all principal components, so the first PC(s) are a denoised
    motion signal.  With ``per_group=True`` (``n_subcarriers`` set) one PCA is
    fitted per antenna pair, as in [Chang23].

    Wraps :class:`sklearn.decomposition.PCA`; must be fitted on training data.
    """

    def __init__(self, n_components: int = 1, n_subcarriers: Optional[int] = None,
                 center: bool = True):
        self.n_components = n_components
        self.n_subcarriers = n_subcarriers
        self.center = center

    def _groups(self, X):
        if self.n_subcarriers is None:
            return [X]
        G = X.reshape(X.shape[0], -1, self.n_subcarriers)
        return [G[:, g] for g in range(G.shape[1])]

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.pcas_ = [PCA(self.n_components).fit(g) for g in self._groups(X)]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.concatenate([p.transform(g) for p, g in zip(self.pcas_, self._groups(X))], 1)


class SubcarrierSelector(BaseEstimator, TransformerMixin):
    """Keep a subset of subcarriers (e.g. drop the ESP32 guard/DC null
    subcarriers, or take every k-th subcarrier to shrink the input).

    Parameters
    ----------
    keep : array-like of int
        Subcarrier positions (within one antenna pair) to keep.
    n_subcarriers : int
        Subcarriers per antenna pair in the input.
    """

    def __init__(self, keep: Sequence[int], n_subcarriers: int):
        self.keep = keep
        self.n_subcarriers = n_subcarriers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        G = X.reshape(X.shape[0], -1, self.n_subcarriers)
        return G[:, :, np.asarray(self.keep)].reshape(X.shape[0], -1)


# ---------------------------------------------------------------------------
# ESP32 data pipeline
# ---------------------------------------------------------------------------
@dataclass
class ESP32CSIParser:
    """Parse the CSV files written by the ESP32 capture pipeline.

    Expected columns: ``timestamp, mac, rssi, channel, subcarrier_count,
    amplitudes, phases, real_parts, imaginary_parts`` where the last four hold
    JSON-style lists.

    The current recordings capture *every* transmitter on the channel
    (promiscuous mode, ~40 MAC addresses).  CSI from different transmitters
    describes different radio links and must not be mixed, so the parser keeps
    one MAC (by default the most frequent one).

    Attributes
    ----------
    mac : str or None
        MAC to keep; ``None`` = most frequent MAC in the file.
    n_subcarriers : int
        Keep packets with exactly this many CSI entries (64 = 20 MHz L-LTF).
    drop_null : bool
        Drop subcarriers that are always zero (DC + guard band: entries
        0 and 27..37 for ESP32 HT20 L-LTF), leaving 52 useful subcarriers.
    use_complex : bool
        Recompute amplitude/phase from the real/imaginary parts (more
        precision than the rounded strings).
    """

    mac: Optional[str] = None
    n_subcarriers: int = 64
    drop_null: bool = True
    use_complex: bool = True
    null_subcarriers: tuple = field(default=(0,) + tuple(range(27, 38)))

    def parse(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        mac = self.mac or df["mac"].value_counts().index[0]
        df = df[(df["mac"] == mac) & (df["subcarrier_count"] == self.n_subcarriers)]
        return df.sort_values("timestamp").reset_index(drop=True)

    def to_arrays(self, df: pd.DataFrame):
        """Return ``t (T,), amplitude (T, S), phase (T, S), rssi (T,)``."""
        if self.use_complex:
            re = np.array([json.loads(v) for v in df["real_parts"]], dtype=float)
            im = np.array([json.loads(v) for v in df["imaginary_parts"]], dtype=float)
            H = re + 1j * im
            amp, ph = np.abs(H), np.angle(H)
        else:
            amp = np.array([json.loads(v) for v in df["amplitudes"]], dtype=float)
            ph = np.array([json.loads(v) for v in df["phases"]], dtype=float)
        if self.drop_null:
            keep = np.setdiff1d(np.arange(self.n_subcarriers), self.null_subcarriers)
            amp, ph = amp[:, keep], ph[:, keep]
        return (df["timestamp"].to_numpy(float), amp, ph, df["rssi"].to_numpy(float))
