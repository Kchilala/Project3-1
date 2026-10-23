"""
csi_models.py
=============

PyTorch re-implementations of the models in ``../papers``.  All models take
a batch of windows ``x`` of shape ``(batch, time, features)`` and return
class logits ``(batch, n_classes)``.

==================  ===============  ==========================================
class               paper            notes
==================  ===============  ==========================================
ShangLSTM           [Shang21]        LSTM baseline (28 hidden units)
ShangLSTMCNN        [Shang21]        **core model**: LSTM(28) -> Conv2d(5 @ 30x28)
                                     -> global max-pool -> MLP
ZhuravchakBiLSTM    [Zhuravchak22]   chunks of 16 packets (overlap 8) -> BiLSTM(256)
                                     -> Dense 512/256/128/n
InceptionTime       [Zhuravchak22]   InceptionTime of Ismail Fawaz et al. 2020
PACSI               [Quy25]          GRE + 4 MCAT layers (temporal & channel
                                     streams) for amplitude and phase,
                                     CNN max-pool, GRN fusion
==================  ===============  ==========================================
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# [Shang21]  LSTM and LSTM-CNN
# ---------------------------------------------------------------------------
class ShangLSTM(nn.Module):
    """LSTM baseline of [Shang21] Sec. 3.3: same LSTM as the LSTM-CNN (28
    hidden units, dropout 0.2), classified from the last hidden state by an
    MLP."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 28,
                 mlp_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, mlp_hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(mlp_hidden, n_classes))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(self.drop(h[-1]))


class ShangLSTMCNN(nn.Module):
    """LSTM-CNN of [Shang21] Sec. 2.2, Fig. 2.

    1. LSTM with ``r = 28`` hidden units over the sequence of ``s`` packets,
       dropout 0.2 -> output ``(b, s, r)``.
    2. "Dimensional promotion" to ``Q`` of shape ``(b, 1, s, r)``.
    3. Conv2d with 5 kernels of size ``30 x 28`` (spanning all hidden units),
       stride 1, ReLU -> ``(b, 5, s - 29, 1)`` (371 x 1 for s = 400, Eq. 7).
    4. Max-pool over time -> one value per kernel, ``S`` of size 5 (Eq. 8).
    5. MLP classifier.

    The MLP size is not given in the paper; we use one hidden layer of 64
    units.  ``kernel_time`` must be <= the sequence length (use a smaller
    value for short ESP32 windows).
    """

    def __init__(self, n_features: int, n_classes: int, hidden: int = 28,
                 n_kernels: int = 5, kernel_time: int = 30, mlp_hidden: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.conv = nn.Conv2d(1, n_kernels, kernel_size=(kernel_time, hidden))
        self.head = nn.Sequential(nn.Linear(n_kernels, mlp_hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(mlp_hidden, n_classes))

    def forward(self, x):
        h, _ = self.lstm(x)                          # (b, s, r)
        q = self.drop(h).unsqueeze(1)                # (b, 1, s, r)
        c = F.relu(self.conv(q)).squeeze(-1)         # (b, k, s-kt+1)
        s = c.amax(dim=-1)                           # (b, k)
        return self.head(s)


# ---------------------------------------------------------------------------
# [Zhuravchak22]  BiLSTM
# ---------------------------------------------------------------------------
class ZhuravchakBiLSTM(nn.Module):
    """LSTM-based model of [Zhuravchak22] Sec. 6.2.

    "The input data-array is processed in 16-time-points chunks with the
    8-point overlapping between them": the window of ``L`` packets is cut
    into chunks of ``chunk`` packets with stride ``chunk_step``; each chunk
    is flattened into one LSTM time step.  One bidirectional LSTM layer with
    256 hidden units, then dense layers 512-256-128-n_classes with ReLU.

    ``input_proj`` (default None = faithful) optionally adds a linear layer
    that maps the flattened chunk (16 x 912 = 14 592 values) to a smaller size
    before the LSTM; the faithful model has ~31 M LSTM parameters.
    """

    def __init__(self, n_features: int, n_classes: int, chunk: int = 16, chunk_step: int = 8,
                 hidden: int = 256, input_proj: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.chunk, self.chunk_step = chunk, chunk_step
        d_in = chunk * n_features
        self.proj = nn.Linear(d_in, input_proj) if input_proj else nn.Identity()
        self.lstm = nn.LSTM(input_proj or d_in, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x):
        b, L, f = x.shape
        chunks = x.unfold(1, self.chunk, self.chunk_step)        # (b, n, f, chunk)
        chunks = chunks.transpose(2, 3).reshape(b, chunks.shape[1], -1)
        out, _ = self.lstm(self.proj(chunks))
        h = out.shape[-1] // 2
        last = torch.cat([out[:, -1, :h], out[:, 0, h:]], -1)    # fwd last, bwd first
        return self.head(last)


# ---------------------------------------------------------------------------
# InceptionTime (Ismail Fawaz et al. 2020), used by [Zhuravchak22]
# ---------------------------------------------------------------------------
class _InceptionModule(nn.Module):
    def __init__(self, c_in: int, nf: int = 32, kernels=(39, 19, 9), bottleneck: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(c_in, bottleneck, 1, bias=False) if c_in > 1 else nn.Identity()
        b_out = bottleneck if c_in > 1 else c_in
        self.convs = nn.ModuleList([nn.Conv1d(b_out, nf, k, padding=k // 2, bias=False)
                                    for k in kernels])
        self.pool_conv = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(c_in, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * (len(kernels) + 1))

    def forward(self, x):
        z = self.bottleneck(x)
        out = torch.cat([c(z) for c in self.convs] + [self.pool_conv(x)], 1)
        return F.relu(self.bn(out))


class InceptionTime(nn.Module):
    """One InceptionTime network: ``depth`` inception modules (bottleneck 32,
    kernels 40/20/10 -> odd 39/19/9 for 'same' padding, max-pool branch, 32
    filters each), residual shortcut every 3 modules, global average pooling,
    linear softmax layer.  The original "InceptionTime" is an ensemble of 5 such
    networks (see :class:`Ensemble`)."""

    def __init__(self, n_features: int, n_classes: int, nf: int = 32, depth: int = 6):
        super().__init__()
        self.modules_ = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        c = n_features
        res_c = n_features
        for d in range(depth):
            self.modules_.append(_InceptionModule(c, nf))
            c = nf * 4
            if d % 3 == 2:
                self.shortcuts.append(nn.Sequential(nn.Conv1d(res_c, c, 1, bias=False),
                                                    nn.BatchNorm1d(c)))
                res_c = c
        self.fc = nn.Linear(c, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)                     # (b, f, t)
        res = x
        for d, m in enumerate(self.modules_):
            x = m(x)
            if d % 3 == 2:
                x = F.relu(x + self.shortcuts[d // 3](res))
                res = x
        return self.fc(x.mean(-1))


class Ensemble(nn.Module):
    """Average the softmax outputs of several independently initialised
    networks (returns log-probabilities so argmax / CE still work)."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        p = torch.stack([F.softmax(m(x), -1) for m in self.members]).mean(0)
        return torch.log(p + 1e-9)


# ---------------------------------------------------------------------------
# [Quy25]  PA-CSI
# ---------------------------------------------------------------------------
class GaussianRangeEncoding(nn.Module):
    """Gaussian Range Encoding ([Quy25] Sec. 3.3.7, from THAT [Li et al. 2021]).

    ``K`` univariate Gaussians with learnable means and standard deviations
    tile the time axis.  For position ``t`` the vector of Gaussian densities
    is normalised over the K Gaussians and multiplied by a learnable
    ``K x d`` embedding matrix; the result is added to the input.  Unlike a
    per-position encoding this encodes *ranges* of time, which tolerates
    variations in activity speed.
    """

    def __init__(self, seq_len: int, d_model: int, k: int = 10):
        super().__init__()
        self.mu = nn.Parameter(torch.linspace(0, seq_len - 1, k))
        self.log_sigma = nn.Parameter(torch.full((k,), math.log(seq_len / k)))
        self.emb = nn.Parameter(torch.randn(k, d_model) * 0.02)
        self.register_buffer("pos", torch.arange(seq_len).float())

    def forward(self, x):                        # x: (b, L, d)
        sigma = self.log_sigma.exp()
        logp = -0.5 * ((self.pos[:, None] - self.mu) / sigma) ** 2 - torch.log(sigma)
        w = torch.softmax(logp, dim=-1)          # (L, K) normalised densities
        return x + w @ self.emb


class MultiScaleConv(nn.Module):
    """Multi-scale CNN of the MCAT layer ([Quy25] Eq. 13):
    ``P_i = ReLU(Dropout(BN(Conv_{j_i}(Y))))`` for every kernel size
    ``j_i``; the ``P_i`` (each ``d / n_kernels`` channels) are concatenated so
    the output has the input width again."""

    def __init__(self, d: int, kernels=(1, 3, 5), dropout: float = 0.1):
        super().__init__()
        n = len(kernels)
        outs = [d // n + (1 if i < d % n else 0) for i in range(n)]
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv1d(d, o, k, padding=k // 2),
                                                  nn.BatchNorm1d(o), nn.Dropout(dropout), nn.ReLU())
                                    for k, o in zip(kernels, outs)])

    def forward(self, x):                        # (b, L, d)
        z = x.transpose(1, 2)
        return torch.cat([c(z) for c in self.convs], 1).transpose(1, 2)


class MCATLayer(nn.Module):
    """Multi-scale Convolution Augmented Transformer layer ([Quy25] Fig. 9):
    multi-head self-attention -> add & LayerNorm -> multi-scale CNN -> add &
    LayerNorm."""

    def __init__(self, d: int, heads: int, kernels=(1, 3, 5), dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)  # no dropout on attention maps (CPU cost)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.msc = MultiScaleConv(d, kernels, dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + a)
        return self.ln2(x + self.msc(x))


class CNNMaxPool(nn.Module):
    """'CNN max-pooling' after the MCAT streams ([Quy25] Fig. 4, as in THAT):
    1-D convolutions with several kernel sizes along the sequence, ReLU,
    global max-pool, concatenation."""

    def __init__(self, d: int, n_filters: int, kernels):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d, n_filters, k, padding=k // 2) for k in kernels])
        self.out_dim = n_filters * len(kernels)

    def forward(self, x):                        # (b, L, d)
        z = x.transpose(1, 2)
        return torch.cat([F.relu(c(z)).amax(-1) for c in self.convs], 1)


class TwoStreamEncoder(nn.Module):
    """Temporal stream (tokens = time steps) and channel stream (tokens =
    subcarrier streams, i.e. the transposed input) for one modality."""

    def __init__(self, seq_len: int, n_feat: int, d_model: int, n_layers: int, h_heads: int,
                 v_heads: int, k_gauss: int, dropout: float, use_gre: bool):
        super().__init__()
        self.in_t = nn.Linear(n_feat, d_model)               # temporal tokens: (L, d_model)
        self.gre = GaussianRangeEncoding(seq_len, d_model, k_gauss) if use_gre else nn.Identity()
        self.temporal = nn.Sequential(*[MCATLayer(d_model, h_heads, (10 + 1, 40 + 1), dropout)
                                        for _ in range(n_layers)])
        self.channel = nn.Sequential(*[MCATLayer(seq_len, v_heads, (2 + 1, 4 + 1), dropout)
                                       for _ in range(n_layers)])
        self.pool_t = CNNMaxPool(d_model, 32, (3, 5, 7))
        self.pool_c = CNNMaxPool(seq_len, 32, (3, 5, 7))
        self.out_dim = self.pool_t.out_dim + self.pool_c.out_dim

    def forward(self, x):                        # (b, L, F)
        t = self.temporal(self.gre(self.in_t(x)))
        c = self.channel(x.transpose(1, 2))      # (b, F, L)
        return torch.cat([self.pool_t(t), self.pool_c(c)], 1)


class GRN(nn.Module):
    """Gated Residual Network fusion of amplitude and phase ([Quy25] Fig. 10):
    ``h = LayerNorm(ELU(W_a a) + ELU(W_p p))``,
    ``out = a' + sigmoid(W_g h) * (W_v h)`` where ``a'`` is the amplitude
    features projected to the output width (residual from the amplitude
    branch, as drawn in Fig. 10)."""

    def __init__(self, d_amp: int, d_phase: int, d: int = 256):
        super().__init__()
        self.wa, self.wp = nn.Linear(d_amp, d), nn.Linear(d_phase, d)
        self.ln = nn.LayerNorm(d)
        self.gate, self.value = nn.Linear(d, d), nn.Linear(d, d)
        self.skip = nn.Linear(d_amp, d)

    def forward(self, a, p):
        h = self.ln(F.elu(self.wa(a)) + (F.elu(self.wp(p)) if p is not None else 0))
        return self.skip(a) + torch.sigmoid(self.gate(h)) * self.value(h)


class PACSI(nn.Module):
    """PA-CSI ([Quy25] Fig. 4).

    Input: ``(b, L, n_amp + n_phase)`` with amplitude columns first.
    Amplitude and phase each go through a two-stream (temporal / channel)
    encoder of ``n_layers`` MCAT layers with Gaussian Range Encoding, are
    reduced by CNN max-pooling, fused by a GRN (256 units) and classified.

    ``use_amp`` / ``use_phase`` / ``use_gre`` / ``use_grn`` switch components
    off for the ablation study (without GRN the two feature vectors are
    concatenated and passed to a linear layer).

    Head counts must divide the token widths: ``h_heads | d_model`` and
    ``v_heads | seq_len`` (the paper used 9 and 50 for its 90 x 1000 inputs).
    """

    def __init__(self, seq_len: int, n_amp: int, n_phase: int, n_classes: int,
                 d_model: int = 96, n_layers: int = 4, h_heads: int = 8, v_heads: int = 10,
                 k_gauss: int = 10, dropout: float = 0.1, grn_dim: int = 256,
                 use_amp: bool = True, use_phase: bool = True, use_gre: bool = True,
                 use_grn: bool = True):
        super().__init__()
        assert use_amp or use_phase
        self.n_amp, self.use_amp, self.use_phase, self.use_grn = n_amp, use_amp, use_phase, use_grn
        kw = dict(seq_len=seq_len, d_model=d_model, n_layers=n_layers, h_heads=h_heads,
                  v_heads=v_heads, k_gauss=k_gauss, dropout=dropout, use_gre=use_gre)
        self.enc_a = TwoStreamEncoder(n_feat=n_amp, **kw) if use_amp else None
        self.enc_p = TwoStreamEncoder(n_feat=n_phase, **kw) if use_phase else None
        d_a = self.enc_a.out_dim if use_amp else 0
        d_p = self.enc_p.out_dim if use_phase else 0
        if use_grn:
            # with a single modality the GRN still gates that modality
            both = use_amp and use_phase
            self.fuse = GRN(d_a if use_amp else d_p, d_p if both else 1, grn_dim)
        else:
            self.fuse = nn.Sequential(nn.Linear(d_a + d_p, grn_dim), nn.ELU())
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(grn_dim, n_classes)

    def forward(self, x):
        a = self.enc_a(x[..., :self.n_amp]) if self.use_amp else None
        p = self.enc_p(x[..., self.n_amp:]) if self.use_phase else None
        if self.use_grn:
            if a is None:
                a, p = p, None
            z = self.fuse(a, p)
        else:
            z = self.fuse(torch.cat([t for t in (a, p) if t is not None], 1))
        return self.out(self.drop(z))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
