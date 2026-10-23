# Wi-Fi CSI presence & motion detection

Detect whether a person is in a room — and whether they are **still or moving** — from Wi-Fi **Channel State Information (CSI)**, with the end goal of running on CSI from our own ESP32 capture pipeline.

The project re-implements the human-activity-recognition (HAR) models and pre-processing from several papers in PyTorch, compares them fairly on a public dataset, and applies the pipeline to our own ESP32 recordings. Everything is documented step by step in **[`csi_har_experiments.ipynb`](csi_har_experiments.ipynb)**.

> **Status:** code and notebook are complete. The full experiment grid (41 runs) is being re-run on the GPU; the HAR result tables in the notebook will be filled in once it finishes.

---

## What is in here

| Part | What it does |
|---|---|
| **Pre-processing classes** — [`csi_preprocessing.py`](csi_preprocessing.py) | Every pre-processing step from the papers as a documented, scikit-learn-style transformer (`fit` / `transform`), chainable in a `Pipeline` and reusable on live data |
| **Models** — [`csi_models.py`](csi_models.py) | **LSTM-CNN** (core model), LSTM, BiLSTM, InceptionTime, PA-CSI |
| **Experiments** — [`csi_experiments.py`](csi_experiments.py), [`run_all.py`](run_all.py) | One function, `run_har(model, task, protocol)`, runs a full experiment and caches the result; `run_all.py` runs the whole grid |
| **ESP32 pipeline** — [`csi_esp32.py`](csi_esp32.py) | Parse our ESP32 CSV files, resample, normalise, leak-free cross-validation, motion detector |
| **Notebook** — [`csi_har_experiments.ipynb`](csi_har_experiments.ipynb) | Explains every step and every decision, shows all plots and results |
| **Data download** — [`download_data.py`](download_data.py) | Fetches the datasets from Google Drive (data is not stored in git) |

### Papers reproduced

| Paper | Used for |
|---|---|
| Shang et al. 2021, *LSTM-CNN network for human activity recognition using WiFi CSI data* | **Core model (LSTM-CNN)**, LSTM / KNN / SVM baselines, Butterworth filter, antenna phase difference and amplitude product inputs |
| Zhuravchak et al. 2022, *Human Activity Recognition based on Wi-Fi CSI Data — A Deep Neural Network Approach* | The training dataset; phase sanitisation, Hampel filter, wavelet de-noising; BiLSTM and InceptionTime |
| Quy et al. 2025, *Enhanced HAR Using Wi-Fi Sensing: Leveraging Phase and Amplitude with Attention Mechanisms* (PA-CSI) | Kalman filter, phase unwrapping, time alignment; PA-CSI model and ablations |
| Korany & Mostofi 2021, *Nocturnal Seizure Detection Using Off-the-Shelf WiFi* | PCA de-noising and a spectral motion detector (no training needed) |
| Chang et al. 2023, *Attention-based Learning for Sleep Apnea and Limb Movement Detection using Wi-Fi CSI* | Resampling irregular packets, per-packet normalisation |
| Guarino et al. 2026, *A survey on CSI-based Wi-Fi sensing datasets and models with a focus on reproducibility* | Evaluation methodology (avoiding leaky splits) |

### Pre-processing classes

| Class | From | Built on |
|---|---|---|
| `HampelFilter` — outlier removal | Zhuravchak, Korany | `scipy.ndimage` |
| `WaveletDenoiser` — DWT shrinkage | Zhuravchak | `pywt` |
| `PhaseSanitizer` — remove linear phase error (CFO/SFO) | Zhuravchak (Sen et al.) | NumPy |
| `PhaseUnwrapper` | PA-CSI | `numpy.unwrap` |
| `PhaseDifference`, `AmplitudeProduct` — antenna-pair features | Shang, Korany | NumPy |
| `ButterworthFilter` — low/band-pass, zero-phase or causal | Shang, ALESAL | `scipy.signal` |
| `KalmanSmoother` | PA-CSI | NumPy |
| `TimeAligner` — bin averaging to a fixed length | PA-CSI | NumPy |
| `MinMaxNormalizer`, `Standardizer` | PA-CSI, ALESAL | scikit-learn |
| `PCADenoiser` | Korany, ALESAL | scikit-learn |
| `SubcarrierSelector`, `ESP32CSIParser`, `UniformResampler` | — / ALESAL | NumPy, pandas |

```python
from sklearn.pipeline import Pipeline
import csi_preprocessing as pp

amp_pipeline = Pipeline([
    ("hampel", pp.HampelFilter(window=5, n_sigma=3)),
    ("wavelet", pp.WaveletDenoiser("db4", level=4)),
    ("scale", pp.Standardizer()),
])
features = amp_pipeline.fit_transform(amplitude)   # (packets, streams)
```

---

## How the models are evaluated

The models are trained on three tasks:

- `activity7` — the 7 activities of the paper
- **`presence_motion3`** — empty / static / moving, which is the project goal
- `static_dynamic2` — Shang's binary task

Each task is evaluated with three protocols:

| Protocol | Test data | Why |
|---|---|---|
| `random` | random 15 % of all windows | what the papers report — but overlapping windows make this **leaky**: a test window shares ~96–98 % of its packets with a training window |
| `cross_session` | two recordings never seen in training | does the model work on a new recording? |
| `cross_room` | a room never seen in training | does the model work in a new room? (closest to deploying in our own room) |

All models are scored on the same target packets with accuracy **and** macro-F1 (the data is imbalanced: walking is 39 % of it).

## Findings so far (ESP32 data)

Our ESP32 recordings consist of three 2-minute sessions: empty room, person standing still, person moving.

- **The ESP32 logs every device on the channel** — 650–750 MAC addresses per file. Only the access point's link is usable, and it arrives at about 3 packets/s with gaps of up to 2 s. The pipeline keeps that one link and resamples it to 5 Hz.
- **Signal strength (RSSI) gives the answer away:** −29 dBm in the empty room versus −41 dBm with a person present. An RSSI-only classifier scores 99 %. The models therefore never see RSSI or absolute amplitude; each packet is normalised across its subcarriers.
- **Results with leak-free blocked 5-fold cross-validation over time:**
  - SVM: 100 %
  - LSTM-CNN: 98 %
  - Motion detector (no training): flags 88 % of *moving* windows and under 1 % of *empty* or *still* windows.
- **What these results do *not* show yet:** whether the models work on another day. Each class comes from a single session, so the classes cannot be separated from session effects. The notebook (section 10) gives a data-collection protocol to fix this.

---

## Setup

Python ≥ 3.12. Dependencies are listed in [`pyproject.toml`](pyproject.toml).

```bash
uv sync                    # or: pip install -e .
python download_data.py    # fetch the data (see Data below)
```

**GPU.** Training uses the best available device (`csi_train.get_device()`):

1. an Intel GPU via PyTorch's `xpu` backend,
2. an NVIDIA GPU (`cuda`),
3. the CPU.

Set `CSI_DEVICE=cpu` to force the CPU. The default PyTorch wheel targets NVIDIA. For an **Intel Arc GPU** (as on the development machine):

```bash
# Intel GPU driver (Ubuntu 24.04)
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc
# PyTorch build for Intel GPUs
uv pip install torch --index-url https://download.pytorch.org/whl/xpu
python -c "import torch; print(torch.xpu.is_available())"
```

## Data

The data is **not stored in git**. Download it from Google Drive with:

```bash
python download_data.py            # all datasets -> data/
python download_data.py esp32      # only our ESP32 recordings
```

| Dataset | Size | Unpacked to |
|---|---|---|
| `har`: Zhuravchak et al. HAR dataset (Atheros CSI tool, 5 GHz, 2×2 antennas, 114 subcarriers, 3 rooms, 7 activities) | 490 MB zip / 1.1 GB | `data/wifi_csi_har_dataset/room_*/<n>/{data,label,label_boxes}.csv` |
| `esp32`: our own ESP32 recordings (2.4 GHz, 1 antenna, 64 subcarriers): empty room, standing still, moving | 1 MB | `data/self_extracted_dataset/*.csv` |

The HAR dataset is also available from its authors at <https://doi.org/10.6084/m9.figshare.14386892.v1>. If you download it from there, unpack it so that `data/wifi_csi_har_dataset/room_1/1/data.csv` exists.

The HAR `data.csv` files have no header: 114 subcarrier-index columns (ignored), then 456 amplitudes, then 456 phases (4 antenna pairs × 114 subcarriers).

**Maintainers: publishing a new data version**

1. Run `python download_data.py --pack`. This writes the zip files to `dist/`, which is git-ignored.
2. Upload them to Google Drive.
3. Share each file as *Anyone with the link → Viewer*.
4. Paste the share links into `DATASETS` in `download_data.py`.

## Running

```bash
python run_all.py --workers 1      # full experiment grid; finished runs are cached in results/ and skipped
jupyter lab csi_har_experiments.ipynb
```

- The notebook loads cached results, so re-running it is fast.
- Set `FORCE = True` in its setup cell to retrain inside the notebook.
- Pre-processed features are cached in `cache/`.

A single experiment:

```python
from csi_experiments import run_har
res = run_har("lstm_cnn", "presence_motion3", "cross_room")
print(res["test"]["accuracy"], res["test"]["f1_macro"])
```

Models: `lstm_cnn`, `lstm_cnn_2ch`, `lstm`, `knn`, `svm`, `bilstm`, `inceptiontime`, `pacsi`, and `pacsi_amp_only` / `pacsi_phase_only` / `pacsi_no_gre` / `pacsi_no_grn`. All hyper-parameters, and every deviation from the papers, are documented in `MODEL_SPECS` in `csi_experiments.py`.

## Project layout

```
csi_preprocessing.py     pre-processing classes
csi_data.py              loading, tasks, per-paper feature recipes, splits, window datasets
csi_models.py            PyTorch models
csi_train.py             training loop, metrics, device selection, result cache
csi_experiments.py       run_har() + MODEL_SPECS
csi_esp32.py             ESP32 pipeline and blocked cross-validation
csi_plots.py             plotting helpers
run_all.py               runs the experiment grid
download_data.py         downloads the datasets from Google Drive into data/
csi_har_experiments.ipynb  the documented notebook
results/                 cached metrics per experiment (JSON)
data/                    datasets (git-ignored, created by download_data.py)
```

## Next steps

1. **Record better ESP32 data.** Use a dedicated sender (`esp-csi` `csi_send` / `csi_recv`) at ≥ 50 packets/s, with many short sessions over several days, people and positions, including transitions.
2. **Evaluate fairly.** Use leave-one-session-out or leave-one-day-out evaluation.
3. **Run it live.** Turn the notebook's inference function into a live detector.
