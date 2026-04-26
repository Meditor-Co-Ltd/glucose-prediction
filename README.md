# Meditor Backend — Glucose Prediction API

## Overview

Flask/Gunicorn inference server for non-invasive blood glucose prediction from optical spectral data.  
Always uses the **baseline-1 (v16) SPECFORMER** model regardless of patient baseline.  
Scale factors and temporal averaging are applied on top of the model output.

---

## Model: baseline-1 (v16 SPECFORMER)

| Property | Value |
|---|---|
| Architecture | SPECFORMER (transformer + conv stem) |
| Version | v16 |
| Training date | 2026-04-21 |
| Training data | `allRecords6.pkl` |
| d_model | 256 |
| Attention heads | 8 |
| Encoder layers | 8 |
| dim_feedforward | 512 |
| Conv kernel size | 15 |
| **Input shape** | **[batch, 10, 67]** |
| Output | μ (predicted glucose) + log σ² (uncertainty) |

---

## Input Preprocessing Pipeline

Raw sensor signals → `[1, 10, 67]` tensor

### Step 1 — Wavelength Binning (`wavelength_binning3`)

| Parameter | Value |
|---|---|
| Function | `wavelength_binning3` (bin_size=3) |
| Wavelength range | 450–648 nm |
| Bin centres | `range(450, 650, 3)` → **67 bins** |
| Bin window | ±3 nm around each centre |
| Outputs per bin | mean and std of: measure, dark, reference, absorption |

Absorption is computed from raw pixels before binning:
```
absorption = -log10( max(measure - dark, ε) / max(reference - dark, ε) )
```

### Step 2 — Channel Stack

With `use_signal_std=False`, `use_absorption=True`:

```
[measure, dark, reference, absorption]  →  4 base channels  [1, 4, 67]
```

### Step 3 — Spectral Derivatives (`add_spectral_derivatives`, order=1)

```
[base | d/dλ(base)]  →  8 channels  [1, 8, 67]
```

### Step 4 — Population Normalization (`population_normalize_only=True`)

The model uses **only** the population-relative channels (no SNV normalization applied).

```
x_pop = (x - pop_avg) / (|pop_avg| + ε)  →  8 channels  [1, 8, 67]
```

Population average file: `baseline-1_average.npy`  shape `(8, 67)`  
Computed from the full training set (all subjects, all baselines).

> `population_normalize_only=True` means the raw/SNV channels are discarded entirely.  
> Only the percent-deviation-from-population channels are fed to the model.

### Step 5 — Time Encoding (`use_time=True`)

Fractional hour-of-day (0.0–24.0) encoded as cyclic sin/cos, broadcast constant across all 67 wavelengths:

```
channel 8:  sin(2π × hour / 24)   [1, 1, 67]
channel 9:  cos(2π × hour / 24)   [1, 1, 67]
```

**Final tensor: `[1, 10, 67]`** = 8 pop-norm channels + 2 time channels

Time source:
- **Server (`app.py`):** extracted from `current_time` sent by the client (ISO format, UTC+9)
- **Batch (`run_predictions.py`):** extracted from `created_at` field of each JSON record
- **Clarke (`run_predictions.py --clarke`):** taken from `time` field of each pkl record (fractional hour)
- **Fallback:** 0.0 (midnight) when time is unavailable

---

## Prediction Flow (`app.py`)

```
Request
  │
  ├── parse measure, measure2, measure3, reference, dark, cal_data
  ├── parse baseline (float), current_time (ISO string), last_glucose_values
  │
  ├── extract time_of_day from current_time
  │
  ├── for each of [measure, measure2, measure3] (skip if empty):
  │     preprocess → [1, 10, 67]
  │     model inference → (μᵢ, σᵢ)
  │
  ├── predicted_glucose = mean(μ₁, μ₂, μ₃)
  ├── sigma            = mean(σ₁, σ₂, σ₃)
  │
  ├── apply scale factor (if baseline in range):
  │     baseline 100–124  →  × SCALE_FACTOR_100  (default 1.0)
  │     baseline ≥ 125    →  × SCALE_FACTOR_125  (default 1.0)
  │
  └── 2-minute rolling average with last_glucose_values:
        filter entries within last 2 min of current_time
        averaged = (predicted + Σ recent) / (1 + n_recent)
```

Response:
```json
{ "predicted_glucose": 142.30, "sigma": 0.0821, "acceptance": 25 }
```

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask inference server |
| `utils.py` | Preprocessing, model loading, inference helpers |
| `models.py` | SPECFORMER and BandEnsembleModel definitions |
| `run_predictions.py` | Batch inference on JSON / Clarke EGA on pkl |
| `baseline-1_configuration.yaml` | Model config (v16) |
| `baseline-1_regression_model.pt` | Trained model weights |
| `baseline-1_average.npy` | Population average array `(8, 67)` |

---

## Running Locally

```bash
# Start server
gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 app:app

# Batch inference on alex data (last 2 months)
python run_predictions.py

# All records
python run_predictions.py --all

# Clarke EGA on diabetic test data
python run_predictions.py --clarke
python run_predictions.py --clarke --pkl diabeticRecords.pkl
```

---

## Docker

```bash
docker build -t meditor-backend .
docker run -p 8000:8000 meditor-backend
```

Required files copied into container:
- `baseline-1_configuration.yaml`
- `baseline-1_regression_model.pt`
- `baseline-1_average.npy`

---

## Scale Factors

Defined at the top of `app.py`:

```python
SCALE_FACTOR_100 = 1.0   # applied when patient baseline is 100–124
SCALE_FACTOR_125 = 1.0   # applied when patient baseline is >= 125
```

Set to `1.0` (no scaling) by default. Update with calibration data when available.

---

## Tunable Constants (`app.py`)

| Constant | Default | Description |
|---|---|---|
| `AVERAGE_WINDOW_MINUTES` | `2` | Rolling average window for `last_glucose_values` |
| `SCALE_FACTOR_100` | `1.0` | Multiplicative scale for baseline 100–124 |
| `SCALE_FACTOR_125` | `1.0` | Multiplicative scale for baseline ≥ 125 |
