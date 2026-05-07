"""
run_inference_test.py
---------------------
Loads the model/config exactly as app.py does on Railway, then runs 1000
predictions against real records from the pkl file referenced in config and
plots the results.

Pre-binned pkl records are used as-is (their feature_vectors are the output
of the same wavelength_binning3 used in the Railway pipeline), so everything
downstream of the binning step is exercised identically.

Usage (from the backend folder):
    python run_inference_test.py
    python run_inference_test.py --pkl D:/11-Meditor/research/allRecords6.pkl
    python run_inference_test.py --n 500
"""

import os
import sys
import argparse
import pickle
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import utils

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--config',  default='baseline-1_configuration.yaml')
parser.add_argument('--model',   default='baseline-1_regression_model.pt')
parser.add_argument('--pop_avg', default='baseline-1_average.npy')
parser.add_argument('--pkl',     default=None,
                    help='Path to records pkl. Defaults to config[data][file], '
                         'then ../research/allRecords6.pkl as fallback.')
parser.add_argument('--n',       type=int, default=1000)
parser.add_argument('--seed',    type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# ── Load model artifacts (exactly as app.py) ─────────────────────────────────
print("Loading config …")
cfg = utils.load_config(args.config)

print("Loading model …")
model = utils.load_model(args.model, cfg)
model.eval()

print("Loading population average …")
pop_avg = np.load(args.pop_avg)
print(f"  pop_avg shape: {pop_avg.shape}")

# ── Locate pkl data ───────────────────────────────────────────────────────────
pkl_path = args.pkl
if pkl_path is None:
    cfg_file = cfg['data']['file']
    candidates = [
        cfg_file,
        os.path.join('..', 'research', cfg_file),
        os.path.join('..', 'research', 'allRecords6.pkl'),
    ]
    for c in candidates:
        if os.path.exists(c):
            pkl_path = c
            break
    if pkl_path is None:
        sys.exit(f"Cannot find pkl. Tried: {candidates}\nPass --pkl <path>")

print(f"Loading data from {pkl_path} …")
with open(pkl_path, 'rb') as f:
    records = pickle.load(f)
print(f"  {len(records)} records total")

# ── Pull config flags (mirrors preprocess_for_inference) ────────────────────
use_signal_std   = cfg['data'].get('use_signal_std', False)
use_absorption   = cfg['data'].get('use_absorption', True)
pop_norm_only    = cfg['data'].get('population_normalize_only', False)
pop_norm         = cfg['data'].get('population_normalize', False)
use_time         = cfg['data'].get('use_time', False)
add_deriv        = cfg['spectral'].get('add_spectral_derivatives', False)
deriv_order      = int(cfg['spectral'].get('derivative_order', 1))

# ── Preprocess a single record (mirrors v16 path in preprocess_for_inference) ─
#
# pkl feature_vectors layout (from generate_training_data2.py / wavelength_binning3):
#   [0] measure_b   [1] measure_std_b   [2] dark_b     [3] dark_std_b
#   [4] reference_b [5] reference_std_b [6] absorption_b [7] absorption_std_b
#
# This is exactly the output of wavelength_binning3, so we start from here
# and follow the identical downstream steps as preprocess_for_inference.
def preprocess_record(feat, time_of_day):
    """feat: np.ndarray shape (8, 67)"""
    if use_signal_std:
        x = np.stack([feat[0], feat[1], feat[2], feat[3], feat[4], feat[5]], axis=0)
    else:
        x = np.stack([feat[0], feat[2], feat[4]], axis=0)   # measure, dark, reference
    x = x[np.newaxis]                                        # [1, C, 67]

    if use_absorption:
        x = np.concatenate([x, feat[6][np.newaxis, np.newaxis]], axis=1)
        if use_signal_std:
            x = np.concatenate([x, feat[7][np.newaxis, np.newaxis]], axis=1)

    if add_deriv:
        x = utils.add_spectral_derivatives(x, deriv_order)

    eps = 1e-8
    if pop_norm or pop_norm_only:
        x_pop = (x - pop_avg[np.newaxis]) / (np.abs(pop_avg[np.newaxis]) + eps)
        if pop_norm_only:
            x_pop_t = torch.from_numpy(x_pop).double()
            if use_time:
                angle = 2 * np.pi * (float(time_of_day or 0.0) / 24.0)
                tc = np.zeros((1, 2, x_pop.shape[2]), dtype=np.float64)
                tc[0, 0] = np.sin(angle)
                tc[0, 1] = np.cos(angle)
                tc_t = torch.from_numpy(tc).double()
                return torch.cat([x_pop_t, tc_t], dim=1)
            return x_pop_t

    x_t = torch.from_numpy(x).double()
    if cfg['normalization'].get('per_channel_norm', False):
        x_t = utils.apply_per_channel_snv(x_t)
    if pop_norm:
        x_pop_t = torch.from_numpy(x_pop).double()
        x_t = torch.cat([x_t, x_pop_t], dim=1)
    if use_time:
        angle = 2 * np.pi * (float(time_of_day or 0.0) / 24.0)
        tc = np.zeros((1, 2, x_t.shape[2]), dtype=np.float64)
        tc[0, 0] = np.sin(angle)
        tc[0, 1] = np.cos(angle)
        x_t = torch.cat([x_t, torch.from_numpy(tc).double()], dim=1)
    return x_t

# ── Run predictions ───────────────────────────────────────────────────────────
sample = random.sample(records, min(args.n, len(records)))

preds, actuals, sigmas = [], [], []
skipped = 0

for rec in sample:
    try:
        feat = np.array(rec['feature_vectors'])   # (8, 67)
        glucose = float(rec['glucose'])
        time_of_day = rec.get('time', None)

        x = preprocess_record(feat, time_of_day)
        mu, lv = utils.regression_inference(model, x)
        preds.append(float(mu.item()))
        actuals.append(glucose)
        sigmas.append(float(lv.item()))
    except Exception as e:
        skipped += 1
        if skipped <= 3:
            print(f"  Skipped record: {e}")

preds   = np.array(preds)
actuals = np.array(actuals)
sigmas  = np.array(sigmas)
errors  = preds - actuals

print(f"\n{'='*55}")
print(f"Predictions: {len(preds)}  (skipped {skipped})")
print(f"  MAE  : {np.mean(np.abs(errors)):.2f} mg/dL")
print(f"  RMSE : {np.sqrt(np.mean(errors**2)):.2f} mg/dL")
from sklearn.metrics import r2_score
print(f"  R²   : {r2_score(actuals, preds):.4f}")
print(f"  MARD : {np.mean(np.abs(errors) / (actuals + 1e-8)) * 100:.1f}%")
print(f"  Pred range : {preds.min():.1f} – {preds.max():.1f} mg/dL")
print(f"  Actual range: {actuals.min():.1f} – {actuals.max():.1f} mg/dL")
print(f"{'='*55}\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    f"Inference test  |  n={len(preds)}  |  "
    f"MAE={np.mean(np.abs(errors)):.1f}  RMSE={np.sqrt(np.mean(errors**2)):.1f}  "
    f"R²={r2_score(actuals, preds):.3f}",
    fontsize=14, fontweight='bold'
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# 1. Predicted vs Actual scatter
ax1 = fig.add_subplot(gs[:, 0])
ax1.scatter(actuals, preds, alpha=0.3, s=10, color='steelblue')
lo = min(actuals.min(), preds.min()) - 5
hi = max(actuals.max(), preds.max()) + 5
ax1.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Ideal')
z = np.polyfit(actuals, preds, 1)
ax1.plot(np.sort(actuals), np.poly1d(z)(np.sort(actuals)),
         'g-', lw=1.5, label=f'Fit: {z[0]:.3f}x+{z[1]:.1f}')
ax1.set_xlabel('Actual glucose (mg/dL)')
ax1.set_ylabel('Predicted glucose (mg/dL)')
ax1.set_title('Predicted vs Actual')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', 'box')

# 2. Error histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(errors, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax2.axvline(0, color='r', lw=1.5, linestyle='--')
ax2.axvline(np.mean(errors), color='orange', lw=1.5,
            label=f'Mean={np.mean(errors):.1f}')
ax2.set_xlabel('Error (pred − actual, mg/dL)')
ax2.set_ylabel('Count')
ax2.set_title('Error distribution')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. Absolute error vs actual glucose
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(actuals, np.abs(errors), alpha=0.3, s=8, color='coral')
ax3.set_xlabel('Actual glucose (mg/dL)')
ax3.set_ylabel('|Error| (mg/dL)')
ax3.set_title('Error magnitude vs glucose level')
ax3.grid(True, alpha=0.3)

# 4. Sigma (log_var) vs absolute error
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(sigmas, np.abs(errors), alpha=0.3, s=8, color='mediumpurple')
ax4.set_xlabel('Model log_var (uncertainty)')
ax4.set_ylabel('|Error| (mg/dL)')
ax4.set_title('Uncertainty vs error')
ax4.grid(True, alpha=0.3)

# 5. Residuals vs predicted
ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(preds, errors, alpha=0.3, s=8, color='teal')
ax5.axhline(0, color='r', lw=1.5, linestyle='--')
ax5.axhline(np.mean(errors) + 2*np.std(errors), color='orange', lw=1,
            linestyle=':', label=f'±2σ={2*np.std(errors):.1f}')
ax5.axhline(np.mean(errors) - 2*np.std(errors), color='orange', lw=1, linestyle=':')
ax5.set_xlabel('Predicted glucose (mg/dL)')
ax5.set_ylabel('Residual (mg/dL)')
ax5.set_title('Residuals vs predicted')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

out_path = 'inference_test_results.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Plot saved → {out_path}")
plt.show()
