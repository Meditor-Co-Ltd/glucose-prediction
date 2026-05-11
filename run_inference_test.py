"""
run_inference_test.py
---------------------
Loads the model/config exactly as app.py does on Railway, then runs 1000
predictions against real records from the pkl file referenced in config and
plots the results.

Uses utils.preprocess_binned_for_inference() which mirrors the post-binning
steps of utils.preprocess_for_inference() used in production. The pkl's
pre-binned feature_vectors (8, 67) feed directly into the same preprocessing
path that Railway uses after wavelength_binning3.

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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import utils

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=float, default=70,
                    help='Patient baseline glucose. Selects model: <100 → baseline80, ≥100 → baseline-1')
parser.add_argument('--config',  default=None, help='Override config path')
parser.add_argument('--model',   default=None, help='Override model path')
parser.add_argument('--pop_avg', default=None, help='Override population average path')
parser.add_argument('--pkl',     default=None,
                    help='Path to records pkl. Defaults to config[data][file], '
                         'then ../research/allRecords6.pkl as fallback.')
parser.add_argument('--n',       type=int, default=1000)
parser.add_argument('--seed',    type=int, default=42)
args = parser.parse_args()

key = utils.select_baseline_key(args.baseline)
if key == 80:
    args.config  = args.config  or 'baseline80_configuration.yaml'
    args.model   = args.model   or 'baseline80_regression_model.pt'
    args.pop_avg = args.pop_avg or 'baseline80_population_average.npy'
else:
    key          = -1
    args.config  = args.config  or 'baseline-1_configuration.yaml'
    args.model   = args.model   or 'baseline-1_regression_model.pt'
    args.pop_avg = args.pop_avg or 'baseline-1_average.npy'
print(f"Using baseline{key} model (baseline={args.baseline})")

random.seed(args.seed)
np.random.seed(args.seed)

# ── Load model artifacts (exactly as app.py) ─────────────────────────────────
print("Loading config …")
cfg = utils.load_config(args.config)

print("Loading model …")
model = utils.load_model(args.model, cfg)
model.eval()

print("Loading population average …")
pop_avg = np.load(args.pop_avg) if os.path.exists(args.pop_avg) else None
if pop_avg is not None:
    print(f"  pop_avg shape: {pop_avg.shape}")
else:
    print("  pop_avg not found — skipping population normalization")

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

# ── Run predictions ───────────────────────────────────────────────────────────
sample = random.sample(records, min(args.n, len(records)))

preds, actuals, sigmas = [], [], []
skipped = 0

for rec in sample:
    try:
        feat        = np.array(rec['feature_vectors'])   # (8, 67)
        glucose     = float(rec['glucose'])
        time_of_day = rec.get('time', None)

        x        = utils.preprocess_binned_for_inference(feat, cfg, pop_avg=pop_avg, time_of_day=time_of_day)
        mu, lv   = utils.regression_inference(model, x)
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
