"""
run_predictions.py — batch inference on a JSON data file, or Clarke EGA on CSV test data.

Usage:
    python run_predictions.py                              # alex_last500.json, last 2 months, baseline 120
    python run_predictions.py --file my_data.json
    python run_predictions.py --file my_data.json --baseline 70
    python run_predictions.py --file my_data.json --months 3
    python run_predictions.py --file my_data.json --all    # no date filter
    python run_predictions.py --no-plot                    # skip histogram
    python run_predictions.py --clarke                     # Clarke EGA on 2026_diabetic_testdata/123/
    python run_predictions.py --clarke --dir path/to/csvs  # custom CSV directory

Baseline routing (matches app.py):
    < 100  → baseline70_regression_model.pt + baseline70_configuration.yaml
    100-119 → baseline100_regression_model.pt + baseline100_configuration.yaml
    ≥ 120  → baseline120_regression_model.pt + baseline120_configuration.yaml
"""

import argparse
import json
import sys
import os
from datetime import datetime, timezone, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_dt(s):
    s = s.replace('+00', '+00:00') if isinstance(s, str) and s.endswith('+00') else s
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.strptime(
            s.split('+')[0].strip(), '%Y-%m-%d %H:%M:%S.%f'
        ).replace(tzinfo=timezone.utc)


def parse_field(v):
    """Parse a field that may be a JSON string or already a list."""
    if isinstance(v, str):
        return json.loads(v)
    return v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def select_baseline_key(baseline_value: float) -> int:
    """Map a patient baseline glucose value to the appropriate model key."""
    if baseline_value < 100:
        return -1
    elif baseline_value < 120:
        return 100
    else:
        return 120


def main():
    parser = argparse.ArgumentParser(description='Run glucose predictions on a JSON data file.')
    parser.add_argument('--file',     default='alex_last500.json', help='Input JSON file')
    parser.add_argument('--baseline', type=float, default=120,
                        help='Patient baseline glucose (default: 120). Selects the model: '
                             '<100 → baseline70, 100-119 → baseline100, ≥120 → baseline120')
    parser.add_argument('--months',   type=float, default=2.0,
                        help='Only predict on last N months of data (default: 2)')
    parser.add_argument('--all',      action='store_true',
                        help='Run on all records, ignoring --months')
    parser.add_argument('--no-plot',  action='store_true', help='Skip histogram')
    parser.add_argument('--output',   default='', help='Save histogram to this path (default: auto)')
    parser.add_argument('--clarke',   action='store_true',
                        help='Run Clarke EGA on diabeticRecords.pkl')
    parser.add_argument('--pkl',      default='diabeticRecords.pkl',
                        help='PKL file for --clarke mode (default: diabeticRecords.pkl)')
    args = parser.parse_args()

    if args.clarke:
        run_clarke(args.pkl, baseline=args.baseline, no_plot=args.no_plot, output=args.output)
        return

    # --- Select model based on baseline ---
    key         = select_baseline_key(args.baseline)
    config_file = f'baseline{key}_configuration.yaml'
    model_file  = f'baseline{key}_regression_model.pt'

    print(f'Baseline:       {args.baseline} → using baseline{key} model')
    print(f'Loading config: {config_file}')
    config = utils.load_config(config_file)
    print(f'Loading model:  {model_file}')
    model = utils.load_model(model_file, config)

    # --- Load data ---
    print(f'Loading data:   {args.file}')
    with open(args.file, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    print(f'Total records:  {len(data)}')

    # --- Date filter ---
    records_with_dt = [(parse_dt(r['created_at']), r) for r in data]
    if args.all:
        selected = sorted(records_with_dt, key=lambda x: x[0])
        date_label = 'all dates'
    else:
        max_dt = max(dt for dt, _ in records_with_dt)
        cutoff = max_dt - timedelta(days=args.months * 30.44)
        selected = sorted(
            [(dt, r) for dt, r in records_with_dt if dt >= cutoff],
            key=lambda x: x[0]
        )
        date_label = f'last {args.months:.0f} months'

    print(f'Selected:       {len(selected)} records ({date_label})')
    if not selected:
        sys.exit('No records matched the filter.')

    # --- Inference ---
    print('\n' + '-' * 55)
    print(f'  {"Date/Time":<22}  {"Glucose (mg/dL)":>16}  {"Sigma":>7}')
    print('-' * 55)

    glucose_vals, timestamps = [], []
    errors = 0
    for dt, r in selected:
        try:
            measure   = np.array(parse_field(r['measure']),    dtype=float)
            reference = np.array(parse_field(r['reference']),  dtype=float)
            dark      = np.array(parse_field(r['dark']),       dtype=float)
            cal_data  = np.array(parse_field(r['cal_data']),   dtype=float)

            x = utils.preprocess_for_inference(measure, reference, dark, cal_data, config)
            mu, log_var = utils.regression_inference(model, x)

            g = float(mu.item())
            s = float(log_var.item())
            glucose_vals.append(g)
            timestamps.append(dt)
            print(f'  {dt.strftime("%Y-%m-%d %H:%M"):<22}  {g:>16.1f}  {s:>7.4f}')
        except Exception as e:
            errors += 1
            print(f'  {dt.strftime("%Y-%m-%d %H:%M"):<22}  ERROR: {e}')

    print('-' * 55)
    if not glucose_vals:
        return
    g_all = np.array(glucose_vals)
    print(f'  n={len(g_all)}  mean={g_all.mean():.1f}  median={np.median(g_all):.1f}'
          f'  min={g_all.min():.1f}  max={g_all.max():.1f}  errors={errors}')

    # --- Daily summary ---
    from collections import defaultdict
    daily = defaultdict(list)
    for dt, g in zip(timestamps, glucose_vals):
        daily[dt.strftime('%Y-%m-%d')].append(g)

    print('\n' + '-' * 57)
    print(f'  {"Date":<12}  {"N":>3}  {"Mean":>8}  {"Min":>8}  {"Max":>8}')
    print('-' * 57)
    for day in sorted(daily):
        vals = np.array(daily[day])
        print(f'  {day:<12}  {len(vals):>3}  {vals.mean():>8.1f}  {vals.min():>8.1f}  {vals.max():>8.1f}')
    print('-' * 57)

    # --- Plots ---
    if args.no_plot:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\nmatplotlib not installed — skipping plots.')
        return

    BG  = '#16213e'
    AX  = '#1a1a2e'
    zone_colors = {'hypo': '#e74c3c', 'normal': '#2ecc71',
                   'prediabetic': '#f39c12', 'diabetic': '#e74c3c'}

    date_start = timestamps[0].strftime('%b %d').replace(' 0', ' ')
    date_end   = timestamps[-1].strftime('%b %d, %Y').replace(' 0', ' ')
    subject    = os.path.splitext(os.path.basename(args.file))[0]
    title_base = f'{subject} ({date_start}–{date_end}, baseline{key})'

    def style_ax(ax, fig):
        ax.set_facecolor(AX)
        fig.patch.set_facecolor(BG)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # --- Plot 1: Distribution histogram ---
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    n_hist, bins, patches = ax1.hist(g_all, bins=20, edgecolor='white', linewidth=0.6, alpha=0.85)

    for patch, left in zip(patches, bins[:-1]):
        if left < 70:      patch.set_facecolor(zone_colors['hypo'])
        elif left < 100:   patch.set_facecolor(zone_colors['normal'])
        elif left < 126:   patch.set_facecolor(zone_colors['prediabetic'])
        else:              patch.set_facecolor(zone_colors['diabetic'])

    ylim_top = ax1.get_ylim()[1]
    for xval, label, color in [
        (70,  'Hypo',    zone_colors['hypo']),
        (100, 'Normal',  zone_colors['normal']),
        (126, 'Diabetic', zone_colors['diabetic']),
    ]:
        ax1.axvline(xval, color=color, linestyle='--', linewidth=1.2, alpha=0.7)
        ax1.text(xval + 1, ylim_top * 0.97, label, color=color, fontsize=8, va='top')

    ax1.axvline(g_all.mean(),     color='white',  linestyle='-',  linewidth=1.5,
                label=f'Mean:   {g_all.mean():.1f} mg/dL')
    ax1.axvline(np.median(g_all), color='yellow', linestyle=':',  linewidth=1.5,
                label=f'Median: {np.median(g_all):.1f} mg/dL')

    ax1.set_title(f'{title_base} — Distribution  n={len(g_all)}', fontsize=12)
    ax1.set_xlabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.legend(fontsize=10)
    style_ax(ax1, fig1)
    plt.tight_layout()

    # --- Plot 2: Per-day 24-hour profiles (10-minute bins) ---
    BIN_MIN = 10  # bin width in minutes
    BINS    = 24 * 60 // BIN_MIN  # 144 bins

    # Build per-day binned averages
    # day_bins[date_str][bin_idx] = list of glucose values
    day_bins = defaultdict(lambda: defaultdict(list))
    for dt, g in zip(timestamps, glucose_vals):
        minutes  = dt.hour * 60 + dt.minute
        bin_idx  = minutes // BIN_MIN
        day_str  = dt.strftime('%Y-%m-%d')
        day_bins[day_str][bin_idx].append(g)

    sorted_days = sorted(day_bins)
    # x values in hours for each bin centre
    bin_hours = np.array([(b * BIN_MIN + BIN_MIN / 2) / 60 for b in range(BINS)])

    # Colour palette — one per day
    cmap   = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(sorted_days))]

    fig2, ax2 = plt.subplots(figsize=(13, 6))

    # Clinical zone shading
    ax2.axhspan(0,   70,  color='#e74c3c', alpha=0.07, zorder=0)
    ax2.axhspan(70,  100, color='#2ecc71', alpha=0.07, zorder=0)
    ax2.axhspan(100, 126, color='#f39c12', alpha=0.07, zorder=0)
    ax2.axhspan(126, 400, color='#e74c3c', alpha=0.07, zorder=0)
    for yval, label, color in [(70, 'Hypo', '#e74c3c'), (100, 'Normal', '#2ecc71'), (126, 'Diabetic', '#e74c3c')]:
        ax2.axhline(yval, color=color, linestyle='--', linewidth=0.7, alpha=0.45, zorder=1)
        ax2.text(23.9, yval + 1, label, color=color, fontsize=7, va='bottom', ha='right')

    # Per-day lines
    all_bin_vals = defaultdict(list)  # for cross-day average
    for day_str, col in zip(sorted_days, colors):
        xs, ys = [], []
        for b in range(BINS):
            if b in day_bins[day_str]:
                mean_g = np.mean(day_bins[day_str][b])
                xs.append(bin_hours[b])
                ys.append(mean_g)
                all_bin_vals[b].append(mean_g)
        if xs:
            ax2.plot(xs, ys, color=col, linewidth=1.4, alpha=0.7,
                     marker='o', markersize=3, label=day_str)

    # Cross-day average line (only bins with ≥2 days)
    avg_xs, avg_ys = [], []
    for b in sorted(all_bin_vals):
        if len(all_bin_vals[b]) >= 2:
            avg_xs.append(bin_hours[b])
            avg_ys.append(np.mean(all_bin_vals[b]))
    if avg_xs:
        ax2.plot(avg_xs, avg_ys, color='white', linewidth=2.5, alpha=0.9,
                 linestyle='-', zorder=5, label='Average')

    ax2.set_ylim(50, 250)
    ax2.set_xlim(0, 24)
    tick_hours = range(0, 25, 2)
    ax2.set_xticks(tick_hours)
    ax2.set_xticklabels([f'{h:02d}:00' for h in tick_hours], rotation=45, ha='right', fontsize=8)
    ax2.set_xlabel('Time of Day', fontsize=12)
    ax2.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax2.set_title(f'{title_base} — 24-Hour Profile per Day  ({BIN_MIN}-min bins)', fontsize=12)
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.3)
    style_ax(ax2, fig2)
    plt.tight_layout()

    # --- Save both plots ---
    base   = os.path.splitext(os.path.basename(args.file))[0]
    suffix = 'all' if args.all else f'last{args.months:.0f}mo'

    if args.output:
        hist_path  = args.output
        trend_path = args.output.replace('.png', '_24h.png')
    else:
        hist_path  = f'{base}_{suffix}_baseline{key}_histogram.png'
        trend_path = f'{base}_{suffix}_baseline{key}_24h_trend.png'

    fig1.savefig(hist_path,  dpi=150, bbox_inches='tight')
    fig2.savefig(trend_path, dpi=150, bbox_inches='tight')
    print(f'\nHistogram saved:  {hist_path}')
    print(f'24h trend saved:  {trend_path}')


# ---------------------------------------------------------------------------
# Clarke Error Grid Analysis
# ---------------------------------------------------------------------------

def clarke_ega_zone(ref, pred):
    """Classify a (ref, pred) pair into Clarke EGA zones A-E.
    Based on Clarke et al. (1987) Diabetes Care 10:622-628.
    """
    # Zone E: extreme opposite errors
    if ref <= 70 and pred >= 180:
        return 'E'
    if ref >= 180 and pred <= 70:
        return 'E'
    # Zone A: within 20% of reference, or both hypoglycemic
    if ref <= 70 and pred <= 70:
        return 'A'
    if abs(pred - ref) / max(ref, 1e-6) <= 0.20:
        return 'A'
    # Zone D: failure to detect dangerous glucose levels
    if ref >= 240 and 70 <= pred <= 180:
        return 'D'
    if ref <= 70 and 70 < pred <= 180:
        return 'D'
    # Zone C: overcorrection
    if 70 <= ref <= 290 and pred >= ref + 110:
        return 'C'
    if ref >= 130 and pred <= (7.0 / 5.0) * ref - 182:
        return 'C'
    return 'B'


def draw_clarke_ega_boundaries(ax):
    """Draw the Clarke EGA zone boundary lines on ax."""
    # ±20% lines (zone A boundaries)
    x = np.linspace(0, 400, 500)
    ax.plot(x, x * 1.20, 'k--', linewidth=0.8, alpha=0.5)
    ax.plot(x, x * 0.80, 'k--', linewidth=0.8, alpha=0.5)
    # Hypo / hyper clinical thresholds
    ax.axvline(70,  color='gray', linewidth=0.6, alpha=0.4)
    ax.axvline(180, color='gray', linewidth=0.6, alpha=0.4)
    ax.axhline(70,  color='gray', linewidth=0.6, alpha=0.4)
    ax.axhline(180, color='gray', linewidth=0.6, alpha=0.4)
    # Zone D upper boundary (ref <= 70, pred > 70) — upper-D top at pred=180
    ax.plot([0, 70],  [180, 180], color='gray', linewidth=0.6, alpha=0.4)
    # Zone D lower boundary (ref >= 240, pred <= 180)
    ax.plot([240, 400], [70, 70], color='gray', linewidth=0.6, alpha=0.4)
    ax.plot([240, 240], [0,  70], color='gray', linewidth=0.6, alpha=0.4)


def preprocess_from_binned(fv, config):
    """
    Convert pre-binned feature_vectors from diabeticRecords.pkl into a model input tensor.

    feature_vectors order (from generate_training_data_diabetic.py):
        [measure, measure_std, dark, dark_std, reference, reference_std, absorption, absorption_std]
    Each element is a numpy array of shape (330,) covering 420–750 nm at 1 nm bins.
    """
    import torch as _torch
    wl_range         = config.get('data', {}).get('wavelength_nm_range', [450, 650])
    use_absorption   = config.get('data', {}).get('use_absorption', True)
    use_signal_std   = config.get('data', {}).get('use_signal_std', True)
    add_deriv        = config.get('spectral', {}).get('add_spectral_derivatives', False)
    deriv_order      = int(config.get('spectral', {}).get('derivative_order', 1))
    per_channel_norm = config.get('normalization', {}).get('per_channel_norm', False)

    measure_b, measure_std_b = np.array(fv[0]), np.array(fv[1])
    dark_b,    dark_std_b    = np.array(fv[2]), np.array(fv[3])
    reference_b, reference_std_b = np.array(fv[4]), np.array(fv[5])
    absorption_b, absorption_std_b = np.array(fv[6]), np.array(fv[7])

    if use_signal_std:
        x = np.stack([measure_b, measure_std_b, dark_b, dark_std_b,
                      reference_b, reference_std_b], axis=0)
    else:
        x = np.stack([measure_b, dark_b, reference_b], axis=0)
    x = np.expand_dims(x, axis=0)  # [1, C, 330]

    start_idx = int(wl_range[0]) - 420
    end_idx   = int(wl_range[1]) - 420
    x = x[:, :, start_idx:end_idx]

    if use_absorption:
        ab = absorption_b[start_idx:end_idx][np.newaxis, np.newaxis, :]  # [1,1,N]
        x = np.concatenate([x, ab], axis=1)
        if use_signal_std:
            ab_std = absorption_std_b[start_idx:end_idx][np.newaxis, np.newaxis, :]
            x = np.concatenate([x, ab_std], axis=1)

    if add_deriv:
        x = utils.add_spectral_derivatives(x, deriv_order)

    x_tensor = _torch.from_numpy(x).double().to(_torch.device('cpu'))
    if per_channel_norm:
        x_tensor = utils.apply_per_channel_snv(x_tensor)
    return x_tensor


def run_clarke(pkl_path, baseline=120, no_plot=False, output=''):
    """Load diabeticRecords.pkl, run inference, and draw Clarke EGA."""
    import pickle

    key         = select_baseline_key(baseline)
    config_file = f'baseline{key}_configuration.yaml'
    model_file  = f'baseline{key}_regression_model.pt'

    print(f'Baseline:       {baseline} → using baseline{key} model')
    print(f'Loading config: {config_file}')
    config = utils.load_config(config_file)
    print(f'Loading model:  {model_file}')
    model = utils.load_model(model_file, config)

    print(f'Loading data:   {pkl_path}')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f'Records:        {len(data)}\n')

    pairs  = []
    errors = 0
    for i, rec in enumerate(data):
        try:
            ref_glucose = float(rec['glucose'])
            fv = rec['feature_vectors']
            x  = preprocess_from_binned(fv, config)
            mu, _ = utils.regression_inference(model, x)
            pairs.append((ref_glucose, float(mu.item())))
        except Exception:
            errors += 1
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{len(data)} processed ...')

    print(f'\nTotal: {len(pairs)} predictions, {errors} errors\n')
    if not pairs:
        return

    refs  = np.array([p[0] for p in pairs])
    preds = np.array([p[1] for p in pairs])
    zones = [clarke_ega_zone(r, p) for r, p in zip(refs, preds)]

    # Zone statistics
    zone_counts = {z: zones.count(z) for z in 'ABCDE'}
    n = len(zones)
    print('Clarke EGA zone distribution:')
    print('-' * 40)
    for z, label in [('A', 'Clinically accurate'),
                     ('B', 'Benign error'),
                     ('C', 'Overcorrection'),
                     ('D', 'Failure to detect'),
                     ('E', 'Dangerous error')]:
        cnt = zone_counts.get(z, 0)
        print(f'  Zone {z} ({label}): {cnt:>4}  ({100*cnt/n:.1f}%)')
    print('-' * 40)

    if no_plot:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\nmatplotlib not installed — skipping plot.')
        return

    ZONE_COLORS = {'A': '#2ecc71', 'B': '#3498db', 'C': '#f39c12',
                   'D': '#e67e22', 'E': '#e74c3c'}
    BG = '#16213e'
    AX = '#1a1a2e'

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX)

    draw_clarke_ega_boundaries(ax)

    for z in 'ABCDE':
        mask = [i for i, zz in enumerate(zones) if zz == z]
        if mask:
            ax.scatter(refs[mask], preds[mask],
                       color=ZONE_COLORS[z], label=f'Zone {z} (n={len(mask)}, {100*len(mask)/n:.0f}%)',
                       alpha=0.7, s=18, edgecolors='none')

    # Zone labels (approximate centres)
    for z, (tx, ty) in [('A', (100, 100)), ('B', (50, 150)),
                         ('C', (100, 280)), ('D', (300, 100)), ('E', (30, 350))]:
        if zone_counts.get(z, 0) > 0:
            ax.text(tx, ty, z, fontsize=18, fontweight='bold',
                    color=ZONE_COLORS[z], alpha=0.4, ha='center', va='center')

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12, color='white')
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12, color='white')
    ax.set_title(f'Clarke Error Grid Analysis  (n={n})\n{os.path.basename(pkl_path)}',
                 fontsize=12, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.3,
              labelcolor='white', facecolor=AX)
    plt.tight_layout()

    out_path = output if output else f'clarke_ega_{os.path.splitext(os.path.basename(pkl_path))[0]}_baseline{key}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nClarke EGA plot saved: {out_path}')


if __name__ == '__main__':
    main()
