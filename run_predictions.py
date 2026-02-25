"""
run_predictions.py — batch inference on a JSON data file.

Usage:
    python run_predictions.py                           # alex_last500.json, last 2 months
    python run_predictions.py --file my_data.json
    python run_predictions.py --file my_data.json --months 3
    python run_predictions.py --file my_data.json --all   # no date filter
    python run_predictions.py --no-plot                 # skip histogram
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

def main():
    parser = argparse.ArgumentParser(description='Run glucose predictions on a JSON data file.')
    parser.add_argument('--file',    default='alex_last500.json', help='Input JSON file')
    parser.add_argument('--config',  default='configuration.yaml', help='YAML config file')
    parser.add_argument('--months',  type=float, default=2.0,
                        help='Only predict on last N months of data (default: 2)')
    parser.add_argument('--all',     action='store_true',
                        help='Run on all records, ignoring --months')
    parser.add_argument('--no-plot', action='store_true', help='Skip histogram')
    parser.add_argument('--output',  default='', help='Save histogram to this path (default: auto)')
    args = parser.parse_args()

    # --- Load config & model ---
    print(f'Loading config: {args.config}')
    config = utils.load_config(args.config)

    model_files = sorted(
        f for f in os.listdir('.')
        if f.startswith('regression_model') and f.endswith(('.pt', '.pth'))
    )
    if not model_files:
        sys.exit('ERROR: No regression_model*.pt/pth found in current directory.')
    model_path = model_files[0]
    print(f'Loading model:  {model_path}')
    model = utils.load_model(model_path, config)

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
    if glucose_vals:
        g = np.array(glucose_vals)
        print(f'  n={len(g)}  mean={g.mean():.1f}  median={np.median(g):.1f}'
              f'  min={g.min():.1f}  max={g.max():.1f}  errors={errors}')

    # --- Histogram ---
    if args.no_plot or not glucose_vals:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\nmatplotlib not installed — skipping histogram.')
        return

    g = np.array(glucose_vals)
    date_start = timestamps[0].strftime('%b %d').replace(' 0', ' ')
    date_end   = timestamps[-1].strftime('%b %d, %Y').replace(' 0', ' ')

    fig, ax = plt.subplots(figsize=(9, 5))
    n, bins, patches = ax.hist(g, bins=20, edgecolor='white', linewidth=0.6, alpha=0.85)

    zone_colors = {'hypo': '#e74c3c', 'normal': '#2ecc71',
                   'prediabetic': '#f39c12', 'diabetic': '#e74c3c'}
    for patch, left in zip(patches, bins[:-1]):
        if left < 70:
            patch.set_facecolor(zone_colors['hypo'])
        elif left < 100:
            patch.set_facecolor(zone_colors['normal'])
        elif left < 126:
            patch.set_facecolor(zone_colors['prediabetic'])
        else:
            patch.set_facecolor(zone_colors['diabetic'])

    ylim_top = ax.get_ylim()[1]
    for xval, label, color in [
        (70,  'Hypo (<70)',   zone_colors['hypo']),
        (100, 'Normal',       zone_colors['normal']),
        (126, 'Diabetic',     zone_colors['diabetic']),
    ]:
        ax.axvline(xval, color=color, linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(xval + 1, ylim_top * 0.97, label, color=color, fontsize=8, va='top')

    ax.axvline(g.mean(),      color='white',  linestyle='-',  linewidth=1.5,
               label=f'Mean:   {g.mean():.1f} mg/dL')
    ax.axvline(np.median(g),  color='yellow', linestyle=':',  linewidth=1.5,
               label=f'Median: {np.median(g):.1f} mg/dL')

    subject = os.path.splitext(os.path.basename(args.file))[0]
    ax.set_title(f'{subject} — Glucose Predictions ({date_start}–{date_end})  n={len(g)}',
                 fontsize=13)
    ax.set_xlabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=10)

    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()

    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.file))[0]
        suffix = 'all' if args.all else f'last{args.months:.0f}mo'
        out_path = f'{base}_{suffix}_histogram.png'

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nHistogram saved: {out_path}')


if __name__ == '__main__':
    main()
