import os
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import warnings
import logging
import sys
import traceback
import utils

AVERAGE_WINDOW_MINUTES = 2

SCALE_FACTOR_100 = 1.2   # multiplicative scale for baseline 100–124
SCALE_FACTOR_125 = 1.4   # multiplicative scale for baseline >= 125

CAL_DEFAULT_HIGH      = 150.0  # used when cal_high is unset (0) on first use
CAL_DEFAULT_LOW       = 120.0  # used when cal_low is unset (0) on first use
CAL_RESCALE_MIN_RANGE = 10.0   # min cal_high − cal_low required to apply linear rescaling
CAL_ROLLING_WINDOW    = 10     # number of recent readings used to compute rolling cal_high / cal_low
USE_CALIBRATION       = False  # set True to apply linear rescaling to predictions

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


# --- Multi-model setup: one model per baseline range ---
BASELINE_KEYS = [-1, 80]

MODEL_FILE_MAP = {
    -1: 'baseline-1_model.pt',
    80: 'baseline80_regression_model.pt',
}

logger.info("=== Starting model initialization ===")
BASELINE_CONFIGS   = {}
BASELINE_MODELS    = {}
BASELINE_POP_AVGS  = {}

for b in BASELINE_KEYS:
    try:
        cfg = utils.load_config(f'baseline{b}_configuration.yaml')
        BASELINE_CONFIGS[b] = cfg
        BASELINE_MODELS[b]  = utils.load_model(MODEL_FILE_MAP.get(b, f'baseline{b}_regression_model.pt'), cfg)
        logger.info(f"baseline{b}: model loaded OK")
    except Exception as e:
        logger.error(f"baseline{b}: failed to load — {e}")
        logger.error(traceback.format_exc())

    for avg_path in [f'baseline{b}_average.npy', f'baseline{b}_population_average.npy']:
        if os.path.exists(avg_path):
            try:
                BASELINE_POP_AVGS[b] = np.load(avg_path)
                logger.info(f"baseline{b}: population average loaded from {avg_path}  shape={BASELINE_POP_AVGS[b].shape}")
            except Exception as e:
                logger.warning(f"baseline{b}: could not load {avg_path} — {e}")
            break

IS_REGRESSION = (
    BASELINE_CONFIGS[-1].get('model', {}).get('num_classes', 1) == 1
    if -1 in BASELINE_CONFIGS else True
)
logger.info(f"IS_REGRESSION={IS_REGRESSION}, models loaded: {list(BASELINE_MODELS.keys())}")
logger.info("=== Model initialization complete ===")

logger.info("=== Environment Information ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PORT environment variable: {os.environ.get('PORT', 'not set')}")
logger.info("=== Environment Information Complete ===")


def predict_from_json(data):
    if not BASELINE_MODELS:
        return {"error": "Models not loaded. Please restart the service."}, 500

    try:
        if isinstance(data, list):
            data = data[0]

        logger.info("=== Incoming request parameters ===")
        for key, val in data.items():
            if isinstance(val, list):
                logger.info(f"  {key}: list[{len(val)}]")
            else:
                logger.info(f"  {key}: {val!r}")
        logger.info("===================================")

        measure   = np.array(data.get("measure",   []))
        measure2  = np.array(data.get("measure2",   []))
        measure3  = np.array(data.get("measure3",   []))
        reference = np.array(data.get("reference", []))
        dark      = np.array(data.get("dark",      []))
        cal_data  = np.array(data.get("cal_data",  []))

        baseline = data.get("baseline", 70)
        if baseline in [None, "None", "null", ""]:
            baseline = 70
        else:
            try:
                baseline = float(baseline)
            except (ValueError, TypeError):
                baseline = 70

        last_glucose_values = data.get("last_glucose_values") or []
        current_time_str    = data.get("current_time", None)

        try:
            cal_high = float(data.get("cal_high") or 0)
        except (ValueError, TypeError):
            cal_high = 0.0

        try:
            cal_low = float(data.get("cal_low") or 0)
        except (ValueError, TypeError):
            cal_low = 0.0


        logger.info(f"cal_high: {data.get('cal_high')!r}  →  {cal_high}")
        logger.info(f"cal_low:  {data.get('cal_low')!r}  →  {cal_low}")

        time_of_day = None
        if current_time_str:
            try:
                _cdt = datetime.fromisoformat(current_time_str)
                time_of_day = _cdt.hour + _cdt.minute / 60.0 + _cdt.second / 3600.0
            except Exception:
                pass

        logger.info(f"Incoming request keys: {list(data.keys())}")
        logger.info(f"Incoming request — baseline={baseline}, "
                    f"measure_len={len(measure)}, reference_len={len(reference)}, "
                    f"measure2_len={len(measure2)}, reference_len={len(reference)}, "
                    f"measure3_len={len(measure3)}, reference_len={len(reference)}, "
                    f"dark_len={len(dark)}, cal_data_len={len(cal_data)}, "
                    f"current_time={current_time_str}, "
                    f"last_glucose_values={last_glucose_values}")

        for label, other in [("measure2", measure2), ("measure3", measure3)]:
            if len(other) == len(measure) and len(measure) > 0:
                denom = np.where(np.abs(measure) > 1e-9, np.abs(measure), 1e-9)
                pct_diff = np.abs(other - measure) / denom * 100
                max_idx = int(np.argmax(pct_diff))
                logger.info(
                    f"measure vs {label}: mean_pct_diff={pct_diff.mean():.2f}%, "
                    f"max_pct_diff={pct_diff[max_idx]:.2f}% at idx={max_idx}"
                )

        if len(measure) == 0 or len(reference) == 0 or len(dark) == 0 or len(cal_data) == 0:
            return {"error": "All data arrays (measure, reference, dark, cal_data) must be non-empty"}, 400


        key = -1
        config  = BASELINE_CONFIGS.get(key)
        mdl     = BASELINE_MODELS.get(key)
        pop_avg = BASELINE_POP_AVGS.get(key)

        if config is None or mdl is None:
            return {"error": f"Model for baseline key {key} is not loaded."}, 500

        logger.info(f"Preprocessing data (baseline={baseline}, model key={key})...")

        if IS_REGRESSION:
            # Step 1: independent inference for each measure
            glucose_preds = []
            sigma_preds   = []

            for label, meas in [("measure", measure), ("measure2", measure2), ("measure3", measure3)]:
                if len(meas) == 0:
                    logger.info(f"Skipping {label}: empty array")
                    continue
                try:
                    logger.info(f"CURRENT TIME={time_of_day}")
                    x_i        = utils.preprocess_for_inference(meas, reference, dark, cal_data, config, pop_avg=pop_avg, time_of_day=time_of_day)
                    mu_i, lv_i = utils.regression_inference(mdl, x_i)
                    g_i        = float(mu_i.item())
                    s_i        = float(lv_i.item())
                    glucose_preds.append(g_i)
                    sigma_preds.append(s_i)
                    logger.info(f"Regression prediction ({label}): glucose={g_i:.2f}, sigma={s_i:.4f}")
                except Exception as e:
                    logger.warning(f"Inference with {label} failed: {e}")

            if not glucose_preds:
                return {"error": "All measure arrays are empty or inference failed."}, 500

            predicted_glucose = sum(glucose_preds) / len(glucose_preds)
            sigma_value       = sum(sigma_preds)   / len(sigma_preds)
            logger.info(f"INDIVIDUAL OUTPUTS — " + ", ".join(
                f"{lbl}: glucose={g:.2f}, sigma={s:.4f}" for lbl, g, s in zip(["measure","measure2","measure3"], glucose_preds, sigma_preds)
            ))
            logger.info(f"AVERAGED across {len(glucose_preds)} measure(s): glucose={predicted_glucose:.2f}, sigma={sigma_value:.4f}")

            # Step 2: scale by baseline range
            if 100 <= baseline < 125:
                scale = SCALE_FACTOR_100
                logger.info(f"Applying SCALE_FACTOR_100={scale} (baseline={baseline})")
            elif baseline >= 125:
                scale = SCALE_FACTOR_125
                logger.info(f"Applying SCALE_FACTOR_125={scale} (baseline={baseline})")
            else:
                scale = 1.0
            predicted_glucose *= scale

            # Step 3: 2-minute rolling average with last_glucose_values
            if current_time_str and isinstance(last_glucose_values, list) and last_glucose_values:
                try:
                    current_dt = datetime.fromisoformat(current_time_str)
                    cutoff     = current_dt - timedelta(minutes=AVERAGE_WINDOW_MINUTES)
                    recent_entries = [
                        entry for entry in last_glucose_values
                        if isinstance(entry, dict)
                        and "timestamp" in entry and "glucose" in entry
                        and datetime.fromisoformat(entry["timestamp"]) >= cutoff
                    ]
                    recent = [e["glucose"] for e in recent_entries]
                    logger.info(f"2-min window [{cutoff.isoformat()} → {current_dt.isoformat()}]: "
                                f"{len(recent_entries)} entries — "
                                + (", ".join(f"{e['timestamp']}={e['glucose']}" for e in recent_entries) or "none"))
                    if recent:
                        averaged = (predicted_glucose + sum(recent)) / (1 + len(recent))
                        logger.info(f"2-min average: current={predicted_glucose:.2f}, history={recent} → {averaged:.2f}")
                        predicted_glucose = averaged
                except Exception as e:
                    logger.warning(f"Could not apply 2-min averaging: {e}")

            # Step 4: update cal_high / cal_low using rolling window of last N readings
            cap_high = 2*baseline if baseline < 100 else 2.2 * baseline

            # save incoming cal values so we can invert adjusted history back to raw scale
            prev_cal_low   = cal_low
            prev_cal_high  = cal_high
            prev_cal_range = prev_cal_high - prev_cal_low
            prev_tgt_range = cap_high - baseline

            def _to_raw(adj):
                if prev_cal_range < CAL_RESCALE_MIN_RANGE:
                    return adj  # rescaling was not applied, value is already raw
                return prev_cal_low + (adj - baseline) * prev_cal_range / prev_tgt_range

            window = [predicted_glucose]
            if isinstance(last_glucose_values, list) and last_glucose_values:
                sorted_history = sorted(
                    [e for e in last_glucose_values if isinstance(e, dict) and "glucose" in e],
                    key=lambda e: e.get("timestamp", ""),
                    reverse=True
                )
                for entry in sorted_history[:CAL_ROLLING_WINDOW - 1]:
                    try:
                        window.append(_to_raw(float(entry["glucose"])))
                    except (ValueError, TypeError):
                        pass
            cal_high = max(window)
            cal_low  = min(window)

            if cal_low < baseline:
                cal_low = baseline
            if cal_high > cap_high:
                cal_high = cap_high

            logger.info(f"cal_high={cal_high:.2f}, cal_low={cal_low:.2f} (cap_high={cap_high:.2f}, baseline={baseline})")

            # Step 5: linear rescaling using cal_high / cal_low
            if USE_CALIBRATION and cal_high - cal_low >= CAL_RESCALE_MIN_RANGE:
                cal_range    = cal_high - cal_low
                target_range = cap_high - baseline
                predicted_glucose = baseline + (predicted_glucose - cal_low) / cal_range * target_range

                if abs(predicted_glucose - baseline) < 1e-6:
                    noise = abs(np.random.normal(0, 1.5))
                    predicted_glucose -= noise
                    logger.info(f"Cal floor noise applied: -{noise:.2f}")
                elif abs(predicted_glucose - cap_high) < 1e-6:
                    noise = abs(np.random.normal(0, 1.5))
                    predicted_glucose += noise
                    logger.info(f"Cal ceiling noise applied: +{noise:.2f}")

                logger.info(f"Cal rescale: [{cal_low:.1f}, {cal_high:.1f}] → [{baseline:.1f}, {cap_high:.1f}], result={predicted_glucose:.2f}")

            tod = time_of_day if time_of_day is not None else 0.0
            time_index = min(10, int(tod / 24.0 * 11))

            logger.info(f"Final prediction: glucose={predicted_glucose:.2f}, sigma={sigma_value:.4f}, baseline_key={key}, baseline={baseline}")
            return {"predicted_glucose": round(predicted_glucose, 2), "sigma": round(sigma_value, 4), "acceptance": 25,
                    "cal_high": round(cal_high, 2), "cal_low": round(cal_low, 2),
                    "glycemic_index": time_index, "display_message": time_index}

        return {"error": "Non-regression models are not supported in this configuration."}, 500

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Prediction failed: {str(e)}"}, 500


@app.route('/', methods=['GET'])
def health_check():
    models_loaded = {f"baseline{k}": (BASELINE_MODELS.get(k) is not None) for k in BASELINE_KEYS}
    all_healthy   = all(models_loaded.values())
    status        = "healthy" if all_healthy else "unhealthy"

    diagnostics = {
        "models_loaded":     models_loaded,
        "is_regression":     IS_REGRESSION,
        "working_directory": os.getcwd(),
        "environment": {
            "PORT":             os.environ.get('PORT', 'not_set'),
            "PYTHONUNBUFFERED": os.environ.get('PYTHONUNBUFFERED', 'not_set'),
        },
        "files_in_root": [f for f in os.listdir('.') if f.endswith(('.pt', '.pth', '.pkl', '.py', '.txt', '.yaml'))],
    }

    logger.info(f"Health check: status={status}, models={models_loaded}")

    return jsonify({
        "status":      status,
        "message":     "Glucose prediction API is running",
        "model_type":  "Regression" if IS_REGRESSION else "Classification",
        "diagnostics": diagnostics,
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not BASELINE_MODELS:
            return jsonify({"error": "Models not loaded. Please restart the service."}), 503

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        result = predict_from_json(data)
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
