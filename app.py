import os
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import warnings
import logging
import sys
import traceback
import utils

SCALE_FACTOR_100 = 1.1
SCALE_FACTOR_125 = 1.2

CAL_DEFAULT_HIGH      = 150.0
CAL_DEFAULT_LOW       = 120.0
CAL_RESCALE_MIN_RANGE = 10.0
CAL_ROLLING_WINDOW    = 10
USE_CALIBRATION       = False

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


logger.info("=== Starting model initialization ===")
try:
    CONFIG  = utils.load_config('baseline-1_configuration.yaml')
    MODEL   = utils.load_model('baseline-1_regression_model.pt', CONFIG)
    logger.info("Model loaded OK")
except Exception as e:
    logger.error(f"Failed to load model — {e}")
    logger.error(traceback.format_exc())
    CONFIG = None
    MODEL  = None

POP_AVG = None
for avg_path in ['baseline-1_average.npy', 'baseline-1_population_average.npy']:
    if os.path.exists(avg_path):
        try:
            POP_AVG = np.load(avg_path)
            logger.info(f"Population average loaded from {avg_path}  shape={POP_AVG.shape}")
        except Exception as e:
            logger.warning(f"Could not load {avg_path} — {e}")
        break

logger.info("=== Model initialization complete ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PORT: {os.environ.get('PORT', 'not set')}")


def compute_display_message(last_glucose_values, glycemic_index, cal_low, cal_high, current_time_str):
    cal_range = cal_high - cal_low

    points = []
    for e in last_glucose_values:
        try:
            ts = datetime.fromisoformat(e["timestamp"])
            g  = float(e["glucose"])
            gi = max(0.0, min(10.0, (g - cal_low) / cal_range * 10.0)) if cal_range > 0 else 5.0
            points.append((ts, gi))
        except (KeyError, ValueError, TypeError):
            pass

    try:
        current_dt = datetime.fromisoformat(current_time_str)
    except Exception:
        return 0

    points.append((current_dt, float(glycemic_index)))
    points.sort(key=lambda x: x[0])

    if len(points) < 2:
        return 0

    now = points[-1][0]

    def contiguous_hours(cond):
        if not cond(points[-1][1]):
            return 0.0
        i = len(points) - 2
        while i >= 0 and cond(points[i][1]):
            i -= 1
        return (points[-1][0] - points[i + 1][0]).total_seconds() / 3600.0

    def had_then_low(window_h, high_cond, low_thresh):
        cutoff = now - timedelta(hours=window_h)
        window = [(ts, gi) for ts, gi in points if ts >= cutoff]
        if len(window) < 2:
            return False
        return any(high_cond(gi) for _, gi in window) and window[-1][1] < low_thresh

    # rule 2 before rule 1 (more specific)
    if contiguous_hours(lambda gi: gi > 7) > 4:
        return 2
    if contiguous_hours(lambda gi: gi > 7) > 2:
        return 1
    if had_then_low(1, lambda gi: gi > 7, 3):
        return 4
    if current_dt.hour >= 22 and glycemic_index > 5:
        return 3
    if had_then_low(2, lambda gi: 4 < gi < 7, 3):
        return 6
    if contiguous_hours(lambda gi: 4 < gi < 7) > 2:
        return 5
    if contiguous_hours(lambda gi: gi < 2) > 3:
        return 7
    return 0


def predict_from_json(data):
    if MODEL is None:
        return {"error": "Model not loaded. Please restart the service."}, 500

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
        measure2  = np.array(data.get("measure2",  []))
        measure3  = np.array(data.get("measure3",  []))
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
        if cal_high == 0:
            cal_high = CAL_DEFAULT_HIGH

        try:
            cal_low = float(data.get("cal_low") or 0)
        except (ValueError, TypeError):
            cal_low = 0.0
        if cal_low == 0:
            cal_low = CAL_DEFAULT_LOW

        logger.info(f"cal_high: {data.get('cal_high')!r}  →  {cal_high}")
        logger.info(f"cal_low:  {data.get('cal_low')!r}  →  {cal_low}")

        time_of_day = None
        if current_time_str:
            try:
                _cdt = datetime.fromisoformat(current_time_str)
                time_of_day = _cdt.hour + _cdt.minute / 60.0 + _cdt.second / 3600.0
            except Exception:
                pass

        logger.info(f"baseline={baseline}, measure_len={len(measure)}, measure2_len={len(measure2)}, "
                    f"measure3_len={len(measure3)}, dark_len={len(dark)}, reference_len={len(reference)}, "
                    f"cal_data_len={len(cal_data)}, current_time={current_time_str}")

        for label, other in [("measure2", measure2), ("measure3", measure3)]:
            if len(other) == len(measure) and len(measure) > 0:
                denom = np.where(np.abs(measure) > 1e-9, np.abs(measure), 1e-9)
                pct_diff = np.abs(other - measure) / denom * 100
                max_idx = int(np.argmax(pct_diff))
                logger.info(f"measure vs {label}: mean_pct_diff={pct_diff.mean():.2f}%, "
                            f"max_pct_diff={pct_diff[max_idx]:.2f}% at idx={max_idx}")

        if len(measure) == 0 or len(reference) == 0 or len(dark) == 0 or len(cal_data) == 0:
            return {"error": "All data arrays (measure, reference, dark, cal_data) must be non-empty"}, 400

        # Step 1: independent inference for each measure
        glucose_preds = []
        sigma_preds   = []

        for label, meas in [("measure", measure), ("measure2", measure2), ("measure3", measure3)]:
            if len(meas) == 0:
                logger.info(f"Skipping {label}: empty array")
                continue
            try:
                logger.info(f"CURRENT TIME={time_of_day}")
                x_i        = utils.preprocess_for_inference(meas, reference, dark, cal_data, CONFIG, pop_avg=POP_AVG, time_of_day=time_of_day)
                mu_i, lv_i = utils.regression_inference(MODEL, x_i)
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
            f"{lbl}: glucose={g:.2f}, sigma={s:.4f}"
            for lbl, g, s in zip(["measure", "measure2", "measure3"], glucose_preds, sigma_preds)
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

        # Step 3: update cal_high / cal_low from history (including current prediction)
        history_vals = [predicted_glucose]
        if isinstance(last_glucose_values, list):
            for e in last_glucose_values:
                try:
                    history_vals.append(float(e["glucose"]))
                except (TypeError, KeyError, ValueError):
                    pass

        if len(history_vals) > 5:
            cal_high = max(history_vals)
            cal_low  = min(history_vals)

        logger.info(f"cal_high={cal_high:.2f}, cal_low={cal_low:.2f} (history_len={len(history_vals)})")

        # Step 4: glycemic_index — predicted_glucose scaled 0–10 between cal_low and cal_high
        cal_range = cal_high - cal_low
        if cal_range > 0:
            glycemic_index = (predicted_glucose - cal_low) / cal_range * 10.0
        else:
            glycemic_index = 5.0
        glycemic_index = int(round(max(0.0, min(10.0, glycemic_index))))

        display_message = compute_display_message(
            last_glucose_values, glycemic_index, cal_low, cal_high, current_time_str or ""
        )

        logger.info(f"Final prediction: glucose={predicted_glucose:.2f}, sigma={sigma_value:.4f}, "
                    f"glycemic_index={glycemic_index}, display_message={display_message}, baseline={baseline}")
        return {"predicted_glucose": round(predicted_glucose, 2), "sigma": round(sigma_value, 4), "acceptance": 25,
                "cal_high": round(cal_high, 2), "cal_low": round(cal_low, 2),
                "glycemic_index": glycemic_index, "display_message": display_message}

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Prediction failed: {str(e)}"}, 500


@app.route('/', methods=['GET'])
def health_check():
    status = "healthy" if MODEL is not None else "unhealthy"
    logger.info(f"Health check: status={status}")
    return jsonify({
        "status":      status,
        "message":     "Glucose prediction API is running",
        "model_type":  "Regression",
        "diagnostics": {
            "model_loaded":      MODEL is not None,
            "working_directory": os.getcwd(),
            "environment": {
                "PORT":             os.environ.get('PORT', 'not_set'),
                "PYTHONUNBUFFERED": os.environ.get('PYTHONUNBUFFERED', 'not_set'),
            },
            "files_in_root": [f for f in os.listdir('.') if f.endswith(('.pt', '.pth', '.pkl', '.py', '.txt', '.yaml'))],
        },
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if MODEL is None:
            return jsonify({"error": "Model not loaded. Please restart the service."}), 503

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
