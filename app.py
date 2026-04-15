import os
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import warnings
import logging
import sys
import traceback
import utils

AVERAGE_LAST_5MIN = False
AVERAGE_WINDOW_MINUTES = 5

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
BASELINE_KEYS = [-1, 100, 120]


logger.info("=== Starting model initialization ===")
BASELINE_CONFIGS   = {}
BASELINE_MODELS    = {}
BASELINE_POP_AVGS  = {}

for b in BASELINE_KEYS:
    try:
        cfg = utils.load_config(f'baseline{b}_configuration.yaml')
        BASELINE_CONFIGS[b] = cfg
        BASELINE_MODELS[b]  = utils.load_model(f'baseline{b}_regression_model.pt', cfg)
        logger.info(f"baseline{b}: model loaded OK")
    except Exception as e:
        logger.error(f"baseline{b}: failed to load — {e}")
        logger.error(traceback.format_exc())

    avg_path = f'baseline{b}_average.npy'
    if os.path.exists(avg_path):
        try:
            BASELINE_POP_AVGS[b] = np.load(avg_path)
            logger.info(f"baseline{b}: population average loaded  shape={BASELINE_POP_AVGS[b].shape}")
        except Exception as e:
            logger.warning(f"baseline{b}: could not load {avg_path} — {e}")

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

        key    = utils.select_baseline_key(baseline)
        config = BASELINE_CONFIGS.get(key)
        mdl    = BASELINE_MODELS.get(key)

        if config is None or mdl is None:
            return {"error": f"Model for baseline key {key} is not loaded."}, 500

        logger.info(f"Preprocessing data (baseline={baseline} → model key={key})...")
        x = utils.preprocess_for_inference(measure, reference, dark, cal_data, config, pop_avg=BASELINE_POP_AVGS.get(key))

        logger.info("Running model inference...")
        if IS_REGRESSION:
            mu, log_var = utils.regression_inference(mdl, x)
            predicted_glucose = float(mu.item())
            sigma_value       = float(log_var.item())

            if AVERAGE_LAST_5MIN and current_time_str and isinstance(last_glucose_values, list) and last_glucose_values:
                try:
                    current_dt = datetime.fromisoformat(current_time_str)
                    cutoff     = current_dt - timedelta(minutes=AVERAGE_WINDOW_MINUTES)
                    recent = [
                        entry["glucose"]
                        for entry in last_glucose_values
                        if isinstance(entry, dict)
                        and "timestamp" in entry
                        and "glucose" in entry
                        and datetime.fromisoformat(entry["timestamp"]) >= cutoff
                    ]
                    if recent:
                        averaged = (predicted_glucose + sum(recent)) / (1 + len(recent))
                        logger.info(f"Averaged with {len(recent)} recent values {recent} → {averaged:.2f}")
                        predicted_glucose = averaged
                except Exception as e:
                    logger.warning(f"Could not apply 5-min averaging: {e}")

            logger.info(f"Regression prediction: glucose={predicted_glucose:.2f}, sigma={sigma_value:.4f}, baseline_key={key}")

            for label, alt_measure in [("measure2", measure2), ("measure3", measure3)]:
                if len(alt_measure) > 0:
                    try:
                        x_alt = utils.preprocess_for_inference(alt_measure, reference, dark, cal_data, config, pop_avg=BASELINE_POP_AVGS.get(key))
                        mu_alt, lv_alt = utils.regression_inference(mdl, x_alt)
                        logger.info(f"Regression prediction ({label}): glucose={float(mu_alt.item()):.2f}, sigma={float(lv_alt.item()):.4f}")
                    except Exception as e:
                        logger.warning(f"Inference with {label} failed: {e}")

            return {"predicted_glucose": round(predicted_glucose, 2), "sigma": round(sigma_value, 4), "acceptance": 25}

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
