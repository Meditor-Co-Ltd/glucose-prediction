import os
import numpy as np
import torch
from flask import Flask, request, jsonify
import warnings
import logging
import sys
import traceback
import utils

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

# --- Config-driven setup ---
CONFIG = utils.load_config('configuration.txt')
IS_REGRESSION = 'regression' in str(CONFIG.get('NUM_CLASSES', '')).lower()


def load_model_with_fallback():
    """Load the single model specified in configuration.txt"""
    try:
        model_file = CONFIG.get('MODEL_FILE')
        if not model_file:
            raise ValueError("MODEL_FILE not set in configuration.txt")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        loaded_model = utils.load_model(model_file)
        logger.info(f"Model loaded successfully from {model_file}")
        return loaded_model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


# Model initialization
logger.info("=== Starting model initialization ===")
try:
    model = load_model_with_fallback()
    logger.info(f"Model initialization completed")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"IS_REGRESSION: {IS_REGRESSION}")
    logger.info(f"Config: USE_ABSORPTION={CONFIG.get('USE_ABSORPTION')}, "
                f"wavelength={CONFIG.get('wavelength_nm_start')}-{CONFIG.get('wavelength_nm_end')}nm, "
                f"ADD_SPECTRAL_DERIVATIVES={CONFIG.get('ADD_SPECTRAL_DERIVATIVES')}, "
                f"PER_CHANNEL_NORM={CONFIG.get('PER_CHANNEL_NORM')}")
except Exception as e:
    logger.error(f"Fatal error during model initialization: {e}")
    model = None
logger.info("=== Model initialization complete ===")

logger.info("=== Environment Information ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PORT environment variable: {os.environ.get('PORT', 'not set')}")
logger.info(f"Model file: {CONFIG.get('MODEL_FILE', 'not set')}")
logger.info("=== Environment Information Complete ===")


def predict_from_json(data):
    if model is None:
        return {"error": "Model not loaded. Please restart the service."}, 500

    try:
        if isinstance(data, list):
            data = data[0]

        measure  = np.array(data.get("measure", []))
        reference = np.array(data.get("reference", []))
        dark     = np.array(data.get("dark", []))
        cal_data = np.array(data.get("cal_data", []))

        if len(measure) == 0 or len(reference) == 0 or len(dark) == 0 or len(cal_data) == 0:
            return {"error": "All data arrays (measure, reference, dark, cal_data) must be non-empty"}, 400

        logger.info("Preprocessing data...")
        x = utils.preprocess_for_inference(measure, reference, dark, cal_data, CONFIG)

        logger.info("Running model inference...")
        if IS_REGRESSION:
            mu, log_var = utils.regression_inference(model, x)
            predicted_glucose = float(mu.item())
            sigma_value = float(log_var.item())
            logger.info(f"Regression prediction: glucose={predicted_glucose:.2f}, sigma={sigma_value:.4f}")
            return {"predicted_glucose": round(predicted_glucose, 2), "sigma": round(sigma_value, 4)}

        return {"error": "Non-regression models are not supported in this configuration."}, 500

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Prediction failed: {str(e)}"}, 500


@app.route('/', methods=['GET'])
def health_check():
    model_file = CONFIG.get('MODEL_FILE', 'unknown')
    status = "healthy" if model is not None else "unhealthy"

    diagnostics = {
        "model_file": model_file,
        "model_file_exists": os.path.exists(model_file) if model_file else False,
        "model_loaded": model is not None,
        "is_regression": IS_REGRESSION,
        "working_directory": os.getcwd(),
        "environment": {
            "PORT": os.environ.get('PORT', 'not_set'),
            "PYTHONUNBUFFERED": os.environ.get('PYTHONUNBUFFERED', 'not_set'),
        },
        "files_in_root": [f for f in os.listdir('.') if f.endswith(('.pt', '.pkl', '.py', '.txt'))],
    }
    if model_file and os.path.exists(model_file):
        diagnostics["model_file_size"] = os.path.getsize(model_file)

    logger.info(f"Health check: status={status}, model_loaded={model is not None}")

    response = {
        "status": status,
        "message": "Glucose prediction API is running",
        "model_loaded": model is not None,
        "model_type": "Regression" if IS_REGRESSION else "Classification",
        "diagnostics": diagnostics,
    }
    if status == "unhealthy":
        error_details = []
        if model_file and not os.path.exists(model_file):
            error_details.append(f"Model file not found: {model_file}")
        if not model:
            error_details.append("Model failed to load")
        response["error_details"] = error_details

    return jsonify(response)


@app.route('/debug-model', methods=['GET'])
def debug_model():
    try:
        model_file = CONFIG.get('MODEL_FILE', '')
        result = {
            "file_exists": os.path.exists(model_file),
            "file_path": model_file,
            "file_size": 0,
            "file_readable": False,
            "torch_loadable": False,
            "model_structure": None,
            "error": None,
        }
        if os.path.exists(model_file):
            result["file_size"] = os.path.getsize(model_file)
            try:
                with open(model_file, 'rb') as f:
                    header = f.read(20)
                    result["file_readable"] = True
                    result["file_header_hex"] = header.hex()
            except Exception as e:
                result["read_error"] = str(e)
            try:
                test_model = torch.jit.load(model_file)
                result["torch_loadable"] = True
                result["model_type"] = str(type(test_model))
                result["model_structure"] = {
                    "is_jit_module": True,
                    "training_mode": test_model.training,
                }
            except Exception as e:
                result["torch_error"] = str(e)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Debug model failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
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
