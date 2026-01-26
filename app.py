import os
import json
import numpy as np
from flask import Flask, request, jsonify
import warnings
import logging
import sys
import traceback
import utils

USE_ONE_MODEL = True
USE_ABSORPTION = True
WAVELENGTH_RANGE = [500, 650]
NORMALIZE_PER_CHANNEL = True

warnings.filterwarnings('ignore')

# Настройка логирования для Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Настройка логирования Flask для Railway
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

def load_model_with_fallback():
    """Загружает PyTorch модель через utils.py"""
    try:
        logger.info("Loading PyTorch model from utils...")
        
        model_path = utils.ALL_CLASSIFICATION_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # Загружаем модель через utils
        ALL_CLASSIFICATION_model = utils.load_model(model_path)
        logger.info(f":white_check_mark: PyTorch model loaded successfully from {model_path}")

        model_path = utils.NORMAL_CLASSIFICATION_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # Загружаем модель через utils
        NORMAL_CLASSIFICATION_model = utils.load_model(model_path)
        logger.info(f":white_check_mark: PyTorch model loaded successfully from {model_path}")

        model_path = utils.DIABETIC_CLASSIFICATION_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # Загружаем модель через utils
        DIABETIC_CLASSIFICATION_model = utils.load_model(model_path)
        logger.info(f":white_check_mark: PyTorch model loaded successfully from {model_path}")

        return NORMAL_CLASSIFICATION_model, DIABETIC_CLASSIFICATION_model, ALL_CLASSIFICATION_model
        
    except Exception as e:
        logger.error(f":x: Failed to load PyTorch model: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to load PyTorch model: {e}")

# Инициализация модели
logger.info("=== Starting PyTorch model initialization ===")
try:
    NORMAL_CLASSIFICATION_model, DIABETIC_CLASSIFICATION_model, ALL_CLASSIFICATION_model = load_model_with_fallback()
    logger.info(f":white_check_mark: Model initialization completed")
    logger.info(f"Model type: {type(ALL_CLASSIFICATION_model)}")
except Exception as e:
    logger.error(f":x: Fatal error during model initialization: {e}")
    logger.info("Setting model to None")
    # NORMAL_model = None
    # DIABETIC_model = None
    NORMAL_CLASSIFICATION_model = None
    DIABETIC_CLASSIFICATION_model = None
    ALL_CLASSIFICATION_model = None
logger.info("=== Model initialization complete ===")

# Добавляем дополнительное логирование для Railway
logger.info("=== Environment Information ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PORT environment variable: {os.environ.get('PORT', 'not set')}")
logger.info(f"Model file exists: {os.path.exists(utils.ALL_CLASSIFICATION_MODEL_PATH)}")
logger.info("=== Environment Information Complete ===")

def predict_from_json(data):
    if NORMAL_CLASSIFICATION_model is None or DIABETIC_CLASSIFICATION_model is None or ALL_CLASSIFICATION_model is None:
        return {"error": "Model not loaded. Please restart the service."}, 500
    
    try:
        # Если данные в списке, берем первый элемент
        if isinstance(data, list):
            data = data[0]
        
        # Извлекаем массивы данных
        measure = np.array(data.get("measure", []))
        reference = np.array(data.get("reference", []))
        dark = np.array(data.get("dark", []))
        cal_data = np.array(data.get("cal_data", []))

        baseline = data.get("baseline", 90)
        if baseline in [None, "None", "null", ""]:
            baseline = 90
        else:
            try:
                baseline = float(baseline)
            except ValueError:
                baseline = 90

        diabetes = data.get("diabetes", 0)
        if diabetes in [None, "None", "null", ""]:
            diabetes = 0
        else:
            try:
                diabetes = int(diabetes)
            except ValueError:
                diabetes = 0

        # Корректировка baseline для диабетиков
        if diabetes == 1 and baseline < 125:
            baseline = 125

        # Проверяем что все массивы не пустые
        if len(measure) == 0 or len(reference) == 0 or len(dark) == 0 or len(cal_data) == 0:
            return {"error": "All data arrays (measure, reference, dark, cal_data) must be non-empty"}, 400
        
        # Препроцессинг данных
        logger.info("Preprocessing data...")
        x_original, x_normalized = utils.preprocess_data(measure, reference, dark, cal_data, baseline, USE_ABSORPTION, WAVELENGTH_RANGE)
        
        # After preprocessing but before model
        print("External data statistics:")
        print(f"  Shape: {x_original.shape}")
        print(f"  Mean: {x_original.mean():.6f}, Std: {x_original.std():.6f}")
        print(f"  Min: {x_original.min():.6f}, Max: {x_original.max():.6f}")
        print(f"  Channel 0 mean: {x_original[:, 0, :].mean():.6f}")
        print(f"  Channel 6 (absorption) mean: {x_original[:, 6, :].mean():.6f}")

        # Инференс модели
        logger.info("Running model inference...")
       
        if USE_ONE_MODEL:
            start_value = 65
            num_classes = 15

            if not NORMALIZE_PER_CHANNEL:
                prediction_rescaled = utils.model_inference(ALL_CLASSIFICATION_model, x_original)
            else:
                prediction_rescaled = utils.model_inference(ALL_CLASSIFICATION_model, x_normalized)
            prediction_array = np.array(prediction_rescaled, dtype=float)
            prediction_array = prediction_array.flatten()

            # # Mask out classes below baseline
            # Class 0 = 65, Class 1 = 75, etc. So baseline 90 means we start from class 3 (95)
            # min_class = max(0, int(np.ceil((baseline - 65) / 10)))
            min_class = max(0, int(np.floor((baseline - 65) / 10)))
            logger.info(f"min class: {min_class}")
            # Create a masked array: set logits below min_class to -inf

        else:
            if baseline < 125:
                if not NORMALIZE_PER_CHANNEL:
                    prediction_rescaled = utils.model_inference(NORMAL_CLASSIFICATION_model, x_original)
                else:
                    prediction_rescaled = utils.model_inference(NORMAL_CLASSIFICATION_model, x_normalized)
            else:
                if not NORMALIZE_PER_CHANNEL:
                    prediction_rescaled = utils.model_inference(DIABETIC_CLASSIFICATION_model, x_original)
                else:
                    prediction_rescaled = utils.model_inference(DIABETIC_CLASSIFICATION_model, x_normalized)
            prediction_array = np.array(prediction_rescaled, dtype=float)
            prediction_array = prediction_array.flatten()
            if baseline < 125:
                num_classes = 15  # between 65 - 205
                start_value = 65
            else:
                num_classes = 8 # between 125 - 205
                start_value = 125

            min_class = max(0, int(np.floor((baseline - start_value) / 10)))
            logger.info(f"min class: {min_class}")              

        masked_logits = prediction_array.copy()
        print(masked_logits)
        masked_logits[:min_class] = -np.inf
        print(masked_logits)
        # Softmax will give 0 probability to -inf logits
        logits_shifted = masked_logits - np.max(masked_logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)
        
        # Compute expectation over feasible classes only
        class_indices = np.arange(num_classes, dtype=float)
        # Compute expected class index (weighted average)
        predicted_class = np.sum(probs * class_indices)

        prediction_value = float(start_value + predicted_class*10)
            
        print(predicted_class)
        print(prediction_value)
        sigma_value = 0
        acceptance_value = 1
        logger.info(f"Classification prediction: {prediction_value}")

        return {"predicted_glucose": round(prediction_value, 2), "sigma": round(sigma_value, 2), "acceptance": round(acceptance_value, 2)}
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Prediction failed: {str(e)}"}, 500

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = "healthy" if ALL_CLASSIFICATION_model is not None else "unhealthy"
    
    # Диагностическая информация
    diagnostics = {
        "model_file_exists": os.path.exists(utils.ALL_CLASSIFICATION_MODEL_PATH),
        "model_file_path": utils.ALL_CLASSIFICATION_MODEL_PATH,
        "model_loaded": ALL_CLASSIFICATION_model is not None,
        "working_directory": os.getcwd(),
        "environment": {
            "PORT": os.environ.get('PORT', 'not_set'),
            "PYTHONUNBUFFERED": os.environ.get('PYTHONUNBUFFERED', 'not_set'),
        },
        "files_in_root": [f for f in os.listdir('.') if f.endswith(('.pt', '.pkl', '.py'))],
    }
    
    # Проверяем размер файла модели
    if os.path.exists(utils.ALL_CLASSIFICATION_MODEL_PATH):
        diagnostics["model_file_size"] = os.path.getsize(utils.ALL_CLASSIFICATION_MODEL_PATH)
    
    # Логируем результат health check
    logger.info(f"Health check: status={status}, model_loaded={ALL_CLASSIFICATION_model is not None}")
    
    response = {
        "status": status, 
        "message": "Glucose prediction API is running (PyTorch version)",
        "model_loaded": ALL_CLASSIFICATION_model is not None,
        "model_type": "PyTorch CNN",
        "diagnostics": diagnostics
    }
    
    # Если что-то не так, добавляем детали
    if status == "unhealthy":
        error_details = []
        if not os.path.exists(utils.ALL_CLASSIFICATION_MODEL_PATH):
            error_details.append(f"Model file not found: {utils.ALL_CLASSIFICATION_MODEL_PATH}")
        if not ALL_CLASSIFICATION_model:
            error_details.append("Model failed to load - check model file and PyTorch installation")
        response["error_details"] = error_details
    
    return jsonify(response)

@app.route('/debug-model', methods=['GET'])
def debug_model():
    """Диагностика PyTorch модели"""
    try:
        model_path = utils.ALL_CLASSIFICATION_MODEL_PATH
        
        result = {
            "file_exists": os.path.exists(model_path),
            "file_path": model_path,
            "file_size": 0,
            "file_readable": False,
            "torch_loadable": False,
            "model_structure": None,
            "error": None
        }
        
        if os.path.exists(model_path):
            # Проверяем размер файла
            result["file_size"] = os.path.getsize(model_path)
            
            # Проверяем читаемость
            try:
                with open(model_path, 'rb') as f:
                    header = f.read(20)
                    result["file_readable"] = True
                    result["file_header_hex"] = header.hex()
            except Exception as e:
                result["read_error"] = str(e)
            
            # Пробуем загрузить через PyTorch
            try:
                logger.info("Debug: Attempting to load model with torch.jit.load...")
                import torch
                test_model = torch.jit.load(model_path)
                result["torch_loadable"] = True
                result["model_type"] = str(type(test_model))
                
                # Получаем информацию о модели
                result["model_structure"] = {
                    "is_jit_module": True,
                    "device": str(next(test_model.parameters()).device),
                    "training_mode": test_model.training
                }
                
                logger.info("Debug: Model loaded successfully in debug mode")
                
            except Exception as e:
                result["torch_error"] = str(e)
                logger.error(f"Debug: Failed to load model: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Debug model failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if ALL_CLASSIFICATION_model is None or NORMAL_CLASSIFICATION_model is None or DIABETIC_CLASSIFICATION_model is None:
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
