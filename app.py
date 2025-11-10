import os
import json
import numpy as np
from flask import Flask, request, jsonify
import warnings
import logging
import sys
import traceback
import utils

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
        
        # Проверяем что файл модели существует
        model_path = utils.NORMAL_TRACED_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # Загружаем модель через utils
        NORMAL_model = utils.load_model(model_path)
        logger.info(f":white_check_mark: PyTorch model loaded successfully from {model_path}")
        # Проверяем что файл модели существует
        model_path = utils.PREDIABETIC_TRACED_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # Загружаем модель через utils
        PREDIABETIC_model = utils.load_model(model_path)
        logger.info(f":white_check_mark: PyTorch model loaded successfully from {model_path}")
        # Проверяем что файл модели существует
        model_path = utils.DIABETIC_TRACED_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # Загружаем модель через utils
        DIABETIC_model = utils.load_model(model_path)
        logger.info(f":white_check_mark: PyTorch model loaded successfully from {model_path}")
        
        model_path = utils.CLASSIFICATION_TRACED_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # Загружаем модель через utils
        CLASSIFICATION_model = utils.load_model(model_path)
        logger.info(f":white_check_mark: PyTorch model loaded successfully from {model_path}")

        return NORMAL_model, PREDIABETIC_model, DIABETIC_model, CLASSIFICATION_model
        
    except Exception as e:
        logger.error(f":x: Failed to load PyTorch model: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to load PyTorch model: {e}")

# Инициализация модели
logger.info("=== Starting PyTorch model initialization ===")
try:
    NORMAL_model, PREDIABETIC_model, DIABETIC_model, CLASSIFICATION_model = load_model_with_fallback()
    logger.info(f":white_check_mark: Model initialization completed")
    logger.info(f"Model type: {type(NORMAL_model)}")
except Exception as e:
    logger.error(f":x: Fatal error during model initialization: {e}")
    logger.info("Setting model to None")
    NORMAL_model = None
    PREDIABETIC_model = None
    DIABETIC_model = None
    CLASSIFICATION_model = None
logger.info("=== Model initialization complete ===")

# Добавляем дополнительное логирование для Railway
logger.info("=== Environment Information ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PORT environment variable: {os.environ.get('PORT', 'not set')}")
logger.info(f"Model file exists: {os.path.exists(utils.NORMAL_TRACED_MODEL_PATH)}")
logger.info("=== Environment Information Complete ===")

def predict_from_json(data):
    """Предсказание с использованием PyTorch модели"""
    if NORMAL_model is None or PREDIABETIC_model is None or DIABETIC_model is None or CLASSIFICATION_model is None:
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
        x, x_original = utils.preprocess_data(measure, reference, dark, cal_data, baseline)
        
        # Инференс модели
        logger.info("Running model inference...")
        # prediction = utils.model_inference(model, x)
        
        # # Масштабирование предсказания
        # logger.info("Rescaling prediction...")
        # prediction_rescaled = utils.rescale_prediction(prediction)
        if baseline < 100:
            prediction_rescaled, sigma = utils.model_inference(NORMAL_model, x)
        elif baseline >= 100 and baseline < 125:
            prediction_rescaled, sigma = utils.model_inference(PREDIABETIC_model, x)
        else:
            prediction_rescaled, sigma = utils.model_inference(DIABETIC_model, x)
        
        # Конвертируем tensor в число
        if hasattr(prediction_rescaled, 'item'):
            prediction_value = prediction_rescaled.item()
        else:
            prediction_value = float(prediction_rescaled)
        if hasattr(sigma, 'item'):
            sigma_value = sigma.item()
        else:
            sigma_value = float(sigma)
        
        logger.info(f"Regression prediction: {prediction_value}")
        if baseline < 100:
            acceptance_value = 0.003*prediction_value+4.540
        elif baseline >= 100 and baseline < 125:
            acceptance_value = 0.013*prediction_value+2.899
        else:
            acceptance_value = 0.018*prediction_value+1.981

        prediction_rescaled = utils.model_inference(CLASSIFICATION_model, x_original)
        print(prediction_rescaled)
        prediction_array = np.array(prediction_rescaled, dtype=float)
        print(np.shape(prediction_array))
        print(prediction_array)

        # If it's a vector of logits or probabilities:
        if prediction_array.ndim > 0 and prediction_array.size > 1:
            predicted_class = int(np.argmax(prediction_array))
            prediction_value = float(65 + predicted_class*10)
        print(predicted_class)
        print(prediction_value)
        sigma_value = 0
        acceptance_value = 1
        logger.info(f"Classificaiton prediction: {prediction_value}")

        return {"predicted_glucose": round(prediction_value, 2), "sigma": round(sigma_value, 2), "acceptance": round(acceptance_value, 2)}
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Prediction failed: {str(e)}"}, 500

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = "healthy" if NORMAL_model is not None else "unhealthy"
    
    # Диагностическая информация
    diagnostics = {
        "model_file_exists": os.path.exists(utils.NORMAL_TRACED_MODEL_PATH),
        "model_file_path": utils.NORMAL_TRACED_MODEL_PATH,
        "model_loaded": NORMAL_model is not None,
        "working_directory": os.getcwd(),
        "environment": {
            "PORT": os.environ.get('PORT', 'not_set'),
            "PYTHONUNBUFFERED": os.environ.get('PYTHONUNBUFFERED', 'not_set'),
        },
        "files_in_root": [f for f in os.listdir('.') if f.endswith(('.pt', '.pkl', '.py'))],
    }
    
    # Проверяем размер файла модели
    if os.path.exists(utils.NORMAL_TRACED_MODEL_PATH):
        diagnostics["model_file_size"] = os.path.getsize(utils.NORMAL_TRACED_MODEL_PATH)
    
    # Логируем результат health check
    logger.info(f"Health check: status={status}, model_loaded={NORMAL_model is not None}")
    
    response = {
        "status": status, 
        "message": "Glucose prediction API is running (PyTorch version)",
        "model_loaded": NORMAL_model is not None,
        "model_type": "PyTorch CNN",
        "diagnostics": diagnostics
    }
    
    # Если что-то не так, добавляем детали
    if status == "unhealthy":
        error_details = []
        if not os.path.exists(utils.NORMAL_TRACED_MODEL_PATH):
            error_details.append(f"Model file not found: {utils.NORMAL_TRACED_MODEL_PATH}")
        if not NORMAL_model:
            error_details.append("Model failed to load - check model file and PyTorch installation")
        response["error_details"] = error_details
    
    return jsonify(response)

@app.route('/debug-model', methods=['GET'])
def debug_model():
    """Диагностика PyTorch модели"""
    try:
        model_path = utils.NORMAL_TRACED_MODEL_PATH
        
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
        if NORMAL_model is None or PREDIABETIC_model is None or DIABETIC_model is None:
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
