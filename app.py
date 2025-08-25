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
        model_path = utils.TRACED_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Загружаем модель через utils
        model = utils.load_model()
        logger.info(f":white_check_mark: PyTorch model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f":x: Failed to load PyTorch model: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to load PyTorch model: {e}")

# Инициализация модели
logger.info("=== Starting PyTorch model initialization ===")
try:
    model = load_model_with_fallback()
    logger.info(f":white_check_mark: Model initialization completed")
    logger.info(f"Model type: {type(model)}")
except Exception as e:
    logger.error(f":x: Fatal error during model initialization: {e}")
    logger.info("Setting model to None")
    model = None
logger.info("=== Model initialization complete ===")

# Добавляем дополнительное логирование для Railway
logger.info("=== Environment Information ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PORT environment variable: {os.environ.get('PORT', 'not set')}")
logger.info(f"Model file exists: {os.path.exists(utils.TRACED_MODEL_PATH)}")
logger.info("=== Environment Information Complete ===")

def predict_from_json(data):
    """Предсказание с использованием PyTorch модели"""
    if model is None:
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
        
        # Проверяем что все массивы не пустые
        if len(measure) == 0 or len(reference) == 0 or len(dark) == 0 or len(cal_data) == 0:
            return {"error": "All data arrays (measure, reference, dark, cal_data) must be non-empty"}, 400
        
        # Препроцессинг данных
        logger.info("Preprocessing data...")
        x = utils.preprocess_data(measure, reference, dark, cal_data)
        
        # Инференс модели
        logger.info("Running model inference...")
        # prediction = utils.model_inference(model, x)
        
        # # Масштабирование предсказания
        # logger.info("Rescaling prediction...")
        # prediction_rescaled = utils.rescale_prediction(prediction)
        prediction_rescaled, _ = utils.model_inference(model, x)
        
        # Конвертируем tensor в число
        if hasattr(prediction_rescaled, 'item'):
            prediction_value = prediction_rescaled.item()
        else:
            prediction_value = float(prediction_rescaled)
        
        logger.info(f"Prediction completed: {prediction_value}")
        
        return {"predicted_glucose": round(prediction_value, 2)}
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Prediction failed: {str(e)}"}, 500

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = "healthy" if model is not None else "unhealthy"
    
    # Диагностическая информация
    diagnostics = {
        "model_file_exists": os.path.exists(utils.TRACED_MODEL_PATH),
        "model_file_path": utils.TRACED_MODEL_PATH,
        "model_loaded": model is not None,
        "working_directory": os.getcwd(),
        "environment": {
            "PORT": os.environ.get('PORT', 'not_set'),
            "PYTHONUNBUFFERED": os.environ.get('PYTHONUNBUFFERED', 'not_set'),
        },
        "files_in_root": [f for f in os.listdir('.') if f.endswith(('.pt', '.pkl', '.py'))],
    }
    
    # Проверяем размер файла модели
    if os.path.exists(utils.TRACED_MODEL_PATH):
        diagnostics["model_file_size"] = os.path.getsize(utils.TRACED_MODEL_PATH)
    
    # Логируем результат health check
    logger.info(f"Health check: status={status}, model_loaded={model is not None}")
    
    response = {
        "status": status, 
        "message": "Glucose prediction API is running (PyTorch version)",
        "model_loaded": model is not None,
        "model_type": "PyTorch CNN",
        "diagnostics": diagnostics
    }
    
    # Если что-то не так, добавляем детали
    if status == "unhealthy":
        error_details = []
        if not os.path.exists(utils.TRACED_MODEL_PATH):
            error_details.append(f"Model file not found: {utils.TRACED_MODEL_PATH}")
        if not model:
            error_details.append("Model failed to load - check model file and PyTorch installation")
        response["error_details"] = error_details
    
    return jsonify(response)

@app.route('/debug-model', methods=['GET'])
def debug_model():
    """Диагностика PyTorch модели"""
    try:
        model_path = utils.TRACED_MODEL_PATH
        
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