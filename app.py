import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
import warnings
import tempfile
import firebase_admin
from firebase_admin import credentials, storage
import logging
import sys

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

def initialize_firebase():
    """Инициализация Firebase Admin SDK"""
    try:
        logger.info("Starting Firebase initialization...")
        
        # Проверяем, не инициализирован ли уже Firebase
        if not firebase_admin._apps:
            # Способ 1: Использование переменной окружения с путем к файлу ключа
            service_account_path = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH')
            logger.info(f"Checking service account path: {service_account_path}")
            
            if service_account_path and os.path.exists(service_account_path):
                logger.info("Found service account file, initializing...")
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'innomax-40d4d.appspot.com'  # Ваш bucket name
                })
                logger.info("✅ Firebase initialized with service account file")
                return True
            
            # Способ 2: Использование JSON строки из переменной окружения
            service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
            logger.info(f"Checking service account JSON env var: {'present' if service_account_json else 'not found'}")
            
            if service_account_json:
                try:
                    # Логируем только первые и последние символы для безопасности
                    json_preview = f"{service_account_json[:50]}...{service_account_json[-50:]}" if len(service_account_json) > 100 else "short_json"
                    logger.info(f"Parsing service account JSON: {json_preview}")
                    
                    service_account_info = json.loads(service_account_json)
                    logger.info(f"JSON parsed successfully, project_id: {service_account_info.get('project_id', 'unknown')}")
                    
                    cred = credentials.Certificate(service_account_info)
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': 'innomax-40d4d.appspot.com'
                    })
                    logger.info("✅ Firebase initialized with service account JSON")
                    return True
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON in FIREBASE_SERVICE_ACCOUNT_JSON: {e}")
                except Exception as e:
                    logger.error(f"❌ Error initializing Firebase with JSON: {e}")
            
            # Способ 3: Попытка использовать Application Default Credentials (для Google Cloud)
            try:
                logger.info("Trying Application Default Credentials...")
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'innomax-40d4d.appspot.com'
                })
                logger.info("✅ Firebase initialized with Application Default Credentials")
                return True
            except Exception as e:
                logger.warning(f"❌ Failed to initialize with default credentials: {e}")
        
        else:
            logger.info("✅ Firebase already initialized")
            return True
            
    except Exception as e:
        logger.error(f"❌ Firebase initialization failed: {e}")
        return False
    
    logger.error("❌ No valid Firebase credentials found")
    logger.info("Available environment variables:")
    for key in os.environ:
        if 'FIREBASE' in key.upper():
            value = os.environ[key]
            # Маскируем чувствительные данные
            if len(value) > 50:
                masked_value = f"{value[:20]}...{value[-20:]}"
            else:
                masked_value = "***masked***"
            logger.info(f"  {key}: {masked_value}")
    
    return False

def download_model_from_firebase():
    """Скачивает модель из Firebase Storage"""
    try:
        logger.info("Getting Firebase Storage bucket...")
        bucket = storage.bucket()
        logger.info(f"Bucket obtained: {bucket.name}")
        
        # Путь к файлу в bucket (без начального слеша)
        blob_name = "random_forest_model_0773_rmse_18.pkl"
        logger.info(f"Looking for blob: {blob_name}")
        blob = bucket.blob(blob_name)
        
        # Проверяем, существует ли файл
        logger.info("Checking if blob exists...")
        if not blob.exists():
            logger.error(f"File {blob_name} not found in bucket")
            raise Exception(f"File {blob_name} not found in bucket")
        
        logger.info("Blob exists, getting metadata...")
        blob.reload()  # Загружаем метаданные
        logger.info(f"Blob size: {blob.size} bytes, updated: {blob.updated}")
        
        # Создаем временный файл для скачивания
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            temp_path = temp_file.name
        
        logger.info(f"Created temp file: {temp_path}")
        
        # Скачиваем файл
        logger.info(f"Starting download of {blob_name} from Firebase Storage...")
        blob.download_to_filename(temp_path)
        
        # Проверяем размер скачанного файла
        file_size = os.path.getsize(temp_path)
        logger.info(f"Downloaded file size: {file_size} bytes")
        
        if file_size == 0:
            raise Exception("Downloaded file is empty")
        
        logger.info("✅ Model downloaded successfully, loading with joblib...")
        
        # Загружаем модель
        model_data = joblib.load(temp_path)
        logger.info(f"Model loaded, type: {type(model_data)}")
        
        # Сохраняем модель локально для будущего использования
        local_path = "model.pkl"
        os.rename(temp_path, local_path)
        logger.info(f"✅ Model saved locally as {local_path}")
        
        return model_data
        
    except Exception as e:
        logger.error(f"Error in download_model_from_firebase: {e}")
        # Удаляем временный файл в случае ошибки
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
        except Exception as cleanup_e:
            logger.warning(f"Failed to cleanup temp file: {cleanup_e}")
        raise Exception(f"Failed to download model from Firebase: {e}")

def load_model_with_fallback():
    """Загружает модель: сначала из локального файла, потом из Firebase"""
    model_path = "model.pkl"
    
    # Попытка 1: Загрузка из локального файла
    if os.path.exists(model_path):
        try:
            logger.info("Loading model from local file...")
            model_data = joblib.load(model_path)
            logger.info("✅ Model loaded successfully from local file")
            return model_data
        except Exception as e:
            logger.error(f"❌ Failed to load local model: {e}")
            logger.info("Trying to download from Firebase...")
    else:
        logger.info("Local model not found, downloading from Firebase...")
    
    # Попытка 2: Инициализация Firebase и скачивание
    logger.info("Initializing Firebase for model download...")
    if not initialize_firebase():
        raise Exception("Failed to initialize Firebase. Check your credentials.")
    
    try:
        model_data = download_model_from_firebase()
        logger.info("✅ Model downloaded and loaded successfully from Firebase")
        return model_data
        
    except Exception as e:
        logger.error(f"Failed to download/load model from Firebase: {e}")
        raise Exception(f"Failed to download/load model from Firebase: {e}")

# Инициализация модели
logger.info("=== Starting model initialization ===")
try:
    model_data = load_model_with_fallback()
    model = model_data['model']
    scaler = model_data.get('scaler', None)
    feature_names = model_data.get('feature_names', [])
    logger.info(f"✅ Model initialization completed. Features: {len(feature_names)}")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Scaler present: {scaler is not None}")
except Exception as e:
    logger.error(f"❌ Fatal error during model initialization: {e}")
    logger.info("Setting model, scaler, feature_names to None/empty")
    model, scaler, feature_names = None, None, []
logger.info("=== Model initialization complete ===")

# Добавляем дополнительное логирование для Railway
logger.info("=== Environment Information ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PORT environment variable: {os.environ.get('PORT', 'not set')}")
logger.info(f"Available environment variables with 'FIREBASE': {[k for k in os.environ.keys() if 'FIREBASE' in k.upper()]}")
logger.info("=== Environment Information Complete ===")

def parse_array_field(x):
    if isinstance(x, str):
        try:
            parsed = json.loads(x)
            return np.array(parsed if isinstance(parsed, list) else [])
        except:
            return np.array([])
    elif isinstance(x, list) or isinstance(x, np.ndarray):
        return np.array(x)
    return np.array([])

def calculate_absorbance_features(measure, reference, dark, cal_data=None):
    try:
        if len(measure) == 0 or len(reference) == 0 or len(dark) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if len(measure) != len(reference) or len(measure) != len(dark):
            raise ValueError("Input arrays must have the same length")
        
        measure_corrected = np.maximum(measure - dark, 1e-10)
        reference_corrected = np.maximum(reference - dark, 1e-10)
        transmittance = np.maximum(measure_corrected / reference_corrected, 1e-6)
        absorbance = -np.log10(transmittance)
        first_deriv = np.gradient(absorbance)
        second_deriv = np.gradient(first_deriv)
        n = len(absorbance)
        q = max(1, n // 4)
        
        features = {
            'abs_mean': np.mean(absorbance),
            'abs_std': np.std(absorbance),
            'abs_max': np.max(absorbance),
            'abs_min': np.min(absorbance),
            'abs_median': np.median(absorbance),
            'abs_range': np.max(absorbance) - np.min(absorbance),
            'abs_q25': np.percentile(absorbance, 25),
            'abs_q75': np.percentile(absorbance, 75),
            'abs_iqr': np.percentile(absorbance, 75) - np.percentile(absorbance, 25),
            'abs_area': np.trapz(absorbance),
            'abs_area_positive': np.trapz(np.maximum(absorbance, 0)),
            'abs_area_negative': np.trapz(np.minimum(absorbance, 0)),
            'abs_quarter1': np.mean(absorbance[:q]),
            'abs_quarter2': np.mean(absorbance[q:2*q]),
            'abs_quarter3': np.mean(absorbance[2*q:3*q]),
            'abs_quarter4': np.mean(absorbance[3*q:]),
            'abs_deriv1_mean': np.mean(np.abs(first_deriv)),
            'abs_deriv1_std': np.std(first_deriv),
            'abs_deriv2_mean': np.mean(np.abs(second_deriv)),
            'abs_deriv2_std': np.std(second_deriv),
            'abs_gradient_start_end': (absorbance[-1] - absorbance[0]) / len(absorbance),
            'abs_gradient_q1_q4': np.mean(absorbance[3*q:]) - np.mean(absorbance[:q]),
            'abs_n_peaks': len([i for i in range(1, len(absorbance)-1) if absorbance[i] > absorbance[i-1] and absorbance[i] > absorbance[i+1]]),
            'abs_n_valleys': len([i for i in range(1, len(absorbance)-1) if absorbance[i] < absorbance[i-1] and absorbance[i] < absorbance[i+1]]),
            'measure_mean': np.mean(measure),
            'measure_std': np.std(measure),
            'reference_mean': np.mean(reference),
            'reference_std': np.std(reference),
            'dark_mean': np.mean(dark),
            'signal_to_noise': np.mean(measure) / np.std(dark) if np.std(dark) > 0 else 0,
        }
        
        if cal_data is not None and len(cal_data) > 0:
            features.update({
                'cal_mean': np.mean(cal_data),
                'cal_std': np.std(cal_data),
                'cal_range': np.max(cal_data) - np.min(cal_data)
            })
        
        return features
    except Exception as e:
        print(f"Error in feature calculation: {e}")
        return None

def predict_from_json(data):
    if model is None:
        return {"error": "Model not loaded. Please restart the service."}, 500
    
    if isinstance(data, list):
        data = data[0]
    
    measure = parse_array_field(data.get("measure"))
    reference = parse_array_field(data.get("reference"))
    dark = parse_array_field(data.get("dark"))
    cal_data = parse_array_field(data.get("cal_data"))

    features = calculate_absorbance_features(measure, reference, dark, cal_data)
    if not features:
        return {"error": "Failed to extract features"}, 400

    if len(feature_names) == 0:
        return {"error": "Feature names not available"}, 500
    
    X = np.array([features.get(feat, 0) for feat in feature_names]).reshape(1, -1)
    
    if scaler:
        X = scaler.transform(X)

    prediction = model.predict(X)[0]
    return {"predicted_glucose": round(float(prediction), 2)}

@app.route('/', methods=['GET'])
def health_check():
    status = "healthy" if model is not None else "unhealthy"
    model_source = "unknown"
    
    if model is not None:
        if os.path.exists("model.pkl"):
            model_source = "local_file"
        else:
            model_source = "downloaded"
    
    firebase_status = "initialized" if firebase_admin._apps else "not_initialized"
    
    # Дополнительная диагностическая информация
    diagnostics = {
        "firebase_credentials_configured": bool(
            os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON') or 
            os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH')
        ),
        "local_model_exists": os.path.exists("model.pkl"),
        "environment": {
            "PORT": os.environ.get('PORT', 'not_set'),
            "PYTHONUNBUFFERED": os.environ.get('PYTHONUNBUFFERED', 'not_set'),
            "has_firebase_json": bool(os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')),
            "has_firebase_path": bool(os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH')),
        }
    }
    
    # Логируем результат health check
    logger.info(f"Health check: status={status}, firebase={firebase_status}, model_loaded={model is not None}")
    
    response = {
        "status": status, 
        "message": "Glucose prediction API is running",
        "model_loaded": model is not None,
        "features_count": len(feature_names),
        "model_source": model_source,
        "firebase_status": firebase_status,
        "diagnostics": diagnostics
    }
    
    # Если что-то не так, добавляем детали
    if status == "unhealthy":
        if not firebase_status == "initialized":
            response["error_details"] = "Firebase not initialized - check credentials"
        elif not model:
            response["error_details"] = "Model failed to load - check Firebase Storage and model file"
    
    return jsonify(response)

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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
