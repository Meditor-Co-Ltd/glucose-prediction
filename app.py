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
import requests

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
            
            # Способ 1: Использование пути к файлу
            service_account_path = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH')
            logger.info(f"Checking FIREBASE_SERVICE_ACCOUNT_PATH: {service_account_path}")
            
            # Способ 2: Создание файла из JSON переменной
            service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
            logger.info(f"Checking FIREBASE_SERVICE_ACCOUNT_JSON: {'present' if service_account_json else 'not found'}")
            
            # Если есть JSON но нет пути к файлу, создаем временный файл
            if service_account_json and not (service_account_path and os.path.exists(service_account_path)):
                logger.info("Creating temporary service account file from JSON...")
                try:
                    temp_path = './temp_firebase_service_account.json'
                    
                    # Проверяем, что JSON валидный
                    service_account_info = json.loads(service_account_json)
                    logger.info(f"JSON parsed successfully, project_id: {service_account_info.get('project_id', 'unknown')}")
                    
                    # Записываем в файл
                    with open(temp_path, 'w') as f:
                        json.dump(service_account_info, f, indent=2)
                    
                    service_account_path = temp_path
                    logger.info(f"✅ Created temporary service account file: {service_account_path}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON in FIREBASE_SERVICE_ACCOUNT_JSON: {e}")
                except Exception as e:
                    logger.error(f"❌ Failed to create service account file: {e}")
            
            # Инициализация с файлом (оригинальным или созданным)
            if service_account_path and os.path.exists(service_account_path):
                logger.info(f"Initializing Firebase with service account file: {service_account_path}")
                try:
                    # Проверяем содержимое файла
                    with open(service_account_path, 'r') as f:
                        file_content = json.load(f)
                        logger.info(f"Service account file loaded, project_id: {file_content.get('project_id', 'unknown')}")
                    
                    cred = credentials.Certificate(service_account_path)
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': 'innomax-40d4d.appspot.com'
                    })
                    logger.info("✅ Firebase initialized with service account file")
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Failed to initialize with service account file: {e}")
            
            # Fallback: прямая инициализация из JSON (если файл не сработал)
            if service_account_json:
                logger.info("Fallback: initializing directly from JSON string...")
                try:
                    service_account_info = json.loads(service_account_json)
                    cred = credentials.Certificate(service_account_info)
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': 'innomax-40d4d.appspot.com'
                    })
                    logger.info("✅ Firebase initialized directly from JSON")
                    return True
                except Exception as e:
                    logger.error(f"❌ Failed to initialize directly from JSON: {e}")
            
            # Способ 3: Application Default Credentials (для Google Cloud)
            try:
                logger.info("Trying Application Default Credentials...")
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'innomax-40d4d.appspot.com'
                })
                logger.info("✅ Firebase initialized with Application Default Credentials")
                return True
            except Exception as e:
                logger.warning(f"❌ Failed with default credentials: {e}")
        
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
            if key.upper() == 'FIREBASE_SERVICE_ACCOUNT_JSON':
                # Маскируем JSON содержимое
                if len(value) > 100:
                    masked_value = f"{value[:30]}...{value[-30:]}"
                else:
                    masked_value = "***JSON_PRESENT***"
            else:
                masked_value = value
            logger.info(f"  {key}: {masked_value}")
    
    return False

def download_model_from_firebase():
    """Скачивает модель из Firebase Storage"""
    try:
        logger.info("Getting Firebase Storage bucket...")
        bucket = storage.bucket()
        logger.info(f"Bucket obtained: {bucket.name}")
        
        # Пробуем разные возможные пути к файлу модели
        possible_paths = [
            "random_forest_model_0773_rmse_18.pkl",
            "models/random_forest_model_0773_rmse_18.pkl",
            "ml/random_forest_model_0773_rmse_18.pkl",
            "model.pkl",
            "data/random_forest_model_0773_rmse_18.pkl"
        ]
        
        blob = None
        blob_name = None
        
        # Ищем файл по разным путям
        for path in possible_paths:
            logger.info(f"Trying path: {path}")
            test_blob = bucket.blob(path)
            
            try:
                if test_blob.exists():
                    logger.info(f"✅ Found model at path: {path}")
                    blob = test_blob
                    blob_name = path
                    break
                else:
                    logger.info(f"❌ Not found at path: {path}")
            except Exception as e:
                logger.warning(f"Error checking path {path}: {e}")
        
        if blob is None:
            # Попробуем получить список файлов для диагностики
            try:
                logger.info("Model not found, listing bucket contents...")
                blobs_list = list(bucket.list_blobs(max_results=10))
                available_files = [b.name for b in blobs_list]
                logger.error(f"Available files in bucket: {available_files}")
            except Exception as list_error:
                logger.error(f"Could not list bucket contents: {list_error}")
            
            raise Exception(f"Model file not found in any of the paths: {possible_paths}")
        
        logger.info("Getting blob metadata...")
        blob.reload()  # Загружаем метаданные
        logger.info(f"Blob size: {blob.size} bytes, updated: {blob.updated}")
        
        if blob.size == 0:
            raise Exception("Model file exists but is empty (0 bytes)")
        
        # Создаем временный файл для скачивания
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            temp_path = temp_file.name
        
        logger.info(f"Created temp file: {temp_path}")
        
        # Скачиваем файл с таймаутом
        logger.info(f"Starting download of {blob_name} from Firebase Storage...")
        
        try:
            # Используем download_to_filename с retry логикой
            blob.download_to_filename(temp_path)
            logger.info("Download completed successfully")
        except Exception as download_error:
            logger.error(f"Download failed: {download_error}")
            # Попробуем альтернативный способ скачивания
            logger.info("Trying alternative download method...")
            try:
                data = blob.download_as_bytes()
                with open(temp_path, 'wb') as f:
                    f.write(data)
                logger.info("Alternative download method succeeded")
            except Exception as alt_error:
                logger.error(f"Alternative download also failed: {alt_error}")
                raise download_error
        
        # Проверяем размер скачанного файла
        file_size = os.path.getsize(temp_path)
        logger.info(f"Downloaded file size: {file_size} bytes")
        
        if file_size == 0:
            raise Exception("Downloaded file is empty")
        
        if file_size != blob.size:
            logger.warning(f"Size mismatch: expected {blob.size}, got {file_size}")
        
        logger.info("✅ Model downloaded successfully, loading with joblib...")
        
        # Загружаем модель
        try:
            model_data = joblib.load(temp_path)
            logger.info(f"Model loaded successfully, type: {type(model_data)}")
            
            # Проверяем структуру модели
            if isinstance(model_data, dict):
                logger.info(f"Model data keys: {list(model_data.keys())}")
            
        except Exception as load_error:
            logger.error(f"Failed to load model with joblib: {load_error}")
            # Попробуем проверить, что это действительно pickle файл
            try:
                with open(temp_path, 'rb') as f:
                    header = f.read(10)
                    logger.info(f"File header (hex): {header.hex()}")
            except:
                pass
            raise load_error
        
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

def download_model_http_fallback():
    """Fallback загрузка модели через HTTP"""
    url = "https://firebasestorage.googleapis.com/v0/b/innomax-40d4d.appspot.com/o/random_forest_model_0773_rmse_18.pkl?alt=media&token=aff5ea98-27a7-4e28-b830-d5f7c731dac5"
    
    try:
        logger.info("Fallback: downloading model via HTTP...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        if len(response.content) == 0:
            raise Exception("HTTP response is empty")
        
        logger.info(f"HTTP download successful, size: {len(response.content)} bytes")
        
        # Сохраняем файл
        with open("model.pkl", 'wb') as f:
            f.write(response.content)
        
        # Загружаем модель
        model_data = joblib.load("model.pkl")
        logger.info("✅ Model loaded successfully via HTTP fallback")
        return model_data
        
    except requests.exceptions.HTTPError as e:
        if hasattr(e, 'response') and e.response.status_code == 402:
            logger.error("HTTP fallback failed: Payment Required (402) - URL expired")
        else:
            logger.error(f"HTTP fallback failed: {e}")
        return None
    except Exception as e:
        logger.error(f"HTTP fallback failed: {e}")
        return None

def load_model_with_fallback():
    """Загружает модель: локальный файл → Firebase Storage → HTTP fallback"""
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
    
    # Попытка 2: Firebase Storage
    if initialize_firebase():
        try:
            model_data = download_model_from_firebase()
            logger.info("✅ Model downloaded and loaded successfully from Firebase Storage")
            return model_data
        except Exception as e:
            logger.error(f"❌ Firebase Storage download failed: {e}")
            logger.info("Trying HTTP fallback...")
    else:
        logger.error("❌ Firebase initialization failed, trying HTTP fallback...")
    
    # Попытка 3: HTTP fallback
    try:
        model_data = download_model_http_fallback()
        if model_data:
            logger.info("✅ Model loaded successfully via HTTP fallback")
            return model_data
        else:
            logger.error("❌ HTTP fallback also failed")
    except Exception as e:
        logger.error(f"❌ HTTP fallback failed: {e}")
    
    # Если все попытки провалились
    raise Exception("Failed to load model from all sources: local file, Firebase Storage, and HTTP fallback")

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
    
    # Расширенная диагностическая информация
    diagnostics = {
        "firebase_credentials_configured": bool(
            os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON') or 
            os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH')
        ),
        "local_model_exists": os.path.exists("model.pkl"),
        "temp_firebase_file_exists": os.path.exists("./temp_firebase_service_account.json"),
        "environment": {
            "PORT": os.environ.get('PORT', 'not_set'),
            "PYTHONUNBUFFERED": os.environ.get('PYTHONUNBUFFERED', 'not_set'),
            "has_firebase_json": bool(os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')),
            "has_firebase_path": bool(os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH')),
            "firebase_path_value": os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH', 'not_set'),
        },
        "file_system": {
            "working_directory": os.getcwd(),
            "files_in_root": [f for f in os.listdir('.') if f.endswith(('.json', '.pkl', '.py'))],
        }
    }
    
    # Проверяем доступ к файлу, если путь указан
    firebase_path = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH')
    if firebase_path:
        diagnostics["firebase_file_check"] = {
            "path": firebase_path,
            "exists": os.path.exists(firebase_path),
            "is_readable": False
        }
        
        if os.path.exists(firebase_path):
            try:
                with open(firebase_path, 'r') as f:
                    content = json.load(f)
                    diagnostics["firebase_file_check"]["is_readable"] = True
                    diagnostics["firebase_file_check"]["project_id"] = content.get('project_id', 'unknown')
            except Exception as e:
                diagnostics["firebase_file_check"]["read_error"] = str(e)
    
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
        error_details = []
        if firebase_status != "initialized":
            error_details.append("Firebase not initialized - check credentials")
        if not model:
            error_details.append("Model failed to load - check Firebase Storage and model file")
        response["error_details"] = error_details
    
    return jsonify(response)

@app.route('/debug-storage', methods=['GET'])
def debug_storage():
    """Диагностика Firebase Storage"""
    try:
        if not firebase_admin._apps:
            return jsonify({"error": "Firebase not initialized"}), 500
        
        logger.info("Debug: Checking Firebase Storage...")
        bucket = storage.bucket()
        logger.info(f"Debug: Connected to bucket: {bucket.name}")
        
        # Список файлов в bucket (ограничиваем для безопасности)
        try:
            blobs = list(bucket.list_blobs(max_results=20))
            files = [{"name": blob.name, "size": blob.size, "updated": str(blob.updated)} for blob in blobs]
            logger.info(f"Debug: Found {len(files)} files in bucket")
        except Exception as e:
            logger.error(f"Debug: Failed to list files: {e}")
            files = []
        
        # Проверяем конкретный файл модели
        target_file = "random_forest_model_0773_rmse_18.pkl"
        target_blob = bucket.blob(target_file)
        
        target_info = {
            "name": target_file,
            "exists": False,
            "size": None,
            "updated": None,
            "error": None
        }
        
        try:
            target_info["exists"] = target_blob.exists()
            if target_info["exists"]:
                target_blob.reload()  # Загружаем метаданные
                target_info["size"] = target_blob.size
                target_info["updated"] = str(target_blob.updated)
                logger.info(f"Debug: Target file exists, size: {target_info['size']}")
            else:
                logger.warning("Debug: Target file does not exist")
        except Exception as e:
            target_info["error"] = str(e)
            logger.error(f"Debug: Error checking target file: {e}")
        
        # Ищем файлы похожие на модель
        model_files = [f for f in files if '.pkl' in f['name'] or 'model' in f['name'].lower()]
        
        return jsonify({
            "bucket_name": bucket.name,
            "total_files": len(files),
            "all_files": files,
            "target_file": target_info,
            "model_files": model_files,
            "debug_info": {
                "firebase_initialized": bool(firebase_admin._apps),
                "bucket_accessible": True
            }
        })
        
    except Exception as e:
        logger.error(f"Debug: Storage debug failed: {e}")
        return jsonify({
            "error": str(e),
            "firebase_initialized": bool(firebase_admin._apps),
            "bucket_accessible": False
        }), 500

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
