import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
import warnings
import tempfile
import firebase_admin
from firebase_admin import credentials, storage

warnings.filterwarnings('ignore')

app = Flask(__name__)

def initialize_firebase():
    """Инициализация Firebase Admin SDK"""
    try:
        # Проверяем, не инициализирован ли уже Firebase
        if not firebase_admin._apps:
            # Способ 1: Использование переменной окружения с путем к файлу ключа
            service_account_path = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH')
            
            if service_account_path and os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'innomax-40d4d.appspot.com'  # Ваш bucket name
                })
                print("✅ Firebase initialized with service account file")
                return True
            
            # Способ 2: Использование JSON строки из переменной окружения
            service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
            
            if service_account_json:
                try:
                    service_account_info = json.loads(service_account_json)
                    cred = credentials.Certificate(service_account_info)
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': 'innomax-40d4d.appspot.com'
                    })
                    print("✅ Firebase initialized with service account JSON")
                    return True
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON in FIREBASE_SERVICE_ACCOUNT_JSON: {e}")
            
            # Способ 3: Попытка использовать Application Default Credentials (для Google Cloud)
            try:
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'innomax-40d4d.appspot.com'
                })
                print("✅ Firebase initialized with Application Default Credentials")
                return True
            except Exception as e:
                print(f"❌ Failed to initialize with default credentials: {e}")
        
        else:
            print("✅ Firebase already initialized")
            return True
            
    except Exception as e:
        print(f"❌ Firebase initialization failed: {e}")
        return False
    
    print("❌ No valid Firebase credentials found")
    return False

def download_model_from_firebase():
    """Скачивает модель из Firebase Storage"""
    try:
        bucket = storage.bucket()
        
        # Путь к файлу в bucket (без начального слеша)
        blob_name = "random_forest_model_0773_rmse_18.pkl"
        blob = bucket.blob(blob_name)
        
        # Проверяем, существует ли файл
        if not blob.exists():
            raise Exception(f"File {blob_name} not found in bucket")
        
        # Создаем временный файл для скачивания
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            temp_path = temp_file.name
        
        # Скачиваем файл
        print(f"Downloading {blob_name} from Firebase Storage...")
        blob.download_to_filename(temp_path)
        
        # Проверяем размер скачанного файла
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise Exception("Downloaded file is empty")
        
        print(f"✅ Model downloaded successfully ({file_size} bytes)")
        
        # Загружаем модель
        model_data = joblib.load(temp_path)
        
        # Сохраняем модель локально для будущего использования
        local_path = "model.pkl"
        os.rename(temp_path, local_path)
        print(f"✅ Model saved locally as {local_path}")
        
        return model_data
        
    except Exception as e:
        # Удаляем временный файл в случае ошибки
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
        raise Exception(f"Failed to download model from Firebase: {e}")

def load_model_with_fallback():
    """Загружает модель: сначала из локального файла, потом из Firebase"""
    model_path = "model.pkl"
    
    # Попытка 1: Загрузка из локального файла
    if os.path.exists(model_path):
        try:
            print("Loading model from local file...")
            model_data = joblib.load(model_path)
            print("✅ Model loaded successfully from local file")
            return model_data
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            print("Trying to download from Firebase...")
    else:
        print("Local model not found, downloading from Firebase...")
    
    # Попытка 2: Инициализация Firebase и скачивание
    if not initialize_firebase():
        raise Exception("Failed to initialize Firebase. Check your credentials.")
    
    try:
        model_data = download_model_from_firebase()
        print("✅ Model downloaded and loaded successfully from Firebase")
        return model_data
        
    except Exception as e:
        raise Exception(f"Failed to download/load model from Firebase: {e}")

# Инициализация модели
try:
    model_data = load_model_with_fallback()
    model = model_data['model']
    scaler = model_data.get('scaler', None)
    feature_names = model_data.get('feature_names', [])
    print(f"Model initialization completed. Features: {len(feature_names)}")
except Exception as e:
    print(f"Fatal error during model initialization: {e}")
    model, scaler, feature_names = None, None, []

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
    
    return jsonify({
        "status": status, 
        "message": "Glucose prediction API is running",
        "model_loaded": model is not None,
        "features_count": len(feature_names),
        "model_source": model_source,
        "firebase_status": firebase_status
    })

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
