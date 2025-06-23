import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
import warnings
import requests
import hashlib

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Функция для скачивания модели с проверкой целостности
def download_model():
    model_path = "model.pkl"
    
    # Удаляем существующий файл если он поврежден
    if os.path.exists(model_path):
        try:
            # Пробуем загрузить существующий файл
            test_load = joblib.load(model_path)
            print("Existing model file is valid")
            return model_path
        except Exception as e:
            print(f"Existing model file is corrupted: {e}")
            os.remove(model_path)
    
    print("Downloading model...")
    url = "https://firebasestorage.googleapis.com/v0/b/innomax-40d4d.appspot.com/o/random_forest_model_0773_rmse_18.pkl?alt=media&token=aff5ea98-27a7-4e28-b830-d5f7c731dac5"
    
    try:
        # Скачиваем с проверкой статуса
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Проверяем, что получили данные
        if len(response.content) == 0:
            raise Exception("Downloaded file is empty")
        
        # Сохраняем во временный файл сначала
        temp_path = model_path + ".tmp"
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        # Проверяем целостность скачанного файла
        try:
            test_load = joblib.load(temp_path)
            # Переименовываем временный файл в основной
            os.rename(temp_path, model_path)
            print("Model downloaded and verified successfully")
            return model_path
        except Exception as e:
            os.remove(temp_path)
            raise Exception(f"Downloaded model file is corrupted: {e}")
            
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

# Функция для загрузки модели с обработкой ошибок
def load_model_safely(model_path):
    try:
        print("Loading model...")
        model_data = joblib.load(model_path)
        
        # Проверяем структуру загруженных данных
        if not isinstance(model_data, dict):
            raise Exception("Model file does not contain expected dictionary structure")
        
        required_keys = ['model']
        for key in required_keys:
            if key not in model_data:
                raise Exception(f"Model file missing required key: {key}")
        
        model = model_data['model']
        scaler = model_data.get('scaler', None)
        feature_names = model_data.get('feature_names', [])
        
        print(f"Model loaded successfully. Features: {len(feature_names)}")
        return model, scaler, feature_names
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Удаляем поврежденный файл
        if os.path.exists(model_path):
            os.remove(model_path)
        raise

# Инициализация модели с обработкой ошибок
try:
    model_path = download_model()
    model, scaler, feature_names = load_model_safely(model_path)
    print("Model initialization completed successfully")
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
        # Проверяем входные данные
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
        q = max(1, n // 4)  # Избегаем деления на ноль
        
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

    # Создаем вектор признаков
    if len(feature_names) == 0:
        return {"error": "Feature names not available"}, 500
    
    X = np.array([features.get(feat, 0) for feat in feature_names]).reshape(1, -1)
    
    # Проверяем размерность
    if X.shape[1] != len(feature_names):
        return {"error": f"Feature dimension mismatch. Expected {len(feature_names)}, got {X.shape[1]}"}, 400
    
    if scaler:
        X = scaler.transform(X)

    prediction = model.predict(X)[0]
    return {"predicted_glucose": round(float(prediction), 2)}

@app.route('/', methods=['GET'])
def health_check():
    status = "healthy" if model is not None else "unhealthy"
    return jsonify({
        "status": status, 
        "message": "Glucose prediction API is running",
        "model_loaded": model is not None,
        "features_count": len(feature_names)
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

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """Эндпоинт для перезагрузки модели"""
    global model, scaler, feature_names
    try:
        # Удаляем существующий файл модели
        model_path = "model.pkl"
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Перезагружаем модель
        model_path = download_model()
        model, scaler, feature_names = load_model_safely(model_path)
        
        return jsonify({
            "status": "success",
            "message": "Model reloaded successfully",
            "features_count": len(feature_names)
        })
    except Exception as e:
        model, scaler, feature_names = None, None, []
        return jsonify({"error": f"Failed to reload model: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)