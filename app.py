import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
import warnings
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging

warnings.filterwarnings('ignore')

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ВАЖНО: Настройки для Railway
app.config['DEBUG'] = False
app.config['TESTING'] = False

# Mock model creation
def create_mock_model():
    """Creates a mock model with the expected structure for testing"""
    logger.info("Creating mock model for testing...")
    
    feature_names = [
        'abs_mean', 'abs_std', 'abs_max', 'abs_min', 'abs_median', 'abs_range',
        'abs_q25', 'abs_q75', 'abs_iqr', 'abs_area', 'abs_area_positive', 
        'abs_area_negative', 'abs_quarter1', 'abs_quarter2', 'abs_quarter3', 
        'abs_quarter4', 'abs_deriv1_mean', 'abs_deriv1_std', 'abs_deriv2_mean', 
        'abs_deriv2_std', 'abs_gradient_start_end', 'abs_gradient_q1_q4', 
        'abs_n_peaks', 'abs_n_valleys', 'measure_mean', 'measure_std', 
        'reference_mean', 'reference_std', 'dark_mean', 'signal_to_noise',
        'cal_mean', 'cal_std', 'cal_range'
    ]
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    n_samples = 100
    X_dummy = np.random.randn(n_samples, len(feature_names))
    y_dummy = np.random.uniform(70, 200, n_samples)
    
    model.fit(X_dummy, y_dummy)
    
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    return model_data

# Initialize model
try:
    # Для Railway всегда используем mock model для простоты
    model_data = create_mock_model()
    model = model_data['model']
    scaler = model_data.get('scaler', None)
    feature_names = model_data.get('feature_names', [])
    is_mock_model = True
    logger.info(f"Model initialization completed. Features: {len(feature_names)}")
except Exception as e:
    logger.error(f"Fatal error during model initialization: {e}")
    model, scaler, feature_names, is_mock_model = None, None, [], True

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
        logger.error(f"Error in feature calculation: {e}")
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
    
    result = {"predicted_glucose": round(float(prediction), 2)}
    if is_mock_model:
        result["warning"] = "Using mock model for testing. Results are not medically accurate."
    
    return result

# Middleware для логирования всех запросов
@app.before_request
def log_request_info():
    logger.info('--- New Request ---')
    logger.info(f'Method: {request.method}')
    logger.info(f'URL: {request.url}')
    logger.info(f'Path: {request.path}')
    logger.info(f'Headers: {dict(request.headers)}')
    logger.info(f'Args: {request.args}')
    if request.method in ['POST', 'PUT', 'PATCH']:
        logger.info(f'Content-Type: {request.content_type}')
        logger.info(f'Data: {request.get_data(as_text=True)[:500]}...')

# ИСПРАВЛЕНИЕ 1: Удаляем trailing slash проблемы
@app.route('/', methods=['GET'], strict_slashes=False)
def health_check():
    status = "healthy" if model is not None else "unhealthy"
    return jsonify({
        "status": status, 
        "message": "Glucose prediction API is running",
        "model_loaded": model is not None,
        "features_count": len(feature_names),
        "model_type": "mock" if is_mock_model else "production",
        "warning": "Using mock model - not medically accurate" if is_mock_model else None,
        "method_received": request.method,
        "url_received": request.url
    })

# ИСПРАВЛЕНИЕ 2: Добавляем поддержку OPTIONS для CORS
@app.route('/predict', methods=['POST', 'OPTIONS'], strict_slashes=False)
def predict():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    # Логируем подробности запроса
    logger.info(f"PREDICT endpoint called with method: {request.method}")
    logger.info(f"Content-Type: {request.content_type}")
    logger.info(f"Is JSON: {request.is_json}")
    
    try:
        if model is None:
            return jsonify({"error": "Model not loaded. Please restart the service."}), 503
        
        # Проверяем метод запроса
        if request.method != 'POST':
            return jsonify({
                "error": f"Method {request.method} not allowed. Use POST.",
                "received_method": request.method,
                "expected_method": "POST"
            }), 405
        
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "content_type": request.content_type,
                "data_received": request.get_data(as_text=True)[:200]
            }), 400
        
        result = predict_from_json(data)
        if isinstance(result, tuple):
            response = jsonify(result[0])
            response.status_code = result[1]
        else:
            response = jsonify(result)
        
        # Добавляем CORS headers
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# ИСПРАВЛЕНИЕ 3: Добавляем debug endpoint
@app.route('/debug', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'], strict_slashes=False)
def debug():
    return jsonify({
        "method": request.method,
        "url": request.url,
        "path": request.path,
        "headers": dict(request.headers),
        "args": dict(request.args),
        "form": dict(request.form),
        "json": request.get_json() if request.is_json else None,
        "data": request.get_data(as_text=True)[:500]
    })

# ИСПРАВЛЕНИЕ 4: Глобальный CORS handler
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ИСПРАВЛЕНИЕ 5: Ловим все 404 для диагностики
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "method": request.method,
        "path": request.path,
        "available_endpoints": [
            "GET /",
            "POST /predict", 
            "GET,POST /debug"
        ]
    }), 404

# ИСПРАВЛЕНИЕ 6: Ловим метод не разрешен
@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "method": request.method,
        "path": request.path,
        "message": "Check if you're using the correct HTTP method"
    }), 405

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    # ВАЖНО: Для Railway отключаем reloader и используем threaded
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=False, 
        use_reloader=False,
        threaded=True
    )