import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
import warnings
import requests

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Функция для скачивания модели
def download_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        print("Downloading model...")
        url = "https://firebasestorage.googleapis.com/v0/b/innomax-40d4d.appspot.com/o/random_forest_model_0740_rmse_17.pkl?alt=media&token=35720e11-2a75-4e6e-82f2-50c635043837"
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully")
    return model_path

# Инициализация модели
model_path = download_model()
model_data = joblib.load(model_path)
model = model_data['model']
scaler = model_data.get('scaler', None)
feature_names = model_data.get('feature_names', [])

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
        measure_corrected = np.maximum(measure - dark, 1e-10)
        reference_corrected = np.maximum(reference - dark, 1e-10)
        transmittance = np.maximum(measure_corrected / reference_corrected, 1e-6)
        absorbance = -np.log10(transmittance)
        first_deriv = np.gradient(absorbance)
        second_deriv = np.gradient(first_deriv)
        n = len(absorbance)
        q = n // 4
        
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
    if isinstance(data, list):
        data = data[0]
    
    measure = parse_array_field(data.get("measure"))
    reference = parse_array_field(data.get("reference"))
    dark = parse_array_field(data.get("dark"))
    cal_data = parse_array_field(data.get("cal_data"))

    features = calculate_absorbance_features(measure, reference, dark, cal_data)
    if not features:
        return {"error": "Failed to extract features"}, 400

    X = np.array([features.get(feat, 0) for feat in feature_names]).reshape(1, -1)
    if scaler:
        X = scaler.transform(X)

    prediction = model.predict(X)[0]
    return {"predicted_glucose": round(float(prediction), 2)}

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Glucose prediction API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        result = predict_from_json(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)