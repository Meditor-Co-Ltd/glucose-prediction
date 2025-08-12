import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import pandas as pd
# import ast
import torch
import pickle

# --- Loading the traced model ---
TRACED_MODEL_PATH = 'cnn_model_traced_20250806.pt' # Must be the same path

def load_model():
    loaded_traced_model = torch.jit.load(TRACED_MODEL_PATH)
    print(f"Traced model successfully loaded from {TRACED_MODEL_PATH}")

    # Set to evaluation mode (important for inference, even for traced models)
    loaded_traced_model.eval()
    print("Model set to evaluation mode (loaded_traced_model.eval()).")

    # Move the model to the appropriate device (CPU or GPU)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    loaded_traced_model.to(device)
    print(f"Model moved to device: {device}")

    return loaded_traced_model

def wavelength_binning(cal_data, measure, dark, reference, bin_size=20):
    
    cal_data = np.asarray(cal_data)
    measure = np.asarray(measure)
    dark = np.asarray(dark)
    reference = np.asarray(reference)
    measure_binned = []
    dark_binned = []
    reference_binned = []
    measure_binned2 = []
    dark_binned2 = []
    reference_binned2 = []
    bins = range(420, 750, bin_size)
    for bin in bins:
        indices = np.argwhere((cal_data >= bin) & (cal_data <= bin+bin_size))
        measure_binned.append(np.nanmean(measure[indices]))
        dark_binned.append(np.nanmean(dark[indices]))
        reference_binned.append(np.nanmean(reference[indices]))
        measure_binned2.append(np.nanstd(measure[indices]))
        dark_binned2.append(np.nanstd(dark[indices]))
        reference_binned2.append(np.nanstd(reference[indices]))

    measure_binned = np.hstack(measure_binned)
    dark_binned = np.hstack(dark_binned)
    reference_binned = np.hstack(reference_binned)
    measure_binned2 = np.hstack(measure_binned2)
    dark_binned2 = np.hstack(dark_binned2)
    reference_binned2 = np.hstack(reference_binned2)
    
    return [measure_binned], [dark_binned], [reference_binned], [measure_binned2], [dark_binned2], [reference_binned2]


def normalize_1d(x):
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min + 1e-8
    x_normalized = (x - x_min) / x_range

    return x_normalized


def normalize_inputs(x):
    with open("calibration_value.pkl", 'rb') as f:
        avg = pickle.load(f)
    normalized_x = (x - avg) / avg
    return normalized_x


def normalize_inputs2(x):
        
    with open("average50.pkl", 'rb') as f:
        avg50 = pickle.load(f)
    with open("average0.pkl", 'rb') as f:
        avg0 = pickle.load(f)
    with open("average100.pkl", 'rb') as f:
        avg100 = pickle.load(f)

    normalized_x0 = (x - avg0) / avg100
    normalized_x50 = (x - avg50) / avg50
    normalized_x100 = (x - avg100) / avg100
    normalized_stacked_x = np.hstack((normalized_x0, normalized_x50, normalized_x100))

    return normalized_stacked_x


def preprocess_data(measure, reference, dark, cal_data):

    # measure = ast.literal_eval(raw_data['measure'])
    # cal_data = ast.literal_eval(raw_data['cal_data'])
    # dark = ast.literal_eval(raw_data['dark'])
    # reference = ast.literal_eval(raw_data['reference'])

    measure, dark, reference, measure_std, dark_std, reference_std = wavelength_binning(cal_data, measure, dark, reference, 1)

    feature_vector = measure + measure_std + dark + dark_std + reference + reference_std # flat list of numerical features

    # Convert to NumPy arrays
    x = np.array(feature_vector)
    # x = normalize_inputs(x)
    x = normalize_inputs2(x)
    print(np.shape(x))
    # y = np.array(glucose_values)
    x = np.expand_dims(x, 0)
    print(np.shape(x))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    x = torch.from_numpy(x).double()
    x = x.to(device)

    return x


def model_inference(model, x):
    
    with torch.no_grad():
        prediction = model(x)

    return prediction


def rescale_prediction(y):
    # input_min = 0.0
    # input_max = 1.0
    output_min = 49.0
    output_max = 235.0

    return y * (output_max-output_min) + output_min
