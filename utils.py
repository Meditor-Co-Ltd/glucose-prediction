import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import pandas as pd
import ast
import torch

# --- Loading the traced model ---
TRACED_MODEL_PATH = 'cnn_model_traced.pt' # Must be the same path

def load_model():
    loaded_traced_model = torch.jit.load(TRACED_MODEL_PATH)
    print(f"Traced model successfully loaded from {TRACED_MODEL_PATH}")

    # Set to evaluation mode (important for inference, even for traced models)
    loaded_traced_model.eval()
    print("Model set to evaluation mode (loaded_traced_model.eval()).")

    # Move the model to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def preprocess_data(measure, reference, dark, cal_data):

    # measure = ast.literal_eval(raw_data['measure'])
    # cal_data = ast.literal_eval(raw_data['cal_data'])
    # dark = ast.literal_eval(raw_data['dark'])
    # reference = ast.literal_eval(raw_data['reference'])

    measure, dark, reference, measure_std, dark_std, reference_std = wavelength_binning(cal_data, measure, dark, reference, 1)

    feature_vector = measure + measure_std + dark + dark_std + reference + reference_std # flat list of numerical features

    # Convert to NumPy arrays
    x = np.array(feature_vector)
    print(np.shape(x))
    # y = np.array(glucose_values)
    x = np.expand_dims(x, 0)
    print(np.shape(x))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x)
    x = x.to(device)

    return x

def model_inference(model, x):
    
    with torch.no_grad():
        prediction = model(x)

    return prediction


def rescale_prediction(y):
    # input_min = 0.0
    # input_max = 1.0
    output_min = 60.0
    output_max = 240.0

    return y * (output_max-output_min) + output_min
