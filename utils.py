import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import pandas as pd
# import ast
import torch
import pickle

# --- Loading the traced model ---
NORMAL_CLASSIFICATION_MODEL_PATH = 'classification_model_500_650_ALL.pt' #'classification_model_500_650_NORMAL.pt'
DIABETIC_CLASSIFICATION_MODEL_PATH = 'classification_model_500_650_DIABETIC.pt'
ALL_CLASSIFICATION_MODEL_PATH = 'classification_model_450_650_ALL_both_normalized.pt'

def load_model(TRACED_MODEL_PATH):
    loaded_traced_model = torch.jit.load(TRACED_MODEL_PATH, map_location='cpu')
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

    # Min-Max normalization (best for absorption spectra)
    sample_min = np.min(x, axis=2, keepdims=True)
    # print(sample_min)
    sample_max = np.max(x, axis=2, keepdims=True)
    # print(sample_max)
    if np.any(sample_max - sample_min < 1e-8):
        # Avoid division by zero for flat channels
        safe_range = np.where(sample_max - sample_min < 1e-8, 1.0, sample_max - sample_min)
        return (x - sample_min) / safe_range
    return (x - sample_min) / (sample_max - sample_min)


def normalize_inputs(x):
    with open("calibration_value.pkl", 'rb') as f:
        avg = pickle.load(f)
    normalized_x = (x - avg) / avg
    return normalized_x


def normalize_inputs2(x, baseline):
    
    if baseline < 100:
        with open("NORMAL_average.pkl", 'rb') as f:
            avg = pickle.load(f)
        with open("NORMAL_average0.pkl", 'rb') as f:
            avg0 = pickle.load(f)
        with open("NORMAL_average50.pkl", 'rb') as f:
            avg50 = pickle.load(f)
        with open("NORMAL_average100.pkl", 'rb') as f:
            avg100 = pickle.load(f)
    elif baseline >= 100 and baseline < 125:
        with open("PREDIABETIC_average.pkl", 'rb') as f:
            avg = pickle.load(f)
        with open("PREDIABETIC_average0.pkl", 'rb') as f:
            avg0 = pickle.load(f)
        with open("PREDIABETIC_average50.pkl", 'rb') as f:
            avg50 = pickle.load(f)
        with open("PREDIABETIC_average100.pkl", 'rb') as f:
            avg100 = pickle.load(f)
    else:
        with open("DIABETIC_average.pkl", 'rb') as f:
            avg = pickle.load(f)
        with open("DIABETIC_average0.pkl", 'rb') as f:
            avg0 = pickle.load(f)
        with open("DIABETIC_average50.pkl", 'rb') as f:
            avg50 = pickle.load(f)
        with open("DIABETIC_average100.pkl", 'rb') as f:
            avg100 = pickle.load(f)

    normalized_x = (x - avg) / avg
    normalized_x0 = (x - avg0) / avg0
    normalized_x50 = (x - avg50) / avg50
    normalized_x100 = (x - avg100) / avg100
    normalized_stacked_x = np.hstack((normalized_x, normalized_x0, normalized_x50, normalized_x100))
    # normalized_stacked_x = np.hstack((normalized_x0, normalized_x50, normalized_x100))

    return normalized_stacked_x


def normalize_per_channel(x_tensor, method):
    """
    Apply per-sample, per-channel min-max normalization.
    Matches NormalizedTensorDataset._normalize_sample() with per_channel_norm=True.
    
    Args:
        x_tensor: torch.Tensor of shape [batch_size, num_channels, seq_length]
    
    Returns:
        normalized tensor of same shape
    """
    
    # Normalize each channel independently
    if method == 'min_max':
        x_min = x_tensor.min(dim=2, keepdim=True)[0]  # Min per channel, shape: (batch, channels, 1)
        x_max = x_tensor.max(dim=2, keepdim=True)[0]  # Max per channel, shape: (batch, channels, 1)
        
        # Avoid division by zero
        x_range = x_max - x_min
        x_range = torch.where(x_range < 1e-8, torch.ones_like(x_range), x_range)
        
        x_normalized = (x_tensor - x_min) / x_range
    elif method == 'snv':
        x_mean = x_tensor.mean(dim=2, keepdim=True)  # shape: (1, 7, 1)
        x_std = x_tensor.std(dim=2, keepdim=True)    # shape: (1, 7, 1)
        safe_std = torch.where(x_std < 1e-8, torch.ones_like(x_std), x_std)
        x_normalized = (x_tensor - x_mean) / safe_std

    return x_normalized


def preprocess_data(measure, reference, dark, cal_data, baseline, use_absorption=False, wavelength_range=None):

    # measure = ast.literal_eval(raw_data['measure'])
    # cal_data = ast.literal_eval(raw_data['cal_data'])
    # dark = ast.literal_eval(raw_data['dark'])
    # reference = ast.literal_eval(raw_data['reference'])

    measure, dark, reference, measure_std, dark_std, reference_std = wavelength_binning(cal_data, measure, dark, reference, 1)

    feature_vector = measure + measure_std + dark + dark_std + reference + reference_std # flat list of numerical features

    # Convert to NumPy arrays
    x = np.array(feature_vector)
    # x = normalize_inputs(x)
    # print(np.shape(x))
    # # y = np.array(glucose_values)
    # x = np.expand_dims(x, 0)
    # print(np.shape(x))

    x = np.expand_dims(x, 0)
    # Apply wavelength range filtering if specified
    if wavelength_range is not None:
        start_nm, end_nm = wavelength_range
        # Assumes data is 420-750nm with 1nm bins (330 total wavelengths)
        # Calculate indices: index = (wavelength - 420)
        start_idx = int(start_nm - 420)
        end_idx = int(end_nm - 420)

        # Validate indices
        if start_idx < 0 or end_idx > 330 or start_idx >= end_idx:
            raise ValueError(f"Invalid wavelength range ({start_nm}, {end_nm}). Must be within 420-750nm.")

        print(f"\nApplying wavelength range filter: {start_nm}-{end_nm}nm (indices {start_idx}-{end_idx})")
        print(f"Original shape: {x.shape}")

        # Slice the wavelength dimension (last dimension)
        x = x[:, :, start_idx:end_idx]

        print(f"Filtered shape: {x.shape}")
        print(f"Using {end_idx - start_idx} wavelengths out of 330")
 
    if use_absorption:
        eps = 1e-8
        num = np.maximum(x[:, 0, :] - x[:, 2, :], eps)
        den = np.maximum(x[:, 4, :] - x[:, 2, :], eps)
        absorption = -np.log10(num/den)
        # print(absorption[0, :])
        absorption = np.expand_dims(absorption, axis=1)
        # print(np.shape(absorption))
        x = np.concatenate([x, absorption], axis=1)
        print(np.shape(x))
    print(np.shape(x))
    device = torch.device("cpu")
    x = torch.from_numpy(x).double().to(device)

    # Apply PER-CHANNEL normalization (matching training)
    x_normalized = normalize_per_channel(x, method='snv')

    return x, x_normalized


def model_inference(model, x):
    """
    Run inference with properly normalized input.
    
    Args:
        model: PyTorch model
        x: Preprocessed and normalized tensor from preprocess_data_corrected()
    
    Returns:
        logits: Raw logits for classification (shape: [1, num_classes])
    """
    model.eval()
    with torch.no_grad():
        logits = model(x)  # Shape: (1, num_classes)

    return logits


def rescale_prediction(y):
    # input_min = 0.0
    # input_max = 1.0
    output_min = 49.0
    output_max = 235.0

    return y * (output_max-output_min) + output_min
