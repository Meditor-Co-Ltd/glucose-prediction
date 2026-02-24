import numpy as np
import torch


def load_config(path='configuration.txt'):
    """
    Parse a configuration.txt file and return a dict of settings.
    Handles key = value lines; casts booleans, ints, and floats automatically.
    """
    config = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('='):
                continue
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if '#' in value:
                value = value[:value.index('#')].strip()
            if value.lower() == 'true':
                config[key] = True
            elif value.lower() == 'false':
                config[key] = False
            else:
                try:
                    config[key] = int(value)
                except ValueError:
                    try:
                        config[key] = float(value)
                    except ValueError:
                        config[key] = value
    return config


def load_model(path):
    model = torch.jit.load(path, map_location='cpu')
    model.eval()
    model.to(torch.device('cpu'))
    print(f"Model loaded from {path}")
    return model


def wavelength_binning(cal_data, measure, dark, reference, bin_size=20):
    cal_data  = np.asarray(cal_data)
    measure   = np.asarray(measure)
    dark      = np.asarray(dark)
    reference = np.asarray(reference)
    measure_binned   = []
    dark_binned      = []
    reference_binned = []
    measure_binned2  = []
    dark_binned2     = []
    reference_binned2 = []
    bins = range(420, 750, bin_size)
    for b in bins:
        indices = np.argwhere((cal_data >= b) & (cal_data <= b + bin_size))
        measure_binned.append(np.nanmean(measure[indices]))
        dark_binned.append(np.nanmean(dark[indices]))
        reference_binned.append(np.nanmean(reference[indices]))
        measure_binned2.append(np.nanstd(measure[indices]))
        dark_binned2.append(np.nanstd(dark[indices]))
        reference_binned2.append(np.nanstd(reference[indices]))
    return (
        [np.hstack(measure_binned)],
        [np.hstack(dark_binned)],
        [np.hstack(reference_binned)],
        [np.hstack(measure_binned2)],
        [np.hstack(dark_binned2)],
        [np.hstack(reference_binned2)],
    )


def add_spectral_derivatives(x_np, derivative_order=1):
    """
    Append spectral derivative channels along axis=1.
    Mirrors add_spectral_derivatives() from the research training code.

    Args:
        x_np: numpy array [batch, channels, wavelengths]
        derivative_order: 1 = first derivative only, 2 = first and second

    Returns:
        numpy array with derivative channels appended
    """
    first_deriv = np.gradient(x_np, axis=2)
    x_np = np.concatenate([x_np, first_deriv], axis=1)
    if derivative_order >= 2:
        second_deriv = np.gradient(first_deriv, axis=2)
        x_np = np.concatenate([x_np, second_deriv], axis=1)
    return x_np


def apply_per_channel_snv(x_tensor):
    """
    Apply per-channel SNV normalization and concatenate with the original.
    Matches NormalizedTensorDataset._normalize_sample(norm_method='snv', per_channel_norm=True).

    Args:
        x_tensor: torch.Tensor [batch, channels, wavelengths]

    Returns:
        torch.Tensor [batch, channels*2, wavelengths]
    """
    mean     = x_tensor.mean(dim=2, keepdim=True)
    std      = x_tensor.std(dim=2, keepdim=True)
    safe_std = torch.where(std < 1e-8, torch.ones_like(std), std)
    x_snv    = (x_tensor - mean) / safe_std
    return torch.cat([x_tensor, x_snv], dim=1)


def preprocess_for_inference(measure, reference, dark, cal_data, config):
    """
    Preprocess raw sensor data for model inference, driven by a config dict.

    Channel order (USE_SIGNAL_STD=True):
        0: measure, 1: measure_std, 2: dark, 3: dark_std, 4: reference, 5: reference_std,
        [6: absorption, 7: absorption_std  if USE_ABSORPTION]
    Channel order (USE_SIGNAL_STD=False):
        0: measure, 1: dark, 2: reference, [3: absorption  if USE_ABSORPTION]

    Pipeline matches the research training:
        1. Wavelength binning (1nm bins, means only)
        2. Stack [measure, dark, reference] -> [1, 3, N_wl]
        3. Filter to configured wavelength range
        4. Append absorption channel if USE_ABSORPTION=True
        5. Add spectral derivatives if ADD_SPECTRAL_DERIVATIVES=True
        6. Convert to torch.double tensor
        7. Apply per-channel SNV + concat if PER_CHANNEL_NORM=True

    Returns:
        torch.Tensor ready for model inference, e.g. [1, 16, 200]
    """
    use_absorption   = config.get('USE_ABSORPTION', True)
    use_signal_std   = config.get('USE_SIGNAL_STD', True)
    wl_start         = int(config.get('wavelength_nm_start', 450))
    wl_end           = int(config.get('wavelength_nm_end', 650))
    add_deriv        = config.get('ADD_SPECTRAL_DERIVATIVES', False)
    deriv_order      = int(config.get('DERIVATIVE_ORDER', 1))
    per_channel_norm = config.get('PER_CHANNEL_NORM', False)

    # Step 1: wavelength binning (means + stds)
    measure_b, dark_b, reference_b, measure_std_b, dark_std_b, reference_std_b = wavelength_binning(
        cal_data, measure, dark, reference, bin_size=1
    )
    measure_b       = np.array(measure_b[0])
    dark_b          = np.array(dark_b[0])
    reference_b     = np.array(reference_b[0])
    measure_std_b   = np.array(measure_std_b[0])
    dark_std_b      = np.array(dark_std_b[0])
    reference_std_b = np.array(reference_std_b[0])

    # Step 2: stack channels
    # With stds: [measure, measure_std, dark, dark_std, reference, reference_std] -> [1, 6, N_wl]
    # Without:   [measure, dark, reference]                                        -> [1, 3, N_wl]
    if use_signal_std:
        x = np.stack([measure_b, measure_std_b, dark_b, dark_std_b, reference_b, reference_std_b], axis=0)
    else:
        x = np.stack([measure_b, dark_b, reference_b], axis=0)
    x = np.expand_dims(x, axis=0)

    # Step 3: wavelength range filter
    start_idx = int(wl_start - 420)
    end_idx   = int(wl_end - 420)
    if start_idx < 0 or end_idx > x.shape[2] or start_idx >= end_idx:
        raise ValueError(f"Invalid wavelength range ({wl_start}, {wl_end}). Must be within 420-750nm.")
    x = x[:, :, start_idx:end_idx]

    # Step 4: absorption channel
    # With stds: channels are 0=measure, 1=measure_std, 2=dark, 3=dark_std, 4=reference, 5=reference_std
    # Without:   channels are 0=measure, 1=dark, 2=reference
    if use_absorption:
        eps = 1e-8
        if use_signal_std:
            m, sigma_m = x[:, 0, :], x[:, 1, :]
            d, sigma_d = x[:, 2, :], x[:, 3, :]
            r, sigma_r = x[:, 4, :], x[:, 5, :]
        else:
            m, d, r = x[:, 0, :], x[:, 1, :], x[:, 2, :]
            sigma_m = sigma_d = sigma_r = np.zeros_like(m)

        num = np.maximum(m - d, eps)
        den = np.maximum(r - d, eps)
        absorption = -np.log10(num / den)
        absorption = np.expand_dims(absorption, axis=1)
        x = np.concatenate([x, absorption], axis=1)

        if use_signal_std:
            # Absorption std via error propagation:
            # absorption = -log10(num/den), num = m-d, den = r-d
            ln10 = np.log(10)
            abs_std = np.sqrt(
                (sigma_m / (num * ln10)) ** 2 +
                ((den - num) * sigma_d / (num * den * ln10)) ** 2 +
                (sigma_r / (den * ln10)) ** 2
            )
            abs_std = np.expand_dims(abs_std, axis=1)
            x = np.concatenate([x, abs_std], axis=1)

    # Step 5: spectral derivatives
    if add_deriv:
        x = add_spectral_derivatives(x, deriv_order)

    # Step 6: to double tensor
    x_tensor = torch.from_numpy(x).double().to(torch.device('cpu'))

    # Step 7: per-channel SNV + concat
    if per_channel_norm:
        x_tensor = apply_per_channel_snv(x_tensor)

    print(f"Preprocessed tensor shape: {x_tensor.shape}")
    return x_tensor


def regression_inference(model, x):
    """
    Run inference with a probabilistic regression model.

    Returns:
        (mu, log_var): both torch.Tensor of shape [1, 1]
    """
    model.eval()
    with torch.no_grad():
        mu, log_var = model(x)
    return mu, log_var
