import torch
import numpy as np


def min_max_norm(x):
    """ Performs min-max normalization --> scales values in range between 1 and 0
        :param x (1-D array like)
        :return: (1-D np.array) min-max normalized input
    """
    mini = min(x)
    maxi = max(x)
    return (x - mini) / (maxi - mini)


def min_max_norm_vectorized(tensor):
    mins = tensor.min(dim=1)[0]
    maxs = tensor.max(dim=1)[0]
    return (tensor - mins.unsqueeze(dim=1)) / (maxs - mins).unsqueeze(dim=1)


def min_max_norm_vectorized_np(array):
    mins = array.min(axis=1)
    maxs = array.max(axis=1)
    return (array - mins[:, None]) / (maxs - mins)[:, None]


def standardize(x, mean=None, std=None):
    """ Performs standardization to zero mean and unit standard deviation. If mean and std are passed those will be used
        instead for standardization (might result in a different distribution)
            :param x (torch.Tensor)
            :param mean (float) custom mean offset (bias)
            :param std (float) custom scaling factor
            :return: (torch.Tensor) standardized input
    """
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    return (x - mean) / std


def resize(transit, size=256):
    """ Resizes a 1D-array with linear interpolation
    :param transit: (1-D np.array) input data
    :param size: (int) target size
    :return: (1-D np.array) resized input
    """
    x_original = np.linspace(0, 100, len(transit),  endpoint=True)
    x_target = np.linspace(0, 100, size, endpoint=True)
    return np.interp(x_target, x_original, transit)


def normalize(x, metric='mean'):
    """ Normalizes the input by the specified metric (default:mean)
    :param x: (1D np.array) input to be normalized
    :param metric: (str) metric used for normalization (either mean or median)
    :return: (1D np.array) normalized input
    """
    if metric=='mean':
        x_norm = x/np.median(x)
    elif metric=='median':
        x_norm = x/np.mean(x)
    else:
        raise RuntimeError(f"Unknown metric type {metric}! Please use 'mean' or 'median'.")
    return x_norm


def time_window_binning(flux, old_time, new_time, median=True):
    """ Performs time-window binning based on old and new time (or phase) values
        :param flux: (1D np.array) flux values of transit record
        :param old_time (1D np.array) original time or phase values of transit record
        :param new_time (1D np.array) target time or phase values of transit record
        :return: (1D np.array) aggregated fluxes (gaps filled)
        """
    phase_bin_edges = np.empty(new_time.size + 1)
    half_phase_step = (new_time[1] - new_time[0]) / 2.
    phase_bin_edges[0] = new_time[0]
    phase_bin_edges[1:-1] = new_time[:-1] + half_phase_step
    phase_bin_edges[-1] = new_time[-1]

    new_flux = np.empty(new_time.size)
    metric = np.median if median else np.mean

    cnt2 = 0
    for p0, (p1, p2) in enumerate(zip(phase_bin_edges[:-1], phase_bin_edges[1:])):
        cnt1 = 0
        # skip cadences out of new_time window
        if p0 == 0:
            while old_time[cnt2] < p1:
                cnt2 += 1
        # determine cadences within current bin
        while p1 <= old_time[cnt1 + cnt2] < p2:
            if cnt1 + cnt2 == len(old_time)-1:
                break
            cnt1 += 1
        # select flux within current bin
        bin_slice = flux[cnt2:cnt1 + cnt2]
        # set start point of new bin
        cnt2 += cnt1
        # if no cadences in current bin assign nan
        new_flux[p0] = np.nan if len(bin_slice) == 0 else metric(bin_slice)

    # repeat edge values for trailing nans
    cnt1 = 0
    while np.isnan(new_flux[cnt1]):
        cnt1 += 1
        if cnt1 == new_time.size:
            return np.zeros_like(new_flux)
    if cnt1 > 0:
        new_flux[:cnt1] = new_flux[cnt1]
    cnt2 = len(new_flux) - 1
    while np.isnan(new_flux[cnt2]):
        cnt2 -= 1
    if cnt2 < len(new_flux) - 1:
        new_flux[1 + cnt2:] = new_flux[cnt2]

    # replace remaining nan values with interpolation
    interpolate_values = new_flux[cnt1:cnt2 + 1]
    nan_locations = np.isnan(interpolate_values)
    target_range = np.linspace(0, 100, len(interpolate_values), endpoint=True)
    data_range = np.linspace(0, 100, len(interpolate_values) - sum(nan_locations), endpoint=True)
    new_flux[cnt1:cnt2 + 1] = np.interp(target_range, data_range, interpolate_values[~nan_locations])

    return new_flux


def horizontal_scale_n_shift(model_to_fit, scale_params, shift_params):
    input_size = model_to_fit.size()[-1]
    corrected_models = torch.ones_like(model_to_fit) * model_to_fit.max(dim=-1, keepdims=True).values
    new_sizes = (3. / scale_params * input_size).round().int().tolist()
    shifts = shift_params.int().tolist()
    for s, (new_size, shift) in enumerate(zip(new_sizes, shifts)):
        mid = input_size//2
        if new_size <= 2:
            continue
        resized = torch.nn.functional.interpolate(model_to_fit[s].unsqueeze(dim=0), new_size, mode='linear',
                                                  align_corners=False)
        if new_size > input_size:
            mid_new = new_size // 2
            start = mid_new - mid
            corrected_models[s, :, :] = resized[:, :, start:start+input_size]
        else:
            left_size = new_size//2
            right_size = new_size - left_size
            corrected_models[s, :, mid-left_size:mid+right_size] = resized
        corrected_models[s] = corrected_models[s].roll(shifts=shift, dims=-1)
    return corrected_models