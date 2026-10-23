"""Smoothing filters exposed through :meth:`gsmath.Spectrum.smooth`."""

import numbers

import numpy as np
import scipy.ndimage
import scipy.signal


def smooth(spectrum, method="savgol", **kwargs):
    """Dispatch to a named filter and return a smoothed spectrum."""
    method = method.lower()

    methods = {
        "savgol": smooth_savgol,
        "savitzky_golay": smooth_savgol,
        "gaussian": smooth_gaussian,
        "moving_average": smooth_moving_average,
        "boxcar": smooth_moving_average,
        "median": smooth_median,
        "median_filter": smooth_median,
    }

    try:
        smoother = methods[method]
    except KeyError:
        raise ValueError(f"Unknown smoothing method: {method!r}") from None

    values = smoother(spectrum.values, **kwargs)

    return spectrum.copy(
        values=values,
        label=f"{spectrum.label} [{method} smoothed]",
    )


def smooth_savgol(y, window_length=11, polyorder=2, **kwargs):
    """Apply a Savitzky-Golay polynomial filter."""
    return scipy.signal.savgol_filter(
        y,
        window_length=window_length,
        polyorder=polyorder,
        **kwargs,
    )


def smooth_gaussian(y, sigma=1.0, **kwargs):
    """Apply a one-dimensional Gaussian filter."""
    return scipy.ndimage.gaussian_filter1d(
        y,
        sigma=sigma,
        **kwargs,
    )


def smooth_moving_average(y, window=5):
    """Apply a centered moving average."""
    _validate_window(window)

    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def smooth_median(y, window=5, **kwargs):
    """Apply a median filter with an odd window."""
    _validate_window(window, odd=True)

    return scipy.ndimage.median_filter(
        y,
        size=window,
        **kwargs,
    )


def _validate_window(window, *, odd=False):
    """Validate filter-window size and optional oddness."""
    if not isinstance(window, numbers.Integral):
        raise TypeError("window must be an integer.")

    if window < 1:
        raise ValueError("window must be >= 1.")

    if odd and window % 2 == 0:
        raise ValueError("window must be odd.")
