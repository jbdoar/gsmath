"""Cleaning raw spectrometer data: spikes out, noise down where the signal is weak.

Two calibration steps for raw spectra (counts, after the dark is subtracted,
before anything scales the noise unevenly -- stray light, responsivity):

- :class:`Despike` replaces one- or two-pixel excursions that can't be light
  -- anything narrower than the instrument's resolution -- and reports each
  one (pixel, wavelength, size in counts and in noise sigmas).
- :class:`AdaptiveSmooth` smooths where the signal is weak and leaves it
  alone where it's strong: a Savitzky-Golay fit blended with the data by
  local signal-to-noise, so a narrow peak keeps its shape while its tails
  lose their noise.

Both judge against the *local* noise, estimated robustly from how far each
pixel sits from a cubic through its neighbours (i-2, i-1, i+1, i+2) -- which
follows a smooth peak closely, so a peak's top isn't mistaken for a spike.

    cleaned = calibration.with_steps(after="SubtractDark",
                                     steps=[Despike(), AdaptiveSmooth()])
    spectrum = measurement.process(cleaned)
    spectrum.metadata["processing"]["reports"]["Despike"]   # the spikes
"""

from __future__ import annotations

import numpy as np

from .calibration import Step

# A pixel's residual from the cubic through its neighbours, for white noise
# of sigma s: sqrt(1 + (1 + 16 + 16 + 1) / 36) * s.
_RESIDUAL_PER_SIGMA = float(np.sqrt(1 + 34 / 36))


def _lagrange_weights(nodes, at):
    """Weights giving the cubic through `nodes` (offsets) at offset `at`."""
    nodes = np.asarray(nodes, dtype=float)
    return np.array([np.prod([(at - m) / (n - m) for m in nodes if m != n]) for n in nodes])


# A pair i, i+1 predicted from i-2, i-1, i+2, i+3.
_PAIR_WEIGHTS = np.array([_lagrange_weights((-2, -1, 2, 3), at) for at in (0, 1)])
_PAIR_RESIDUAL_PER_SIGMA = float(np.sqrt(1 + np.sum(_PAIR_WEIGHTS[0] ** 2)))


def neighbour_prediction(values: np.ndarray) -> np.ndarray:
    """Each pixel predicted by the cubic through i-2, i-1, i+1, i+2 (the two
    pixels at each end predicted as themselves)."""
    v = np.asarray(values, dtype=float)
    prediction = v.copy()
    prediction[2:-2] = (-v[:-4] + 4 * v[1:-3] + 4 * v[3:-1] - v[4:]) / 6
    return prediction


def local_noise(values: np.ndarray, window: int = 31) -> np.ndarray:
    """Per-pixel noise sigma: the robust spread (1.4826 x median absolute
    deviation) of the neighbour residuals over a `window` around each pixel."""
    residual = np.asarray(values, dtype=float) - neighbour_prediction(values)
    half = window // 2
    padded = np.pad(np.abs(residual), half, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    mad = np.median(windows, axis=1)
    return np.maximum(1.4826 * mad / _RESIDUAL_PER_SIGMA, np.finfo(float).tiny)


class Despike(Step):
    """Replace excursions of one or two pixels more than `threshold` local
    noise sigmas -- and more than `relative` of the local signal -- from what
    their neighbours predict, by that prediction: narrower than any real
    spectral feature. Reports each one.

    `relative` keeps the top of a narrow, bright line (where even a cubic
    misses by a fraction of a percent, many shot-noise sigmas) from passing
    for a spike."""

    def __init__(self, threshold: float = 10.0, window: int = 31, relative: float = 0.02, *,
                 optional: bool = False):
        super().__init__(optional=optional)
        self.threshold, self.window, self.relative = float(threshold), int(window), float(relative)

    def apply(self, data, measurement):
        values = np.asarray(data.values, dtype=float)
        cleaned, spikes = despike(values, self.threshold, self.window, self.relative)
        wavelength = np.asarray(data.wavelength, dtype=float)
        report = [{"pixel": int(i), "wavelength": round(float(wavelength[i]), 3),
                   "value": float(values[i]), "replaced_by": float(cleaned[i]),
                   "sigmas": round(float(z), 1)} for i, z in spikes]
        metadata = dict(getattr(data, "metadata", {}) or {})
        metadata["step_report"] = report
        return data.copy(values=cleaned, metadata=metadata)

    def parameters(self):  # predates the default (int window): kept for saved fingerprints
        return {"threshold": self.threshold, "window": self.window, "relative": self.relative}

    def arguments(self):
        return {"threshold": self.threshold, "window": self.window, "relative": self.relative}


def despike(values: np.ndarray, threshold: float = 10.0, window: int = 31,
            relative: float = 0.02, max_spikes: int = 50
            ) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """(cleaned values, [(pixel, sigmas)]) -- see :class:`Despike`.

    Worst first: a pixel beyond `threshold` noise sigmas (and `relative` of
    the signal) from the cubic through its neighbours, or a pair of pixels
    both beyond it, the same way, from the cubic through the pair's outer
    neighbours -- whichever stands out more is replaced, and the rest looked
    at again: a spike pulls its neighbours' predictions, so they're only
    judged once it's gone."""
    v = np.asarray(values, dtype=float)
    cleaned = v.copy()
    if v.size < 7:
        return cleaned, []
    sigma = local_noise(v, window)
    spikes = []
    for _ in range(max_spikes):
        prediction = neighbour_prediction(cleaned)
        residual = cleaned - prediction
        z = residual / (sigma * _RESIDUAL_PER_SIGMA)
        z[:2] = z[-2:] = 0  # no neighbours on one side
        single = (np.abs(z) > threshold) & (np.abs(residual) > relative * np.abs(prediction))
        pair_prediction, pair_z = _pairs(cleaned, sigma)
        both = np.stack([cleaned[2:-3], cleaned[3:-2]], axis=1)
        big = np.abs(both - pair_prediction) > relative * np.abs(pair_prediction)
        pair = (np.all(np.abs(pair_z) > threshold, axis=1) & np.all(big, axis=1)
                & (pair_z[:, 0] * pair_z[:, 1] > 0))
        best_single = np.max(np.abs(z[single])) if single.any() else 0.0
        best_pair = np.max(np.min(np.abs(pair_z[pair]), axis=1)) if pair.any() else 0.0
        if best_single == 0.0 and best_pair == 0.0:
            break
        if best_pair > best_single:
            k = int(np.flatnonzero(pair)[np.argmax(np.min(np.abs(pair_z[pair]), axis=1))])
            start = k + 2
            cleaned[start:start + 2] = pair_prediction[k]
            spikes += [(start, float(pair_z[k, 0])), (start + 1, float(pair_z[k, 1]))]
        else:
            i = int(np.flatnonzero(single)[np.argmax(np.abs(z[single]))])
            cleaned[i] = prediction[i]
            spikes.append((i, float(z[i])))
    return cleaned, sorted(spikes)


def _pairs(v, sigma):
    """For each pair (i, i+1), i = 2 .. n-4: the pair predicted by the cubic
    through i-2, i-1, i+2, i+3, and each pixel's excess in noise sigmas."""
    neighbours = np.stack([v[:-5], v[1:-4], v[4:-1], v[5:]], axis=1)
    prediction = neighbours @ _PAIR_WEIGHTS.T
    both = np.stack([v[2:-3], v[3:-2]], axis=1)
    noise = np.stack([sigma[2:-3], sigma[3:-2]], axis=1) * _PAIR_RESIDUAL_PER_SIGMA
    return prediction, (both - prediction) / noise


class AdaptiveSmooth(Step):
    """Savitzky-Golay smoothing (`window` samples, `order`) blended with the
    data by local signal-to-noise: weight 1 / (1 + (snr / `snr`)^2) on the
    smoothed values, so where signal is `snr` noise sigmas the two count
    equally, well above it the data stand, well below it they're smoothed."""

    def __init__(self, window: int = 11, order: int = 3, snr: float = 30.0, *,
                 optional: bool = False):
        super().__init__(optional=optional)
        self.window, self.order, self.snr = int(window), int(order), float(snr)

    def apply(self, data, measurement):
        values = np.asarray(data.values, dtype=float)
        smoothed, weight = adaptive_smooth(values, self.window, self.order, self.snr)
        metadata = dict(getattr(data, "metadata", {}) or {})
        metadata["step_report"] = {"mean_weight": round(float(weight.mean()), 4)}
        return data.copy(values=smoothed, metadata=metadata)

    def parameters(self):  # predates the default (int window): kept for saved fingerprints
        return {"window": self.window, "order": self.order, "snr": self.snr}

    def arguments(self):
        return {"window": self.window, "order": self.order, "snr": self.snr}


def adaptive_smooth(values: np.ndarray, window: int = 11, order: int = 3, snr: float = 30.0
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(smoothed values, the weight given the smoothing at each pixel)."""
    from scipy.signal import savgol_filter

    v = np.asarray(values, dtype=float)
    if v.size <= window:
        return v.copy(), np.zeros_like(v)
    smooth = savgol_filter(v, window, order)
    local_snr = np.abs(smooth) / local_noise(v)
    weight = 1.0 / (1.0 + (local_snr / snr) ** 2)
    return weight * smooth + (1 - weight) * v, weight
