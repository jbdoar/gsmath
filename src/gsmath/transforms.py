"""Pure transformations that return new :class:`~gsmath.Spectrum` objects.

:func:`trim`, :func:`interpolate` and :func:`evaluate` work on any
:class:`~gsmath.Series` -- a spectrum, a line cut, a camera's sensitivities
-- along its axis, every channel alike, keeping per-sample fields (a line
cut's positions) in step.
"""

import numbers

import numpy as np
import scipy.interpolate


def trim(spectrum, xmin, xmax):
    """Select samples within inclusive bounds on the axis (wavelength)."""
    mask = (spectrum.axis >= xmin) & (spectrum.axis <= xmax)
    per_sample = {name: getattr(spectrum, name)[mask] for name in spectrum._PER_SAMPLE
                  if getattr(spectrum, name) is not None}
    return spectrum.copy(
        **{spectrum.AXIS: spectrum.axis[mask]},
        values=spectrum.values[mask],
        label=f"{spectrum.label} [trimmed]",
        **per_sample,
    )


def interpolate(
    spectrum,
    xnew=None,
    *,
    startwl=None,
    stopwl=None,
    step=None,
    spline_order=1,
    extrapolate=False,
):
    """Interpolate onto an explicit or regularly constructed wavelength grid.

    Values outside the measured domain are produced only when ``extrapolate``
    is enabled; callers can use :func:`pad` first to establish explicit edge
    values before resampling.
    """
    if xnew is not None and step is not None:
        raise ValueError("Specify either xnew or step, not both.")

    if spline_order < 1:
        raise ValueError("spline_order must be >= 1.")

    if xnew is None:
        if step is None:
            raise ValueError("Must specify either xnew or step.")

        start = spectrum.axis[0] if startwl is None else startwl
        stop = spectrum.axis[-1] if stopwl is None else stopwl

        xnew = np.arange(start, stop + step, step)

    xnew = np.asarray(xnew, dtype=float)

    if xnew.ndim != 1:
        raise ValueError("xnew must be one-dimensional.")
    if np.any(np.diff(xnew) <= 0):
        raise ValueError("the new wavelengths must be strictly increasing; to look up "
                         "values at wavelengths in any order, call the spectrum: "
                         "spectrum([650, 450])")

    ynew = evaluate(spectrum, xnew, spline_order=spline_order, extrapolate=extrapolate)
    # Per-sample positions (a line cut's x, y) are linear along the axis.
    per_sample = {
        name: scipy.interpolate.make_interp_spline(spectrum.axis, getattr(spectrum, name), k=1)(
            xnew, extrapolate=True)
        for name in spectrum._PER_SAMPLE if getattr(spectrum, name) is not None}

    return spectrum.copy(
        **{spectrum.AXIS: xnew},
        values=ynew,
        label=f"{spectrum.label} [interpolated]",
        **per_sample,
    )


def evaluate(spectrum, wavelength, *, spline_order=1, extrapolate=False):
    """The spectrum's interpolated values at ``wavelength`` (any order, any
    shape), as an array -- with a trailing channel axis for a multi-channel
    series; NaN outside the measured range unless ``extrapolate``."""
    if spline_order < 1:
        raise ValueError("spline_order must be >= 1.")
    spline = scipy.interpolate.make_interp_spline(spectrum.axis, spectrum.values,
                                                  k=spline_order)
    return spline(np.asarray(wavelength, dtype=float), extrapolate=extrapolate)


def pad(spectrum, mode="zero", *, value=None):
    """Fill non-finite values outside the finite data span."""
    mode = mode.lower()
    values = spectrum.values.copy()
    valid = np.isfinite(values)

    if mode not in {"zero", "constant", "edge"}:
        raise ValueError(
            f"Unknown padding mode: {mode!r}. "
            "Expected 'zero', 'constant', or 'edge'."
        )

    if mode == "constant":
        if value is None:
            raise ValueError(
                "value must be provided when mode='constant'."
            )

        if not isinstance(value, numbers.Real):
            raise TypeError("value must be a real number.")

    if not np.any(valid):
        if mode == "edge":
            raise ValueError(
                "Cannot edge-pad a spectrum with no finite values."
            )

        fill = 0.0 if mode == "zero" else value
        values[:] = fill

    else:
        finite_indices = np.flatnonzero(valid)
        first = finite_indices[0]
        last = finite_indices[-1]

        if mode == "zero":
            left_fill = right_fill = 0.0
        elif mode == "constant":
            left_fill = right_fill = value
        else:
            left_fill = values[first]
            right_fill = values[last]

        values[:first] = left_fill
        values[last + 1:] = right_fill

    return spectrum.copy(
        values=values,
        label=f"{spectrum.label} [{mode} padded]",
    )


def normalize(spectrum, by="peak", value=1.0):
    """Scale a spectrum so the selected measure equals ``value``."""
    if not isinstance(value, numbers.Real):
        raise TypeError("value must be a real number.")

    if spectrum.values.size == 0:
        raise ValueError("Cannot normalize an empty spectrum.")

    if isinstance(by, str):
        key = by.lower()

        if key == "peak":
            if not np.any(np.isfinite(spectrum.values)):
                raise ValueError(
                    "Cannot normalize a spectrum with no finite values."
                )

            current = np.nanmax(spectrum.values)

        elif key in {"integral", "integrated", "integrated_power"}:
            current = spectrum.power().integral

        elif key in {"y", "photometric"}:
            current = spectrum.color().Y

        else:
            raise ValueError(
                f"Unknown normalization mode: {by!r}"
            )

    elif isinstance(by, numbers.Real):
        wavelength = float(by)

        if not (
            spectrum.wavelength[0]
            <= wavelength
            <= spectrum.wavelength[-1]
        ):
            raise ValueError(
                "Normalization wavelength is outside "
                "the spectrum range."
            )

        current = np.interp(
            wavelength,
            spectrum.wavelength,
            spectrum.values,
        )

    else:
        raise TypeError(
            "by must be a normalization mode string "
            "or wavelength."
        )

    if not np.isfinite(current):
        raise ValueError(
            "Normalization quantity is not finite."
        )

    if current == 0:
        raise ValueError("Cannot normalize by zero.")

    scale = value / current

    return spectrum.copy(
        values=spectrum.values * scale,
        label=f"{spectrum.label} [{by} normalized]",
    )


def calibrate_wavelength(spectrum, calibration):
    """Apply a fitted pixel solution or correct the existing wavelength axis.

    WavelengthCalibration uses sample indices and outputs nm. Legacy callables
    and ascending-order coefficients operate on the existing wavelength axis.
    """
    from .wavelength import WavelengthCalibration
    if isinstance(calibration, WavelengthCalibration):
        return calibration.apply(spectrum)
    if callable(calibration):
        wavelength = calibration(spectrum.wavelength)
    else:
        coefficients = np.asarray(calibration, dtype=float)
        wavelength = np.polynomial.polynomial.polyval(
            spectrum.wavelength,
            coefficients,
        )

    wavelength = np.asarray(wavelength, dtype=float)

    if wavelength.shape != spectrum.wavelength.shape:
        raise ValueError("Calibration must preserve wavelength array shape.")

    if np.any(np.diff(wavelength) <= 0):
        raise ValueError("Corrected wavelength must be strictly increasing.")

    return spectrum.copy(
        wavelength=wavelength,
        label=f"{spectrum.label} [wavelength calibrated]",
    )


def shift_wavelength(spectrum, shift):
    """Translate the wavelength axis without changing spectral values."""
    return calibrate_wavelength(
        spectrum,
        [shift, 1.0],
    )


TAIL_TRIMS = ("first_nonpositive", "none")


def trim_tails(spectrum, how="first_nonpositive"):
    """Zero a spectrum's tails, keeping its length and grid.

    ``"first_nonpositive"``: walking outward from the (highest) peak, zero
    everything from the first value <= 0 on, on each side -- so noise beyond
    an emission band goes, even where it rises above zero again. ``"none"``:
    only negative values become 0. Non-finite values count as 0.
    """
    if how not in TAIL_TRIMS:
        raise ValueError(f"Unknown tail trim: {how!r}. Expected one of {TAIL_TRIMS}.")

    values = np.nan_to_num(np.asarray(spectrum.values, dtype=float))

    if how == "first_nonpositive" and values.size and values.max() > 0:
        peak = int(np.argmax(values))
        below = np.flatnonzero(values[:peak] <= 0)
        if below.size:
            values[:below[-1] + 1] = 0.0
        above = np.flatnonzero(values[peak:] <= 0)
        if above.size:
            values[peak + above[0]:] = 0.0

    return spectrum.copy(
        values=np.clip(values, 0.0, None),
        label=f"{spectrum.label} [tails trimmed]",
    )


def effective_range(spectrum, threshold=0.0):
    """The effective wavelength range: the first and last wavelengths whose
    value exceeds ``threshold``, relative to the peak (0.01: above 1% of
    the peak; the default 0: nonzero), or None if nothing is positive."""
    values = np.nan_to_num(np.asarray(spectrum.values, dtype=float))
    peak = values.max() if values.size else 0.0
    if not peak > 0:
        return None
    on = np.flatnonzero(values > threshold * peak)
    if not on.size:  # threshold >= 1: the peak itself
        on = np.flatnonzero(values >= peak)
    return float(spectrum.wavelength[on[0]]), float(spectrum.wavelength[on[-1]])
