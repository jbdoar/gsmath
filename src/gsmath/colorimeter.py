"""Turning a camera into an imaging colorimeter: its spectral sensitivity,
and the colour matrix from its channels to CIE X, Y, Z.

A camera's channel signals are its spectral sensitivities integrated
against the light; X, Y, Z are the colour-matching functions integrated
against the same light. Unless the sensitivities happen to be a linear mix
of the colour-matching functions (the Luther condition), no matrix maps one
to the other exactly for every spectrum -- so the matrix is fitted on the
kinds of light the camera will measure, and checked on others.

Two ways to get the (camera signal, X Y Z) pairs to fit:

- **Spectrally.** Measure the camera's :class:`SpectralSensitivity` with a
  monochromator (:meth:`SpectralSensitivity.from_monochromator`); then any
  known spectrum -- a calibrated tunable source's settings, a display's
  primaries -- gives its camera signal by :meth:`~SpectralSensitivity.response`
  and its X, Y, Z by ``spectrum.color().XYZ``, without imaging it.
- **By measurement.** Image patches the reference instrument (a
  spectroradiometer) also measures: the camera signal is each patch's mean
  (:func:`gsmath.mosaic.site_means` straight from a raw mosaic), the X, Y, Z
  the instrument's.

What the matrix is fitted on matters more than how: fit it on the kind of
light it will measure. A display's colours are all mixes of its three
primaries, so a matrix fitted on that display's red, green, blue (and white)
is exact for every colour it shows. For a tunable source, fit on mixes of its
own channels. No single matrix measures arbitrary narrow-band light well;
that needs more channels, such as a filter-wheel camera.

Either way, :func:`fit_color_matrix` fits the matrix and reports how well it
does; :meth:`ColorMatrixFit.validate` checks it on held-out samples, and
:meth:`ColorMatrixFit.step` puts it in a calibration::

    sensitivity = SpectralSensitivity.from_monochromator(wl, signals, band_radiance)
    training = [source.spectrum(setting) for setting in settings]
    fit = fit_color_matrix(sensitivity.responses(training),
                           [s.color().XYZ for s in training],
                           method="uv_Y", unit=...)
    fit.validate(sensitivity.responses(held_out), [s.color().XYZ for s in held_out])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from . import units
from .analysis.base import Result
from .series import Series


class SpectralSensitivity(Series):
    """A camera's channel sensitivities on one wavelength grid: a
    :class:`~gsmath.Series` with values ``(wavelengths, channels)``, the
    signal per unit of light at each wavelength."""
    AXIS = "wavelength"
    AXIS_LABEL = "Wavelength"
    VALUE_LABEL = "Sensitivity"
    _FIELDS = ("wavelength", "values", "channels", "value_unit", "wavelength_unit", "label",
               "metadata")
    _REMOVABLE = {"wavelength_unit", "value_unit"}

    def __init__(self, wavelength, values, channels=("R", "G", "B"), value_unit=None,
                 wavelength_unit="nm", label="Spectral sensitivity", metadata=None):
        wavelength = np.asarray(wavelength, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if wavelength.ndim != 1 or values.shape[0] != wavelength.size:
            raise ValueError("values must have one row per wavelength")
        if np.any(np.diff(wavelength) <= 0):
            raise ValueError("wavelengths must increase")
        channels = (channels,) if isinstance(channels, str) else tuple(channels)
        if len(channels) != values.shape[1]:
            raise ValueError(f"{len(channels)} channel names for {values.shape[1]} columns")
        self._set_series(wavelength, values, label, wavelength_unit, value_unit, channels,
                         metadata)
        self._seal()

    @classmethod
    def from_monochromator(cls, wavelength, signal, reference, *, channels=("R", "G", "B"),
                           signal_unit=None, reference_unit=None, wavelength_unit="nm",
                           label="Spectral sensitivity"):
        """Sensitivity from a monochromator scan: at each ``wavelength``, the
        camera's ``signal`` per channel (``(n, channels)``, dark-subtracted,
        e.g. counts/ms) over the ``reference`` -- the band's radiance (or
        irradiance) from a calibrated detector, integrated over the band.
        Responses to spectral radiance then come out in the signal's unit.
        (The band's width isn't deconvolved: keep it narrow next to the
        sensitivities' features.)"""
        signal = np.asarray(signal, dtype=float)
        reference = np.asarray(reference, dtype=float)
        if signal.ndim == 1:
            signal = signal[:, None]
        if reference.shape != (signal.shape[0],):
            raise ValueError("one reference value per wavelength")
        if np.any(reference <= 0):
            raise ValueError("reference values must be positive")
        unit = None
        if signal_unit is not None or reference_unit is not None:
            unit = units.divide_units(signal_unit, reference_unit)
        return cls(wavelength, signal / reference[:, None], channels, unit, wavelength_unit,
                   label, {"source": "monochromator"})

    @classmethod
    def from_spectra(cls, spectra, channels=None, *, label="Spectral sensitivity"):
        """Sensitivity from one Spectrum per channel, all on one grid."""
        spectra = list(spectra)
        first = spectra[0]
        for spectrum in spectra[1:]:
            if not np.array_equal(spectrum.wavelength, first.wavelength) \
                    or spectrum.wavelength_unit != first.wavelength_unit \
                    or spectrum.value_unit != first.value_unit:
                raise ValueError("channel spectra must share their grid and units")
        names = channels if channels is not None else tuple(s.label for s in spectra)
        return cls(first.wavelength, np.column_stack([s.values for s in spectra]), names,
                   first.value_unit, first.wavelength_unit, label)

    def channel(self, name):
        """One channel's sensitivity as a Spectrum (a list of names: a
        SpectralSensitivity of those channels)."""
        from .spectrum import Spectrum

        if not isinstance(name, str):
            names = list(name)
            return SpectralSensitivity(self.wavelength, self.values[:, [self.channels.index(n)
                                                                        for n in names]],
                                       names, self.value_unit, self.wavelength_unit, self.label,
                                       self.metadata)
        i = self.channels.index(name)
        return Spectrum(self.wavelength, self.values[:, i], label=name,
                        wavelength_unit=self.wavelength_unit, value_unit=self.value_unit)

    def response(self, spectrum):
        """The camera's signal per channel for light of ``spectrum`` (a
        spectral density), as a quantity array (unitless if either has no
        unit): the sensitivity integrated against it, which is taken as 0
        outside its wavelength range."""
        wavelength = units.convert_values(spectrum.wavelength, spectrum.wavelength_unit,
                                          self.wavelength_unit)
        light = np.interp(self.wavelength, wavelength, np.asarray(spectrum.values, dtype=float),
                          left=0.0, right=0.0)
        signal = np.trapezoid(self.values * light[:, None], self.wavelength, axis=0)
        unit = self.response_unit(spectrum)
        return signal if unit is None else units.Q_(signal, unit)

    def responses(self, spectra):
        """``(spectra, channels)`` responses to several spectra."""
        found = [self.response(s) for s in spectra]
        if found and units.is_quantity(found[0]):
            unit = found[0].units
            return units.Q_(np.array([f.to(unit).magnitude for f in found]), unit)
        return np.array(found)

    def response_unit(self, spectrum):
        if self.value_unit is None or spectrum.value_unit is None:
            return None
        return units.multiply_units(units.multiply_units(self.value_unit, spectrum.value_unit),
                                    self.wavelength_unit)

    def normalized(self):
        """Each channel scaled to a peak of 1 (unitless)."""
        return SpectralSensitivity(self.wavelength, self.values / self.values.max(axis=0),
                                   self.channels, None, self.wavelength_unit, self.label,
                                   dict(self.metadata))


FIT_METHODS = ("xyz", "uv_Y", "delta_e2000")


@dataclass(frozen=True, eq=False)
class ColorMatrixFit(Result):
    """A fitted colour matrix and how well it reproduces its training set.

    Works as the matrix itself (``numpy.asarray(fit)``), e.g. in
    :meth:`Image.apply_matrix <gsmath.Image.apply_matrix>`."""
    matrix: np.ndarray
    method: str
    measured: np.ndarray
    """The training camera signals, ``(samples, channels)``."""
    reference: np.ndarray
    """The training X, Y, Z, ``(samples, 3)``."""
    matrix_unit: object = None
    """The matrix's unit: X, Y, Z's per camera signal's."""
    white: np.ndarray | None = None
    """The reference white's X, Y, Z for CIEDE2000 (on the samples' scale)."""
    channels: tuple = ("X", "Y", "Z")

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.matrix, dtype=dtype)

    def predict(self, measured):
        """X, Y, Z from camera signals ``(samples, channels)``."""
        return _magnitude(measured) @ self.matrix.T

    def errors(self, measured, reference) -> dict:
        """Per-sample errors of the prediction against ``reference`` X, Y, Z:
        ``delta_uv_prime``, ``delta_xy``, ``delta_Y`` (relative), and
        ``delta_e2000`` if the fit has a white."""
        return _errors(self.predict(measured), _magnitude(reference), self.white)

    @cached_property
    def training_errors(self) -> dict:
        return self.errors(self.measured, self.reference)

    def validate(self, measured, reference) -> dict:
        """Mean, RMS and maximum of each error over held-out samples."""
        return _report(self.errors(measured, reference))

    @property
    def units(self):
        """The matrix's unit; the fit's errors are dimensionless (ΔY relative)."""
        return {"matrix": self.matrix_unit, "samples": units.ureg.dimensionless,
                "training": units.ureg.dimensionless}

    @property
    def summary(self) -> dict:
        return {"method": self.method, "samples": len(self.measured),
                "matrix": self.matrix.tolist(), "unit": units.format_unit(self.matrix_unit),
                "training": _report(self.training_errors)}

    def step(self, *, optional=False):
        """This matrix as a :class:`~gsmath.ColorMatrix` calibration step."""
        from .calibration import ColorMatrix
        return ColorMatrix(self.matrix, channels=self.channels, unit=self.matrix_unit, optional=optional)


def fit_color_matrix(measured, reference, *, method="xyz", weighting="relative", weights=None,
                     preserve=None, white=None, tolerance=(0.001, 0.01), unit=None):
    """Fit the matrix ``M`` taking camera signals to X, Y, Z (``M @ signal``).

    ``measured`` is ``(samples, channels)`` camera signals, ``reference``
    ``(samples, 3)`` X, Y, Z (arrays or quantity arrays; with quantities the
    matrix's ``unit`` follows). At least as many samples as channels, of
    distinct colours. ``method``:

    - ``"xyz"`` -- linear least squares on X, Y, Z. ``weighting``
      ``"relative"`` (default) divides each sample by its X + Y + Z, so dim
      and bright samples count alike (dividing by Y alone would let deep
      violet and red samples, with almost no luminance, dominate the fit);
      ``"absolute"`` doesn't; ``weights`` multiplies in more. ``preserve``: a sample index whose X, Y, Z must come out exactly
      (a white).
    - ``"uv_Y"`` -- minimise chromaticity and luminance error together:
      Δu'v' / ``tolerance[0]`` and relative ΔY / ``tolerance[1]``, from the
      ``"xyz"`` fit as a start.
    - ``"delta_e2000"`` -- minimise CIEDE2000 against ``white``.

    Returns a :class:`ColorMatrixFit`.
    """
    from scipy.optimize import least_squares

    measured_unit, reference_unit = _unit_of(measured), _unit_of(reference)
    if unit is None and (measured_unit is not None or reference_unit is not None):
        unit = units.divide_units(reference_unit, measured_unit)
    measured, reference = _magnitude(measured), _magnitude(reference)
    if measured.ndim != 2 or reference.ndim != 2 or reference.shape[1] != 3 \
            or measured.shape[0] != reference.shape[0]:
        raise ValueError("measured and reference must both be (samples, channels), "
                         "with X, Y, Z as reference's channels")
    if method not in FIT_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {FIT_METHODS}")
    if weighting not in ("relative", "absolute"):
        raise ValueError("weighting must be 'relative' or 'absolute'")
    samples, channels = measured.shape
    if samples < channels:
        raise ValueError(f"fitting a matrix needs at least {channels} patches")
    if np.linalg.matrix_rank(measured) < channels:
        raise ValueError("the patches' colours don't span the camera's channels")
    white = None if white is None else np.asarray(_magnitude(white), dtype=float)

    row_weights = np.ones(samples) if weights is None else np.asarray(weights, dtype=float)
    if weighting == "relative":
        row_weights = row_weights / np.abs(reference.sum(axis=1))
    matrix = _weighted_lstsq(measured, reference, row_weights, preserve)

    if method == "uv_Y":
        uv_tol, Y_tol = tolerance
        target_uv = _uv_prime(reference)

        def residual(flat):
            predicted = measured @ flat.reshape(3, channels).T
            duv = (_uv_prime(predicted) - target_uv) / uv_tol
            dY = (predicted[:, 1] / reference[:, 1] - 1) / Y_tol
            return np.concatenate([duv.ravel(), dY])

        matrix = least_squares(residual, matrix.ravel()).x.reshape(3, channels)
    elif method == "delta_e2000":
        if white is None:
            raise ValueError("fitting for CIEDE2000 needs a white")
        from .analysis.colorimetry import xyz_to_lab
        from .analysis.delta_e import delta_e_2000

        target_Lab = xyz_to_lab(reference, white)

        def residual(flat):
            predicted = measured @ flat.reshape(3, channels).T
            return delta_e_2000(xyz_to_lab(predicted, white), target_Lab)

        matrix = least_squares(residual, matrix.ravel()).x.reshape(3, channels)

    return ColorMatrixFit(matrix, method, measured, reference, units.parse(unit), white)


def _weighted_lstsq(measured, reference, weights, preserve):
    """Least squares per output row, optionally exact at sample `preserve`."""
    A = measured * weights[:, None]
    B = reference * weights[:, None]
    if preserve is None:
        solution, *_ = np.linalg.lstsq(A, B, rcond=None)
        return solution.T
    # Minimise |A m - b|^2 subject to r . m = target, per output row (KKT).
    r = measured[int(preserve)]
    n = A.shape[1]
    kkt = np.zeros((n + 1, n + 1))
    kkt[:n, :n] = 2 * A.T @ A
    kkt[:n, n] = r
    kkt[n, :n] = r
    rows = []
    for k in range(3):
        rhs = np.concatenate([2 * A.T @ B[:, k], [reference[int(preserve), k]]])
        rows.append(np.linalg.solve(kkt, rhs)[:n])
    return np.array(rows)


def _uv_prime(XYZ):
    from .analysis.colorimetry import xyz_to_uv_prime
    return xyz_to_uv_prime(XYZ)


def _errors(predicted, reference, white):
    from .analysis.colorimetry import xyz_to_xy

    errors = {
        "delta_uv_prime": np.hypot(*(_uv_prime(predicted) - _uv_prime(reference)).T),
        "delta_xy": np.hypot(*(xyz_to_xy(predicted) - xyz_to_xy(reference)).T),
        "delta_Y": predicted[:, 1] / reference[:, 1] - 1,
    }
    if white is not None:
        from .analysis.colorimetry import xyz_to_lab
        from .analysis.delta_e import delta_e_2000
        errors["delta_e2000"] = delta_e_2000(xyz_to_lab(predicted, white),
                                             xyz_to_lab(reference, white))
    return errors


def _report(errors):
    report = {}
    for name, values in errors.items():
        magnitude = np.abs(values)
        report[name] = {"mean": float(np.mean(magnitude)),
                        "rms": float(np.sqrt(np.mean(values ** 2))),
                        "max": float(np.max(magnitude))}
    return report


def _unit_of(values):
    if units.is_quantity(values):
        return values.units
    if isinstance(values, (list, tuple)) and values and units.is_quantity(values[0]):
        return values[0].units
    return None


def _magnitude(values):
    if units.is_quantity(values):
        return np.asarray(values.magnitude, dtype=float)
    if isinstance(values, (list, tuple)) and values and units.is_quantity(values[0]):
        unit = values[0].units
        return np.array([v.to(unit).magnitude for v in values], dtype=float)
    return np.asarray(values, dtype=float)
