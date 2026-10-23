"""Core sampled-spectrum value object and its high-level operations."""

import numpy as np

from . import units
from .profile import SPECTRUM_PROFILE
from .sampled import read_only
from .series import Series


class Spectrum(Series):
    """Represent sampled values on a wavelength axis with units and metadata.

    This is the library's central object: a one-channel :class:`~gsmath.Series`
    whose axis is ``wavelength``. Arithmetic is unit-aware, while transforms
    and analyses delegate to focused modules.
    """
    AXIS = "wavelength"
    AXIS_LABEL = "Wavelength"
    _FIELDS = ("wavelength", "values", "label", "wavelength_unit", "value_unit", "metadata")
    _REMOVABLE = {"wavelength_unit", "value_unit"}
    channels = None
    """A spectrum is one channel."""

    def __init__(
            self,
            wavelength,
            values,
            label="Spectrum",
            wavelength_unit="nm",
            value_unit=None,
            metadata=None,
    ):
        """Create a spectrum, validating arrays and normalizing unit inputs:
        wavelengths one-dimensional, finite and strictly increasing; values
        numeric, one per wavelength. (An empty spectrum is allowed.)"""
        wavelength, values = np.asarray(wavelength), np.asarray(values)
        if wavelength.ndim != 1 or wavelength.dtype.kind not in "iuf":
            raise ValueError("wavelengths must be a one-dimensional array of numbers")
        if not np.all(np.isfinite(wavelength)):
            raise ValueError("wavelengths must be finite")
        if np.any(np.diff(wavelength) <= 0):
            raise ValueError("wavelengths must be strictly increasing (sort the data, or "
                             "reverse a descending scan)")
        if values.dtype.kind not in "iufcb":
            raise TypeError(f"values must be numbers, not {values.dtype}")
        if values.shape != wavelength.shape:
            raise ValueError(f"one value per wavelength: {values.shape} values for "
                             f"{wavelength.size} wavelengths")
        self.wavelength = read_only(wavelength)
        self.values = read_only(values)
        self.label = label
        self.wavelength_unit = units.parse(wavelength_unit)
        self.value_unit = units.parse(value_unit)
        self.metadata = {} if metadata is None else dict(metadata)
        self._seal()

    def __str__(self):
        wl_min = self.wavelength[0]
        wl_max = self.wavelength[-1]
        n = len(self.wavelength)

        unit = (
            f" {self.wavelength_unit:~}"
            if self.wavelength_unit is not None
            else ""
        )
        
        return (
            f"{self.label} "
            f"({wl_min:g}-{wl_max:g}{unit}, {n} points)"
        )

    def __repr__(self):
        return (
            f"Spectrum(label={self.label!r}, "
            f"range=({self.wavelength[0]:g}, {self.wavelength[-1]:g}), "
            f"points={len(self.wavelength)}, "
            f"wavelength_unit={str(self.wavelength_unit)!r}, "
            f"value_unit={str(self.value_unit)!r})"
        )

    def __call__(self, wavelength, *, spline_order=1, extrapolate=False):
        """Interpolate the spectrum at one wavelength or an array of them, in
        any order."""
        from .transforms import evaluate

        result = evaluate(self, np.atleast_1d(wavelength), spline_order=spline_order,
                          extrapolate=extrapolate)
        if np.ndim(wavelength) == 0:
            return float(result[0])
        return result

    @classmethod
    def from_csv(cls, *args, **kwargs):
        """Construct a spectrum using :func:`gsmath.io.from_csv`."""
        from .io import from_csv
        return from_csv(*args, **kwargs)

    @classmethod
    def from_lighttouch(cls, *args, **kwargs):
        """Construct a spectrum from a Gamma Scientific LightTouch export."""
        from .io import from_lighttouch
        return from_lighttouch(*args, **kwargs)

    @classmethod
    def read(cls, path, *, format=None, **kwargs):
        """Read a spectrum using an inferred or explicitly named format.

        Native ``.gsmath``, generic ``.csv``, and LightTouch amplitude ``.cal``
        files are inferred by suffix. For LightTouch scans named ``.csv`` or
        ``.std``, specify ``format="lighttouch"``.
        """
        from .io import read

        return read(path, format=format, **kwargs)

    def write(self, path, *, format=None, **kwargs):
        """Write this spectrum using an inferred or explicitly named format.

        The native ``gsmath`` format preserves units, labels, array dtypes, and
        JSON-compatible metadata. Generic CSV contains wavelength/value rows
        and therefore requires callers to manage contextual metadata.
        """
        from .io import write

        return write(self, path, format=format, **kwargs)
    
    # process
    def trim(self, xmin, xmax):
        """Return samples within the inclusive wavelength interval."""
        from .transforms import trim
        return trim(self, xmin, xmax)

    def interpolate(self, *args, **kwargs):
        """Interpolate values onto an explicit or regularly spaced grid."""
        from .transforms import interpolate
        return interpolate(self, *args, **kwargs)

    def pad(self, mode="zero", *, value=None):
        """Return a copy configured for constant or edge extrapolation."""
        from .transforms import pad
        return pad(self, mode=mode, value=value)

    def normalize(self, by="peak", value=1.0):
        """Scale by peak, area, photometric value, or a supplied scalar."""
        from .transforms import normalize
        return normalize(self, by=by, value=value)

    def calibrate_wavelength(self, calibration):
        """Apply a fitted WavelengthCalibration, or a legacy axis correction.

        Fitted results use pixel indices; coefficient arrays (ascending powers)
        and callables correct the existing wavelength coordinates.
        """
        from .transforms import calibrate_wavelength
        return calibrate_wavelength(self, calibration)

    def shift_wavelength(self, shift):
        """Translate every wavelength by ``shift``."""
        from .transforms import shift_wavelength
        return shift_wavelength(self, shift)

    def trim_tails(self, how="first_nonpositive"):
        """Zero the tails beyond the first nonpositive value either side of
        the peak (see :func:`gsmath.transforms.trim_tails`)."""
        from .transforms import trim_tails
        return trim_tails(self, how=how)

    def effective_range(self, threshold=0.0):
        """(first, last) wavelength above ``threshold`` relative to the peak
        (default: nonzero), or None."""
        from .transforms import effective_range
        return effective_range(self, threshold=threshold)
    
    def smooth(self, method="savgol", **kwargs):
        """Return a smoothed copy using :mod:`gsmath.smooth`."""
        from .smooth import smooth

        return smooth(self, method=method, **kwargs)

    def copy(self, **changes):
        """Copy the spectrum with selected fields replaced (``wavelength``,
        ``values``, ``label``, ``wavelength_unit``, ``value_unit``,
        ``metadata``). ``value_unit=None`` makes the copy unitless, as for
        every kind of sampled data; leave it out to keep the unit."""
        return super().copy(**changes)

    def to_wavelength_unit(self, unit):
        """Return a copy with wavelengths converted to ``unit``."""
        unit = units.parse(unit)
        
        wavelength = units.convert_values(self.wavelength, self.wavelength_unit, unit)

        return self.copy(wavelength=wavelength, wavelength_unit=unit)

    # summary / comparison
    def profile(self):
        """The default profile for spectra: SPECTRUM_PROFILE (power, peaks,
        colour, statistics)."""
        return SPECTRUM_PROFILE

    # plotting
    def plot(self, *, graph=None, ax=None, label=None, **plot_kwargs):
        """Add the spectrum to a new or existing spectrum graph."""
        from .plotting import SpectrumGraph

        if graph is not None and ax is not None:
            raise ValueError("Specify either graph or ax, not both.")

        if graph is None:
            graph = SpectrumGraph(ax=ax)

        graph.add(self, label=label, **plot_kwargs)

        return graph

    # analysis
    def peaks(self, **kwargs):
        """Return a cached peak analysis."""
        from .analysis import PeakAnalysis
        return self._get_analysis(PeakAnalysis, **kwargs)
    
    def color(self, **kwargs):
        """Return a cached colorimetric analysis."""
        from .analysis import ColorAnalysis

        if "white" in kwargs and not isinstance(kwargs["white"], str):
            kwargs["white"] = tuple(kwargs["white"])

        return self._get_analysis(ColorAnalysis, **kwargs)
    
    def power(self, **kwargs):
        """Return a cached integrated-power analysis."""
        from .analysis import PowerAnalysis
        return self._get_analysis(PowerAnalysis, **kwargs)
    
    def stats(self, **kwargs):
        """Return a cached descriptive-statistics analysis."""
        from .analysis import StatisticsAnalysis
        return self._get_analysis(StatisticsAnalysis, **kwargs)

    def _align(self, other):
        if not isinstance(other, Spectrum):
            return NotImplemented
        return super()._align(other)
