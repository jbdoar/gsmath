"""One-dimensional sampled data: values along an axis, in one or more channels.

A :class:`Series` holds values against a single axis -- ``(n,)`` for one
channel, ``(n, channels)`` for several, named like an image's channels. Its
kinds name the axis for what it is:

- :class:`~gsmath.Spectrum` -- one channel against ``wavelength``;
- :class:`~gsmath.SpectralSensitivity` -- a camera's channels against
  ``wavelength``;
- :class:`~gsmath.LineCut` -- an image's channels against ``distance``
  along a line.

and a plain ``Series(x, values)`` is for anything else (a signal against
time, temperature, source level). Whatever the axis is called, :attr:`axis`
and :attr:`axis_unit` reach it, so the same code serves them all::

    cut = image.line((100, 240), (500, 240))
    cut["G"].peaks().fwhm          # width of a spot, in the cut's distance unit
    cut.plot()
"""

from __future__ import annotations

import numpy as np

from . import units
from .sampled import KEEP, Sampled, read_only


class Series(Sampled):
    """Values along one axis, in one or more named channels."""

    AXIS = "x"
    """The name of the axis attribute (``wavelength``, ``distance``, ...);
    its unit is ``<AXIS>_unit``."""
    AXIS_LABEL = "x"
    VALUE_LABEL = None
    _FIELDS = ("x", "values", "label", "x_unit", "value_unit", "channels", "metadata", "x_label")
    _REMOVABLE = {"x_unit", "value_unit", "channels", "x_label"}
    _PER_SAMPLE = ()
    """Further fields with one entry per sample, kept in step when resampling."""

    def __init__(self, x, values, label="Series", x_unit=None, value_unit=None, channels=None,
                 metadata=None, x_label=None):
        """``x_label`` names the axis on plots ("Exposure", "Frequency")."""
        self._set_series(x, values, label, x_unit, value_unit, channels, metadata)
        self.x_label = x_label
        self._seal()

    def _set_series(self, axis, values, label, axis_unit, value_unit, channels, metadata):
        axis = np.asarray(axis)
        values = np.asarray(values)
        if axis.ndim != 1:
            raise ValueError(f"the {self.AXIS} axis must be one-dimensional")
        if values.ndim not in (1, 2) or values.shape[0] != axis.size:
            raise ValueError(f"values must be (n,) or (n, channels), with one row per "
                             f"{self.AXIS} ({axis.size})")
        if values.dtype.hasobject:
            raise TypeError("values cannot use object dtype")
        setattr(self, self.AXIS, read_only(axis))
        setattr(self, f"{self.AXIS}_unit", units.parse(axis_unit))
        self.values = read_only(values)
        self.label = label
        self.value_unit = units.parse(value_unit)
        self.channels = _channels(channels, values)
        self.metadata = {} if metadata is None else dict(metadata)

    # the axis, whatever it's called
    @property
    def axis(self) -> np.ndarray:
        return getattr(self, self.AXIS)

    @property
    def axis_unit(self):
        return getattr(self, f"{self.AXIS}_unit")

    def __len__(self):
        return len(self.axis)

    @property
    def n_channels(self) -> int:
        return 1 if self.values.ndim == 1 else self.values.shape[1]

    def __repr__(self):
        channels = f", channels={self.channels!r}" if self.channels else ""
        return (f"{type(self).__name__}(label={self.label!r}, points={len(self)}{channels}, "
                f"{self.AXIS}_unit={str(self.axis_unit)!r}, "
                f"value_unit={str(self.value_unit)!r})")

    # copies
    def copy(self, **changes):
        """Copy with selected fields replaced. ``None`` keeps the axis,
        values, label and metadata; for an optional field (units, channels)
        it removes it."""
        unknown = set(changes) - set(self._FIELDS)
        if unknown:
            raise TypeError(f"{type(self).__name__}.copy() got unknown field(s) {sorted(unknown)}")
        fields = {}
        for name in self._FIELDS:
            value = changes.get(name, KEEP)
            if value is KEEP or (value is None and name not in self._REMOVABLE):
                value = getattr(self, name)  # arrays are read-only: shared
                if isinstance(value, dict):
                    value = value.copy()
            fields[name] = value
        return type(self)(**fields)

    # channels
    def channel(self, name):
        """One named channel (values ``(n,)``), or a list of names as a
        multi-channel series."""
        if self.channels is None:
            raise ValueError(f"{self.label!r} has no named channels")
        names = [name] if isinstance(name, str) else list(name)
        missing = [n for n in names if n not in self.channels]
        if missing:
            raise KeyError(f"{self.label!r} has no channel {missing[0]!r}; it has {self.channels}")
        if self.values.ndim == 1:
            values = self.values
        else:
            index = [self.channels.index(n) for n in names]
            values = self.values[:, index[0]] if isinstance(name, str) else self.values[:, index]
        return self.copy(values=values, channels=tuple(names))

    def __getitem__(self, key):
        """``series["G"]`` or ``series[["R", "G"]]``: channels."""
        if isinstance(key, (str, list)):
            return self.channel(key)
        raise TypeError("index a series by channel name: series['G'] or series[['R', 'G']]")

    def _single(self, channel, what):
        """This series as one channel, for an analysis of one curve."""
        if channel is not None:
            return self.channel(channel)
        if self.n_channels > 1:
            raise ValueError(f"{what} is of one channel: pick one of {self.channels}")
        return self

    # resampling
    def trim(self, low, high):
        """The samples with the axis within ``[low, high]`` (inclusive)."""
        from .transforms import trim
        return trim(self, low, high)

    def interpolate(self, axis=None, *, start=None, stop=None, step=None, spline_order=1,
                    extrapolate=False):
        """Resample every channel onto new ``axis`` values (increasing), or a
        regular grid of ``step`` from ``start`` to ``stop`` (default: the
        series' own span); NaN outside unless ``extrapolate``. Spline
        ``order`` 1 is linear. See :func:`gsmath.transforms.interpolate`."""
        from .transforms import interpolate
        return interpolate(self, axis, startwl=start, stopwl=stop, step=step,
                           spline_order=spline_order, extrapolate=extrapolate)

    def __call__(self, axis, *, spline_order=1, extrapolate=False):
        """The interpolated values at ``axis`` values in any order: a number
        (or one per channel) for one value, else an array."""
        from .transforms import evaluate
        found = evaluate(self, np.atleast_1d(axis), spline_order=spline_order,
                         extrapolate=extrapolate)
        return found[0] if np.ndim(axis) == 0 else found

    # arithmetic: the same axis, and the same channels (or one channel for all)
    def _align(self, other):
        if not isinstance(other, Series) or other.AXIS != self.AXIS:
            return NotImplemented
        name = self.AXIS.capitalize()
        if self.axis_unit != other.axis_unit:
            raise ValueError(f"{name} units must match.")
        if not np.array_equal(self.axis, other.axis):
            raise ValueError(f"{name} arrays must match exactly.")
        left, right = self.values, other.values
        changes = {}
        if left.ndim == 2 and right.ndim == 2:
            if self.channels != other.channels:
                raise ValueError(f"Channels must match: {self.channels} vs {other.channels}.")
        elif left.ndim == 2:
            right = np.asarray(right)[:, None]
        elif right.ndim == 2:
            left = np.asarray(left)[:, None]
            changes["channels"] = other.channels
        return left, right, changes

    # analyses
    def peaks(self, channel=None, **kwargs):
        """Peak position, width (FWHM) and centroid of one channel's curve
        (see :class:`gsmath.analysis.PeakAnalysis`; on a series, its
        ``*_wavelength`` results are positions along the axis)."""
        from .analysis import PeakAnalysis
        return self._single(channel, "a peak analysis")._get_analysis(PeakAnalysis, **kwargs)

    def stats(self, channel=None):
        """Descriptive statistics of one channel's values."""
        from .analysis import StatisticsAnalysis
        return self._single(channel, "statistics")._get_analysis(StatisticsAnalysis)

    # summary / comparison
    def profile(self):
        """The default profile for a series: for each channel (or the one
        curve), its peak (``peak``, ``peak_value``, ``fwhm``, ``centroid``,
        positions in the axis unit) and statistics (``mean``, ``min``,
        ``max``, ``std``)."""
        from .profile import Field, Profile

        def curve(name):
            def at(getter, unit):
                return Field(lambda s: getter(s._single(name, "a summary")),
                             unit=lambda s: unit(s))

            axis, value = (lambda s: s.axis_unit), (lambda s: s.value_unit)
            return {
                "peak": at(lambda c: c.peaks().peak_wavelength, axis),
                "peak_value": at(lambda c: c.peaks().peak_value, value),
                "fwhm": at(lambda c: c.peaks().fwhm, axis),
                "centroid": at(lambda c: c.peaks().centroid_wavelength, axis),
                "mean": at(lambda c: c.stats().mean, value),
                "min": at(lambda c: c.stats().min, value),
                "max": at(lambda c: c.stats().max, value),
                "std": at(lambda c: c.stats().std, value),
            }

        if self.n_channels == 1:
            return Profile(name=type(self).__name__, fields=curve(None))
        return Profile(name=type(self).__name__,
                       fields={name: curve(name) for name in self.channels})

    def plot(self, ax=None, **kwargs):
        """Values against the axis, one line per channel; returns the axes."""
        import matplotlib.pyplot as plt

        from .image_plotting import LINE_STYLES, channel_colours

        ax = plt.subplots()[1] if ax is None else ax
        values = self.values if self.values.ndim == 2 else self.values[:, None]
        names = self.channels if self.channels and len(self.channels) == values.shape[1] \
            else tuple(str(i) for i in range(values.shape[1]))
        # Colour plus line style: curves often end at the same level, where
        # direct labels would collide, and red/green alone isn't enough.
        for i, (name, colour) in enumerate(zip(names, channel_colours(names))):
            ax.plot(self.axis, values[:, i], label=name, color=colour,
                    **{"linewidth": 1.5, "linestyle": LINE_STYLES[i % len(LINE_STYLES)],
                       **kwargs})
        unit = units.format_unit(self.axis_unit, pretty=True)
        name = getattr(self, "x_label", None) or self.AXIS_LABEL
        ax.set_xlabel(f"{name} ({unit})" if unit else name)
        value_unit = units.format_unit(self.value_unit, pretty=True)
        if self.VALUE_LABEL:
            ax.set_ylabel(f"{self.VALUE_LABEL} ({value_unit})" if value_unit else self.VALUE_LABEL)
        elif value_unit:
            ax.set_ylabel(value_unit)
        if values.shape[1] > 1:
            ax.legend()
        return ax


def _channels(channels, values):
    if channels is None:
        return None if values.ndim == 1 else tuple(str(i) for i in range(values.shape[1]))
    channels = (channels,) if isinstance(channels, str) else tuple(str(c) for c in channels)
    expected = 1 if values.ndim == 1 else values.shape[1]
    if len(channels) != expected:
        raise ValueError(f"{len(channels)} channel names for {expected} channel(s)")
    if len(set(channels)) != len(channels):
        raise ValueError("channel names must be unique")
    return channels
