"""Characterising how a camera acquires: dark signal, linearity, flat field.

These are measured once per camera (and setting: gain, binning, temperature)
and then applied by calibration steps:

- :class:`DarkModel` -- each pixel's dark signal as bias + rate x exposure,
  fitted from dark frames at several exposures -- and, from darks at several
  sensor temperatures, how bias and rate change with temperature (for an
  uncooled camera); its :class:`SubtractDarkModel` step removes the dark for
  any measurement's exposure and temperature, without a dark frame for each.
  It also estimates read noise (from repeated frames) and finds hot pixels.
- :class:`MaskSaturated` -- marks pixels at or above the sensor's white level
  as NaN, first thing, so clipped values can't pass for light.
- :class:`Linearity` -- signal against exposure (or source level) per
  channel: gain, deviation from proportionality, where saturation sets in,
  and an optional :class:`Linearize` correction step (:meth:`Linearity.step`).
- :func:`build_flat` -- a flat field from frames of a uniform source,
  dark-corrected and averaged, optionally smoothed within each colour's
  sites of a raw mosaic.

A colorimeter's calibration then reads, for raw Bayer data::

    Calibration("colorimeter", [
        MaskSaturated(), dark.step(), FlatField(flat), Demosaic(), PerExposure(),
        fit.step(),
    ])

Measure the spectral sensitivity (or fitting patches) with frames processed
through the same dark and flat steps, so the matrix sees the signal it will
be applied to.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from . import units
from .analysis.base import Result
from .calibration import Step
from .image import Image


def _seconds(exposure) -> float:
    if not units.is_quantity(exposure):
        raise TypeError("exposures must be time quantities, e.g. Q_(10, 'ms')")
    return float(exposure.to("s").magnitude)


def _celsius(temperature) -> float:
    if not (units.is_quantity(temperature) and temperature.check("[temperature]")):
        raise TypeError("temperatures must be temperature quantities, e.g. Q_(31.5, 'degC')")
    return float(temperature.to("degC").magnitude)


def _frames(frames, exposures, temperatures=None):
    """(first frame, [(seconds, celsius or None, values)]), one per frame.
    Measurements bring their own exposure and temperature."""
    frames = list(frames)
    if exposures is None:
        exposures = [getattr(f, "exposure", None) for f in frames]
        if temperatures is None:
            temperatures = [getattr(f, "temperature", None) for f in frames]
        frames = [getattr(f, "raw", f) for f in frames]
    exposures = list(exposures)
    temperatures = [None] * len(frames) if temperatures is None else list(temperatures)
    if len(exposures) != len(frames) or len(temperatures) != len(frames):
        raise ValueError("one exposure (and temperature, if any) per frame")
    known = [t is not None for t in temperatures]
    if any(known) and not all(known):
        raise ValueError("give every frame a temperature, or none")
    if len({f.values.shape for f in frames}) != 1:
        raise ValueError("all frames must have the same shape")
    return frames[0], [(_seconds(e), None if T is None else _celsius(T),
                        np.asarray(f.values, dtype=float))
                       for f, e, T in zip(frames, exposures, temperatures)]


def _temperature(temperatures, i):
    """The i-th frame's temperature from ``temperatures``: None, one quantity
    for all, or one per frame."""
    if temperatures is None or units.is_quantity(temperatures) and np.ndim(
            temperatures.magnitude) == 0:
        return temperatures
    return temperatures[i]


def _read_noise(variance):
    """Root of the pixels' mean variance, leaving out the noisiest 1%."""
    variance = variance.ravel()
    return float(np.sqrt(np.mean(variance[variance <= np.percentile(variance, 99)])))


def _modelled(bias, rate, t, *, slope=None, doubling=None, reference=None, temperature=None):
    """bias(T) + rate(T) t, as arrays; T, the reference in degC."""
    bias, rate = np.asarray(bias, dtype=float), np.asarray(rate, dtype=float)
    if temperature is not None and reference is not None:
        dT = temperature - reference
        if slope is not None:
            bias = bias + np.asarray(slope, dtype=float) * dT
        if doubling is not None:
            rate = rate * 2.0 ** (dT / doubling)
    return bias + rate * t


@dataclass(frozen=True, eq=False)
class DarkModel(Result):
    """Each pixel's dark signal: ``bias + rate * exposure`` -- and, fitted
    from darks at two or more sensor temperatures, how it changes with
    temperature T::

        bias(T) = bias + bias_slope * (T - T0)
        rate(T) = rate * 2 ** ((T - T0) / doubling)

    with ``bias`` and ``rate`` at the reference temperature T0. Dark current
    grows about exponentially with temperature, doubling every several
    kelvin; ``doubling`` is one value for the whole sensor (a per-pixel
    exponent would mostly fit noise). A small bias drift is linear per pixel."""
    bias: Image
    """Signal at zero exposure (counts), at the reference temperature."""
    rate: Image
    """Dark signal per second (counts/s), at the reference temperature."""
    read_noise: float | None = None
    """Temporal noise (counts rms) at the shortest exposure with repeated
    frames: the root of the pixels' mean variance, leaving out the noisiest
    1% (hot pixels). Fitted across temperatures, the variance is of the
    frames' residuals from the model."""
    residual_rms: float = 0.0
    """RMS of the fit's residuals over all pixels and frames (counts)."""
    exposures: tuple = ()
    """The exposures fitted (s)."""
    metadata: dict = field(default_factory=dict)
    bias_slope: Image | None = None
    """The bias's change per kelvin (counts/K), when fitted across temperatures."""
    doubling: float | None = None
    """Kelvin over which the dark rate doubles, when fitted across temperatures."""
    reference_temperature: float | None = None
    """T0 (degC): the temperature ``bias`` and ``rate`` hold for -- the darks'
    temperature, or by default their mean when they span several."""
    temperatures: tuple = ()
    """The lowest and highest sensor temperatures fitted (degC), if known."""
    by_temperature: tuple = ()
    """Per temperature (to the nearest degree) fitted: ``(degC, median rate
    measured at that temperature or None, residual rms)``."""

    @property
    def temperature_dependent(self) -> bool:
        return self.doubling is not None

    @classmethod
    def fit(cls, frames, exposures=None, temperatures=None, *, reference=None):
        """Fit from dark frames: Images, with ``exposures`` (and optionally
        ``temperatures``) as quantities, or Measurements, using their
        ``exposure`` and ``temperature``. At least two distinct exposures;
        with frames at two or more temperatures, also the temperature
        dependence, best from several exposures at each of several
        temperatures (each frame is fitted as taken, so the temperature
        needn't repeat exactly). ``reference`` (a temperature) sets T0."""
        first, data = _frames(frames, exposures, temperatures)
        times = np.array([t for t, _, _ in data])
        if len(np.unique(np.round(times, 12))) < 2:
            raise ValueError("fitting a dark model needs frames at two or more exposures")
        celsius = None if data[0][1] is None else np.array([T for _, T, _ in data])
        values = np.stack([v for _, _, v in data])
        unit = first.value_unit
        rate_unit = units.divide_units(unit, "s")
        if celsius is not None and np.ptp(celsius) > 0:
            return cls._fit_temperatures(first, times, celsius, values, reference, rate_unit)

        groups = {}
        for t, v in zip(times, values):
            groups.setdefault(round(float(t), 12), []).append(v)
        times = np.array(sorted(groups))
        stacks = [np.stack(groups[t]) for t in sorted(groups)]
        means = np.stack([stack.mean(axis=0) for stack in stacks])
        centred = times - times.mean()
        rate = np.tensordot(centred, means - means.mean(axis=0), axes=1) / np.sum(centred ** 2)
        bias = means.mean(axis=0) - rate * times.mean()
        residual = means - (bias[None] + times.reshape(-1, *[1] * bias.ndim) * rate[None])
        repeats = [stack for stack in stacks if stack.shape[0] > 1]
        temperature = {} if celsius is None else {
            "reference_temperature": float(celsius[0]),
            "temperatures": (float(celsius[0]), float(celsius[0]))}
        return cls(
            bias=first.copy(values=bias, label="dark bias"),
            rate=first.copy(values=rate, value_unit=rate_unit, label="dark rate"),
            read_noise=_read_noise(repeats[0].var(axis=0, ddof=1)) if repeats else None,
            residual_rms=float(np.sqrt(np.mean(residual ** 2))),
            exposures=tuple(float(t) for t in times), **temperature)

    @classmethod
    def _fit_temperatures(cls, first, times, celsius, values, reference, rate_unit):
        from scipy.optimize import minimize_scalar

        T0 = round(float(celsius.mean()), 1) if reference is None else _celsius(reference)
        dT = celsius - T0
        n, shape = len(times), values.shape[1:]
        Y = values.reshape(n, -1)
        sample = Y[:, ::max(1, Y.shape[1] // 20000)]  # enough pixels to find the exponent

        def design(k):  # k = ln 2 / doubling
            return np.column_stack([np.ones(n), dT, times * np.exp(k * dT)])

        conditions = {(round(float(t), 12), round(float(T), 3)) for t, T in zip(times, celsius)}
        if len(conditions) < 4 or np.linalg.matrix_rank(design(0.1)) < 3:
            raise ValueError("fitting the dark's temperature dependence needs frames at four or "
                             "more combinations of exposure and temperature")

        def misfit(k):
            X = design(k)
            return np.sum((sample - X @ np.linalg.lstsq(X, sample, rcond=None)[0]) ** 2)

        # Dark current rises with temperature: doubling every 1 K or more slowly.
        k = float(minimize_scalar(misfit, bounds=(1e-6, np.log(2)), method="bounded",
                                  options={"xatol": 1e-10}).x)
        X = design(k)
        coefficients = np.linalg.lstsq(X, Y, rcond=None)[0]
        residual = Y - X @ coefficients
        bias, slope, rate = (c.reshape(shape) for c in coefficients)

        shortest = np.isclose(times, times.min())
        read_noise = None
        if shortest.sum() > 1:  # residual variance, for the 3 fitted parameters per pixel
            variance = np.mean(residual[shortest] ** 2, axis=0) * n / (n - 3)
            read_noise = _read_noise(variance)
        by_temperature = []
        for T in np.unique(np.round(celsius)):
            at = np.round(celsius) == T
            measured = None
            if len(np.unique(np.round(times[at], 12))) > 1:  # a rate of its own
                t, y = times[at], Y[at] - slope.ravel() * dT[at, None]
                centred = t - t.mean()
                measured = float(np.median(centred @ (y - y.mean(axis=0)) / np.sum(centred ** 2)))
            by_temperature.append((float(T), measured, float(np.sqrt(np.mean(residual[at] ** 2)))))
        return cls(
            bias=first.copy(values=bias, label="dark bias"),
            rate=first.copy(values=rate, value_unit=rate_unit, label="dark rate"),
            read_noise=read_noise,
            residual_rms=float(np.sqrt(np.mean(residual ** 2))),
            exposures=tuple(float(t) for t in np.unique(np.round(times, 12))),
            bias_slope=first.copy(values=slope, value_unit=units.divide_units(first.value_unit, "K"),
                                  label="dark bias slope"),
            doubling=float(np.log(2) / k), reference_temperature=T0,
            temperatures=(float(celsius.min()), float(celsius.max())),
            by_temperature=tuple(by_temperature))

    def dark(self, exposure, temperature=None) -> Image:
        """The modelled dark frame for an ``exposure`` (a time quantity) --
        and the sensor ``temperature``, which a temperature-dependent model needs."""
        if temperature is None and self.temperature_dependent:
            raise ValueError("this dark model depends on temperature: give the sensor's")
        celsius = None if temperature is None else _celsius(temperature)
        values = _modelled(self.bias.values, self.rate.values, _seconds(exposure),
                           slope=None if self.bias_slope is None else self.bias_slope.values,
                           doubling=self.doubling, reference=self.reference_temperature,
                           temperature=celsius)
        at = "" if temperature is None else f", {temperature:~}"
        return self.bias.copy(values=values, label=f"dark ({exposure:~}{at})")

    def rate_at(self, temperature) -> float:
        """The median pixel's dark rate at ``temperature`` (counts/s)."""
        median = float(np.median(self.rate.values))
        if not self.temperature_dependent:
            return median
        return median * 2.0 ** ((_celsius(temperature) - self.reference_temperature) / self.doubling)

    def hot_pixels(self, threshold=6.0):
        """Pixels whose dark rate is more than ``threshold`` robust sigmas
        above the median: a :class:`~gsmath.Mask` region."""
        from .roi import Mask

        rate = np.asarray(self.rate.values, dtype=float)
        median = np.median(rate)
        spread = 1.4826 * np.median(np.abs(rate - median))
        return Mask(rate > median + threshold * max(spread, np.finfo(float).tiny))

    def step(self, *, optional=False):
        """The step subtracting this model's dark from a measurement."""
        temperature = {}
        if self.reference_temperature is not None:
            celsius = lambda value: units.Q_(value, "degC")  # noqa: E731
            temperature = {
                "bias_slope": self.bias_slope,
                "doubling": None if self.doubling is None else units.Q_(self.doubling, "K"),
                "reference_temperature": celsius(self.reference_temperature),
                "lowest_temperature": celsius(self.temperatures[0]),
                "highest_temperature": celsius(self.temperatures[1])}
        return SubtractDarkModel(self.bias, self.rate, **temperature, optional=optional)

    @property
    def units(self):
        signal = self.bias.value_unit
        return {"exposures": units.parse("s"), "bias_median": signal, "read_noise": signal,
                "residual_rms": signal, "rate_median": self.rate.value_unit,
                "hot_pixels": units.ureg.dimensionless,
                "reference_temperature": units.parse("degC"), "temperatures": units.parse("degC"),
                "doubling": units.parse("K"),
                "bias_slope_median": None if self.bias_slope is None
                else self.bias_slope.value_unit}

    @property
    def summary(self):
        found = {"exposures": list(self.exposures),
                 "bias_median": float(np.median(self.bias.values)),
                 "rate_median": float(np.median(self.rate.values)),
                 "read_noise": self.read_noise, "residual_rms": self.residual_rms,
                 "hot_pixels": int(self.hot_pixels().array.sum())}
        if self.reference_temperature is not None:
            found["reference_temperature"] = self.reference_temperature
            found["temperatures"] = list(self.temperatures)
        if self.temperature_dependent:
            found["doubling"] = self.doubling
            found["bias_slope_median"] = float(np.median(self.bias_slope.values))
        return found

    def plot(self, ax=None):
        """The median pixel's dark rate against sensor temperature: the
        model, and the rate measured at each temperature with frames at
        several exposures. Returns the axes."""
        import matplotlib.pyplot as plt

        from matplotlib.ticker import ScalarFormatter

        from .image_plotting import INK, SECONDARY_INK

        if not self.temperature_dependent:
            raise ValueError("this dark model was fitted at one temperature")
        ax = plt.subplots()[1] if ax is None else ax
        low, high = self.temperatures
        T = np.linspace(low - 1, high + 1, 100)
        model = [self.rate_at(units.Q_(t, "degC")) for t in T]
        ax.plot(T, model, color=INK, linewidth=2)
        middle = len(T) // 2  # above the rising line, clear of the points
        ax.annotate("model", (T[middle], model[middle]), xytext=(-8, 8),
                    textcoords="offset points", ha="right", color=INK, fontsize="small")
        measured = [(t, r) for t, r, _ in self.by_temperature if r is not None and r > 0]
        if measured:
            ax.plot(*zip(*measured), linestyle="none", marker="o", markersize=8,
                    color=SECONDARY_INK, markeredgecolor="white")
            ax.annotate("measured", measured[0], xytext=(8, -14), textcoords="offset points",
                        color=INK, fontsize="small")
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_title(f"Dark rate doubles every {self.doubling:.1f} K", loc="left")
        ax.set_xlabel("Sensor temperature (°C)")
        ax.set_ylabel(f"Median dark rate ({units.format_unit(self.rate.value_unit)})")
        return ax


class SubtractDarkModel(Step):
    """Subtract ``bias + rate * exposure``: a modelled dark for the
    measurement's exposure (see :class:`DarkModel`) -- and, for a model
    fitted across temperatures, its sensor temperature, which it then needs.
    It warns when that temperature is outside the range the model was fitted
    over (or, for a model from one temperature, differs from it by more
    than 1 K): the dark is extrapolated, or not adjusted at all."""

    removes_dark = True

    def __init__(self, bias, rate, *, bias_slope=None, doubling=None, reference_temperature=None,
                 lowest_temperature=None, highest_temperature=None, optional=False):
        super().__init__(optional=optional)
        self.bias, self.rate, self.bias_slope = bias, rate, bias_slope
        self.doubling = doubling
        self.reference_temperature = reference_temperature
        self.lowest_temperature = lowest_temperature
        self.highest_temperature = highest_temperature
        if (bias_slope is not None or doubling is not None) and reference_temperature is None:
            raise ValueError("a temperature-dependent dark needs its reference_temperature")

    @property
    def temperature_dependent(self) -> bool:
        return self.bias_slope is not None or self.doubling is not None

    def missing(self, measurement):
        if measurement.exposure is None:
            return "an exposure time"
        if self.temperature_dependent and measurement.temperature is None:
            return "a sensor temperature"
        return None

    def apply(self, data, measurement):
        t = _seconds(measurement.exposure)
        if self.bias.shape != data.shape:
            raise ValueError(f"the dark model's shape {self.bias.shape} doesn't match "
                             f"the data {data.shape}")
        celsius = None
        if measurement.temperature is not None and self.reference_temperature is not None:
            celsius = _celsius(measurement.temperature)
            self._check_range(celsius)
        dark = _modelled(
            self.bias.values, self.rate.values, t,
            slope=None if self.bias_slope is None else self.bias_slope.values,
            doubling=None if self.doubling is None else float(self.doubling.to("K").magnitude),
            reference=None if self.reference_temperature is None else _celsius(
                self.reference_temperature),
            temperature=celsius)
        if np.ndim(data.values) == 3 and np.ndim(dark) == 2:
            dark = dark[:, :, None]
        if self.bias.value_unit != data.value_unit:
            dark = units.convert_values(dark, self.bias.value_unit, data.value_unit)
        return data.copy(values=np.asarray(data.values, dtype=float) - dark)

    def _check_range(self, celsius):
        low = high = _celsius(self.reference_temperature)
        if self.lowest_temperature is not None:
            low, high = _celsius(self.lowest_temperature), _celsius(self.highest_temperature)
        margin = 0.0 if self.temperature_dependent else 1.0
        if celsius < low - margin or celsius > high + margin:
            fitted = f"{low:g} °C" if low == high else f"{low:g}-{high:g} °C"
            how = "extrapolated" if self.temperature_dependent else "not adjusted for it"
            warnings.warn(f"sensor temperature {celsius:g} °C is outside the dark model's "
                          f"{fitted}: the dark is {how}", stacklevel=4)

    def arguments(self):
        found = {"bias": self.bias, "rate": self.rate}
        for name in ("bias_slope", "doubling", "reference_temperature", "lowest_temperature",
                     "highest_temperature"):
            if getattr(self, name) is not None:  # absent for a model from one temperature
                found[name] = getattr(self, name)
        return found


class MaskSaturated(Step):
    """Make pixels at or above ``level`` NaN -- by default 98% of the white
    level a raw file records (``image.mosaic.white_level``). Place it
    first, on the raw counts. (Demosaicing spreads a NaN to its neighbours,
    which a clipped pixel would have corrupted anyway.)"""

    def __init__(self, level=None, *, fraction=0.98, optional=False):
        super().__init__(optional=optional)
        self.level = None if level is None else float(level)
        self.fraction = float(fraction)

    def apply(self, data, measurement):
        level = self.level
        if level is None:
            white = data.mosaic.white_level if getattr(data, "mosaic", None) else None
            if white is None:
                raise ValueError("MaskSaturated needs a level: the data records no white level")
            level = self.fraction * float(white)
        values = np.asarray(data.values, dtype=float)
        return data.copy(values=np.where(values >= level, np.nan, values))

    def arguments(self):
        arguments = {"fraction": self.fraction}
        if self.level is not None:
            arguments["level"] = self.level
        return arguments


@dataclass(frozen=True, eq=False)
class Linearity(Result):
    """Signal against stimulus (exposure, or source level) per channel."""
    stimulus: np.ndarray
    signal: np.ndarray
    """``(points, channels)``, dark-subtracted."""
    channels: tuple
    gain: np.ndarray
    """Signal per unit stimulus, fitted through the origin over the
    unsaturated points' lower half."""
    saturated: np.ndarray
    """``(points, channels)``: True from where the signal stops rising."""
    stimulus_unit: object = None
    signal_unit: object = None

    @property
    def units(self):
        return {"gain": None if self.signal_unit is None
                else units.divide_units(self.signal_unit, self.stimulus_unit),
                "max_deviation": units.ureg.dimensionless, "saturation": self.signal_unit}

    @property
    def curve(self):
        """Signal against stimulus, per channel, as a :class:`~gsmath.Series`."""
        from .series import Series
        return Series(self.stimulus, self.signal, label="Linearity", x_unit=self.stimulus_unit,
                      value_unit=self.signal_unit, channels=self.channels, x_label="Stimulus")

    @classmethod
    def fit(cls, stimulus, signal, *, channels=None, clipped=None, signal_unit=None):
        """From stimuli (numbers or a quantity array) and the dark-subtracted
        mean signal at each, ``(points,)`` or ``(points, channels)`` (a
        quantity array, or numbers in ``signal_unit``); ``clipped`` (the same
        shape) marks points known to be saturated."""
        unit = stimulus.units if units.is_quantity(stimulus) else None
        if units.is_quantity(signal):
            signal_unit, signal = signal.units, signal.magnitude
        stimulus = np.asarray(getattr(stimulus, "magnitude", stimulus), dtype=float)
        signal = np.asarray(signal, dtype=float)
        if signal.ndim == 1:
            signal = signal[:, None]
        if signal.shape[0] != stimulus.size or stimulus.size < 3:
            raise ValueError("at least three points, one signal row per stimulus")
        order = np.argsort(stimulus)
        stimulus, signal = stimulus[order], signal[order]
        clipped = np.zeros(signal.shape, dtype=bool) if clipped is None else             np.asarray(clipped, dtype=bool).reshape(signal.shape)[order]
        names = tuple(channels) if channels is not None else \
            tuple(str(i) for i in range(signal.shape[1]))
        gains, saturated = [], []
        positive = np.flatnonzero(stimulus > 0)
        if positive.size == 0:
            raise ValueError("the stimuli must include positive values")
        start = int(positive[0])
        for column, known in zip(signal.T, clipped.T):
            # A point is saturated if known to be clipped, or once the signal
            # rises less than a quarter of what the trend so far predicts.
            gain = column[start] / stimulus[start]
            flags = known.copy()
            for i in range(start + 1, column.size):
                expected = gain * (stimulus[i] - stimulus[i - 1])
                if flags[i] or flags[i - 1] or (column[i] - column[i - 1]) < 0.25 * expected:
                    flags[i] = True
                else:
                    good = ~flags[:i + 1]
                    gain = np.sum(stimulus[:i + 1][good] * column[:i + 1][good]) / \
                        np.sum(stimulus[:i + 1][good] ** 2)
            usable = ~flags
            lower = usable & (column <= 0.5 * column[usable].max())
            if lower.sum() < 2:
                lower = usable
            gains.append(np.sum(stimulus[lower] * column[lower]) / np.sum(stimulus[lower] ** 2))
            saturated.append(flags)
        return cls(stimulus, signal, names, np.array(gains), np.column_stack(saturated), unit,
                   units.parse(signal_unit))

    @classmethod
    def from_frames(cls, frames, stimulus, *, roi=None, dark=None, temperatures=None):
        """From frames of a stable uniform source at each stimulus: each
        frame's mean (per colour of a raw mosaic, per channel otherwise) over
        ``roi``, less ``dark`` -- an Image, or a :class:`DarkModel` with
        ``stimulus`` as exposures (and ``temperatures``, one or one per
        frame, for a temperature-dependent model). Pixels marked NaN are skipped. If a raw
        frame records its white level, a point is saturated as soon as any
        pixel of that colour in the region reaches 98% of it -- vignetting
        clips the centre while the mean still rises."""
        from .mosaic import mosaic_of, site_means

        signals, clipped, names = [], [], None
        for i, frame in enumerate(frames):
            mosaic = mosaic_of(frame)
            if mosaic is not None and mosaic.white_level is not None:
                region = np.ones(frame.shape, bool) if roi is None else roi.mask(frame)
                values = np.asarray(frame.values, dtype=float)
                clipped.append([np.any(values[mask & region] >= 0.98 * mosaic.white_level)
                                for mask in mosaic.masks(frame.shape).values()])
            if isinstance(dark, DarkModel):
                frame = frame - dark.dark(stimulus[i], _temperature(temperatures, i))
            elif dark is not None:
                frame = frame - dark
            if mosaic_of(frame) is not None:
                signals.append(site_means(frame, roi))
                names = ("R", "G", "B")
            else:
                signals.append(np.atleast_1d(frame.stats(roi=roi).mean))
                names = frame.channels
        return cls.fit(stimulus, np.array(signals), channels=names,
                       clipped=np.array(clipped) if len(clipped) == len(signals) else None,
                       signal_unit=frame.value_unit)

    @property
    def deviation(self) -> np.ndarray:
        """``signal / (gain * stimulus) - 1`` per point, NaN where saturated
        or the stimulus is 0."""
        with np.errstate(divide="ignore", invalid="ignore"):
            ideal = self.stimulus[:, None] * self.gain[None, :]
            deviation = self.signal / ideal - 1
        return np.where(self.saturated | (ideal == 0), np.nan, deviation)

    @property
    def max_deviation(self) -> np.ndarray:
        """Largest |deviation| over the unsaturated points, per channel."""
        return np.nanmax(np.abs(self.deviation), axis=0)

    @property
    def saturation(self) -> np.ndarray:
        """The highest unsaturated signal per channel (NaN if none saturated)."""
        found = []
        for column, flags in zip(self.signal.T, self.saturated.T):
            found.append(column[~flags].max() if flags.any() else np.nan)
        return np.array(found)

    def step(self, *, optional=False):
        """A :class:`Linearize` step mapping each channel's measured signal to
        ``gain * stimulus`` over the unsaturated points."""
        measured, ideal = [], []
        for k in range(len(self.channels)):
            keep = ~self.saturated[:, k]
            measured.append(self.signal[keep, k])
            ideal.append(self.gain[k] * self.stimulus[keep])
        length = min(len(m) for m in measured)
        return Linearize(np.column_stack([m[:length] for m in measured]),
                         np.column_stack([i[:length] for i in ideal]),
                         channels=self.channels, optional=optional)

    @property
    def summary(self):
        return {name: {"gain": float(self.gain[k]), "max_deviation": float(self.max_deviation[k]),
                       "saturation": float(self.saturation[k])}
                for k, name in enumerate(self.channels)}

    def plot(self, ax=None):
        """Deviation from linear against signal, per channel; returns the axes."""
        import matplotlib.pyplot as plt

        from .image_plotting import LINE_STYLES, channel_colours

        ax = plt.subplots()[1] if ax is None else ax
        for k, (name, colour) in enumerate(zip(self.channels, channel_colours(self.channels))):
            ax.plot(self.signal[:, k], 100 * self.deviation[:, k], marker="o", markersize=4,
                    color=colour, linestyle=LINE_STYLES[k % len(LINE_STYLES)], label=name)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_xlabel("Signal")
        ax.set_ylabel("Deviation from linear (%)")
        ax.legend()
        return ax


class Linearize(Step):
    """Map measured signal to linear signal by a per-channel table (from
    :meth:`Linearity.step`): linear interpolation inside it, the
    table's end slope beyond it. For a raw mosaic, the table's channels
    R, G, B apply to the matching colour's sites."""

    def __init__(self, measured, ideal, *, channels=("R", "G", "B"), optional=False):
        super().__init__(optional=optional)
        self.measured = np.asarray(measured, dtype=float)
        self.ideal = np.asarray(ideal, dtype=float)
        self.channels = tuple(channels)
        if self.measured.shape != self.ideal.shape or self.measured.ndim != 2 \
                or self.measured.shape[1] != len(self.channels) or self.measured.shape[0] < 2:
            raise ValueError("measured and ideal are (points, channels) tables, 2+ points")

    def _map(self, values, k):
        x, y = self.measured[:, k], self.ideal[:, k]
        order = np.argsort(x)
        x, y = x[order], y[order]
        inside = np.interp(values, x, y)
        low = y[0] + (values - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
        high = y[-1] + (values - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
        return np.where(values < x[0], low, np.where(values > x[-1], high, inside))

    def apply(self, data, measurement):
        from .mosaic import mosaic_of

        values = np.asarray(data.values, dtype=float)
        out = values.copy()
        mosaic = mosaic_of(data)
        if mosaic is not None:
            for colour, mask in mosaic.masks(data.shape).items():
                out[mask] = self._map(values[mask], self.channels.index(colour))
        elif values.ndim == 3:
            for k, name in enumerate(data.channels):
                out[:, :, k] = self._map(values[:, :, k], self.channels.index(name))
        else:
            out = self._map(values, 0)
        return data.copy(values=out)

    def arguments(self):
        return {"measured": self.measured, "ideal": self.ideal, "channels": list(self.channels)}


def build_flat(frames, *, dark=None, exposures=None, temperatures=None, sigma=None,
               label="flat"):
    """A flat field from frames of a uniform source (an integrating sphere,
    a diffuser): their mean, less ``dark`` (an Image, or a :class:`DarkModel`
    with the frames' ``exposures`` and ``temperatures``, or Measurements'
    own), optionally smoothed
    by ``sigma`` pixels -- within each colour's own sites for a raw mosaic, so
    colours never mix. Smoothing suppresses the frames' noise but biases
    the outermost few pixels where the flat curves steeply; keep ``sigma``
    to a few pixels, or average more frames instead. Use it with
    :class:`~gsmath.FlatField`, which normalises it."""
    import scipy.ndimage

    from .mosaic import mosaic_of

    frames = list(frames)
    if exposures is None and frames and hasattr(frames[0], "raw"):
        exposures = [m.exposure for m in frames]
        if temperatures is None:
            temperatures = [m.temperature for m in frames]
        frames = [m.raw for m in frames]
    values = []
    for i, frame in enumerate(frames):
        v = np.asarray(frame.values, dtype=float)
        if isinstance(dark, DarkModel):
            v = v - dark.dark(exposures[i], _temperature(temperatures, i)).values
        elif dark is not None:
            v = v - np.asarray(dark.values, dtype=float)
        values.append(v)
    flat = np.mean(values, axis=0)
    first = frames[0]
    if sigma is not None:
        if mosaic_of(first) is not None:
            for i in (0, 1):
                for j in (0, 1):
                    flat[i::2, j::2] = scipy.ndimage.gaussian_filter(
                        flat[i::2, j::2], sigma / 2, mode="nearest")
        else:
            flat = first.copy(values=flat).smooth("gaussian", sigma=sigma).values
    metadata = {**first.metadata, "flat": {"frames": len(frames), "sigma": sigma}}
    return first.copy(values=flat, label=label, metadata=metadata)
