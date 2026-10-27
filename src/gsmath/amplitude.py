"""Amplitude (responsivity) calibration from a scan of a standard source.

The responsivity is the standard's certified spectral values, interpolated to
the scan's wavelengths, divided by the scan's dark-subtracted counts per
millisecond (after stray-light correction, when a matrix is given)::

    responsivity = standard / (SLC @ counts / integration_time)

A native calibration is normalised by integration time: ``PerExposure`` then
``Responsivity`` in calibrated units per (count/ms), with the stray-light
step first. Applied to a measurement at any integration time it gives
calibrated values directly, and two native calibrations compare directly.
A LightTouch ``.cal`` keeps that format's convention, factors per count at a
recorded integration time; :func:`read_amplitude_calibration` converts it.

The standard's unit (radiance or irradiance) sets the calibration's unit.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis.base import Result
from .calibration import Calibration, PerExposure, Responsivity, StrayLight
from .comparison import compare
from .measurement import Measurement
from .profile import Field, Profile
from .spectrum import Spectrum
from . import units

LIGHTTOUCH_INVALID = 10000.0
"""LightTouch's factor for samples inside the standard's range without usable counts."""

_LIGHTTOUCH_QUANTITIES = {
    "radiance": ("Radiance (µW/cm²/sr)", "uW / cm^2 / sr / nm"),
    "irradiance": ("Irradiance (µW/cm²)", "uW / cm^2 / nm"),
    "counts": ("Counts", "count"),
}


def lighttouch_unit(spectrum):
    """The spectral unit named by a LightTouch file's third header line, or None.

    ``Radiance (µW/cm²/sr)`` is µW/cm²/sr/nm, ``Irradiance (µW/cm²)`` µW/cm²/nm
    and ``Counts`` counts. For an amplitude ``.cal`` this is the calibrated
    quantity; its factors are that per count.
    """
    header = spectrum.metadata.get("header") or []
    words = header[2].split() if len(header) > 2 else []
    entry = _LIGHTTOUCH_QUANTITIES.get(words[0].lower()) if words else None
    return None if entry is None else entry[1]


def unit_text(unit):
    """A unit as plain ASCII for consoles and file headers: ``uW ms / cm^2 / count / nm / sr``."""
    text = units.format_unit(unit)
    if text is None:
        return "no unit"
    return (text.replace("µ", "u").replace("μ", "u").replace(" ** ", "^").replace("**", "^")
            .replace(" * ", " ").replace("counts", "count"))


def quantity_of(unit):
    """'radiance' or 'irradiance' for a spectral unit, or one per count or per (count/ms)."""
    unit = units.parse(unit)
    if unit is None:
        return None
    count, ms = units.ureg.count, units.ureg.ms
    # The steradian is dimensionless, so only the unit's name tells radiance
    # from irradiance (as in ColorAnalysis).
    reference = units.parse(_LIGHTTOUCH_QUANTITIES["irradiance"][1])
    for candidate in (unit, unit * count, unit * count / ms):
        if units.Q_(1, candidate).is_compatible_with(reference):
            return "radiance" if "steradian" in str(unit) else "irradiance"
    return None


def on_wavelengths(calibration, wavelength, *, atol_nm=1e-3):
    """``calibration`` with its responsivity on exactly ``wavelength``.

    Files store wavelengths rounded (a ``.cal`` to 6 decimals), so a
    calibration of these very pixels need not match a scan's axis bit for bit.
    Within ``atol_nm`` the scan's axis is adopted; otherwise this refuses:
    the calibration is for different pixels.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    steps = []
    for step in calibration.steps:
        response = getattr(step, "responsivity", None)
        if isinstance(step, Responsivity) and isinstance(response, Spectrum):
            if (len(response.wavelength) != len(wavelength)
                    or not np.allclose(response.wavelength, wavelength, rtol=0, atol=atol_nm)):
                raise ValueError("the calibration's wavelengths don't match the scan's pixels")
            step = Responsivity(response.copy(wavelength=wavelength.copy()))
        steps.append(step)
    return Calibration(calibration.name, steps, metadata=dict(calibration.metadata))


def _integration_time_ms(scan, integration_time_ms):
    if integration_time_ms is None:
        metadata = getattr(scan, "metadata", {}) or {}
        integration_time_ms = metadata.get("integration_time_ms", metadata.get("integration_time"))
    if integration_time_ms is None or not np.isfinite(integration_time_ms) or integration_time_ms <= 0:
        raise ValueError("the scan's integration time is unknown; pass integration_time_ms")
    return float(integration_time_ms)


@dataclass(frozen=True, repr=False)
class AmplitudeCalibrationResult(Result):
    """A responsivity derived from a standard, with what limits it.

    ``valid`` marks samples inside the standard's range with positive
    corrected counts; elsewhere the responsivity is zero. ``low_signal``
    marks valid samples below ``low_signal_fraction`` of the peak counts:
    usable, but noisy.
    """
    responsivity: Spectrum
    counts_per_ms: Spectrum
    standard: Spectrum
    integration_time_ms: float
    valid: np.ndarray
    low_signal: np.ndarray
    stray_light: object = None
    name: str = "amplitude calibration"

    units = {"standard_range_nm": "nm", "valid_range_nm": "nm", "integration_time_ms": "ms"}

    @property
    def quantity(self):
        return quantity_of(self.responsivity.value_unit)

    @property
    def calibration(self):
        """The native, integration-time-normalised gsmath Calibration."""
        steps = [] if self.stray_light is None else [StrayLight(self.stray_light)]
        steps += [PerExposure(), Responsivity(self.responsivity)]
        return Calibration(self.name, steps, metadata={
            "convention": "PerExposure + Responsivity per (count/ms)",
            "calibration_integration_time_ms": self.integration_time_ms,
            "standard_range_nm": self.summary["standard_range_nm"],
        })

    def lighttouch_factors(self):
        """LightTouch factors: µW/cm²(/sr)/nm per count at the calibration integration time.

        Zero outside the standard's range; LightTouch's invalid value (10000)
        inside it where the counts were not positive.
        """
        quantity = self.quantity
        if quantity is None:
            raise ValueError("only radiance or irradiance calibrations can be written as LightTouch .cal")
        target = units.parse(_LIGHTTOUCH_QUANTITIES[quantity][1]) / units.ureg.count
        per_count = units.convert_values(
            np.asarray(self.responsivity.values, dtype=float) / self.integration_time_ms,
            units.parse(self.responsivity.value_unit) / units.ureg.ms, target)
        inside = np.isfinite(self._standard_on_grid())
        factors = np.where(self.valid, per_count, np.where(inside, LIGHTTOUCH_INVALID, 0.0))
        return self.responsivity.copy(values=factors, value_unit=units.format_unit(target))

    def write(self, stem):
        """Write ``stem.gsmath`` (native) and ``stem.cal`` (LightTouch); return both paths."""
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        native = self.calibration.write(stem.with_suffix(".gsmath"))
        lighttouch = self.lighttouch_factors().write(
            stem.with_suffix(".cal"), integration_time_ms=self.integration_time_ms,
            quantity=self.quantity)
        return native, lighttouch

    def _standard_on_grid(self):
        wavelength = np.asarray(self.responsivity.wavelength, dtype=float)
        standard = self.standard
        inside = (wavelength >= standard.wavelength[0]) & (wavelength <= standard.wavelength[-1])
        values = np.full(len(wavelength), np.nan)
        values[inside] = np.interp(wavelength[inside], standard.wavelength, standard.values)
        return values

    @property
    def summary(self):
        wavelength = np.asarray(self.responsivity.wavelength, dtype=float)
        valid = wavelength[self.valid]
        return {
            "quantity": self.quantity,
            "unit": unit_text(self.responsivity.value_unit),
            "integration_time_ms": self.integration_time_ms,
            "standard_range_nm": [float(self.standard.wavelength[0]), float(self.standard.wavelength[-1])],
            "valid_range_nm": [float(valid.min()), float(valid.max())] if len(valid) else None,
            "valid_samples": int(self.valid.sum()),
            "low_signal_samples": int(self.low_signal.sum()),
            "stray_light_corrected": self.stray_light is not None,
        }

    def __str__(self):
        s = self.summary
        lines = [f"Amplitude calibration: {s['quantity'] or 'unknown quantity'}, {s['unit']}",
                 f"  integration time {s['integration_time_ms']:g} ms; "
                 f"stray light {'corrected' if s['stray_light_corrected'] else 'not corrected'}",
                 f"  standard {s['standard_range_nm'][0]:.1f}-{s['standard_range_nm'][1]:.1f} nm; "
                 f"{s['valid_samples']} valid samples"
                 + (f" ({s['valid_range_nm'][0]:.1f}-{s['valid_range_nm'][1]:.1f} nm)" if s['valid_range_nm'] else "")]
        if s["low_signal_samples"]:
            lines.append(f"  {s['low_signal_samples']} samples below the low-signal limit: "
                         "their responsivity is noisy")
        return "\n".join(lines)


def amplitude_calibration(scan, standard, *, stray_light=None, integration_time_ms=None,
                          name="amplitude calibration", low_signal_fraction=0.01):
    """Derive a responsivity from a dark-subtracted scan of a standard source.

    ``scan`` is a Spectrum of dark-subtracted counts on the instrument's
    wavelength axis; its integration time comes from ``integration_time_ms``
    or its metadata (``integration_time_ms`` or LightTouch's
    ``integration_time``). ``standard`` holds the source's certified spectral
    values with their unit; a LightTouch file's unit is read from its header
    (:func:`lighttouch_unit`). ``stray_light`` is a
    :class:`~gsmath.StrayLightCorrection` on the scan's wavelengths.
    """
    integration = _integration_time_ms(scan, integration_time_ms)
    wavelength = np.asarray(scan.wavelength, dtype=float)
    counts = np.asarray(scan.values, dtype=float)
    if counts.ndim != 1 or not np.all(np.isfinite(counts)):
        raise ValueError("the scan must be a finite one-dimensional spectrum")
    standard_unit = standard.value_unit or lighttouch_unit(standard)
    if quantity_of(standard_unit) is None:
        raise ValueError("the standard needs a radiance or irradiance unit "
                         f"(it has {standard.value_unit!r})")
    standard = standard.copy(value_unit=standard_unit)
    raw = scan.copy(value_unit="count")
    if stray_light is not None:
        measurement = Measurement(raw=raw, exposure=units.Q_(integration, "ms"))
        raw = measurement.process(Calibration("stray light", [StrayLight(stray_light)]))
    counts_per_ms = np.asarray(raw.values, dtype=float) / integration

    inside = (wavelength >= standard.wavelength[0]) & (wavelength <= standard.wavelength[-1])
    certified = np.full(len(wavelength), np.nan)
    certified[inside] = np.interp(wavelength[inside], standard.wavelength, standard.values)
    valid = inside & (counts_per_ms > 0)
    if not valid.any():
        raise ValueError("no positive counts overlap the standard's wavelength range")
    responsivity = np.zeros(len(wavelength))
    responsivity[valid] = certified[valid] / counts_per_ms[valid]
    peak = counts_per_ms[valid].max()
    low_signal = valid & (counts_per_ms < low_signal_fraction * peak)

    unit = units.parse(standard_unit) / (units.ureg.count / units.ureg.ms)
    response = Spectrum(wavelength, responsivity, label=f"{name} responsivity",
                        wavelength_unit=scan.wavelength_unit, value_unit=units.format_unit(unit))
    per_ms = Spectrum(wavelength, counts_per_ms, label="counts per ms",
                      wavelength_unit=scan.wavelength_unit, value_unit="count / ms")
    return AmplitudeCalibrationResult(response, per_ms, standard, integration, valid, low_signal,
                                      stray_light, name)


def read_amplitude_calibration(path, *, wavelength=None):
    """Read an amplitude calibration, LightTouch ``.cal`` or native ``.gsmath``.

    Returns a native, integration-time-normalised gsmath Calibration. A
    ``.cal`` becomes ``PerExposure`` + ``Responsivity(factor x integration
    time)`` in its header's unit per (count/ms). A native file must contain a
    ``PerExposure`` step; one without is in the old per-count convention and
    is refused.
    """
    path = Path(path)
    if path.suffix.lower() == ".gsmath":
        calibration = Calibration.read(path)
        names = [type(step).__name__ for step in calibration.steps]
        if "Responsivity" not in names:
            raise ValueError(f"{path.name} has no Responsivity step")
        if "PerExposure" not in names or names.index("PerExposure") > names.index("Responsivity"):
            raise ValueError(f"{path.name} is not normalised by integration time (no PerExposure "
                             "before Responsivity); regenerate it")
        return calibration
    from .io import from_lighttouch
    factors = from_lighttouch(path)
    integration = factors.metadata.get("integration_time")
    if not integration or integration <= 0:
        raise ValueError(f"{path.name}: no positive [Integration Time ms] header line")
    unit = lighttouch_unit(factors)
    if quantity_of(unit) is None:
        raise ValueError(f"{path.name}: header names no radiance or irradiance unit")
    per_count_ms = units.parse(unit) / (units.ureg.count / units.ureg.ms)
    # LightTouch marks samples without usable counts with 10000; they have no
    # responsivity, and applying 10000 would put spikes in calibrated spectra.
    values = np.asarray(factors.values, dtype=float)
    values = np.where(values == LIGHTTOUCH_INVALID, 0.0, values)
    response = factors.copy(values=values * integration,
                            value_unit=units.format_unit(per_count_ms),
                            label=f"{path.stem} responsivity")
    return Calibration(path.stem, [PerExposure(), Responsivity(response)], metadata={
        "convention": "PerExposure + Responsivity per (count/ms)",
        "calibration_integration_time_ms": float(integration),
        "source": path.name,
    })


def responsivity_of(calibration):
    """The responsivity Spectrum of a native amplitude Calibration."""
    for step in calibration.steps:
        if isinstance(step, Responsivity) and isinstance(step.responsivity, Spectrum):
            return step.responsivity
    raise ValueError("the calibration has no spectral Responsivity step")


# Evaluation: calibrated measurement against a certified source.

@dataclass(frozen=True)
class _Photometry:
    label: str
    Y: float
    x: float
    y: float
    cct: float


def _photometry(spectrum, label):
    color = spectrum.color(mode="absolute")
    return _Photometry(label, float(color.Y), float(color.x), float(color.y), float(color.cct))


@dataclass(frozen=True, repr=False)
class AmplitudeEvaluation(Result):
    """A calibrated measurement's (il)luminance, x, y and CCT against certified values."""
    measured: _Photometry
    certified: _Photometry
    comparison: object
    Y_unit: str
    ratio: dict

    @property
    def passed(self):
        return bool(self.comparison.passed)

    @property
    def summary(self):
        return {"passed": self.passed, "Y_measured": self.measured.Y, "Y_certified": self.certified.Y,
                "x_measured": self.measured.x, "x_certified": self.certified.x,
                "y_measured": self.measured.y, "y_certified": self.certified.y,
                "cct_measured": self.measured.cct, "cct_certified": self.certified.cct,
                **{f"ratio_{key}": value for key, value in self.ratio.items()}}

    def rows(self):
        result = []
        for path, record in self.comparison.records.items():
            relative = record.field.mode == "relative"
            result.append({
                "quantity": path, "certified": record.reference, "measured": record.measured,
                "difference": record.difference, "tolerance": record.field.tolerance,
                "relative": relative, "result": "PASS" if record.passed else "FAIL",
            })
        return result

    def table(self):
        names = {"Y": f"{'Luminance' if 'cd' in self.Y_unit else 'Illuminance'} ({self.Y_unit})",
                 "x": "CIE x", "y": "CIE y", "cct": "CCT (K)"}
        digits = {"Y": 5, "x": 4, "y": 4, "cct": 0}
        header = ["Quantity", "Certified", "Measured", "Difference", "Tolerance", "Result"]
        body = []
        for row in self.rows():
            q, d = row["quantity"], digits[row["quantity"]]
            fmt = (lambda v: f"{v:.{d}g}") if q == "Y" else (lambda v: f"{v:.{d}f}")
            difference = f"{100 * row['difference']:+.2f} %" if row["relative"] else f"{row['difference']:+.{d}f}"
            tolerance = f"{100 * row['tolerance']:.2f} %" if row["relative"] else f"{row['tolerance']:g}"
            body.append([names[q], fmt(row["certified"]), fmt(row["measured"]), difference,
                         tolerance, row["result"]])
        widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]
        lines = ["  ".join([r[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(r[1:], widths[1:])])
                 for r in [header, ["-" * w for w in widths]] + body]
        ratio = self.ratio
        lines += ["", f"Spectral ratio measured/certified, {ratio['band_nm'][0]:g}-{ratio['band_nm'][1]:g} nm: "
                      f"median {ratio['median']:.4f}, 5-95% {ratio['p5']:.4f}-{ratio['p95']:.4f}",
                  f"{'PASS' if self.passed else 'FAIL'}"]
        return "\n".join(lines)

    def __str__(self):
        return self.table()


def evaluate_amplitude(measured, certified, *, Y_tolerance=0.02, xy_tolerance=0.002,
                       cct_tolerance=30.0, band_nm=(380.0, 780.0)):
    """Compare a calibrated measurement of a source with its certified values.

    (Il)luminance is compared relatively (``Y_tolerance`` is a fraction), x
    and y and CCT absolutely, with :func:`gsmath.compare`. The spectral ratio
    in ``band_nm`` (on the certified grid, where it is above 10% of its peak)
    is reported but not judged. Both spectra need radiometric units; a
    LightTouch file's unit is read from its header.
    """
    certified = certified.copy(value_unit=certified.value_unit or lighttouch_unit(certified))
    for name, spectrum in (("measured", measured), ("certified", certified)):
        if quantity_of(spectrum.value_unit) is None:
            raise ValueError(f"the {name} spectrum needs a radiance or irradiance unit")
    if quantity_of(measured.value_unit) != quantity_of(certified.value_unit):
        raise ValueError("the measured and certified spectra are different quantities")
    ours, theirs = _photometry(measured, "measured"), _photometry(certified, "certified")
    profile = Profile("amplitude evaluation", {
        "Y": Field(getter=lambda p: p.Y, mode="relative", tolerance=Y_tolerance),
        "x": Field(getter=lambda p: p.x, tolerance=xy_tolerance),
        "y": Field(getter=lambda p: p.y, tolerance=xy_tolerance),
        "cct": Field(getter=lambda p: p.cct, unit="K", tolerance=cct_tolerance),
    })
    comparison = compare(ours, theirs, profile=profile)

    wavelength = np.asarray(certified.wavelength, dtype=float)
    values = units.convert_values(np.asarray(measured.values, dtype=float), measured.value_unit,
                                  certified.value_unit)
    resampled = np.interp(wavelength, measured.wavelength, values, left=np.nan, right=np.nan)
    in_band = (wavelength >= band_nm[0]) & (wavelength <= band_nm[1])
    reference = np.asarray(certified.values, dtype=float)
    bright = reference > 0.1 * np.nanmax(reference[in_band]) if in_band.any() else in_band
    selected = in_band & bright & np.isfinite(resampled) & (reference > 0)
    ratio = resampled[selected] / reference[selected]
    p5, p50, p95 = np.percentile(ratio, [5, 50, 95]) if len(ratio) else (np.nan,) * 3
    Y_unit = unit_text(measured.color(mode="absolute").XYZ_unit)
    return AmplitudeEvaluation(ours, theirs, comparison, Y_unit,
                               {"band_nm": list(band_nm), "median": float(p50), "p5": float(p5),
                                "p95": float(p95), "samples": int(selected.sum())})


def compare_responsivity(calibration, reference, *, band_nm=(380.0, 780.0), tolerance=0.02):
    """Ratio of two native amplitude calibrations' responsivities within a band.

    Returns a dict with the median, 5-95 % range and largest deviation from 1
    of ``calibration / reference`` on ``calibration``'s wavelengths, and
    ``passed`` when the 5-95 % range lies within ``tolerance`` of 1. The
    single worst sample is reported but not judged: it is usually the
    noisiest pixel at the band's edge.
    """
    ours, theirs = responsivity_of(calibration), responsivity_of(reference)
    values = units.convert_values(np.asarray(theirs.values, dtype=float), theirs.value_unit,
                                  ours.value_unit)
    wavelength = np.asarray(ours.wavelength, dtype=float)
    resampled = np.interp(wavelength, theirs.wavelength, values, left=np.nan, right=np.nan)
    mine = np.asarray(ours.values, dtype=float)
    selected = ((wavelength >= band_nm[0]) & (wavelength <= band_nm[1]) & np.isfinite(resampled)
                & (resampled > 0) & (mine > 0))
    if not selected.any():
        raise ValueError("the calibrations share no valid samples in the band")
    ratio = mine[selected] / resampled[selected]
    p5, p50, p95 = np.percentile(ratio, [5, 50, 95])
    worst = float(ratio[np.argmax(abs(ratio - 1))])
    return {"band_nm": list(band_nm), "samples": int(selected.sum()), "median": float(p50),
            "p5": float(p5), "p95": float(p95), "worst": worst, "tolerance": tolerance,
            "passed": bool(abs(p5 - 1) <= tolerance and abs(p95 - 1) <= tolerance),
            "wavelength": wavelength[selected].tolist(), "ratio": ratio.tolist()}
