"""High dynamic range: one reading made of several exposures.

An :class:`ExposureSeries` holds one :class:`~gsmath.Measurement` per
exposure (integration) time -- each with its own raw data and dark -- and
combines them when processed: every measurement is taken through the
calibration up to and including its dark subtraction (``combine_after``),
the results are combined element by element, and the rest of the
calibration runs once, on the combined data. It works the same for a
spectrometer's spectra and a camera's images -- a pixel of either clips
alike::

    series = ExposureSeries([short, medium, long], saturation=65535)
    spectrum = series.process(calibration)
    frame = ExposureSeries(raw_frames).process(camera)   # white level from the raw file

Combining: an element's rate (dark-subtracted counts per unit exposure) is
the exposure-weighted mean over the exposures where it is usable -- its raw
value below ``saturation`` and its processed value finite (so a pixel that
an earlier MaskSaturated step made NaN is simply left out) -- which comes to
its summed counts over its summed exposure. Where an element is usable in no
exposure, the shortest exposure's value is used (NaN if it was masked) and
it is counted in the report. The combined data is
expressed as counts at the longest exposure, so the remaining steps
(PerExposure, ...) apply to it as to a single reading of that exposure.

Combining before the remaining steps matters for any step that mixes
pixels: a stray-light correction would spread a saturated pixel's error into
its neighbours, so it must only see the combined data.

The processing report summarises how many elements used how many
exposures, whatever the data's size; :meth:`ExposureSeries.exposures_used`
gives the map itself, shaped like the data.

The series is stored with :meth:`write` / :meth:`read` like a Measurement, so
it can be processed (and combined) again with a later calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import units
from .measurement import Measurement


SATURATION_FRACTION = 0.98
"""``saturation="auto"``: this fraction of the recorded white level."""


@dataclass(frozen=True, eq=False)
class ExposureSeries:
    measurements: tuple
    """One Measurement per exposure, each with its exposure time."""
    saturation: float | str | None = "auto"
    """Raw value (counts) from which an element counts as saturated; None if
    nothing saturates; ``"auto"`` (the default): 98% of the white level the
    raw data records (a raw image's ``mosaic.white_level``, as
    MaskSaturated uses), else nothing saturates."""
    label: str = ""
    settings: dict = field(default_factory=dict)
    """Acquisition settings: the exposure multiple, limits, ..."""
    instrument: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    """The calibration in force when recorded, as a reference."""

    def __post_init__(self):
        measurements = tuple(self.measurements)
        object.__setattr__(self, "measurements", measurements)
        if not measurements:
            raise ValueError("an exposure series needs at least one measurement")
        if not all(isinstance(m, Measurement) for m in measurements):
            raise TypeError("an exposure series holds Measurements")
        if any(m.exposure is None for m in measurements):
            raise ValueError("every measurement in an exposure series needs its exposure time")
        if len({type(m.raw) for m in measurements}) > 1:
            raise TypeError("every measurement in an exposure series must hold the same kind of data")
        if len({np.shape(_values(m.raw)) for m in measurements}) > 1:
            raise ValueError("every measurement in an exposure series must have the same shape")
        if not (self.saturation is None or self.saturation == "auto"
                or isinstance(self.saturation, (int, float, np.number))):
            raise TypeError("saturation is a raw value, None or 'auto'")

    @property
    def saturation_level(self) -> float | None:
        """The raw value from which an element counts as saturated (None:
        nothing saturates), with ``"auto"`` resolved."""
        if self.saturation != "auto":
            return None if self.saturation is None else float(self.saturation)
        mosaic = getattr(self.measurements[0].raw, "mosaic", None)
        white = None if mosaic is None else mosaic.white_level
        return None if white is None else SATURATION_FRACTION * float(white)

    def usable(self) -> np.ndarray:
        """``(exposures, *shape)``: True where an exposure's raw value is
        finite and below the saturation level."""
        raw = np.array([_values(m.raw) for m in self.measurements], dtype=float)
        level = self.saturation_level
        return np.isfinite(raw) if level is None else np.isfinite(raw) & (raw < level)

    def exposures_used(self):
        """How many exposures each element can use (from the raw values),
        shaped like the data -- an Image or Spectrum of counts (0: saturated
        in every exposure)."""
        used = self.usable().sum(axis=0)
        return _with_values(self.measurements[0].raw, used, label=f"{self.label} exposures used")

    @property
    def exposures(self) -> list:
        """The exposure times, in the order measured."""
        return [m.exposure for m in self.measurements]

    def process(self, calibration, *, combine_after: str | None = None,
                until: str | None = None):
        """Take each measurement through `calibration` up to and including
        the step `combine_after` -- by default its dark subtraction, the
        first step that ``removes_dark`` -- combine them (see the module
        docstring), then apply the remaining steps, stopping after `until`."""
        names = [step.name for step in calibration.steps]
        if combine_after is None:
            darks = [step.name for step in calibration.steps if step.removes_dark]
            if not darks:
                raise ValueError(f"{calibration.name!r} has no dark subtraction (no step "
                                 "removes_dark); name the step to combine after")
            combine_after = darks[0]
        if combine_after not in names:
            raise ValueError(f"{calibration.name!r} has no step {combine_after!r} to combine after")
        if until is not None and until not in names[names.index(combine_after):]:
            raise ValueError(f"{calibration.name!r}: step {until!r} does not come "
                             f"at or after {combine_after!r}")

        parts = [m.process(calibration, until=combine_after) for m in self.measurements]
        combined, report = self._combine(parts)
        before = parts[0].metadata["processing"]

        reference = self.measurements[int(np.argmax(self._times()))]
        combined = combined.copy(metadata={k: v for k, v in combined.metadata.items()
                                           if k != "processing"})
        single = Measurement(raw=combined, exposure=reference.exposure, label=self.label,
                             settings=self.settings, instrument=self.instrument,
                             metadata=self.metadata, calibration=self.calibration)
        if until == combine_after:
            result = combined.copy(label=self.label or combined.label)
            rest = {"applied": [], "skipped": []}
        else:
            result = calibration.process(single, until=until, after=combine_after)
            rest = result.metadata["processing"]

        skipped = sorted({name for part in parts
                          for name in part.metadata["processing"]["skipped"]},
                         key=names.index) + rest["skipped"]
        reports = dict(before.get("reports", {}))
        reports.update(rest.get("reports", {}))
        reports["ExposureSeries"] = report
        metadata = dict(result.metadata)
        metadata["processing"] = {
            "calibration": calibration.name,
            "fingerprint": calibration.fingerprint,
            "applied": before["applied"] + rest["applied"],
            "skipped": skipped,
            "complete": until is None and not skipped,
            "combined_after": combine_after,
            "reports": reports,
        }
        return result.copy(metadata=metadata)

    def write(self, path, *, indent=2):
        from .io.native import write_exposure_series
        return write_exposure_series(self, path, indent=indent)

    @classmethod
    def read(cls, path):
        from .io.native import read_exposure_series
        return read_exposure_series(path)

    def _times(self) -> np.ndarray:
        """Exposure times as plain numbers, in the unit of the first."""
        unit = self.measurements[0].exposure.units
        return np.array([float(m.exposure.to(unit).magnitude) for m in self.measurements])

    def _combine(self, parts):
        """The exposure-weighted mean rate of each pixel over the exposures
        where it isn't saturated, as counts at the longest exposure."""
        times = self._times()
        signal = np.array([_values(part) for part in parts], dtype=float)
        usable = self.usable() & np.isfinite(signal)
        weights = usable * times.reshape((-1,) + (1,) * (signal.ndim - 1))
        used = usable.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(usable, signal, 0.0).sum(axis=0) / weights.sum(axis=0)
        shortest = int(np.argmin(times))
        saturated = used == 0
        rate = np.where(saturated, signal[shortest] / times[shortest], rate)

        combined = rate * times.max()
        reference = parts[int(np.argmax(times))]
        unit = units.format_unit(self.measurements[0].exposure.units)
        counts = np.bincount(used.ravel(), minlength=len(times) + 1)
        report = {
            "exposures": [float(t) for t in times],
            "exposure_unit": unit,
            "saturation": self.saturation_level,
            # elements by how many exposures they used: its size doesn't
            # grow with the data (exposures_used() gives the map)
            "elements_by_exposures_used": {str(k): int(n) for k, n in enumerate(counts)},
            "saturated_in_all": int(saturated.sum()),
        }
        return _with_values(reference, combined), report


def _values(data):
    return getattr(data, "values", data)


def _with_values(data, values, **changes):
    if hasattr(data, "copy") and hasattr(data, "values"):
        if changes:  # a map of counts: no unit
            return data.copy(values=values, value_unit=None, **changes)
        return data.copy(values=values)
    return values
