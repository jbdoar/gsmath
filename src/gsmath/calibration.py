"""Declarative measurement equations: a Calibration is an ordered list of Steps.

Like a :class:`~gsmath.Profile` describes what to extract from data, a
:class:`Calibration` describes how raw instrument data becomes physical data,
and :meth:`Measurement.process <gsmath.Measurement.process>` walks it. Most
steps are plain unit-aware arithmetic on the data object, so they work for
any data type that supports it; a few are specific to one (StrayLight works
on spectra)::

    radiance = Calibration("spectral radiance", [
        SubtractDark(),
        StrayLight(correction),
        PerExposure(),
        Responsivity(responsivity),   # e.g. µW/cm²/sr/nm per (count/ms)
    ])
    spectrum = measurement.process(radiance)

A calibration saves with :meth:`Calibration.write` (steps and all their
data) and is identified by its content :attr:`~Calibration.fingerprint`, so a
measurement can refer to the calibration it was recorded under.

A step whose input a measurement lacks (SubtractDark without a dark) raises,
unless it is ``optional``, when it is skipped. The processed data's metadata
records the calibration's name and fingerprint and which steps were applied
and skipped, so partly processed data can't pass for fully calibrated data.
"""

from __future__ import annotations

import hashlib
import json
import numbers

import numpy as np

from . import units


STEP_TYPES: dict[str, type] = {}
"""Step classes by name, for reading calibrations back; every subclass of
Step registers itself."""


class Step:
    """One stage of a measurement equation."""

    removes_dark = False
    """True for a step that subtracts the dark (or background) signal --
    whatever it is called, so code that needs "after the dark" finds it."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        STEP_TYPES[cls.__name__] = cls

    def __init__(self, *, optional: bool = False):
        self.optional = optional

    @property
    def name(self) -> str:
        return type(self).__name__

    def missing(self, measurement) -> str | None:
        """What this step needs that `measurement` lacks, or None."""
        return None

    def apply(self, data, measurement):
        raise NotImplementedError

    def parameters(self) -> dict:
        """What determines this step's effect, for the calibration's
        fingerprint: by default its :meth:`arguments`, each fingerprinted.
        (Steps whose fingerprints predate this default keep their own, so
        saved calibrations still load.)"""
        return {name: fingerprint_value(value) for name, value in self.arguments().items()}

    def arguments(self) -> dict:
        """The constructor arguments besides ``optional``, for saving the step."""
        return {}

    def __repr__(self) -> str:
        return f"{self.name}({'optional' if self.optional else ''})"


class SubtractDark(Step):
    """Subtract the measurement's dark (or background) data."""

    removes_dark = True

    def missing(self, measurement):
        return None if measurement.dark is not None else "a dark"

    def apply(self, data, measurement):
        return data - measurement.dark


class PerExposure(Step):
    """Divide by the measurement's exposure (integration) time."""

    def missing(self, measurement):
        return None if measurement.exposure is not None else "an exposure time"

    def apply(self, data, measurement):
        exposure = measurement.exposure
        if np.ndim(exposure.magnitude) == 1:  # one per channel
            channels = getattr(data, "n_channels", None)
            if channels is None or np.ndim(data.values) != 3 or channels != exposure.magnitude.size:
                raise ValueError(f"{exposure.magnitude.size} exposures need data with as many "
                                 "channels")
        return data / exposure


class Responsivity(Step):
    """Multiply by a responsivity: a number or quantity, or data on the same
    grid (a Spectrum for spectra; per-pixel data for images)."""

    def __init__(self, responsivity, *, optional: bool = False):
        super().__init__(optional=optional)
        self.responsivity = responsivity

    def apply(self, data, measurement):
        return data * self.responsivity

    def arguments(self):
        return {"responsivity": self.responsivity}


class FlatField(Step):
    """Divide by a flat field: the response to uniform light, normalised
    per channel to its mean (or used as given with ``normalize=False``) so
    the correction evens out the response without changing the units. An
    image of the measurement's shape; a single-channel flat applies to
    every channel. On a raw colour-filter mosaic, each colour's sites are
    normalised to their own mean, so the flat source's colour doesn't
    change the data's colour balance."""

    def __init__(self, flat, *, normalize: bool = True, optional: bool = False):
        super().__init__(optional=optional)
        self.flat = flat
        self.normalize = bool(normalize)

    def apply(self, data, measurement):
        from .mosaic import mosaic_of

        flat = np.asarray(self.flat.values, dtype=float)
        if flat.shape[:2] != data.values.shape[:2]:
            raise ValueError(f"the flat field's shape {flat.shape[:2]} doesn't match "
                             f"the data {data.values.shape[:2]}")
        mosaic = mosaic_of(data)
        if self.normalize and mosaic is not None and flat.ndim == 2:
            flat = flat.copy()
            for mask in mosaic.masks(flat.shape).values():
                flat[mask] /= np.nanmean(flat[mask])
        elif self.normalize:
            flat = flat / np.nanmean(flat, axis=(0, 1))
        if flat.ndim == 2 and data.values.ndim == 3:
            flat = flat[:, :, None]
        return data.copy(values=np.asarray(data.values, dtype=float) / flat)

    def arguments(self):
        return {"flat": self.flat, "normalize": self.normalize}


class ColorMatrix(Step):
    """Mix an image's channels by a matrix (``(outputs, inputs)``): camera
    R, G, B to X, Y, Z, say, after demosaicing. ``unit`` is the matrix's own
    unit, e.g. ``"cd/m^2 / (count/ms)"`` for a matrix fitted against a
    spectroradiometer (see :func:`gsmath.fit_color_matrix`)."""

    def __init__(self, matrix, *, channels=("X", "Y", "Z"), unit=None, optional: bool = False):
        super().__init__(optional=optional)
        self.matrix = np.asarray(matrix, dtype=float)
        self.channels = tuple(channels)
        self.unit = units.parse(unit)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != len(self.channels):
            raise ValueError("matrix must be (outputs, inputs), one row per output channel")

    def apply(self, data, measurement):
        from .image_transforms import apply_matrix
        return apply_matrix(data, self.matrix, channels=self.channels, unit=self.unit)

    def arguments(self):
        arguments = {"matrix": self.matrix, "channels": list(self.channels)}
        if self.unit is not None:
            arguments["unit"] = units.format_unit(self.unit)
        return arguments


class StrayLight(Step):
    """Apply a stray-light correction matrix to a spectrum."""

    def __init__(self, correction, *, optional: bool = False):
        super().__init__(optional=optional)
        self.correction = correction

    def apply(self, data, measurement):
        if not np.array_equal(data.wavelength, self.correction.wavelength):
            raise ValueError("the spectrum's wavelength grid doesn't match the stray-light matrix")
        return data.copy(values=self.correction.apply_matrix(np.asarray(data.values, dtype=float)))

    def parameters(self):  # predates the default: kept for saved fingerprints
        return {"wavelength": fingerprint_value(self.correction.wavelength),
                "matrix": fingerprint_value(self.correction.values)}

    def arguments(self):
        return {"correction": self.correction}


class WavelengthAxis(Step):
    """Apply an identified lamp solution to the same detector sample layout.

    Place after dark subtraction and any corrections tied to the original
    grid, and before wavelength-based analysis or resampling. Responsivity
    spectra used afterward must be on the calibrated grid.
    """

    def __init__(self, solution, *, optional=False):
        from .wavelength import WavelengthCalibration
        if not isinstance(solution, WavelengthCalibration):
            raise TypeError("solution must be a WavelengthCalibration")
        super().__init__(optional=optional)
        self.solution = solution

    def apply(self, data, measurement):
        result = self.solution.apply(data)
        metadata = dict(result.metadata)
        metadata["step_report"] = self.solution.report
        return result.copy(metadata=metadata)

    def parameters(self):  # predates the default: kept for saved fingerprints
        return self.solution.report

    def arguments(self):
        return {"solution": self.solution}


class Calibration:
    """A named, ordered list of steps from raw data to physical data."""

    def __init__(self, name: str, steps, *, metadata: dict | None = None):
        self.name = name
        self.steps = list(steps)
        self.metadata = dict(metadata or {})

    def __repr__(self) -> str:
        return f"Calibration({self.name!r}, {self.steps!r})"

    @property
    def reference(self) -> dict:
        """How a measurement or result refers to this calibration."""
        return {"name": self.name, "fingerprint": self.fingerprint}

    def write(self, path, *, indent=2):
        """Save to the native ``.gsmath`` format (steps and their data),
        e.g. once per run as ``calibrations/<fingerprint>.gsmath``."""
        from .io.native import write_calibration
        return write_calibration(self, path, indent=indent)

    @classmethod
    def read(cls, path):
        """Load a saved calibration; raises if its contents no longer match
        the fingerprint it was saved with."""
        from .io.native import read_calibration
        return read_calibration(path)

    @property
    def fingerprint(self) -> str:
        """A hash of the steps and everything that determines their effect."""
        description = [[step.name, step.optional, step.parameters()] for step in self.steps]
        text = json.dumps([self.name, description], sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def with_steps(self, steps, *, after: str | None = None, name: str | None = None):
        """This calibration with `steps` inserted after the step named
        `after` (at the start if None): e.g. cleaning raw data right after
        the dark is subtracted. A new calibration, with its own fingerprint."""
        steps = list(steps)
        names = [step.name for step in self.steps]
        if after is not None and after not in names:
            raise ValueError(f"{self.name!r} has no step {after!r}")
        at = 0 if after is None else names.index(after) + 1
        added = "+".join(step.name for step in steps)
        return Calibration(name or f"{self.name} + {added}",
                           [*self.steps[:at], *steps, *self.steps[at:]],
                           metadata=dict(self.metadata))

    def process(self, measurement, *, until: str | None = None, after: str | None = None):
        """Apply the steps to `measurement`'s raw data, stopping after the
        step named `until` if given. With `after`, start after the step of
        that name instead, as on data that has been through the steps before
        it (e.g. exposures combined after SubtractDark); the result is then
        not marked complete, as those steps were not applied here."""
        names = [step.name for step in self.steps]
        for name in (until, after):
            if name is not None and name not in names:
                raise ValueError(f"{self.name!r} has no step {name!r}")
        steps = self.steps if after is None else self.steps[names.index(after) + 1:]
        if until is not None and until not in {step.name for step in steps}:
            raise ValueError(f"{self.name!r}: step {until!r} does not come after {after!r}")
        data, applied, skipped, reports = measurement.raw, [], [], {}
        for step in steps:
            need = step.missing(measurement)
            if need is not None:
                if not step.optional:
                    raise ValueError(f"{self.name!r}: {step.name} needs {need}")
                skipped.append(step.name)
            else:
                data = step.apply(data, measurement)
                applied.append(step.name)
                found = getattr(data, "metadata", None) or {}
                if "step_report" in found:  # what the step did (Despike: the spikes)
                    found = dict(found)
                    reports[step.name] = found.pop("step_report")
                    data = data.copy(metadata=found)
            if step.name == until:
                break
        metadata = dict(getattr(measurement.raw, "metadata", {}) or {})
        metadata["processing"] = {
            "calibration": self.name,
            "fingerprint": self.fingerprint,
            "applied": applied,
            "skipped": skipped,
            "complete": until is None and after is None and not skipped,
        }
        if after is not None:
            metadata["processing"]["after"] = after
        if reports:
            metadata["processing"]["reports"] = reports
        label = measurement.label or getattr(measurement.raw, "label", "")
        return data.copy(label=label, metadata=metadata)


def fingerprint_value(value):
    """A JSON-able stand-in for a step argument, determined by its content:
    None, flags and text as themselves, numbers as floats, lists item by
    item, quantities as magnitude and unit, spectra and images by their data
    and units, arrays by a hash of their bytes."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Real):
        return float(value)
    if units.is_quantity(value):
        return [float(value.magnitude), units.format_unit(value.units)]
    if isinstance(value, (list, tuple)):
        return [fingerprint_value(item) for item in value]
    if hasattr(value, "values") and hasattr(value, "wavelength"):  # a Spectrum
        return [fingerprint_value(value.wavelength), fingerprint_value(value.values),
                units.format_unit(value.value_unit)]
    if hasattr(value, "values") and hasattr(value, "pixel_map"):  # an Image
        pixel_map = value.pixel_map
        found = [fingerprint_value(value.values), units.format_unit(value.value_unit),
                 value.channels and list(value.channels),
                 pixel_map and [fingerprint_value(pixel_map.x), fingerprint_value(pixel_map.y),
                                units.format_unit(pixel_map.unit)]]
        if getattr(value, "mosaic", None) is not None:
            found.append(value.mosaic.to_dict())
        return found
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes() + str(array.dtype).encode()).hexdigest()[:16]
