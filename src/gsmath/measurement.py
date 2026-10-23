"""What an instrument recorded, kept so it can be processed (again) later.

A :class:`Measurement` holds raw data -- a gsmath data object in instrument
units, e.g. a Spectrum in counts on the instrument's wavelength axis -- with
what it was recorded with: an optional dark (background) of the same kind,
the exposure (integration) time, acquisition settings, the instrument's
identity and free metadata. A :class:`~gsmath.Calibration` turns it into
physical data::

    spectrum = measurement.process(calibration)

Measurements store with :meth:`write` / :meth:`read` in the native ``.gsmath``
format, so a run can be re-processed with a later calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import units


@dataclass(frozen=True, eq=False)
class Measurement:
    raw: object
    """Raw data in instrument units (a Spectrum in counts, ...)."""
    dark: object = None
    """Dark or background data of the same kind, if recorded."""
    exposure: object = None
    """Exposure (integration) time, a quantity such as ``units.Q_(160.8, 'ms')``
    -- or one per channel (``Q_([10, 40, 20], 'ms')``), for channels taken
    separately, e.g. through a filter wheel."""
    label: str = ""
    settings: dict = field(default_factory=dict)
    """Acquisition settings: averages, gain, pixel format, ..."""
    instrument: dict = field(default_factory=dict)
    """Model, serial number, driver, ..."""
    metadata: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    """The calibration in force when recorded, as a reference
    (:attr:`Calibration.reference`: name and fingerprint) -- the calibration
    itself is saved once, separately."""
    temperature: object = None
    """The sensor's temperature during the acquisition, a quantity such as
    ``units.Q_(31.5, 'degC')``: what a temperature-dependent dark model needs."""

    def __post_init__(self):
        if self.dark is not None and type(self.dark) is not type(self.raw):
            raise TypeError("dark must be the same kind of data as raw")
        if self.exposure is not None and not units.is_quantity(self.exposure):
            raise TypeError("exposure must be a quantity with a time unit, e.g. Q_(10, 'ms')")
        if self.exposure is not None and np.ndim(self.exposure.magnitude) > 1:
            raise ValueError("exposure is one time, or one per channel")
        if self.temperature is not None and not (units.is_quantity(self.temperature)
                                                 and self.temperature.check("[temperature]")):
            raise TypeError("temperature must be a temperature quantity, e.g. Q_(31.5, 'degC')")

    def process(self, calibration, *, until: str | None = None):
        """Apply `calibration`'s steps; see :meth:`Calibration.process`."""
        return calibration.process(self, until=until)

    def write(self, path, *, indent=2):
        from .io.native import write_measurement
        return write_measurement(self, path, indent=indent)

    @classmethod
    def read(cls, path):
        from .io.native import read_measurement
        return read_measurement(path)
