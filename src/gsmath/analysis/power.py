"""Integrated power and spectral centroid calculations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .base import AnalysisResult
from .. import units
from ..constants import PLANCK_CONSTANT, SPEED_OF_LIGHT

if TYPE_CHECKING:
    from ..spectrum import Spectrum, SpectrumView



@dataclass(frozen=True)
class PowerAnalysis(AnalysisResult):
    """Compute integral measures over a spectrum's wavelength axis."""
    # this is also where whole-spectrum centroid_wavelength, peak_spectral_power, mean power, weighted power, photopic power, photon flux, etc belong.
    spectrum: Spectrum | SpectrumView

    @property
    def units(self) -> dict[str, object]:
        """Return units derived from the spectrum axes."""
        return {
            "integral": units.multiply_units(
                self.spectrum.value_unit,
                self.spectrum.wavelength_unit,
            ),
            "centroid_wavelength": self.spectrum.wavelength_unit,
            "photon_flux": self._photon_flux_unit,
        }

    @property
    def summary(self) -> dict[str, object]:
        """Return integral and centroid values."""
        summary = {
            "integral" : self.integral,
            "centroid_wavelength" : self.centroid_wavelength,
        }
        if self._photon_flux_unit is not None:
            summary["photon_flux"] = self.photon_flux
        return summary
    
    @cached_property
    def integral(self):
        """Integrate spectral values using the trapezoidal rule."""
        x = self.spectrum.wavelength
        y = self.spectrum.values

        if self._is_empty:
            return np.nan
        
        return np.trapezoid(y, x)

    @cached_property
    def centroid_wavelength(self):
        """Return the power-weighted mean wavelength."""
        x = self.spectrum.wavelength
        y = self.spectrum.values

        area = np.trapezoid(y, x)
        if not np.isfinite(area) or area == 0:
            return np.nan
      
        return np.trapezoid(x * y, x) / area

    @cached_property
    def _photon_flux_unit(self):
        """Photon-rate unit for radiometric spectral densities, if known."""
        if self.spectrum.value_unit is None or self.spectrum.wavelength_unit is None:
            return None

        integrated = units.multiply_units(
            self.spectrum.value_unit,
            self.spectrum.wavelength_unit,
        )
        quantity = units.Q_(1, integrated)
        power = units.ureg.W
        area = units.ureg.m**2
        is_irradiance = quantity.is_compatible_with(power / area)
        if not (quantity.is_compatible_with(power) or is_irradiance):
            return None

        result = units.ureg.photon / units.ureg.s
        if is_irradiance:
            result /= area
        if "steradian" in str(integrated):
            result /= units.ureg.sr
        return result

    @cached_property
    def photon_flux(self):
        """Integrate photon rate from a radiometric spectral density.

        Geometry in the input unit is retained: spectral power produces
        photons/s, irradiance produces photons/(m^2 s), and radiance retains
        its steradian denominator. The result unit is convertible to molar
        photon flux, including ``umol / m^2 / s`` for PPFD-style reporting.
        """
        target_unit = self._photon_flux_unit
        if target_unit is None:
            raise ValueError(
                "Photon flux requires radiometric spectral values such as "
                "W/nm, W/m^2/nm, or W/m^2/sr/nm."
            )
        if self._is_empty:
            return np.nan

        wavelength_m = units.convert_values(
            self.spectrum.wavelength,
            self.spectrum.wavelength_unit,
            units.ureg.m,
        )
        if np.any(wavelength_m <= 0):
            raise ValueError("Photon flux requires positive wavelengths.")

        photon_energy = units.Q_(
            PLANCK_CONSTANT * SPEED_OF_LIGHT / wavelength_m,
            units.ureg.J,
        )
        spectral_rate = (
            units.Q_(self.spectrum.values, self.spectrum.value_unit)
            / photon_energy
            * units.ureg.photon
        )
        integrated = units.Q_(
            np.trapezoid(spectral_rate.magnitude, self.spectrum.wavelength),
            spectral_rate.units * self.spectrum.wavelength_unit,
        )
        return integrated.to(target_unit).magnitude
