"""Integrated power and spectral centroid calculations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .base import AnalysisResult
from .. import units

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
        }

    @property
    def summary(self) -> dict[str, object]:
        """Return integral and centroid values."""
        return {
            "integral" : self.integral,
            "centroid_wavelength" : self.centroid_wavelength,
            }
    
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
