"""Descriptive statistics for sampled spectral values."""

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
class StatisticsAnalysis(AnalysisResult):
    """Expose cached scalar statistics for a spectrum."""
    spectrum: Spectrum | SpectrumView

    @property
    def units(self) -> dict[str, object]:
        """Return value units for statistics and dimensionless sample count."""
        value = self.spectrum.value_unit
        return {
            "mean": value,
            "min": value,
            "max": value,
            "std": value,
            "median": value,
            "rms": value,
            "dynamic_range": value,
            "n": units.ureg.dimensionless,
        }

    @property
    def summary(self) -> dict[str, object]:
        """Return all supported descriptive statistics."""
        return {
            "mean" : self.mean,
            "min" : self.min,
            "max" : self.max,
            "std" : self.std,
            "median" : self.median,
            "rms" : self.rms,
            "dynamic_range": self.dynamic_range,
            "n" : self.n,
            }
    
    @cached_property
    def mean(self) -> float:
        """Return the arithmetic mean."""
        if self._is_empty:
            return np.nan
        return np.mean(self.spectrum.values)

    @cached_property
    def min(self) -> float:
        """Return the minimum value."""
        if self._is_empty:
            return np.nan
        return np.min(self.spectrum.values)

    @cached_property
    def max(self) -> float:
        """Return the maximum value."""
        if self._is_empty:
            return np.nan
        return np.max(self.spectrum.values)

    @cached_property
    def std(self) -> float:
        """Return the population standard deviation."""
        if self._is_empty:
            return np.nan
        return np.std(self.spectrum.values)

    @cached_property
    def median(self) -> float:
        """Return the median value."""
        if self._is_empty:
            return np.nan
        return np.median(self.spectrum.values)

    @cached_property
    def rms(self) -> float:
        """Return the root-mean-square value."""
        if self._is_empty:
            return np.nan
        return np.sqrt(np.mean(self.spectrum.values**2))

    @cached_property
    def dynamic_range(self) -> float:
        """Return maximum minus minimum."""
        return self.max - self.min

    @cached_property
    def n(self) -> int:
        """Return the number of samples."""
        return self.spectrum.values.size
