"""Peak detection and position/width estimators for spectral features -- or
any one-channel :class:`~gsmath.Series` (positions along its axis; the
``*_wavelength`` names are kept from spectra)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize
import scipy.signal

from .base import AnalysisResult
from ..lineshapes import gaussian

if TYPE_CHECKING:
    from ..spectrum import Spectrum, SpectrumView
    

@dataclass(frozen=True)
class PeakAnalysis(AnalysisResult):
    """Detect peaks and refine their positions using geometric and fitted methods.

    SciPy performs initial peak detection. Derived properties then operate on
    local windows, with ``primary_only`` selecting the strongest peak for
    scalar-oriented workflows.
    """
    spectrum: Spectrum | SpectrumView
    primary_only: bool = True
    prominence: float | None = None
    width: float | None = 1
    rel_height: float = 0.5
    poly_order: int = 2
    fit_radius: int = 3
    interp_points: int = 1000

    @property
    def units(self) -> dict[str, object]:
        """Return units for wavelength-, width-, and value-like results."""
        wavelength = self.spectrum.axis_unit
        value = self.spectrum.value_unit
        return {
            "peak_wavelength": wavelength,
            "peak_wavelengths": wavelength,
            "peak_value": value,
            "peak_values": value,
            "center_wavelength": wavelength,
            "center_wavelengths": wavelength,
            "fwhm": wavelength,
            "fwhms": wavelength,
            "peak_wavelength_polyfit": wavelength,
            "peak_wavelengths_polyfit": wavelength,
            "peak_wavelength_gaussian": wavelength,
            "peak_wavelengths_gaussian": wavelength,
            "centroid_wavelength": wavelength,
            "centroid_wavelengths": wavelength,
        }

    def __len__(self) -> int:
        return len(self.peak_indices)

    @property
    def summary(self) -> dict[str, object]:
        """Return the configured scalar or multi-peak result fields."""
        if self.primary_only:
            return {
                "peak_wavelength" : self.peak_wavelength,
                "peak_value" : self.peak_value,
                "center_wavelength" : self.center_wavelength,
                "fwhm" : self.fwhm,
                "peak_wavelength_polyfit" : self.peak_wavelength_polyfit,
                # "peak_wavelength_gaussian" : self.peak_wavelength_gaussian,
                "centroid_wavelength" : self.centroid_wavelength,
                }
        return {
            "peak_count" : len(self),
            "peak_wavelengths" : self.peak_wavelengths,
            "peak_values" : self.peak_values,
            "center_wavelengths" : self.center_wavelengths,
            "fwhms" : self.fwhms,
            "peak_wavelengths_polyfit" : self.peak_wavelengths_polyfit,
            # "peak_wavelengths_gaussian" : self.peak_wavelengths_gaussian,
            "centroid_wavelengths" : self.centroid_wavelengths,
            }

    @cached_property
    def all_peaks_result(self):
        """Return every detected peak and its derived measurements."""
        y0 = self.spectrum.values

        if y0.size == 0:
            return np.array([], dtype=int), {}

        if not np.any(np.isfinite(y0)):
            return np.array([], dtype=int), {}

        ymax = np.nanmax(y0)

        if not np.isfinite(ymax) or ymax <= 0:
            return np.array([], dtype=int), {}
        
        return scipy.signal.find_peaks(
            y0 / ymax,
            prominence=self.prominence,
            width=self.width,
            rel_height=self.rel_height,
            )

    @cached_property
    def selected_peaks_result(self):
        """Return either the primary peak or all peaks per configuration."""
        peaks, props = self.all_peaks_result

        if not self.primary_only or len(peaks) == 0:
            return peaks, props

        y = self.spectrum.values
        j = np.nanargmax(y[peaks])

        peaks = peaks[[j]]
        props = {k: np.asarray(v)[[j]] for k, v in props.items()}
        return peaks, props
    
    @property
    def peak_indices(self):
        """Return array indices found by :func:`scipy.signal.find_peaks`."""
        peaks, props = self.selected_peaks_result
        return peaks

    @property
    def properties(self):
        """Return the property mapping produced during peak detection."""
        peaks, props = self.selected_peaks_result
        return props

    @property
    def peak_wavelengths(self):
        """Return sample-grid wavelengths at detected peak indices."""
        return self.spectrum.axis[self.peak_indices]

    @property
    def peak_wavelength(self):
        """Return the primary peak's sample-grid wavelength."""
        self._require_primary_only("peak_wavelength")
        return self._primary(self.peak_wavelengths)

    @property
    def peak_values(self):
        """Return spectral values at all detected peak indices."""
        return self.spectrum.values[self.peak_indices]

    @property
    def peak_value(self):
        """Return the primary detected peak's spectral value."""
        self._require_primary_only("peak_value")
        return self._primary(self.peak_values)

    @property
    def center_wavelengths(self):
        """Return half-height midpoint wavelengths for detected peaks."""
        x = self.spectrum.axis
        i = np.arange(x.size)
        centers = (self.properties["left_ips"] + self.properties["right_ips"]) / 2
        return np.interp(centers, i, x)

    @property
    def center_wavelength(self):
        """Return the primary peak's half-height midpoint wavelength."""
        self._require_primary_only("center_wavelength")
        return self._primary(self.center_wavelengths)
    
    @property
    def fwhms(self):
        """Return full widths at the configured relative peak height."""
        x = self.spectrum.axis
        i = np.arange(x.size)
        left = np.interp(self.properties['left_ips'], i, x)
        right = np.interp(self.properties['right_ips'], i, x)
        return right - left

    @property
    def fwhm(self):
        """Return the primary peak's full width at half maximum."""
        self._require_primary_only("fwhm")
        return self._primary(self.fwhms)

    @property
    def peak_wavelengths_polyfit(self):
        """Refine peak locations from local polynomial vertex fits."""
        results = []

        for x_fit, y_fit, x_dense in self._peak_windows():
            if x_fit.size < self.poly_order + 1:
                results.append(np.nan)
                continue
            
            coeff = np.polyfit(x_fit, y_fit, self.poly_order)
            y_dense = np.polyval(coeff, x_dense)
            results.append(x_dense[np.nanargmax(y_dense)])

        return np.asarray(results)

    @property
    def peak_wavelength_polyfit(self):
        """Return the primary peak's polynomial-refined wavelength."""
        self._require_primary_only("peak_wavelength_polyfit")
        return self._primary(self.peak_wavelengths_polyfit)    

    @property
    def peak_wavelengths_gaussian(self):
        """Refine peak locations by fitting local Gaussian line shapes."""
        results = []
        
        for x_fit, y_fit, x_dense in self._peak_windows():
            if x_fit.size < 4:
                results.append(np.nan)
                continue
                
            p0 = [
                np.nanmax(y_fit) - np.nanmin(y_fit),
                x_fit[np.nanargmax(y_fit)],
                np.ptp(x_fit)/4,
                np.nanmin(y_fit)
            ]

            try:
                popt, _ = scipy.optimize.curve_fit(gaussian, x_fit, y_fit, p0=p0)
                y_dense = gaussian(x_dense, *popt)
                results.append(x_dense[np.argmax(y_dense)])
            except RuntimeError:
                results.append(np.nan)

        return np.asarray(results)

    @property
    def peak_wavelength_gaussian(self):
        """Return the primary peak's Gaussian-fit wavelength."""
        self._require_primary_only("peak_wavelength_gaussian")
        return self._primary(self.peak_wavelengths_gaussian)
    
    @property
    def centroid_wavelengths(self):
        """Return value-weighted centroids within each peak window."""
        results = []

        for x_fit, y_fit, _ in self._peak_windows():
            area = np.trapezoid(y_fit, x_fit)

            if area == 0:
                results.append(np.nan)
                continue

            results.append(np.trapezoid(x_fit * y_fit, x_fit) / area)

        return np.asarray(results)

    @property
    def centroid_wavelength(self):
        """Return the primary peak's local value-weighted centroid."""
        self._require_primary_only("centroid_wavelength")
        return self._primary(self.centroid_wavelengths)

    # convenience functions for getting single (primary) peak
    def _primary(self, values):
        """Select the strongest peak's value from a per-peak array."""
        return values[0] if values.size else np.nan
    
    def _require_primary_only(self, name):
        """Reject scalar-only properties when multiple peaks were requested."""
        if not self.primary_only:
            raise ValueError(
                f"{name} is only available when primary_only=True. "
                "Use the plural property instead."
            )
    
    def _peak_windows(self):
        """Yield non-overlapping local index windows around detected peaks."""
        x = self.spectrum.axis
        y = self.spectrum.values

        for i0 in self.peak_indices:
            lo = max(0, i0 - self.fit_radius)
            hi = min(len(x), i0 + self.fit_radius + 1)

            yield (
                x[lo:hi],
                y[lo:hi],
                np.linspace(x[lo], x[hi-1], self.interp_points),
            )
