"""CIE tristimulus, chromaticity, and correlated color calculations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .base import AnalysisResult
from .. import units
from ..constants import K_M
from ..reference.cmfs import CMFS
from ..reference.illuminants import ILLUMINANTS
from .colorimetry import (  # noqa: F401  (xyz_to_lab, ... re-exported)
    Colorimetry, reference_white_XYZ, validate_white, xyz_to_lab, xyz_to_luv, xyz_to_uv,
    xyz_to_uv_prime, xyz_to_xy,
)
from .planckian import ColorTemperatureResult

if TYPE_CHECKING:
    from ..spectrum import Spectrum, SpectrumView


def _cross_2d(left, right):
    return left[0] * right[1] - left[1] * right[0]


def _ray_segment_intersections(origin, direction, points, *, close=False):
    """Return ``(ray distance, segment fraction, segment index)`` hits."""
    count = len(points) if close else len(points) - 1
    hits = []
    tolerance = 64 * np.finfo(float).eps
    for index in range(count):
        start = points[index]
        end = points[(index + 1) % len(points)]
        segment = end - start
        denominator = _cross_2d(direction, segment)
        if abs(denominator) <= tolerance:
            continue
        offset = start - origin
        distance = _cross_2d(offset, segment) / denominator
        fraction = _cross_2d(offset, direction) / denominator
        if distance > tolerance and -tolerance <= fraction <= 1 + tolerance:
            hits.append((distance, np.clip(fraction, 0.0, 1.0), index))
    return hits


def _wavelength_at_intersection(hit, wavelength):
    _, fraction, index = hit
    return wavelength[index] + fraction * (wavelength[index + 1] - wavelength[index])


def _dominant_wavelengths(xy, white_xy, cmf):
    """Dominant/complementary wavelength and excitation purity in CIE xy."""
    xy = np.asarray(xy, dtype=float)
    white_xy = np.asarray(white_xy, dtype=float)
    direction = xy - white_xy
    if not np.all(np.isfinite(direction)):
        return np.nan, np.nan, np.nan
    if np.linalg.norm(direction) <= 64 * np.finfo(float).eps:
        return np.nan, np.nan, 0.0

    total = cmf.values.sum(axis=1)
    valid = np.isfinite(total) & (total > 0)
    locus = cmf.values[valid, :2] / total[valid, np.newaxis]
    wavelength = cmf.wavelength[valid]

    boundary_hits = _ray_segment_intersections(
        white_xy, direction, locus, close=True
    )
    if not boundary_hits:
        return np.nan, np.nan, np.nan
    boundary = min(boundary_hits, key=lambda hit: hit[0])
    purity = 1.0 / boundary[0]
    purple_index = len(locus) - 1

    if boundary[2] != purple_index:
        return _wavelength_at_intersection(boundary, wavelength), np.nan, purity

    opposite_hits = _ray_segment_intersections(white_xy, -direction, locus)
    if not opposite_hits:
        return np.nan, np.nan, purity
    complementary = _wavelength_at_intersection(
        min(opposite_hits, key=lambda hit: hit[0]), wavelength
    )
    return np.nan, complementary, purity


@dataclass(frozen=True)
class ColorAnalysis(AnalysisResult, Colorimetry):
    """Compute colorimetric quantities for one spectrum and CIE observer.

    ``mode="relative"`` computes unscaled tristimulus values and does not
    assign them a physical unit. ``mode="absolute"`` requires a radiometric
    spectral density and returns photometric quantities. The default ``auto``
    mode selects absolute behavior only when the value units support it.
    """
    spectrum: Spectrum | SpectrumView
    observer: str = '2deg'
    mode: str = "auto"
    white: str | tuple[float, float, float] = "D65"

    def __post_init__(self):
        if self.mode not in {"auto", "relative", "absolute"}:
            raise ValueError(
                "color mode must be 'auto', 'relative', or 'absolute'"
            )

        if not isinstance(self.white, str):
            white = tuple(np.asarray(self.white, dtype=float))
            object.__setattr__(self, "white", white)
            validate_white(white)
        elif self.white not in ILLUMINANTS:
            raise ValueError(
                f"Unknown reference white {self.white!r}; expected a bundled "
                "illuminant name or an XYZ triple."
            )

        if self.spectrum.wavelength_unit is None:
            raise ValueError(
                "Color analysis requires a physical wavelength unit."
            )

        if self.mode == "absolute" and self._photometric_basis is None:
            raise ValueError(
                "Absolute colorimetry requires spectral radiometric values "
                "compatible with W/nm, W/sr/nm, W/m^2/nm, or "
                "W/m^2/sr/nm."
            )

    @property
    def units(self) -> dict[str, object]:
        """Return physical units for supported color quantities."""
        return {
            "XYZ": self.XYZ_unit,
            "X": self.XYZ_unit,
            "Y": self.XYZ_unit,
            "Z": self.XYZ_unit,
            "xy": units.ureg.dimensionless,
            "x": units.ureg.dimensionless,
            "y": units.ureg.dimensionless,
            "uv": units.ureg.dimensionless,
            "u": units.ureg.dimensionless,
            "v": units.ureg.dimensionless,
            "uv_prime": units.ureg.dimensionless,
            "u_prime": units.ureg.dimensionless,
            "v_prime": units.ureg.dimensionless,
            "cct": units.ureg.kelvin,
            "duv": units.ureg.dimensionless,
            "planck.cct": units.ureg.kelvin,
            "planck.duv": units.ureg.dimensionless,
            "dominant_wavelength": units.ureg.nm,
            "complementary_wavelength": units.ureg.nm,
            "excitation_purity": units.ureg.dimensionless,
            "Lab": units.ureg.dimensionless,
            "Luv": units.ureg.dimensionless,
            "L_star": units.ureg.dimensionless,
            "a_star": units.ureg.dimensionless,
            "b_star": units.ureg.dimensionless,
            "u_star": units.ureg.dimensionless,
            "v_star": units.ureg.dimensionless,
        }

    @cached_property
    def effective_mode(self) -> str:
        """Return the explicit mode selected directly or by ``auto``."""
        if self.mode == "auto":
            return "absolute" if self._photometric_basis is not None else "relative"
        return self.mode

    @cached_property
    def XYZ_unit(self):
        """Return the absolute photometric unit, or ``None`` when relative."""
        if self.effective_mode == "relative":
            return None
        return self._photometric_basis[1]

    @property
    def summary(self):
        """Return scalar color results keyed by observer and quantity."""
        return {
            self.observer: {
                "X" : self.X,
                "Y" : self.Y,
                "Z" : self.Z,
                "x" : self.x,
                "y" : self.y,
                "u" : self.u,
                "v" : self.v,
                "u_prime" : self.u_prime,
                "v_prime" : self.v_prime,
                "planck.cct" : self.planck.cct,
                "planck.duv" : self.planck.duv,
                "dominant_wavelength" : self.dominant_wavelength,
                "complementary_wavelength" : self.complementary_wavelength,
                "excitation_purity" : self.excitation_purity,
                "L_star" : self.L_star,
                "a_star" : self.a_star,
                "b_star" : self.b_star,
                "u_star" : self.u_star,
                "v_star" : self.v_star,
            },
        }

    @cached_property
    def cmf(self):
        """Return the selected observer's color-matching functions."""
        return CMFS[self.observer]
    
    @cached_property
    def XYZ(self):
        """Integrate the spectrum against matching functions to obtain XYZ."""
        if self._is_empty:
            return np.array([np.nan, np.nan, np.nan])

        wavelength_nm = units.convert_values(
            self.spectrum.wavelength,
            self.spectrum.wavelength_unit,
            units.ureg.nm,
        )
        values = self.spectrum.values

        scale = 1.0
        if self.effective_mode == "absolute":
            radiometric_unit, _ = self._photometric_basis
            values = units.convert_values(
                values,
                self.spectrum.value_unit,
                radiometric_unit / units.ureg.nm,
            )
            scale = K_M

        # Unit conversion can place an otherwise exact endpoint a few ulps
        # beyond the CMF grid (for example, 0.4 um -> 399.999... nm). Treat
        # only that roundoff margin as in-range; genuinely missing regions
        # remain zero rather than inheriting an endpoint value.
        query = self.cmf.wavelength
        scale_nm = max(1.0, np.max(np.abs(wavelength_nm)))
        tolerance = 16 * np.finfo(float).eps * scale_nm
        in_range = (
            (query >= wavelength_nm[0] - tolerance)
            & (query <= wavelength_nm[-1] + tolerance)
        )
        sampled = np.zeros_like(query, dtype=float)
        sampled[in_range] = np.interp(
            np.clip(query[in_range], wavelength_nm[0], wavelength_nm[-1]),
            wavelength_nm,
            values,
        )
        tristimulus = np.trapezoid(
            sampled[:, np.newaxis] * self.cmf.values,
            self.cmf.wavelength,
            axis=0,
        )
        return scale * tristimulus

    @cached_property
    def _photometric_basis(self):
        """Return canonical radiometric-density and photometric units."""
        value_unit = self.spectrum.value_unit
        if value_unit is None:
            return None

        integrated_unit = units.multiply_units(
            value_unit,
            self.spectrum.wavelength_unit,
        )
        has_solid_angle = "steradian" in str(integrated_unit)

        power = units.ureg.W
        area = units.ureg.m**2
        solid_angle = units.ureg.sr

        if units.Q_(1, integrated_unit).is_compatible_with(power / area):
            if has_solid_angle:
                return power / area / solid_angle, units.ureg.cd / area
            return power / area, units.ureg.lux

        if units.Q_(1, integrated_unit).is_compatible_with(power):
            if has_solid_angle:
                return power / solid_angle, units.ureg.cd
            return power, units.ureg.lm

        return None

    @cached_property
    def white_XYZ(self):
        """Return the selected reference white normalized to ``Y_n = 100``."""
        return reference_white_XYZ(self.white, self.observer)

    @property
    def X(self):
        """Return the X tristimulus component."""
        return self.XYZ[0]

    @property
    def Y(self):
        """Return the Y tristimulus component."""
        return self.XYZ[1]

    @property
    def Z(self):
        """Return the Z tristimulus component."""
        return self.XYZ[2]

    @cached_property
    def xy(self):
        """Return CIE 1931 ``(x, y)`` chromaticity coordinates."""
        return self._xy
    @property
    def x(self):
        """Return the CIE 1931 x coordinate."""
        return self.xy[0]

    @property
    def y(self):
        """Return the CIE 1931 y coordinate."""
        return self.xy[1]

    @cached_property
    def xyY(self):
        """Return chromaticity coordinates with the Y tristimulus value."""
        return np.array([self.x, self.y, self.Y])
    
    @cached_property
    def uv(self):
        """Return CIE 1960 UCS ``(u, v)`` coordinates."""
        return self._uv
    @property
    def u(self):
        """Return the CIE 1960 u coordinate."""
        return self.uv[0]

    @property
    def v(self):
        """Return the CIE 1960 v coordinate."""
        return self.uv[1]
    
    @cached_property
    def uv_prime(self):
        """Return CIE 1976 UCS ``(u', v')`` coordinates."""
        return self._uv_prime
    @property
    def u_prime(self):
        """Return the CIE 1976 u-prime coordinate."""
        return self.uv_prime[0]

    @property
    def v_prime(self):
        """Return the CIE 1976 v-prime coordinate."""
        return self.uv_prime[1]

    @cached_property
    def Lab(self):
        """Return CIE 1976 ``(L*, a*, b*)`` for the reference white."""
        return self._Lab
    @property
    def L_star(self):
        """Return CIE 1976 perceptual lightness."""
        return self.Lab[0]

    @property
    def a_star(self):
        """Return the CIELAB green-red opponent coordinate."""
        return self.Lab[1]

    @property
    def b_star(self):
        """Return the CIELAB blue-yellow opponent coordinate."""
        return self.Lab[2]

    @cached_property
    def Luv(self):
        """Return CIE 1976 ``(L*, u*, v*)`` for the reference white."""
        return self._Luv
    @property
    def u_star(self):
        """Return the CIELUV u-star opponent coordinate."""
        return self.Luv[1]

    @property
    def v_star(self):
        """Return the CIELUV v-star opponent coordinate."""
        return self.Luv[2]

    @property
    def _XYZ(self):
        return self.XYZ
    @cached_property
    def planck(self):
        """Estimate correlated color temperature and Duv."""
        cct, duv = self._cct_duv
        return ColorTemperatureResult(cct=float(cct), duv=float(duv))
    @property
    def cct(self):
        """Return the Planckian correlated color temperature in kelvin."""
        return self.planck.cct

    @property
    def duv(self):
        """Return signed distance from the Planckian locus."""
        return self.planck.duv

    @cached_property
    def _dominant_wavelength_result(self):
        """Dominant/complementary wavelength and excitation purity."""
        white_xy = xyz_to_xy(self.white_XYZ)
        return _dominant_wavelengths(self.xy, white_xy, self.cmf)

    @property
    def dominant_wavelength(self):
        """CIE dominant wavelength in nm, or NaN for purple/achromatic light."""
        return self._dominant_wavelength_result[0]

    @property
    def complementary_wavelength(self):
        """CIE complementary wavelength in nm for purple light, otherwise NaN."""
        return self._dominant_wavelength_result[1]

    @property
    def excitation_purity(self):
        """Chromaticity distance from white relative to the locus boundary."""
        return self._dominant_wavelength_result[2]

    def delta_e_2000(self, other, **kwargs):
        """Return CIEDE2000 difference from another color or LAB triple.

        Another colour (a spectrum's, a region's, an image's) must use the
        same reference white, and another spectrum's analysis the same
        observer. A raw ``(L*, a*, b*)`` triple is interpreted in this
        analysis's LAB coordinate system.
        """
        if isinstance(other, ColorAnalysis) and self.observer != other.observer:
            raise ValueError("CIEDE2000 requires matching colorimetric observers.")
        return self._delta_e_2000(other, **kwargs)
