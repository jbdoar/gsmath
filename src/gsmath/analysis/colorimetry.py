"""Colorimetry from CIE X, Y, Z: one set of formulas for every shape.

Every function takes X, Y, Z along the last axis -- one triple, a list of
samples, an image's pixels -- and returns results of the matching shape.
:class:`Colorimetry` derives all the usual quantities from an object's
``_XYZ`` and ``white_XYZ``; :class:`~gsmath.ColorAnalysis` (a spectrum),
:class:`~gsmath.TristimulusColor` (one colour) and
:class:`~gsmath.ImageColor` (a map) are all Colorimetry, so they agree
exactly.

A value with a non-finite component, or whose chromaticity is undefined
(X + Y + Z = 0), gives NaN.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from ..reference.cmfs import CMFS
from ..reference.illuminants import ILLUMINANTS
from .delta_e import delta_e_2000

_CIE_DELTA = 6 / 29


def _cie_f(values):
    """The piecewise CIE 1976 lightness transfer function."""
    values = np.asarray(values, dtype=float)
    return np.where(values > _CIE_DELTA ** 3, np.cbrt(values),
                    values / (3 * _CIE_DELTA ** 2) + 4 / 29)


def _xyz(XYZ):
    XYZ = np.asarray(XYZ, dtype=float)
    if XYZ.ndim == 0 or XYZ.shape[-1] != 3:
        raise ValueError("XYZ and white_XYZ must each contain three values.")
    return XYZ


def validate_white(white_XYZ):
    """A reference white as a finite, positive XYZ triple."""
    white_XYZ = np.asarray(white_XYZ, dtype=float)
    if white_XYZ.shape != (3,):
        raise ValueError("XYZ and white_XYZ must each contain three values.")
    if not np.all(np.isfinite(white_XYZ)) or np.any(white_XYZ <= 0):
        raise ValueError("white_XYZ values must be finite and positive.")
    return white_XYZ


def _nan_where_nonfinite(result, XYZ):
    """NaN for every triple whose XYZ had a non-finite component."""
    bad = ~np.all(np.isfinite(XYZ), axis=-1)
    return np.where(bad[..., np.newaxis], np.nan, result)


def xyz_to_xy(XYZ):
    """CIE 1931 chromaticity (x, y)."""
    XYZ = _xyz(XYZ)
    total = XYZ.sum(axis=-1)
    valid = np.isfinite(total) & (total != 0)
    xy = XYZ[..., :2] / np.where(valid, total, 1.0)[..., np.newaxis]
    return np.where(valid[..., np.newaxis], xy, np.nan)


def xyz_to_uv_prime(XYZ):
    """CIE 1976 UCS chromaticity (u', v')."""
    XYZ = _xyz(XYZ)
    X, Y, Z = np.moveaxis(XYZ, -1, 0)
    denominator = X + 15 * Y + 3 * Z
    valid = np.isfinite(denominator) & (denominator != 0)
    safe = np.where(valid, denominator, 1.0)
    uv = np.stack([4 * X / safe, 9 * Y / safe], axis=-1)
    return np.where(valid[..., np.newaxis], uv, np.nan)


def xyz_to_uv(XYZ):
    """CIE 1960 UCS chromaticity (u, v): u = u', v = 2/3 v'."""
    return xyz_to_uv_prime(XYZ) * np.array([1.0, 2.0 / 3.0])


def xyz_to_lab(XYZ, white_XYZ):
    """CIELAB (L*, a*, b*) for a reference white."""
    XYZ, white_XYZ = _xyz(XYZ), validate_white(white_XYZ)
    fx, fy, fz = np.moveaxis(_cie_f(XYZ / white_XYZ), -1, 0)
    Lab = np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)
    return _nan_where_nonfinite(Lab, XYZ)


def xyz_to_luv(XYZ, white_XYZ):
    """CIELUV (L*, u*, v*) for a reference white; black has u* = v* = 0."""
    XYZ, white_XYZ = _xyz(XYZ), validate_white(white_XYZ)
    L = 116 * _cie_f(XYZ[..., 1] / white_XYZ[1]) - 16
    u_prime, v_prime = np.moveaxis(xyz_to_uv_prime(XYZ), -1, 0)
    white_u_prime, white_v_prime = xyz_to_uv_prime(white_XYZ)
    with np.errstate(invalid="ignore"):
        Luv = np.stack([L, 13 * L * (u_prime - white_u_prime),
                        13 * L * (v_prime - white_v_prime)], axis=-1)
    Luv = np.where((L == 0)[..., np.newaxis], 0.0, Luv)
    return _nan_where_nonfinite(Luv, XYZ)


def xyz_to_cct(XYZ):
    """Correlated colour temperature (K) and Duv (Robertson's method):
    a pair of arrays of the samples' shape."""
    from .planckian import cct_robertson_array

    uv = xyz_to_uv(XYZ)
    return cct_robertson_array(uv[..., 0], uv[..., 1])


def reference_white_XYZ(white, observer="2deg"):
    """A reference white's XYZ: a bundled illuminant's name (normalised to
    ``Y_n = 100`` for `observer`), or an XYZ triple, validated."""
    if not isinstance(white, str):
        return validate_white(white)
    if white not in ILLUMINANTS:
        raise ValueError(f"Unknown reference white {white!r}; expected a bundled "
                         "illuminant name or an XYZ triple.")
    cmf = CMFS[observer]
    illuminant = ILLUMINANTS[white]
    values = np.interp(cmf.wavelength, illuminant.wavelength, illuminant.values,
                       left=0.0, right=0.0)
    XYZ = np.trapezoid(values[:, np.newaxis] * cmf.values, cmf.wavelength, axis=0)
    if not np.isfinite(XYZ[1]) or XYZ[1] <= 0:
        raise ValueError(f"Reference white {white!r} has invalid luminance.")
    return XYZ * (100 / XYZ[1])


class Colorimetry:
    """Quantities derived from ``self._XYZ`` (last axis X, Y, Z) and
    ``self.white_XYZ``, cached: arrays of the samples' shape."""

    @cached_property
    def _xy(self):
        return xyz_to_xy(self._XYZ)

    @cached_property
    def _uv(self):
        return xyz_to_uv(self._XYZ)

    @cached_property
    def _uv_prime(self):
        return xyz_to_uv_prime(self._XYZ)

    @cached_property
    def _Lab(self):
        return xyz_to_lab(self._XYZ, self.white_XYZ)

    @cached_property
    def _Luv(self):
        return xyz_to_luv(self._XYZ, self.white_XYZ)

    @cached_property
    def _cct_duv(self):
        return xyz_to_cct(self._XYZ)

    def _delta_e_2000(self, reference, **kwargs):
        """CIEDE2000 from a reference: another Colorimetry with the same
        white (pixel by pixel if it's a map), or a Lab triple."""
        if isinstance(reference, Colorimetry):
            if not np.allclose(reference.white_XYZ, self.white_XYZ):
                raise ValueError("CIEDE2000 requires matching reference whites.")
            reference = reference._Lab
        return delta_e_2000(self._Lab, np.asarray(reference, dtype=float), **kwargs)
