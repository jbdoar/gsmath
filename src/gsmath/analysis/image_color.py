"""Colour of every pixel of a tristimulus (X, Y, Z) image, or of a region.

:class:`ImageColor` turns an image with channels ``("X", "Y", "Z")`` into
maps -- each an :class:`~gsmath.Image` on the same pixels, with the same
pixel map -- of chromaticity (x, y; CIE 1960 u, v; CIE 1976 u', v'),
CIELAB and CIELUV, correlated colour temperature and Duv, and CIEDE2000
difference from a reference. :meth:`ImageColor.mean` gives the colour of a
region's mean XYZ as numbers (a :class:`TristimulusColor`)::

    color = xyz_image.color(white=display_white_XYZ)
    color.uv_prime                 # an Image, channels u', v'
    color.cct.stats(roi=spot).mean
    color.mean(roi=spot).xy        # the spot's mean colour
    color.delta_e_2000(reference)  # a map of CIEDE2000

CIELAB, CIELUV and CIEDE2000 depend on the reference white being on the
same scale as the image's XYZ: for absolute XYZ (cd/m²), pass the white's
measured XYZ in those units; a bundled illuminant's name gives a white with
``Y = 100``, for XYZ already scaled to that.

A camera's R, G, B become X, Y, Z by a colour matrix
(:meth:`Image.apply_matrix <gsmath.Image.apply_matrix>` or the
:class:`~gsmath.ColorMatrix` calibration step), which
:func:`gsmath.fit_color_matrix` fits (see :mod:`gsmath.colorimeter`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .. import units
from .base import Result
from .colorimetry import Colorimetry, reference_white_XYZ, xyz_to_uv, xyz_to_xy  # noqa: F401

XYZ_CHANNELS = ("X", "Y", "Z")


def _color_units(XYZ_unit):
    one = units.ureg.dimensionless
    return {"X": XYZ_unit, "Y": XYZ_unit, "Z": XYZ_unit, "x": one, "y": one, "u_prime": one,
            "v_prime": one, "cct": units.ureg.kelvin, "duv": one, "L_star": one, "a_star": one,
            "b_star": one}


@dataclass(frozen=True, eq=False)
class TristimulusColor(Colorimetry, Result):
    """The colour of one XYZ triple: a region's mean, say."""
    XYZ: np.ndarray
    white_XYZ: np.ndarray
    XYZ_unit: object = None

    def __post_init__(self):
        object.__setattr__(self, "XYZ", np.asarray(self.XYZ, dtype=float))
        object.__setattr__(self, "white_XYZ", np.asarray(self.white_XYZ, dtype=float))

    @property
    def _XYZ(self):
        return self.XYZ

    X = property(lambda self: float(self.XYZ[0]))
    Y = property(lambda self: float(self.XYZ[1]))
    Z = property(lambda self: float(self.XYZ[2]))
    xy = property(lambda self: self._xy)
    x = property(lambda self: float(self._xy[0]))
    y = property(lambda self: float(self._xy[1]))
    uv = property(lambda self: self._uv)
    uv_prime = property(lambda self: self._uv_prime)
    u_prime = property(lambda self: float(self._uv_prime[0]))
    v_prime = property(lambda self: float(self._uv_prime[1]))
    Lab = property(lambda self: self._Lab)
    Luv = property(lambda self: self._Luv)
    cct = property(lambda self: float(self._cct_duv[0]))
    duv = property(lambda self: float(self._cct_duv[1]))

    @property
    def units(self):
        return _color_units(self.XYZ_unit)

    @property
    def summary(self):
        return {"X": self.X, "Y": self.Y, "Z": self.Z, "x": self.x, "y": self.y,
                "u_prime": self.u_prime, "v_prime": self.v_prime, "cct": self.cct,
                "duv": self.duv, "L_star": float(self.Lab[0]), "a_star": float(self.Lab[1]),
                "b_star": float(self.Lab[2])}

    def delta_e_2000(self, reference, **kwargs):
        """CIEDE2000 from another colour (same white) or a Lab triple."""
        return float(self._delta_e_2000(reference, **kwargs))


@dataclass(frozen=True, eq=False)
class ImageColor(Colorimetry, Result):
    """Per-pixel colour of an X, Y, Z image (see the module notes)."""
    image: object
    white: object = "D65"
    observer: str = "2deg"

    def __post_init__(self):
        missing = [c for c in XYZ_CHANNELS if c not in (self.image.channels or ())]
        if missing:
            raise ValueError(f"{self.image.label!r} isn't a tristimulus image: it has no "
                             f"channel {missing[0]!r} (channels {self.image.channels}); "
                             "convert camera R, G, B with apply_matrix first")
        if not isinstance(self.white, str):
            object.__setattr__(self, "white", tuple(np.asarray(self.white, dtype=float)))

    @cached_property
    def white_XYZ(self):
        return reference_white_XYZ(self.white, self.observer)

    @cached_property
    def _XYZ(self):
        return np.asarray(self.image[list(XYZ_CHANNELS)].values, dtype=float)

    def _map(self, values, name, channels=None, unit=None):
        return self.image.copy(values=values, channels=channels, value_unit=unit,
                               label=f"{self.image.label} {name}")

    @property
    def XYZ(self):
        return self.image[list(XYZ_CHANNELS)]

    @property
    def Y(self):
        """Luminance (in the image's unit)."""
        return self.image["Y"]

    @property
    def xy(self):
        return self._map(self._xy, "xy", ("x", "y"))

    @property
    def uv(self):
        return self._map(self._uv, "uv", ("u", "v"))

    @property
    def uv_prime(self):
        return self._map(self._uv_prime, "u'v'", ("u'", "v'"))

    @property
    def Lab(self):
        return self._map(self._Lab, "Lab", ("L*", "a*", "b*"))

    @property
    def Luv(self):
        return self._map(self._Luv, "Luv", ("L*", "u*", "v*"))

    @property
    def cct(self):
        """Correlated colour temperature (K), NaN where the colour is undefined."""
        return self._map(self._cct_duv[0], "CCT", unit="K")

    @property
    def duv(self):
        return self._map(self._cct_duv[1], "Duv")

    def delta_e_2000(self, reference, **kwargs):
        """A CIEDE2000 map against a reference: another ImageColor of the same
        shape (pixel by pixel), one colour (a TristimulusColor or a
        Spectrum's ColorAnalysis), or a Lab triple -- the same white."""
        if isinstance(reference, ImageColor) and reference.image.shape != self.image.shape:
            raise ValueError("CIEDE2000 between images needs the same shape")
        return self._map(self._delta_e_2000(reference, **kwargs), "ΔE00")

    def mean(self, roi=None) -> TristimulusColor:
        """The colour of the mean XYZ over the image or a region."""
        XYZ = self.XYZ.stats(roi=roi).mean
        return TristimulusColor(XYZ, self.white_XYZ, self.image.value_unit)

    @property
    def label(self):
        return self.image.label

    @property
    def units(self):
        return _color_units(self.image.value_unit)

    @property
    def summary(self):
        """The mean colour's summary."""
        return self.mean().summary
