"""Where each pixel of an image looks or sits: pixel-to-angle or -position maps.

A :class:`PixelMap` gives every pixel centre an ``(x, y)`` coordinate in one
unit -- degrees of field for a camera looking at a display, millimetres on
the object for a microscope. It is to an :class:`~gsmath.Image` what the
wavelength axis is to a Spectrum. Two layouts:

- *separable*: ``x`` is one value per column and ``y`` one per row (a linear
  pitch, or any monotonic axis);
- *full*: ``x`` and ``y`` are arrays of the image's shape, for a mapping that
  isn't separable (lens distortion, a tilted target).

Row 0 is the top of the image; a mapping whose ``y`` increases upward has a
negative row pitch::

    angles = PixelMap.linear((1200, 1600), step=(0.05, -0.05), unit="deg")
    lens = PixelMap.from_function((1200, 1600), f_theta, unit="deg")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import units
from ._utils import factor_pair


@dataclass(frozen=True, eq=False)
class PixelMap:
    x: np.ndarray
    """Horizontal coordinate: one per column, or one per pixel."""
    y: np.ndarray
    """Vertical coordinate: one per row, or one per pixel."""
    unit: object = None

    def __post_init__(self):
        x, y = np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float)
        if x.ndim != y.ndim or x.ndim not in (1, 2):
            raise ValueError("x and y must both be 1-D (separable) or both 2-D (per pixel)")
        if x.ndim == 2 and x.shape != y.shape:
            raise ValueError("per-pixel x and y must have the same shape")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "unit", units.parse(self.unit))

    @classmethod
    def linear(cls, shape, step, *, center=None, unit=None):
        """A uniform pitch: ``step`` per pixel (one number, or ``(x, y)``),
        zero at ``center`` -- a ``(column, row)`` pixel position, fractional
        allowed, by default the middle of the image."""
        rows, cols = shape
        sx, sy = np.broadcast_to(np.asarray(step, dtype=float), (2,))
        cx, cy = ((cols - 1) / 2, (rows - 1) / 2) if center is None else center
        return cls((np.arange(cols) - cx) * sx, (np.arange(rows) - cy) * sy, unit)

    @classmethod
    def from_function(cls, shape, function, *, unit=None):
        """A full map from ``function(columns, rows) -> (x, y)``, called once
        with arrays of every pixel's column and row index."""
        rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
        x, y = function(cols.astype(float), rows.astype(float))
        return cls(np.broadcast_to(x, shape), np.broadcast_to(y, shape), unit)

    @property
    def separable(self) -> bool:
        return self.x.ndim == 1

    @property
    def shape(self) -> tuple[int, int]:
        """The (rows, columns) of image this maps."""
        return (len(self.y), len(self.x)) if self.separable else self.x.shape

    def grid(self):
        """``(X, Y)``: each pixel's coordinates, as arrays of the image's shape."""
        if self.separable:
            return np.meshgrid(self.x, self.y)
        return self.x, self.y

    def sliced(self, rows: slice, cols: slice) -> PixelMap:
        """The map of ``image[rows, cols]``."""
        if self.separable:
            return PixelMap(self.x[cols], self.y[rows], self.unit)
        return PixelMap(self.x[rows, cols], self.y[rows, cols], self.unit)

    def binned(self, factor) -> PixelMap:
        """The map of the image binned by `factor` (one number, or ``(rows,
        columns)``): each block's mean coordinates, a partial block at the far
        edges dropped, as in binning the image."""
        fy, fx = factor_pair(factor)

        def block_mean(a, factors):
            trimmed = a[tuple(slice(0, n // f * f) for n, f in zip(a.shape, factors))]
            shape = [m for n, f in zip(trimmed.shape, factors) for m in (n // f, f)]
            return trimmed.reshape(shape).mean(axis=tuple(range(1, 2 * trimmed.ndim, 2)))

        if self.separable:
            return PixelMap(block_mean(self.x, (fx,)), block_mean(self.y, (fy,)), self.unit)
        return PixelMap(block_mean(self.x, (fy, fx)), block_mean(self.y, (fy, fx)), self.unit)

    def rotated90(self, k: int = 1) -> PixelMap:
        """The map of the image rotated by ``numpy.rot90(values, k)``: each
        pixel keeps its coordinates, so a separable map becomes a per-pixel
        one unless the turn is a half turn."""
        k %= 4
        if self.separable and k == 2:
            return PixelMap(self.x[::-1], self.y[::-1], self.unit)
        x, y = self.grid()
        return PixelMap(np.rot90(x, k), np.rot90(y, k), self.unit)

    def at(self, cols, rows):
        """``(x, y)`` at fractional pixel positions, interpolated linearly
        (extrapolated linearly beyond the edges for a separable map)."""
        cols, rows = np.asarray(cols, dtype=float), np.asarray(rows, dtype=float)
        if self.separable:
            return _linear(self.x, cols), _linear(self.y, rows)
        import scipy.ndimage

        positions = np.stack([rows.ravel(), cols.ravel()])
        sample = lambda a: scipy.ndimage.map_coordinates(a, positions, order=1, mode="nearest")
        return sample(self.x).reshape(cols.shape), sample(self.y).reshape(rows.shape)

    def pixel_of(self, x, y):
        """``(column, row)``: the fractional pixel position of coordinates
        ``(x, y)`` -- for a separable map with monotonic axes."""
        if not self.separable:
            raise ValueError("finding pixels from coordinates needs a separable pixel map")
        return (_inverse(self.x, np.asarray(x, dtype=float), "x"),
                _inverse(self.y, np.asarray(y, dtype=float), "y"))

    def to_unit(self, unit) -> PixelMap:
        unit = units.parse(unit)
        return PixelMap(units.convert_values(self.x, self.unit, unit),
                        units.convert_values(self.y, self.unit, unit), unit)

    def magnitude(self, value) -> float:
        """A coordinate as a number in this map's unit (quantities converted)."""
        if units.is_quantity(value):
            return units.convert_quantity(value, self.unit)
        return float(value)

    def __eq__(self, other):
        if not isinstance(other, PixelMap):
            return NotImplemented
        return (self.unit == other.unit and self.separable == other.separable
                and np.array_equal(self.x, other.x) and np.array_equal(self.y, other.y))

    __hash__ = object.__hash__

    def __repr__(self):
        layout = "separable" if self.separable else "per-pixel"
        unit = units.format_unit(self.unit)
        return f"PixelMap({layout}, shape={self.shape}, unit={unit!r})"


def _linear(axis, positions):
    """`axis` (one value per pixel) at fractional `positions`, linear inside
    and extrapolated from the end pixels outside."""
    if axis.size == 1:
        return np.full(positions.shape, axis[0])
    index = np.arange(axis.size)
    inside = np.interp(positions, index, axis)
    below = axis[0] + (positions - 0) * (axis[1] - axis[0])
    above = axis[-1] + (positions - (axis.size - 1)) * (axis[-1] - axis[-2])
    return np.where(positions < 0, below, np.where(positions > axis.size - 1, above, inside))


def _inverse(axis, values, name):
    """Fractional pixel positions of `values` on a monotonic `axis`."""
    steps = np.diff(axis)
    if axis.size > 1 and not (np.all(steps > 0) or np.all(steps < 0)):
        raise ValueError(f"the pixel map's {name} axis isn't monotonic")
    if axis.size > 1 and steps[0] < 0:
        return (axis.size - 1) - _inverse(axis[::-1], values, name)
    if axis.size == 1:
        return np.zeros(values.shape)
    index = np.arange(axis.size, dtype=float)
    inside = np.interp(values, axis, index)
    below = (values - axis[0]) / (axis[1] - axis[0])
    above = (axis.size - 1) + (values - axis[-1]) / (axis[-1] - axis[-2])
    return np.where(values < axis[0], below, np.where(values > axis[-1], above, inside))
