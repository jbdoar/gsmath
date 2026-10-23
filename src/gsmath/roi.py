"""Regions of interest: which pixels of an image an analysis looks at.

A region gives a boolean mask for an image (:meth:`ROI.mask`). A pixel is in
the region when its *centre* is, measured either in pixel indices (column
``x``, row ``y``; the default) or, with ``coords=True``, in the image's
:class:`~gsmath.PixelMap` (degrees, millimetres, ...; quantities are
converted to the map's unit). Regions combine::

    spot = Circle(0, 0, 2.5, coords=True)             # 2.5 deg about the axis
    ring = Circle(0, 0, 10, coords=True) - spot        # an annulus
    image.stats(roi=ring)

Rectangles are half-open like slicing: ``Rectangle(0, 0, 10, 20)`` in pixels
is ``image[0:20, 0:10]``.

A region only looks at the pixels in its bounding window
(:meth:`ROI.window`), so a small spot on a large frame costs what the spot
does, not what the frame does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FULL = (slice(None), slice(None))


class ROI:
    """A region of an image."""

    coords: bool = False

    def mask(self, image) -> np.ndarray:
        """A boolean (rows, columns) array: True for the pixels in the region."""
        window, local = self.submask(image)
        found = np.zeros(image.shape, dtype=bool)
        found[window] = local
        return found

    def window(self, image) -> tuple:
        """``(row slice, column slice)``: a box of the image holding every
        pixel of the region (the whole image if it can't be bounded)."""
        return FULL

    def submask(self, image, window=None):
        """``(window, mask)``: the region's pixels within ``window`` (by
        default its own :meth:`window`) as a boolean array of the window's
        shape -- so ``image.values[window][mask]`` are its pixels."""
        window = self.window(image) if window is None else window
        return window, self._local(image, window)

    def _local(self, image, window) -> np.ndarray:
        x, y = _centres(image, self.coords, window)
        return self._contains(x, y, image)

    def _contains(self, x, y, image) -> np.ndarray:
        raise NotImplementedError

    def _box(self, image, xs, ys):
        """The window around points ``xs``, ``ys`` (in this region's
        coordinates), a pixel of margin each side; FULL if a pixel map
        can't be inverted."""
        try:
            if self.coords:
                cols, rows = _pixel_map(image).pixel_of(np.asarray(xs, dtype=float),
                                                        np.asarray(ys, dtype=float))
            else:
                cols, rows = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        except ValueError:  # a per-pixel map: no inverse lookup
            return FULL
        if not (np.all(np.isfinite(cols)) and np.all(np.isfinite(rows))):
            return FULL
        height, width = image.shape
        r0, r1 = int(np.floor(np.min(rows))) - 1, int(np.ceil(np.max(rows))) + 2
        c0, c1 = int(np.floor(np.min(cols))) - 1, int(np.ceil(np.max(cols))) + 2
        return (slice(min(max(r0, 0), height), min(max(r1, 0), height)),
                slice(min(max(c0, 0), width), min(max(c1, 0), width)))

    def __and__(self, other):
        return _Combined(np.logical_and, self, other)

    def __or__(self, other):
        return _Combined(np.logical_or, self, other)

    def __sub__(self, other):
        return _Combined(lambda a, b: a & ~b, self, other)

    def __invert__(self):
        return _Not(self)


@dataclass(frozen=True)
class Rectangle(ROI):
    x0: float
    y0: float
    x1: float
    y1: float
    coords: bool = False

    def _contains(self, x, y, image):
        (x0, x1), (y0, y1) = _bounds(image, self.coords, self.x0, self.x1, self.y0, self.y1)
        return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)

    def window(self, image):
        (x0, x1), (y0, y1) = _bounds(image, self.coords, self.x0, self.x1, self.y0, self.y1)
        return self._box(image, [x0, x1, x0, x1], [y0, y0, y1, y1])


@dataclass(frozen=True)
class Circle(ROI):
    cx: float
    cy: float
    radius: float
    coords: bool = False

    def _contains(self, x, y, image):
        cx, cy, r = (_magnitude(image, self.coords, v) for v in (self.cx, self.cy, self.radius))
        return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

    def window(self, image):
        cx, cy, r = (_magnitude(image, self.coords, v) for v in (self.cx, self.cy, self.radius))
        return self._box(image, [cx - r, cx + r, cx - r, cx + r], [cy - r, cy - r, cy + r, cy + r])


@dataclass(frozen=True)
class Polygon(ROI):
    vertices: tuple
    """``((x, y), ...)`` in order around the polygon."""
    coords: bool = False

    def __post_init__(self):
        object.__setattr__(self, "vertices", tuple(tuple(v) for v in self.vertices))

    def _contains(self, x, y, image):
        from matplotlib.path import Path

        vertices = [[_magnitude(image, self.coords, a) for a in v] for v in self.vertices]
        points = np.column_stack([np.ravel(x), np.ravel(y)])
        return Path(vertices).contains_points(points).reshape(np.shape(x))

    def window(self, image):
        vertices = np.array([[_magnitude(image, self.coords, a) for a in v]
                             for v in self.vertices])
        return self._box(image, vertices[:, 0], vertices[:, 1])


@dataclass(frozen=True, eq=False)
class Mask(ROI):
    """An explicit boolean array of the image's (rows, columns)."""
    array: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "array", np.asarray(self.array, dtype=bool))

    def mask(self, image):
        self._check(image)
        return self.array.copy()

    def _local(self, image, window):
        self._check(image)
        return self.array[window]

    def _check(self, image):
        if self.array.shape != image.shape:
            raise ValueError(f"mask shape {self.array.shape} doesn't match the image {image.shape}")


@dataclass(frozen=True)
class _Combined(ROI):
    op: object
    a: ROI
    b: ROI

    def window(self, image):
        a, b = self.a.window(image), self.b.window(image)
        if self.op is np.logical_or:
            return _union(a, b, image)
        if self.op is np.logical_and:
            return _intersection(a, b, image)
        return a  # a - b lies within a

    def _local(self, image, window):
        return self.op(self.a._local(image, window), self.b._local(image, window))


@dataclass(frozen=True)
class _Not(ROI):
    a: ROI

    def _local(self, image, window):
        return ~self.a._local(image, window)


def _centres(image, coords, window=FULL):
    rows, cols = window
    if coords:
        return _pixel_map(image).sliced(rows, cols).grid()
    height, width = image.shape
    return np.meshgrid(np.arange(width, dtype=float)[cols], np.arange(height, dtype=float)[rows])


def _span(window, image):
    """Each slice of a window as (start, stop) in the image."""
    return [s.indices(n)[:2] for s, n in zip(window, image.shape)]


def _union(a, b, image):
    return tuple(slice(min(p[0], q[0]), max(p[1], q[1]))
                 for p, q in zip(_span(a, image), _span(b, image)))


def _intersection(a, b, image):
    spans = [(max(p[0], q[0]), min(p[1], q[1])) for p, q in zip(_span(a, image), _span(b, image))]
    return tuple(slice(start, max(start, stop)) for start, stop in spans)


def _pixel_map(image):
    if image.pixel_map is None:
        raise ValueError(f"{image.label!r} has no pixel map; use pixel coordinates (coords=False)")
    return image.pixel_map


def _magnitude(image, coords, value):
    return _pixel_map(image).magnitude(value) if coords else float(value)


def _bounds(image, coords, x0, x1, y0, y1):
    """Each pair in increasing order (a map's y may run upward)."""
    x0, x1, y0, y1 = (_magnitude(image, coords, v) for v in (x0, x1, y0, y1))
    return sorted((x0, x1)), sorted((y0, y1))
