"""Values along a line through an image: a row, a column, or any segment.

A :class:`LineCut` is a :class:`~gsmath.Series` of the values, per channel,
against the distance along the line from the start, and keeps each sample's
``(x, y)`` coordinates -- in the image's pixel-map unit when it has one,
else in pixels. Like any series it plots, and finds a peak and its width::

    across = image.line((100, 240), (500, 240), width=5)   # 5-pixel band
    across.distance, across.values
    spot = image.line((-3, 0), (3, 0), coords=True)            # in degrees
    spot["Y"].peaks().fwhm                                      # in degrees
"""

from __future__ import annotations

import numpy as np

from . import units
from .sampled import read_only
from .series import Series


class LineCut(Series):
    """An image's values along a line: a :class:`~gsmath.Series` against
    ``distance`` from the first sample, which also keeps each sample's
    ``(x, y)`` position (in the image's pixel-map unit, else in pixels)."""
    AXIS = "distance"
    AXIS_LABEL = "Distance"
    _FIELDS = ("distance", "values", "x", "y", "label", "distance_unit", "value_unit",
               "channels", "metadata")
    _REMOVABLE = {"distance_unit", "value_unit", "channels"}
    _PER_SAMPLE = ("x", "y")

    def __init__(self, distance, values, x=None, y=None, label="Line cut", distance_unit=None,
                 value_unit=None, channels=None, metadata=None):
        self._set_series(distance, values, label, distance_unit, value_unit, channels, metadata)
        self.x = None if x is None else read_only(x)
        self.y = None if y is None else read_only(y)
        self._seal()


def line(image, start, end, *, coords=False, width=1, num=None, order=1, label=None):
    """Sample ``image`` along the straight line (in pixels) from ``start`` to
    ``end``, each an ``(x, y)`` in pixels -- column, row -- or, with
    ``coords``, in the image's pixel-map coordinates.

    ``num`` samples (default: about one per pixel of length), each the mean
    of ``width`` samples spaced one pixel apart across the line; spline
    ``order`` 1 is linear interpolation. The cut's ``label`` says where it
    is -- ``"panel (100, 240)-(500, 240)"`` -- so cuts of one image are told
    apart in plots, tables and collections.
    """
    from .image_transforms import sample
    from .mosaic import require_demosaiced

    require_demosaiced(image, "a line cut")
    width = int(width)
    if width < 1:
        raise ValueError("width must be at least 1")
    if coords:
        if image.pixel_map is None:
            raise ValueError(f"{image.label!r} has no pixel map; give start and end in pixels")
        mag = image.pixel_map.magnitude
        ends = [(mag(start[0]), mag(start[1])), (mag(end[0]), mag(end[1]))]
        (c0, c1), (r0, r1) = image.pixel_map.pixel_of(
            [ends[0][0], ends[1][0]], [ends[0][1], ends[1][1]])
        where = f" {units.format_unit(image.pixel_map.unit, pretty=True)}"
    else:
        (c0, r0), (c1, r1) = (float(v) for v in start), (float(v) for v in end)
        ends, where = [(c0, r0), (c1, r1)], ""
    if label is None:
        (a, b), (c, d) = ends
        label = f"{image.label} ({a:g}, {b:g})-({c:g}, {d:g}){where}"
        if width > 1:
            label += f", {width} px wide"
    length = float(np.hypot(c1 - c0, r1 - r0))
    if length == 0:
        raise ValueError("start and end are the same point")
    num = int(np.ceil(length)) + 1 if num is None else int(num)
    if num < 2:
        raise ValueError("a line cut needs at least 2 samples")

    t = np.linspace(0.0, 1.0, num)
    cols, rows = c0 + t * (c1 - c0), r0 + t * (r1 - r0)
    normal = np.array([-(r1 - r0), c1 - c0]) / length  # (column, row) across the line
    offsets = np.arange(width) - (width - 1) / 2
    values = np.mean([sample(image, cols + o * normal[0], rows + o * normal[1], order=order)
                      for o in offsets], axis=0)

    if image.pixel_map is not None:
        x, y = image.pixel_map.at(cols, rows)
        distance_unit = image.pixel_map.unit
    else:
        x, y = cols, rows
        distance_unit = units.parse("px")
    distance = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return LineCut(distance, values, x, y, label=label, distance_unit=distance_unit,
                   value_unit=image.value_unit, channels=image.channels,
                   metadata={"start": [float(c0), float(r0)], "end": [float(c1), float(r1)],
                             "width": width})


def row(image, index, *, label=None):
    """The line cut along row ``index``, one sample per pixel ("panel row 240")."""
    return line(image, (0, index), (image.width - 1, index), num=image.width, order=1,
                label=f"{image.label} row {index}" if label is None else label)


def column(image, index, *, label=None):
    """The line cut down column ``index``, one sample per pixel ("panel column 320")."""
    return line(image, (index, 0), (index, image.height - 1), num=image.height, order=1,
                label=f"{image.label} column {index}" if label is None else label)
