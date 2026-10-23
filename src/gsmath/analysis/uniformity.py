"""Uniformity of a display's luminance and colour, from an X, Y, Z image.

The active area is the whole image, a pixel :class:`~gsmath.Rectangle`, or a
:class:`~gsmath.Quad` (e.g. from :func:`~gsmath.find_display`) -- so a
keystoned frame can be measured as it is: positions are fractions ``(u, v)``
of the active area (0, 0 its top-left corner, 1, 1 its bottom-right), mapped
into the image through the area's homography, and each sample spot is a
circle on the display, whatever shape it takes in the frame::

    result = xyz.uniformity(display, points=grid(3, 3, inset=0.1), spot=0.04)
    result.luminance["min_max_ratio"]
    result.color["max_delta_uv_prime"]
    result.points                       # one row per spot
    result.full_field(sigma=2)          # every pixel of the active area

Each spot's colour comes from its mean X, Y, Z (not the mean of x and y).
The *reference* for "from centre" metrics and the maps is the spot nearest
the middle of the area. No standard's layout or formula is implied: choose
the points and spot size your standard or specification calls for.

Luminance metrics (``result.luminance``): ``min``, ``max``, ``mean``,
``centre``, ``min_max_ratio`` (min/max), ``min_mean_ratio`` (min/mean),
``nonuniformity`` (1 - min/max), ``max_deviation_from_mean`` and
``max_deviation_from_centre`` (largest |Y - ref| / ref) and ``cv``
(population standard deviation / mean).

Colour metrics (``result.color``): ``x_range``, ``y_range``,
``max_delta_xy`` and ``max_delta_uv_prime`` (the largest difference between
any two spots), ``max_delta_xy_from_centre`` and
``max_delta_uv_prime_from_centre``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations

import numpy as np

from .. import units
from .base import Result
from .colorimetry import xyz_to_uv_prime, xyz_to_xy
from .image_color import XYZ_CHANNELS


def grid(rows, cols, *, inset=0.1):
    """``rows`` x ``cols`` points, evenly spaced, the outer ones ``inset``
    (a fraction of the area, or ``(across, down)``) in from the edges: a
    list of ``(u, v)``, in reading order."""
    across, down = np.broadcast_to(np.asarray(inset, dtype=float), (2,))
    if not (0 <= across < 0.5 and 0 <= down < 0.5):
        raise ValueError("inset must be at least 0 and less than half the area")
    us = [0.5] if cols == 1 else np.linspace(across, 1 - across, cols)
    vs = [0.5] if rows == 1 else np.linspace(down, 1 - down, rows)
    return [(float(u), float(v)) for v in vs for u in us]


def _uniformity_units(Y):
    """Units of uniformity metrics: luminance levels in the image's unit,
    ratios, deviations and chromaticity differences dimensionless."""
    one = units.ureg.dimensionless
    return {"luminance.min": Y, "luminance.max": Y, "luminance.mean": Y,
            "luminance.centre": Y, "luminance": one, "color": one, "points": one,
            "spot": one, "inset": one, "sigma": units.parse("px"),
            "percentiles": units.ureg.percent}


@dataclass(frozen=True, eq=False)
class Uniformity(Result):
    """Luminance and colour uniformity over sample spots (see the module);
    check it against a specification with :meth:`check`."""
    image: object
    area: object = None
    points: tuple = field(default_factory=lambda: tuple(grid(3, 3)))
    spot: float = 0.04
    """Spot diameter as a fraction of the active area's width."""

    def __post_init__(self):
        object.__setattr__(self, "points", tuple(tuple(map(float, p)) for p in self.points))
        if not self.points:
            raise ValueError("uniformity needs at least one sample point")
        if any(not (0 <= u <= 1 and 0 <= v <= 1) for u, v in self.points):
            raise ValueError("sample points are fractions of the active area, 0 to 1")
        if not 0 < self.spot < 1:
            raise ValueError("spot is a fraction of the active area's width, between 0 and 1")
        if self.image.values.ndim == 2 and self.image.channels is None:
            object.__setattr__(self, "image", self.image.copy(channels=("Y",)))  # luminance
        if "Y" not in (self.image.channels or ()):
            raise ValueError(f"{self.image.label!r} has no Y channel; uniformity needs "
                             "an X, Y, Z (or at least Y) image")

    # geometry
    @cached_property
    def _frame(self):
        from ..perspective import Quad, homography
        from ..roi import Rectangle

        area = self.area
        if area is None:
            w, h = self.image.width, self.image.height
            corners = [(-0.5, -0.5), (w - 0.5, -0.5), (w - 0.5, h - 0.5), (-0.5, h - 0.5)]
        elif isinstance(area, Rectangle) and not area.coords:
            (x0, x1), (y0, y1) = sorted((area.x0, area.x1)), sorted((area.y0, area.y1))
            corners = [(x0 - 0.5, y0 - 0.5), (x1 - 0.5, y0 - 0.5), (x1 - 0.5, y1 - 0.5),
                       (x0 - 0.5, y1 - 0.5)]
        elif isinstance(area, Quad):
            corners = area.corners
            if area.coords:
                corners = np.column_stack(self.image.pixel_map.pixel_of(corners[:, 0],
                                                                        corners[:, 1]))
        else:
            raise TypeError("the active area must be None, a pixel Rectangle or a Quad")
        corners = np.asarray(corners, dtype=float)
        return corners, homography([(0, 0), (1, 0), (1, 1), (0, 1)], corners)

    def to_pixels(self, u, v):
        """Image pixel positions (column, row) of area fractions ``(u, v)``."""
        from ..perspective import project
        return project(self._frame[1], u, v)

    @cached_property
    def aspect(self):
        """The active area's width / height, from its sides in the image."""
        from ..perspective import Quad
        width, height = Quad(tuple(map(tuple, self._frame[0]))).size
        return width / height

    def spot_region(self, u, v):
        """The spot at ``(u, v)``: a circle on the display, as a Polygon."""
        from ..roi import Polygon

        angle = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        radius = self.spot / 2
        cols, rows = self.to_pixels(u + radius * np.cos(angle),
                                    v + radius * self.aspect * np.sin(angle))
        return Polygon(tuple(zip(cols, rows)))

    def area_region(self, inset=0.0):
        """The active area (less ``inset`` of it at each edge), as a Polygon."""
        from ..roi import Polygon

        u = np.array([inset, 1 - inset, 1 - inset, inset])
        v = np.array([inset, inset, 1 - inset, 1 - inset])
        return Polygon(tuple(zip(*self.to_pixels(u, v))))

    # samples
    @cached_property
    def _XYZ(self):
        """Mean X, Y, Z (NaN where a channel is missing) per spot."""
        names = [c for c in XYZ_CHANNELS if c in self.image.channels]
        selected = self.image[names] if len(names) > 1 else self.image["Y"]
        rows = []
        for u, v in self.points:
            region = self.spot_region(u, v)
            count = region.submask(self.image)[1].sum()
            if count < 9:
                raise ValueError(f"the spot at ({u:g}, {v:g}) covers {count} pixel(s); "
                                 "make spot larger (at least 9 pixels)")
            mean = np.atleast_1d(selected.stats(roi=region).mean)
            rows.append([mean[names.index(c)] if c in names else np.nan for c in XYZ_CHANNELS])
        return np.array(rows)

    @property
    def has_color(self):
        return all(c in self.image.channels for c in XYZ_CHANNELS)

    @cached_property
    def reference(self) -> int:
        """Index of the spot nearest the middle of the area."""
        return int(np.argmin([np.hypot(u - 0.5, v - 0.5) for u, v in self.points]))

    @cached_property
    def points_table(self):
        """One dict per spot: u, v, column, row, X, Y, Z, x, y, u', v'."""
        xy = xyz_to_xy(self._XYZ)
        uv = xyz_to_uv_prime(self._XYZ)
        table = []
        for i, (u, v) in enumerate(self.points):
            col, row = self.to_pixels(u, v)
            X, Y, Z = self._XYZ[i]
            table.append({"u": u, "v": v, "column": float(col), "row": float(row),
                          "X": float(X), "Y": float(Y), "Z": float(Z),
                          "x": float(xy[i, 0]), "y": float(xy[i, 1]),
                          "u'": float(uv[i, 0]), "v'": float(uv[i, 1])})
        return table

    @property
    def points_frame(self):
        """The spots as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.points_table)

    # metrics
    @cached_property
    def luminance(self) -> dict:
        return _luminance_metrics(self._XYZ[:, 1], self._XYZ[self.reference, 1])

    @cached_property
    def color(self) -> dict:
        if not self.has_color:
            raise ValueError("colour uniformity needs X, Y and Z")
        return _color_metrics(xyz_to_xy(self._XYZ), xyz_to_uv_prime(self._XYZ), self.reference)

    @property
    def label(self):
        return self.image.label

    @property
    def units(self):
        return _uniformity_units(self.image.value_unit)

    @property
    def summary(self) -> dict:
        result = {"points": len(self.points), "spot": self.spot,
                  "luminance": dict(self.luminance)}
        if self.has_color:
            result["color"] = dict(self.color)
        return result

    def full_field(self, *, sigma=None, inset=0.0, percentiles=None):
        """Uniformity over every pixel of the active area (less ``inset``).

        ``sigma`` (px) smooths X, Y, Z first, to keep noise from setting the
        extremes; ``percentiles`` ``(low, high)``, e.g. ``(1, 99)``, uses
        those instead of the minimum and maximum. Returns a
        :class:`FullFieldUniformity` with the metrics and the maps."""
        return FullFieldUniformity(self, sigma, inset, percentiles)

    def plot(self, ax=None, **kwargs):
        """Luminance with the spots outlined and each labelled with its
        Y relative to the reference; returns the axes."""
        from ..image_plotting import INK

        ax = self.image["Y"].plot(ax, coords=False, **kwargs)
        reference = self._XYZ[self.reference, 1]
        for i, (u, v) in enumerate(self.points):
            outline = np.array(self.spot_region(u, v).vertices + self.spot_region(u, v).vertices[:1])
            ax.plot(outline[:, 0], outline[:, 1], color=INK, linewidth=1)
            col, row = self.to_pixels(u, v)
            ax.annotate(f"{self._XYZ[i, 1] / reference:.3f}", (col, row), xytext=(0, 9),
                        textcoords="offset points", ha="center", color=INK, fontsize="small",
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none",
                              "alpha": 0.8})
        corners = np.vstack([self._frame[0], self._frame[0][:1]])
        ax.plot(corners[:, 0], corners[:, 1], color=INK, linewidth=1, linestyle="--")
        return ax


@dataclass(frozen=True, eq=False)
class FullFieldUniformity(Result):
    """Uniformity over every pixel of the active area, with maps."""
    sampled: Uniformity
    sigma: float | None = None
    inset: float = 0.0
    percentiles: tuple | None = None

    @cached_property
    def _mask(self):
        return self.sampled.area_region(self.inset).mask(self.sampled.image)

    @cached_property
    def _image(self):
        image = self.sampled.image
        names = [c for c in XYZ_CHANNELS if c in image.channels]
        image = image[names] if len(names) > 1 else image["Y"]
        return image if self.sigma is None else image.smooth("gaussian", sigma=self.sigma)

    def _channel(self, name):
        """One channel of the (smoothed) frame as a (rows, columns) array."""
        image = self._image
        return image.values if image.values.ndim == 2 else image[name].values

    @cached_property
    def _pixels(self) -> dict:
        """The active area's pixels: one 1-D array per channel present.
        (Separate arrays, selected once: whole-frame copies and (N, 3)
        column slicing are what make per-pixel colour slow on large frames.)"""
        names = [c for c in XYZ_CHANNELS if c in self._image.channels] \
            if self._image.values.ndim == 3 else ["Y"]
        return {name: np.asarray(self._channel(name), dtype=float)[self._mask] for name in names}

    @property
    def _reference_XYZ(self):
        return self.sampled._XYZ[self.sampled.reference]

    @cached_property
    def luminance(self) -> dict:
        Y = self._pixels["Y"]
        Y = Y[np.isfinite(Y)]
        extremes = None if self.percentiles is None else np.percentile(Y, self.percentiles)
        return _luminance_metrics(Y, self._reference_XYZ[1], extremes)

    @cached_property
    def color(self) -> dict:
        if not self.sampled.has_color:
            raise ValueError("colour uniformity needs X, Y and Z")
        X, Y, Z = (self._pixels[c] for c in XYZ_CHANNELS)
        with np.errstate(divide="ignore", invalid="ignore"):
            total, denominator = X + Y + Z, X + 15 * Y + 3 * Z
            x, y = X / total, Y / total
            u, v = 4 * X / denominator, 9 * Y / denominator
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
        if not valid.all():
            x, y, u, v = x[valid], y[valid], u[valid], v[valid]
        x0, y0 = xyz_to_xy(self._reference_XYZ)
        u0, v0 = xyz_to_uv_prime(self._reference_XYZ)
        # Squared distances: the root of the largest is the largest root.
        # (A percentile interpolates between values, so it takes the roots.)
        from_ref_xy = (x - x0) ** 2 + (y - y0) ** 2
        from_ref_uv = (u - u0) ** 2 + (v - v0) ** 2
        if self.percentiles is not None:
            high = self.percentiles[1]
            limit = lambda d: float(np.percentile(np.sqrt(d), high))
            spread = lambda c: float(np.percentile(c, high) - np.percentile(c, 100 - high))
        else:
            limit = lambda d: float(np.sqrt(np.max(d)))
            spread = lambda c: float(np.ptp(c))
        return {"x_range": spread(x), "y_range": spread(y),
                "max_delta_xy_from_centre": limit(from_ref_xy),
                "max_delta_uv_prime_from_centre": limit(from_ref_uv)}

    @property
    def relative_luminance(self):
        """Y / the reference spot's Y, NaN outside the area: an Image."""
        image = self.sampled.image
        Y = np.asarray(self._channel("Y"), dtype=float)
        return image["Y"].copy(values=np.where(self._mask, Y / self._reference_XYZ[1], np.nan),
                               value_unit=None, channels=None,
                               label=f"{image.label} relative luminance")

    @property
    def delta_uv_prime(self):
        """Δu'v' from the reference spot, NaN outside the area: an Image."""
        if not self.sampled.has_color:
            raise ValueError("colour uniformity needs X, Y and Z")
        X, Y, Z = (np.asarray(self._channel(c), dtype=float) for c in XYZ_CHANNELS)
        u0, v0 = xyz_to_uv_prime(self._reference_XYZ)
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = X + 15 * Y + 3 * Z
            distance = np.hypot(4 * X / denominator - u0, 9 * Y / denominator - v0)
        image = self.sampled.image
        return image["Y"].copy(values=np.where(self._mask & np.isfinite(distance), distance,
                                               np.nan),
                               value_unit=None, channels=None, label=f"{image.label} Δu'v'")

    @property
    def label(self):
        return self.sampled.image.label

    @property
    def units(self):
        return _uniformity_units(self.sampled.image.value_unit)

    @property
    def summary(self):
        result = {"sigma": self.sigma, "inset": self.inset,
                  "percentiles": None if self.percentiles is None else list(self.percentiles),
                  "luminance": dict(self.luminance)}
        if self.sampled.has_color:
            result["color"] = dict(self.color)
        return result


def _luminance_metrics(Y, reference, extremes=None):
    """Luminance metrics; `extremes` (low, high) replace the minimum and
    maximum (percentiles), for the ratios and deviations alike."""
    Y = np.asarray(Y, dtype=float)
    mean = float(np.mean(Y))
    low, high = (float(np.min(Y)), float(np.max(Y))) if extremes is None else map(float, extremes)
    deviation = lambda ref: max(abs(high - ref), abs(low - ref)) / ref
    return {
        "min": low, "max": high, "mean": mean, "centre": float(reference),
        "min_max_ratio": low / high,
        "min_mean_ratio": low / mean,
        "nonuniformity": 1 - low / high,
        "max_deviation_from_mean": float(deviation(mean)),
        "max_deviation_from_centre": float(deviation(reference)),
        "cv": float(np.std(Y) / mean),
    }


def _color_metrics(xy, uv, reference):
    pairs = list(combinations(range(len(xy)), 2))

    def largest(points):
        return max((float(np.hypot(*(points[i] - points[j]))) for i, j in pairs), default=0.0)

    return {
        "x_range": float(np.ptp(xy[:, 0])), "y_range": float(np.ptp(xy[:, 1])),
        "max_delta_xy": largest(xy), "max_delta_uv_prime": largest(uv),
        "max_delta_xy_from_centre": float(np.max(np.hypot(*(xy - xy[reference]).T))),
        "max_delta_uv_prime_from_centre": float(np.max(np.hypot(*(uv - uv[reference]).T))),
    }
