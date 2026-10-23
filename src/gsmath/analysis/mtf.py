"""Sharpness from a slanted edge: the ISO 12233 e-SFR method.

An image (or a region of one) containing a single straight edge, tilted a
few degrees from vertical or horizontal (2-10° is typical), gives the
system's modulation transfer function across that edge::

    edge = image.edge_mtf(roi=Rectangle(400, 300, 480, 400))
    edge.mtf50                       # cycles per pixel, per channel
    edge.frequency, edge.mtf         # the curve
    edge.frequency_in("1/deg")       # with a pixel map: cycles per degree

The method: each row's edge position is the centroid of its derivative
across the edge (refined inside a Hamming window), and a line through
those gives the edge's angle. Every pixel is then placed by its
perpendicular distance from that line -- the tilt spreads the rows over
sub-pixel phases -- and averaged into bins a quarter of a pixel wide: an
oversampled edge spread function (ESF). Its derivative, the line spread
function (LSF), is windowed (Hamming) and Fourier transformed; the MTF is its
magnitude, normalised to 1 at zero frequency and corrected for the
derivative filter's own response. Only a whole number of phase cycles of
rows is used, so every bin is filled evenly.

The data should be linear in light (calibrated values or raw counts after
the dark); an encoded image (sRGB, gamma) needs linearising first. Each
channel is analysed on its own, so colour images also show lateral
chromatic aberration as a shift of the edge between channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import units
from .base import Result


@dataclass(frozen=True, eq=False)
class EdgeMTF(Result):
    """The MTF across one slanted edge, per channel."""
    frequency: np.ndarray
    """Cycles per pixel, across the edge."""
    mtf: np.ndarray
    """``(n,)``, or ``(n, channels)``."""
    esf: np.ndarray
    """The oversampled edge spread function, per channel."""
    lsf: np.ndarray
    """The line spread function (before windowing), per channel."""
    position: np.ndarray
    """Each ESF/LSF sample's distance from the edge (px)."""
    angle: np.ndarray
    """The edge's tilt (degrees) from vertical (or horizontal), per channel."""
    offset: np.ndarray
    """Where each channel's edge crosses the region's middle row, relative to
    the first channel's (px): lateral chromatic aberration."""
    orientation: str
    """``"vertical"`` or ``"horizontal"``: which way the edge runs."""
    channels: tuple | None = None
    pitch: float | None = None
    """Pixel-map distance per pixel across the edge, if the image has a map."""
    pitch_unit: object = None
    metadata: dict = field(default_factory=dict)

    @property
    def curve(self):
        """The MTF against frequency (cycles/px) as a :class:`~gsmath.Series`."""
        from ..series import Series
        return Series(self.frequency, self.mtf, label="MTF", x_unit="1/px",
                      channels=self.channels if self.mtf.ndim == 2 else None,
                      x_label="Frequency")

    @property
    def mtf50(self):
        """Frequency (cycles/px) where the MTF falls to 0.5, per channel."""
        return self.frequency_at(0.5)

    @property
    def mtf10(self):
        """Frequency (cycles/px) where the MTF falls to 0.1, per channel."""
        return self.frequency_at(0.1)

    def frequency_at(self, level):
        """Where the MTF first falls to ``level`` (cycles/px; NaN if it
        doesn't within the range), per channel."""
        curves = self.mtf if self.mtf.ndim == 2 else self.mtf[:, None]
        found = np.array([_crossing(self.frequency, curve, level) for curve in curves.T])
        return found if self.mtf.ndim == 2 else float(found[0])

    def at(self, frequency):
        """The MTF at ``frequency`` (cycles/px), interpolated, per channel."""
        curves = self.mtf if self.mtf.ndim == 2 else self.mtf[:, None]
        found = np.array([np.interp(frequency, self.frequency, c) for c in curves.T])
        return found if self.mtf.ndim == 2 else float(found[0])

    def frequency_in(self, unit=None):
        """The frequency axis in cycles per pixel-map unit (``"1/deg"``,
        ``"1/mm"``, ...; by default per the map's own unit)."""
        if self.pitch is None:
            raise ValueError("the image had no pixel map, so frequencies are per pixel only")
        per_unit = units.Q_(self.frequency / self.pitch, 1 / self.pitch_unit)
        return per_unit if unit is None else per_unit.to(unit)

    @property
    def units(self):
        """MTF50 and MTF10 in cycles per pixel, the edge's tilt in degrees,
        the channels' offsets in pixels."""
        per_pixel = units.parse("1/px")
        return {"mtf50": per_pixel, "mtf10": per_pixel, "angle": units.parse("deg"),
                "offset": units.parse("px")}

    @property
    def summary(self):
        def value(v, i):
            return float(np.atleast_1d(v)[i])

        names = self.channels or ("value",)
        result = {name: {"mtf50": value(self.mtf50, i), "mtf10": value(self.mtf10, i),
                         "angle": value(self.angle, i), "offset": value(self.offset, i)}
                  for i, name in enumerate(names)}
        return result if self.mtf.ndim == 2 else result[names[0]]

    def plot(self, ax=None, *, per=None, **kwargs):
        """Plot MTF against frequency (cycles/px, or ``per`` a unit such as
        ``"1/deg"``) up to the oversampled limit; returns the axes."""
        import matplotlib.pyplot as plt

        from ..image_plotting import channel_colours, label_line

        ax = plt.subplots()[1] if ax is None else ax
        x = self.frequency if per is None else self.frequency_in(per).magnitude
        label = "cycles/px" if per is None else units.format_unit(units.parse(per), pretty=True)
        curves = self.mtf if self.mtf.ndim == 2 else self.mtf[:, None]
        names = self.channels if self.mtf.ndim == 2 else ("MTF",)
        mtf50 = np.atleast_1d(self.mtf50)
        scale = 1.0 if per is None else float(x[1] / self.frequency[1])
        shown = self.frequency <= 1.0
        for i, (name, colour) in enumerate(zip(names, channel_colours(names))):
            ax.plot(x[shown], curves[shown, i], label=name, color=colour,
                    **{"linewidth": 1.5, **kwargs})
            if len(names) > 1 and np.isfinite(mtf50[i]):
                label_line(ax, mtf50[i] * scale, 0.5, name)
        ax.set_xlabel(f"Frequency ({label})")
        ax.set_ylabel("MTF")
        ax.set_ylim(bottom=0)
        if len(names) > 1:
            ax.legend()
        return ax


def edge_mtf(image, roi=None, *, oversampling=4) -> EdgeMTF:
    """The MTF across the slanted edge in ``image`` (or in the bounding box
    of ``roi``); see the module notes."""
    from ..mosaic import require_demosaiced

    require_demosaiced(image, "an edge MTF")
    if roi is not None:
        image = image.crop(roi)
    oversampling = int(oversampling)
    if oversampling < 1:
        raise ValueError("oversampling must be a positive integer")
    values = np.asarray(image.values, dtype=float)
    planes = values[:, :, None] if values.ndim == 2 else values
    if not np.all(np.isfinite(planes)):
        raise ValueError("the edge region has non-finite pixels")

    reference = planes.mean(axis=2)
    across_columns = np.abs(np.diff(reference, axis=1)).sum()
    across_rows = np.abs(np.diff(reference, axis=0)).sum()
    orientation = "vertical" if across_columns >= across_rows else "horizontal"
    if orientation == "horizontal":
        planes = planes.transpose(1, 0, 2)
    if planes.shape[0] < 4 or planes.shape[1] < 8:
        raise ValueError(f"the edge region {image.shape} is too small")

    results = [_one_channel(planes[:, :, i], oversampling) for i in range(planes.shape[2])]
    # Common bins: the positions every channel covers.
    low = max(r["position"][0] for r in results)
    high = min(r["position"][-1] for r in results)
    step = 1.0 / oversampling
    position = np.arange(np.ceil(low / step), np.floor(high / step) + 1) * step
    esf = np.column_stack([np.interp(position, r["position"], r["esf"]) for r in results])
    frequency, mtf, lsf = _mtf(esf, step)
    middle = (planes.shape[0] - 1) / 2
    crossing = np.array([r["slope"] * middle + r["intercept"] for r in results])

    pitch, pitch_unit = _pitch(image, orientation, results[0]["slope"])
    single = values.ndim == 2
    return EdgeMTF(
        frequency=frequency,
        mtf=mtf[:, 0] if single else mtf,
        esf=esf[:, 0] if single else esf,
        lsf=lsf[:, 0] if single else lsf,
        position=position,
        angle=_squeeze(np.degrees(np.arctan([r["slope"] for r in results])), single),
        offset=_squeeze(crossing - crossing[0], single),
        orientation=orientation,
        channels=None if single else image.channels,
        pitch=pitch,
        pitch_unit=pitch_unit,
        metadata={"rows_used": int(results[0]["rows"]), "oversampling": oversampling,
                  "empty_bins": int(max(r["empty"] for r in results))},
    )


def _one_channel(plane, oversampling):
    """Edge line and oversampled ESF of one channel (edge running down the rows)."""
    rows, cols = plane.shape
    derivative = np.diff(plane, axis=1)  # at columns 0.5, 1.5, ...
    x = np.arange(cols - 1) + 0.5
    centroids = _centroids(derivative, x, None)
    slope, intercept = np.polyfit(np.arange(rows), centroids, 1)
    for _ in range(2):  # refine inside a Hamming window about the line
        centroids = _centroids(derivative, x, slope * np.arange(rows) + intercept)
        slope, intercept = np.polyfit(np.arange(rows), centroids, 1)

    if abs(slope) * rows < 1:
        raise ValueError(f"the edge is too close to vertical or horizontal "
                         f"({np.degrees(np.arctan(slope)):.2f}°) for this region: it must "
                         "shift by at least a pixel from top to bottom; tilt it 2-10°")
    if abs(slope) > np.tan(np.radians(45)):
        raise ValueError("the edge is tilted too far (over 45°)")
    used = int(np.floor(np.floor(rows * abs(slope)) / abs(slope)))  # whole phase cycles
    used = max(min(used, rows), 2)

    angle = np.arctan(slope)
    r, c = np.mgrid[0:used, 0:cols].astype(float)
    distance = (c - (slope * r + intercept)) * np.cos(angle)  # perpendicular, px
    edge_rows = slope * np.arange(used) + intercept
    low = np.max((0 - edge_rows) * np.cos(angle))
    high = np.min((cols - 1 - edge_rows) * np.cos(angle))
    step = 1.0 / oversampling
    first, last = int(np.ceil(low / step)), int(np.floor(high / step))
    index = np.round(distance / step).astype(int)
    keep = (index >= first) & (index <= last)
    counts = np.bincount(index[keep] - first, minlength=last - first + 1)
    sums = np.bincount(index[keep] - first, weights=plane[:used][keep], minlength=last - first + 1)
    position = np.arange(first, last + 1) * step
    filled = counts > 0
    if filled.sum() < 8:
        raise ValueError("too few pixels either side of the edge")
    esf = np.interp(position, position[filled], sums[filled] / counts[filled])
    return {"position": position, "esf": esf, "slope": slope, "intercept": intercept,
            "rows": used, "empty": int((~filled).sum())}


def _centroids(derivative, x, centres):
    """Each row's centroid of |derivative|, inside a Hamming window about
    `centres` if given."""
    weights = np.abs(derivative)
    if centres is not None:
        half = derivative.shape[1] / 2
        offset = (x[None, :] - centres[:, None]) / half
        window = np.where(np.abs(offset) <= 1, 0.54 + 0.46 * np.cos(np.pi * offset), 0.0)
        weights = weights * window
    totals = weights.sum(axis=1)
    if np.any(totals == 0):
        raise ValueError("a row of the edge region has no edge in it")
    return (weights * x).sum(axis=1) / totals


def _mtf(esf, step):
    """MTF from oversampled ESFs (columns), spacing `step` px."""
    lsf = np.gradient(esf, axis=0)  # central difference: [-1/2, 0, 1/2]
    lsf = lsf * np.sign(lsf.sum(axis=0))  # dark-to-light or light-to-dark alike
    n = lsf.shape[0]
    index = np.arange(n)[:, None]
    centre = (lsf * index).sum(axis=0) / lsf.sum(axis=0)
    half = np.maximum(centre, n - 1 - centre)
    window = 0.54 + 0.46 * np.cos(np.pi * (index - centre) / half)
    spectrum = np.abs(np.fft.rfft(lsf * window, axis=0))
    frequency = np.fft.rfftfreq(n, d=step)  # cycles per pixel
    mtf = spectrum / spectrum[0]
    # The central difference spans two bins: its response is sinc(2 f step).
    correction = 1 / np.maximum(np.sinc(2 * frequency * step), 0.1)
    return frequency, mtf * correction[:, None], lsf


def _pitch(image, orientation, slope):
    """Pixel-map distance per pixel across the edge, at the region's centre."""
    pixel_map = image.pixel_map
    if pixel_map is None:
        return None, None
    cx, cy = (image.width - 1) / 2, (image.height - 1) / 2
    normal = np.array([1.0, -slope]) / np.hypot(1.0, slope)  # (column, row), vertical edge
    if orientation == "horizontal":
        normal = normal[::-1]
    x, y = pixel_map.at(np.array([cx - normal[0] / 2, cx + normal[0] / 2]),
                        np.array([cy - normal[1] / 2, cy + normal[1] / 2]))
    return float(np.hypot(x[1] - x[0], y[1] - y[0])), pixel_map.unit


def _crossing(frequency, curve, level):
    below = np.flatnonzero(curve <= level)
    if below.size == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    f0, f1, m0, m1 = frequency[i - 1], frequency[i], curve[i - 1], curve[i]
    return float(f0 + (m0 - level) * (f1 - f0) / (m0 - m1))


def _squeeze(values, single):
    values = np.asarray(values, dtype=float)
    return float(values[0]) if single else values
