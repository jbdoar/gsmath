"""Descriptive statistics and histograms of image pixels, per channel.

Statistics ignore NaN pixels (masked or invalid), so ``n`` counts the finite
values. A single-channel image gives numbers; a multi-channel image gives an
array with one value per channel, and its summary is keyed by channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .base import Result
from .. import units

if TYPE_CHECKING:
    from ..image import Image

_STATISTICS = ("mean", "min", "max", "std", "median", "rms", "dynamic_range", "n")


@dataclass(frozen=True, eq=False)
class ImageStatistics(Result):
    """Cached per-channel statistics of an image's pixels, or a region's."""
    image: Image
    roi: object = None

    @property
    def label(self):
        return self.image.label

    @property
    def units(self) -> dict[str, object]:
        value = self.image.value_unit
        return {name: value for name in _STATISTICS[:-1]} | {"n": units.ureg.dimensionless}

    @property
    def summary(self) -> dict[str, object]:
        if self.image.n_channels == 1:
            return {name: _scalar(getattr(self, name)) for name in _STATISTICS}
        return {channel: {name: _scalar(getattr(self, name)[i]) for name in _STATISTICS}
                for i, channel in enumerate(self.image.channels)}

    @cached_property
    def pixels(self) -> np.ndarray:
        """The pixels analysed, as float: ``(n,)`` or ``(n, channels)``."""
        return np.asarray(self.image.pixels(self.roi), dtype=float)

    @property
    def _is_empty(self) -> bool:
        return self.pixels.shape[0] == 0

    def _reduce(self, function):
        if self._is_empty:
            return np.full(self.pixels.shape[1:], np.nan)[()]
        return function(self.pixels, axis=0)[()]

    @cached_property
    def mean(self):
        return self._reduce(np.nanmean)

    @cached_property
    def min(self):
        return self._reduce(np.nanmin)

    @cached_property
    def max(self):
        return self._reduce(np.nanmax)

    @cached_property
    def std(self):
        """Population standard deviation."""
        return self._reduce(np.nanstd)

    @cached_property
    def median(self):
        return self._reduce(np.nanmedian)

    @cached_property
    def rms(self):
        return self._reduce(lambda v, axis: np.sqrt(np.nanmean(v ** 2, axis=axis)))

    @cached_property
    def dynamic_range(self):
        return self.max - self.min

    @cached_property
    def n(self):
        """Finite pixels, per channel."""
        return np.sum(np.isfinite(self.pixels), axis=0)[()]

    def percentile(self, q):
        """The ``q``-th percentile(s) (0-100), per channel."""
        return self._reduce(lambda v, axis: np.nanpercentile(v, q, axis=axis))


@dataclass(frozen=True, eq=False)
class Histogram(Result):
    """Pixel counts in bins shared by every channel."""
    counts: np.ndarray
    """``(bins,)``, or ``(channels, bins)``."""
    edges: np.ndarray
    channels: tuple | None
    value_unit: object

    @property
    def units(self):
        return {"low": self.value_unit, "high": self.value_unit,
                "bins": units.ureg.dimensionless, "n": units.ureg.dimensionless}

    @property
    def summary(self):
        """The bins' span, and the pixels counted (per channel)."""
        counts = self.counts.sum(axis=-1)
        n = int(counts) if self.counts.ndim == 1 else {
            channel: int(c) for channel, c in zip(self.channels, counts)}
        return {"bins": len(self.edges) - 1, "low": float(self.edges[0]),
                "high": float(self.edges[-1]), "n": n}

    @property
    def centers(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)


def histogram(image, bins=256, *, range=None, roi=None) -> Histogram:
    """Histogram the finite pixels of ``image`` (or a region), each channel
    over the same bins: by default spanning every channel's values."""
    pixels = np.asarray(image.pixels(roi), dtype=float)
    finite = pixels[np.isfinite(pixels)]
    if range is None:
        range = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
    edges = np.histogram_bin_edges(finite, bins=bins, range=range)
    if pixels.ndim == 1:
        counts = np.histogram(pixels[np.isfinite(pixels)], bins=edges)[0]
    else:
        counts = np.stack([np.histogram(c[np.isfinite(c)], bins=edges)[0] for c in pixels.T])
    return Histogram(counts, edges, image.channels, image.value_unit)


def _scalar(value):
    return value.item() if isinstance(value, np.generic) else value
