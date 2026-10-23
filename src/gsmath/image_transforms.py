"""Pure transformations that return new :class:`~gsmath.Image` objects.

Each pixel keeps its coordinates: flipping or rotating rearranges the pixel
map with the values (so a separable map becomes per-pixel after a quarter
turn), binning averages it, and resampling interpolates it. To put an image
on a chosen coordinate grid -- a regular angle grid, say -- use
:func:`interpolate` with that grid's :class:`~gsmath.PixelMap`.

A raw colour-filter mosaic can be flipped, rotated and sliced (the pattern
follows); anything that would mix neighbouring pixels -- smoothing, binning,
resampling, line cuts -- asks for it to be demosaiced first.
"""

import numbers

import numpy as np
import scipy.ndimage

from .mosaic import require_demosaiced
from ._utils import factor_pair
from .pixelmap import PixelMap

SMOOTHING = ("gaussian", "box", "median")
NORMALIZATIONS = ("peak", "mean", "median")


def flip(image, axis="horizontal"):
    """Mirror left-right (``"horizontal"``) or top-bottom (``"vertical"``)."""
    if axis not in ("horizontal", "vertical"):
        raise ValueError(f"axis must be 'horizontal' or 'vertical', not {axis!r}")
    rows, cols = (slice(None), slice(None, None, -1)) if axis == "horizontal" else \
        (slice(None, None, -1), slice(None))
    flipped = image[rows, cols]
    return flipped.copy(label=f"{image.label} [flipped {axis}]")


def rotate90(image, k=1):
    """Rotate by ``k`` quarter turns counter-clockwise (``numpy.rot90``)."""
    k = int(k) % 4
    h, w = image.shape
    old = {0: lambda i, j: (i, j),
           1: lambda i, j: (j, w - 1 - i),
           2: lambda i, j: (h - 1 - i, w - 1 - j),
           3: lambda i, j: (h - 1 - j, i)}[k]
    return image.copy(
        values=np.rot90(image.values, k, axes=(0, 1)).copy(),
        pixel_map=None if image.pixel_map is None else image.pixel_map.rotated90(k),
        mosaic=None if image.mosaic is None else image.mosaic.remapped(old, "rotating"),
        label=f"{image.label} [rotated {90 * k}]",
    )


def bin(image, factor, *, method="mean"):
    """Combine ``factor`` x ``factor`` blocks (or ``(rows, columns)``) into one
    pixel by their mean or sum; a partial block at the far edges is dropped.
    A NaN pixel makes its block NaN."""
    require_demosaiced(image, "binning")
    if method not in ("mean", "sum"):
        raise ValueError(f"method must be 'mean' or 'sum', not {method!r}")
    fy, fx = factor_pair(factor)
    rows, cols = image.height // fy, image.width // fx
    if rows == 0 or cols == 0:
        raise ValueError(f"can't bin a {image.shape} image by {(fy, fx)}")
    values = np.asarray(image.values[:rows * fy, :cols * fx], dtype=float)
    blocks = values.reshape(rows, fy, cols, fx, *values.shape[2:])
    combined = blocks.mean(axis=(1, 3)) if method == "mean" else blocks.sum(axis=(1, 3))
    return image.copy(
        values=combined,
        pixel_map=None if image.pixel_map is None else image.pixel_map.binned((fy, fx)),
        label=f"{image.label} [binned {fy}x{fx}]",
    )


def smooth(image, method="gaussian", **kwargs):
    """Filter each channel spatially.

    - ``"gaussian"``: ``sigma`` in pixels (one, or ``(rows, columns)``);
    - ``"box"``: the mean over a ``size`` x ``size`` window;
    - ``"median"``: the median over a ``size`` x ``size`` window.

    Gaussian and box filters skip NaN pixels (each output is the weighted
    mean of the finite pixels near it; NaN pixels stay NaN). The median
    needs finite data.
    """
    require_demosaiced(image, "smoothing")
    if method not in SMOOTHING:
        raise ValueError(f"Unknown smoothing method {method!r}; expected one of {SMOOTHING}")
    values = np.asarray(image.values, dtype=float)
    finite = np.isfinite(values)

    if method == "median":
        if not finite.all():
            raise ValueError("median smoothing needs finite data; mask NaNs another way first")
        size = _window(kwargs.pop("size", 3))
        smoothed = _per_channel(values, lambda v: scipy.ndimage.median_filter(
            v, size=size, mode="reflect"))
    else:
        if method == "gaussian":
            sigma = kwargs.pop("sigma", 1.0)
            spatial = lambda v: scipy.ndimage.gaussian_filter(v, sigma=sigma, mode="reflect")
        else:
            size = _window(kwargs.pop("size", 3))
            spatial = lambda v: scipy.ndimage.uniform_filter(v, size=size, mode="reflect")
        weights = _per_channel(finite.astype(float), spatial)
        sums = _per_channel(np.where(finite, values, 0.0), spatial)
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = np.where(finite, sums / weights, np.nan)
    if kwargs:
        raise TypeError(f"unexpected arguments for {method} smoothing: {sorted(kwargs)}")
    return image.copy(values=smoothed, label=f"{image.label} [{method} smoothed]")


def normalize(image, by="peak", value=1.0, *, roi=None, per_channel=False):
    """Scale so the image's (or a region's) peak, mean or median equals
    ``value`` -- the same scale for every channel, or each channel on its
    own with ``per_channel`` (a white balance against a neutral region)."""
    if not isinstance(value, numbers.Real):
        raise TypeError("value must be a real number.")
    if by not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization {by!r}; expected one of {NORMALIZATIONS}")
    stats = image.stats(roi=roi)
    current = np.asarray({"peak": stats.max, "mean": stats.mean, "median": stats.median}[by])
    if not per_channel and current.ndim:
        current = np.asarray({"peak": np.nanmax, "mean": np.nanmean,
                              "median": np.nanmedian}[by](stats.pixels))
    if not np.all(np.isfinite(current)):
        raise ValueError("Normalization quantity is not finite.")
    if np.any(current == 0):
        raise ValueError("Cannot normalize by zero.")
    return image.copy(values=np.asarray(image.values, dtype=float) * (value / current),
                      label=f"{image.label} [{by} normalized]")


def scale_channels(image, gains):
    """Multiply each channel by its gain: a mapping ``{"R": 1.2, ...}``
    (unnamed channels keep gain 1) or one gain per channel, in order."""
    if image.channels is None or image.values.ndim != 3:
        raise ValueError(f"{image.label!r} has no channels to scale")
    if isinstance(gains, dict):
        unknown = set(gains) - set(image.channels)
        if unknown:
            raise KeyError(f"{image.label!r} has no channel {sorted(unknown)[0]!r}")
        gains = [gains.get(name, 1.0) for name in image.channels]
    gains = np.asarray(gains, dtype=float)
    if gains.shape != (image.n_channels,):
        raise ValueError(f"{gains.size} gains for {image.n_channels} channels")
    return image.copy(values=np.asarray(image.values, dtype=float) * gains,
                      label=f"{image.label} [channels scaled]")


def resize(image, shape, *, order=1):
    """Resample to ``shape`` (rows, columns) over the same field: pixel
    centres stay aligned with the original pixels' footprint (spline
    ``order``: 0 nearest, 1 bilinear, 3 cubic). For shrinking by a whole
    factor, :func:`bin` averages without interpolating."""
    require_demosaiced(image, "resizing")
    rows, cols = (int(n) for n in shape)
    if rows < 1 or cols < 1:
        raise ValueError(f"shape must be positive, not {shape!r}")
    row_at = (np.arange(rows) + 0.5) * image.height / rows - 0.5
    col_at = (np.arange(cols) + 0.5) * image.width / cols - 0.5
    pixel_map = image.pixel_map
    if pixel_map is not None:
        if pixel_map.separable:
            x, _ = pixel_map.at(col_at, np.zeros(1))
            _, y = pixel_map.at(np.zeros(1), row_at)
            pixel_map = PixelMap(x, y, pixel_map.unit)
        else:
            c, r = np.meshgrid(col_at, row_at)
            pixel_map = PixelMap(*pixel_map.at(c, r), pixel_map.unit)
    c, r = np.meshgrid(col_at, row_at)
    return image.copy(values=sample(image, c, r, order=order), pixel_map=pixel_map,
                      label=f"{image.label} [resized]")


def interpolate(image, pixel_map, *, order=1, fill=np.nan):
    """Resample onto the coordinates of ``pixel_map`` (separable or per
    pixel), e.g. a regular angle grid. The image's own map must be separable
    and monotonic; target points outside it are ``fill``."""
    require_demosaiced(image, "interpolating")
    if image.pixel_map is None:
        raise ValueError(f"{image.label!r} has no pixel map to interpolate from")
    target = pixel_map.to_unit(image.pixel_map.unit)
    x, y = target.grid()
    cols, rows = image.pixel_map.pixel_of(x, y)
    values = sample(image, cols, rows, order=order)
    outside = (cols < -0.5) | (cols > image.width - 0.5) | (rows < -0.5) | (rows > image.height - 0.5)
    values[outside] = fill
    return image.copy(values=values, pixel_map=pixel_map, label=f"{image.label} [interpolated]")


def sample(image, cols, rows, *, order=1):
    """Values at fractional pixel positions (arrays of one shape), per
    channel: ``cols.shape`` or ``cols.shape + (channels,)``. Beyond the
    edges, the nearest edge value."""
    cols, rows = np.asarray(cols, dtype=float), np.asarray(rows, dtype=float)
    positions = np.stack([rows.ravel(), cols.ravel()])
    values = np.asarray(image.values, dtype=float)

    def one(plane):
        return scipy.ndimage.map_coordinates(plane, positions, order=order, mode="nearest",
                                             prefilter=order > 1).reshape(cols.shape)

    if values.ndim == 2:
        return one(values)
    return np.stack([one(values[:, :, i]) for i in range(values.shape[2])], axis=-1)


def _per_channel(values, function):
    if values.ndim == 2:
        return function(values)
    return np.stack([function(values[:, :, i]) for i in range(values.shape[2])], axis=-1)


def _window(size):
    if not isinstance(size, numbers.Integral) or size < 1:
        raise ValueError(f"size must be a positive integer, not {size!r}")
    return int(size)


def apply_matrix(image, matrix, *, channels=("X", "Y", "Z"), unit=None):
    """Mix channels by a matrix: each pixel's output is ``matrix @ pixel``,
    e.g. camera R, G, B to X, Y, Z. ``matrix`` is ``(outputs, inputs)``;
    ``channels`` names the outputs; ``unit`` is the matrix's own unit (the
    result's unit is the image's times it)."""
    from . import units

    require_demosaiced(image, "a colour matrix")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != image.n_channels:
        raise ValueError(f"a {matrix.shape} matrix can't apply to {image.n_channels} channel(s)")
    if len(channels) != matrix.shape[0]:
        raise ValueError(f"{len(channels)} channel names for {matrix.shape[0]} matrix rows")
    values = np.asarray(image.values, dtype=float)
    if values.ndim == 2:
        values = values[:, :, None]
    mixed = values @ matrix.T
    return image.copy(values=mixed[:, :, 0] if mixed.shape[2] == 1 else mixed,
                      channels=tuple(channels),
                      value_unit=units.multiply_units(image.value_unit, unit),
                      label=f"{image.label} [matrix]")
