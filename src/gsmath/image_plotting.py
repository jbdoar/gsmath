"""Matplotlib display of images, with axes in pixel-map units.

One channel is drawn through a colormap with a colorbar in the image's unit:
a single-hue sequential ramp (light = low) for magnitudes, or, when the data
has both signs in earnest (Duv, a*, b*, a difference -- the smaller side at
least a tenth of the larger), a diverging blue-gray-red ramp centred on zero. R, G, B images are drawn as colour, scaled so ``vmax`` (by
default the 99.9th percentile) is full scale; pass ``srgb=True`` to encode
linear data for display. Other multi-channel images need a ``channel``.

Axes are the pixel map's coordinates when the image has one (``coords``),
else pixel indices; row 0 is at the top, and a map whose y increases upward
reads upward. A uniform separable map is drawn as an image; any other map as
a mesh.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import units

SEQUENTIAL = LinearSegmentedColormap.from_list("gsmath_sequential", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
"""One hue, light to dark: magnitude (luminance, CCT, colour difference)."""

DIVERGING = LinearSegmentedColormap.from_list("gsmath_diverging", [
    "#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a9a8", "#e34948", "#a32a29"])
"""Blue to a neutral gray to red: signed values, the gray at zero."""

_RGB = (("R", "G", "B"), ("R", "G", "B", "A"))


def plot(image, ax=None, *, channel=None, coords=True, cmap=None, vmin=None, vmax=None,
         colorbar=True, srgb=False, title=None):
    """Draw ``image`` on ``ax`` (a new figure by default); returns the axes."""
    import matplotlib.pyplot as plt

    from .mosaic import require_demosaiced

    require_demosaiced(image, "displaying colour")
    if channel is not None:
        image = image.channel(channel)
    rgb = image.channels in _RGB and image.values.ndim == 3
    if image.values.ndim == 3 and not rgb:
        raise ValueError(f"{image.label!r} has channels {image.channels}; "
                         "choose one to plot with channel=")
    ax = plt.subplots()[1] if ax is None else ax
    values = np.asarray(image.values, dtype=float)
    pixel_map = image.pixel_map if coords else None

    if rgb:
        values = values[:, :, :3]
        top = np.nanpercentile(values, 99.9) if vmax is None else vmax
        values = np.clip(values / (top if top > 0 else 1.0), 0, 1)
        if srgb:
            values = np.where(values <= 0.0031308, 12.92 * values,
                              1.055 * values ** (1 / 2.4) - 0.055)
        artist = _draw(ax, values, pixel_map)
    else:
        norm, cmap = _scale(values, cmap, vmin, vmax)
        artist = _draw(ax, values, pixel_map, cmap=cmap, norm=norm)
        if colorbar:
            bar = ax.figure.colorbar(artist, ax=ax)
            name = image.channels[0] if image.channels else ""
            unit = units.format_unit(image.value_unit, pretty=True)
            bar.set_label(f"{name} ({unit})".strip() if unit else name)

    unit = "px" if pixel_map is None else units.format_unit(pixel_map.unit, pretty=True)
    ax.set_xlabel(f"x ({unit})" if unit else "x")
    ax.set_ylabel(f"y ({unit})" if unit else "y")
    ax.set_title(image.label if title is None else title)
    ax.set_aspect("equal")
    return ax


def _scale(values, cmap, vmin, vmax):
    """A norm and colormap: diverging about zero for signed data."""
    from matplotlib.colors import Normalize

    finite = values[np.isfinite(values)]
    low = vmin if vmin is not None else (float(finite.min()) if finite.size else 0.0)
    high = vmax if vmax is not None else (float(finite.max()) if finite.size else 1.0)
    # Diverging only when both signs matter: a dark-subtracted frame's noise
    # dipping below zero shouldn't turn a luminance image red and blue.
    if low < 0 < high and min(-low, high) >= 0.1 * max(-low, high):
        if vmin is None and vmax is None:  # symmetric, so the midpoint gray is zero
            span = max(-low, high)
            low, high = -span, span
        return TwoSlopeNorm(0.0, low, high), DIVERGING if cmap is None else cmap
    if high <= low:
        high = low + 1.0
    return Normalize(low, high), SEQUENTIAL if cmap is None else cmap


def _draw(ax, values, pixel_map, **kwargs):
    if pixel_map is None:
        rows, cols = values.shape[:2]
        return ax.imshow(values, origin="upper", extent=(-0.5, cols - 0.5, rows - 0.5, -0.5),
                         interpolation="nearest", **kwargs)
    if pixel_map.separable and _uniform(pixel_map.x) and _uniform(pixel_map.y):
        x, y = pixel_map.x, pixel_map.y
        sx = x[1] - x[0] if x.size > 1 else 1.0
        sy = y[1] - y[0] if y.size > 1 else 1.0
        extent = (x[0] - sx / 2, x[-1] + sx / 2, y[-1] + sy / 2, y[0] - sy / 2)
        return ax.imshow(values, origin="upper", extent=extent, interpolation="nearest",
                         **kwargs)
    if values.ndim == 3:
        raise ValueError("colour display needs a uniform separable pixel map; "
                         "plot one channel, or pixel axes with coords=False")
    X, Y = pixel_map.grid()
    artist = ax.pcolormesh(X, Y, values, shading="nearest", **kwargs)
    if pixel_map.separable and pixel_map.y.size > 1 and pixel_map.y[1] > pixel_map.y[0]:
        ax.invert_yaxis()  # row 0 at the top
    return artist


def _uniform(axis):
    steps = np.diff(axis)
    return steps.size == 0 or np.allclose(steps, steps[0], rtol=1e-6, atol=0)


# Lines per channel: R, G, B in their own colours; anything else in the
# palette's fixed categorical order (assigned in order, never cycled).
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
               "#e34948")
_CHANNEL_COLOURS = {"R": "#e34948", "G": "#008300", "B": "#2a78d6"}
LINE_STYLES = ("-", "--", ":", "-.")
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"


def channel_colours(names):
    """A colour per channel name. Red and green are close for protan
    viewers, so label lines directly as well (see :func:`label_line`)."""
    names = tuple(names)
    if names and all(name in _CHANNEL_COLOURS for name in names):
        return [_CHANNEL_COLOURS[name] for name in names]
    if len(names) > len(CATEGORICAL):
        raise ValueError(f"{len(names)} channels is more than can be told apart by colour; "
                         "plot fewer at once")
    return list(CATEGORICAL[:len(names)])


def label_line(ax, x, y, text):
    """Name a line directly beside the point (x, y), in ink, not its colour."""
    ax.annotate(text, (x, y), xytext=(6, 4), textcoords="offset points", color=INK,
                fontsize="small")
