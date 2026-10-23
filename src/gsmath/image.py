"""Sampled images: pixel values with channels, units, coordinates and metadata.

An :class:`Image` is to a camera what a :class:`~gsmath.Spectrum` is to a
spectrometer. Its values are a ``(rows, columns)`` array for one channel --
intensity, luminance, a raw sensor mosaic -- or ``(rows, columns, channels)``
with named channels (``("R", "G", "B")``, ``("X", "Y", "Z")``). It carries the
value unit, a label, metadata, and optionally a :class:`~gsmath.PixelMap`
giving each pixel's angle or position.

Like a Spectrum, operations return new images, arithmetic is unit-aware and
requires matching grids, and an image works in a
:class:`~gsmath.Measurement` and through a :class:`~gsmath.Calibration`::

    raw = Image(frame, value_unit="count", channels=("R", "G", "B"))
    centre = raw[400:800, 600:1000]            # pixel_map sliced with it
    green = raw["G"]
    raw.stats(roi=Circle(800, 600, 50)).mean   # per channel
"""

import numpy as np

from . import units
from .pixelmap import PixelMap
from .sampled import KEEP as _KEEP
from .sampled import Sampled, read_only
from .sampled import as_float as _float


class Image(Sampled):
    """Pixel values on a (rows, columns) grid, with units and metadata."""

    def __init__(
            self,
            values,
            label="Image",
            value_unit=None,
            channels=None,
            pixel_map=None,
            metadata=None,
            mosaic=None,
    ):
        """Create an image, validating the array, channels and pixel map.
        ``mosaic`` (a :class:`~gsmath.mosaic.Mosaic`, or a pattern such as
        ``"RGGB"``) marks single-channel raw colour-filter data."""
        values = np.asarray(values)
        if values.ndim not in (2, 3):
            raise ValueError("Image values must be (rows, columns) or (rows, columns, channels).")
        if values.dtype.hasobject:
            raise TypeError("Image values cannot use object dtype.")
        self.values = read_only(values)
        self.label = label
        self.value_unit = units.parse(value_unit)
        self.channels = _channels(channels, values)
        if pixel_map is not None:
            if not isinstance(pixel_map, PixelMap):
                raise TypeError("pixel_map must be a PixelMap")
            if pixel_map.shape != values.shape[:2]:
                raise ValueError(f"pixel map shape {pixel_map.shape} doesn't match "
                                 f"the image {values.shape[:2]}")
        self.pixel_map = pixel_map
        from .mosaic import Mosaic
        self.mosaic = Mosaic.coerce(mosaic)
        if self.mosaic is not None and values.ndim != 2:
            raise ValueError("a colour-filter mosaic is single-channel (rows, columns) data")
        self.metadata = {} if metadata is None else dict(metadata)
        self._seal()

    # shape
    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns)."""
        return self.values.shape[:2]

    @property
    def height(self) -> int:
        return self.values.shape[0]

    @property
    def width(self) -> int:
        return self.values.shape[1]

    @property
    def n_channels(self) -> int:
        return 1 if self.values.ndim == 2 else self.values.shape[2]

    def __str__(self):
        channels = f", {''.join(self.channels)}" if self.channels else ""
        unit = f" {self.value_unit:~}" if self.value_unit is not None else ""
        return f"{self.label} ({self.height}x{self.width}{channels}{unit})"

    def __repr__(self):
        return (f"Image(label={self.label!r}, shape={self.values.shape}, "
                f"channels={self.channels!r}, value_unit={str(self.value_unit)!r}, "
                f"pixel_map={self.pixel_map!r}"
                + (f", mosaic={self.mosaic.pattern!r}" if self.mosaic else "") + ")")

    # selection
    def __getitem__(self, key):
        """``image[rows, cols]`` (slices), ``image["G"]`` (a channel),
        ``image[["R", "G"]]`` (channels) or ``image[rows, cols, "G"]``."""
        if isinstance(key, (str, list)):
            return self.channel(key)
        if not isinstance(key, tuple) or len(key) not in (2, 3):
            raise TypeError("index an image as image[rows, cols], image[rows, cols, channel] "
                            "or image[channel]")
        rows, cols = key[:2]
        if not (isinstance(rows, slice) and isinstance(cols, slice)):
            raise TypeError("rows and columns must be slices (image[i:i+1, :] for one row)")
        image = self.copy(
            values=self.values[rows, cols],
            pixel_map=None if self.pixel_map is None else self.pixel_map.sliced(rows, cols),
            mosaic=None if self.mosaic is None else self.mosaic.sliced(rows, cols, self.shape),
        )
        return image if len(key) == 2 else image.channel(key[2])

    def channel(self, name):
        """One named channel as a single-channel image, or a list of names as
        a multi-channel image."""
        if self.channels is None:
            raise ValueError(f"{self.label!r} has no named channels")
        names = [name] if isinstance(name, str) else list(name)
        missing = [n for n in names if n not in self.channels]
        if missing:
            raise KeyError(f"{self.label!r} has no channel {missing[0]!r}; it has {self.channels}")
        if not isinstance(name, str) and tuple(names) == self.channels:
            return self  # all of them, in order: nothing to select
        index = [self.channels.index(n) for n in names]
        if self.values.ndim == 2:
            values = self.values
        elif isinstance(name, str):
            values = self.values[:, :, index[0]]
        else:
            values = self.values[:, :, index]
        return self.copy(values=values, channels=tuple(names))

    def crop(self, roi):
        """The smallest rectangle of the image containing ``roi``'s pixels."""
        rows, cols = np.nonzero(roi.mask(self))
        if rows.size == 0:
            raise ValueError("the region contains no pixels of the image")
        return self[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    def coordinates(self):
        """``(X, Y)``: each pixel's mapped coordinates, or its column and row
        indices if the image has no pixel map."""
        if self.pixel_map is not None:
            return self.pixel_map.grid()
        return np.meshgrid(np.arange(self.width), np.arange(self.height))

    # io
    @classmethod
    def read(cls, path, *, format=None, **kwargs):
        """Read an image using an inferred or explicitly named format: native
        ``.gsmath``, PNG, JPEG, TIFF, or a camera raw file (as its mosaic)."""
        from .io.formats import read_image
        return read_image(path, format=format, **kwargs)

    def write(self, path, *, format=None, **kwargs):
        """Write this image: native ``.gsmath`` keeps everything; TIFF keeps
        the data exactly; PNG and JPEG need uint8 (or uint16 PNG) values."""
        from .io.formats import write_image
        return write_image(self, path, format=format, **kwargs)

    # copies
    def copy(
            self,
            *,
            values=None,
            label=None,
            value_unit=_KEEP,
            channels=_KEEP,
            pixel_map=_KEEP,
            metadata=None,
            mosaic=_KEEP,
    ):
        """Copy the image with selected fields replaced (``value_unit=None``,
        ``channels=None``, ``pixel_map=None`` or ``mosaic=None`` removes them)."""
        return type(self)(
            self.values if values is None else values,  # read-only: shared
            label=self.label if label is None else label,
            value_unit=self.value_unit if value_unit is _KEEP else value_unit,
            channels=self.channels if channels is _KEEP else channels,
            pixel_map=self.pixel_map if pixel_map is _KEEP else pixel_map,
            metadata=self.metadata.copy() if metadata is None else metadata,
            mosaic=self.mosaic if mosaic is _KEEP else mosaic,
        )

    # transforms
    def flip(self, axis="horizontal"):
        """Mirror left-right (``"horizontal"``) or top-bottom (``"vertical"``)."""
        from .image_transforms import flip
        return flip(self, axis)

    def rotate90(self, k=1):
        """Rotate by ``k`` quarter turns counter-clockwise."""
        from .image_transforms import rotate90
        return rotate90(self, k)

    def bin(self, factor, *, method="mean"):
        """Combine ``factor`` x ``factor`` blocks by their mean or sum."""
        from .image_transforms import bin
        return bin(self, factor, method=method)

    def smooth(self, method="gaussian", **kwargs):
        """Filter each channel spatially (gaussian ``sigma``, box or median ``size``)."""
        from .image_transforms import smooth
        return smooth(self, method=method, **kwargs)

    def normalize(self, by="peak", value=1.0, *, roi=None, per_channel=False):
        """Scale so the peak, mean or median (of a region) equals ``value``."""
        from .image_transforms import normalize
        return normalize(self, by=by, value=value, roi=roi, per_channel=per_channel)

    def scale_channels(self, gains):
        """Multiply each channel by a gain (``{"R": 1.2, ...}`` or a sequence)."""
        from .image_transforms import scale_channels
        return scale_channels(self, gains)

    def resize(self, shape, *, order=1):
        """Resample to ``shape`` (rows, columns) over the same field."""
        from .image_transforms import resize
        return resize(self, shape, order=order)

    def interpolate(self, pixel_map, *, order=1, fill=np.nan):
        """Resample onto the coordinates of another pixel map."""
        from .image_transforms import interpolate
        return interpolate(self, pixel_map, order=order, fill=fill)

    def line(self, start, end, *, coords=False, width=1, num=None, order=1, label=None):
        """Values along a line, as a LineCut (see :func:`gsmath.linecut.line`)."""
        from .linecut import line
        return line(self, start, end, coords=coords, width=width, num=num, order=order,
                    label=label)

    def row(self, index, *, label=None):
        """The line cut along one row."""
        from .linecut import row
        return row(self, index, label=label)

    def column(self, index, *, label=None):
        """The line cut down one column."""
        from .linecut import column
        return column(self, index, label=label)

    def apply_matrix(self, matrix, *, channels=("X", "Y", "Z"), unit=None):
        """Mix channels by a matrix, e.g. camera R, G, B to X, Y, Z."""
        from .image_transforms import apply_matrix
        return apply_matrix(self, matrix, channels=channels, unit=unit)

    def find_display(self, *, channel=None, threshold=None):
        """The quadrilateral of a bright display (see :mod:`gsmath.perspective`)."""
        from .perspective import find_display
        return find_display(self, channel=channel, threshold=threshold)

    def rectify(self, quad, shape=None, *, size=None, order=1, fill=np.nan):
        """Resample so ``quad`` fills a rectangle: keystone correction."""
        from .perspective import rectify
        return rectify(self, quad, shape, size=size, order=order, fill=fill)

    def demosaic(self, method="bilinear"):
        """R, G, B from a raw colour-filter mosaic (see :mod:`gsmath.mosaic`)."""
        from .mosaic import demosaic
        return demosaic(self, method)

    # analysis
    def stats(self, *, roi=None):
        """Per-channel statistics over the image or a region (cached)."""
        from .analysis.image import ImageStatistics
        return self._get_analysis(ImageStatistics, roi=roi)

    def color(self, *, white="D65", observer="2deg"):
        """Per-pixel colour of an X, Y, Z image (cached); see
        :class:`gsmath.ImageColor`. ``white``: an illuminant name (Y = 100)
        or an XYZ triple on the image's scale."""
        from .analysis.image_color import ImageColor
        if not isinstance(white, str):
            white = tuple(float(v) for v in white)
        return self._get_analysis(ImageColor, white=white, observer=observer)

    # summary / comparison
    def profile(self):
        """The default profile for images: every channel's statistics
        (``"R.mean"``, ``"G.std"``, ...; for one channel, ``"mean"``, ...)."""
        from .profile import summary_profile
        return summary_profile(self.stats(), name="image", of=lambda image: image.stats())

    def plot(self, ax=None, *, channel=None, coords=True, cmap=None, vmin=None, vmax=None,
             colorbar=True, srgb=False, title=None):
        """Draw the image (axes in pixel-map units); see :mod:`gsmath.image_plotting`."""
        from .image_plotting import plot
        return plot(self, ax, channel=channel, coords=coords, cmap=cmap, vmin=vmin,
                    vmax=vmax, colorbar=colorbar, srgb=srgb, title=title)

    def uniformity(self, area=None, *, points=None, spot=0.04):
        """Luminance and colour uniformity over sample spots within the
        active ``area`` (None, a pixel Rectangle or a Quad); ``points`` are
        ``(u, v)`` fractions of it (default a 3 x 3 grid), ``spot`` the spot
        diameter as a fraction of its width. See :mod:`gsmath.analysis.uniformity`."""
        from .analysis.uniformity import Uniformity, grid
        return Uniformity(self, area, tuple(grid(3, 3) if points is None else points), spot)

    def edge_mtf(self, roi=None, *, oversampling=4):
        """MTF across a slanted edge (ISO 12233 e-SFR); see :mod:`gsmath.analysis.mtf`."""
        from .analysis.mtf import edge_mtf
        return edge_mtf(self, roi, oversampling=oversampling)

    def histogram(self, bins=256, *, range=None, roi=None):
        """Per-channel histogram over the image or a region."""
        from .analysis.image import histogram
        return histogram(self, bins=bins, range=range, roi=roi)

    def pixels(self, roi=None) -> np.ndarray:
        """The values of the pixels in ``roi`` (all by default): shape
        ``(n,)`` for one channel, ``(n, channels)`` for several."""
        if roi is None:
            return self.values.reshape(-1, *self.values.shape[2:])
        window, mask = roi.submask(self)  # only the region's box of the frame
        return self.values[window][mask]

    def _align(self, other):
        """The same grid, channels, pixel map and colour-filter pattern; a
        single-channel image combines with every channel of a multi-channel
        one (a gain map applied to R, G and B alike)."""
        if not isinstance(other, Image):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"Image shapes must match: {self.shape} vs {other.shape}.")
        if self.values.ndim == 3 and other.values.ndim == 3 and self.channels != other.channels:
            raise ValueError(f"Image channels must match: {self.channels} vs {other.channels}.")
        if (self.pixel_map is not None and other.pixel_map is not None
                and self.pixel_map != other.pixel_map):
            raise ValueError("Image pixel maps must match.")
        if (self.mosaic is not None and other.mosaic is not None
                and self.mosaic.pattern != other.mosaic.pattern):
            raise ValueError("Image colour-filter patterns must match.")
        left, right = _float(self.values), _float(other.values)
        if left.ndim == 3 and right.ndim == 2:
            right = right[:, :, None]
        elif left.ndim == 2 and right.ndim == 3:
            left = left[:, :, None]
        wider = self if self.values.ndim >= other.values.ndim else other
        return left, right, {
            "channels": wider.channels,
            "pixel_map": self.pixel_map if self.pixel_map is not None else other.pixel_map,
            "mosaic": self.mosaic if self.mosaic is not None else other.mosaic}


def _channels(channels, values):
    if channels is None:
        return None if values.ndim == 2 else tuple(str(i) for i in range(values.shape[2]))
    channels = (channels,) if isinstance(channels, str) else tuple(str(c) for c in channels)
    expected = 1 if values.ndim == 2 else values.shape[2]
    if len(channels) != expected:
        raise ValueError(f"{len(channels)} channel names for {expected} channel(s)")
    if len(set(channels)) != len(channels):
        raise ValueError("channel names must be unique")
    return channels
