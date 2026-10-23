"""Colour-filter mosaics: raw sensor data, before it becomes R, G, B.

A raw frame is a single-channel :class:`~gsmath.Image` with a
:class:`Mosaic` (``image.mosaic``): which colour each pixel saw -- the 2x2
Bayer pattern from the top-left pixel in reading order (``"RGGB"``,
``"BGGR"``, ``"GRBG"``, ``"GBRG"``) -- and, as read from the file, the black
level (one per pattern position) and white level. The image keeps the
pattern right for its top-left pixel when it is sliced, flipped or rotated,
and the mosaic stays through dark subtraction, flat fielding and other
per-pixel steps, so it can be demosaiced at any point in a calibration::

    camera = Calibration("camera", [
        SubtractDark(),      # or BlackLevel() without a dark frame
        FlatField(flat),     # a raw flat, same mosaic
        Demosaic(),
        PerExposure(),
    ])

Both demosaicing methods are linear in the data, so they don't disturb a
radiometric calibration:

- ``"bilinear"``: full resolution; each missing colour is the average of its
  nearest neighbours of that colour (measured values kept as they are).
- ``"superpixel"``: half resolution; each 2x2 block becomes one pixel (its R,
  the mean of its two Gs, its B) with no interpolation at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage

from .calibration import Step

CFA_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")
METHODS = ("bilinear", "superpixel")

_KERNELS = {
    "R": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float),
    "G": np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=float),
}
_KERNELS["B"] = _KERNELS["R"]
# The same kernels as sums of outer products (rows, columns): R and B's is
# [1 2 1] x [1 2 1]; G's plus-shape is [0 1 0] x [1 2 1] + [1 2 1] x [0 1 0].
_ONE, _TRIANGLE = np.array([0.0, 1.0, 0.0]), np.array([1.0, 2.0, 1.0])
_KERNEL_TERMS = {"R": [(_TRIANGLE, _TRIANGLE)], "B": [(_TRIANGLE, _TRIANGLE)],
                 "G": [(_ONE, _TRIANGLE), (_TRIANGLE, _ONE)]}


@dataclass(frozen=True)
class Mosaic:
    """A raw image's colour-filter layout and sensor levels."""
    pattern: str
    """The 2x2 pattern from the top-left pixel, in reading order."""
    black_level: tuple | None = None
    """The black level at each pattern position (same order), if known."""
    white_level: float | None = None
    """The sensor's clipping level, if known."""

    def __post_init__(self):
        if self.pattern not in CFA_PATTERNS:
            raise ValueError(f"unsupported colour-filter pattern {self.pattern!r}; "
                             f"expected one of {CFA_PATTERNS}")
        if self.black_level is not None:
            black = np.broadcast_to(np.asarray(self.black_level, dtype=float), (4,))
            object.__setattr__(self, "black_level", tuple(float(b) for b in black))
        if self.white_level is not None:
            object.__setattr__(self, "white_level", float(self.white_level))

    @classmethod
    def coerce(cls, value):
        """A Mosaic from a Mosaic, a pattern string, or a stored dict."""
        if value is None or isinstance(value, Mosaic):
            return value
        if isinstance(value, str):
            return cls(value)
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, data):
        return cls(data["cfa"], data.get("black_level"), data.get("white_level"))

    def to_dict(self) -> dict:
        """As stored: ``{"cfa": ..., "black_level": [...], "white_level": ...}``."""
        stored = {"cfa": self.pattern}
        if self.black_level is not None:
            stored["black_level"] = list(self.black_level)
        if self.white_level is not None:
            stored["white_level"] = self.white_level
        return stored

    def masks(self, shape) -> dict[str, np.ndarray]:
        """For each of R, G, B: where in an image of ``shape`` it was measured."""
        return site_masks(self.pattern, shape)

    def remapped(self, old_position, what="rearranging") -> Mosaic:
        """The mosaic of an image rearranged so its new pixel ``(i, j)`` was
        old pixel ``old_position(i, j)``; raises if that doesn't leave a
        Bayer mosaic (e.g. slicing with step 2)."""
        order = []
        for i in (0, 1):
            for j in (0, 1):
                r, c = old_position(i, j)
                order.append((r % 2) * 2 + c % 2)
        pattern = "".join(self.pattern[k] for k in order)
        if pattern not in CFA_PATTERNS:
            raise ValueError(f"{what} doesn't leave a colour-filter mosaic; demosaic first")
        black = None if self.black_level is None else tuple(self.black_level[k] for k in order)
        return Mosaic(pattern, black, self.white_level)

    def sliced(self, rows: slice, cols: slice, shape) -> Mosaic:
        """The mosaic of ``image[rows, cols]``."""
        (r0, _, rs), (c0, _, cs) = rows.indices(shape[0]), cols.indices(shape[1])
        return self.remapped(lambda i, j: (r0 + i * rs, c0 + j * cs), "slicing with that step")

    def black_frame(self, shape) -> np.ndarray:
        """The black level at every pixel of an image of ``shape``."""
        if self.black_level is None:
            raise ValueError("the mosaic has no recorded black level")
        black = np.asarray(self.black_level).reshape(2, 2)
        return np.tile(black, (shape[0] // 2 + 1, shape[1] // 2 + 1))[:shape[0], :shape[1]]


def site_masks(pattern: str, shape) -> dict[str, np.ndarray]:
    """For each of R, G, B: where in an image of `shape` that colour was measured."""
    masks = {}
    for colour in "RGB":
        mask = np.zeros(shape, dtype=bool)
        for i, j in _sites(pattern, colour):
            mask[i::2, j::2] = True
        masks[colour] = mask
    return masks


def _sites(pattern, colour):
    """The (row, column) parities, within the 2 x 2 tile, where `colour` sits."""
    return [divmod(k, 2) for k, c in enumerate(pattern) if c == colour]


def _bilinear_weights(pattern, colour, shape):
    """How much measured data of `colour` the bilinear kernel sees around
    each pixel -- the normalised convolution's denominator. The mask is a
    sum of outer products of parity vectors and the kernel one or two outer
    products, so this needs only 1-D convolutions: no pass over the frame
    until the final products."""
    rows, cols = shape
    weights = np.zeros(shape)
    for i, j in _sites(pattern, colour):
        on_rows = (np.arange(rows) % 2 == i).astype(float)
        on_cols = (np.arange(cols) % 2 == j).astype(float)
        for a, b in _KERNEL_TERMS[colour]:
            weights += np.outer(np.convolve(on_rows, a, mode="same"),
                                np.convolve(on_cols, b, mode="same"))
    return weights


def mosaic_of(data) -> Mosaic | None:
    """``data``'s mosaic, or None (for an image without one, or a spectrum)."""
    return getattr(data, "mosaic", None)


def require_mosaic(image) -> Mosaic:
    if mosaic_of(image) is None:
        raise ValueError(f"{image.label!r} isn't a colour-filter mosaic (it has no mosaic)")
    return image.mosaic


def require_demosaiced(image, what: str):
    """Raise if `image` is a raw mosaic, which `what` would mix the colours of."""
    if mosaic_of(image) is not None:
        raise ValueError(f"{what} would mix the colours of {image.label!r}'s raw mosaic; "
                         "demosaic it first")


def demosaic(image, method: str = "bilinear"):
    """The R, G, B image from a raw mosaic image (see the module notes)."""
    mosaic = require_mosaic(image)
    if method not in METHODS:
        raise ValueError(f"unknown demosaic method {method!r}; expected one of {METHODS}")
    pattern = mosaic.pattern
    values = np.asarray(image.values, dtype=float)
    metadata = {**image.metadata, "demosaic": {"method": method, "cfa": pattern}}

    if method == "superpixel":
        rows, cols = image.height // 2 * 2, image.width // 2 * 2
        blocks = values[:rows, :cols].reshape(rows // 2, 2, cols // 2, 2).transpose(0, 2, 1, 3)
        blocks = blocks.reshape(rows // 2, cols // 2, 4)
        rgb = np.stack([blocks[:, :, [i for i, c in enumerate(pattern) if c == colour]].mean(axis=2)
                        for colour in "RGB"], axis=-1)
        pixel_map = None if image.pixel_map is None else image.pixel_map.binned(2)
        return image.copy(values=rgb, channels=("R", "G", "B"), pixel_map=pixel_map,
                          mosaic=None, metadata=metadata)

    rgb = np.empty((*image.shape, 3))
    for k, (colour, mask) in enumerate(mosaic.masks(image.shape).items()):
        # Normalised convolution: weighted mean of the measured neighbours,
        # which also handles the edges (no data beyond them, no weight).
        sums = scipy.ndimage.convolve(np.where(mask, values, 0.0), _KERNELS[colour],
                                      mode="constant")
        plane = rgb[:, :, k]
        np.divide(sums, _bilinear_weights(pattern, colour, image.shape), out=plane)
        plane[mask] = values[mask]  # measured sites keep their value
    return image.copy(values=rgb, channels=("R", "G", "B"), mosaic=None, metadata=metadata)


def subtract_black_level(image):
    """The mosaic minus the black level recorded for each pattern site."""
    mosaic = require_mosaic(image)
    if mosaic.black_level is None:
        raise ValueError(f"{image.label!r} has no recorded black level")
    return image.copy(values=np.asarray(image.values, dtype=float) - mosaic.black_frame(image.shape))


class Demosaic(Step):
    """Turn a raw mosaic into R, G, B (``method``: bilinear or superpixel)."""

    def __init__(self, method: str = "bilinear", *, optional: bool = False):
        if method not in METHODS:
            raise ValueError(f"unknown demosaic method {method!r}; expected one of {METHODS}")
        super().__init__(optional=optional)
        self.method = method

    def apply(self, data, measurement):
        return demosaic(data, self.method)

    def arguments(self):
        return {"method": self.method}


class BlackLevel(Step):
    """Subtract the black level the raw file records, per mosaic site: for
    when no dark frame was taken. A dark frame (SubtractDark) also removes
    the dark signal and fixed pattern, so prefer it when there is one."""

    def apply(self, data, measurement):
        return subtract_black_level(data)


def site_means(image, roi=None):
    """The mean of each colour's sites -- R, G (both G sites), B -- of a raw
    mosaic, over the image or a region: a uniform patch's camera signal
    without demosaicing. NaN pixels are skipped."""
    mosaic = require_mosaic(image)
    values = np.asarray(image.values, dtype=float)
    region = np.ones(image.shape, dtype=bool) if roi is None else roi.mask(image)
    means = []
    for colour, mask in mosaic.masks(image.shape).items():
        chosen = values[mask & region]
        chosen = chosen[np.isfinite(chosen)]
        if chosen.size == 0:
            raise ValueError(f"the region has no {colour} sites")
        means.append(chosen.mean())
    return np.array(means)
