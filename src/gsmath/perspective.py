"""Finding a display in an image, and correcting its keystone.

A camera that isn't square-on to a display sees its rectangular active area
as a general quadrilateral. :func:`find_display` locates that quadrilateral
in an image of the lit display (a full-white frame works best; the corners
found can be reused for every other frame of the same setup), and
:func:`rectify` resamples the image so the display fills a rectangle::

    display = image.find_display()          # a Quad: corners TL, TR, BR, BL
    display.edge_rms                        # px; large if the edges are curved
    flat = image.rectify(display, size=(Q_(344, "mm"), Q_(194, "mm")))
    image.stats(roi=display)                # a Quad is also a region

The correction is a homography -- exact for a flat display seen through a
lens without distortion; correct lens distortion first if there is any (the
edge fit's residual shows it). Resampling interpolates values, which suits
intensive quantities (luminance, radiance, colour); per-pixel totals such as
counts aren't conserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage

from .roi import Polygon


@dataclass(frozen=True)
class Quad(Polygon):
    """Four corners -- top-left, top-right, bottom-right, bottom-left -- in
    pixel coordinates (column, row), or in the pixel map's with ``coords``."""
    edge_rms: float | None = None
    """RMS distance (px) of the detected edge pixels from the fitted straight
    edges: about 0.5 for straight edges (the pixels either side of an edge
    sit about half a pixel from it), more if the edges curve."""

    def __post_init__(self):
        super().__post_init__()
        if len(self.vertices) != 4:
            raise ValueError("a Quad has four corners: top-left, top-right, "
                             "bottom-right, bottom-left")

    @property
    def corners(self) -> np.ndarray:
        """The corners as a (4, 2) array of (x, y)."""
        return np.array(self.vertices, dtype=float)

    @property
    def size(self) -> tuple[float, float]:
        """(width, height): the mean lengths of opposite sides."""
        tl, tr, br, bl = self.corners
        return ((np.hypot(*(tr - tl)) + np.hypot(*(br - bl))) / 2,
                (np.hypot(*(bl - tl)) + np.hypot(*(br - tr))) / 2)


def homography(source, target) -> np.ndarray:
    """The 3x3 projective map taking four ``source`` points to ``target``
    points (each (4, 2) of x, y)."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.shape != (4, 2) or target.shape != (4, 2):
        raise ValueError("a homography needs four (x, y) points each side")
    rows, rhs = [], []
    for (x, y), (u, v) in zip(source, target):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.extend([u, v])
    try:
        h = np.linalg.solve(np.array(rows), np.array(rhs))
    except np.linalg.LinAlgError:
        raise ValueError("three of the points are collinear") from None
    return np.append(h, 1.0).reshape(3, 3)


def project(H, x, y):
    """Apply homography ``H`` to points ``(x, y)`` (arrays of one shape)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w


def find_display(image, *, channel=None, threshold=None) -> Quad:
    """The quadrilateral of a bright display on a darker surround.

    Thresholds the image (``channel``: by default Y, else G, else the mean of
    the channels; ``threshold``: by default Otsu's), keeps the largest bright
    region with its holes filled, and fits a straight line to each edge --
    to the pixels just inside and just outside it, so the line sits on the
    transition -- for sub-pixel corners.
    """
    from .mosaic import require_demosaiced

    require_demosaiced(image, "finding the display")
    values = _luminance(image, channel)
    finite = np.isfinite(values)
    if threshold is None:
        threshold = otsu_threshold(values[finite])
    bright = finite & (values > threshold)
    labels, count = scipy.ndimage.label(bright)
    if count == 0:
        raise ValueError(f"nothing in {image.label!r} is brighter than {threshold:g}")
    largest = 1 + int(np.argmax(scipy.ndimage.sum_labels(bright, labels, range(1, count + 1))))
    region = scipy.ndimage.binary_fill_holes(labels == largest)
    if region.sum() < 100:
        raise ValueError(f"the brightest region of {image.label!r} is too small to be a display")

    inner = region & ~scipy.ndimage.binary_erosion(region)
    outer = ~region & scipy.ndimage.binary_dilation(region)
    rows, cols = np.nonzero(inner | outer)
    points = np.column_stack([cols, rows]).astype(float)

    corners = _extreme_corners(region)
    for _ in range(3):
        lines, residuals = [], []
        for k in range(4):
            line, residual = _fit_edge(points, corners[k], corners[(k + 1) % 4])
            lines.append(line)
            residuals.append(residual)
        corners = np.array([_intersect(lines[k - 1], lines[k]) for k in range(4)])
    edge_rms = float(np.sqrt(np.mean(np.square(np.concatenate(residuals)))))
    return Quad(tuple(map(tuple, corners)), edge_rms=edge_rms)


def rectify(image, quad, shape=None, *, size=None, order=1, fill=np.nan):
    """Resample ``image`` so the quadrilateral ``quad`` fills a rectangle.

    ``shape`` (rows, columns) defaults to the quad's mean side lengths, so
    the display keeps about the resolution it had. The output keeps each
    pixel's source coordinates (interpolated from the image's pixel map),
    unless ``size`` -- the display's (width, height), numbers or quantities
    -- gives it a pixel map in display units, zero at the centre.
    """
    from . import units
    from .image_transforms import sample
    from .mosaic import require_demosaiced
    from .pixelmap import PixelMap

    require_demosaiced(image, "rectifying")
    if not isinstance(quad, Quad):
        raise TypeError("rectify needs a Quad (e.g. from find_display)")
    corners = quad.corners
    if quad.coords:
        corners = np.column_stack(image.pixel_map.pixel_of(corners[:, 0], corners[:, 1]))
        quad = Quad(tuple(map(tuple, corners)))
    if shape is None:
        width, height = quad.size
        shape = (max(int(round(height)), 1), max(int(round(width)), 1))
    rows, cols = shape
    frame = [(-0.5, -0.5), (cols - 0.5, -0.5), (cols - 0.5, rows - 0.5), (-0.5, rows - 0.5)]
    H = homography(frame, corners)
    out_cols, out_rows = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    src_cols, src_rows = project(H, out_cols, out_rows)
    values = sample(image, src_cols, src_rows, order=order)
    outside = ((src_cols < -0.5) | (src_cols > image.width - 0.5)
               | (src_rows < -0.5) | (src_rows > image.height - 0.5))
    values[outside] = fill

    if size is not None:
        (width, height), unit = _size(size)
        pixel_map = PixelMap.linear(shape, (width / cols, height / rows), unit=unit)
    elif image.pixel_map is not None:
        pixel_map = PixelMap(*image.pixel_map.at(src_cols, src_rows), image.pixel_map.unit)
    else:
        pixel_map = None
    metadata = {**image.metadata, "rectified": {
        "corners": corners.tolist(), "shape": [rows, cols],
        **({"size": [width, height], "unit": units.format_unit(unit)} if size is not None else {}),
    }}
    return image.copy(values=values, pixel_map=pixel_map, metadata=metadata,
                      label=f"{image.label} [rectified]")


def otsu_threshold(values, bins=256) -> float:
    """The threshold best separating `values` into two classes (Otsu)."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0 or values.min() == values.max():
        raise ValueError("can't threshold values that are all the same")
    counts, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(counts)
    total = weight[-1]
    mean = np.cumsum(counts * centres)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mean[-1] * weight / total - mean) ** 2 / (weight * (total - weight))
    return float(edges[1:][np.nanargmax(between[:-1])])


def _luminance(image, channel):
    if channel is not None:
        return np.asarray(image.channel(channel).values, dtype=float)
    if image.values.ndim == 2:
        return np.asarray(image.values, dtype=float)
    for name in ("Y", "G"):
        if name in image.channels:
            return np.asarray(image.channel(name).values, dtype=float)
    return np.nanmean(np.asarray(image.values, dtype=float), axis=2)


def _extreme_corners(region):
    """Rough corners: the region's extreme pixels along the diagonals."""
    rows, cols = np.nonzero(region)
    s, d = cols + rows, cols - rows
    picks = [np.argmin(s), np.argmax(d), np.argmax(s), np.argmin(d)]  # TL, TR, BR, BL
    return np.array([[cols[i], rows[i]] for i in picks], dtype=float)


def _fit_edge(points, start, end):
    """A line (point, unit direction) through the boundary points near the
    middle 80% of the segment start-end, by total least squares."""
    direction = end - start
    length = np.hypot(*direction)
    direction = direction / length
    normal = np.array([-direction[1], direction[0]])
    offset = points - start
    along = offset @ direction / length
    across = np.abs(offset @ normal)
    near = (along > 0.1) & (along < 0.9) & (across < 2.0 + 0.02 * length)
    if near.sum() < 4:
        raise ValueError("couldn't find a straight display edge")
    chosen = points[near]
    centre = chosen.mean(axis=0)
    _, _, vt = np.linalg.svd(chosen - centre)
    fitted = vt[0]
    residual = (chosen - centre) @ np.array([-fitted[1], fitted[0]])
    return (centre, fitted), residual


def _intersect(a, b):
    (p, u), (q, v) = a, b
    t = np.linalg.solve(np.column_stack([u, -v]), q - p)[0]
    return p + t * u


def _size(size):
    """((width, height), unit) from numbers or quantities."""
    from . import units

    width, height = size
    if units.is_quantity(width) != units.is_quantity(height):
        raise ValueError("give the display size as two numbers or two quantities")
    if not units.is_quantity(width):
        return (float(width), float(height)), None
    return (float(width.magnitude), float(units.convert_quantity(height, width.units))), width.units
