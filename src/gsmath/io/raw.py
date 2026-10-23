"""Camera raw files (DNG, CR2/CR3, NEF, ARW, ...), through LibRaw via rawpy
(``pip install gsmath[images]``).

A raw file reads as the sensor's mosaic, untouched: one channel of counts on
the visible area, with ``image.mosaic`` giving the colour-filter pattern,
per-site black level and white level (see :mod:`gsmath.mosaic`),
and ``metadata["camera"]`` what the file says about the camera and the shot.
No black level, white balance, colour matrix or tone curve is applied:
demosaic and calibrate it explicitly. The shutter time, when the file has
one, is ``metadata["camera"]["exposure_s"]`` (seconds) -- for a
:class:`~gsmath.Measurement`, ``exposure=Q_(seconds, "s")``.
"""

from pathlib import Path

import numpy as np

from ..image import Image
from .raster import require, source

SUFFIXES = (".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf",
            ".orf", ".rw2", ".pef", ".srw", ".3fr", ".fff", ".iiq", ".mos", ".erf", ".kdc",
            ".mrw", ".x3f")


def read_raw(path, *, label=None, value_unit="count", metadata=None):
    """Read a camera raw file's mosaic as a single-channel Image."""
    rawpy = require("rawpy", "Reading camera raw files")
    path = Path(path)
    with rawpy.imread(str(path)) as raw:
        if raw.raw_type != rawpy.RawType.Flat:
            raise ValueError(f"{path.name}: only colour-filter (Bayer) raw data is supported")
        values = raw.raw_image_visible.copy()
        colours = raw.raw_colors_visible
        description = raw.color_desc.decode("ascii")
        if raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
            raise ValueError(f"{path.name}: only 2x2 colour-filter patterns are supported")
        sites = [int(c) for c in colours[:2, :2].ravel()]
        black = [float(b) for b in raw.black_level_per_channel]
        from ..mosaic import Mosaic

        mosaic = Mosaic("".join(description[c] for c in sites), [black[c] for c in sites],
                        float(raw.white_level))
        other = raw.other
        camera = {
            "camera_whitebalance": [float(v) for v in raw.camera_whitebalance],
            "daylight_whitebalance": [float(v) for v in raw.daylight_whitebalance],
            "rgb_xyz_matrix": np.asarray(raw.rgb_xyz_matrix, dtype=float).tolist(),
            "aperture": _positive(other.aperture),
            "focal_length_mm": _positive(other.focal_length),
            "iso": _positive(other.iso_speed),
            "exposure_s": _positive(other.shutter_speed),
            "timestamp": None if other.timestamp.year <= 1970 else other.timestamp.isoformat(),
        }
    return Image(values, label=path.stem if label is None else label, value_unit=value_unit,
                 mosaic=mosaic,
                 metadata={"source": source(path, "raw", values), "camera": camera,
                           **(metadata or {})})


def _positive(value):
    value = float(value)
    return value if value > 0 else None
