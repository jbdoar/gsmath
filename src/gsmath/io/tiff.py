"""TIFF images, through tifffile (``pip install gsmath[images]``).

TIFF keeps any integer or float data exactly, with any number of channels.
gsmath writes the image's label, value unit, channels, metadata and a
separable pixel map into the TIFF's description, and restores them when it
reads the file back; a per-pixel map is kept only by the native format. A TIFF
from elsewhere is read as its data with R, G, B channel names if it is RGB.
"""

import json
from pathlib import Path

import numpy as np

from .._utils import to_builtin
from ..image import Image
from ..pixelmap import PixelMap
from .raster import default_channels, require, source

_DESCRIPTION_KEY = "gsmath"


def read_tiff(path, *, page=None, label=None, value_unit=None, channels=None, metadata=None):
    """Read one page of a TIFF file (``page``: required if it has several)."""
    tifffile = require("tifffile", "Reading TIFF images")
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        if page is None:
            if len(tif.pages) > 1:
                raise ValueError(f"{path.name} has {len(tif.pages)} pages; pass page=")
            page = 0
        tiff_page = tif.pages[page]
        values = tiff_page.asarray()
        axes = tiff_page.axes
        description = tiff_page.description
        rgb = tiff_page.photometric == tifffile.PHOTOMETRIC.RGB
    if "S" in axes and axes.index("S") == 0 and values.ndim == 3:  # planar samples
        values = np.moveaxis(values, 0, -1)
    if values.ndim not in (2, 3):
        raise ValueError(f"{path.name}: can't read a TIFF page of axes {axes!r} as an image")

    saved = _saved(description)
    if saved is not None:
        pixel_map = saved.get("pixel_map")
        stored_metadata = dict(saved.get("metadata", {}))
        mosaic = saved.get("mosaic", stored_metadata.pop("mosaic", None))
        return Image(
            values,
            label=saved.get("label", path.stem) if label is None else label,
            value_unit=saved.get("value_unit") if value_unit is None else value_unit,
            channels=saved.get("channels") if channels is None else channels,
            pixel_map=None if pixel_map is None else PixelMap(**pixel_map),
            metadata={**stored_metadata, **(metadata or {})},
            mosaic=mosaic,
        )
    if channels is None:
        channels = default_channels(values) if rgb or values.ndim == 2 else None
    return Image(values, label=path.stem if label is None else label, value_unit=value_unit,
                 channels=channels,
                 metadata={"source": source(path, "tiff", values), **(metadata or {})})


def write_tiff(image, path, *, compression=None):
    """Write an image as TIFF, exactly (``compression``: e.g. "zlib", "lzw")."""
    tifffile = require("tifffile", "Writing TIFF images")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.channels in (("R", "G", "B"), ("R", "G", "B", "A"))
    tifffile.imwrite(path, image.values, photometric="rgb" if rgb else "minisblack",
                     planarconfig="contig" if image.values.ndim == 3 else None,
                     compression=compression, description=_description(image), metadata=None)
    return path


def _description(image):
    from .. import units

    pixel_map = image.pixel_map
    saved = {
        "label": image.label,
        "value_unit": units.format_unit(image.value_unit),
        "channels": None if image.channels is None else list(image.channels),
        "pixel_map": None if pixel_map is None or not pixel_map.separable else {
            "x": pixel_map.x.tolist(), "y": pixel_map.y.tolist(),
            "unit": units.format_unit(pixel_map.unit)},
        "metadata": to_builtin(image.metadata),
    }
    if image.mosaic is not None:
        saved["mosaic"] = image.mosaic.to_dict()
    try:
        return json.dumps({_DESCRIPTION_KEY: saved}, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("TIFF image metadata must be JSON-compatible and finite.") from exc


def _saved(description):
    try:
        found = json.loads(description)
    except (TypeError, ValueError):
        return None
    return found.get(_DESCRIPTION_KEY) if isinstance(found, dict) else None
