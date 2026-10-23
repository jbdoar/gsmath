"""PNG and JPEG images, through imagecodecs (``pip install gsmath[images]``).

These are display formats: their values are encoded (usually sRGB, gamma
applied), not linear measurements, so images read from them have no value
unit unless one is given. PNG keeps 8- and 16-bit data exactly; JPEG is 8-bit
and lossy. Writing requires integer data already in range -- ``uint8``, or
``uint16`` for PNG -- rather than guessing a scaling.
"""

import importlib
from pathlib import Path

import numpy as np

from ..image import Image

_WRITABLE = {"png": (np.uint8, np.uint16), "jpeg": (np.uint8,)}


def require(module, purpose):
    """Import an optional dependency, saying how to install it if missing."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"{purpose} needs {module}: pip install 'gsmath[images]'") from exc


def default_channels(values):
    """Channel names for an array read from a file: R, G, B (and A)."""
    if values.ndim == 3:
        return {3: ("R", "G", "B"), 4: ("R", "G", "B", "A"), 2: ("L", "A")}.get(values.shape[2])
    return None


def source(path, format, values):
    # Provenance travels into native and TIFF output. Keep the useful filename
    # without disclosing a user's account name, network share, or directory
    # layout when that output is shared.
    return {"path": Path(path).name, "format": format, "dtype": values.dtype.name}


def read_raster(path, *, format, label=None, value_unit=None, channels=None, metadata=None):
    """Read a PNG or JPEG file as an Image."""
    imagecodecs = require("imagecodecs", f"Reading {format.upper()} images")
    path = Path(path)
    decode = imagecodecs.png_decode if format == "png" else imagecodecs.jpeg8_decode
    values = decode(path.read_bytes())
    return Image(values, label=path.stem if label is None else label, value_unit=value_unit,
                 channels=default_channels(values) if channels is None else channels,
                 metadata={"source": source(path, format, values), **(metadata or {})})


def write_raster(image, path, *, format, quality=95):
    """Write an image's integer values as PNG or JPEG (``quality`` 1-100)."""
    imagecodecs = require("imagecodecs", f"Writing {format.upper()} images")
    values = image.values
    allowed = _WRITABLE[format]
    if values.dtype not in allowed:
        names = " or ".join(np.dtype(t).name for t in allowed)
        raise TypeError(f"{format.upper()} stores {names} values; {image.label!r} is "
                        f"{values.dtype.name} -- convert it first (image.copy(values=...)), "
                        "or write TIFF or native .gsmath to keep it as it is")
    if image.n_channels not in (1, 3, 4) or (format == "jpeg" and image.n_channels == 4):
        raise ValueError(f"can't write {image.n_channels} channels as {format.upper()}")
    if format == "png":
        data = imagecodecs.png_encode(values)
    else:
        data = imagecodecs.jpeg8_encode(values, level=quality)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
