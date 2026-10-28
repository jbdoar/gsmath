"""File formats for ``Spectrum``/``Image`` ``read``/``write`` and ``gsmath.read``.

:data:`FORMATS` lists each format once: its name, the names and suffixes
that select it, and its readers and writers for each kind of data
(``"spectrum"``, ``"image"``). Native ``.gsmath`` files hold any record (see
:mod:`gsmath.io.native`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .from_csv import from_csv
from .from_lighttouch import from_lighttouch
from .native import (read_image as read_native_image, read_native, read_record,
                     write_image as write_native_image, write_native)
from .raster import read_raster, write_raster
from .raw import SUFFIXES as RAW_SUFFIXES, read_raw
from .tiff import read_tiff, write_tiff
from .to_csv import to_csv
from .to_lighttouch import to_lighttouch


@dataclass(frozen=True)
class FileFormat:
    """One file format and what it reads and writes."""
    name: str
    aliases: tuple
    suffixes: tuple
    readers: dict = field(default_factory=dict)
    """kind -> ``reader(path, **options)``"""
    writers: dict = field(default_factory=dict)
    """kind -> ``writer(obj, path, **options)``"""
    unwritable: str = ""
    """Why it can't be written, if it can't."""

    def names(self):
        return (self.name, *self.aliases)


FORMATS: list[FileFormat] = [
    FileFormat("gsmath", ("native",), (".gsmath",),
               {"spectrum": read_native, "image": read_native_image},
               {"spectrum": write_native, "image": write_native_image}),
    FileFormat("csv", (), (".csv",), {"spectrum": from_csv}, {"spectrum": to_csv}),
    FileFormat("lighttouch", ("light-touch",), (".cal", ".std"),
               {"spectrum": from_lighttouch}, {"spectrum": to_lighttouch}),
    FileFormat("png", (), (".png",),
               {"image": lambda path, **kw: read_raster(path, format="png", **kw)},
               {"image": lambda image, path, **kw: write_raster(image, path, format="png", **kw)}),
    FileFormat("jpeg", ("jpg",), (".jpg", ".jpeg"),
               {"image": lambda path, **kw: read_raster(path, format="jpeg", **kw)},
               {"image": lambda image, path, **kw: write_raster(image, path, format="jpeg", **kw)}),
    FileFormat("tiff", ("tif",), (".tif", ".tiff"), {"image": read_tiff}, {"image": write_tiff}),
    FileFormat("raw", (), RAW_SUFFIXES, {"image": read_raw},
               unwritable="camera raw files can be read but not written; "
                          "write TIFF or native .gsmath instead"),
]


def _kind_of(obj):
    from ..image import Image
    return "image" if isinstance(obj, Image) else "spectrum"


def resolve(path, format=None, kind=None) -> FileFormat:
    """The format named ``format``, or the one ``path``'s suffix implies,
    among those that handle ``kind`` (any kind if None)."""
    candidates = [f for f in FORMATS
                  if kind is None or kind in f.readers or kind in f.writers]
    what = f"{kind} format" if kind else "format"
    if format is not None:
        name = str(format).strip().lower()
        for candidate in candidates:
            if name in candidate.names():
                return candidate
        supported = ", ".join(sorted(n for c in candidates for n in c.names()))
        raise ValueError(f"Unknown {what} {format!r}; supported formats: {supported}.")
    suffix = Path(path).suffix.lower()
    for candidate in candidates:
        if suffix in candidate.suffixes:
            return candidate
    raise ValueError(f"Cannot infer {what} from suffix {suffix!r}; specify format explicitly.")


def read(path, *, format=None, kind="spectrum", **kwargs):
    """Read ``kind`` of data through the selected format."""
    selected = resolve(path, format, kind)
    return selected.readers[kind](path, **kwargs)


def write(obj, path, *, format=None, **kwargs):
    """Write a spectrum or image through the selected format."""
    kind = _kind_of(obj)
    selected = resolve(path, format, kind)
    if kind not in selected.writers:
        raise ValueError(selected.unwritable or f"{selected.name} files can't be written")
    return selected.writers[kind](obj, path, **kwargs)


def read_image(path, *, format=None, **kwargs):
    """Read an image through the selected format."""
    return read(path, format=format, kind="image", **kwargs)


def write_image(image, path, *, format=None, **kwargs):
    """Write an image through the selected format."""
    return write(image, path, format=format, **kwargs)


def read_any(path, *, format=None, **kwargs):
    """Read any gsmath file: a native ``.gsmath`` record comes back as what it
    holds (by its ``format`` field); other formats as the one kind of data
    they hold (csv, lighttouch: spectra; png, jpeg, tiff, camera raw: images)."""
    selected = resolve(path, format)
    if selected.name == "gsmath":
        return read_record(path)
    (kind,) = selected.readers
    return selected.readers[kind](path, **kwargs)
