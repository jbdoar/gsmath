"""Versioned, lossless native serialization: the ``.gsmath`` records.

A record is JSON: ``{"format": ..., "version": ..., <section>: {...}}``.
Records holding images are written as a zip container instead -- the same
JSON record as ``record.json``, with each array beside it as an ``.npy`` file
rather than base64 text -- so a frame of millions of pixels stays compact and
quick to load. Readers accept either.

Two registries say what can be stored:

- :data:`CODECS` -- how each kind of *value* is stored (a spectrum, an image,
  a quantity, an array, ...), as ``{"kind": ..., ...}``. Step arguments and
  measurement data go through them. :func:`encode_value` and
  :func:`decode_value` use them.
- :data:`RECORDS` -- each kind of *file*: its format name and version, its
  section, how its object is stored, and whether it needs a container.
  :func:`write_record` and :func:`read_record` use them.

Adding something savable is registering a codec or a record.
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import json
import numbers
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import units
from .._utils import to_builtin
from ..image import Image
from ..pixelmap import PixelMap
from ..spectrum import Spectrum


# ---- arrays -------------------------------------------------------------

def _unit_text(unit):
    """Serialize an optional Pint unit using its abbreviated expression."""
    return None if unit is None else f"{unit:~}"


_RECORD_MEMBER = "record.json"
_ARRAY_FILES = contextvars.ContextVar("gsmath_native_array_files", default=None)
"""While a container record is being built: arrays to store as files."""
_LARGEST_ARRAY = contextvars.ContextVar("gsmath_native_largest_array", default=None)
"""While a JSON record is being built: the largest array's size so far (bytes)."""

LARGE_ARRAY_BYTES = 1 << 20
"""A record holding an array larger than this (1 MiB) is written as a
container, whatever it holds: arrays in JSON are base64 text, a third larger
and read twice over. Small records -- spectra, typical calibrations -- stay
JSON you can read."""


@contextlib.contextmanager
def _arrays_as_files():
    """Collect the arrays of payloads built inside as container files."""
    files = {}
    token = _ARRAY_FILES.set(files)
    try:
        yield files
    finally:
        _ARRAY_FILES.reset(token)


def _array_payload(array):
    """Serialize values alongside dtype information needed for reconstruction."""
    array = np.ascontiguousarray(array)
    if array.dtype.hasobject:
        raise TypeError("Native spectrum arrays cannot use object dtype.")
    files = _ARRAY_FILES.get()
    if files is not None:
        name = f"arrays/{len(files)}.npy"
        files[name] = array
        return {"dtype": array.dtype.str, "shape": array.shape, "file": name}
    largest = _LARGEST_ARRAY.get()
    if largest is not None:
        largest[0] = max(largest[0], array.nbytes)
    return {
        "dtype": array.dtype.str,
        "shape": array.shape,
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _decode_array(payload):
    """Reconstruct an array from its validated dtype, shape, and bytes."""
    dtype = np.dtype(payload["dtype"])
    if dtype.hasobject:
        raise ValueError("Native spectrum arrays cannot use object dtype.")
    shape = tuple(payload["shape"])
    if "array" in payload:  # loaded from a container file
        array = payload["array"]
        if array.dtype != dtype or array.shape != shape:
            raise ValueError("Native array file does not match its dtype and shape.")
        return array
    raw = base64.b64decode(payload["data"], validate=True)
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected:
        raise ValueError("Native spectrum array byte count does not match its shape.")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


# ---- data objects ------------------------------------------------------

def _spectrum_payload(spectrum):
    return {
        "wavelength": _array_payload(spectrum.wavelength),
        "values": _array_payload(spectrum.values),
        "label": spectrum.label,
        "wavelength_unit": _unit_text(spectrum.wavelength_unit),
        "value_unit": _unit_text(spectrum.value_unit),
        "metadata": to_builtin(spectrum.metadata),
    }


def _spectrum_from_payload(data):
    return Spectrum(
        wavelength=_decode_array(data["wavelength"]),
        values=_decode_array(data["values"]),
        label=data["label"],
        wavelength_unit=data["wavelength_unit"],
        value_unit=data["value_unit"],
        metadata=data["metadata"],
    )


def _image_payload(image):
    pixel_map = image.pixel_map
    payload = {
        "values": _array_payload(image.values),
        "label": image.label,
        "value_unit": _unit_text(image.value_unit),
        "channels": None if image.channels is None else list(image.channels),
        "pixel_map": None if pixel_map is None else {
            "x": _array_payload(pixel_map.x),
            "y": _array_payload(pixel_map.y),
            "unit": _unit_text(pixel_map.unit),
        },
        "metadata": to_builtin(image.metadata),
    }
    if image.mosaic is not None:
        payload["mosaic"] = image.mosaic.to_dict()
    return payload


def _image_from_payload(data):
    pixel_map = data["pixel_map"]
    metadata = dict(data["metadata"])
    mosaic = data.get("mosaic", metadata.pop("mosaic", None))  # older files: in metadata
    return Image(
        _decode_array(data["values"]),
        label=data["label"],
        value_unit=data["value_unit"],
        channels=data["channels"],
        pixel_map=None if pixel_map is None else PixelMap(
            _decode_array(pixel_map["x"]), _decode_array(pixel_map["y"]), pixel_map["unit"]),
        metadata=metadata,
        mosaic=mosaic,
    )


def _wavelength_calibration_payload(value):
    return {"coefficients": _array_payload(value.coefficients),
            "pixel_domain": list(value.pixel_domain),
            "fit_diagnostics": value.fit_diagnostics,
            "matches": [asdict(m) for m in value.matches],
            "peaks": [asdict(p) for p in value.peaks]}


def _wavelength_calibration_from_payload(payload):
    from ..wavelength import CalibrationPeak, LineMatch, WavelengthCalibration

    coefficients = _decode_array(payload["coefficients"])
    coefficients.setflags(write=False)
    return WavelengthCalibration(coefficients,
                                 tuple(LineMatch(**m) for m in payload["matches"]),
                                 tuple(CalibrationPeak(**p) for p in payload["peaks"]),
                                 tuple(payload["pixel_domain"]), payload.get("fit_diagnostics"))


def _matrix_payload(matrix):
    from ..straylight import _grid, _matrix

    wavelength, values = np.asarray(matrix.wavelength), np.asarray(matrix.values)
    _matrix(values, _grid(wavelength))
    return {"wavelength": _array_payload(wavelength), "values": _array_payload(values)}


def _matrix_from_payload(cls, payload):
    from ..straylight import _grid, _matrix

    wavelength, values = _decode_array(payload["wavelength"]), _decode_array(payload["values"])
    _matrix(values, _grid(wavelength))
    return cls(wavelength, values)


def _is(module, name):
    """A type test that imports the type only when asked (avoiding cycles)."""
    def test(value):
        import importlib
        return isinstance(value, getattr(importlib.import_module(module, __package__), name))
    return test


def _straylight(name):
    def decode(payload):
        import importlib
        cls = getattr(importlib.import_module("..straylight", __package__), name)
        return _matrix_from_payload(cls, payload)
    return decode


# ---- codecs: how values are stored -------------------------------------

@dataclass(frozen=True)
class Codec:
    """How one kind of value is stored: ``{"kind": kind, ...}``.

    ``encode`` gives the stored fields (or, with ``section``, the body stored
    under that one key); ``decode`` takes them back. ``container`` values
    hold arrays large enough to store as files."""
    kind: str
    matches: Callable
    encode: Callable
    decode: Callable
    section: str | None = None
    container: bool = False


CODECS: list[Codec] = [
    Codec("wavelength_calibration", _is("..wavelength", "WavelengthCalibration"),
          _wavelength_calibration_payload, _wavelength_calibration_from_payload),
    Codec("spectrum", lambda v: isinstance(v, Spectrum), _spectrum_payload,
          _spectrum_from_payload, section="spectrum"),
    Codec("image", lambda v: isinstance(v, Image), _image_payload, _image_from_payload,
          section="image", container=True),
    Codec("stray_light", _is("..straylight", "StrayLightCorrection"), _matrix_payload,
          _straylight("StrayLightCorrection")),
    Codec("quantity", units.is_quantity,
          lambda v: {"value": float(v.magnitude), "unit": _unit_text(v.units)},
          lambda p: units.Q_(p["value"], p["unit"])),
    Codec("flag", lambda v: isinstance(v, bool), lambda v: {"value": v}, lambda p: p["value"]),
    Codec("number", lambda v: isinstance(v, numbers.Real), lambda v: {"value": float(v)},
          lambda p: p["value"]),
    Codec("text", lambda v: isinstance(v, str), lambda v: {"value": v}, lambda p: p["value"]),
    Codec("texts", lambda v: isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v),
          lambda v: {"value": list(v)}, lambda p: p["value"]),
    Codec("array", lambda v: isinstance(v, np.ndarray), lambda v: {"array": _array_payload(v)},
          lambda p: _decode_array(p["array"])),
]
"""Tried in order: the first whose ``matches`` accepts a value stores it."""


def codec_for(value) -> Codec:
    for codec in CODECS:
        if codec.matches(value):
            return codec
    raise TypeError(f"can't save a value of type {type(value).__name__}")


def _codec_named(kind) -> Codec:
    for codec in CODECS:
        if codec.kind == kind:
            return codec
    raise ValueError(f"unknown stored value kind {kind!r}")


def encode_value(value) -> dict:
    """``{"kind": ..., ...}`` for a value (see :data:`CODECS`)."""
    codec = codec_for(value)
    body = codec.encode(value)
    return {"kind": codec.kind, **({codec.section: body} if codec.section else body)}


def decode_value(payload):
    """The value stored by :func:`encode_value`."""
    codec = _codec_named(payload["kind"])
    return codec.decode(payload[codec.section] if codec.section else payload)


# ---- records: the kinds of file ----------------------------------------

class RecordError(ValueError):
    """A record that reads as JSON but fails a check of its own (shown as is)."""


@dataclass(frozen=True)
class Record:
    """One kind of ``.gsmath`` file."""
    format: str
    version: int
    section: str
    what: str
    """What it holds, for messages."""
    matches: Callable
    encode: Callable
    decode: Callable
    container: Callable = lambda obj: False
    """Whether ``obj`` is always written as a container (images); any record
    with an array over LARGE_ARRAY_BYTES is one regardless."""
    write_file: Callable | None = None
    """For a record with its own envelope (a checksum beside the section):
    ``write_file(obj, path, indent)``, used instead of the standard writer."""
    read_file: Callable | None = None
    """... and ``read_file(path)``, instead of the standard reader."""


def _measurement_payload(measurement):
    codec = codec_for(measurement.raw)
    if codec.section is None:
        raise TypeError(f"can't write a measurement of {type(measurement.raw).__name__}")
    exposure = measurement.exposure
    payload = {
        "kind": codec.kind,
        "raw": codec.encode(measurement.raw),
        "dark": None if measurement.dark is None else codec.encode(measurement.dark),
        "exposure": None if exposure is None else {
            "value": to_builtin(np.asarray(exposure.magnitude, dtype=float)),
            "unit": _unit_text(exposure.units)},
        "label": measurement.label,
        "settings": to_builtin(measurement.settings),
        "instrument": to_builtin(measurement.instrument),
        "metadata": to_builtin(measurement.metadata),
        "calibration": to_builtin(measurement.calibration),
    }
    if measurement.temperature is not None:  # written only when known
        temperature = measurement.temperature
        payload["temperature"] = {"value": float(temperature.magnitude),
                                  "unit": _unit_text(temperature.units)}
    return payload


def _measurement_from_payload(data):
    from ..measurement import Measurement

    codec = _codec_named(data["kind"])
    if codec.section is None:
        raise ValueError(f"unknown measurement kind {data['kind']!r}")
    exposure, temperature = data["exposure"], data.get("temperature")
    return Measurement(
        raw=codec.decode(data["raw"]),
        dark=None if data["dark"] is None else codec.decode(data["dark"]),
        exposure=None if exposure is None else units.Q_(
            np.asarray(exposure["value"], dtype=float) if isinstance(exposure["value"], list)
            else exposure["value"], exposure["unit"]),
        temperature=None if temperature is None else units.Q_(temperature["value"],
                                                             temperature["unit"]),
        label=data["label"],
        settings=data["settings"],
        instrument=data["instrument"],
        metadata=data["metadata"],
        calibration=data.get("calibration", {}),
    )


def _exposure_series_payload(series):
    return {
        "measurements": [_measurement_payload(m) for m in series.measurements],
        "saturation": series.saturation if series.saturation in (None, "auto")
        else float(series.saturation),
        "label": series.label,
        "settings": to_builtin(series.settings),
        "instrument": to_builtin(series.instrument),
        "metadata": to_builtin(series.metadata),
        "calibration": to_builtin(series.calibration),
    }


def _exposure_series_from_payload(data):
    from ..exposures import ExposureSeries

    return ExposureSeries(
        measurements=[_measurement_from_payload(m) for m in data["measurements"]],
        saturation=data["saturation"],
        label=data["label"],
        settings=data["settings"],
        instrument=data["instrument"],
        metadata=data["metadata"],
        calibration=data["calibration"],
    )


def _calibration_payload(calibration):
    return {
        "name": calibration.name,
        "fingerprint": calibration.fingerprint,
        "metadata": to_builtin(calibration.metadata),
        "steps": [{"type": step.name, "optional": step.optional,
                   "arguments": {k: encode_value(v) for k, v in step.arguments().items()}}
                  for step in calibration.steps],
    }


def _calibration_from_payload(data):
    from ..calibration import STEP_TYPES, Calibration

    steps = []
    for step in data["steps"]:
        try:
            cls = STEP_TYPES[step["type"]]
        except KeyError:
            raise ValueError(f"unknown step type {step['type']!r}") from None
        arguments = {k: decode_value(v) for k, v in step["arguments"].items()}
        steps.append(cls(**arguments, optional=step["optional"]))
    calibration = Calibration(data["name"], steps, metadata=data["metadata"])
    stored = data["fingerprint"]
    if calibration.fingerprint != stored:
        raise RecordError(f"contents don't match the saved fingerprint {stored} "
                          f"(now {calibration.fingerprint})")
    return calibration


def _calibration_container(calibration):
    return any(codec_for(v).container
               for step in calibration.steps for v in step.arguments().values())


RECORDS: list[Record] = [
    Record("gsmath.spectrum", 1, "spectrum", "spectrum", lambda o: isinstance(o, Spectrum),
           _spectrum_payload, _spectrum_from_payload),
    Record("gsmath.image", 1, "image", "image", lambda o: isinstance(o, Image),
           _image_payload, _image_from_payload, container=lambda o: True),
    Record("gsmath.measurement", 1, "measurement", "measurement",
           _is("..measurement", "Measurement"), _measurement_payload, _measurement_from_payload,
           container=lambda m: codec_for(m.raw).container),
    Record("gsmath.exposure_series", 1, "series", "exposure series",
           _is("..exposures", "ExposureSeries"), _exposure_series_payload,
           _exposure_series_from_payload,
           container=lambda s: any(codec_for(m.raw).container for m in s.measurements)),
    Record("gsmath.calibration", 1, "calibration", "calibration",
           _is("..calibration", "Calibration"), _calibration_payload, _calibration_from_payload,
           container=_calibration_container),
    Record("gsmath.straylight_distribution", 1, "straylight", "stray-light matrix",
           _is("..straylight", "StrayLightDistribution"), _matrix_payload,
           _straylight("StrayLightDistribution")),
    Record("gsmath.straylight_correction", 1, "straylight", "stray-light matrix",
           _is("..straylight", "StrayLightCorrection"), _matrix_payload,
           _straylight("StrayLightCorrection")),
    # A fitted factory line model keeps its own checksummed envelope.
    Record("gsmath.factory_lines", 1, "model", "factory line model",
           _is("..linefit", "FactoryLineModel"), None, None,
           write_file=lambda model, path, indent: model.write(path),
           read_file=lambda path: _factory_lines().read(path)),
]


def _factory_lines():
    from ..linefit import FactoryLineModel
    return FactoryLineModel


def _record_named(format_name) -> Record:
    for record in RECORDS:
        if record.format == format_name:
            return record
    raise ValueError(f"unknown gsmath record {format_name!r}; "
                     f"expected one of {sorted(r.format for r in RECORDS)}")


def record_for(obj) -> Record:
    for record in RECORDS:
        if record.matches(obj):
            return record
    raise TypeError(f"no native record stores a {type(obj).__name__}")


def write_record(obj, path, *, record=None, indent=2):
    """Write ``obj`` as its native record (or as ``record``, a format name)."""
    record = record_for(obj) if record is None else _record_named(record)
    if record.write_file is not None:
        return record.write_file(obj, path, indent)
    files = None
    if not record.container(obj):
        largest = [0]
        token = _LARGEST_ARRAY.set(largest)
        try:
            payload = {"format": record.format, "version": record.version,
                       record.section: record.encode(obj)}
        finally:
            _LARGEST_ARRAY.reset(token)
        if largest[0] <= LARGE_ARRAY_BYTES:
            return _write_payload(payload, path, indent, record.what)
    with _arrays_as_files() as files:  # large arrays as .npy files in a container
        payload = {"format": record.format, "version": record.version,
                   record.section: record.encode(obj)}
    return _write_payload(payload, path, indent, record.what, files)


def read_record(path, *, record=None):
    """The object in a native file -- whatever its ``format`` says, or
    checked to be ``record`` (a format name)."""
    path = Path(path)
    if record is None:
        try:
            record = record_format(path)
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Could not read native gsmath file {path}.") from exc
        try:
            chosen = _record_named(record)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from None
    else:
        chosen = _record_named(record)
    if chosen.read_file is not None:
        return chosen.read_file(path)
    payload = _read_payload(path, chosen.format, chosen.version, chosen.what)
    try:
        return chosen.decode(payload[chosen.section])
    except RecordError as exc:
        raise RecordError(f"{path}: {exc}") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid native {chosen.what} payload in {path}: {exc}") from exc


# ---- files ------------------------------------------------------------

def _write_payload(payload, path, indent, what, files=None):
    """Write a JSON record, or a container if its arrays are `files`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(payload, indent=indent, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Native {what} metadata must be JSON-compatible and finite.") from exc
    if files is None:
        path.write_text(text + "\n", encoding="utf-8")
        return path
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as container:
        container.writestr(_RECORD_MEMBER, text + "\n")
        for name, array in files.items():
            with container.open(name, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, array, allow_pickle=False)
    return path


def _load_record(path, *, arrays=True):
    """The JSON record in `path`, with a container's array files loaded in
    place of their references (unless `arrays` is False)."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        return json.loads(path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(path) as container:
        payload = json.loads(container.read(_RECORD_MEMBER).decode("utf-8"))
        if not arrays:
            return payload

        def load(node):
            if isinstance(node, dict):
                if set(node) == {"dtype", "shape", "file"}:
                    with container.open(node["file"]) as member:
                        array = np.lib.format.read_array(member, allow_pickle=False)
                    return {"dtype": node["dtype"], "shape": node["shape"], "array": array}
                return {key: load(value) for key, value in node.items()}
            if isinstance(node, list):
                return [load(item) for item in node]
            return node

        return load(payload)


def record_format(path):
    """The ``format`` a native record declares, without loading its arrays."""
    record = _load_record(path, arrays=False)
    return record.get("format") if isinstance(record, dict) else None


def _read_payload(path, format_name, version, what):
    path = Path(path)
    try:
        payload = _load_record(path)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Could not read native {what} file {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid native {what} payload in {path}.")
    if payload.get("format") != format_name:
        raise ValueError(f"Not a {format_name!r} file: {path}")
    if payload.get("version") != version:
        raise ValueError(f"Unsupported native {what} version {payload.get('version')!r}.")
    return payload


# ---- the established entry points -------------------------------------

FORMAT_NAME = "gsmath.spectrum"
IMAGE_FORMAT = "gsmath.image"
MEASUREMENT_FORMAT = "gsmath.measurement"
EXPOSURE_SERIES_FORMAT = "gsmath.exposure_series"
CALIBRATION_FORMAT = "gsmath.calibration"
STRAYLIGHT_FORMATS = {"distribution": "gsmath.straylight_distribution",
                      "correction": "gsmath.straylight_correction"}


def write_native(spectrum, path, *, indent=2):
    """Write a spectrum to the versioned native ``.gsmath`` JSON format."""
    return write_record(spectrum, path, record=FORMAT_NAME, indent=indent)


def read_native(path):
    """Read and validate a spectrum stored in native ``.gsmath`` format."""
    return read_record(path, record=FORMAT_NAME)


def write_image(image, path, *, indent=2):
    """Write an image to the native ``.gsmath`` format (a container)."""
    return write_record(image, path, record=IMAGE_FORMAT, indent=indent)


def read_image(path):
    """Read and validate an image stored in native ``.gsmath`` format."""
    return read_record(path, record=IMAGE_FORMAT)


def write_measurement(measurement, path, *, indent=2):
    """Write a Measurement: JSON for spectra, a container for images."""
    return write_record(measurement, path, record=MEASUREMENT_FORMAT, indent=indent)


def read_measurement(path):
    """Read a Measurement written by :func:`write_measurement`."""
    return read_record(path, record=MEASUREMENT_FORMAT)


def write_exposure_series(series, path, *, indent=2):
    """Write an ExposureSeries (every exposure's Measurement, the saturation
    level, settings, ...): JSON for spectra, a container for images."""
    return write_record(series, path, record=EXPOSURE_SERIES_FORMAT, indent=indent)


def read_exposure_series(path):
    """Read an ExposureSeries written by :func:`write_exposure_series`."""
    return read_record(path, record=EXPOSURE_SERIES_FORMAT)


def write_calibration(calibration, path, *, indent=2):
    """Write a Calibration -- its steps and all their data -- with its
    fingerprint: JSON, or a container if a step holds an image."""
    return write_record(calibration, path, record=CALIBRATION_FORMAT, indent=indent)


def read_calibration(path):
    """Read a Calibration, checking its contents still match the
    fingerprint it was saved with."""
    return read_record(path, record=CALIBRATION_FORMAT)


def write_straylight_matrix(matrix, path, *, kind, indent=2):
    """Save a stray-light matrix with its wavelength grid and array dtype."""
    return write_record(matrix, path, record=STRAYLIGHT_FORMATS[kind], indent=indent)


def read_straylight_matrix(path, *, kind):
    """Read and validate the grid and values of a native stray-light file."""
    matrix = read_record(path, record=STRAYLIGHT_FORMATS[kind])
    return matrix.wavelength, matrix.values


# Older names, used by other modules.
_value_payload = encode_value
_value_from_payload = decode_value
