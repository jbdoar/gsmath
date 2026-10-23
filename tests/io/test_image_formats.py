import json
import zipfile

import numpy as np
import pytest

import gsmath
from gsmath import Calibration, FlatField, Image, Measurement, PixelMap, Spectrum
from gsmath.mosaic import Mosaic
from gsmath.units import Q_

pytest.importorskip("imagecodecs")
tifffile = pytest.importorskip("tifffile")


def ramp(shape, dtype, channels=None, **kwargs):
    values = (np.arange(np.prod(shape)) * 37 % np.iinfo(dtype).max).astype(dtype).reshape(shape)
    return Image(values, channels=channels, **kwargs)


# native container
def test_image_records_are_containers_of_npy_arrays(tmp_path):
    image = ramp((40, 60, 3), np.uint16, ("R", "G", "B"),
                 pixel_map=PixelMap.linear((40, 60), 0.1, unit="deg"))
    path = image.write(tmp_path / "frame.gsmath")
    with zipfile.ZipFile(path) as container:
        assert sorted(container.namelist()) == ["arrays/0.npy", "arrays/1.npy", "arrays/2.npy",
                                                "record.json"]
        record = json.loads(container.read("record.json"))
    assert record["format"] == "gsmath.image"
    assert record["image"]["values"] == {"dtype": "<u2", "shape": [40, 60, 3], "file": "arrays/0.npy"}
    again = Image.read(path)
    np.testing.assert_array_equal(again.values, image.values)
    assert again.pixel_map == image.pixel_map


def test_spectrum_records_stay_json(tmp_path):
    path = Spectrum([1, 2], [3, 4]).write(tmp_path / "s.gsmath")
    assert not zipfile.is_zipfile(path)
    json.loads(path.read_text())


def test_calibrations_holding_images_are_containers(tmp_path):
    flat = Calibration("flat", [FlatField(ramp((4, 6), np.uint16), normalize=False)])
    path = flat.write(tmp_path / "flat.gsmath")
    assert zipfile.is_zipfile(path)
    loaded = gsmath.read(path)
    assert loaded.fingerprint == flat.fingerprint and loaded.steps[0].normalize is False
    measurement = Measurement(raw=ramp((4, 6), np.uint16) + 1.0)
    assert zipfile.is_zipfile(measurement.write(tmp_path / "m.gsmath"))


# PNG and JPEG
@pytest.mark.parametrize("shape, dtype, channels", [
    ((5, 7), np.uint8, None),
    ((5, 7), np.uint16, None),
    ((5, 7, 3), np.uint8, ("R", "G", "B")),
    ((5, 7, 3), np.uint16, ("R", "G", "B")),  # Pillow would truncate this to 8 bits
    ((5, 7, 4), np.uint16, ("R", "G", "B", "A")),
])
def test_png_round_trips_exactly(tmp_path, shape, dtype, channels):
    image = ramp(shape, dtype, channels)
    again = Image.read(image.write(tmp_path / "a.png"))
    np.testing.assert_array_equal(again.values, image.values)
    assert again.values.dtype == dtype and again.channels == channels
    assert again.label == "a" and again.value_unit is None
    assert again.metadata["source"]["format"] == "png"
    assert again.metadata["source"]["path"] == "a.png"
    assert str(tmp_path) not in str(again.metadata)


def test_jpeg_is_lossy_eight_bit(tmp_path):
    image = Image(np.full((16, 16, 3), [200, 100, 50], dtype=np.uint8), channels=("R", "G", "B"))
    again = gsmath.read(image.write(tmp_path / "a.jpg", quality=98))
    assert isinstance(again, Image) and again.channels == ("R", "G", "B")
    np.testing.assert_allclose(again.values, image.values, atol=3)


def test_display_formats_refuse_data_they_cant_hold(tmp_path):
    with pytest.raises(TypeError, match="convert it first"):
        Image(np.zeros((2, 2))).write(tmp_path / "a.png")
    with pytest.raises(TypeError, match="uint8"):
        ramp((2, 2), np.uint16).write(tmp_path / "a.jpg")
    with pytest.raises(ValueError, match="channels"):
        ramp((2, 2, 2), np.uint8).write(tmp_path / "a.png")


# TIFF
@pytest.mark.parametrize("values", [
    np.arange(12, dtype=np.uint16).reshape(3, 4),
    np.linspace(-1, 1, 36).reshape(3, 4, 3),
    np.arange(60, dtype=np.float32).reshape(3, 4, 5),
])
def test_tiff_round_trips_data_and_gsmath_fields(tmp_path, values):
    channels = ("R", "G", "B") if values.ndim == 3 and values.shape[2] == 3 else None
    image = Image(values, label="frame 7", value_unit="cd/m^2", channels=channels,
                  pixel_map=PixelMap.linear(values.shape[:2], 0.25, unit="mm"),
                  metadata={"gain": 2, "note": "µ-test"})
    again = Image.read(image.write(tmp_path / "a.tif", compression="zlib"))
    np.testing.assert_array_equal(again.values, image.values)
    assert again.values.dtype == values.dtype
    assert again.label == "frame 7" and again.value_unit == image.value_unit
    assert again.channels == image.channels and again.pixel_map == image.pixel_map
    assert again.metadata == {"gain": 2, "note": "µ-test"}


def test_a_foreign_tiff_reads_as_its_data(tmp_path):
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    tifffile.imwrite(tmp_path / "rgb.tif", rgb, photometric="rgb")
    image = Image.read(tmp_path / "rgb.tif", value_unit="count")
    assert image.channels == ("R", "G", "B") and image.value_unit == gsmath.units.parse("count")
    np.testing.assert_array_equal(image.values, rgb)
    planar = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)
    tifffile.imwrite(tmp_path / "planar.tif", planar, photometric="rgb", planarconfig="separate")
    np.testing.assert_array_equal(Image.read(tmp_path / "planar.tif").values,
                                  np.moveaxis(planar, 0, -1))
    tifffile.imwrite(tmp_path / "pages.tif", np.zeros((2, 3, 4), dtype=np.uint8),
                     photometric="minisblack")
    with pytest.raises(ValueError, match="pages"):
        Image.read(tmp_path / "pages.tif")
    assert Image.read(tmp_path / "pages.tif", page=1).shape == (3, 4)


# camera raw
def synthetic_dng(path, mosaic, *, cfa=(0, 1, 1, 2), black=512, white=4095, shutter=(1, 100)):
    """A minimal DNG that LibRaw reads: one CFA page with the required tags."""
    identity = (1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1)
    tags = [
        (271, "s", 0, "gsmath", True), (272, "s", 0, "synthetic", True),
        (50706, "B", 4, (1, 4, 0, 0), True), (50707, "B", 4, (1, 1, 0, 0), True),
        (50708, "s", 0, "gsmath synthetic", True),
        (33421, "H", 2, (2, 2), True), (33422, "B", 4, cfa, True),
        (50710, "B", 3, (0, 1, 2), True), (50711, "H", 1, (1,), True),
        (50714, "H", 1, (black,), True), (50717, "H", 1, (white,), True),
        (50721, "2i", 9, identity, True), (50778, "H", 1, (21,), True),
        (33434, "2I", 1, shutter, True),
    ]
    tifffile.imwrite(path, mosaic, photometric=32803, extratags=tags, metadata=None, subfiletype=0)
    return path


def test_a_raw_file_reads_as_its_mosaic(tmp_path):
    pytest.importorskip("rawpy")
    mosaic = (np.arange(64 * 96).reshape(64, 96) % 3000 + 512).astype(np.uint16)
    image = gsmath.read(synthetic_dng(tmp_path / "shot.dng", mosaic, cfa=(1, 0, 2, 1)))
    assert isinstance(image, Image) and image.channels is None
    np.testing.assert_array_equal(image.values, mosaic)
    assert image.value_unit == gsmath.units.parse("count")
    assert image.mosaic == Mosaic("GRBG", [512.0] * 4, 4095.0)
    assert "mosaic" not in image.metadata
    assert image.metadata["camera"]["exposure_s"] == pytest.approx(0.01)
    rgb = image.demosaic()
    assert rgb.channels == ("R", "G", "B") and rgb.metadata["demosaic"]["cfa"] == "GRBG"
    with pytest.raises(ValueError, match="read but not written"):
        image.write(tmp_path / "out.dng")


def test_a_raw_frames_mosaic_survives_tiff(tmp_path):
    raw = Image(np.arange(48, dtype=np.uint16).reshape(6, 8), value_unit="count",
                mosaic=Mosaic("GBRG", [60, 61, 62, 63], 4095))
    again = Image.read(raw.write(tmp_path / "raw.tif"))
    assert again.mosaic == raw.mosaic and "mosaic" not in again.metadata
    np.testing.assert_array_equal(again.values, raw.values)
