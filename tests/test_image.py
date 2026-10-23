import numpy as np
import pytest

import gsmath
from gsmath import (
    Calibration, Circle, FlatField, Image, Mask, Measurement, PerExposure, PixelMap, Polygon,
    Rectangle, Responsivity, Spectrum, SubtractDark,
)
from gsmath.units import Q_


def rgb(shape=(4, 6), *, unit="count", pixel_map=None, dtype=np.uint16):
    rows, cols = shape
    values = np.stack([np.full(shape, 100), np.full(shape, 200), np.full(shape, 300)], axis=-1)
    values = values + np.arange(rows * cols).reshape(shape)[:, :, None]
    return Image(values.astype(dtype), label="rgb", value_unit=unit, channels=("R", "G", "B"),
                 pixel_map=pixel_map)


# construction
def test_an_image_validates_its_shape_channels_and_pixel_map():
    with pytest.raises(ValueError, match="rows, columns"):
        Image(np.zeros(5))
    with pytest.raises(ValueError, match="channel"):
        Image(np.zeros((2, 2, 3)), channels=("R", "G"))
    with pytest.raises(ValueError, match="unique"):
        Image(np.zeros((2, 2, 2)), channels=("R", "R"))
    with pytest.raises(ValueError, match="pixel map shape"):
        Image(np.zeros((2, 3)), pixel_map=PixelMap.linear((3, 2), 1.0))
    assert Image(np.zeros((2, 2, 3))).channels == ("0", "1", "2")
    assert Image(np.zeros((2, 2))).channels is None
    assert Image(np.zeros((2, 2)), channels="Y").channels == ("Y",)


def test_shape_properties():
    image = rgb((4, 6))
    assert image.shape == (4, 6) and image.height == 4 and image.width == 6
    assert image.n_channels == 3 and Image(np.zeros((4, 6))).n_channels == 1
    assert "4x6" in str(image)


# selection
def test_slicing_keeps_the_pixel_map_aligned():
    image = rgb((4, 6), pixel_map=PixelMap.linear((4, 6), 0.5, unit="deg"))
    part = image[1:3, 2:5]
    assert part.shape == (2, 3) and part.channels == ("R", "G", "B")
    np.testing.assert_array_equal(part.values, image.values[1:3, 2:5])
    np.testing.assert_array_equal(part.pixel_map.x, image.pixel_map.x[2:5])
    np.testing.assert_array_equal(part.pixel_map.y, image.pixel_map.y[1:3])
    with pytest.raises(TypeError, match="slices"):
        image[1, :]


def test_channel_selection():
    image = rgb()
    green = image["G"]
    assert green.values.ndim == 2 and green.channels == ("G",)
    np.testing.assert_array_equal(green.values, image.values[:, :, 1])
    rb = image[["R", "B"]]
    assert rb.channels == ("R", "B") and rb.values.shape == (4, 6, 2)
    assert image[0:2, 0:2, "B"].shape == (2, 2)
    with pytest.raises(KeyError, match="no channel 'X'"):
        image["X"]


def test_crop_to_a_region():
    image = rgb((10, 10))
    cropped = image.crop(Circle(5, 5, 2))
    assert cropped.shape == (5, 5)
    np.testing.assert_array_equal(cropped.values, image.values[3:8, 3:8])


# arithmetic
def test_integer_raw_data_subtracts_without_wrapping():
    raw = Image(np.full((2, 2), 5, dtype=np.uint16), value_unit="count")
    dark = Image(np.full((2, 2), 10, dtype=np.uint16), value_unit="count")
    difference = raw - dark
    np.testing.assert_array_equal(difference.values, -5.0)
    assert difference.value_unit == gsmath.units.parse("count")


def test_arithmetic_carries_units():
    image = rgb(dtype=float)
    rate = image / Q_(100, "ms")
    assert rate.value_unit == gsmath.units.parse("count / ms")
    np.testing.assert_allclose(rate.values, image.values / 100)
    with pytest.raises(ValueError, match="unitless scalar"):
        image + 1
    assert np.all((image + Q_(1, "count")).values == image.values + 1)
    assert np.all((-image).values == -image.values)


def test_image_arithmetic_checks_grids_and_broadcasts_one_channel():
    image = rgb(dtype=float)
    gain = Image(np.full((4, 6), 2.0), value_unit="uW/count")
    product = image * gain
    assert product.channels == ("R", "G", "B")
    np.testing.assert_allclose(product.values, image.values * 2)
    assert product.value_unit == gsmath.units.parse("uW")
    with pytest.raises(ValueError, match="shapes"):
        image + rgb((3, 6), dtype=float)
    other = Image(image.values, value_unit="count", channels=("X", "Y", "Z"))
    with pytest.raises(ValueError, match="channels"):
        image - other
    mapped = image.copy(pixel_map=PixelMap.linear((4, 6), 1.0, unit="mm"))
    with pytest.raises(ValueError, match="pixel maps"):
        mapped - image.copy(pixel_map=PixelMap.linear((4, 6), 2.0, unit="mm"))
    assert (mapped - image).pixel_map == mapped.pixel_map


def test_copy_can_remove_the_pixel_map_and_channels():
    image = rgb(pixel_map=PixelMap.linear((4, 6), 1.0))
    assert image.copy().pixel_map == image.pixel_map
    assert image.copy(pixel_map=None).pixel_map is None
    assert Image(np.zeros((2, 2)), channels="Y").copy(channels=None).channels is None
    assert image.copy().value_unit == image.value_unit
    assert image.copy(value_unit=None).value_unit is None


# statistics
def test_statistics_are_per_channel_and_respect_regions():
    image = rgb((4, 6), dtype=float)
    stats = image.stats()
    np.testing.assert_allclose(stats.mean, image.values.mean(axis=(0, 1)))
    np.testing.assert_array_equal(stats.n, [24, 24, 24])
    assert set(stats.summary) == {"R", "G", "B"}
    region = image.stats(roi=Rectangle(0, 0, 2, 1))
    np.testing.assert_allclose(region.mean, image.values[0:1, 0:2].mean(axis=(0, 1)))
    assert image.stats() is stats  # cached
    assert stats.quantity("mean").units == gsmath.units.parse("count")


def test_statistics_ignore_nan_pixels():
    values = np.array([[1.0, 2.0], [np.nan, 3.0]])
    stats = Image(values).stats()
    assert stats.mean == pytest.approx(2.0) and stats.n == 3
    assert stats.percentile(50) == pytest.approx(2.0)
    assert set(stats.summary) >= {"mean", "std", "rms", "n"}
    assert np.isnan(Image(values).stats(roi=Mask(np.zeros((2, 2)))).mean)


def test_histogram_shares_bins_across_channels():
    image = rgb((4, 6), dtype=float)
    histogram = image.histogram(bins=10)
    assert histogram.counts.shape == (3, 10)
    np.testing.assert_array_equal(histogram.counts.sum(axis=1), [24, 24, 24])
    assert histogram.edges[0] == image.values.min() and histogram.edges[-1] == image.values.max()
    mono = Image(np.arange(10.0).reshape(2, 5)).histogram(bins=5, range=(0, 10))
    np.testing.assert_array_equal(mono.counts, [2, 2, 2, 2, 2])
    np.testing.assert_allclose(mono.centers, [1, 3, 5, 7, 9])


# regions and pixel maps
def test_regions_in_pixels():
    image = Image(np.zeros((5, 5)))
    assert Rectangle(1, 1, 3, 4).mask(image).sum() == 6
    assert Circle(2, 2, 1).mask(image).sum() == 5
    ring = Circle(2, 2, 1.5) - Circle(2, 2, 0.5)
    assert ring.mask(image).sum() == 8
    assert (~Rectangle(0, 0, 5, 5)).mask(image).sum() == 0
    assert (Rectangle(0, 0, 2, 5) | Rectangle(3, 0, 5, 5)).mask(image).sum() == 20
    triangle = Polygon([(-0.5, -0.5), (4, -0.5), (-0.5, 4)])  # column + row <= 3
    assert triangle.mask(image).sum() == 10
    with pytest.raises(ValueError, match="mask shape"):
        Mask(np.ones((2, 2))).mask(image)


def test_regions_in_mapped_coordinates():
    angles = PixelMap.linear((101, 101), step=(0.1, -0.1), unit="deg")  # y up
    image = Image(np.zeros((101, 101)), pixel_map=angles)
    spot = Circle(0, 0, Q_(0.55, "deg").to("arcmin"), coords=True)
    np.testing.assert_array_equal(spot.mask(image), Circle(50, 50, 5.5).mask(image))
    upper = Rectangle(-6, 0.05, 6, 6, coords=True).mask(image)
    assert upper[:50].all() and not upper[50:].any()  # y increases upward
    with pytest.raises(ValueError, match="no pixel map"):
        Circle(0, 0, 1, coords=True).mask(Image(np.zeros((3, 3))))


def test_pixel_maps():
    linear = PixelMap.linear((3, 5), step=2.0, unit="mm")
    np.testing.assert_allclose(linear.x, [-4, -2, 0, 2, 4])
    np.testing.assert_allclose(linear.y, [-2, 0, 2])
    assert linear.separable and linear.shape == (3, 5)
    shifted = PixelMap.linear((3, 5), step=1.0, center=(0, 0))
    np.testing.assert_allclose(shifted.x, np.arange(5))
    radial = PixelMap.from_function((3, 5), lambda c, r: (np.hypot(c, r), r), unit="deg")
    assert not radial.separable and radial.shape == (3, 5)
    x, y = linear.grid()
    assert x.shape == (3, 5) and y[2, 0] == 2
    assert linear.to_unit("um").x[0] == pytest.approx(-4000)
    assert radial.sliced(slice(1, 3), slice(0, 2)).shape == (2, 2)


# measurement, calibration, persistence
@pytest.fixture
def frame():
    raw = rgb((4, 6))
    dark = Image(np.full((4, 6, 3), 50, dtype=np.uint16), label="dark", value_unit="count",
                 channels=("R", "G", "B"))
    return Measurement(raw=raw, dark=dark, exposure=Q_(10, "ms"), label="frame 1")


@pytest.fixture
def camera():
    flat = Image(np.linspace(0.5, 1.5, 24).reshape(4, 6), label="flat")
    return Calibration("camera", [SubtractDark(), FlatField(flat), PerExposure(),
                                  Responsivity(Q_(2, "cd/m^2 * ms / count"))])


def test_an_image_measurement_processes_through_a_calibration(frame, camera):
    result = frame.process(camera)
    assert isinstance(result, Image) and result.label == "frame 1"
    flat = camera.steps[1].flat.values
    expected = (frame.raw.values - 50.0) / (flat / flat.mean())[:, :, None] / 10 * 2
    np.testing.assert_allclose(result.values, expected)
    assert result.to_value_unit("cd/m^2") is not None
    assert result.metadata["processing"]["complete"]


def test_flat_field_checks_its_shape(frame):
    small = Image(np.ones((2, 2)))
    with pytest.raises(ValueError, match="flat field"):
        frame.process(Calibration("bad", [FlatField(small)]))


def test_native_image_round_trip(tmp_path):
    image = rgb(pixel_map=PixelMap.linear((4, 6), 0.1, unit="deg"))
    image.metadata["camera"] = {"gain": np.int64(2)}
    path = image.write(tmp_path / "frame.gsmath")
    again = Image.read(path)
    np.testing.assert_array_equal(again.values, image.values)
    assert again.values.dtype == np.uint16 and again.channels == image.channels
    assert again.pixel_map == image.pixel_map and again.value_unit == image.value_unit
    assert again.metadata == {"camera": {"gain": 2}}
    assert isinstance(gsmath.read(path), Image)
    with pytest.raises(ValueError, match="Not a 'gsmath.spectrum' file"):
        Spectrum.read(path)
    with pytest.raises(ValueError, match="image format"):
        Image.read(tmp_path / "frame.bmp")


def test_image_measurements_and_calibrations_round_trip(tmp_path, frame, camera):
    measurement = Measurement.read(frame.write(tmp_path / "m.gsmath"))
    assert isinstance(measurement.raw, Image) and isinstance(measurement.dark, Image)
    loaded = Calibration.read(camera.write(tmp_path / "c.gsmath"))
    assert loaded.fingerprint == camera.fingerprint
    np.testing.assert_array_equal(measurement.process(loaded).values, frame.process(camera).values)


def test_the_fingerprint_follows_image_data():
    def calibration(values, unit=None, pixel_map=None):
        return Calibration("c", [Responsivity(Image(values, value_unit=unit, pixel_map=pixel_map))])

    ones = np.ones((2, 2))
    assert calibration(ones).fingerprint == calibration(ones.copy()).fingerprint
    assert calibration(ones).fingerprint != calibration(ones * 2).fingerprint
    assert calibration(ones).fingerprint != calibration(ones, "W").fingerprint
    assert (calibration(ones).fingerprint
            != calibration(ones, pixel_map=PixelMap.linear((2, 2), 1.0)).fingerprint)


def _brute_force(roi, image):
    """The region's mask from every pixel of the frame (no window)."""
    from gsmath.roi import FULL, _Combined, _Not, Mask

    if isinstance(roi, Mask):
        return roi.array.copy()
    if isinstance(roi, _Combined):
        return roi.op(_brute_force(roi.a, image), _brute_force(roi.b, image))
    if isinstance(roi, _Not):
        return ~_brute_force(roi.a, image)
    return roi._local(image, FULL)


@pytest.mark.parametrize("make", [
    lambda: Rectangle(3.2, 1.5, 17.9, 9.0),
    lambda: Rectangle(-5, -5, 4, 3),                         # over the corner
    lambda: Rectangle(30, 30, 40, 40),                       # outside: empty
    lambda: Circle(10, 6, 4.3),
    lambda: Circle(0, 0, 3),                                 # clipped at the corner
    lambda: Polygon(((2, 1), (15, 3), (12, 11), (4, 9))),
    lambda: Circle(0.5, -0.2, 0.6, coords=True),             # in map units
    lambda: Rectangle(Q_(-30, "arcmin"), -0.4, 0.9, 0.3, coords=True),
    lambda: Circle(10, 6, 5) - Circle(10, 6, 2),             # an annulus
    lambda: Circle(5, 5, 3) | Rectangle(14, 2, 19, 6),
    lambda: Circle(8, 6, 4) & Rectangle(6, 0, 20, 20),
    lambda: ~Circle(10, 6, 3),
    lambda: Mask(np.arange(240).reshape(12, 20) % 7 == 0),
])
def test_a_region_only_looks_within_its_window_yet_finds_the_same_pixels(make):
    image = Image(np.arange(240.0).reshape(12, 20),
                  pixel_map=PixelMap.linear((12, 20), (0.1, -0.1), unit="deg"))
    roi = make()
    expected = _brute_force(roi, image)
    np.testing.assert_array_equal(roi.mask(image), expected)
    np.testing.assert_array_equal(image.pixels(roi), image.values[expected])
    rows, cols = roi.window(image)
    outside = np.ones(image.shape, bool)
    outside[rows, cols] = False
    assert not np.any(expected & outside)  # the window holds the whole region
