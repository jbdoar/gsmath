import numpy as np
import pytest

from gsmath import (
    BlackLevel, Calibration, Demosaic, FlatField, Image, Measurement, PerExposure, PixelMap,
    SubtractDark, demosaic,
)
from gsmath.mosaic import Mosaic, site_masks
from gsmath.units import Q_


def mosaic_of(rgb, pattern="RGGB", black_level=None):
    """Sample an (rows, cols, 3) scene through a colour-filter pattern."""
    masks = site_masks(pattern, rgb.shape[:2])
    values = sum(np.where(masks[c], rgb[:, :, i], 0) for i, c in enumerate("RGB"))
    return Image(values, value_unit="count",
                 mosaic=Mosaic(pattern, black_level))


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_bilinear_keeps_measured_samples_and_fills_flat_fields(pattern):
    scene = np.empty((8, 10, 3))
    scene[:] = [100.0, 200.0, 300.0]
    rgb = demosaic(mosaic_of(scene, pattern))
    assert rgb.channels == ("R", "G", "B")
    np.testing.assert_allclose(rgb.values, scene)  # edges included
    assert rgb.mosaic is None and rgb.metadata["demosaic"]["method"] == "bilinear"


def test_bilinear_is_exact_for_a_linear_ramp_away_from_the_edges():
    rows, cols = np.mgrid[0:12, 0:14].astype(float)
    scene = np.stack([rows + cols, 2 * rows, 3 * cols + 5], axis=-1)
    raw = mosaic_of(scene, "RGGB")
    rgb = raw.demosaic()
    np.testing.assert_allclose(rgb.values[1:-1, 1:-1], scene[1:-1, 1:-1])
    measured = site_masks("RGGB", raw.shape)
    for i, colour in enumerate("RGB"):
        np.testing.assert_array_equal(rgb.values[:, :, i][measured[colour]],
                                      raw.values[measured[colour]])


def test_superpixel_averages_each_block():
    scene = np.random.default_rng(1).uniform(0, 1000, (6, 9, 3))
    raw = mosaic_of(scene, "GBRG")
    raw = raw.copy(pixel_map=PixelMap.linear(raw.shape, 1.0, unit="mm"))
    rgb = demosaic(raw, "superpixel")
    assert rgb.shape == (3, 4)  # odd last column dropped
    block = raw.values[0:2, 2:4]  # G B / R G
    np.testing.assert_allclose(rgb.values[0, 1], [block[1, 0], block[[0, 1], [0, 1]].mean(), block[0, 1]])
    np.testing.assert_allclose(rgb.pixel_map.x, [-3.5, -1.5, 0.5, 2.5])
    np.testing.assert_allclose(rgb.pixel_map.y, [-2.0, 0.0, 2.0])


def test_slicing_keeps_the_pattern_aligned():
    rows, cols = np.mgrid[0:8, 0:8].astype(float)
    scene = np.stack([rows, cols, rows * cols], axis=-1)
    raw = mosaic_of(scene, "RGGB", [1, 2, 3, 4])
    part = raw[1:7, 3:8]  # starts on the B site
    assert part.mosaic.pattern == "BGGR"
    assert part.mosaic.black_level == (4, 3, 2, 1)
    np.testing.assert_allclose(part.demosaic().values[1:-1, 1:-1],
                               raw.demosaic().values[1:7, 3:8][1:-1, 1:-1])
    with pytest.raises(ValueError, match="doesn.t leave a colour-filter mosaic"):
        raw[::2, :]


def test_black_level_per_site():
    raw = Image(np.full((4, 4), 100.0), value_unit="count",
                mosaic=Mosaic("RGGB", [10, 20, 30, 40]))
    np.testing.assert_array_equal(BlackLevel().apply(raw, None).values[:2, :2], [[90, 80], [70, 60]])
    with pytest.raises(ValueError, match="no recorded black level"):
        BlackLevel().apply(Image(np.zeros((2, 2)), mosaic="RGGB"), None)


def test_demosaic_needs_a_mosaic():
    with pytest.raises(ValueError, match="isn't a colour-filter mosaic"):
        Image(np.zeros((4, 4))).demosaic()
    with pytest.raises(ValueError, match="patterns must match"):
        Image(np.zeros((4, 4)), mosaic="RGGB") - Image(np.zeros((4, 4)), mosaic="BGGR")
    with pytest.raises(ValueError, match="unsupported colour-filter pattern"):
        Image(np.zeros((4, 4)), mosaic="RGBW")
    with pytest.raises(ValueError, match="single-channel"):
        Image(np.zeros((4, 4, 3)), mosaic="RGGB")
    with pytest.raises(ValueError, match="unknown demosaic method"):
        Demosaic("nearest")


def test_a_raw_measurement_calibrates_through_a_demosaic(tmp_path):
    scene = np.empty((6, 8, 3))
    scene[:] = [400.0, 800.0, 1200.0]
    raw = mosaic_of(scene + 50, "RGGB", [50] * 4)
    dark = mosaic_of(np.full((6, 8, 3), 50.0), "RGGB")
    flat = Image(np.ones((6, 8)))
    camera = Calibration("camera", [SubtractDark(), FlatField(flat), Demosaic(), PerExposure()])
    measurement = Measurement(raw=raw, dark=dark, exposure=Q_(4, "ms"))
    result = measurement.process(camera)
    np.testing.assert_allclose(result.values, scene / 4)
    assert result.channels == ("R", "G", "B") and str(result.value_unit) == "count / millisecond"
    loaded = Calibration.read(camera.write(tmp_path / "camera.gsmath"))
    assert loaded.steps[2].method == "bilinear" and loaded.fingerprint == camera.fingerprint
    assert Calibration("c", [Demosaic("superpixel")]).fingerprint != Calibration("c", [Demosaic()]).fingerprint
