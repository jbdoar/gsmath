import numpy as np
import pytest

from gsmath import Circle, Image, LineCut, PixelMap, Rectangle
from gsmath.mosaic import Mosaic, site_masks
from gsmath.units import Q_, parse


def ramp(shape=(6, 8), *, channels=False, unit="count", pixel_map=None):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    plane = 2 * cols + 3 * rows + 1
    if channels:
        return Image(np.stack([plane, 10 * plane, 100 * plane], axis=-1), label="ramp",
                     value_unit=unit, channels=("R", "G", "B"), pixel_map=pixel_map)
    return Image(plane, label="ramp", value_unit=unit, pixel_map=pixel_map)


def mosaic(shape=(8, 10), pattern="RGGB", seed=0):
    scene = np.random.default_rng(seed).uniform(100, 1000, (*shape, 3))
    masks = site_masks(pattern, shape)
    values = sum(np.where(masks[c], scene[:, :, i], 0) for i, c in enumerate("RGB"))
    return Image(values, value_unit="count",
                 mosaic=Mosaic(pattern, [1, 2, 3, 4]))


def coordinates_follow_pixels(before, after, old_position):
    """Every pixel of `after` has the coordinates its source pixel had."""
    bx, by = before.pixel_map.grid()
    ax, ay = after.pixel_map.grid()
    for i in range(after.height):
        for j in range(after.width):
            r, c = old_position(i, j)
            assert (ax[i, j], ay[i, j]) == (bx[r, c], by[r, c])


# orientation
def test_flip_mirrors_values_and_coordinates():
    image = ramp(channels=True, pixel_map=PixelMap.linear((6, 8), 0.5, unit="mm"))
    left_right = image.flip()
    np.testing.assert_array_equal(left_right.values, image.values[:, ::-1])
    coordinates_follow_pixels(image, left_right, lambda i, j: (i, 7 - j))
    up_down = image.flip("vertical")
    np.testing.assert_array_equal(up_down.values, image.values[::-1])
    np.testing.assert_array_equal(up_down.flip("vertical").values, image.values)
    with pytest.raises(ValueError, match="horizontal"):
        image.flip("diagonal")


@pytest.mark.parametrize("k", [1, 2, 3, -1])
def test_rotation_keeps_each_pixels_coordinates(k):
    image = ramp((4, 6), pixel_map=PixelMap.linear((4, 6), (1.0, -2.0), unit="deg"))
    turned = image.rotate90(k)
    np.testing.assert_array_equal(turned.values, np.rot90(image.values, k))
    h, w = image.shape
    old = {1: lambda i, j: (j, w - 1 - i), 2: lambda i, j: (h - 1 - i, w - 1 - j),
           3: lambda i, j: (h - 1 - j, i)}[k % 4]
    coordinates_follow_pixels(image, turned, old)
    assert turned.pixel_map.separable == (k % 4 == 2)
    np.testing.assert_array_equal(turned.rotate90(-k).values, image.values)


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
@pytest.mark.parametrize("change", [
    lambda im: im.rotate90(1), lambda im: im.rotate90(2), lambda im: im.rotate90(3),
    lambda im: im.flip(), lambda im: im.flip("vertical"),
])
def test_rearranging_a_mosaic_commutes_with_demosaicing(pattern, change):
    raw = mosaic((8, 10), pattern)
    np.testing.assert_allclose(change(raw).demosaic().values, change(raw.demosaic()).values)


def test_a_rearranged_mosaic_has_the_right_pattern():
    raw = mosaic((8, 10), "RGGB")
    assert raw.rotate90().mosaic.pattern == "GBRG"
    assert raw.flip().mosaic.pattern == "GRBG"
    assert raw.flip().mosaic.black_level == (2, 1, 4, 3)
    odd = mosaic((8, 9), "RGGB")
    assert odd.flip().mosaic.pattern == "RGGB"  # odd width: last column is R/G
    with pytest.raises(ValueError, match="demosaic first"):
        raw[:, ::2]


# resolution
def test_bin_averages_blocks_and_their_coordinates():
    image = ramp((6, 9), channels=True, pixel_map=PixelMap.linear((6, 9), 1.0, unit="mm"))
    binned = image.bin(3)
    assert binned.shape == (2, 3) and binned.channels == ("R", "G", "B")
    np.testing.assert_allclose(binned.values[1, 2], image.values[3:6, 6:9].mean(axis=(0, 1)))
    np.testing.assert_allclose(binned.pixel_map.x, [-3, 0, 3])
    np.testing.assert_allclose(binned.pixel_map.y, [-1.5, 1.5])
    summed = image.bin((2, 3), method="sum")
    assert summed.shape == (3, 3)
    np.testing.assert_allclose(summed.values[0, 0], image.values[0:2, 0:3].sum(axis=(0, 1)))
    assert image.bin(4).shape == (1, 2)  # partial blocks dropped
    with pytest.raises(ValueError, match="positive integers"):
        image.bin(1.5)
    with pytest.raises(ValueError, match="demosaic it first"):
        mosaic().bin(2)


def test_resize_keeps_the_field():
    image = ramp((6, 8), pixel_map=PixelMap.linear((6, 8), 1.0, unit="mm"))
    half = image.resize((3, 4))
    np.testing.assert_allclose(half.values, image.bin(2).values)  # exact for a linear ramp
    np.testing.assert_allclose(half.pixel_map.x, image.bin(2).pixel_map.x)
    np.testing.assert_allclose(half.pixel_map.y, image.bin(2).pixel_map.y)
    double = image.resize((12, 16))
    assert double.shape == (12, 16)
    centres = (np.arange(16) + 0.5) / 2 - 0.5, (np.arange(12) + 0.5) / 2 - 0.5
    cols, rows = np.meshgrid(*centres)
    inside = (slice(1, -1), slice(1, -1))  # beyond the outer pixel centres: edge values
    np.testing.assert_allclose(double.values[inside], (2 * cols + 3 * rows + 1)[inside])
    np.testing.assert_allclose(double.pixel_map.x, centres[0] - 3.5)


def test_interpolate_onto_another_grid():
    source = PixelMap.linear((6, 8), (0.5, -0.5), unit="deg")  # y up
    image = ramp((6, 8), pixel_map=source)
    target = PixelMap(np.array([-1.0, -0.25, 0.5]), np.array([1.0, 0.0]), "deg")
    resampled = image.interpolate(target)
    assert resampled.shape == (2, 3) and resampled.pixel_map == target
    cols, rows = source.pixel_of(*target.grid())
    np.testing.assert_allclose(resampled.values, 2 * cols + 3 * rows + 1)
    arcmin = PixelMap(target.x * 60, target.y * 60, "arcmin")
    np.testing.assert_allclose(image.interpolate(arcmin).values, resampled.values)
    beyond = image.interpolate(PixelMap(np.array([0.0, 50.0]), np.array([0.0]), "deg"))
    assert np.isfinite(beyond.values[0, 0]) and np.isnan(beyond.values[0, 1])
    with pytest.raises(ValueError, match="no pixel map"):
        ramp().interpolate(target)


# values
def test_smoothing_skips_nan_pixels():
    values = np.full((7, 7), 5.0)
    values[3, 3] = np.nan
    for method, kwargs in [("gaussian", {"sigma": 1.5}), ("box", {"size": 3})]:
        smoothed = Image(values).smooth(method, **kwargs)
        assert np.isnan(smoothed.values[3, 3])
        np.testing.assert_allclose(np.delete(smoothed.values.ravel(), 24), 5.0)
    with pytest.raises(ValueError, match="finite"):
        Image(values).smooth("median")
    spiky = np.zeros((5, 5))
    spiky[2, 2] = 100
    np.testing.assert_array_equal(Image(spiky).smooth("median", size=3).values, 0)
    with pytest.raises(TypeError, match="unexpected"):
        Image(spiky).smooth("box", sigma=2)
    with pytest.raises(ValueError, match="demosaic it first"):
        mosaic().smooth()


def test_smoothing_is_per_channel():
    image = ramp(channels=True)
    smoothed = image.smooth("gaussian", sigma=1.0)
    np.testing.assert_allclose(smoothed.values[:, :, 1], 10 * smoothed.values[:, :, 0])


def test_normalize():
    image = ramp(channels=True)
    assert np.nanmax(image.normalize().values) == pytest.approx(1.0)
    centre = Rectangle(2, 2, 4, 4)
    balanced = image.normalize("mean", roi=centre, per_channel=True)
    np.testing.assert_allclose(balanced.stats(roi=centre).mean, [1, 1, 1])
    assert balanced.value_unit == image.value_unit
    with pytest.raises(ValueError, match="zero"):
        Image(np.zeros((2, 2))).normalize()


def test_scale_channels():
    image = ramp(channels=True)
    scaled = image.scale_channels({"R": 2.0, "B": 0.5})
    np.testing.assert_allclose(scaled.values, image.values * [2.0, 1.0, 0.5])
    np.testing.assert_allclose(image.scale_channels([1, 2, 3]).values, image.values * [1, 2, 3])
    with pytest.raises(KeyError):
        image.scale_channels({"X": 1})
    with pytest.raises(ValueError, match="no channels"):
        ramp().scale_channels([1])


# line cuts
def test_a_line_cut_samples_along_a_line():
    image = ramp((10, 12), channels=True)
    across = image.line((1, 4), (9, 4))
    assert isinstance(across, LineCut) and len(across) == 9
    np.testing.assert_allclose(across.values[:, 0], 2 * np.arange(1, 10) + 3 * 4 + 1)
    np.testing.assert_allclose(across.distance, np.arange(9))
    assert across.distance_unit == parse("px") and across.channels == ("R", "G", "B")
    banded = image.line((1, 4), (9, 4), width=5)
    np.testing.assert_allclose(banded.values, across.values)  # linear across the band
    diagonal = image.channel("R").line((0, 0), (9, 9), num=4)
    np.testing.assert_allclose(diagonal.values, 5 * np.array([0, 3, 6, 9]) + 1)
    np.testing.assert_allclose(diagonal.distance[-1], 9 * np.sqrt(2))
    assert across.channel("G").values.ndim == 1


def test_line_cuts_in_mapped_coordinates():
    image = ramp((11, 11), pixel_map=PixelMap.linear((11, 11), 0.2, unit="deg"))
    spot = image.line((Q_(-60, "arcmin"), 0), (1, 0), coords=True)
    np.testing.assert_allclose(spot.x, np.linspace(-1, 1, 11))
    np.testing.assert_allclose(spot.distance, np.linspace(0, 2, 11))
    assert spot.distance_unit == parse("deg")
    np.testing.assert_allclose(spot.values, 2 * np.arange(11) + 3 * 5 + 1)
    np.testing.assert_allclose(image.row(5).values, spot.values)
    np.testing.assert_allclose(image.column(0).values, 3 * np.arange(11) + 1)


def test_pixel_map_lookups():
    up = PixelMap.linear((5, 5), (1.0, -1.0))
    cols, rows = up.pixel_of([-2.0, 0.0, 3.0], [2.0, 0.0, -3.0])
    np.testing.assert_allclose(cols, [0, 2, 5])
    np.testing.assert_allclose(rows, [0, 2, 5])  # beyond the edge: extrapolated
    x, y = up.at(np.array([0.5]), np.array([4.5]))
    np.testing.assert_allclose([x[0], y[0]], [-1.5, -2.5])
    with pytest.raises(ValueError, match="separable"):
        PixelMap.from_function((2, 2), lambda c, r: (c + r, r)).pixel_of(0, 0)


def test_line_cuts_are_labelled_by_where_they_are():
    from gsmath import Collection

    image = ramp((11, 11), pixel_map=PixelMap.linear((11, 11), 0.2, unit="deg")).copy(
        label="panel")
    assert image.row(5).label == "panel row 5" and image.column(2).label == "panel column 2"
    assert image.line((1, 4), (9, 4)).label == "panel (1, 4)-(9, 4)"
    assert image.line((1, 4), (9, 4), width=3).label == "panel (1, 4)-(9, 4), 3 px wide"
    assert image.line((-1, 0), (1, 0), coords=True).label == "panel (-1, 0)-(1, 0) deg"
    assert image.line((1, 4), (9, 4), label="across").label == "across"
    cuts = Collection([image.row(i) for i in (2, 5, 8)])  # unique labels: a collection
    assert cuts.labels == ("panel row 2", "panel row 5", "panel row 8")
