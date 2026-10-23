import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gsmath import Image, Rectangle, Uniformity, grid, homography
from gsmath import xyz_to_uv_prime
from gsmath.perspective import project

WHITE = np.array([95.047, 100.0, 108.883])


def xyz_from(Y, x, y):
    """X, Y, Z (last axis) from luminance and chromaticity."""
    Y, x, y = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (Y, x, y)))
    return np.stack([x * Y / y, Y, (1 - x - y) * Y / y], axis=-1)


def display(u, v):
    """A display's luminance and chromaticity by position across (u) and down (v)."""
    r2 = (u - 0.5) ** 2 + (v - 0.5) ** 2
    return 250 * (1 - 0.6 * r2), 0.3127 + 0.02 * r2, 0.3290 - 0.015 * (u - 0.5)


def square_on(shape=(120, 200)):
    rows, cols = shape
    u, v = np.meshgrid((np.arange(cols) + 0.5) / cols, (np.arange(rows) + 0.5) / rows)
    return Image(xyz_from(*display(u, v)), label="square-on", value_unit="cd/m^2",
                 channels=("X", "Y", "Z"))


def keystoned(corners, shape=(210, 300), supersample=4):
    """The same display seen at an angle, its active area at `corners`."""
    rows, cols = shape
    to_display = homography(corners, [(0, 0), (1, 0), (1, 1), (0, 1)])
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    r, c = np.mgrid[0:rows, 0:cols].astype(float)
    total = np.zeros((*shape, 3))
    for dy in offsets:
        for dx in offsets:
            u, v = project(to_display, c + dx, r + dy)
            inside = ((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1))[..., None]
            total += np.where(inside, xyz_from(*display(np.clip(u, 0, 1), np.clip(v, 0, 1))),
                              xyz_from(0.5, 0.3127, 0.3290))
    return Image(total / supersample ** 2, label="keystoned", value_unit="cd/m^2",
                 channels=("X", "Y", "Z"))


def quadrants():
    """Four flat quadrants of known Y and chromaticity."""
    Y = np.array([[100.0, 80.0], [90.0, 120.0]])
    x = np.array([[0.31, 0.32], [0.30, 0.315]])
    y = np.array([[0.33, 0.34], [0.32, 0.33]])
    values = np.repeat(np.repeat(xyz_from(Y, x, y), 40, axis=0), 50, axis=1)
    return Image(values, channels=("X", "Y", "Z")), Y.ravel(), x.ravel(), y.ravel()


def test_grid_layouts():
    assert grid(1, 1) == [(0.5, 0.5)]
    assert grid(2, 3, inset=0.1) == [(0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
                                     (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)]
    assert grid(3, 3, inset=(0.0, 0.25))[3] == (0.0, 0.5)
    with pytest.raises(ValueError, match="inset"):
        grid(3, 3, inset=0.5)


def test_metrics_on_known_quadrants():
    image, Y, x, y = quadrants()
    result = image.uniformity(points=grid(2, 2, inset=0.25), spot=0.1)
    np.testing.assert_allclose([p["Y"] for p in result.points_table], Y)
    np.testing.assert_allclose([p["x"] for p in result.points_table], x)
    lum = result.luminance
    assert (lum["min"], lum["max"], lum["mean"]) == (80.0, 120.0, 97.5)
    assert lum["min_max_ratio"] == pytest.approx(80 / 120)
    assert lum["nonuniformity"] == pytest.approx(1 - 80 / 120)
    assert lum["min_mean_ratio"] == pytest.approx(80 / 97.5)
    assert lum["max_deviation_from_mean"] == pytest.approx(22.5 / 97.5)
    assert lum["cv"] == pytest.approx(np.std(Y) / 97.5)
    # all four spots are equally far from the middle: the first is the reference
    assert result.reference == 0 and lum["centre"] == 100.0
    assert lum["max_deviation_from_centre"] == pytest.approx(0.2)
    color = result.color
    assert color["x_range"] == pytest.approx(0.02) and color["y_range"] == pytest.approx(0.02)
    pairs = [np.hypot(x[i] - x[j], y[i] - y[j]) for i in range(4) for j in range(i + 1, 4)]
    assert color["max_delta_xy"] == pytest.approx(max(pairs))
    uv = xyz_to_uv_prime(xyz_from(Y, x, y))
    assert color["max_delta_uv_prime_from_centre"] == pytest.approx(
        np.max(np.hypot(*(uv - uv[0]).T)))
    assert set(result.summary) == {"points", "spot", "luminance", "color"}


def test_spots_measure_the_display_where_they_are():
    result = square_on().uniformity(points=grid(3, 3, inset=0.1), spot=0.04)
    for point in result.points_table:
        Y, x, y = display(point["u"], point["v"])
        assert point["Y"] == pytest.approx(Y, rel=2e-3)
        assert (point["x"], point["y"]) == pytest.approx((x, y), abs=2e-5)
    assert result.reference == 4 and result.luminance["centre"] == pytest.approx(250, rel=1e-3)


def test_a_keystoned_frame_measures_like_a_square_on_one():
    corners = [(52.3, 40.7), (251.6, 55.2), (243.1, 190.4), (60.8, 171.9)]
    frame = keystoned(corners)
    quad = frame.find_display()
    tilted = frame.uniformity(quad, points=grid(3, 3, inset=0.15), spot=0.06)
    straight = square_on().uniformity(points=grid(3, 3, inset=0.15), spot=0.06)
    np.testing.assert_allclose([p["Y"] for p in tilted.points_table],
                               [p["Y"] for p in straight.points_table], rtol=3e-3)
    np.testing.assert_allclose([p["u'"] for p in tilted.points_table],
                               [p["u'"] for p in straight.points_table], atol=2e-5)
    assert tilted.luminance["min_max_ratio"] == pytest.approx(
        straight.luminance["min_max_ratio"], abs=3e-3)
    col, row = tilted.to_pixels(0.0, 1.0)
    assert (col, row) == pytest.approx(tuple(quad.corners[3]), abs=1e-9)


def test_area_as_a_rectangle():
    frame = np.zeros((100, 140, 3))
    frame[20:80, 30:110] = square_on((60, 80)).values
    image = Image(frame, channels=("X", "Y", "Z"))
    boxed = image.uniformity(Rectangle(30, 20, 110, 80), spot=0.05)
    whole = square_on((60, 80)).uniformity(spot=0.05)
    np.testing.assert_allclose([p["Y"] for p in boxed.points_table],
                               [p["Y"] for p in whole.points_table])


def test_full_field():
    image = square_on()
    result = image.uniformity(spot=0.04)
    field = result.full_field()
    Y = image["Y"].values
    assert field.luminance["max"] == pytest.approx(Y.max())
    assert field.luminance["min_max_ratio"] == pytest.approx(Y.min() / Y.max())
    relative = field.relative_luminance
    assert relative.value_unit is None
    assert np.nanmax(relative.values) == pytest.approx(Y.max() / result.luminance["centre"])
    delta = field.delta_uv_prime
    assert np.nanmin(delta.values) < 1e-4 and field.color["max_delta_uv_prime_from_centre"] > 0
    trimmed = result.full_field(percentiles=(1, 99))
    assert trimmed.luminance["min"] == pytest.approx(np.percentile(Y, 1))
    assert trimmed.luminance["mean"] == pytest.approx(field.luminance["mean"])
    inset = result.full_field(inset=0.1)
    assert inset.luminance["min"] > field.luminance["min"]  # edges excluded
    assert np.isnan(inset.relative_luminance.values[0, 0])
    smoothed = result.full_field(sigma=2.0)
    assert set(smoothed.summary) == {"sigma", "inset", "percentiles", "luminance", "color"}


def test_luminance_only_images():
    Y = square_on()["Y"].values
    unnamed = Image(Y).uniformity(spot=0.05)
    assert unnamed.luminance["centre"] == pytest.approx(250, rel=1e-3)
    assert not unnamed.has_color
    with pytest.raises(ValueError, match="needs X, Y and Z"):
        unnamed.color
    assert "color" not in unnamed.summary


def test_bad_inputs_are_explained():
    image = square_on()
    with pytest.raises(ValueError, match="fractions of the active area"):
        image.uniformity(points=[(1.2, 0.5)])
    with pytest.raises(ValueError, match="make spot larger"):
        image.uniformity(spot=0.005).luminance
    with pytest.raises(ValueError, match="no Y channel"):
        Image(np.ones((10, 10, 3)), channels=("R", "G", "B")).uniformity()
    with pytest.raises(TypeError, match="active area"):
        image.uniformity("everything").luminance


def test_plot_and_table():
    result = square_on().uniformity(spot=0.06)
    ax = result.plot()
    assert len(ax.texts) == 9
    frame = result.points_frame
    assert list(frame.columns) == ["u", "v", "column", "row", "X", "Y", "Z", "x", "y", "u'", "v'"]
    assert isinstance(result, Uniformity)
