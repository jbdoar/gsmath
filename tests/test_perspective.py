import numpy as np
import pytest

from gsmath import Image, PixelMap, Quad, homography
from gsmath.perspective import otsu_threshold, project
from gsmath.units import Q_, parse

KEYSTONED = [(52.3, 40.7), (251.6, 55.2), (243.1, 190.4), (60.8, 171.9)]  # TL, TR, BR, BL


def content(u, v):
    """What the display shows, by position across (u) and down (v) it."""
    return 200 + 40 * u - 30 * v


def render(corners, shape=(210, 300), *, noise=0.0, supersample=6, distort=0.0, channels=False):
    """A camera image of the display, whose active area projects to `corners`,
    anti-aliased by supersampling; `distort` adds barrel distortion."""
    rows, cols = shape
    to_display = homography(corners, [(0, 0), (1, 0), (1, 1), (0, 1)])
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    r, c = np.mgrid[0:rows, 0:cols].astype(float)
    total = np.zeros(shape)
    for dy in offsets:
        for dx in offsets:
            x, y = c + dx, r + dy
            if distort:
                cx, cy = (cols - 1) / 2, (rows - 1) / 2
                scale = 1 + distort * ((x - cx) ** 2 + (y - cy) ** 2) / (cols / 2) ** 2
                x, y = cx + (x - cx) * scale, cy + (y - cy) * scale
            u, v = project(to_display, x, y)
            inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
            total += np.where(inside, content(u, v), 2.0)
    values = total / supersample ** 2 + np.random.default_rng(0).normal(0, noise, shape)
    if channels:
        values = np.stack([values * 0.9, values, values * 1.1], axis=-1)
        return Image(values, value_unit="cd/m^2", channels=("X", "Y", "Z"))
    return Image(values, value_unit="cd/m^2")


def test_homography_maps_the_four_points():
    source = [(0, 0), (10, 0), (10, 5), (0, 5)]
    H = homography(source, KEYSTONED)
    x, y = project(H, *np.array(source, dtype=float).T)
    np.testing.assert_allclose(np.column_stack([x, y]), KEYSTONED, atol=1e-9)
    back = homography(KEYSTONED, source)
    np.testing.assert_allclose(project(back, *project(H, 3.0, 2.0)), (3.0, 2.0))
    with pytest.raises(ValueError, match="collinear"):
        homography([(0, 0), (1, 1), (2, 2), (0, 1)], source)


@pytest.mark.parametrize("corners", [
    KEYSTONED,
    [(30.5, 30.5), (270.5, 30.5), (270.5, 170.5), (30.5, 170.5)],  # square-on
    [(80.2, 20.9), (230.4, 44.1), (262.7, 185.3), (41.6, 160.2)],  # rotated
])
def test_finds_corners_to_a_fraction_of_a_pixel(corners):
    quad = render(corners, noise=1.0).find_display()
    assert isinstance(quad, Quad)
    np.testing.assert_allclose(quad.corners, corners, atol=0.25)
    assert quad.edge_rms == pytest.approx(0.55, abs=0.1)


def test_curved_edges_show_in_the_fit_residual():
    straight = render(KEYSTONED).find_display().edge_rms
    curved = render(KEYSTONED, distort=0.3).find_display().edge_rms  # edges bow ~2 px
    assert curved > 1.5 * straight and curved > 0.9


def test_rectify_recovers_the_displayed_content():
    image = render(KEYSTONED)
    flat = image.rectify(image.find_display(), (100, 150))
    assert flat.shape == (100, 150)
    u, v = np.meshgrid((np.arange(150) + 0.5) / 150, (np.arange(100) + 0.5) / 100)
    np.testing.assert_allclose(flat.values[2:-2, 2:-2], content(u, v)[2:-2, 2:-2], atol=0.5)
    assert flat.metadata["rectified"]["shape"] == [100, 150]


def test_rectify_defaults_and_display_units():
    image = render(KEYSTONED, channels=True)
    quad = image.find_display()
    width, height = quad.size
    flat = image.rectify(quad)
    assert flat.shape == (round(height), round(width)) and flat.channels == ("X", "Y", "Z")
    sized = image.rectify(quad, (100, 160), size=(Q_(344, "mm"), Q_(21.5, "cm")))
    assert sized.pixel_map.unit == parse("mm")
    assert sized.pixel_map.x[-1] - sized.pixel_map.x[0] == pytest.approx(344 * 159 / 160)
    assert sized.pixel_map.y[-1] - sized.pixel_map.y[0] == pytest.approx(215 * 99 / 100)
    with pytest.raises(ValueError, match="two numbers or two quantities"):
        image.rectify(quad, size=(Q_(344, "mm"), 215))


def test_rectify_keeps_source_coordinates_by_default():
    image = render(KEYSTONED).copy(pixel_map=PixelMap.linear((210, 300), 0.1, unit="deg"))
    quad = image.find_display()
    flat = image.rectify(quad, (40, 60))
    assert not flat.pixel_map.separable and flat.pixel_map.unit == parse("deg")
    frame = [(-0.5, -0.5), (59.5, -0.5), (59.5, 39.5), (-0.5, 39.5)]
    source = project(homography(frame, quad.corners), 0.0, 0.0)  # first pixel's source
    expected = image.pixel_map.at(np.array([source[0]]), np.array([source[1]]))
    np.testing.assert_allclose([flat.pixel_map.x[0, 0], flat.pixel_map.y[0, 0]],
                               np.ravel(expected), atol=1e-9)


def test_a_quad_in_mapped_coordinates():
    image = render(KEYSTONED).copy(pixel_map=PixelMap.linear((210, 300), 0.1, unit="deg"))
    corners = image.find_display().corners
    x, y = image.pixel_map.at(corners[:, 0], corners[:, 1])
    mapped = Quad(tuple(zip(x, y)), coords=True)
    np.testing.assert_allclose(image.rectify(mapped, (50, 80)).values,
                               image.rectify(image.find_display(), (50, 80)).values)


def test_a_quad_is_a_region():
    image = render(KEYSTONED)
    quad = image.find_display()
    inside = image.stats(roi=quad)
    assert inside.percentile(1) > 150  # display pixels (a corner pixel can be part-lit)
    (ax, ay), (bx, by) = quad.corners[2] - quad.corners[0], quad.corners[3] - quad.corners[1]
    area = 0.5 * abs(ax * by - ay * bx)  # half the diagonals' cross product
    assert quad.mask(image).sum() == pytest.approx(area, rel=0.02)
    with pytest.raises(ValueError, match="four corners"):
        Quad(((0, 0), (1, 0), (1, 1)))


def test_detection_failures_are_explained():
    with pytest.raises(ValueError, match="all the same"):
        Image(np.ones((50, 50))).find_display()
    speck = np.zeros((50, 50))
    speck[10:13, 10:13] = 100
    with pytest.raises(ValueError, match="too small"):
        Image(speck).find_display()
    with pytest.raises(ValueError, match="demosaic"):
        Image(np.ones((4, 4)), mosaic="RGGB").find_display()


def test_otsu_splits_two_populations():
    values = np.concatenate([np.full(500, 10.0), np.full(300, 200.0)])
    assert 10 < otsu_threshold(values) < 200
