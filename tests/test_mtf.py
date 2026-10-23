import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import ndtr

from gsmath import EdgeMTF, Image, PixelMap, Rectangle
from gsmath.units import Q_, parse


def edge(shape=(80, 64), angle=5.0, sigma=0.7, *, low=20.0, high=200.0, shift=0.0,
         supersample=16, noise=0.0, seed=1):
    """A straight edge through the centre, tilted `angle` degrees from
    vertical (moving right going down), blurred by a Gaussian of `sigma` px
    and integrated over each pixel's area -- so its MTF is exactly
    exp(-2 pi^2 sigma^2 f^2) |sinc f|."""
    rows, cols = shape
    t = np.radians(angle)
    normal = np.array([np.cos(t), -np.sin(t)])
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    r, c = np.mgrid[0:rows, 0:cols].astype(float)
    total = np.zeros(shape)
    for dy in offsets:
        for dx in offsets:
            d = (c + dx - (cols - 1) / 2 - shift) * normal[0] + (r + dy - (rows - 1) / 2) * normal[1]
            total += low + (high - low) * ndtr(d / sigma)
    return total / supersample ** 2 + np.random.default_rng(seed).normal(0, noise, shape)


def true_mtf(f, sigma):
    return np.exp(-2 * np.pi ** 2 * sigma ** 2 * f ** 2) * np.abs(np.sinc(f))


def true_mtf50(sigma):
    return brentq(lambda f: true_mtf(f, sigma) - 0.5, 0.01, 1.0)


@pytest.mark.parametrize("angle, sigma", [(5, 0.7), (3, 1.0), (8, 0.5), (-5, 0.7)])
def test_matches_the_analytic_mtf(angle, sigma):
    result = Image(edge(angle=angle, sigma=sigma)).edge_mtf()
    assert isinstance(result, EdgeMTF) and result.orientation == "vertical"
    assert result.angle == pytest.approx(angle, abs=0.01)
    shown = result.frequency <= 0.75
    np.testing.assert_allclose(result.mtf[shown], true_mtf(result.frequency[shown], sigma),
                               atol=0.015)
    assert result.mtf50 == pytest.approx(true_mtf50(sigma), rel=0.01)
    assert result.mtf[0] == pytest.approx(1.0)
    assert result.metadata["empty_bins"] == 0


def test_horizontal_edges_and_either_polarity():
    vertical = Image(edge()).edge_mtf()
    horizontal = Image(edge().T).edge_mtf()
    assert horizontal.orientation == "horizontal"
    np.testing.assert_allclose(horizontal.mtf50, vertical.mtf50, rtol=1e-6)
    reversed_ = Image(edge()[:, ::-1]).edge_mtf()  # light to dark
    assert reversed_.mtf50 == pytest.approx(vertical.mtf50, rel=1e-3)
    assert reversed_.angle == pytest.approx(-5.0, abs=0.01)


def test_noise_and_a_region_of_a_larger_frame():
    frame = np.full((200, 240), 20.0)
    frame[60:140, 100:164] = edge(noise=1.0)
    region = Rectangle(100, 60, 164, 140)
    result = Image(frame).edge_mtf(region)
    assert result.mtf50 == pytest.approx(true_mtf50(0.7), rel=0.03)


def test_colour_channels_each_get_their_own_curve_and_shift():
    rgb = np.stack([edge(sigma=1.0, shift=0.3), edge(sigma=0.7), edge(sigma=0.5, shift=-0.2)],
                   axis=-1)
    result = Image(rgb, channels=("R", "G", "B")).edge_mtf()
    assert result.mtf.shape[1] == 3 and result.channels == ("R", "G", "B")
    np.testing.assert_allclose(result.mtf50, [true_mtf50(s) for s in (1.0, 0.7, 0.5)], rtol=0.02)
    np.testing.assert_allclose(result.offset, [0.0, -0.3, -0.5], atol=0.02)
    assert set(result.summary) == {"R", "G", "B"}
    assert set(result.summary["G"]) == {"mtf50", "mtf10", "angle", "offset"}
    curve = result.curve  # the same curves as a Series
    assert curve.channels == ("R", "G", "B") and curve.x_unit == parse("1/px")
    np.testing.assert_allclose(curve["B"].values, result.mtf[:, 2])


def test_frequencies_in_pixel_map_units():
    image = Image(edge(), pixel_map=PixelMap.linear((80, 64), 0.02, unit="deg"))
    result = image.edge_mtf()
    assert result.pitch == pytest.approx(0.02)
    per_degree = result.frequency_in()
    np.testing.assert_allclose(per_degree.magnitude, result.frequency / 0.02)
    assert str(per_degree.units) == "1 / degree"
    per_arcmin = result.frequency_in("1/arcmin")
    np.testing.assert_allclose(per_arcmin.magnitude, result.frequency / 0.02 / 60)
    with pytest.raises(ValueError, match="no pixel map"):
        Image(edge()).edge_mtf().frequency_in()


def test_mtf10_at_and_plot():
    result = Image(edge(sigma=0.7)).edge_mtf()
    expected = brentq(lambda f: true_mtf(f, 0.7) - 0.1, 0.01, 1.0)
    assert result.mtf10 == pytest.approx(expected, rel=0.02)
    assert result.at(result.mtf50) == pytest.approx(0.5, abs=1e-3)
    ax = result.plot()
    assert ax.get_xlabel() == "Frequency (cycles/px)"
    mapped = Image(edge(), pixel_map=PixelMap.linear((80, 64), 0.02, unit="deg")).edge_mtf()
    assert mapped.plot(per="1/deg").get_xlabel() == "Frequency (1/deg)"


def test_unusable_edges_are_explained():
    with pytest.raises(ValueError, match="too close to vertical"):
        Image(edge(angle=0.2)).edge_mtf()
    with pytest.raises(ValueError, match="no edge"):
        Image(np.full((40, 40), 5.0)).edge_mtf()
    with pytest.raises(ValueError, match="non-finite"):
        Image(np.where(np.eye(80, 64) > 0, np.nan, edge())).edge_mtf()
    with pytest.raises(ValueError, match="demosaic"):
        Image(edge(), mosaic="RGGB").edge_mtf()
