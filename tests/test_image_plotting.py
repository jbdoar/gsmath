import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gsmath import Image, PixelMap
from gsmath.image_plotting import DIVERGING, SEQUENTIAL


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_a_magnitude_map_in_map_units():
    image = Image(np.arange(12.0).reshape(3, 4), label="luminance", value_unit="cd/m^2",
                  channels="Y", pixel_map=PixelMap.linear((3, 4), (0.5, -0.5), unit="deg"))
    ax = image.plot()
    assert ax.get_xlabel() == "x (deg)" and ax.get_title() == "luminance"
    artist = ax.images[0]
    assert artist.cmap is SEQUENTIAL
    np.testing.assert_allclose(artist.get_extent(), [-1.0, 1.0, -0.75, 0.75])  # y up
    assert ax.figure.axes[1].get_ylabel() == "Y (cd/m²)"


def test_signed_data_is_diverging_about_zero():
    ax = Image(np.array([[-1.0, 0.2], [0.5, 3.0]])).plot()
    norm = ax.images[0].norm
    assert ax.images[0].cmap is DIVERGING
    assert (norm.vmin, norm.vcenter, norm.vmax) == (-3.0, 0.0, 3.0)


def test_noise_below_zero_stays_sequential():
    frame = np.full((4, 4), 200.0)
    frame[0, :2] = [-3.0, 0.5]  # dark-subtracted noise
    assert Image(frame).plot().images[0].cmap is SEQUENTIAL


def test_pixel_axes_and_meshes():
    image = Image(np.ones((3, 4)), pixel_map=PixelMap.linear((3, 4), 1.0, unit="mm"))
    ax = image.plot(coords=False)
    assert ax.get_xlabel() == "x (px)"
    np.testing.assert_allclose(ax.images[0].get_extent(), [-0.5, 3.5, 2.5, -0.5])
    mesh = image.rotate90().plot()  # per-pixel map: a mesh
    assert len(mesh.collections) == 1 and not mesh.images


def test_colour_images():
    rgb = Image(np.random.default_rng(0).uniform(0, 2, (4, 5, 3)), channels=("R", "G", "B"))
    shown = rgb.plot(srgb=True).images[0].get_array()
    assert shown.shape == (4, 5, 3) and shown.max() <= 1.0
    with pytest.raises(ValueError, match="choose one"):
        Image(np.ones((2, 2, 3)), channels=("X", "Y", "Z")).plot()
    assert Image(np.ones((2, 2, 3)), channels=("X", "Y", "Z")).plot(channel="Y").images
    with pytest.raises(ValueError, match="demosaic"):
        Image(np.ones((4, 4)), mosaic="RGGB").plot()


def test_channel_lines_have_identity_colours_and_a_second_cue():
    from matplotlib.colors import to_hex

    from gsmath.image_plotting import CATEGORICAL, channel_colours

    assert channel_colours(("R", "G", "B")) == ["#e34948", "#008300", "#2a78d6"]
    assert channel_colours(("X", "Y", "Z")) == list(CATEGORICAL[:3])  # fixed order
    with pytest.raises(ValueError, match="told apart"):
        channel_colours([str(i) for i in range(9)])
    rgb = Image(np.random.default_rng(0).uniform(0, 1, (6, 8, 3)), channels=("R", "G", "B"))
    lines = rgb.line((0, 3), (7, 3)).plot().get_lines()
    assert [to_hex(line.get_color()) for line in lines] == ["#e34948", "#008300", "#2a78d6"]
    assert len({line.get_linestyle() for line in lines}) == 3
