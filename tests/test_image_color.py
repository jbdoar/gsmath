import numpy as np
import pytest

from gsmath import (
    ILLUMINANTS, Calibration, ColorMatrix, Demosaic, Image, ImageColor, Measurement, PixelMap,
    Rectangle, Spectrum, TristimulusColor, fit_color_matrix, xyz_to_lab, xyz_to_luv,
)
from gsmath.analysis.planckian import cct_robertson, cct_robertson_array
from gsmath.mosaic import site_masks
from gsmath.units import Q_, parse

NAMES = ["A", "D65", "F11", "D50", "C", "F2"]


@pytest.fixture(scope="module")
def spectra():
    return [Spectrum(ILLUMINANTS[n].wavelength, ILLUMINANTS[n].values, label=n) for n in NAMES]


@pytest.fixture(scope="module")
def xyz_image(spectra):
    """A 2x3 X, Y, Z image whose pixels are the illuminants' tristimulus values."""
    XYZ = np.array([s.color(white="D65").XYZ for s in spectra]).reshape(2, 3, 3)
    return Image(XYZ, label="illuminants", channels=("X", "Y", "Z"),
                 pixel_map=PixelMap.linear((2, 3), 1.0, unit="deg"))


def test_every_pixel_matches_the_spectral_colour(spectra, xyz_image):
    color = xyz_image.color(white="D65")
    for k, spectrum in enumerate(spectra):
        i, j = divmod(k, 3)
        reference = spectrum.color(white="D65")
        np.testing.assert_allclose(color.xy.values[i, j], reference.xy)
        np.testing.assert_allclose(color.uv.values[i, j], reference.uv)
        np.testing.assert_allclose(color.uv_prime.values[i, j], reference.uv_prime)
        np.testing.assert_allclose(color.Lab.values[i, j], reference.Lab)
        np.testing.assert_allclose(color.Luv.values[i, j], reference.Luv)
        assert color.cct.values[i, j] == pytest.approx(reference.cct)
        assert color.duv.values[i, j] == pytest.approx(reference.duv, abs=1e-12)
        assert color.delta_e_2000(reference).values[i, j] == pytest.approx(0, abs=1e-9)


def test_colour_maps_are_images_on_the_same_pixels(xyz_image):
    color = xyz_image.color()
    assert isinstance(color, ImageColor) and xyz_image.color() is color  # cached
    assert color.uv_prime.channels == ("u'", "v'") and color.Lab.channels == ("L*", "a*", "b*")
    assert color.cct.value_unit == parse("K") and color.cct.channels is None
    for unitless in (color.xy, color.uv, color.uv_prime, color.Lab, color.Luv, color.duv,
                     color.delta_e_2000([50, 0, 0])):
        assert unitless.value_unit is None
    assert color.xy.pixel_map == xyz_image.pixel_map
    assert color.Y.channels == ("Y",)
    np.testing.assert_allclose(color.cct.stats(roi=Rectangle(0, 0, 1, 1)).mean,
                               color.cct.values[0, 0])


def test_a_regions_mean_colour(spectra, xyz_image):
    left = Rectangle(0, 0, 1, 2)  # column 0: A and D50
    mean = xyz_image.color(white="D65").mean(roi=left)
    assert isinstance(mean, TristimulusColor)
    XYZ = (spectra[0].color().XYZ + spectra[3].color().XYZ) / 2
    np.testing.assert_allclose(mean.XYZ, XYZ)
    np.testing.assert_allclose(mean.xy, XYZ[:2] / XYZ.sum())
    assert mean.cct == pytest.approx(cct_robertson(*mean.uv).cct)
    assert set(mean.summary) >= {"x", "y", "u_prime", "v_prime", "cct", "duv", "L_star"}
    assert mean.delta_e_2000(mean) == pytest.approx(0)


def test_whites_must_match_for_colour_differences(xyz_image, spectra):
    d65 = xyz_image.color(white="D65")
    with pytest.raises(ValueError, match="matching reference whites"):
        d65.delta_e_2000(xyz_image.color(white="D50"))
    with pytest.raises(ValueError, match="matching reference whites"):
        d65.delta_e_2000(spectra[0].color(white="A"))
    same = d65.delta_e_2000(xyz_image.color(white="D65"))
    np.testing.assert_allclose(same.values, 0, atol=1e-9)
    grey = TristimulusColor(_xyz_for_L(50, d65.white_XYZ), d65.white_XYZ)  # Lab (50, 0, 0)
    np.testing.assert_allclose(d65.delta_e_2000([50, 0, 0]).values,
                               d65.delta_e_2000(grey).values, atol=1e-6)


def _xyz_for_L(L, white):
    """XYZ of the neutral with lightness L* under `white`."""
    fy = (L + 16) / 116
    return white * (fy ** 3 if fy > 6 / 29 else 3 * (6 / 29) ** 2 * (fy - 4 / 29))


def test_an_absolute_white_on_the_images_scale():
    white = np.array([190.1, 200.0, 217.8])  # a display's white, cd/m^2
    image = Image(np.broadcast_to(white, (2, 2, 3)), value_unit="cd/m^2",
                  channels=("X", "Y", "Z"))
    color = image.color(white=white)
    np.testing.assert_allclose(color.Lab.values, [[[100, 0, 0]] * 2] * 2, atol=1e-9)
    assert color.Y.value_unit == parse("cd/m^2") and color.xy.value_unit is None
    assert color.mean().XYZ_unit == parse("cd/m^2")


def test_colour_needs_xyz_channels():
    with pytest.raises(ValueError, match="isn't a tristimulus image"):
        Image(np.ones((2, 2, 3)), channels=("R", "G", "B")).color()


def test_vectorised_helpers_agree_with_one_triple(xyz_image):
    XYZ = xyz_image.values
    white = [95.047, 100.0, 108.883]
    for converter in (xyz_to_lab, xyz_to_luv):
        mapped = converter(XYZ, white)
        for i, j in np.ndindex(2, 3):
            np.testing.assert_allclose(mapped[i, j], converter(XYZ[i, j], white))
    edge = np.array([[0, 0, 0], [np.nan, 1, 1], [1, 0, 0]], dtype=float)
    np.testing.assert_array_equal(xyz_to_luv(edge, white)[0], [0, 0, 0])
    assert np.all(np.isnan(xyz_to_lab(edge, white)[1]))
    cct, duv = cct_robertson_array([0.2, np.nan], [0.3, 0.3])
    assert cct[0] == cct_robertson(0.2, 0.3).cct and np.isnan(cct[1]) and np.isnan(duv[1])


# colour matrices
CAMERA = np.array([[0.9, 0.3, 0.1], [0.2, 1.1, 0.2], [0.05, 0.2, 1.3]])  # XYZ -> camera RGB


def test_fit_color_matrix_recovers_a_camera(spectra):
    XYZ = np.array([s.color().XYZ for s in spectra])
    rgb = XYZ @ CAMERA.T
    matrix = fit_color_matrix(rgb, XYZ)
    np.testing.assert_allclose(matrix, np.linalg.inv(CAMERA), atol=1e-10)
    with pytest.raises(ValueError, match="at least 3 patches"):
        fit_color_matrix(rgb[:2], XYZ[:2])
    with pytest.raises(ValueError, match="don't span"):
        fit_color_matrix(np.ones((4, 3)), XYZ[:4])


def test_apply_matrix_mixes_channels_with_units():
    rgb = Image(np.random.default_rng(0).uniform(0, 1, (3, 4, 3)), value_unit="count/ms",
                channels=("R", "G", "B"))
    matrix = np.linalg.inv(CAMERA)
    XYZ = rgb.apply_matrix(matrix, unit="cd/m^2 / (count/ms)")
    assert XYZ.channels == ("X", "Y", "Z")
    np.testing.assert_allclose(XYZ.values, rgb.values @ matrix.T)
    assert XYZ.to_value_unit("cd/m^2") is not None
    luminance = rgb.apply_matrix(matrix[1:2], channels=("Y",))
    assert luminance.values.ndim == 2
    with pytest.raises(ValueError, match="can't apply"):
        rgb.apply_matrix(np.eye(2), channels=("a", "b"))


def test_raw_to_colour_through_a_calibration(tmp_path):
    scene_XYZ = np.empty((8, 10, 3))
    scene_XYZ[:] = [95.0, 100.0, 108.9]
    camera_rgb = scene_XYZ @ CAMERA.T
    masks = site_masks("RGGB", (8, 10))
    raw = Image(sum(np.where(masks[c], camera_rgb[:, :, i], 0) for i, c in enumerate("RGB")),
                value_unit="count", mosaic="RGGB")
    pipeline = Calibration("colour camera", [
        Demosaic(), ColorMatrix(np.linalg.inv(CAMERA), unit="1/count")])
    XYZ = Measurement(raw=raw).process(pipeline)
    np.testing.assert_allclose(XYZ.values, scene_XYZ)
    assert str(XYZ.value_unit) == "dimensionless"
    np.testing.assert_allclose(XYZ.color().mean().xy, [0.3127, 0.3291], atol=2e-4)
    loaded = Calibration.read(pipeline.write(tmp_path / "colour.gsmath"))
    assert loaded.fingerprint == pipeline.fingerprint
    assert loaded.steps[1].channels == ("X", "Y", "Z") and loaded.steps[1].unit == parse("1/count")
    np.testing.assert_allclose(Measurement(raw=raw).process(loaded).values, scene_XYZ)
