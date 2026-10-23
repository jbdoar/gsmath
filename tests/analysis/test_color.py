# tests/analysis/test_color.py

import numpy as np
import pytest

from gsmath.analysis.color import ColorAnalysis, xyz_to_lab, xyz_to_luv
from gsmath import units
from gsmath.constants import K_M
from gsmath.reference.cmfs import CMFS
from gsmath.spectrum import Spectrum


@pytest.fixture
def spectrum():
    wavelength = np.arange(360.0, 831.0)
    values = np.ones_like(wavelength)

    return Spectrum(
        wavelength=wavelength,
        values=values,
        label="Test",
    )


@pytest.fixture
def color(spectrum):
    return ColorAnalysis(
        spectrum=spectrum,
        observer="2deg",
    )


def test_default_observer(spectrum):
    color = ColorAnalysis(spectrum)

    assert color.observer == "2deg"


def test_cmf(color):
    assert color.cmf is CMFS["2deg"]


def test_xyz(color):
    cmf = CMFS["2deg"]

    interpolated = np.interp(
        cmf.wavelength,
        color.spectrum.wavelength,
        color.spectrum.values,
    )

    expected = np.array([
        np.trapezoid(
            interpolated * cmf.xbar,
            cmf.wavelength,
        ),
        np.trapezoid(
            interpolated * cmf.ybar,
            cmf.wavelength,
        ),
        np.trapezoid(
            interpolated * cmf.zbar,
            cmf.wavelength,
        ),
    ])

    np.testing.assert_allclose(color.XYZ, expected)


def test_unitless_values_select_relative_colorimetry(color):
    assert color.effective_mode == "relative"
    assert color.XYZ_unit is None


def test_relative_colorimetry_is_wavelength_unit_invariant():
    values = [0.0, 1.0, 0.0]
    nm = Spectrum([400, 500, 600], values, wavelength_unit="nm")
    um = Spectrum([0.4, 0.5, 0.6], values, wavelength_unit="um")

    np.testing.assert_allclose(nm.color().XYZ, um.color().XYZ)


def test_interpolation_zero_fills_unmeasured_wavelengths():
    spectrum = Spectrum([500, 550, 600], [1.0, 1.0, 1.0])
    color = spectrum.color(mode="relative")
    expected_values = np.interp(
        color.cmf.wavelength,
        spectrum.wavelength,
        spectrum.values,
        left=0.0,
        right=0.0,
    )
    expected = np.trapezoid(
        expected_values[:, np.newaxis] * color.cmf.values,
        color.cmf.wavelength,
        axis=0,
    )

    np.testing.assert_allclose(color.XYZ, expected)


@pytest.mark.parametrize(
    ("value_unit", "expected_unit"),
    [
        ("W/nm", units.ureg.lm),
        ("W/sr/nm", units.ureg.cd),
        ("W/m^2/nm", units.ureg.lux),
        ("W/m^2/sr/nm", units.ureg.cd / units.ureg.m**2),
    ],
)
def test_auto_mode_recognizes_absolute_radiometric_inputs(
    value_unit,
    expected_unit,
):
    spectrum = Spectrum(
        [500, 550, 600],
        [1.0, 1.0, 1.0],
        value_unit=value_unit,
    )
    color = spectrum.color()

    assert color.effective_mode == "absolute"
    assert color.XYZ_unit == expected_unit
    assert color.unit("Y") == expected_unit
    assert color.quantity("Y").units == expected_unit
    assert color.quantity("XYZ").units == expected_unit


def test_absolute_radiance_uses_physical_unit_conversions():
    spectrum = Spectrum(
        [500, 550, 600],
        [1.0, 1.0, 1.0],
        value_unit="uW/cm^2/sr/nm",
    )
    relative = spectrum.color(mode="relative")
    absolute = spectrum.color(mode="absolute")

    assert absolute.Y == pytest.approx(relative.Y * K_M * 0.01)
    assert absolute.XYZ_unit == units.ureg.cd / units.ureg.m**2


def test_absolute_colorimetry_is_unit_invariant():
    nm = Spectrum(
        [400, 500, 600],
        [1.0, 2.0, 1.0],
        wavelength_unit="nm",
        value_unit="W/m^2/nm",
    )
    um = Spectrum(
        [0.4, 0.5, 0.6],
        [1000.0, 2000.0, 1000.0],
        wavelength_unit="um",
        value_unit="W/m^2/um",
    )

    np.testing.assert_allclose(nm.color().XYZ, um.color().XYZ)


def test_explicit_relative_mode_does_not_claim_photometric_units():
    spectrum = Spectrum(
        [500, 550, 600],
        [1.0, 1.0, 1.0],
        value_unit="W/m^2/nm",
    )
    color = spectrum.color(mode="relative")

    assert color.effective_mode == "relative"
    assert color.XYZ_unit is None


def test_cct_and_duv_expose_intrinsic_units(color):
    assert color.cct == color.planck.cct
    assert color.duv == color.planck.duv
    assert color.quantity("cct").units == units.ureg.kelvin
    assert color.quantity("duv").units == units.ureg.dimensionless


def test_xyz_to_lab_known_d65_red_example():
    result = xyz_to_lab(
        [41.24, 21.26, 1.93],
        [95.047, 100.0, 108.883],
    )

    np.testing.assert_allclose(
        result,
        [53.2329, 80.1093, 67.2201],
        atol=1e-4,
    )


def test_xyz_to_luv_known_d65_red_example():
    result = xyz_to_luv(
        [41.24, 21.26, 1.93],
        [95.047, 100.0, 108.883],
    )

    np.testing.assert_allclose(
        result,
        [53.2329, 175.0530, 37.7505],
        atol=1e-4,
    )


def test_reference_white_maps_to_neutral_lab_and_luv():
    white = [95.047, 100.0, 108.883]

    np.testing.assert_allclose(xyz_to_lab(white, white), [100, 0, 0])
    np.testing.assert_allclose(xyz_to_luv(white, white), [100, 0, 0])


def test_zero_xyz_maps_to_zero_lab_and_luv():
    white = [95.047, 100.0, 108.883]

    np.testing.assert_allclose(xyz_to_lab([0, 0, 0], white), [0, 0, 0])
    np.testing.assert_allclose(xyz_to_luv([0, 0, 0], white), [0, 0, 0])


def test_nonfinite_xyz_maps_to_nan_lab_and_luv():
    white = [95.047, 100.0, 108.883]

    assert np.all(np.isnan(xyz_to_lab([np.nan, 1, 1], white)))
    assert np.all(np.isnan(xyz_to_luv([np.nan, 1, 1], white)))


@pytest.mark.parametrize("converter", [xyz_to_lab, xyz_to_luv])
def test_perceptual_conversion_validates_xyz_and_white(converter):
    with pytest.raises(ValueError, match="three values"):
        converter([1, 2], [95.047, 100.0, 108.883])
    with pytest.raises(ValueError, match="finite and positive"):
        converter([1, 2, 3], [95.047, 0, 108.883])


def test_named_d65_white_depends_on_observer(spectrum):
    white_2deg = spectrum.color(observer="2deg").white_XYZ
    white_10deg = spectrum.color(observer="10deg").white_XYZ

    np.testing.assert_allclose(
        white_2deg,
        [95.047, 100.0, 108.883],
        atol=0.001,
    )
    np.testing.assert_allclose(
        white_10deg,
        [94.811, 100.0, 107.305],
        atol=0.001,
    )


def test_custom_xyz_reference_white_is_supported_and_cached(spectrum):
    color = spectrum.color(white=[95.047, 100.0, 108.883])

    np.testing.assert_allclose(color.white_XYZ, [95.047, 100.0, 108.883])
    assert color is spectrum.color(white=(95.047, 100.0, 108.883))


def test_unknown_named_reference_white_is_rejected(spectrum):
    with pytest.raises(ValueError, match="Unknown reference white"):
        spectrum.color(white="not-an-illuminant")


def test_lab_and_luv_properties_are_dimensionless(color):
    np.testing.assert_allclose(
        color.Lab,
        [color.L_star, color.a_star, color.b_star],
    )
    np.testing.assert_allclose(
        color.Luv,
        [color.L_star, color.u_star, color.v_star],
    )
    assert color.quantity("Lab").units == units.ureg.dimensionless
    assert color.quantity("Luv").units == units.ureg.dimensionless


def test_absolute_mode_rejects_nonradiometric_values():
    spectrum = Spectrum(
        [500, 550, 600],
        [1.0, 1.0, 1.0],
        value_unit="count/nm",
    )

    assert spectrum.color().effective_mode == "relative"
    with pytest.raises(ValueError, match="radiometric values"):
        spectrum.color(mode="absolute")


def test_colorimetry_rejects_unitless_wavelength_axis():
    spectrum = Spectrum(
        [500, 550, 600],
        [1.0, 1.0, 1.0],
        wavelength_unit=None,
    )

    with pytest.raises(ValueError, match="physical wavelength unit"):
        spectrum.color()


def test_colorimetry_rejects_unknown_mode(spectrum):
    with pytest.raises(ValueError, match="color mode"):
        spectrum.color(mode="unknown")


def test_xyz_scalar_properties(color):
    X, Y, Z = color.XYZ

    assert color.X == pytest.approx(X)
    assert color.Y == pytest.approx(Y)
    assert color.Z == pytest.approx(Z)


def test_xy(color):
    X, Y, Z = color.XYZ
    denominator = X + Y + Z

    expected = np.array([
        X / denominator,
        Y / denominator,
    ])

    np.testing.assert_allclose(color.xy, expected)


def test_xy_scalar_properties(color):
    x, y = color.xy

    assert color.x == pytest.approx(x)
    assert color.y == pytest.approx(y)


def test_xyY(color):
    np.testing.assert_allclose(
        color.xyY,
        [
            color.x,
            color.y,
            color.Y,
        ],
    )


def test_uv(color):
    denominator = (
        -2 * color.x
        + 12 * color.y
        + 3
    )

    expected = np.array([
        4 * color.x / denominator,
        6 * color.y / denominator,
    ])

    np.testing.assert_allclose(color.uv, expected)


def test_uv_scalar_properties(color):
    u, v = color.uv

    assert color.u == pytest.approx(u)
    assert color.v == pytest.approx(v)


def test_uv_prime(color):
    denominator = (
        -2 * color.x
        + 12 * color.y
        + 3
    )

    expected = np.array([
        4 * color.x / denominator,
        9 * color.y / denominator,
    ])

    np.testing.assert_allclose(
        color.uv_prime,
        expected,
    )


def test_uv_prime_scalar_properties(color):
    u_prime, v_prime = color.uv_prime

    assert color.u_prime == pytest.approx(u_prime)
    assert color.v_prime == pytest.approx(v_prime)


def test_summary_contains_only_scalar_leaves(color):
    summary = color.summary["2deg"]

    assert set(summary) == {
        "X",
        "Y",
        "Z",
        "x",
        "y",
        "u",
        "v",
        "u_prime",
        "v_prime",
        "planck.cct",
        "planck.duv",
        "L_star",
        "a_star",
        "b_star",
        "u_star",
        "v_star",
    }

    for value in summary.values():
        assert np.isscalar(value)


def test_summary_does_not_include_coordinate_arrays(color):
    summary = color.summary["2deg"]

    assert "XYZ" not in summary
    assert "xy" not in summary
    assert "xyY" not in summary
    assert "uv" not in summary
    assert "uv_prime" not in summary


def test_summary_uses_selected_observer(spectrum):
    color = ColorAnalysis(
        spectrum=spectrum,
        observer="10deg",
    )

    assert color.summary == {
        "10deg": {
            "X": pytest.approx(color.X),
            "Y": pytest.approx(color.Y),
            "Z": pytest.approx(color.Z),
            "x": pytest.approx(color.x),
            "y": pytest.approx(color.y),
            "u": pytest.approx(color.u),
            "v": pytest.approx(color.v),
            "u_prime": pytest.approx(color.u_prime),
            "v_prime": pytest.approx(color.v_prime),
            "planck.cct": pytest.approx(color.planck.cct),
            "planck.duv": pytest.approx(color.planck.duv),
            "L_star": pytest.approx(color.L_star),
            "a_star": pytest.approx(color.a_star),
            "b_star": pytest.approx(color.b_star),
            "u_star": pytest.approx(color.u_star),
            "v_star": pytest.approx(color.v_star),
        },
    }


def test_empty_spectrum_returns_nan_xyz():
    spectrum = Spectrum(
        wavelength=np.array([]),
        values=np.array([]),
        label="Empty",
    )

    color = ColorAnalysis(spectrum)

    assert np.all(np.isnan(color.XYZ))
    assert np.isnan(color.X)
    assert np.isnan(color.Y)
    assert np.isnan(color.Z)


def test_empty_spectrum_returns_nan_coordinates():
    spectrum = Spectrum(
        wavelength=np.array([]),
        values=np.array([]),
        label="Empty",
    )

    color = ColorAnalysis(spectrum)

    assert np.all(np.isnan(color.xy))
    assert np.all(np.isnan(color.xyY))
    assert np.all(np.isnan(color.uv))
    assert np.all(np.isnan(color.uv_prime))

    assert np.isnan(color.x)
    assert np.isnan(color.y)
    assert np.isnan(color.u)
    assert np.isnan(color.v)
    assert np.isnan(color.u_prime)
    assert np.isnan(color.v_prime)


def test_zero_spectrum_returns_nan_chromaticity():
    wavelength = np.arange(360.0, 831.0)

    spectrum = Spectrum(
        wavelength=wavelength,
        values=np.zeros_like(wavelength),
        label="Zero",
    )

    color = ColorAnalysis(spectrum)

    np.testing.assert_allclose(
        color.XYZ,
        [0.0, 0.0, 0.0],
        atol=1e-12,
    )

    assert np.all(np.isnan(color.xy))
    assert np.all(np.isnan(color.uv))
    assert np.all(np.isnan(color.uv_prime))
