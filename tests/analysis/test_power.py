import numpy as np
import pytest

from gsmath import Spectrum, units


@pytest.fixture
def flat_spectrum():
    return Spectrum(
        wavelength=[400, 500, 600],
        values=[2.0, 2.0, 2.0],
        label="Flat",
    )


def test_integral(flat_spectrum):
    result = flat_spectrum.power()

    assert result.integral == pytest.approx(400.0)


def test_centroid_wavelength_flat_spectrum(flat_spectrum):
    result = flat_spectrum.power()

    assert result.centroid_wavelength == pytest.approx(500.0)


def test_centroid_wavelength_asymmetric_spectrum():
    spectrum = Spectrum(
        wavelength=[0, 1, 2],
        values=[0.0, 0.0, 2.0],
    )

    result = spectrum.power()

    assert result.centroid_wavelength == pytest.approx(2.0)


def test_summary(flat_spectrum):
    result = flat_spectrum.power()

    assert result.summary == {
        "integral": pytest.approx(400.0),
        "centroid_wavelength": pytest.approx(500.0),
    }


def test_empty_spectrum():
    spectrum = Spectrum(
        wavelength=[],
        values=[],
    )

    result = spectrum.power()

    assert np.isnan(result.integral)
    assert np.isnan(result.centroid_wavelength)


def test_zero_area_centroid_is_nan():
    spectrum = Spectrum(
        wavelength=[400, 500, 600],
        values=[0.0, 0.0, 0.0],
    )

    result = spectrum.power()

    assert result.integral == pytest.approx(0.0)
    assert np.isnan(result.centroid_wavelength)


def test_nonfinite_area_centroid_is_nan():
    spectrum = Spectrum(
        wavelength=[400, 500, 600],
        values=[1.0, np.nan, 1.0],
    )

    result = spectrum.power()

    assert np.isnan(result.integral)
    assert np.isnan(result.centroid_wavelength)


def test_integral_nonuniform_wavelength_spacing():
    spectrum = Spectrum(
        wavelength=[0, 1, 3],
        values=[2.0, 2.0, 2.0],
    )

    result = spectrum.power()

    assert result.integral == pytest.approx(6.0)
    assert result.centroid_wavelength == pytest.approx(1.5)


def test_photon_flux_from_spectral_power_density():
    spectrum = Spectrum(
        wavelength=[500, 600],
        values=[1.0, 1.0],
        value_unit="W/nm",
    )

    result = spectrum.power()

    # 100 W distributed uniformly from 500 to 600 nm has a photon rate
    # equal to its power divided by the photon energy at the 550 nm mean.
    expected = 100 * 550e-9 / (6.62607015e-34 * 299792458.0)
    assert result.photon_flux == pytest.approx(expected)
    assert result.quantity("photon_flux").units == units.ureg.photon / units.ureg.s


def test_photon_flux_is_wavelength_unit_invariant_and_converts_to_moles():
    nm = Spectrum([500, 600], [1.0, 1.0], value_unit="W/m^2/nm")
    um = Spectrum([0.5, 0.6], [1000.0, 1000.0],
                  wavelength_unit="um", value_unit="W/m^2/um")

    np.testing.assert_allclose(nm.power().photon_flux, um.power().photon_flux)
    molar = nm.power().quantity("photon_flux").to("umol / m^2 / s")
    assert molar.magnitude == pytest.approx(459.76409760114785)


def test_photon_flux_preserves_radiance_geometry():
    spectrum = Spectrum(
        [500, 600], [1.0, 1.0], value_unit="W/m^2/sr/nm"
    )

    assert spectrum.power().quantity("photon_flux").units == (
        units.ureg.photon / units.ureg.m**2 / units.ureg.sr / units.ureg.s
    )


@pytest.mark.parametrize("value_unit", [None, "count/nm", "J/nm"])
def test_photon_flux_rejects_nonradiometric_rate_units(value_unit):
    spectrum = Spectrum([500, 600], [1.0, 1.0], value_unit=value_unit)

    with pytest.raises(ValueError, match="radiometric spectral values"):
        _ = spectrum.power().photon_flux


def test_power_summary_includes_photon_flux_only_when_available(flat_spectrum):
    radiometric = Spectrum([500, 600], [1.0, 1.0], value_unit="W/nm")

    assert "photon_flux" in radiometric.power().summary
    assert "photon_flux" not in flat_spectrum.power().summary


def test_empty_radiometric_spectrum_has_nan_photon_flux():
    result = Spectrum([], [], value_unit="W/nm").power()

    assert np.isnan(result.photon_flux)


def test_photon_flux_rejects_nonpositive_wavelengths():
    spectrum = Spectrum([0, 1], [1.0, 1.0], value_unit="W/nm")

    with pytest.raises(ValueError, match="positive wavelengths"):
        _ = spectrum.power().photon_flux
