import pytest

from gsmath import Spectrum, units


@pytest.fixture
def radiometric_spectrum():
    return Spectrum(
        wavelength=[400, 500, 600],
        values=[1.0, 2.0, 1.0],
        wavelength_unit="nm",
        value_unit="uW / cm^2 / sr / nm",
    )


def test_power_units_are_derived_from_both_spectrum_axes(
    radiometric_spectrum,
):
    analysis = radiometric_spectrum.power()

    assert analysis.unit("integral") == units.ureg.uW / units.ureg.cm**2 / units.ureg.sr
    assert analysis.unit("centroid_wavelength") == units.ureg.nm
    assert analysis.quantity("integral").magnitude == pytest.approx(
        analysis.integral
    )
    assert analysis.quantity("integral").units == analysis.unit("integral")


def test_peak_units_distinguish_wavelengths_and_values(radiometric_spectrum):
    analysis = radiometric_spectrum.peaks()

    assert analysis.unit("fwhm") == radiometric_spectrum.wavelength_unit
    assert analysis.unit("peak_value") == radiometric_spectrum.value_unit
    assert analysis.quantity("peak_wavelength").units == units.ureg.nm


def test_statistic_units_follow_spectral_values(radiometric_spectrum):
    analysis = radiometric_spectrum.stats()

    assert analysis.unit("mean") == radiometric_spectrum.value_unit
    assert analysis.unit("dynamic_range") == radiometric_spectrum.value_unit
    assert analysis.unit("n") == units.ureg.dimensionless
    assert analysis.quantity("n") == 3 * units.ureg.dimensionless


def test_unknown_value_unit_cannot_be_claimed_as_physical_quantity():
    analysis = Spectrum([400, 500, 600], [1, 2, 1]).stats()

    assert analysis.unit("mean") is None
    with pytest.raises(ValueError, match="No physical unit is known"):
        analysis.quantity("mean")


def test_unknown_result_name_has_clear_error(radiometric_spectrum):
    with pytest.raises(KeyError, match="no unit metadata"):
        radiometric_spectrum.power().unit("missing")
