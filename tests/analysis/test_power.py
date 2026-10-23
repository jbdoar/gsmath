import numpy as np
import pytest

from gsmath import Spectrum


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
