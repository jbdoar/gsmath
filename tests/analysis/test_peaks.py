import numpy as np
import pytest

from gsmath import Spectrum


@pytest.fixture
def single_peak_spectrum():
    return Spectrum(
        wavelength=[400, 410, 420, 430, 440],
        values=[0.0, 1.0, 2.0, 1.0, 0.0],
        label="Single peak",
    )


@pytest.fixture
def two_peak_spectrum():
    return Spectrum(
        wavelength=[400, 410, 420, 430, 440, 450, 460],
        values=[0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0],
        label="Two peaks",
    )


# basic detection
def test_single_peak_count(single_peak_spectrum):
    result = single_peak_spectrum.peaks()

    assert len(result) == 1


def test_single_peak_index(single_peak_spectrum):
    result = single_peak_spectrum.peaks()

    np.testing.assert_array_equal(result.peak_indices, [2])


def test_single_peak_wavelength(single_peak_spectrum):
    result = single_peak_spectrum.peaks()

    assert result.peak_wavelength == pytest.approx(420.0)


def test_single_peak_value(single_peak_spectrum):
    result = single_peak_spectrum.peaks()

    assert result.peak_value == pytest.approx(2.0)


# multiple peaks and primary selection
def test_all_peaks(two_peak_spectrum):
    result = two_peak_spectrum.peaks(primary_only=False)

    np.testing.assert_array_equal(
        result.peak_indices,
        [1, 3, 5],
    )

    np.testing.assert_allclose(
        result.peak_wavelengths,
        [410, 430, 450],
    )

    np.testing.assert_allclose(
        result.peak_values,
        [2, 1, 3],
    )


def test_primary_peak_is_tallest(two_peak_spectrum):
    result = two_peak_spectrum.peaks(primary_only=True)

    assert len(result) == 1
    assert result.peak_wavelength == pytest.approx(450.0)
    assert result.peak_value == pytest.approx(3.0)


# singular properties should reject multi-peak mode
@pytest.mark.parametrize(
    "property_name",
    [
        "peak_wavelength",
        "peak_value",
        "center_wavelength",
        "fwhm",
        "peak_wavelength_polyfit",
        "peak_wavelength_gaussian",
        "centroid_wavelength",
    ],
)
def test_singular_properties_require_primary_only(
    two_peak_spectrum,
    property_name,
):
    result = two_peak_spectrum.peaks(primary_only=False)

    with pytest.raises(ValueError):
        getattr(result, property_name)


# empty and invalid spectra
def test_empty_spectrum():
    spectrum = Spectrum([], [])

    result = spectrum.peaks()

    assert len(result) == 0
    assert result.peak_indices.size == 0
    assert np.isnan(result.peak_wavelength)
    assert np.isnan(result.peak_value)


def test_zero_spectrum_has_no_peaks():
    spectrum = Spectrum(
        [400, 500, 600],
        [0.0, 0.0, 0.0],
    )

    result = spectrum.peaks()

    assert len(result) == 0
    assert np.isnan(result.peak_wavelength)


def test_negative_spectrum_has_no_peaks():
    spectrum = Spectrum(
        [400, 500, 600],
        [-1.0, -0.5, -1.0],
    )

    result = spectrum.peaks()

    assert len(result) == 0


def test_all_nan_spectrum_has_no_peaks():
    spectrum = Spectrum(
        [400, 500, 600],
        [np.nan, np.nan, np.nan],
    )

    result = spectrum.peaks()

    assert len(result) == 0


def test_all_nan_spectrum_has_no_peaks():
    spectrum = Spectrum(
        [400, 500, 600],
        [np.nan, np.nan, np.nan],
    )

    result = spectrum.peaks()

    assert len(result) == 0
    assert result.peak_indices.size == 0
    assert np.isnan(result.peak_wavelength)


# width and center
def test_center_wavelength(single_peak_spectrum):
    result = single_peak_spectrum.peaks()

    assert result.center_wavelength == pytest.approx(420.0)


def test_fwhm(single_peak_spectrum):
    result = single_peak_spectrum.peaks()

    assert result.fwhm == pytest.approx(20.0)


def test_polyfit_peak_wavelength(single_peak_spectrum):
    result = single_peak_spectrum.peaks(
        fit_radius=2,
        poly_order=2,
    )

    assert result.peak_wavelength_polyfit == pytest.approx(
        420.0,
        abs=0.1,
    )


def test_gaussian_peak_wavelength():
    wavelength = np.linspace(400, 440, 81)
    values = np.exp(
        -0.5 * ((wavelength - 421.5) / 4.0) ** 2
    )

    spectrum = Spectrum(wavelength, values)
    result = spectrum.peaks(fit_radius=15)

    assert result.peak_wavelength_gaussian == pytest.approx(
        421.5,
        abs=0.1,
    )


# centroid
def test_peak_centroid_symmetric(single_peak_spectrum):
    result = single_peak_spectrum.peaks(fit_radius=2)

    assert result.centroid_wavelength == pytest.approx(420.0)


# insufficient fitting points
def test_polyfit_returns_nan_when_window_too_small():
    spectrum = Spectrum(
        [400, 410, 420],
        [0.0, 1.0, 0.0],
    )

    result = spectrum.peaks(
        poly_order=4,
        fit_radius=1,
    )

    assert np.isnan(result.peak_wavelength_polyfit)


def test_gaussian_returns_nan_when_window_too_small():
    spectrum = Spectrum(
        [400, 410, 420],
        [0.0, 1.0, 0.0],
    )

    result = spectrum.peaks(fit_radius=1)

    assert np.isnan(result.peak_wavelength_gaussian)


# summaries
def test_primary_summary(single_peak_spectrum):
    summary = single_peak_spectrum.peaks().summary

    assert set(summary) == {
        "peak_wavelength",
        "peak_value",
        "center_wavelength",
        "fwhm",
        "peak_wavelength_polyfit",
        "centroid_wavelength",
    }

    assert summary["peak_wavelength"] == pytest.approx(420.0)
    assert summary["peak_value"] == pytest.approx(2.0)
    assert np.isfinite(summary["center_wavelength"])
    assert np.isfinite(summary["fwhm"])
    assert np.isfinite(summary["peak_wavelength_polyfit"])
    assert np.isfinite(summary["centroid_wavelength"])


def test_multiple_peak_summary(two_peak_spectrum):
    summary = two_peak_spectrum.peaks(primary_only=False).summary

    assert set(summary) == {
        "peak_count",
        "peak_wavelengths",
        "peak_values",
        "center_wavelengths",
        "fwhms",
        "peak_wavelengths_polyfit",
        "centroid_wavelengths",
    }

    assert summary["peak_count"] == 3

    np.testing.assert_allclose(
        summary["peak_wavelengths"],
        [410, 430, 450],
    )

    for key in (
        "peak_values",
        "center_wavelengths",
        "fwhms",
        "peak_wavelengths_polyfit",
        "centroid_wavelengths",
    ):
        assert summary[key].shape == (3,)
