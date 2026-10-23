import numpy as np
import pytest

from gsmath import Spectrum
from gsmath.transforms import (
    interpolate,
    normalize,
    pad,
    trim,
)


@pytest.fixture
def spectrum():
    return Spectrum(
        wavelength=[400, 500, 600],
        values=[1.0, 2.0, 3.0],
        label="Test",
    )


# trim
def test_trim(spectrum):
    result = trim(spectrum, 450, 550)

    np.testing.assert_array_equal(result.wavelength, [500])
    np.testing.assert_array_equal(result.values, [2])


def test_trim_empty(spectrum):
    result = trim(spectrum, 700, 800)

    assert result.wavelength.size == 0
    assert result.values.size == 0


def test_trim_preserves_metadata(spectrum):
    result = trim(spectrum, 450, 550)

    assert result.wavelength_unit == spectrum.wavelength_unit
    assert result.value_unit == spectrum.value_unit
    assert result.metadata == spectrum.metadata


def test_trim_updates_label(spectrum):
    result = trim(spectrum, 450, 550)

    assert result.label == "Test [trimmed]"


def test_interpolate_exact_points(spectrum):
    result = interpolate(spectrum, xnew=spectrum.wavelength)

    np.testing.assert_allclose(result.values, spectrum.values)


def test_pad_leaves_fully_finite_spectrum_unchanged(spectrum):
    result = pad(spectrum, "edge")

    np.testing.assert_allclose(result.values, spectrum.values)


def test_normalize_does_not_modify_original(spectrum):
    original = spectrum.values.copy()

    normalize(spectrum)

    np.testing.assert_array_equal(spectrum.values, original)


# interpolate
def test_interpolate_xnew(spectrum):
    result = interpolate(spectrum, xnew=[450, 550])

    np.testing.assert_allclose(result.values, [1.5, 2.5])


def test_interpolate_step(spectrum):
    result = interpolate(spectrum, step=50)

    np.testing.assert_array_equal(
        result.wavelength,
        [400, 450, 500, 550, 600],
    )


def test_interpolate_both_xnew_and_step(spectrum):
    with pytest.raises(ValueError):
        interpolate(spectrum, [450], step=10)


def test_interpolate_requires_xnew_or_step(spectrum):
    with pytest.raises(ValueError):
        interpolate(spectrum)


def test_interpolate_invalid_spline_order(spectrum):
    with pytest.raises(ValueError):
        interpolate(spectrum, [450], spline_order=0)


def test_interpolate_non_1d_xnew(spectrum):
    with pytest.raises(ValueError):
        interpolate(spectrum, np.zeros((2,2)))


# Explicit grids
def test_interpolate_outside_range_returns_nan(spectrum):
    result = interpolate(spectrum, xnew=[350, 450, 650])

    assert np.isnan(result.values[0])
    assert np.isnan(result.values[-1])
    assert result.values[1] == pytest.approx(1.5)


def test_interpolate_empty(spectrum):
    result = interpolate(spectrum, xnew=[])

    assert result.values.size == 0


def test_unsorted_queries_are_lookups_not_spectra(spectrum):
    np.testing.assert_allclose(spectrum([500, 400, 450]), [2, 1, 1.5])
    with pytest.raises(ValueError, match="call the spectrum"):
        interpolate(spectrum, xnew=[500, 400, 450])


# pad
def test_zero_pad():
    s = Spectrum(
        [400,500,600,700],
        [np.nan,1,2,np.nan],
    )

    result = pad(s)

    np.testing.assert_allclose(
        result.values,
        [0,1,2,0],
        equal_nan=True,
    )


def test_constant_pad():
    s = Spectrum(
        [400, 500, 600, 700],
        [np.nan, 1.0, 2.0, np.nan],
    )

    result = pad(s, "constant", value=42)

    np.testing.assert_allclose(
        result.values,
        [42.0, 1.0, 2.0, 42.0],
        equal_nan=True,
    )


def test_edge_pad():
    s = Spectrum(
        [400, 500, 600, 700],
        [np.nan, 1.0, 2.0, np.nan],
    )

    result = pad(s, "edge")

    np.testing.assert_allclose(
        result.values,
        [1.0, 1.0, 2.0, 2.0],
        equal_nan=True,
    )


def test_pad_invalid_mode(spectrum):
    with pytest.raises(ValueError):
        pad(spectrum, "banana")


def test_constant_requires_value(spectrum):
    with pytest.raises(ValueError):
        pad(spectrum, "constant")


def test_constant_requires_numeric_value(spectrum):
    with pytest.raises(TypeError):
        pad(spectrum, "constant", value="abc")


def test_zero_pad_all_nan():
    s = Spectrum(
        [400, 500],
        [np.nan, np.nan],
    )

    result = pad(s, "zero")

    np.testing.assert_array_equal(
        result.values,
        [0.0, 0.0],
    )


def test_constant_pad_all_nan():
    s = Spectrum(
        [400, 500],
        [np.nan, np.nan],
    )

    result = pad(s, "constant", value=42)

    np.testing.assert_array_equal(
        result.values,
        [42.0, 42.0],
    )


def test_edge_pad_all_nan():
    s = Spectrum(
        [400, 500],
        [np.nan, np.nan],
    )

    with pytest.raises(ValueError):
        pad(s, "edge")

# normalize
def test_normalize_peak(spectrum):
    result = normalize(spectrum)

    np.testing.assert_allclose(
        result.values,
        [1/3, 2/3, 1],
    )


def test_normalize_peak_to_value(spectrum):
    result = normalize(spectrum, value=10)

    assert np.max(result.values) == pytest.approx(10)


def test_normalize_wavelength(spectrum):
    result = normalize(spectrum, by=500)

    np.testing.assert_allclose(
        result.values,
        [0.5,1,1.5],
    )


def test_normalize_invalid_mode(spectrum):
    with pytest.raises(ValueError):
        normalize(spectrum, by="banana")


def test_normalize_invalid_type(spectrum):
    with pytest.raises(TypeError):
        normalize(spectrum, by=[])


def test_normalize_non_numeric_value(spectrum):
    with pytest.raises(TypeError):
        normalize(spectrum, value="abc")


def test_normalize_outside_range(spectrum):
    with pytest.raises(ValueError):
        normalize(spectrum, by=700)


# trim_tails
def test_trim_tails_zeroes_from_the_first_nonpositive_value_outward():
    from gsmath.transforms import trim_tails

    s = Spectrum(wavelength=np.arange(9) + 400.0,
                 values=[0.3, -0.1, 0.2, 1.0, 3.0, 1.0, 0.0, 0.4, 0.1], label="s")
    trimmed = trim_tails(s)
    np.testing.assert_array_equal(trimmed.values, [0, 0, 0.2, 1.0, 3.0, 1.0, 0, 0, 0])
    np.testing.assert_array_equal(trimmed.wavelength, s.wavelength)
    np.testing.assert_array_equal(s.trim_tails("none").values,
                                  [0.3, 0, 0.2, 1.0, 3.0, 1.0, 0, 0.4, 0.1])
    with pytest.raises(ValueError, match="Unknown tail trim"):
        trim_tails(s, "sometimes")


def test_trim_tails_of_nothing_positive_is_all_zero():
    s = Spectrum(wavelength=[400.0, 401.0, 402.0], values=[-1.0, np.nan, 0.0])
    np.testing.assert_array_equal(s.trim_tails().values, [0, 0, 0])


# effective_range
def test_effective_range():
    s = Spectrum(wavelength=[400.0, 401, 402, 403, 404, 405],
                 values=[0.0, 0.005, 0.5, 1.0, 0.02, 0.0])
    assert s.effective_range() == (401.0, 404.0)
    assert s.effective_range(0.01) == (402.0, 404.0)
    assert s.effective_range(1.0) == (403.0, 403.0)
    assert Spectrum(wavelength=[400.0, 401.0], values=[0.0, -1.0]).effective_range() is None
