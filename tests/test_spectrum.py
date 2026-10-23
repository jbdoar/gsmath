import numpy as np
import pint
import pytest

from gsmath import Spectrum, units


@pytest.fixture
def spectrum():
    return Spectrum(
        wavelength=[400, 500, 600],
        values=[1.0, 2.0, 3.0],
        label="Test",
    )


# construction
def test_init():
    s = Spectrum([400, 500], [1, 2])

    np.testing.assert_array_equal(s.wavelength, [400, 500])
    np.testing.assert_array_equal(s.values, [1, 2])


def test_label():
    s = Spectrum([400], [1], label="ABC")

    assert s.label == "ABC"


def test_metadata_copied():
    metadata = {"a": 1}

    s = Spectrum([400], [1], metadata=metadata)

    metadata["a"] = 2

    assert s.metadata == {"a": 1}


# units
def test_default_wavelength_unit():
    s = Spectrum([400], [1])

    assert s.wavelength_unit == units.ureg.nm


def test_call_interpolates_at_scalar_wavelength(spectrum):
    assert spectrum(450) == pytest.approx(1.5)


def test_call_interpolates_at_multiple_wavelengths(spectrum):
    result = spectrum([425, 475])

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [1.25, 1.75])


def test_to_wavelength_unit():
    s = Spectrum([1000], [1])

    s2 = s.to_wavelength_unit("um")

    np.testing.assert_allclose(s2.wavelength, [1])
    assert s2.wavelength_unit == units.ureg.um


def test_to_value_unit():
    s = Spectrum([400], [1000], value_unit="uW")

    s2 = s.to_value_unit("mW")

    np.testing.assert_allclose(s2.values, [1])


# unary operators
def test_neg(spectrum):
    np.testing.assert_allclose((-spectrum).values, [-1, -2, -3])


def test_abs():
    s = Spectrum([1,2], [-2,3])

    np.testing.assert_allclose(abs(s).values, [2,3])


def test_power(spectrum):
    np.testing.assert_allclose((spectrum**2).values, [1,4,9])


# binary operators
def test_add_spectrum():
    s1 = Spectrum([400,500], [1,2])
    s2 = Spectrum([400,500], [3,4])

    np.testing.assert_allclose((s1+s2).values, [4,6])


def test_subtract_spectrum():
    ...


def test_multiply_scalar():
    ...


def test_divide_scalar():
    ...


def test_add_scalar_unitless():
    s = Spectrum([400], [1])

    assert (s+2).values[0] == 3


# error conditions
def test_add_mismatched_wavelengths():
    s1 = Spectrum([400,500], [1,2])
    s2 = Spectrum([401,501], [1,2])

    with pytest.raises(ValueError):
        s1 + s2


def test_add_mismatched_wavelength_units():
    s1 = Spectrum([400], [1], wavelength_unit="nm")
    s2 = Spectrum([0.4], [1], wavelength_unit="um")

    with pytest.raises(ValueError):
        s1 + s2


def test_add_unitless_scalar_to_unit_values():
    s = Spectrum([400], [1], value_unit="W")

    with pytest.raises(ValueError):
        s + 1


# reverse operators
def test_radd():
    s = Spectrum([400], [1])

    assert (2+s).values[0] == 3


def test_rsub():
    s = Spectrum([400], [1])

    assert (2-s).values[0] == 1


def test_rmul(spectrum):
    result = 2 * spectrum

    np.testing.assert_allclose(result.values, [2, 4, 6])


def test_rtruediv(spectrum):
    result = 1 / spectrum

    np.testing.assert_allclose(result.values, [1, 0.5, 1/3])


# internal copy behavior
def test_operations_return_new_object(spectrum):
    s2 = spectrum * 2

    assert s2 is not spectrum
    assert s2.values is not spectrum.values
    assert s2.wavelength is spectrum.wavelength  # unchanged, read-only: shared


def test_spectra_are_immutable_so_analyses_are_cached_for_good():
    spectrum = Spectrum([400., 500.], [1., 1.])
    assert spectrum.power() is spectrum.power()
    with pytest.raises(ValueError, match="read-only"):
        spectrum.values[:] = 2.
    with pytest.raises(ValueError, match="read-only"):
        spectrum.wavelength[:] = [400., 600.]
    with pytest.raises(AttributeError, match="immutable: use copy"):
        spectrum.values = np.array([2., 2.])
    changed = spectrum.copy(values=[2., 2.])  # a change is a new spectrum
    assert changed.power().integral == pytest.approx(200.)
    assert spectrum.power().integral == pytest.approx(100.)
    spectrum.metadata["note"] = "metadata stays a free-form dict"


def test_a_spectrum_checks_its_arrays():
    Spectrum([], [])  # empty is fine
    Spectrum([400], [1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        Spectrum([500, 400], [1.0, 2.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        Spectrum([400, 400], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        Spectrum([400, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match="one value per wavelength"):
        Spectrum([400, 500], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="one value per wavelength"):
        Spectrum([400, 500], [[1.0, 2.0]])
    with pytest.raises(ValueError, match="one-dimensional array of numbers"):
        Spectrum(["a", "b"], [1.0, 2.0])
    with pytest.raises(TypeError, match="must be numbers"):
        Spectrum([400, 500], ["1", "2"])
    Spectrum([400, 500], [np.nan, 1.0])  # missing values are data, not an error
