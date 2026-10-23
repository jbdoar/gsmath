import numpy as np
import pytest

from gsmath import Spectrum


@pytest.fixture
def spectrum():
    return Spectrum(
        wavelength=[400, 500, 600, 700],
        values=[1.0, 2.0, 3.0, 4.0],
        label="Test",
    )


def test_mean(spectrum):
    assert spectrum.stats().mean == pytest.approx(2.5)


def test_min(spectrum):
    assert spectrum.stats().min == pytest.approx(1.0)


def test_max(spectrum):
    assert spectrum.stats().max == pytest.approx(4.0)


def test_std(spectrum):
    assert spectrum.stats().std == pytest.approx(
        np.std([1.0, 2.0, 3.0, 4.0])
    )


def test_median(spectrum):
    assert spectrum.stats().median == pytest.approx(2.5)


def test_rms(spectrum):
    expected = np.sqrt(
        np.mean(np.array([1.0, 2.0, 3.0, 4.0]) ** 2)
    )

    assert spectrum.stats().rms == pytest.approx(expected)


def test_dynamic_range(spectrum):
    assert spectrum.stats().dynamic_range == pytest.approx(3.0)


def test_n(spectrum):
    assert spectrum.stats().n == 4


def test_summary(spectrum):
    result = spectrum.stats()

    assert result.summary == {
        "mean": pytest.approx(2.5),
        "min": pytest.approx(1.0),
        "max": pytest.approx(4.0),
        "std": pytest.approx(np.std([1.0, 2.0, 3.0, 4.0])),
        "median": pytest.approx(2.5),
        "rms": pytest.approx(np.sqrt(7.5)),
        "dynamic_range": pytest.approx(3.0),
        "n": 4,
    }

def test_empty_spectrum():
    spectrum = Spectrum(
        wavelength=[],
        values=[],
    )

    result = spectrum.stats()

    assert np.isnan(result.mean)
    assert np.isnan(result.min)
    assert np.isnan(result.max)
    assert np.isnan(result.std)
    assert np.isnan(result.median)
    assert np.isnan(result.rms)
    assert np.isnan(result.dynamic_range)
    assert result.n == 0


def test_constant_spectrum():
    spectrum = Spectrum(
        wavelength=[400, 500, 600],
        values=[5.0, 5.0, 5.0],
    )

    result = spectrum.stats()

    assert result.mean == pytest.approx(5.0)
    assert result.std == pytest.approx(0.0)
    assert result.rms == pytest.approx(5.0)
    assert result.dynamic_range == pytest.approx(0.0)


def test_nan_values_propagate():
    spectrum = Spectrum(
        wavelength=[400, 500, 600],
        values=[1.0, np.nan, 3.0],
    )

    result = spectrum.stats()

    assert np.isnan(result.mean)
    assert np.isnan(result.min)
    assert np.isnan(result.max)
    assert np.isnan(result.std)
    assert np.isnan(result.median)
    assert np.isnan(result.rms)
    assert np.isnan(result.dynamic_range)
    assert result.n == 3
