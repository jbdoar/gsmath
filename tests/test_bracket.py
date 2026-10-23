import numpy as np
import pytest

import gsmath
from gsmath import Spectrum, bracket


def test_steady_drift_cancels_for_numbers():
    # The reference falls 1 per step; B is 10% above it at the middle step.
    b = bracket(100.0, 99.0 * 1.10, 98.0)
    assert b.reference == pytest.approx(99.0)
    assert b.effect == pytest.approx(0.10)
    assert b.difference == pytest.approx(9.9)
    assert b.drift == pytest.approx(-0.02)


def test_spectra_bracket_pixel_by_pixel():
    # A dark that wanders between two dark scans: the mean is what to subtract.
    wavelength = np.arange(400.0, 405.0)
    dark_before = Spectrum(wavelength, np.full(5, 100.0))
    dark_after = Spectrum(wavelength, np.full(5, 104.0))
    scan = Spectrum(wavelength, np.array([102.0, 202.0, 302.0, 402.0, 502.0]))
    b = bracket(dark_before, scan, dark_after)
    np.testing.assert_allclose(b.reference.values, 102.0)
    np.testing.assert_allclose(b.difference.values, [0.0, 100.0, 200.0, 300.0, 400.0])
    np.testing.assert_allclose(b.drift.values, 0.04)


def test_exported():
    assert gsmath.bracket is bracket and gsmath.Bracketed
