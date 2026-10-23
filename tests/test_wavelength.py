import numpy as np
import pytest

from gsmath import (CalibrationPeak, Spectrum, calibrate_wavelength,
                    detect_calibration_peaks, identify_calibration_lines)
from gsmath.wavelength import _assign


def candidates(pixels):
    return [CalibrationPeak(float(p), 1, 1, 2) for p in pixels]


def test_assignment_maximizes_valid_matches():
    # Forcing every row to a real line would sacrifice the exact match.
    rows, cols = _assign(np.array([0., 10.]), np.array([10., 100.]), 1)
    np.testing.assert_array_equal(rows, [1])
    np.testing.assert_array_equal(cols, [0])
    rows, cols = _assign(np.array([10., 10.1]), np.array([10.]), 1)
    assert len(rows) == len(set(cols)) == 1


def test_peak_refinement_and_flat_signal():
    x = np.arange(100)
    peaks = detect_calibration_peaks(20 + 100 * np.exp(-0.5 * ((x - 40.3) / 2)**2))
    assert len(peaks) == 1
    assert peaks[0].pixel == pytest.approx(40.3, abs=0.02)
    assert detect_calibration_peaks(np.ones(100)) == ()
    y = np.zeros(30)
    y[10:14] = 20
    assert detect_calibration_peaks(y)[0].pixel == 11.5


@pytest.mark.parametrize('direction', ['increasing', 'decreasing'])
def test_quadratic_recovery_missing_and_false_peaks(direction):
    pixels = np.array([20, 77, 143, 211, 302, 391, 503, 612, 733, 855, 967.])
    reference = 350 + 0.4 * pixels + 0.00004 * pixels**2
    observed = np.delete(pixels, [2, 7])
    observed = np.r_[observed, 330, 690]
    if direction == 'decreasing':
        observed = 1000 - observed
    # Input order is deliberately unrelated to pixel order.
    result = identify_calibration_lines(candidates(observed), reference,
        degree=2, iterations=2500, tolerance_nm=0.05, pixel_domain=(0, 1000), direction=direction)
    assert len(result.matches) == 9
    expected_pixels = pixels if direction == 'increasing' else 1000 - pixels
    np.testing.assert_allclose(result.wavelength(expected_pixels), reference, atol=1e-7)
    assert result.rms_nm < 1e-7
    assert len(result.unmatched_peak_indices) == 2
    assert result.matched_pixel_fraction > 0.9


def test_spectrum_uses_sample_index_and_preserves_input():
    pixels = np.array([20, 57, 103, 169, 248, 341.])
    x = np.arange(400)
    y = sum((i + 1) * 100 * np.exp(-0.5 * ((x - p) / 2)**2)
            for i, p in enumerate(pixels))
    spectrum = Spectrum(x * 8 + 10000, y)
    options = dict(degree=1, iterations=300, tolerance_nm=0.01,
                   direction='increasing', dispersion_bounds=(0.4, 0.6))
    result = calibrate_wavelength(spectrum, 400 + pixels * 0.5, **options)
    again = calibrate_wavelength(y, 400 + pixels * 0.5, **options)
    np.testing.assert_allclose(result.wavelength(pixels), 400 + pixels * 0.5, atol=1e-8)
    np.testing.assert_array_equal(result.coefficients, again.coefficients)
    np.testing.assert_array_equal(spectrum.wavelength, x * 8 + 10000)
    assert result.pixel_domain == (0, 399)


@pytest.mark.parametrize('options', [dict(degree=0), dict(iterations=0),
    dict(tolerance_nm=0), dict(min_matches=3), dict(direction='sideways'),
    dict(pixel_domain=(3, 2)), dict(dispersion_bounds=(-1, 1))])
def test_invalid_fit_options(options):
    with pytest.raises(ValueError):
        identify_calibration_lines(candidates(np.arange(10)), np.arange(10), **options)


def test_invalid_data_and_insufficient_consensus():
    for y in ([1, np.nan, 2], [1, np.inf, 2], [[1, 2, 3]], [1, 2]):
        with pytest.raises(ValueError):
            detect_calibration_peaks(y)
    with pytest.raises(ValueError, match='unique'):
        identify_calibration_lines(candidates(np.arange(6)), [1, 2, 3, 4, 5, 5])
    with pytest.raises(ValueError):
        identify_calibration_lines(candidates([1, 2, 3]), [400, 500, 600])
    with pytest.raises(RuntimeError):
        identify_calibration_lines(candidates(np.arange(6)), np.arange(6) + 400,
                                   iterations=20, wavelength_bounds_nm=(100, 200))


def test_monotonicity_checked_beyond_matched_region():
    pixels = np.array([10, 20, 30, 40, 50.])
    # Increasing where observed, but turns over inside the full detector.
    reference = 400 + pixels - 0.005 * pixels**2
    with pytest.raises(RuntimeError):
        identify_calibration_lines(candidates(pixels), reference, degree=2, iterations=100,
            tolerance_nm=1e-7, pixel_domain=(0, 200), direction='increasing')


@pytest.fixture
def lamp_solution():
    pixels = np.array([20, 57, 103, 169, 248, 341.])
    x = np.arange(400)
    y = sum(100 * np.exp(-0.5 * ((x - p) / 2)**2) for p in pixels)
    return calibrate_wavelength(y, 400 + pixels * 0.5, degree=1,
        iterations=300, tolerance_nm=0.01, direction='increasing')


def test_apply_solution_through_existing_spectrum_api(lamp_solution):
    s = Spectrum(np.arange(400) / 1000, np.arange(400),
                 wavelength_unit='um', value_unit='count', metadata={'instrument': 'test'})
    result = s.calibrate_wavelength(lamp_solution)
    np.testing.assert_allclose(result.wavelength, 400 + np.arange(400) * 0.5)
    np.testing.assert_array_equal(result.values, s.values)
    assert str(result.wavelength_unit) == 'nanometer'
    assert result.value_unit == s.value_unit
    assert result.metadata['instrument'] == 'test'
    assert result.metadata['wavelength_calibration']['rms_nm'] < 1e-8
    assert 'wavelength_calibration' not in s.metadata
    np.testing.assert_array_equal(s.wavelength, np.arange(400) / 1000)
    with pytest.raises(ValueError, match='same full pixel domain'):
        s.trim(0, 0.1).calibrate_wavelength(lamp_solution)


def test_decreasing_solution_reverses_values_with_axis(lamp_solution):
    from dataclasses import replace
    decreasing = replace(lamp_solution, coefficients=np.array([-0.5, 600]))
    s = Spectrum(np.arange(400), np.arange(400))
    result = s.calibrate_wavelength(decreasing)
    np.testing.assert_allclose(result.wavelength, (600 - np.arange(400) * 0.5)[::-1])
    np.testing.assert_array_equal(result.values, s.values[::-1])


def test_pipeline_save_reload_and_reprocess(lamp_solution, tmp_path):
    from gsmath import Calibration, Measurement, SubtractDark, WavelengthAxis
    raw = Spectrum(np.arange(400), np.arange(400) + 10, value_unit='count')
    dark = Spectrum(np.arange(400), np.full(400, 10), value_unit='count')
    measurement = Measurement(raw, dark=dark, label='sample')
    pipeline = Calibration('lamp calibration', [SubtractDark(), lamp_solution.to_step()])
    saved = pipeline.write(tmp_path / 'lamp.gsmath')
    loaded = Calibration.read(saved)
    result = measurement.process(loaded)
    assert isinstance(loaded.steps[1], WavelengthAxis)
    assert loaded.steps[1].solution.matches == lamp_solution.matches
    assert loaded.steps[1].solution.peaks == lamp_solution.peaks
    assert loaded.fingerprint == pipeline.fingerprint
    np.testing.assert_allclose(result.wavelength, 400 + np.arange(400) * 0.5)
    np.testing.assert_array_equal(result.values, np.arange(400))
    processing = result.metadata['processing']
    assert processing['applied'] == ['SubtractDark', 'WavelengthAxis']
    assert processing['reports']['WavelengthAxis'] == lamp_solution.report
    assert result.label == 'sample'
    np.testing.assert_array_equal(raw.values, np.arange(400) + 10)
    # Fit metadata are covered by the saved fingerprint.
    saved.write_text(saved.read_text().replace('"wavelength_nm": 410.0', '"wavelength_nm": 411.0'))
    with pytest.raises(ValueError, match='fingerprint'):
        Calibration.read(saved)


def test_legacy_axis_corrections_still_use_existing_axis():
    s = Spectrum([400., 500., 600.], [1, 2, 3])
    np.testing.assert_allclose(s.calibrate_wavelength([2, 1]).wavelength, [402, 502, 602])
    np.testing.assert_allclose(s.calibrate_wavelength(lambda x: x + 3).wavelength, [403, 503, 603])


def test_default_is_fourth_order():
    pixels = np.array([10, 27, 54, 83, 121, 163, 210.])
    expected = np.array([1e-9, -2e-7, 3e-5, 0.5, 400])
    lines = np.polyval(expected, pixels)
    result = identify_calibration_lines(candidates(pixels), lines,
        iterations=100, tolerance_nm=0.1, direction='increasing')
    assert len(result.coefficients) == 5
    np.testing.assert_allclose(result.wavelength(pixels), lines, atol=1e-8)
    assert len(result.matches) == 7


def test_default_requires_more_than_five_matches():
    with pytest.raises(ValueError, match='at least 6'):
        identify_calibration_lines(candidates(np.arange(5)), np.arange(5) + 400)
