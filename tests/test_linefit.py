import json

import numpy as np
import pytest
from scipy.special import ndtr

from gsmath import (Calibration, CalibrationPeak, FactoryLineModel, LineMatch,
                    LineSpreadFunction, Measurement, Spectrum, WavelengthCalibration,
                    detect_calibration_peaks, fit_factory_lines, fit_lamp_lines)


def synthetic_pair():
    n = 600
    x = np.arange(n)
    coefficients = np.array([1e-10, -8e-8, 2e-5, .48, 400.])
    wave = np.polyval(coefficients, x)
    refs = np.array([415, 440, 477, 479, 515, 550, 552, 600, 650, 675.])
    centers = np.interp(refs, wave, x)
    areas = np.array([900, 700, 1100, 550, 1000, 950, 450, 1200, 950, 700.])

    def scan(sigma, shift, gain):
        c = centers+shift
        # Independent analytic integration of Gaussian lines across each pixel.
        profile = ndtr((x[:, None]+.5-c)/sigma)-ndtr((x[:, None]-.5-c)/sigma)
        return Spectrum(wave, 30 + .01*x + gain*(profile@areas))

    peaks = tuple(CalibrationPeak(float(c), 100, 100, 2) for c in centers)
    initial_coefficients = coefficients.copy()
    initial_coefficients[-1] += .15
    matches = tuple(LineMatch(i, float(c), float(line), float(np.polyval(initial_coefficients, c)),
                              float(np.polyval(initial_coefficients, c)-line))
                    for i, (c, line) in enumerate(zip(centers, refs)))
    initial = WavelengthCalibration(initial_coefficients, matches, peaks, (0, n-1))
    return scan(.9, 0, 1), scan(2.5, -1, 1.5), initial, refs, coefficients


@pytest.fixture(scope='module')
def fitted_pair():
    narrow, wide, initial, refs, truth = synthetic_pair()
    model, result = fit_factory_lines(narrow, wide, initial=initial, reference_lines=refs,
                                      max_nfev=500, metadata={'instrument': 'synthetic'})
    return model, result, narrow, wide, truth


def test_integrated_shape_area_and_mean():
    x = np.arange(-150, 151)
    shape = LineSpreadFunction((2, 2, 2), .3, 5)
    y = shape.profiles(x+200, [200.25], 401)[:, 0]
    assert y.sum() == pytest.approx(1, abs=1e-10)
    assert np.sum(x*y) == pytest.approx(.25, abs=1e-5)
    assert np.all(y >= -1e-14)


def test_factory_recovers_blended_wavelengths_and_slit_shift(fitted_pair):
    model, result, narrow, wide, truth = fitted_pair
    x = np.arange(40, 560)
    assert all(len(c.coefficients) == 5 for c in result.calibrations)
    np.testing.assert_allclose(result.calibrations[0].wavelength(x), np.polyval(truth, x), atol=.12)
    np.testing.assert_allclose(result.calibrations[1].wavelength(x), np.polyval(truth, x+1), atol=.12)
    assert max(result.normalized_rms) < .015
    assert len(result.reference_lines) == 10
    assert all(np.isnan(r[~m]).all() for r, m in zip(result.residuals, result.masks))
    for modeled, component, background, mask in zip(result.models, result.components,
                                                   result.backgrounds, result.masks):
        np.testing.assert_allclose(modeled[mask], component[mask].sum(axis=1)+background[mask])
    assert model.metadata['instrument'] == 'synthetic'
    assert result.calibrations[1].peaks == detect_calibration_peaks(wide.values)
    assert result.calibrations[1].peaks != result.calibrations[0].peaks
    for match in result.calibrations[1].matches:
        assert match.pixel == result.calibrations[1].peaks[match.peak_index].pixel


def test_factory_roundtrip_field_fit_and_pipeline(fitted_pair, tmp_path):
    model, result, narrow, wide, truth = fitted_pair
    path = model.write(tmp_path / 'factory.gsmath')
    loaded = FactoryLineModel.read(path)
    np.testing.assert_array_equal(loaded.relative_areas, model.relative_areas)
    import gsmath
    assert isinstance(gsmath.read(path), FactoryLineModel)  # registered with gsmath.read
    assert loaded.metadata == model.metadata
    assert loaded.working_shape == model.working_shape
    # A field exposure with different overall gain and background needs no narrow scan.
    field = wide.copy(values=wide.values*1.3 + 25)
    fitted = loaded.fit(field, max_nfev=500)
    x = np.arange(40, 560)
    np.testing.assert_allclose(fitted.calibrations[0].wavelength(x), np.polyval(truth, x+1), atol=.15)
    pipeline = Calibration('joint', [fitted.calibrations[0].to_step()])
    again = Calibration.read(pipeline.write(tmp_path / 'cal.gsmath'))
    output = Measurement(field).process(again)
    np.testing.assert_array_equal(output.values, field.values)
    assert output.metadata['processing']['reports']['WavelengthAxis']['joint_fit']['degree'] == 4
    payload = json.loads(path.read_text())
    payload['model']['relative_areas'][0] *= 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='checksum'):
        FactoryLineModel.read(path)


def test_invalid_inputs_and_nonconvergence():
    a, b, initial, refs, truth = synthetic_pair()
    with pytest.raises(ValueError, match='same detector'):
        fit_factory_lines(a, b.values[:-1], initial=initial)
    with pytest.raises(ValueError, match='degree'):
        fit_lamp_lines(a, initial=initial, degree=0)
    with pytest.raises(ValueError, match='full pixel domain'):
        fit_lamp_lines(a.values[:-1], initial=initial)
    with pytest.raises(RuntimeError, match='converge'):
        fit_factory_lines(a, b, initial=initial, reference_lines=refs, max_nfev=1)
    with pytest.raises(ValueError, match='unsaturated'):
        fit_lamp_lines(a, initial=initial, reference_lines=refs, saturation=1)


def test_field_allows_drift_noise_and_changed_line_ratios(fitted_pair):
    model, result, narrow, wide, truth = fitted_pair
    x = np.arange(len(wide.values))
    refs = model.reference_lines
    centers = np.interp(refs, np.polyval(truth, x), x)-.7
    # Different width, drift, line ratios, gain, background, and detector noise.
    areas = np.array([900, 700, 1100, 550, 1000, 950, 450, 1200, 950, 700.])
    areas *= np.array([1.1, .9, .9, 1.2, 1.1, 1.1, .8, 1., .9, 1.1])
    profiles = ndtr((x[:, None]+.5-centers)/2.65)-ndtr((x[:, None]-.5-centers)/2.65)
    signal = 45 + .02*x + profiles@areas*1.7 + np.random.default_rng(3).normal(0, .15, len(x))
    fitted = model.fit(signal)
    check = np.arange(40, 560)
    np.testing.assert_allclose(fitted.calibrations[0].wavelength(check),
                               np.polyval(truth, check+.7), atol=.15)
    assert fitted.normalized_rms[0] < .015


def test_training_seed_discards_heldout_and_full_fit_coefficients():
    from dataclasses import replace
    from gsmath.wavelength_validation import _training_calibration
    a, b, initial, refs, truth = synthetic_pair()
    mask = np.ones(len(a.values), dtype=bool)
    first = _training_calibration(initial, [515.], mask, 4)
    altered = replace(initial, coefficients=np.array([1., 12345.]),
                      matches=tuple(replace(m, pixel=m.pixel+50) if m.wavelength_nm == 515 else m
                                    for m in initial.matches))
    second = _training_calibration(altered, [515.], mask, 4)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    assert all(m.wavelength_nm != 515 for m in first.matches)
    assert len(first.coefficients) == 5


def test_heldout_validation_excludes_pixels_and_predicts_known_line(monkeypatch):
    from gsmath import validate_factory_lines
    import gsmath.wavelength_validation as validation
    a, b, initial, refs, truth = synthetic_pair()
    original = validation.fit_factory_lines
    checked = []

    def checked_fit(*args, **kwargs):
        assert 515 not in kwargs['reference_lines']
        assert all(m.wavelength_nm != 515 for m in kwargs['initial'].matches)
        result = original(*args, **kwargs)
        assert all(not np.any(used & ~allowed) for used, allowed in
                   zip(result[1].masks, kwargs['sample_masks']))
        checked.append(True)
        return result

    monkeypatch.setattr(validation, 'fit_factory_lines', checked_fit)
    report = validate_factory_lines(a, b, [515.], narrow_initial=initial,
                                    working_initial=initial, reference_lines=refs)
    assert checked and not report.failures
    assert len(report.records) == 2
    assert all(not row.extrapolated for row in report.records)
    assert max(abs(row.joint_error_nm) for row in report.records) < .15
    assert report.assess(max_error_nm=.15, max_rms_nm=.15)['working']['joint']['passes_prediction_limits']


def test_validation_missing_identity_is_a_failure_not_a_pass():
    from dataclasses import replace
    from gsmath import validate_factory_lines
    a, b, initial, refs, truth = synthetic_pair()
    missing = replace(initial, matches=tuple(m for m in initial.matches if m.wavelength_nm != 515))
    report = validate_factory_lines(a, b, [515.], narrow_initial=missing,
                                    working_initial=initial, reference_lines=refs)
    assert len(report.failures) == 1 and not report.records
    assert not report.assess(max_error_nm=1, max_rms_nm=1)['working']['joint']['passes_prediction_limits']
    assert report.summary()['working']['joint']['rms_nm'] is None
    with pytest.raises(ValueError, match='reference_lines'):
        validate_factory_lines(a, b, [999], reference_lines=refs)


def test_joint_shift_budget_allows_constant_shift_beyond_per_coefficient_share():
    from dataclasses import replace
    a, b, initial, refs, truth = synthetic_pair()
    coefficients = initial.coefficients.copy()
    coefficients[-1] += .8
    initial = replace(initial, coefficients=coefficients)
    fitted = fit_lamp_lines(a, initial=initial, reference_lines=refs, max_shift_nm=2.)
    x = np.arange(40, 560)
    np.testing.assert_allclose(fitted.calibrations[0].wavelength(x), np.polyval(truth, x), atol=.12)
    full = np.arange(len(a.values))
    assert np.max(abs(fitted.calibrations[0].wavelength(full)-initial.wavelength(full))) <= 2.+1e-8


def test_sample_masks_validate_shape_and_dtype():
    a, b, initial, refs, truth = synthetic_pair()
    with pytest.raises(ValueError, match='boolean'):
        fit_lamp_lines(a, initial=initial, sample_masks=[np.ones(len(a.values))])
    with pytest.raises(ValueError, match='one boolean mask'):
        fit_lamp_lines(a, initial=initial, sample_masks=[])
