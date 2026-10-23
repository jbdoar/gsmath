from dataclasses import replace

import pytest

from gsmath import HeldOutLine, WavelengthValidation


def report(errors=(.1, .5, .6), failures=()):
    rows = tuple(HeldOutLine(w, scan, e, e, 10., 10., .01, w == 800., 12, .02, ())
                 for scan in ('narrow', 'working') for w, e in zip([400., 500., 800.], errors))
    return WavelengthValidation(rows, tuple(failures), (400., 500., 800., 900.), 4, 20.)


def test_default_is_per_peak_half_nm_and_includes_extrapolation():
    result = report().assess()
    values = result['working']['joint']
    assert values['status'] == 'fail'
    assert values['max_error_nm'] == .5 and values['max_rms_nm'] is None
    assert values['violations'] == [{'wavelength_nm': 800., 'error_nm': .6}]
    assert values['scope'] == 'full_instrument'


def test_working_range_filters_errors_failures_and_inventory_inclusively():
    r = report(failures=[{'wavelength_nm': 900., 'reason': 'unscorable'}])
    result = r.assess(working_range_nm=(400, 500), required_lines_nm=[400, 500, 800, 900])
    assert result['working']['joint']['status'] == 'pass'
    assert result['working']['joint']['count'] == 2
    assert result['working']['joint']['coverage_complete']


def test_low_rms_cannot_hide_one_bad_peak():
    r = report(errors=(0., 0., .51))
    values = r.assess(required_lines_nm=[400, 500, 800])['working']['joint']
    assert values['rms_nm'] < .5 and values['status'] == 'fail'


def test_missing_inventory_and_unscored_required_peaks_are_incomplete():
    r = report(errors=(.1, .2, .3))
    r = replace(r, requested_lines=(400., 500., 800.))
    values = r.assess()['working']['joint']
    assert values['passes_prediction_limits'] and values['status'] == 'incomplete'
    values = r.assess(required_lines_nm=[400, 500, 800, 900])['working']['joint']
    assert values['status'] == 'incomplete' and values['unscored_lines_nm'] == [900.]


def test_slit_specific_inventory_and_tolerance_override():
    r = report()
    result = r.assess(max_error_nm=.7, required_lines_nm={'narrow': [400, 500, 800], 'working': [400, 500]})
    assert result['narrow']['joint']['status'] == 'pass'
    assert result['working']['joint']['count'] == 2
    assert r.assess(max_error_nm=.7, max_rms_nm=.1,
                    required_lines_nm=[400, 500, 800])['working']['joint']['status'] == 'fail'


def test_interpolation_only_cannot_silently_drop_required_endpoint():
    values = report(errors=(.1, .2, .3)).assess(
        interpolation_only=True, required_lines_nm=[400, 500, 800])['working']['joint']
    assert values['status'] == 'incomplete' and values['unscored_lines_nm'] == [800.]


@pytest.mark.parametrize('options', [dict(max_error_nm=0), dict(max_rms_nm=-1),
    dict(working_range_nm=(700, 400)), dict(required_lines_nm=[400, 400]),
    dict(required_lines_nm={'typo': [400]})])
def test_invalid_policy(options):
    with pytest.raises(ValueError):
        report().assess(**options)
