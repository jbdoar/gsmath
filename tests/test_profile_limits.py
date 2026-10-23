import tomllib

import numpy as np
import pytest

from gsmath import Spectrum, compare
from gsmath.profile import SPECTRUM_PROFILE, Field

LIMITS_TOML = """
[peaks.fwhm]
tolerance = 3.0

[power.integral]
mode = "relative"
tolerance = 0.1
uncertainty = 0.02

[color.2deg.x]
tolerance = 0.005
"""


def gaussian_spectrum(center, height=1.0):
    wavelength = np.arange(380.0, 781.0)
    values = height * np.exp(-0.5 * ((wavelength - center) / 10) ** 2)
    return Spectrum(wavelength, values, value_unit="uW / cm^2 / sr / nm")


def test_with_limits_selects_and_updates_fields_from_toml():
    profile = SPECTRUM_PROFILE.with_limits(tomllib.loads(LIMITS_TOML))

    assert profile.name == SPECTRUM_PROFILE.name
    assert set(profile.fields) == {"peaks", "power", "color"}
    assert set(profile.fields["peaks"]) == {"fwhm"}
    assert set(profile.fields["color"]["2deg"]) == {"x"}

    integral = profile.fields["power"]["integral"]
    assert (integral.mode, integral.tolerance, integral.uncertainty) == ("relative", 0.1, 0.02)
    assert integral.getter is SPECTRUM_PROFILE.fields["power"]["integral"].getter
    # The source profile is untouched.
    assert SPECTRUM_PROFILE.fields["power"]["integral"].tolerance is None


def test_with_limits_drives_comparison_pass_fail():
    profile = SPECTRUM_PROFILE.with_limits(tomllib.loads(LIMITS_TOML), name="acceptance")
    assert profile.name == "acceptance"

    result = compare(gaussian_spectrum(550, 1.05), gaussian_spectrum(550), profile=profile)
    assert result.passed is True
    assert list(result.records) == ["peaks.fwhm", "power.integral", "color.2deg.x"]
    assert result.records["power.integral"].difference == pytest.approx(0.05)

    result = compare(gaussian_spectrum(550, 1.2), gaussian_spectrum(550), profile=profile)
    assert result.failed == ("power.integral",)


def test_with_limits_rejects_unknown_field_path():
    with pytest.raises(KeyError, match="peaks.peak_wavelenght"):
        SPECTRUM_PROFILE.with_limits({"peaks": {"peak_wavelenght": {"tolerance": 1}}})


def test_with_limits_rejects_unknown_setting_and_mode():
    with pytest.raises(ValueError, match="tolerence"):
        SPECTRUM_PROFILE.with_limits({"peaks": {"fwhm": {"tolerence": 1}}})
    with pytest.raises(ValueError, match="Unknown comparison mode"):
        SPECTRUM_PROFILE.with_limits({"peaks": {"fwhm": {"mode": "absolute"}}})


def test_with_limits_rejects_non_mapping_entries():
    with pytest.raises(TypeError, match="peaks.fwhm"):
        SPECTRUM_PROFILE.with_limits({"peaks": {"fwhm": 3.0}})
    with pytest.raises(TypeError, match="peaks"):
        SPECTRUM_PROFILE.with_limits({"peaks": 3.0})
    with pytest.raises(TypeError):
        SPECTRUM_PROFILE.with_limits([("peaks", {})])


def test_with_limits_on_custom_profile():
    from gsmath.profile import Profile

    custom = Profile("custom", {"peak": Field(lambda s: s.peaks().peak_wavelength)})
    limited = custom.with_limits({"peak": {"tolerance": 1.0}})
    assert limited.fields["peak"].tolerance == 1.0
    assert limited.name == "custom"


def test_power_centroid_is_whole_spectrum_unlike_local_peaks_centroid():
    # A narrow peak at 500 nm plus a broad shoulder at 600 nm: the local
    # centroid stays at the peak, the power centroid moves toward the shoulder.
    wavelength = np.arange(380.0, 781.0)
    values = (np.exp(-0.5 * ((wavelength - 500) / 5) ** 2)
              + 0.3 * np.exp(-0.5 * ((wavelength - 600) / 40) ** 2))
    spectrum = Spectrum(wavelength, values, value_unit="uW / cm^2 / sr / nm")

    power = SPECTRUM_PROFILE.fields["power"]["centroid_wavelength"].getter(spectrum)
    local = SPECTRUM_PROFILE.fields["peaks"]["centroid_wavelength"].getter(spectrum)
    expected = np.trapezoid(wavelength * values, wavelength) / np.trapezoid(values, wavelength)

    assert power == pytest.approx(expected)
    assert power > 570
    assert local == pytest.approx(500, abs=0.5)
