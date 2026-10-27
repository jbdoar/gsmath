import numpy as np
import pytest

from gsmath import Calibration, Measurement, Responsivity, Spectrum, StrayLightCorrection
from gsmath.amplitude import (LIGHTTOUCH_INVALID, amplitude_calibration, compare_responsivity,
                              evaluate_amplitude, lighttouch_unit, read_amplitude_calibration,
                              responsivity_of)
from gsmath.units import Q_

RADIANCE = "uW / cm^2 / sr / nm"
IRRADIANCE = "uW / cm^2 / nm"
WAVELENGTH = np.linspace(350.0, 930.0, 600)  # the instrument's pixels
IT = 250.0  # ms


def planck(wavelength, temperature=3000.0):
    """A QTH-like spectrum, peak-normalised to 100."""
    x = np.asarray(wavelength) * 1e-9
    values = 1 / (x**5 * (np.exp(1.4388e-2 / (x * temperature)) - 1))
    return 100 * values / values.max()


def truth_responsivity():
    """Calibrated units per (count/ms): low at the ends like a silicon detector."""
    return 0.002 + 0.02 * np.exp(-0.5 * ((WAVELENGTH - 650) / 160) ** 2) ** -1 / 30


def standard(unit=RADIANCE):
    grid = np.arange(340.0, 941.0, 1.0)
    return Spectrum(grid, planck(grid), label="standard", value_unit=unit)


def scan_of(source, integration_time=IT, gain=1.0):
    """Dark-subtracted counts for ``source`` at ``integration_time``."""
    values = np.interp(WAVELENGTH, source.wavelength, source.values)
    counts = gain * values / truth_responsivity() * integration_time
    return Spectrum(WAVELENGTH, counts, label="scan", value_unit="count",
                    metadata={"integration_time_ms": integration_time})


def apply(calibration, scan):
    measurement = Measurement(raw=scan, exposure=Q_(scan.metadata["integration_time_ms"], "ms"))
    return measurement.process(calibration)


@pytest.fixture(scope="module")
def result():
    return amplitude_calibration(scan_of(standard()), standard())


def test_responsivity_is_standard_over_counts_per_ms(result):
    np.testing.assert_allclose(result.responsivity.values, truth_responsivity(), rtol=1e-9)
    assert result.quantity == "radiance"
    assert result.valid.all() and not result.low_signal.all()


def test_native_calibration_is_normalised_by_integration_time(result):
    # The same source at another integration time calibrates to the same values.
    for integration_time in (IT, 40.0, 3000.0):
        measured = apply(result.calibration, scan_of(standard(), integration_time))
        np.testing.assert_allclose(measured.values, np.interp(WAVELENGTH, standard().wavelength,
                                                              standard().values), rtol=1e-9)
    names = [type(step).__name__ for step in result.calibration.steps]
    assert names == ["PerExposure", "Responsivity"]


def test_lighttouch_cal_and_native_file_read_back_the_same(result, tmp_path):
    native, cal = result.write(tmp_path / "amplitude")
    from_cal = responsivity_of(read_amplitude_calibration(cal))
    from_native = responsivity_of(read_amplitude_calibration(native))
    np.testing.assert_allclose(from_cal.values, from_native.values, rtol=1e-7)
    assert cal.read_text(encoding="utf-8").splitlines()[2].startswith("Radiance")
    assert read_amplitude_calibration(cal).metadata["calibration_integration_time_ms"] == IT


def test_irradiance_standard_gives_an_irradiance_calibration(tmp_path):
    result = amplitude_calibration(scan_of(standard(IRRADIANCE)), standard(IRRADIANCE))
    assert result.quantity == "irradiance"
    _, cal = result.write(tmp_path / "irradiance")
    assert cal.read_text(encoding="utf-8").splitlines()[2].startswith("Irradiance")
    assert lighttouch_unit(Spectrum.from_lighttouch(cal)) == IRRADIANCE


def test_outside_the_standard_and_nonpositive_counts_are_invalid(tmp_path):
    narrow = Spectrum(np.arange(400.0, 801.0), planck(np.arange(400.0, 801.0)), value_unit=RADIANCE)
    scan = scan_of(standard())
    values = np.asarray(scan.values).copy()
    values[300] = -5.0
    result = amplitude_calibration(scan.copy(values=values), narrow)
    inside = (WAVELENGTH >= 400) & (WAVELENGTH <= 800)
    assert not result.valid[~inside].any() and not result.valid[300]
    factors = np.asarray(result.lighttouch_factors().values)
    assert np.all(factors[~inside] == 0) and factors[300] == LIGHTTOUCH_INVALID
    # Read back, LightTouch's invalid marker is no responsivity, not a huge one.
    _, cal = result.write(tmp_path / "narrow")
    assert np.asarray(responsivity_of(read_amplitude_calibration(cal)).values)[300] == 0


def test_stray_light_correction_comes_first():
    n = len(WAVELENGTH)
    matrix = np.eye(n) - 0.001 * np.ones((n, n)) / n
    slc = StrayLightCorrection(WAVELENGTH, matrix)
    scan = scan_of(standard())
    result = amplitude_calibration(scan, standard(), stray_light=slc)
    names = [type(step).__name__ for step in result.calibration.steps]
    assert names == ["StrayLight", "PerExposure", "Responsivity"]
    measured = apply(result.calibration, scan_of(standard(), 60.0))
    np.testing.assert_allclose(measured.values, np.interp(WAVELENGTH, standard().wavelength,
                                                          standard().values), rtol=1e-9)


def test_old_per_count_native_files_are_refused(result, tmp_path):
    old = Calibration("old", [Responsivity(result.responsivity)])
    path = old.write(tmp_path / "old.gsmath")
    with pytest.raises(ValueError, match="not normalised by integration time"):
        read_amplitude_calibration(path)


def test_standard_needs_a_radiometric_unit():
    with pytest.raises(ValueError, match="radiance or irradiance unit"):
        amplitude_calibration(scan_of(standard()), standard().copy(value_unit=None))


def test_evaluate_passes_a_good_calibration_and_fails_a_scaled_one(result):
    measured = apply(result.calibration, scan_of(standard(), 100.0))
    good = evaluate_amplitude(measured, standard())
    assert good.passed
    assert abs(good.comparison.records["Y"].difference) < 1e-6
    assert "Luminance (cd / m^2)" in good.table()
    bright = evaluate_amplitude(measured.copy(values=np.asarray(measured.values) * 1.05), standard())
    assert not bright.passed
    assert bright.comparison.failed == ("Y",)


def test_compare_two_calibrations_by_band_ratio(result):
    same = compare_responsivity(result.calibration, result.calibration)
    assert same["passed"] and same["median"] == pytest.approx(1.0)
    shifted = result.responsivity.copy(values=np.asarray(result.responsivity.values) * 1.03)
    other = Calibration("other", [result.calibration.steps[0], Responsivity(shifted)])
    worse = compare_responsivity(other, result.calibration)
    assert not worse["passed"] and worse["median"] == pytest.approx(1.03)


def test_cli_calibrate_show_evaluate_compare(tmp_path, capsys):
    from gsmath.scripts.amplitude import main
    standard_path = standard().write(tmp_path / "standard.cal", quantity="radiance")  # LightTouch format
    scan_path = scan_of(standard()).write(tmp_path / "scan.gsmath")
    output = tmp_path / "out"
    assert main(["calibrate", str(scan_path), "--standard", str(standard_path),
                 "--output", str(output)]) == 0
    for name in ("amplitude.gsmath", "amplitude.cal", "amplitude.png", "amplitude.json"):
        assert (output / name).exists()
    assert main(["show", str(output / "amplitude.cal")]) == 0
    assert "uW ms / cm^2 / count / nm / sr" in capsys.readouterr().out
    other_scan = scan_of(standard(), 80.0).write(tmp_path / "other.gsmath")
    for calibration in ("amplitude.cal", "amplitude.gsmath"):
        assert main(["evaluate", str(other_scan), "--calibration", str(output / calibration),
                     "--certified", str(standard_path)]) == 0
    assert main(["compare", str(output / "amplitude.gsmath"), str(output / "amplitude.cal")]) == 0
    assert main([]) == 2
