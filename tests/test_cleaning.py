import numpy as np
import pytest

from gsmath import (
    AdaptiveSmooth, Calibration, Despike, Measurement, PerExposure, Spectrum, SubtractDark,
)
from gsmath.cleaning import adaptive_smooth, despike
from gsmath.units import Q_

WAVELENGTH = np.linspace(177.5, 1102.3, 1024)  # approximately 0.9 nm per sample


def line(center=368.0, fwhm=10.0, height=50000.0, noise=0.0, seed=0):
    """A narrow LED line in counts, with shot-like noise."""
    sigma = fwhm / 2.3548
    clean = height * np.exp(-0.5 * ((WAVELENGTH - center) / sigma) ** 2) + 300
    rng = np.random.default_rng(seed)
    return clean + noise * np.sqrt(clean) * rng.standard_normal(WAVELENGTH.size), clean


def test_a_narrow_bright_line_is_not_a_spike():
    values, _ = line(noise=1.0)
    _, spikes = despike(values)
    assert spikes == []


def test_one_and_two_pixel_spikes_are_found_and_replaced():
    values, clean = line(center=450, noise=1.0)
    values[700] += 400          # a cosmic ray in the tail
    values[800:802] -= 600      # a hot pair in the dark, subtracted
    cleaned, spikes = despike(values)
    assert [i for i, _ in spikes] == [700, 800, 801]
    assert abs(cleaned[700] - clean[700]) < 4 * np.sqrt(clean[700])
    assert np.all(np.abs(cleaned[800:802] - clean[800:802]) < 4 * np.sqrt(clean[800:802]))


def test_adaptive_smoothing_keeps_the_peak_and_quiets_the_tails():
    values, clean = line(noise=1.0)
    smoothed, weight = adaptive_smooth(values, 11, 3, 30)
    peak = int(np.argmax(clean))
    assert weight[peak] < 0.01 and abs(smoothed[peak] / values[peak] - 1) < 1e-4
    tail = slice(700, 1000)
    assert np.std(smoothed[tail] - clean[tail]) < 0.7 * np.std(values[tail] - clean[tail])


def measurement(values, dark=300.0):
    raw = Spectrum(WAVELENGTH, values, value_unit="count", label="raw")
    return Measurement(raw=raw, dark=raw.copy(values=np.full(WAVELENGTH.size, dark)),
                       exposure=Q_(100, "ms"))


def test_steps_report_what_they_did_in_the_processing_record():
    values, _ = line(center=450, noise=1.0)
    values[700] += 400
    base = Calibration("instrument", [SubtractDark(), PerExposure()])
    cleaned = base.with_steps([Despike(), AdaptiveSmooth()], after="SubtractDark")
    assert [s.name for s in cleaned.steps] == ["SubtractDark", "Despike", "AdaptiveSmooth",
                                               "PerExposure"]
    assert cleaned.name == "instrument + Despike+AdaptiveSmooth"
    assert cleaned.fingerprint != base.fingerprint
    spectrum = measurement(values).process(cleaned)
    processing = spectrum.metadata["processing"]
    spike, = processing["reports"]["Despike"]
    assert spike["pixel"] == 700 and spike["wavelength"] == pytest.approx(WAVELENGTH[700], abs=1e-3)
    assert spike["sigmas"] > 10 and "mean_weight" in processing["reports"]["AdaptiveSmooth"]
    assert "step_report" not in spectrum.metadata


def test_cleaning_steps_save_and_read_back(tmp_path):
    cal = Calibration("c", [SubtractDark(), Despike(threshold=8, relative=0.05),
                            AdaptiveSmooth(window=7, snr=50)])
    path = tmp_path / "c.gsmath"
    cal.write(path)
    back = Calibration.read(path)
    assert back.fingerprint == cal.fingerprint
    assert back.steps[1].threshold == 8 and back.steps[2].window == 7


def test_with_steps_refuses_an_unknown_anchor():
    with pytest.raises(ValueError, match="no step 'Dark'"):
        Calibration("c", [SubtractDark()]).with_steps([Despike()], after="Dark")
