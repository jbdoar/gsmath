"""Series, comparisons and checks as native files: gsmath.series,
gsmath.comparison and gsmath.check records."""

import json
import math

import numpy as np
import pytest

import gsmath
from gsmath import Collection, Image, LineCut, Series, Spectrum, compare
from gsmath.colorimeter import SpectralSensitivity
from gsmath.comparison import Check, Comparison, check
from gsmath.profile import SPECTRUM_PROFILE


def led(label, peak=500.0, scale=1.0):
    x = np.linspace(380.0, 780.0, 81)
    return Spectrum(x, scale * np.exp(-0.5 * ((x - peak) / 10.0) ** 2) + 0.01, label=label,
                    value_unit="uW/cm^2/sr/nm", metadata={"serial": label})


def same_series(after, before):
    assert type(after) is type(before)
    np.testing.assert_array_equal(after.axis, before.axis)
    np.testing.assert_array_equal(after.values, before.values)
    assert after.values.dtype == before.values.dtype
    assert (after.label, after.channels, after.metadata) == (before.label, before.channels,
                                                             before.metadata)
    assert (after.axis_unit, after.value_unit) == (before.axis_unit, before.value_unit)


def test_a_series_round_trips_as_readable_json(tmp_path):
    series = Series([10.0, 20.0, 40.0], [[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]],
                    label="Peak vs exposure", x_unit="ms", value_unit="nm", channels=("A", "B"),
                    metadata={"source": {"workspace": "SN-2066"}}, x_label="Exposure")
    path = series.write(tmp_path / "trend.gsmath")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert (record["format"], record["version"], record["series"]["type"]) == \
        ("gsmath.series", 1, "series")
    loaded = gsmath.read(path)
    same_series(loaded, series)
    assert loaded.x_label == "Exposure"
    assert Series.read(path).label == "Peak vs exposure"


def test_line_cuts_and_sensitivities_keep_their_kind(tmp_path):
    image = Image(np.arange(400, dtype=float).reshape(20, 20), label="panel", value_unit="cd/m^2")
    cut = image.line((2, 10), (17, 10))
    loaded = gsmath.read(cut.write(tmp_path / "cut.gsmath"))
    same_series(loaded, cut)
    np.testing.assert_array_equal(loaded.x, cut.x)
    np.testing.assert_array_equal(loaded.y, cut.y)
    assert LineCut.read(tmp_path / "cut.gsmath").distance_unit == cut.distance_unit

    sensitivity = SpectralSensitivity([400.0, 500.0, 600.0], np.eye(3), value_unit="count/nJ")
    same_series(gsmath.read(sensitivity.write(tmp_path / "rgb.gsmath")), sensitivity)
    with pytest.raises(ValueError, match="not a LineCut"):
        LineCut.read(tmp_path / "rgb.gsmath")


def test_a_series_subclass_without_a_record_is_refused(tmp_path):
    class Custom(Series):
        pass

    with pytest.raises(TypeError, match="can't save a Custom"):
        Custom([1.0, 2.0], [3.0, 4.0]).write(tmp_path / "custom.gsmath")


def test_spectra_keep_their_own_record(tmp_path):
    path = led("lamp").write(tmp_path / "lamp.gsmath")
    assert json.loads(path.read_text())["format"] == "gsmath.spectrum"


def test_a_comparison_round_trips_with_its_limits_and_outcome(tmp_path):
    profile = SPECTRUM_PROFILE.with_limits({
        "peaks": {"peak_wavelength": {"tolerance": 2.0, "uncertainty": 0.5},
                  "fwhm": {"tolerance": 0.1}},
        "power": {"integral": {"mode": "percent", "tolerance": 5.0}},
    })
    result = compare(led("450nm", 451.0), led("450nm", 450.0, 1.02), profile=profile)
    path = result.write(tmp_path / "comparison.gsmath")
    loaded = Comparison.read(path)
    assert isinstance(gsmath.read(path), Comparison)
    assert loaded.label == result.label
    assert loaded.passed == result.passed
    assert loaded.failed == result.failed
    assert loaded.summary == result.summary
    assert loaded.summary_units() == result.summary_units()
    for (path_before, before), (path_after, after) in zip(result.records.items(),
                                                          loaded.records.items()):
        assert path_after == path_before
        assert after.field.mode == before.field.mode
        assert after.field.tolerance == before.field.tolerance
        assert after.unit == before.unit
    with pytest.raises(TypeError, match="read from a saved result"):
        loaded.records["peaks.fwhm"].field.getter(led("x"))


def test_non_finite_values_survive(tmp_path):
    zero = Spectrum([400.0, 500.0, 600.0], [0.0, 0.0, 0.0], label="dark", value_unit="count")
    profile = SPECTRUM_PROFILE.with_limits({"power": {"integral": {"mode": "relative"}}})
    result = compare(zero, zero, profile=profile)
    difference = result.records["power.integral"].difference
    loaded = Comparison.read(result.write(tmp_path / "nan.gsmath"))
    after = loaded.records["power.integral"].difference
    assert (math.isnan(after) if math.isnan(difference) else after == difference)
    json.loads((tmp_path / "nan.gsmath").read_text())  # still strict JSON


def test_a_collection_of_comparisons_is_one_file(tmp_path):
    profile = SPECTRUM_PROFILE.with_limits({"peaks": {"peak_wavelength": {"tolerance": 1.0}}})
    measured = Collection([led("450nm", 450.5), led("520nm", 523.0)], label="SN-2066")
    reference = Collection([led("450nm", 450.0), led("520nm", 520.0)], label="hx0489")
    results = compare(measured, reference, profile=profile)
    path = results.write(tmp_path / "results.gsmath")
    record = json.loads(path.read_text())
    assert [i["format"] for i in record["collection"]["items"]] == ["gsmath.comparison"] * 2
    loaded = gsmath.read(path)
    assert loaded.labels == ("450nm", "520nm")
    assert [c.passed for c in loaded] == [True, False]
    assert loaded["520nm"].failed == ("peaks.peak_wavelength",)


def test_a_check_round_trips(tmp_path):
    profile = SPECTRUM_PROFILE.with_limits({"peaks": {"peak_wavelength": {"minimum": 445.0,
                                                                          "maximum": 455.0}}})
    result = check(led("450nm", 465.0), profile)
    loaded = Check.read(result.write(tmp_path / "check.gsmath"))
    assert loaded.summary == result.summary
    assert loaded.failed == ("peaks.peak_wavelength",)
    assert loaded.records["peaks.peak_wavelength"].field.maximum == 455.0
