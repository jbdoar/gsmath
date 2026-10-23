"""One result protocol: summaries with units, export, limits and comparisons."""

from collections.abc import Mapping

import numpy as np
import pytest
from scipy.special import ndtr

from gsmath import (
    Collection, DarkModel, Image, Linearity, Spectrum, fit_color_matrix, to_dataframe, to_dict,
)
from gsmath.analysis.base import Result
from gsmath.comparison import Check
from gsmath.units import Q_, parse


def xyz_image(scale=1.0, shape=(120, 160), seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    falloff = 1 - 0.2 * ((cols / shape[1] - 0.5) ** 2 + (rows / shape[0] - 0.5) ** 2)
    Y = 100 * scale * falloff + rng.normal(0, 0.2, shape)
    return Image(np.stack([0.95 * Y, Y, 1.09 * Y], axis=-1), label="panel", value_unit="cd/m^2",
                 channels=("X", "Y", "Z"))


def edge_image():
    rows, cols = np.mgrid[0:80, 0:64].astype(float)
    return Image(ndtr((cols - 32 - 0.1 * (rows - 40)) / 0.8) * 200 + 10, value_unit="count")


def results():
    """One of every kind of result, by name."""
    image = xyz_image()
    t = np.array([1.0, 2, 4, 8])
    darks = [Image(np.full((4, 6), 100.0) + 20 * ms / 1000, value_unit="count") for ms in t]
    signals = np.random.default_rng(1).uniform(10, 100, (6, 3))
    fit = fit_color_matrix(signals, signals @ np.array([[0.5, 0.3, 0.2], [0.2, 0.7, 0.1],
                                                        [0.0, 0.1, 0.9]]).T,
                           unit="cd/m^2 / count")
    return {
        "image statistics": image.stats(),
        "histogram": image.histogram(16),
        "image colour": image.color(),
        "mean colour": image.color().mean(),
        "edge MTF": edge_image().edge_mtf(),
        "uniformity": image.uniformity(),
        "full-field uniformity": image.uniformity().full_field(),
        "dark model": DarkModel.fit(darks, [Q_(ms, "ms") for ms in t]),
        "linearity": Linearity.fit(Q_(t, "ms"), np.column_stack([10 * t, 20 * t]),
                                   channels=("R", "G"), signal_unit="count"),
        "colour matrix": fit,
        "spectrum peaks": Spectrum(np.arange(500.0, 600), np.exp(-((np.arange(500.0, 600) - 550)
                                                                  / 10) ** 2),
                                   value_unit="W/nm").peaks(),
    }


def leaves(mapping, path=()):
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            yield from leaves(value, (*path, key))
        else:
            yield ".".join((*path, str(key))), value


@pytest.mark.parametrize("name", list(results()))
def test_every_result_declares_the_unit_of_every_number(name):
    result = results()[name]
    assert isinstance(result, Result)
    units = dict(leaves(result.summary_units()))
    for path, value in leaves(result.summary):
        if isinstance(value, (bool, str)) or value is None:
            continue
        assert units[path] is not None, f"{name}: no unit for {path}"
    to_dict(result, flat=True, include_units=True)  # exports
    assert len(to_dataframe(result, include_units=True)) == 1


def test_units_by_path_name_or_group():
    uniformity = xyz_image().uniformity()
    assert uniformity.unit("luminance.mean") == parse("cd/m^2")
    assert uniformity.unit("luminance.min_max_ratio") == parse("dimensionless")  # the group's
    assert uniformity.unit("color.max_delta_uv_prime") == parse("dimensionless")
    stats = xyz_image().stats()
    assert stats.unit("G.mean") == parse("cd/m^2") == stats.unit("mean")  # any channel
    assert uniformity.quantity("luminance.mean").units == parse("cd/m^2")
    assert uniformity.value("luminance.cv") == uniformity.luminance["cv"]
    with pytest.raises(KeyError, match="no unit metadata"):
        stats.unit("width")
    with pytest.raises(KeyError, match="no result"):
        uniformity.value("luminance.nothing")


def test_include_units_pairs_each_number_with_its_unit():
    found = to_dict(xyz_image().uniformity(), include_units=True)
    assert found["luminance"]["mean"]["unit"] == "cd / m ** 2"
    assert found["luminance"]["cv"] == {"value": pytest.approx(found["luminance"]["cv"]["value"]),
                                        "unit": "dimensionless"}


def test_a_uniformity_specification_passes_or_fails():
    uniformity = xyz_image().uniformity()
    spec = {"luminance": {"min_max_ratio": {"minimum": 0.9},
                          "mean": {"minimum": 90, "maximum": 110}},
            "color": {"max_delta_uv_prime": {"maximum": 0.004}}}
    result = uniformity.check(spec)
    assert isinstance(result, Check) and result.passed is True and result.failed == ()
    assert result.label == "panel"
    record = result.records["luminance.min_max_ratio"]
    assert record.value == uniformity.luminance["min_max_ratio"] and record.passed
    strict = uniformity.check({"luminance": {"min_max_ratio": {"minimum": 0.99}}})
    assert strict.passed is False and strict.failed == ("luminance.min_max_ratio",)
    summary = strict.summary["results"]["luminance"]["min_max_ratio"]
    assert summary["minimum"] == 0.99 and summary["passed"] is False
    assert uniformity.check(uniformity.profile().with_limits(spec)).passed  # a Profile works too


def test_checks_report_fields_without_limits_and_refuse_typos():
    uniformity = xyz_image().uniformity()
    reported = uniformity.check({"luminance": {"mean": {}}})
    assert reported.passed is None and "passed" not in reported.summary["results"]["luminance"]["mean"]
    with pytest.raises(KeyError, match="luminance.minimum_ratio"):
        uniformity.check({"luminance": {"minimum_ratio": {"minimum": 0.8}}})
    with pytest.raises(ValueError, match="Unknown setting"):
        uniformity.check({"luminance": {"mean": {"lowest": 0.8}}})


def test_nan_fails_a_limit():
    from gsmath.comparison import check
    from gsmath.profile import Field, Profile

    profile = Profile(fields={"v": Field(lambda value: value, maximum=1.0)})
    assert check(np.nan, profile).passed is False
    assert check(np.array([0.5, 0.9]), profile).passed is True
    assert check(np.array([0.5, 1.1]), profile).passed is False


def test_results_compare_with_results_of_their_kind():
    measured, reference = xyz_image(1.03).uniformity(), xyz_image(seed=1).uniformity()
    comparison = measured.compare(reference, {"luminance": {"mean": {"tolerance": 0.02,
                                                                     "mode": "relative"}}})
    assert comparison.passed is False and comparison.failed == ("luminance.mean",)
    everything = measured.compare(reference)
    assert everything.passed is None and "color.max_delta_xy" in everything.records


def test_images_summarise_compare_and_check():
    image = xyz_image()
    summary = image.summary()
    assert set(summary) == {"X", "Y", "Z"} and summary["Y"]["mean"] == pytest.approx(
        image.stats().mean[1])
    assert image.summary_units()["Y"]["mean"] == parse("cd/m^2")
    assert image.compare(xyz_image(1.01)).records["Y.mean"].difference < 0
    check = image.check({"Y": {"mean": {"minimum": 90}, "std": {"maximum": 0.1}}})
    assert check.failed == ("Y.std",)
    batch = Collection([xyz_image().copy(label="a"), xyz_image(0.8).copy(label="b")])
    checks = [image.check({"Y": {"mean": {"minimum": 90}}}) for image in batch]
    assert [c.passed for c in checks] == [True, False]
    frame = to_dataframe(batch, include_units=True)
    assert list(frame["label"]) == ["a", "b"] and "Y.mean [cd / m ** 2]" in frame.columns


def test_spectra_check_against_limits():
    wl = np.arange(500.0, 600)
    spectrum = Spectrum(wl, np.exp(-4 * np.log(2) * ((wl - 550) / 20) ** 2))
    assert spectrum.check({"peaks": {"fwhm": {"maximum": 25}}}).passed is True
    assert spectrum.check({"peaks": {"fwhm": {"maximum": 15}}}).failed == ("peaks.fwhm",)


def test_one_interface_for_data_collections_and_results():
    """check(limits, *, profile=None) and compare(reference, limits=None,
    *, profile=None) for every kind."""
    from gsmath.profile import SPECTRUM_PROFILE

    wl = np.arange(500.0, 600)
    narrow = Spectrum(wl, np.exp(-4 * np.log(2) * ((wl - 550) / 20) ** 2), label="narrow",
                      value_unit="W/nm")
    wide = narrow.copy(values=np.exp(-4 * np.log(2) * ((wl - 550) / 30) ** 2), label="wide")
    image = xyz_image()
    things = {
        "spectrum": (narrow, wide, {"peaks": {"fwhm": {"maximum": 25, "tolerance": 1}}}),
        "image": (image, xyz_image(1.1), {"Y": {"mean": {"minimum": 90, "tolerance": 5}}}),
        "result": (image.uniformity(), xyz_image(1.1).uniformity(),
                   {"luminance": {"mean": {"minimum": 90, "tolerance": 5}}}),
        "collection": (Collection([narrow, wide]), narrow,
                       {"peaks": {"fwhm": {"maximum": 25, "tolerance": 1}}}),
    }
    for name, (thing, reference, limits) in things.items():
        assert thing.check(limits).passed is not None, name
        assert thing.check(thing.profile().with_limits(limits)).passed is not None, name
        assert thing.compare(reference).passed is None, name          # nothing judged
        assert thing.compare(reference, limits).passed is not None, name
        assert thing.compare(reference, profile=thing.profile()).passed is None, name
    # limits apply to another profile when given one
    only_peaks = SPECTRUM_PROFILE.with_limits({"peaks": {"fwhm": {}}})
    assert set(narrow.summary(only_peaks)) == {"peaks"}
    check = narrow.check({"peaks": {"fwhm": {"maximum": 25}}}, profile=only_peaks)
    assert check.passed and list(check.records) == ["peaks.fwhm"]
