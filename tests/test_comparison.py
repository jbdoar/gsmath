from types import SimpleNamespace

import numpy as np
import pytest

from gsmath import Collection, report, units
from gsmath.comparison import Comparison, ComparisonRecord, _compare_value, _evaluate_value, compare
from gsmath.profile import Field, Profile


def field(**kwargs):
    return Field(lambda value: value.value, **kwargs)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("difference", 2.0), ("relative", 0.2), ("percent", 20.0), ("ratio", 1.2)],
)
def test_compare_value_modes(mode, expected):
    assert _compare_value(12.0, 10.0, field(mode=mode)) == pytest.approx(expected)


def test_compare_value_array():
    result = _compare_value([1.5, 1.0, 5.0], [1.0, 2.0, 3.0], field())
    np.testing.assert_allclose(result, [0.5, -1.0, 2.0])


def test_compare_value_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="different shapes"):
        _compare_value([1.0, 2.0], [1.0], field())


def test_compare_value_rejects_non_numeric_values():
    with pytest.raises(TypeError, match="must be numeric"):
        _compare_value("10deg", "2deg", field())


def test_compare_value_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown comparison mode"):
        _compare_value(2.0, 1.0, field(mode="banana"))


def test_compare_value_uses_custom_comparator():
    comparison_field = field(comparator=lambda measured, reference: measured == reference)
    assert _compare_value("2deg", "2deg", comparison_field) is True


@pytest.mark.parametrize(
    ("mode", "difference", "tolerance", "expected"),
    [
        ("difference", 0.5, 1.0, True),
        ("difference", 1.5, 1.0, False),
        ("ratio", 1.05, 0.1, True),
        ("ratio", 1.2, 0.1, False),
    ],
)
def test_evaluate_value(mode, difference, tolerance, expected):
    assert _evaluate_value(difference, field(mode=mode, tolerance=tolerance)) is expected


def test_evaluate_value_without_tolerance():
    assert _evaluate_value(100.0, field()) is None


def test_compare_returns_nested_comparison():
    profile = Profile(
        name="test",
        fields={"measurement": {"value": field(tolerance=1.0, uncertainty=0.2)}},
    )
    result = compare(
        SimpleNamespace(value=10.5, label="sample"),
        SimpleNamespace(value=10.0, label="standard"),
        profile=profile,
    )

    assert isinstance(result, Comparison)
    assert result.label == "sample"
    assert result.passed is True
    assert result.failed == ()
    assert result.evaluated is True
    assert result.leaves == (
        ComparisonRecord(
            measured=10.5,
            reference=10.0,
            difference=pytest.approx(0.5),
            field=profile.fields["measurement"]["value"],
            passed=True,
        ),
    )
    assert result.summary == {
        "label": "sample",
        "passed": True,
        "failed": (),
        "results": {
            "measurement": {
                "value": {
                    "measured": 10.5,
                    "reference": 10.0,
                    "difference": pytest.approx(0.5),
                    "mode": "difference",
                    "tolerance": 1.0,
                    "uncertainty": 0.2,
                    "passed": True,
                }
            }
        },
    }


def test_compare_reports_failed_paths():
    profile = Profile(
        fields={"first": field(tolerance=0.1), "nested": {"second": field(tolerance=0.1)}}
    )
    result = compare(
        SimpleNamespace(value=2.0),
        SimpleNamespace(value=1.0),
        profile=profile,
    )
    assert result.passed is False
    assert result.failed == ("first", "nested.second")


def test_compare_without_tolerances_is_unevaluated():
    result = compare(
        SimpleNamespace(value=2.0),
        SimpleNamespace(value=1.0),
        profile=Profile(fields={"value": field()}),
    )
    assert result.passed is None
    assert result.failed == ()
    assert result.evaluated is False
    assert set(result.summary) == {"label", "results"}


def test_compare_rejects_invalid_profile():
    with pytest.raises(TypeError, match="profile must be a Profile"):
        compare(SimpleNamespace(), SimpleNamespace(), profile="test")


def test_compare_converts_compatible_field_units():
    profile = Profile(
        fields={
            "wavelength": Field(
                lambda value: value.value,
                unit=lambda value: value.unit,
            )
        }
    )
    measured = SimpleNamespace(value=500.0, unit=units.ureg.nm)
    reference = SimpleNamespace(value=0.5, unit=units.ureg.um)

    result = compare(measured, reference, profile=profile)
    record = result.leaves[0]

    assert record.reference == pytest.approx(500.0)
    assert record.difference == pytest.approx(0.0)
    assert record.unit == units.ureg.nm


def test_comparison_report_labels_physical_and_relative_units():
    profile = Profile(
        fields={
            "absolute": Field(
                lambda value: value.value,
                unit="nm",
            ),
            "relative": Field(
                lambda value: value.value,
                mode="relative",
                unit="nm",
            ),
        }
    )
    result = compare(
        SimpleNamespace(value=12.0),
        SimpleNamespace(value=10.0),
        profile=profile,
    )

    dataframe = report(result, include_units=True)

    assert "results.absolute.difference [nm]" in dataframe
    assert "results.relative.measured [nm]" in dataframe
    assert "results.relative.difference [dimensionless]" in dataframe


def test_compare_collections_pairs_items_by_label():
    measured = Collection(
        [SimpleNamespace(value=2.0, label="a"), SimpleNamespace(value=4.0, label="b")],
        label="measured",
    )
    reference = Collection(
        [SimpleNamespace(value=3.0, label="b"), SimpleNamespace(value=1.0, label="a")],
        label="reference",
    )
    result = compare(measured, reference, profile=Profile(fields={"value": field()}))

    assert isinstance(result, Collection)
    assert result.label == "measured"
    assert [item.label for item in result] == ["a", "b"]
    assert [item.leaves[0].difference for item in result] == [1.0, 1.0]


def test_compare_collection_rejects_mismatched_labels():
    measured = Collection([SimpleNamespace(value=1.0, label="a")])
    reference = Collection([SimpleNamespace(value=1.0, label="b")])
    with pytest.raises(ValueError, match="matching labels"):
        compare(measured, reference, profile=Profile(fields={"value": field()}))


def test_compare_single_value_against_collection_is_rejected():
    reference = Collection([SimpleNamespace(value=1.0, label="a")])
    with pytest.raises(TypeError, match="reference Collection"):
        compare(
            SimpleNamespace(value=1.0),
            reference,
            profile=Profile(fields={"value": field()}),
        )
