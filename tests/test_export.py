import pandas as pd
import pytest

from gsmath import Collection, Spectrum, report, to_dataframe, to_dict


def spectrum(label="sample", *, wavelength_unit="nm"):
    return Spectrum(
        [400, 500, 600],
        [1.0, 2.0, 1.0],
        label=label,
        wavelength_unit=wavelength_unit,
        value_unit="uW / nm",
    )


def test_to_dict_can_export_unit_metadata():
    result = to_dict(spectrum(), include_units=True)

    assert result["power"]["integral"] == {
        "value": 300.0,
        "unit": "µW",
    }


def test_dataframe_puts_units_in_labels_and_keeps_numeric_cells():
    result = to_dataframe(spectrum(), include_units=True)

    assert "power.integral [µW]" in result
    assert "peaks.fwhm [nm]" in result
    assert "color.2deg.x [dimensionless]" in result
    assert pd.api.types.is_numeric_dtype(result["power.integral [µW]"])


def test_report_can_include_units_in_column_labels():
    result = report(spectrum(), include_units=True)

    assert "statistics.mean [µW / nm]" in result


def test_collection_unit_labels_require_consistent_units():
    collection = Collection(
        [
            spectrum("nm", wavelength_unit="nm"),
            spectrum("um", wavelength_unit="um"),
        ]
    )

    with pytest.raises(ValueError, match="different units"):
        to_dataframe(collection, include_units=True)
