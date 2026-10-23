# tests/test_units.py

import numpy as np
import pint
import pytest

from gsmath import units


def test_parse_none():
    assert units.parse(None) is None


def test_parse_string():
    assert units.parse("nm") == units.ureg.nanometer


def test_parse_unit():
    unit = units.ureg.microwatt
    assert units.parse(unit) is unit


def test_custom_count_unit():
    assert units.parse("count") == units.ureg.count


def test_custom_pixel_unit():
    assert units.parse("pixel") == units.ureg.pixel


def test_is_quantity():
    assert units.is_quantity(5 * units.ureg.nm)
    assert not units.is_quantity(5)
    assert not units.is_quantity(np.array([1, 2, 3]))


def test_convert_values():
    values = np.array([1000.0, 2000.0])

    converted = units.convert_values(values, "nm", "um")

    np.testing.assert_allclose(converted, [1.0, 2.0])


def test_convert_values_unitless_returns_copy():
    values = np.array([1.0, 2.0])

    converted = units.convert_values(values, None, None)

    np.testing.assert_array_equal(converted, values)
    assert converted is not values


def test_convert_values_incompatible_units():
    with pytest.raises(pint.DimensionalityError):
        units.convert_values([1.0], "nm", "s")


def test_convert_quantity():
    quantity = np.array([1000.0, 2000.0]) * units.ureg.nm

    converted = units.convert_quantity(quantity, "um")

    np.testing.assert_allclose(converted, [1.0, 2.0])
    

def test_convert_dimensionless_quantity():
    quantity = np.array([1.0, 2.0]) * units.ureg.dimensionless

    converted = units.convert_quantity(quantity, None)

    np.testing.assert_array_equal(converted, [1.0, 2.0])


def test_add_converts_rhs_to_lhs_unit():
    values, unit = units.add(
        np.array([1.0]),
        "m",
        np.array([100.0]),
        "cm",
    )

    np.testing.assert_allclose(values, [2.0])
    assert unit == units.ureg.m


def test_subtract_converts_rhs_to_lhs_unit():
    values, unit = units.subtract(
        np.array([2.0]),
        "m",
        np.array([100.0]),
        "cm",
    )

    np.testing.assert_allclose(values, [1.0])
    assert unit == units.ureg.m


def test_multiply_combines_units():
    values, unit = units.multiply(
        np.array([2.0]),
        "m",
        np.array([3.0]),
        "s",
    )

    np.testing.assert_allclose(values, [6.0])
    assert unit == units.ureg.m * units.ureg.s


def test_divide_combines_units():
    values, unit = units.divide(
        np.array([6.0]),
        "m",
        np.array([2.0]),
        "s",
    )

    np.testing.assert_allclose(values, [3.0])
    assert unit == units.ureg.m / units.ureg.s


def test_format_unit_pretty():
    from gsmath.units import format_unit
    assert format_unit("uW/cm^2/nm/sr", pretty=True) == "µW/cm²/nm/sr"
    assert format_unit("uW/cm^2/nm/sr") == "µW / cm ** 2 / nm / sr"
    assert format_unit(None, pretty=True) is None


def test_is_compatible():
    from gsmath.units import is_compatible
    assert is_compatible("uW/cm^2/nm/sr", "W/m^2/nm/sr")
    assert not is_compatible("uW/cm^2/nm", "nm")
    assert is_compatible(None, None) and not is_compatible(None, "nm")
