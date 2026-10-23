"""Pint-backed unit parsing, conversion, and arithmetic helpers."""

import numpy as np
# units.py

import pint


ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

ureg.define("count = [count] = counts")
ureg.define("pixel = [pixel] = px")
# Relate an individual photon to the exact Avogadro constant so photon rates
# can also be expressed in molar units for applications such as PPFD.
ureg.define("photon = mole / 6.02214076e23 = photons")

UnitLike = str | pint.Unit | None


def parse(unit: UnitLike) -> pint.Unit | None:
    """Normalize a string or Pint unit while preserving ``None``."""
    if unit is None or is_unit(unit):
        return unit

    return ureg.Unit(unit)


def multiply_units(left: UnitLike, right: UnitLike) -> pint.Unit | None:
    """Return the product of two optional units.

    ``None`` represents a unitless operand, matching the way Spectrum stores
    unitless axes and values.
    """
    return _multiply_units(parse(left), parse(right))


def divide_units(left: UnitLike, right: UnitLike) -> pint.Unit | None:
    """Return the quotient of two optional units (``None`` is unitless)."""
    return _divide_units(parse(left), parse(right))


def format_unit(unit: UnitLike, *, pretty: bool = False) -> str | None:
    """Return a compact, presentation-friendly unit label.

    ``pretty`` uses superscript powers and no spaces (``µW/cm²/nm/sr``), for
    plot axes and report headings; otherwise ``µW / cm ** 2 / nm / sr``.
    """
    unit = parse(unit)

    if unit is None:
        return None

    if unit == ureg.dimensionless:
        return "dimensionless"

    # Pint >= 0.25 writes the micro prefix as Greek mu (U+03BC); keep the
    # micro sign (U+00B5) so labels don't change with the Pint version.
    return f"{unit:{'~P' if pretty else '~'}}".replace("μ", "µ")


def is_compatible(left: UnitLike, right: UnitLike) -> bool:
    """Whether values in one unit can be converted to the other. ``None``
    (unitless) is compatible only with ``None``.

    Pint treats steradians as dimensionless, so radiance and irradiance units
    count as compatible here; tell such quantities apart by name.
    """
    left, right = parse(left), parse(right)
    if left is None or right is None:
        return left is None and right is None
    return left.is_compatible_with(right)


def convert_values(
    values,
    source: UnitLike,
    target: UnitLike,
) -> np.ndarray:
    """Convert numeric values between compatible units."""
    values = np.asarray(values)
    source = parse(source)
    target = parse(target)

    if source is None or target is None:
        if source is target:
            return values.copy()

        raise ValueError(
            "Cannot convert between unitless and unit-bearing values."
        )

    return Q_(values, source).to(target).magnitude


def convert_quantity(quantity, target: UnitLike):
    """Convert a Pint quantity and return its magnitude."""
    if not is_quantity(quantity):
        raise TypeError("Expected a Pint quantity.")

    target = parse(target)

    if target is None:
        if quantity.dimensionless:
            return quantity.magnitude

        raise ValueError(
            "Cannot convert a unit-bearing quantity to unitless values."
        )

    return quantity.to(target).magnitude


def add(
    left_values,
    left_unit: UnitLike,
    right_values,
    right_unit: UnitLike,
):
    """Add arrays after converting the right operand to the left unit."""
    left_unit = parse(left_unit)
    right_unit = parse(right_unit)

    right_values = convert_values(
        right_values,
        right_unit,
        left_unit,
    )

    return np.asarray(left_values) + right_values, left_unit


def subtract(
    left_values,
    left_unit: UnitLike,
    right_values,
    right_unit: UnitLike,
):
    """Subtract arrays after converting the right operand to the left unit."""
    left_unit = parse(left_unit)
    right_unit = parse(right_unit)

    right_values = convert_values(
        right_values,
        right_unit,
        left_unit,
    )

    return np.asarray(left_values) - right_values, left_unit


def multiply(
    left_values,
    left_unit: UnitLike,
    right_values,
    right_unit: UnitLike,
):
    """Multiply arrays and return their combined unit."""
    left_unit = parse(left_unit)
    right_unit = parse(right_unit)

    return (
        np.asarray(left_values) * np.asarray(right_values),
        _multiply_units(left_unit, right_unit),
    )


def divide(
    left_values,
    left_unit: UnitLike,
    right_values,
    right_unit: UnitLike,
):
    """Divide arrays and return their quotient unit."""
    left_unit = parse(left_unit)
    right_unit = parse(right_unit)

    return (
        np.asarray(left_values) / np.asarray(right_values),
        _divide_units(left_unit, right_unit),
    )


def multiply_scalar(
    values,
    unit: UnitLike,
    scalar,
):
    """Multiply by a scalar or quantity, carrying quantity units."""
    unit = parse(unit)

    if is_quantity(scalar):
        return (
            np.asarray(values) * scalar.magnitude,
            _multiply_units(unit, scalar.units),
        )

    return np.asarray(values) * scalar, unit


def divide_scalar(
    values,
    unit: UnitLike,
    scalar,
):
    """Divide by a scalar or quantity, carrying quantity units."""
    unit = parse(unit)

    if is_quantity(scalar):
        return (
            np.asarray(values) / scalar.magnitude,
            _divide_units(unit, scalar.units),
        )

    return np.asarray(values) / scalar, unit


def is_unit(obj) -> bool:
    """Return whether ``obj`` is a Pint unit."""
    return isinstance(obj, pint.Unit)


def is_quantity(obj) -> bool:
    """Return whether ``obj`` is a Pint quantity."""
    return isinstance(obj, pint.Quantity)


def _multiply_units(
    left: pint.Unit | None,
    right: pint.Unit | None,
) -> pint.Unit | None:
    """Combine optional units for multiplication."""
    if left is None:
        return right

    if right is None:
        return left

    return left * right


def _divide_units(
    left: pint.Unit | None,
    right: pint.Unit | None,
) -> pint.Unit | None:
    """Combine optional units for division."""
    if right is None:
        return left

    if left is None:
        return ureg.dimensionless / right

    return left / right
