"""Small conversion, numeric, and nested-mapping helpers used internally."""
from collections.abc import Mapping

import numpy as np


# generic conversion
def to_builtin(value):
    """Recursively convert NumPy objects to built-in Python types."""
    if isinstance(value, Mapping):
        return {
            key: to_builtin(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(to_builtin(item) for item in value)

    if isinstance(value, list):
        return [to_builtin(item) for item in value]

    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())

    if isinstance(value, np.generic):
        return value.item()

    return value


def safe_divide(numerator, denominator):
    """Divide numerically, returning NaN wherever the denominator is zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )


def factor_pair(factor):
    """``(rows, columns)`` from one positive integer or a pair (binning)."""
    fy, fx = np.broadcast_to(np.asarray(factor), (2,)).tolist()
    if not all(isinstance(f, int) and f >= 1 for f in (fy, fx)):
        raise ValueError(f"binning factors must be positive integers, not {factor!r}")
    return fy, fx


# nested mapping utilities
def format_path(path):
    """Render a tuple of nested mapping keys as a dotted path."""
    return ".".join(map(str, path))


def get_nested(mapping, path):
    """Retrieve a value from a mapping using a dotted key path."""
    value = mapping

    for key in path.split("."):
        try:
            value = value[key]
        except (KeyError, TypeError) as exc:
            raise KeyError(path) from exc

    return value


def set_nested(mapping, path, value):
    """Set a dotted path, creating missing intermediate dictionaries."""
    keys = path.split(".")
    current = mapping
    for key in keys[:-1]:
        current = current.setdefault(key, {})

    current[keys[-1]] = value


def map_nested(mapping, func, *, _path=()):
    """Transform every leaf while preserving a nested mapping's shape."""
    result = {}

    for key, value in mapping.items():
        path = (*_path, key)

        if isinstance(value, Mapping):
            result[key] = map_nested(value, func, _path=path)
        else:
            result[key] = func(path, value)

    return result


def map2_nested(left, right, func, *, _path=()):
    """Transform paired leaves from two structurally identical mappings."""
    _validate_matching_keys(left, right, path=_path)

    result = {}

    for key in left:
        path = (*_path, key)
        left_value = left[key]
        right_value = right[key]

        left_is_mapping = isinstance(left_value, Mapping)
        right_is_mapping = isinstance(right_value, Mapping)

        if left_is_mapping != right_is_mapping:
            raise ValueError(f"Structure mismatch at {format_path(path)!r}.")

        if left_is_mapping:
            result[key] = map2_nested(left_value, right_value, func, _path=path)
        else:
            result[key] = func(path, left_value, right_value)

    return result


def _validate_matching_keys(left, right, *, path=()):
    """Raise ValueError if two nested mappings differ in structure."""
    left_is_mapping = isinstance(left, Mapping)
    right_is_mapping = isinstance(right, Mapping)

    if left_is_mapping != right_is_mapping:
        raise ValueError(
            f"Structure mismatch at {format_path(path)!r}: "
            f"{type(left).__name__} != {type(right).__name__}"
        )

    if not left_is_mapping:
        return

    left_keys = set(left)
    right_keys = set(right)

    if left_keys != right_keys:
        missing = sorted(left_keys - right_keys)
        extra = sorted(right_keys - left_keys)

        raise ValueError(
            f"Key mismatch at {format_path(path)!r}; "
            f"missing: {missing}; extra: {extra}"
        )
