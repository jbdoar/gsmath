"""Convert summaries and comparisons into built-ins and tabular forms."""

from collections.abc import Mapping
from pprint import pprint

import pandas as pd

from .collection import Collection
from ._utils import format_path, to_builtin
from . import units


def to_dict(value, *, flat=False, include_units=False):
    """Return an export-safe dictionary, optionally including unit metadata."""
    if isinstance(value, Mapping):
        mapping = value

    elif isinstance(value, Collection):
        mapping = {
            item.label: _get_summary(item, include_units=include_units)
            for item in value
        }

    else:
        mapping = _get_summary(value, include_units=include_units)

    mapping = to_builtin(mapping)

    if flat:
        return _flatten_nested(
            mapping,
        )

    return mapping


def print_result(value, *, flat=False, include_units=False):
    """Print a summary-like object."""
    data = to_dict(value, flat=flat, include_units=include_units)

    if flat:
        _print_mapping(data)
    else:
        pprint(data, sort_dicts=False)


def to_dataframe(value, *, include_units=False):
    """Convert a summary or comparison into a pandas DataFrame.

    With ``include_units=True``, known units are appended to column labels so
    the underlying cells remain numeric and useful to pandas.
    """
    if isinstance(value, Collection):
        rows = []

        for item in value:
            row = {
                "label": item.label,
                **_flatten_nested(
                    to_builtin(_get_summary(item)),
                ),
            }
            rows.append(row)

        dataframe = pd.DataFrame(rows)
    else:
        dataframe = pd.DataFrame([
            to_dict(
                value,
                flat=True,
            )
        ])

    if include_units:
        dataframe = _label_unit_columns(dataframe, value)

    return dataframe


def _get_summary(value, *, include_units=False):
    """Resolve objects exposing ``summary`` while accepting mappings directly."""
    try:
        summary = value.summary
    except AttributeError:
        raise TypeError(
            f"{type(value).__name__} does not expose a summary."
        ) from None

    if callable(summary):
        summary = summary(include_units=include_units)
    elif include_units and hasattr(value, "summary_units"):
        # A result's summary property: pair each number with its unit.
        summary = _with_units(summary, value.summary_units())

    if not isinstance(summary, Mapping):
        raise TypeError(
            f"{type(value).__name__}.summary must be a mapping."
        )

    return summary


def _with_units(summary, unit_tree):
    """``{"value": ..., "unit": ...}`` leaves, as summarize(include_units=True) gives."""
    result = {}
    for key, value in summary.items():
        unit = unit_tree.get(key) if isinstance(unit_tree, Mapping) else None
        if isinstance(value, Mapping):
            result[key] = _with_units(value, unit or {})
        else:
            result[key] = {"value": value, "unit": units.format_unit(unit)}
    return result


def _label_unit_columns(dataframe, value):
    """Append resolved unit labels without converting numeric cells."""
    candidates = list(value) if isinstance(value, Collection) else [value]
    mappings = []

    for candidate in candidates:
        resolver = getattr(candidate, "summary_units", None)
        if resolver is not None:
            mappings.append(_flatten_nested(resolver()))

    if not mappings:
        return dataframe

    first = mappings[0]
    if any(mapping != first for mapping in mappings[1:]):
        raise ValueError(
            "Cannot label report columns for items with different units."
        )

    renamed = {}
    for column, unit in first.items():
        label = units.format_unit(unit)
        if label is not None and column in dataframe.columns:
            renamed[column] = f"{column} [{label}]"

    return dataframe.rename(columns=renamed)


def _walk_nested(mapping, *, _path=()):
    """Yield ``(path, value)`` pairs for each leaf of a nested mapping."""
    for key, value in mapping.items():
        path = (*_path, key)

        if isinstance(value, Mapping):
            yield from _walk_nested(value, _path=path)
        else:
            yield path, value

            
def _flatten_nested(mapping):
    """Flatten a nested mapping using dotted paths."""
    return {
        format_path(path): value
        for path, value in _walk_nested(mapping)
    }


def _print_mapping(mapping):
    """Print a mapping with aligned keys."""
    width = max((len(str(key)) for key in mapping), default=0)

    for key, value in mapping.items():
        print(f"{key:<{width}} : {value}")
