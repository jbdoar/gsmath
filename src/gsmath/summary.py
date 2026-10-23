"""Evaluate declarative profiles against values and collections."""

from collections.abc import Mapping

from .collection import Collection
from . import units


def summarize(value, profile, *, include_units=False):
    """Build a nested summary by invoking every getter in ``profile``."""
    # if value is a Collection, recurse over the data
    if isinstance(value, Collection):
        # we require that labels be unique
        return {
            item.label: summarize(item, profile, include_units=include_units)
            for item in value
        }
    
    # otherwise recurse over the profile
    return _summarize(value, profile.fields, include_units=include_units)


def _summarize(value, fields, *, include_units=False):
    """Recursively evaluate a nested mapping of :class:`Field` objects."""
    # walk a nested dictionary of Fields
    result = {}

    for name, field in fields.items():
        if isinstance(field, Mapping):
            result[name] = _summarize(
                value,
                field,
                include_units=include_units,
            )
        else:
            result_value = field.getter(value)
            if include_units:
                result[name] = {
                    "value": result_value,
                    "unit": units.format_unit(field.get_unit(value)),
                }
            else:
                result[name] = result_value

    return result


def summarize_units(value, profile):
    """Return a tree of resolved Pint units parallel to a profile summary."""
    if isinstance(value, Collection):
        return {
            item.label: summarize_units(item, profile)
            for item in value
        }

    return _summarize_units(value, profile.fields)


def _summarize_units(value, fields):
    result = {}
    for name, field in fields.items():
        if isinstance(field, Mapping):
            result[name] = _summarize_units(value, field)
        else:
            result[name] = field.get_unit(value)
    return result
