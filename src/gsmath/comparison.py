"""Profile-driven comparisons for individual values and labeled collections."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .collection import Collection
from .profile import Field, Profile
from .summary import summarize
from ._utils import safe_divide, to_builtin
from . import units


@dataclass(frozen=True)
class ComparisonRecord:
    """Capture the inputs, difference, rule, and outcome for one field."""
    measured: object
    reference: object
    difference: object
    field: Field
    passed: bool | None
    unit: object = None


@dataclass(frozen=True)
class Comparison:
    """
    The result of comparing two individual values.

    A comparison of collections is represented as a
    Collection of comparison objects.
    """
    
    tree: dict
    label: str = ""

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"label={self.label!r}, "
            f"passed={self.passed!r})"
        )

    @property
    def leaves(self):
        """Return all ComparisonRecord leaves."""
        return tuple(_iter_records(self.tree))

    @property
    def passed(self):
        """
        Overall evaluation result.

        Returns
        -------
        True
            All evaluated fields passed.
        False
            At least one evaluated field failed.
        None
            No fields were evaluated because no tolerances were specified.
        """
        results = [
            record.passed
            for record in self.leaves
            if record.passed is not None
        ]

        return all(results) if results else None

    @property
    def records(self):
        """ComparisonRecord leaves keyed by dotted field path, in tree order."""
        return dict(_iter_record_items(self.tree))

    @property
    def failed(self):
        """
        Dotted paths of failed comparison fields.

        Returns an empty tuple when no fields failed or when no
        fields were evaluated.
        """
        return tuple(
            path
            for path, record in _iter_record_items(self.tree)
            if record.passed is False
        )

    @property
    def summary(self):
        """Return a nested, export-friendly comparison summary."""
        result = {
            "label": self.label,
            "results": _summarize_tree(self.tree),
        }

        if self.evaluated:
            result["passed"] = self.passed
            result["failed"] = self.failed

        return result

    @property
    def evaluated(self):
        """Return whether at least one field supplied a tolerance."""
        return any(
            record.passed is not None
            for record in self.leaves
        )

    def summary_units(self):
        """Return units parallel to the comparison summary tree."""
        result = {
            "label": None,
            "results": _summarize_unit_tree(self.tree),
        }

        if self.evaluated:
            result["passed"] = None
            result["failed"] = None

        return result


def compare(measured, reference, *, profile):
    """
    Compare two values according to a Profile.

    Supported forms:

        compare(value, value)
            -> Comparison
    
        compare(collection, collection)
            -> Collection[Comparison]
    
        compare(collection, value)
            -> Collection[Comparison]

    When comparing two collections, items are paired by label.
    When comparing a collection against one value, that value
    is used as the reference for every collection item.
    """
    if not isinstance(profile, Profile):
        raise TypeError("profile must be a Profile")

    if isinstance(measured, Collection):
        return _compare_collection(
            measured,
            reference,
            profile=profile,
        )

    if isinstance(reference, Collection):
        raise TypeError(
            "Cannot compare a single measured value "
            "against a reference Collection"
        )

    measured_summary = summarize(measured, profile)
    reference_summary = summarize(reference, profile)

    return Comparison(
        tree=_compare_nested(
            measured_summary,
            reference_summary,
            profile.fields,
            measured_source=measured,
            reference_source=reference,
        ),
        label=getattr(measured, "label", ""),
    )


def _compare_collection(measured, reference, *, profile):
    """
    Compare each member of a measured Collection.

    A reference Collection is paired by label. A single reference
    value is compared against every measured collection member.
    """
    if isinstance(reference, Collection):
        _validate_matching_collection_labels(
            measured,
            reference,
        )

        comparisons = (
            compare(
                item,
                reference[item.label],
                profile=profile,
            )
            for item in measured
        )

    else:
        comparisons = (
            compare(
                item,
                reference,
                profile=profile,
            )
            for item in measured
        )

    return Collection(
        comparisons,
        label=measured.label,
        metadata=measured.metadata,
    )


def _validate_matching_collection_labels(measured, reference):
    """Require measured and reference collections to have equal labels."""
    measured_labels = set(measured.labels)
    reference_labels = set(reference.labels)

    if measured_labels == reference_labels:
        return

    missing_reference = measured_labels - reference_labels
    missing_measured = reference_labels - measured_labels

    details = []

    if missing_reference:
        details.append(
            "missing from reference: "
            f"{sorted(missing_reference)!r}"
        )

    if missing_measured:
        details.append(
            "missing from measured: "
            f"{sorted(missing_measured)!r}"
        )

    raise ValueError(
        "Measured and reference collections must have "
        "matching labels; "
        + "; ".join(details)
    )


def _compare_nested(
    measured,
    reference,
    fields,
    *,
    measured_source,
    reference_source,
):
    """
    Walk matching summary trees using the corresponding Field tree.
    """
    result = {}

    for name, field in fields.items():
        if name not in measured:
            raise KeyError(
                f"Measured summary is missing field {name!r}"
            )

        if name not in reference:
            raise KeyError(
                f"Reference summary is missing field {name!r}"
            )

        if isinstance(field, Mapping):
            result[name] = _compare_nested(
                measured[name],
                reference[name],
                field,
                measured_source=measured_source,
                reference_source=reference_source,
            )

        elif isinstance(field, Field):
            result[name] = _compare_leaf(
                measured[name],
                reference[name],
                field,
                measured_source=measured_source,
                reference_source=reference_source,
            )

        else:
            raise TypeError(
                f"Profile entry {name!r} must be a Field "
                "or nested mapping"
            )

    return result


def _compare_leaf(
    measured,
    reference,
    field,
    *,
    measured_source,
    reference_source,
):
    """Compare and evaluate one pair of summary leaves."""
    measured_unit = field.get_unit(measured_source)
    reference_unit = field.get_unit(reference_source)
    if reference_unit != measured_unit:
        reference = units.convert_values(
            reference,
            reference_unit,
            measured_unit,
        )

    difference = _compare_value(
        measured,
        reference,
        field,
    )

    return ComparisonRecord(
        measured=measured,
        reference=reference,
        difference=difference,
        field=field,
        passed=_evaluate_value(difference, field),
        unit=measured_unit,
    )


def _compare_value(measured, reference, field):
    """
    Compare one pair of summary leaves.

    This is the main seam for adding quantity-aware arithmetic later.
    """
    if field.comparator is not None:
        return field.comparator(measured, reference)

    measured_array = np.asarray(measured)
    reference_array = np.asarray(reference)

    if measured_array.shape != reference_array.shape:
        raise ValueError(
            "Cannot compare values with different shapes: "
            f"{measured_array.shape} and "
            f"{reference_array.shape}"
        )

    try:
        difference = measured_array - reference_array
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Comparison fields must be numeric unless "
            "a custom comparator is supplied"
        ) from exc

    if field.mode == "difference":
        result = difference

    elif field.mode == "relative":
        result = safe_divide(
            difference,
            reference_array,
        )

    elif field.mode == "percent":
        result = 100 * safe_divide(
            difference,
            reference_array,
        )

    elif field.mode == "ratio":
        result = safe_divide(
            measured_array,
            reference_array,
        )

    else:
        raise ValueError(
            f"Unknown comparison mode: {field.mode!r}"
        )

    return to_builtin(result)


def _evaluate_value(difference, field):
    """
    Evaluate one comparison result against its Field tolerance.
    """
    if field.tolerance is None:
        return None

    difference_array = np.asarray(difference)
    tolerance_array = np.asarray(field.tolerance)

    if field.mode == "ratio":
        error = difference_array - 1
    else:
        error = difference_array

    return bool(
        np.all(
            np.abs(error) <= tolerance_array
        )
    )


def _iter_records(tree):
    """Yield every ComparisonRecord in a nested comparison tree."""
    for value in tree.values():
        if isinstance(value, Mapping):
            yield from _iter_records(value)

        elif isinstance(value, ComparisonRecord):
            yield value

        else:
            raise TypeError(
                "Comparison tree leaves must be ComparisonRecord "
                f"objects, got {type(value).__name__}."
            )


def _iter_record_items(tree, parent=()):
    """Yield dotted paths and ComparisonRecord leaves."""
    for name, value in tree.items():
        path = (*parent, name)

        if isinstance(value, Mapping):
            yield from _iter_record_items(
                value,
                parent=path,
            )

        elif isinstance(value, ComparisonRecord):
            yield ".".join(path), value

        else:
            raise TypeError(
                "Comparison tree leaves must be ComparisonRecord "
                f"objects, got {type(value).__name__}."
            )


def _summarize_tree(tree):
    """
    Convert a comparison tree into nested builtin-oriented dictionaries.
    """
    result = {}

    for name, value in tree.items():
        if isinstance(value, Mapping):
            result[name] = _summarize_tree(value)

        elif isinstance(value, ComparisonRecord):
            record = {
                "measured": to_builtin(value.measured),
                "reference": to_builtin(value.reference),
                "difference": to_builtin(value.difference),
                "mode": value.field.mode,
            }

            if value.field.tolerance is not None:
                record["tolerance"] = to_builtin(
                    value.field.tolerance
                )
                record["passed"] = value.passed

            if value.field.uncertainty is not None:
                record["uncertainty"] = to_builtin(
                    value.field.uncertainty
                )

            result[name] = record

        else:
            raise TypeError(
                "Comparison tree leaves must be ComparisonRecord "
                f"objects, got {type(value).__name__}."
            )

    return result


def _summarize_unit_tree(tree):
    """Build unit metadata matching :attr:`Comparison.summary`."""
    result = {}

    for name, value in tree.items():
        if isinstance(value, Mapping):
            result[name] = _summarize_unit_tree(value)
        elif isinstance(value, ComparisonRecord):
            difference_unit = (
                value.unit
                if value.field.mode == "difference"
                else units.ureg.dimensionless
            )
            record = {
                "measured": value.unit,
                "reference": value.unit,
                "difference": difference_unit,
                "mode": None,
            }
            if value.field.tolerance is not None:
                record["tolerance"] = difference_unit
                record["passed"] = None
            if value.field.uncertainty is not None:
                record["uncertainty"] = value.unit
            result[name] = record
        else:
            raise TypeError(
                "Comparison tree leaves must be ComparisonRecord "
                f"objects, got {type(value).__name__}."
            )

    return result


@dataclass(frozen=True)
class CheckRecord:
    """One field's value, its limits and whether it is within them."""
    value: object
    field: Field
    passed: bool | None
    unit: object = None


@dataclass(frozen=True)
class Check:
    """A value checked against a specification's limits: each field's
    ``minimum`` and/or ``maximum``, pass or fail. Fields without limits are
    reported but not judged."""
    tree: dict
    label: str = ""

    def __repr__(self):
        return f"{type(self).__name__}(label={self.label!r}, passed={self.passed!r})"

    @property
    def records(self):
        """CheckRecord leaves keyed by dotted field path, in tree order."""
        return dict(_iter_items(self.tree, CheckRecord))

    @property
    def passed(self):
        """True if every judged field is within its limits, False if any
        isn't, None if no field has limits."""
        results = [r.passed for r in self.records.values() if r.passed is not None]
        return all(results) if results else None

    @property
    def failed(self):
        """Dotted paths of the fields outside their limits."""
        return tuple(path for path, r in self.records.items() if r.passed is False)

    @property
    def summary(self):
        """A nested, export-friendly summary: each field's value, limits
        and outcome, then the overall result."""
        def walk(tree):
            found = {}
            for name, value in tree.items():
                if isinstance(value, Mapping):
                    found[name] = walk(value)
                    continue
                record = {"value": to_builtin(value.value)}
                for limit in ("minimum", "maximum"):
                    if getattr(value.field, limit) is not None:
                        record[limit] = to_builtin(getattr(value.field, limit))
                if value.passed is not None:
                    record["passed"] = value.passed
                found[name] = record
            return found

        return {"label": self.label, "results": walk(self.tree), "passed": self.passed,
                "failed": self.failed}

    def summary_units(self):
        """Units parallel to :attr:`summary`."""
        def walk(tree):
            found = {}
            for name, value in tree.items():
                if isinstance(value, Mapping):
                    found[name] = walk(value)
                    continue
                record = {"value": value.unit}
                for limit in ("minimum", "maximum"):
                    if getattr(value.field, limit) is not None:
                        record[limit] = value.unit
                if value.passed is not None:
                    record["passed"] = None
                found[name] = record
            return found

        return {"label": None, "results": walk(self.tree), "passed": None, "failed": None}


def check(value, profile):
    """Check ``value`` against the limits in ``profile``: every field with a
    ``minimum`` and/or ``maximum`` passes when all of its numbers are within
    them (NaN fails). A Collection gives a Collection of checks."""
    if not isinstance(profile, Profile):
        raise TypeError("profile must be a Profile")
    if isinstance(value, Collection):
        return Collection([check(item, profile) for item in value])

    def walk(fields):
        found = {}
        for name, field in fields.items():
            if isinstance(field, Mapping):
                found[name] = walk(field)
            elif isinstance(field, Field):
                measured = field.getter(value)
                found[name] = CheckRecord(measured, field, _within(measured, field),
                                          field.get_unit(value))
            else:
                raise TypeError(f"Profile entry {name!r} must be a Field or nested mapping")
        return found

    return Check(walk(profile.fields), label=getattr(value, "label", "") or "")


def _within(value, field):
    if field.minimum is None and field.maximum is None:
        return None
    values = np.asarray(value, dtype=float)
    ok = np.isfinite(values)
    if field.minimum is not None:
        ok &= values >= np.asarray(field.minimum, dtype=float)
    if field.maximum is not None:
        ok &= values <= np.asarray(field.maximum, dtype=float)
    return bool(np.all(ok))


def _iter_items(tree, kind, parent=()):
    for name, value in tree.items():
        path = (*parent, name)
        if isinstance(value, Mapping):
            yield from _iter_items(value, kind, path)
        elif isinstance(value, kind):
            yield ".".join(path), value
        else:
            raise TypeError(f"tree leaves must be {kind.__name__} objects, "
                            f"got {type(value).__name__}.")
