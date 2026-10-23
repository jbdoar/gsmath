"""What every result offers, whatever it was computed from.

A :class:`Result` -- a spectrum's peaks or colour, an image's statistics,
a display's uniformity, an edge's MTF, a camera's dark model -- has a
``summary``: a nested mapping of its numbers. ``units`` declares their
units, by summary path (``"luminance.min"``) or by name (``"min"``,
whichever channel or group it is in). With those, every result can:

- give one number with its unit: ``result.quantity("luminance.min")``;
- export: ``result.to_dict(flat=True, include_units=True)``,
  ``result.to_dataframe(include_units=True)``;
- be checked against limits, pass or fail::

      uniformity.check({"luminance": {"min_max_ratio": {"minimum": 0.8}},
                        "color": {"max_delta_uv_prime": {"maximum": 0.01}}})

- be compared with another result of its kind:
  ``measured.compare(reference, {"luminance": {"mean": {"tolerance": 0.02,
  "mode": "relative"}}})``.

Both use :meth:`Result.profile`, the result's summary as a
:class:`~gsmath.profile.Profile`.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Protocol

import numpy as np

from ..comparable import Comparable


class SpectrumLike(Protocol):
    """Structural type required by analysis implementations."""
    values: np.ndarray
    wavelength_unit: object
    value_unit: object


class Result(Comparable):
    """A result with a nested ``summary`` and declared ``units``."""

    @property
    def units(self) -> dict[str, object]:
        """Units of the summary's numbers, by dotted path or by name.

        Subclasses override this for the physical quantities they expose. A
        missing key means the result has no declared unit metadata.
        """
        return {}

    @property
    def summary(self) -> Mapping:
        """The result's numbers as a nested mapping."""
        raise NotImplementedError

    def unit(self, name):
        """The declared unit of one result: a summary path
        (``"luminance.min"``, ``"G.mean"``) or a name. A path matches its
        most specific declaration: the whole path, then shorter endings of
        it down to the last name, then the groups containing it (a default
        for all of ``"luminance"``)."""
        units = self.units
        parts = name.split(".")
        for start in range(len(parts)):
            key = ".".join(parts[start:])
            if key in units:
                return units[key]
        for end in range(len(parts) - 1, 0, -1):  # a group's default: "luminance"
            key = ".".join(parts[:end])
            if key in units:
                return units[key]
        raise KeyError(f"{type(self).__name__} has no unit metadata for {name!r}.")

    def value(self, name):
        """One result by summary path (``"luminance.min"``), or attribute name."""
        found = _lookup(self.summary, name)
        if found is _MISSING:
            if "." not in name and hasattr(self, name):
                return getattr(self, name)
            raise KeyError(f"{type(self).__name__} has no result {name!r}")
        return found

    def quantity(self, name):
        """Return one result as a Pint quantity.

        Results whose physical unit is unknown use ``None`` metadata and
        cannot safely be converted into a quantity.
        """
        from .. import units

        unit = self.unit(name)
        if unit is None:
            raise ValueError(f"No physical unit is known for {name!r}.")
        value = getattr(self, name) if hasattr(type(self), name) else self.value(name)
        return units.Q_(value, unit)

    def summary_units(self):
        """The units of the summary's numbers, as a tree like the summary
        (None where none is declared)."""
        def walk(mapping, path):
            found = {}
            for key, value in mapping.items():
                here = (*path, str(key))
                if isinstance(value, Mapping):
                    found[key] = walk(value, here)
                else:
                    try:
                        found[key] = self.unit(".".join(here))
                    except KeyError:
                        found[key] = None
            return found

        return walk(self.summary, ())

    # export
    def to_dict(self, *, flat=False, include_units=False):
        """The summary as built-in types; see :func:`gsmath.export.to_dict`."""
        from ..export import to_dict
        return to_dict(self, flat=flat, include_units=include_units)

    def to_dataframe(self, *, include_units=False):
        """The summary as a one-row DataFrame; see :func:`gsmath.export.to_dataframe`."""
        from ..export import to_dataframe
        return to_dataframe(self, include_units=include_units)

    # limits and comparisons
    def profile(self, name=None):
        """This result's summary as a :class:`~gsmath.profile.Profile`: one
        field per number, with its unit. Give it limits with
        ``with_limits``; :meth:`check` and :meth:`compare` do that for you."""
        from ..profile import summary_profile
        return summary_profile(self, name=name or type(self).__name__)

    def __str__(self):
        lines = [f"{type(self).__name__}:"]
        for key, value in self.summary.items():
            lines.append(f" {key}: {value}")
        return "\n".join(lines)


class AnalysisResult(Result, ABC):
    """A spectrum analysis: a :class:`Result` of one spectrum."""
    spectrum: SpectrumLike

    @property
    @abstractmethod
    def summary(self):
        """Return an export-friendly mapping of analysis results."""
        ...

    def __repr__(self):
        return str(self)

    @property
    def _is_empty(self) -> bool:
        """Return whether the analyzed spectrum contains no values."""
        return self.spectrum.values.size == 0


_MISSING = object()


def _lookup(mapping, path):
    """The value at a dotted ``path`` in a nested mapping. Keys may contain
    dots themselves (``"planck.cct"``), so the longest matching key wins."""
    if not isinstance(mapping, Mapping):
        return _MISSING
    if path in mapping:
        return mapping[path]
    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        head = ".".join(parts[:cut])
        if head in mapping:
            found = _lookup(mapping[head], ".".join(parts[cut:]))
            if found is not _MISSING:
                return found
    return _MISSING


def lookup(mapping, path):
    """The value at a dotted ``path`` in a nested summary (KeyError if none)."""
    found = _lookup(mapping, path)
    if found is _MISSING:
        raise KeyError(path)
    return found
