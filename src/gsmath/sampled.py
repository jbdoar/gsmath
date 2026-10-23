"""What every kind of sampled data shares: values with a unit, a label and
metadata; unit-aware arithmetic; unit conversion; and cached analyses.

:class:`~gsmath.Spectrum`, :class:`~gsmath.Series` (and its line cuts and
spectral sensitivities) and :class:`~gsmath.Image` build on :class:`Sampled`.
Each says how its samples line up with another's -- the same wavelengths,
the same grid and channels -- in :meth:`Sampled._align`; everything else
about arithmetic is the same for all of them:

- data with data combines sample by sample, units combining too;
- a scalar or quantity scales every sample (adding one needs matching units);
- integer data (raw counts) becomes float first, so a dark can't wrap around.

``copy`` replaces the fields it is given. For an optional field such as
``value_unit``, passing ``None`` removes it; leaving it out keeps it.

Sampled data is immutable: its arrays are read-only and its fields can't be
reassigned once it is made -- every operation returns new data. So copies
share arrays rather than duplicating them, and an analysis, once computed,
is cached on the object for good. (The object takes over the arrays it is
given: keep no writable reference to them, or a change there would show.)
"""

from __future__ import annotations

import hashlib
import numbers

import numpy as np

from . import units
from .comparable import Summarised

KEEP = object()
"""Default of ``copy`` arguments that may be set to None: keep the field."""


class Sampled(Summarised):
    """Values with a unit, a label and metadata. Subclasses store
    ``values``, ``label``, ``value_unit`` and ``metadata`` and implement
    ``copy`` and :meth:`_align`."""

    values: np.ndarray
    label: str
    value_unit: object
    metadata: dict

    def _seal(self):
        """End of construction: from here on the fields can't be reassigned."""
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value):
        if self.__dict__.get("_sealed") and not name.startswith("_"):
            raise AttributeError(f"{type(self).__name__} is immutable: use "
                                 f"copy({name}=...) for one with a new {name}")
        object.__setattr__(self, name, value)

    # arithmetic
    def __add__(self, other):
        return self._binary_op(other, "add", "+")

    def __sub__(self, other):
        return self._binary_op(other, "subtract", "-")

    def __mul__(self, other):
        return self._binary_op(other, "multiply", "*")

    def __truediv__(self, other):
        return self._binary_op(other, "divide", "/")

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, numbers.Real):
            return self.copy(values=np.subtract(other, as_float(self.values)),
                             label=f"{other} - ({self.label})")
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, numbers.Real):
            return self.copy(values=np.divide(other, as_float(self.values)),
                             label=f"{other} / ({self.label})")
        return NotImplemented

    def __neg__(self):
        return self.copy(values=-as_float(self.values), label=f"-({self.label})")

    def __abs__(self):
        return self.copy(values=np.abs(self.values), label=f"abs({self.label})")

    def __pow__(self, exponent):
        if not isinstance(exponent, numbers.Real):
            return NotImplemented
        return self.copy(values=np.power(as_float(self.values), exponent),
                         label=f"({self.label}) ** {exponent}")

    def _align(self, other):
        """``(left values, right values, copy changes)`` for combining with
        another Sampled -- raising if their samples don't line up -- or
        NotImplemented for a kind this one doesn't combine with."""
        return NotImplemented

    def _binary_op(self, other, operation, symbol):
        """Data/data and data/scalar arithmetic, with the result's unit."""
        if isinstance(other, Sampled):
            aligned = self._align(other)
            if aligned is NotImplemented:
                return NotImplemented
            left, right, changes = aligned
            values, value_unit = getattr(units, operation)(
                left, self.value_unit, right, other.value_unit)
            return self.copy(values=values, value_unit=value_unit,
                             label=f"({self.label}) {symbol} ({other.label})", **changes)

        if isinstance(other, numbers.Real) or units.is_quantity(other):
            values = as_float(self.values)
            if operation in {"add", "subtract"}:
                if units.is_quantity(other):
                    scalar = units.convert_quantity(other, self.value_unit)
                elif self.value_unit is not None:
                    raise ValueError(
                        "Cannot add or subtract a unitless scalar from unit-bearing values.")
                else:
                    scalar = other
                values = values + scalar if operation == "add" else values - scalar
                value_unit = self.value_unit
            else:
                helper = units.multiply_scalar if operation == "multiply" else units.divide_scalar
                values, value_unit = helper(values, self.value_unit, other)
            return self.copy(values=values, value_unit=value_unit,
                             label=f"({self.label}) {symbol} {other}")

        return NotImplemented

    # units
    def to_value_unit(self, unit):
        """Return a copy with values converted to ``unit``."""
        unit = units.parse(unit)
        return self.copy(values=units.convert_values(self.values, self.value_unit, unit),
                         value_unit=unit)

    # analyses
    def _get_analysis(self, cls, **kwargs):
        """``cls(self, **kwargs)``, computed once: the data can't change, so
        neither can the analysis."""
        cache = self.__dict__.setdefault("_analysis_cache", {})
        key = (cls, tuple(sorted(kwargs.items(), key=lambda item: item[0])))
        try:
            hash(key)
        except TypeError:  # an unhashable argument (an ROI, say): not cached
            return cls(self, **kwargs)
        if key not in cache:
            cache[key] = cls(self, **kwargs)
        return cache[key]


def read_only(array):
    """``array`` as a read-only array (a read-only view if it was writable)."""
    array = np.asarray(array)
    if array.flags.writeable:
        array = array.view()
        array.flags.writeable = False
    return array


def as_float(values):
    """Integer data (raw counts) as float, so a dark doesn't wrap around."""
    values = np.asarray(values)
    return values.astype(np.float64) if values.dtype.kind in "biu" else values


def array_signature(array):
    """A content key for an array: shape, dtype and a hash of its bytes."""
    array = np.ascontiguousarray(array)
    digest = hashlib.blake2b(array.tobytes(), digest_size=16).digest()
    return array.shape, array.dtype.str, digest
