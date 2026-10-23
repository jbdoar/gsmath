"""Bracketed readings: a reading taken between two of a reference.

Read the reference (A), the state of interest (B), then the reference again
(A): against the mean of the two A readings, any steady drift over the three
-- a light source warming, a detector's dark wandering -- cancels, and the
drift itself is measured. Works on anything with arithmetic: plain numbers,
numpy arrays, Spectra.

    b = bracket(before, during, after)
    b.reference      # (before + after) / 2: e.g. the dark to subtract
    b.effect         # during / reference - 1: e.g. light gained by a boost
    b.drift          # after / before - 1: how much the reference moved

Drift that isn't steady over the bracket (a fast exponential warm-up) isn't
cancelled: settle first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Bracketed:
    before: Any
    during: Any
    after: Any

    @property
    def reference(self) -> Any:
        """The reference as it was during B, if it drifted steadily: the
        mean of the readings before and after."""
        return (self.before + self.after) / 2

    @property
    def effect(self) -> Any:
        """B against the reference at the time, as a fraction (0.01 = 1%)."""
        return self.during / self.reference - 1

    @property
    def difference(self) -> Any:
        """B minus the reference at the time (e.g. a dark-subtracted scan)."""
        return self.during - self.reference

    @property
    def drift(self) -> Any:
        """How much the reference moved over the bracket, as a fraction."""
        return self.after / self.before - 1


def bracket(before: Any, during: Any, after: Any) -> Bracketed:
    """Bundle a reading (`during`) with the reference readings either side."""
    return Bracketed(before, during, after)
