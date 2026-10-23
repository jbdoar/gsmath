"""Summaries, limits and comparisons: one interface for everything gsmath
measures.

Data (spectra, series, images), collections of them, and results (a
uniformity, a dark model, a peak analysis) are all summarised by a
:class:`~gsmath.profile.Profile` -- a tree of named numbers with units -- and
each has a default one, :meth:`Comparable.profile`. On top of it, all of
them::

    thing.check(limits)                   # pass or fail against minimum/maximum
    thing.compare(reference, limits)      # against another of its kind (tolerance, mode)

``limits`` is a mapping shaped like the summary, as loaded from TOML, JSON or
YAML (``{"peaks": {"fwhm": {"maximum": 25}}}``), applied to the default
profile, or to ``profile=`` if given; or a Profile that already has them.

Data and collections are summarised *by* a profile, so ``summary`` is a
method taking one (:class:`Summarised`); a result *is* its summary, so for
results it is a property.
"""

from __future__ import annotations


class Comparable:
    """Has a default profile, so it can be checked and compared."""

    def profile(self):
        """The default profile: what :meth:`check` and :meth:`compare` use."""
        raise NotImplementedError(f"{type(self).__name__} has no default profile")

    def check(self, limits, *, profile=None):
        """Pass or fail against limits: a mapping whose leaves set
        ``minimum`` and/or ``maximum`` (applied to ``profile``, by default
        :meth:`profile`), or a Profile with them. Returns a
        :class:`~gsmath.comparison.Check` (a Collection of them for a
        collection)."""
        from .comparison import check
        return check(self, self._limited(limits, profile))

    def compare(self, reference, limits=None, *, profile=None):
        """Compare with ``reference`` -- another of the same kind, or for a
        collection one reference for all or a collection paired by label --
        field by field. ``limits`` sets ``tolerance`` and ``mode`` per field
        (applied to ``profile``, by default :meth:`profile`); without them
        every field is compared and none judged."""
        from .comparison import compare
        if limits is None:
            chosen = self.profile() if profile is None else profile
        else:
            chosen = self._limited(limits, profile)
        return compare(self, reference, profile=chosen)

    def _limited(self, limits, profile=None):
        from .profile import Profile
        if isinstance(limits, Profile):
            return limits
        return (self.profile() if profile is None else profile).with_limits(limits)


class Summarised(Comparable):
    """Summarised by a profile: data and collections of it."""

    def summary(self, profile=None, *, include_units=False):
        """Evaluate ``profile`` (by default :meth:`profile`): a nested mapping
        of numbers, or of ``{"value", "unit"}`` with ``include_units``."""
        from .summary import summarize
        return summarize(self, self.profile() if profile is None else profile,
                         include_units=include_units)

    def summary_units(self, profile=None):
        """The units of :meth:`summary`, as a tree like it."""
        from .summary import summarize_units
        return summarize_units(self, self.profile() if profile is None else profile)
