"""Declarative field profiles shared by summaries and comparisons."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from . import units

COMPARISON_MODES = ("difference", "relative", "percent", "ratio")

# Field settings a limits mapping may set; getters and units stay in code.
# tolerance/mode/uncertainty judge a comparison with a reference;
# minimum/maximum judge a value on its own (see gsmath.comparison.check).
LIMIT_KEYS = ("tolerance", "mode", "uncertainty", "minimum", "maximum")


@dataclass(frozen=True)
class Profile:
    """Name a nested tree of fields to extract from an object."""
    name: str = ""
    fields: dict = field(default_factory=dict)

    def with_limits(self, limits, *, name=None):
        """Return a profile of the fields named in ``limits``, with new settings.

        ``limits`` mirrors this profile's field tree (as loaded from a TOML,
        JSON or YAML file). Each leaf names one field and maps to the settings
        to apply: ``tolerance``, ``mode`` and ``uncertainty`` for comparisons,
        ``minimum`` and ``maximum`` for checks of the value itself
        (:func:`~gsmath.comparison.check`). Fields that are
        not named are left out, so the mapping selects what gets compared as
        well as how. Unknown paths and settings raise, so a typo cannot
        silently drop a check::

            limits = {"peaks": {"fwhm": {"tolerance": 3.0}},
                      "power": {"integral": {"mode": "relative", "tolerance": 0.1}}}
            profile = SPECTRUM_PROFILE.with_limits(limits)
        """
        if not isinstance(limits, Mapping):
            raise TypeError("limits must be a mapping of field paths to settings")
        return Profile(
            name=self.name if name is None else name,
            fields=_apply_limits(self.fields, limits, ()),
        )


@dataclass(frozen=True)
class Field:
    """Describe how to extract, compare, and evaluate one summary leaf."""
    getter: Callable
    mode: str = "difference"
    tolerance: object = None
    uncertainty: object = None
    comparator: Callable | None = None
    unit: object = None
    minimum: object = None
    """The least acceptable value (in the field's unit), for checks."""
    maximum: object = None
    """The greatest acceptable value (in the field's unit), for checks."""

    def get_unit(self, value):
        """Resolve this field's static or source-dependent unit."""
        unit = self.unit(value) if callable(self.unit) else self.unit
        return units.parse(unit)


def _apply_limits(fields, limits, path):
    """Select and update the fields named by a nested limits mapping."""
    selected = {}

    for name, settings in limits.items():
        dotted = ".".join((*path, name))

        if name not in fields:
            raise KeyError(f"Unknown profile field {dotted!r}.")

        node = fields[name]

        if not isinstance(settings, Mapping):
            raise TypeError(
                f"Limits for {dotted!r} must be a mapping, "
                f"got {type(settings).__name__}."
            )

        if not isinstance(node, Field):
            selected[name] = _apply_limits(node, settings, (*path, name))
            continue

        unknown = set(settings) - set(LIMIT_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown setting(s) {sorted(unknown)} for {dotted!r}; "
                f"expected any of {list(LIMIT_KEYS)}."
            )

        mode = settings.get("mode", node.mode)
        if mode not in COMPARISON_MODES:
            raise ValueError(
                f"Unknown comparison mode {mode!r} for {dotted!r}; "
                f"expected one of {list(COMPARISON_MODES)}."
            )

        selected[name] = replace(node, **settings)

    return selected


def summary_profile(result, *, name="", of=None):
    """A Profile with one field per number in ``result``'s summary (a
    :class:`~gsmath.analysis.base.Result`), each with its declared unit.

    Each field reads its number from the object it is given -- the result
    itself, or ``of(source)`` for a source the result is computed from, e.g.
    ``of=lambda image: image.stats()`` -- so the profile can summarise and
    compare other objects of the same kind."""
    of = (lambda source: source) if of is None else of

    def fields(mapping, path):
        found = {}
        for key, value in mapping.items():
            here = (*path, str(key))
            if isinstance(value, Mapping):
                nested = fields(value, here)
                if nested:
                    found[key] = nested
            elif _numeric(value):
                found[key] = _summary_field(".".join(here), of)
        return found

    return Profile(name=name, fields=fields(result.summary, ()))


def _summary_field(path, of):
    from .analysis.base import lookup

    def unit(source):
        try:
            return of(source).unit(path)
        except KeyError:
            return None

    return Field(lambda source: lookup(of(source).summary, path), unit=unit)


def _numeric(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return False
    try:
        import numpy as np
        return np.asarray(value).dtype.kind in "iuf"
    except (TypeError, ValueError):
        return False


def from_analysis(method, fields, *, options=None, relative=()):
    """Profile fields read from one of an object's analyses -- no getters to
    write. ``method`` names the analysis (``"peaks"``, ``"color"``,
    ``"stats"``, ...), called with ``options`` (``{"observer": "2deg"}``);
    ``fields`` names the results to take: a sequence of names, or a mapping
    ``{field name: result name}`` to rename (``{"cct": "planck.cct"}``,
    dotted for nested results). Each field's unit is the one the analysis
    declares for it; those named in ``relative`` compare in relative mode::

        Profile("LED", {
            "peaks": from_analysis("peaks", ["peak_wavelength", "fwhm"]),
            "photometry": from_analysis("color", ["Y"], options={"observer": "2deg"},
                                   relative=["Y"]),
        })
    """
    from operator import attrgetter

    options = dict(options or {})
    names = dict(fields) if isinstance(fields, Mapping) else {name: name for name in fields}
    unknown = set(relative) - set(names)
    if unknown:
        raise KeyError(f"relative names fields not taken: {sorted(unknown)}")

    def result_of(source):
        return getattr(source, method)(**options)

    def field(name, path):
        value = attrgetter(path)
        return Field(lambda source: value(result_of(source)),
                     mode="relative" if name in relative else "difference",
                     unit=lambda source: result_of(source).unit(path))

    return {name: field(name, path) for name, path in names.items()}


SPECTRUM_PROFILE = Profile(
    name="spectrum",
    fields={
        "power": from_analysis("power", ["integral", "centroid_wavelength"],
                          relative=["integral"]),
        # power's centroid is the whole spectrum's power-weighted mean
        # wavelength; peaks' is local to a few samples around the strongest peak
        "peaks": from_analysis("peaks", ["peak_wavelength", "peak_wavelength_polyfit",
                                    "centroid_wavelength", "center_wavelength", "fwhm"]),
        "color": {
            "2deg": from_analysis(
                "color",
                {"X": "X", "Y": "Y", "Z": "Z", "x": "x", "y": "y", "u_prime": "u_prime",
                 "v_prime": "v_prime", "cct": "planck.cct", "duv": "planck.duv",
                 "L_star": "L_star", "a_star": "a_star", "b_star": "b_star",
                 "u_star": "u_star", "v_star": "v_star"},
                options={"observer": "2deg"}, relative=["X", "Y", "Z"]),
        },
        "statistics": from_analysis("stats", ["min", "max", "mean", "std"]),
    },
)
