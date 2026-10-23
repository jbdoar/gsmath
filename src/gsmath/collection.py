"""Ordered, label-addressable collections used for batch spectrum workflows."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .comparable import Summarised


@dataclass(frozen=True)
class Collection(Sequence, Summarised):
    """Store uniquely labeled objects with sequence and label lookup semantics."""
    items: tuple = ()
    label: str = ""
    metadata: dict = field(default_factory=dict)

    def __init__(
        self,
        items: Iterable = (),
        label: str = "",
        metadata: dict | None = None,
    ):
        """Create a collection and require every item label to be unique."""
        items = tuple(items)

        for item in items:
            if not hasattr(item, "label"):
                raise TypeError(
                    "Collection items must expose a label."
                )

        labels = tuple(item.label for item in items)

        if len(labels) != len(set(labels)):
            raise ValueError("Item labels must be unique.")

        object.__setattr__(self, "items", items)
        object.__setattr__(self, "label", label)
        object.__setattr__(
            self,
            "metadata",
            {} if metadata is None else dict(metadata),
        )

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.copy(items=self.items[key])

        if isinstance(key, str):
            try:
                return next(
                    item
                    for item in self
                    if item.label == key
                )
            except StopIteration:
                raise KeyError(key) from None

        return self.items[key]

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"label={self.label!r}, "
            f"size={len(self)})"
        )

    @property
    def labels(self):
        """Return item labels in collection order."""
        return tuple(item.label for item in self)

    def copy(
        self,
        *,
        items=None,
        label=None,
        metadata=None,
    ):
        """Return a collection copy with optional data and metadata changes."""
        return type(self)(
            self.items if items is None else items,
            label=self.label if label is None else label,
            metadata=(
                self.metadata
                if metadata is None
                else metadata
            ),
        )

    def append(self, item):
        """Append an item after validating its label."""
        return self.copy(
            items=(*self.items, item)
        )

    def extend(self, items):
        """Append several items while preserving label uniqueness."""
        return self.copy(
            items=(*self.items, *tuple(items))
        )

    def remove_at(self, index=-1):
        """Remove and return the item at ``index``."""
        index = range(len(self))[index]

        return self.copy(
            items=(
                item
                for i, item in enumerate(self)
                if i != index
            )
        )

    def filter(self, predicate):
        """Return items for which ``predicate`` is true."""
        return self.copy(
            items=(
                item
                for item in self
                if predicate(item)
            )
        )

    def exclude(self, predicate):
        """Return items for which ``predicate`` is false."""
        return self.filter(
            lambda item: not predicate(item)
        )

    def sort(self, *, key=None, reverse=False):
        """Return a sorted copy without mutating this collection."""
        return self.copy(
            items=sorted(
                self,
                key=key,
                reverse=reverse,
            )
        )

    def map(self, function):
        """Apply ``function`` to every item and return a new collection."""
        return self.copy(
            items=(
                function(item)
                for item in self
            )
        )

    # summaries, limits and comparisons, item by item
    def profile(self):
        """The first item's default profile: what :meth:`summary`,
        :meth:`check` and :meth:`compare` use unless given one. (An image's
        or series' default depends on its channels, so keep a collection to
        one kind with the same channels.)"""
        if not self.items:
            raise ValueError("an empty collection has no profile")
        first = self.items[0]
        if not callable(getattr(first, "profile", None)):
            raise TypeError(f"{type(first).__name__} items have no default profile; "
                            "pass a profile")
        return first.profile()

    @property
    def passed(self):
        """For a collection of checks or comparisons: True if every item
        passed, False if any failed, None if none was judged."""
        results = [item.passed for item in self if getattr(item, "passed", None) is not None]
        return all(results) if results else None

    @property
    def failed(self):
        """For a collection of checks or comparisons: the labels of the
        items that failed."""
        return tuple(item.label for item in self if getattr(item, "passed", None) is False)

    # export
    def to_dict(self, *, flat=False, include_units=False):
        """``{label: summary}`` as built-in types; see :func:`gsmath.export.to_dict`."""
        from .export import to_dict
        return to_dict(self, flat=flat, include_units=include_units)

    def to_dataframe(self, *, include_units=False):
        """One row per item; see :func:`gsmath.export.to_dataframe`."""
        from .export import to_dataframe
        return to_dataframe(self, include_units=include_units)
