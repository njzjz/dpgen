"""Small compatibility helpers for supported Python versions."""

from itertools import zip_longest


def zip_strict(*iterables):
    """Yield corresponding items, raising ValueError for unequal lengths.

    This provides the length check of ``zip(..., strict=True)`` on Python 3.9.
    Inputs are consumed lazily; the mismatch is reported when the first
    incomplete tuple would be yielded. No input is materialized in memory.
    """
    missing = object()
    for values in zip_longest(*iterables, fillvalue=missing):
        if any(value is missing for value in values):
            raise ValueError("zip_strict() arguments have different lengths")
        yield values
