"""Common helpers for formula functions."""
from __future__ import annotations

from typing import Iterable, Iterator, Any


def flatten(args: Iterable[Any]) -> Iterator[Any]:
    """Recursively yield scalars from *args* (flattens nested iterables)."""
    for a in args:
        if isinstance(a, (list, tuple, set)):
            yield from flatten(a)
        else:
            yield a
