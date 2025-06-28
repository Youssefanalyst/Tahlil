"""Text-related spreadsheet functions."""
from __future__ import annotations

import re
from .common import flatten


def LEN(text):  # noqa: N802
    return len(str(text))


def LEFT(text, n):  # noqa: N802
    return str(text)[: int(n)]


def RIGHT(text, n):  # noqa: N802
    return str(text)[-int(n):]


def CONCAT(*args):  # noqa: N802
    return "".join(str(a) for a in flatten(args))


def TRIM(text):  # noqa: N802
    return str(text).strip()


def UPPER(text):  # noqa: N802
    return str(text).upper()


def LOWER(text):  # noqa: N802
    return str(text).lower()


def SUBSTITUTE(text, old, new, instance=None):  # noqa: N802
    text = str(text)
    old = str(old)
    new = str(new)
    if instance is None:
        return text.replace(old, new)
    instance = int(instance)
    parts = text.split(old)
    if instance <= 0 or instance > len(parts) - 1:
        return text  # out of range – no change
    parts[instance - 1] += new
    return old.join(parts)


def FIND(substr, text, start=1):  # noqa: N802
    try:
        pos = str(text).index(str(substr), int(start) - 1)
        return pos + 1
    except ValueError:
        return 0


def REPLACE(text, start, length, new_text):  # noqa: N802
    s = str(text)
    start = int(start) - 1
    length = int(length)
    return s[:start] + str(new_text) + s[start + length:]


def VALUE(text):  # noqa: N802
    try:
        return float(text)
    except Exception:
        return "#ERR"


def REPT(text, n):  # noqa: N802
    return str(text) * int(n)


def SEARCH(find_text, within_text, start_num=1):  # noqa: N802
    """Case-insensitive search (1-based). Returns 0 if not found."""
    try:
        pos = str(within_text).lower().find(str(find_text).lower(), int(start_num) - 1)
        return pos + 1 if pos != -1 else 0
    except Exception:
        return 0


# Python format mini-language & strftime wrapper
import datetime as _dt
import numbers as _nums


def TEXT(value, fmt):  # noqa: N802
    try:
        if isinstance(value, (_dt.date, _dt.datetime)):
            return _dt.datetime.fromisoformat(str(value)).strftime(str(fmt))
        if isinstance(value, _nums.Number):
            return format(float(value), str(fmt))
        return format(str(value), str(fmt))
    except Exception:
        return str(value)
