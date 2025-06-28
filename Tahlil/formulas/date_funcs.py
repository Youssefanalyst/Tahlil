"""Date/Time spreadsheet functions."""
from __future__ import annotations

import datetime as _dt
import calendar as _cal


def TODAY():  # noqa: N802
    return _dt.date.today().isoformat()


def NOW():  # noqa: N802
    return _dt.datetime.now().isoformat(sep=" ", timespec="seconds")


def YEAR(date):  # noqa: N802
    return _dt.date.fromisoformat(str(date)).year


def MONTH(date):  # noqa: N802
    return _dt.date.fromisoformat(str(date)).month


def DAY(date):  # noqa: N802
    return _dt.date.fromisoformat(str(date)).day


def HOUR(dt):  # noqa: N802
    return _dt.datetime.fromisoformat(str(dt)).hour


def MINUTE(dt):  # noqa: N802
    return _dt.datetime.fromisoformat(str(dt)).minute


def SECOND(dt):  # noqa: N802
    return _dt.datetime.fromisoformat(str(dt)).second


def WEEKDAY(date, first_day=1):  # noqa: N802
    """Return 1=Monday..7=Sunday (Excel style) or 1=Sunday..7=Saturday if first_day=0."""
    wd = _dt.date.fromisoformat(str(date)).weekday()  # 0=Mon .. 6=Sun
    if int(first_day) == 0:
        # Sunday-based
        return (wd + 1) % 7 + 1
    return wd + 1


def WEEKNUM(date):  # noqa: N802 – ISO week number
    return _dt.date.fromisoformat(str(date)).isocalendar()[1]

# --- New advanced date funcs ---

def _shift_months(dt: _dt.date, months: int) -> _dt.date:
    months_total = dt.month - 1 + int(months)
    year = dt.year + months_total // 12
    month = months_total % 12 + 1
    day = min(dt.day, _cal.monthrange(year, month)[1])
    return _dt.date(year, month, day)


def EDATE(start_date, months):  # noqa: N802
    """Date shifted by *months* months (ISO string)."""
    dt = _dt.date.fromisoformat(str(start_date))
    return _shift_months(dt, int(months)).isoformat()


def EOMONTH(start_date, months=0):  # noqa: N802
    dt = _shift_months(_dt.date.fromisoformat(str(start_date)), int(months))
    last_day = _cal.monthrange(dt.year, dt.month)[1]
    return _dt.date(dt.year, dt.month, last_day).isoformat()


def WORKDAY(start_date, days):  # noqa: N802
    """Add workdays (Mon-Fri)."""
    dt = _dt.date.fromisoformat(str(start_date))
    inc = 1 if int(days) >= 0 else -1
    remaining = abs(int(days))
    while remaining:
        dt += _dt.timedelta(days=inc)
        if dt.weekday() < 5:  # Mon-Fri
            remaining -= 1
    return dt.isoformat()
