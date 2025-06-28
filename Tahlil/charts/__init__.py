"""Charts package exposing plotting helpers and option dataclass.

Keeps public surface stable::

    from charts import plot_chart, ChartOptions
"""
from __future__ import annotations

from importlib import import_module as _im
from types import ModuleType as _Mod

# Import submodules lazily to avoid heavy matplotlib import at startup.
_plotting: _Mod = _im("charts.plotting")
_options: _Mod = _im("charts.options")

plot_chart = _plotting.plot_chart  # type: ignore[attr-defined]
ChartOptions = _options.ChartOptions  # type: ignore[attr-defined]

__all__ = ["plot_chart", "ChartOptions"]
