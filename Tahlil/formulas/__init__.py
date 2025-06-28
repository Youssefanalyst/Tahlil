"""Central package exposing spreadsheet formula functions.

Import this package (`import formulas as f`) to access all spreadsheet functions
(e.g. `f.SUM`, `f.TEXT`). Functions are grouped into sub-modules by domain
but re-exported here for convenience. Only callables with ALL-CAPS names are
exported. This makes auto-registration trivial for the DataFrameModel formula
engine.
"""


from importlib import import_module
from types import ModuleType as _MT
from inspect import isfunction as _isfunc

# --- Import sub-modules ---
_submods = [
    "text_funcs",
    "date_funcs",
]

for _name in _submods:
    globals()[_name] = import_module(f"{__name__}.{_name}")

# Collect ALL-CAPS callables as public API
_public_funcs = {}
for _mod in (globals()[m] for m in _submods):
    for _n, _obj in _mod.__dict__.items():
        if _n.isupper() and callable(_obj):
            _public_funcs[_n] = _obj

globals().update(_public_funcs)
__all__ = list(_public_funcs)

# compute_value kept for backward compatibility
from .engine import compute_value  # noqa: E402



