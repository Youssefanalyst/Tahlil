"""Plugin/Extension framework scaffold for the spreadsheet application.

Extensions live as modules inside the ``extensions`` package. Each extension
can expose one of:

1. A module-level function ``activate(app)`` accepting the running
   ``MainWindow`` instance. This is simplest.
2. A subclass of ``Extension`` implementing ``activate``.

At startup, the application calls ``extensions.load_extensions(app)`` which
imports every sub-module and attempts to activate it.

This file purposely keeps the API minimal for now; more hooks can be
introduced later (e.g., life-cycle events, settings, unload etc.).
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType
from typing import List

__all__ = ["Extension", "load_extensions"]


class Extension:
    """Base class helper for class-based plugins."""

    name: str = "Unnamed"
    version: str = "0.1"
    description: str = ""

    def activate(self, app):  # noqa: D401, D400
        """Run plugin setup using *app* (the MainWindow instance)."""
        raise NotImplementedError


def _activate_module(mod: ModuleType, app) -> bool:
    """Try to activate *mod*. Returns True on success."""
    if hasattr(mod, "activate") and callable(mod.activate):
        try:
            mod.activate(app)
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[Extension] activate() failed in {mod.__name__}: {exc}", file=sys.stderr)
            return False
    # fall-back: look for Extension subclasses
    activated = False
    for obj in mod.__dict__.values():
        if isinstance(obj, type) and issubclass(obj, Extension) and obj is not Extension:
            try:
                obj().activate(app)
                activated = True
            except Exception as exc:  # pragma: no cover
                print(f"[Extension] class activate failed in {mod.__name__}: {exc}", file=sys.stderr)
    return activated


def load_extensions(app, package_name: str = __name__) -> List[str]:
    """Import and activate all extension sub-modules.

    Returns list of loaded module names (successful activations only).
    """
    loaded: List[str] = []
    pkg = sys.modules[package_name]
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue  # skip private modules
        fullname = f"{package_name}.{info.name}"
        try:
            mod = importlib.import_module(fullname)
        except Exception as exc:  # pragma: no cover
            print(f"[Extension] failed to import {fullname}: {exc}", file=sys.stderr)
            continue
        if _activate_module(mod, app):
            loaded.append(fullname)
    return loaded
