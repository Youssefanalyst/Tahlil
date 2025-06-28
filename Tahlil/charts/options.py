"""Dataclass encapsulating chart rendering options."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class ChartOptions:
    palette: str = "Default"
    labels: bool = False
    dual: bool = False
    # group/pie-related
    group_col: Optional[str] = None
    val_col: Optional[str] = None
    agg: str = "Sum"
    df_full: Optional[pd.DataFrame] = None
    rows: Optional[List[int]] = None
    # bubble chart
    x_col: Optional[str] = None
    y_col: Optional[str] = None
    size_col: Optional[str] = None
    # extra future-proof
    extra: Dict[str, Any] = field(default_factory=dict)

    # dict-like helper for backward-compat with existing plot_chart logic
    def get(self, key: str, default: Any = None) -> Any:  # noqa: D401
        """Mimic dict.get() to ease transition."""
        return getattr(self, key, self.extra.get(key, default))

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("extra", None)
        d.update(self.extra)
        return d
