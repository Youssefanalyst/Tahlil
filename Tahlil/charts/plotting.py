"""Plotting utilities originally defined in root ``charts.py`` now housed here.

All chart rendering code lives in this module. The package-level ``charts``
(exposed via ``charts.__init__``) re-exports :func:`plot_chart` and
:class:`ChartOptions` for external callers, so existing imports continue to
work (``from charts import plot_chart``).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from .options import ChartOptions

__all__ = ["plot_chart", "ChartOptions"]

# --- helpers -----------------------------------------------------------------

def _pie_or_donut(ax: Axes, series: pd.Series, donut: bool = False) -> None:
    """Render Pie/Donut chart from a 1-D numeric *series*."""
    series = series.dropna().astype(float)
    series = series[series > 0]
    if series.empty or math.isclose(series.sum(), 0.0):
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    wedges, _texts, _autotexts = ax.pie(
        series.values, labels=series.index, autopct="%1.1f%%"
    )
    if donut:
        centre = plt.Circle((0, 0), 0.70, fc="white")
        ax.add_artist(centre)
        ax.set_aspect("equal")


def _radar(ax: Axes, df: pd.DataFrame) -> None:
    """Radar/spider chart for ≥3 numeric columns."""
    categories = list(df.columns)
    if len(categories) < 3:
        return
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for idx, (_, row) in enumerate(df.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, label=str(idx))
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.grid(True)
    if len(df) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))


def _apply_palette(ax: Axes, palette_name: str) -> None:
    from matplotlib import cm

    if palette_name == "Tableau":
        ax.set_prop_cycle(color=cm.tab10.colors)
    elif palette_name == "Pastel":
        ax.set_prop_cycle(color=cm.Pastel1.colors)
    elif palette_name == "Dark":
        ax.set_prop_cycle(color=cm.Dark2.colors)


# --- public API --------------------------------------------------------------

def plot_chart(
    ax: Axes,
    chart_type: str,
    df_numeric: pd.DataFrame,
    params: Union[ChartOptions, Dict[str, Any], None] = None,
) -> None:  # noqa: C901
    """Draw *chart_type* on *ax* from *df_numeric*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    chart_type : str
        One of supported chart types.
    df_numeric : pandas.DataFrame
        Already-cleaned numeric selection.
    params : dict | ChartOptions | None
        Extra context captured during chart insertion (e.g. x_col for Bubble).
    """
    if params is None:
        opt = ChartOptions()
    elif isinstance(params, ChartOptions):
        opt = params
    else:
        if isinstance(params, dict):
            field_names = {f.name for f in ChartOptions.__dataclass_fields__.values()}
            known_kwargs = {k: v for k, v in params.items() if k in field_names}
            opt = ChartOptions(**known_kwargs)
            # store extra
            opt.extra.update({k: v for k, v in params.items() if k not in field_names})
        else:
            opt = ChartOptions()

    params_dict = opt.to_dict()
    palette = params_dict.get("palette", "Default")
    _apply_palette(ax, palette)

    dual = params_dict.get("dual", False)
    labels_flag = params_dict.get("labels", False)

    if chart_type == "Line":
        if dual and len(df_numeric.columns) >= 2:
            primary_cols = df_numeric.columns[:-1]
            sec_col = df_numeric.columns[-1]
            df_numeric[primary_cols].plot(ax=ax)
            ax2 = ax.twinx()
            df_numeric[[sec_col]].plot(ax=ax2, linestyle="--")
            ax2.set_ylabel(sec_col)
        else:
            df_numeric.plot(ax=ax)

    elif chart_type == "Area":
        df_numeric.plot.area(ax=ax, stacked=False)

    elif chart_type == "Stacked Bar":
        df_numeric.plot.bar(ax=ax, stacked=True)

    elif chart_type == "Bar":
        df_numeric.plot.bar(ax=ax, stacked=False)

    elif chart_type in ("Pie", "Donut"):
        if df_numeric.shape[1] != 1:
            raise ValueError("Pie/Donut chart requires exactly one numeric column.")
        _pie_or_donut(ax, df_numeric.iloc[:, 0], donut=chart_type == "Donut")

    elif chart_type == "Heatmap":
        import seaborn as sns  # type: ignore

        sns.heatmap(df_numeric, ax=ax, annot=labels_flag, fmt=".1f", cmap="viridis")

    elif chart_type == "Bubble":
        x_col = params_dict.get("x_col") or df_numeric.columns[0]
        y_col = params_dict.get("y_col") or df_numeric.columns[1]
        size_col = params_dict.get("size_col") or df_numeric.columns[2]
        ax.scatter(
            df_numeric[x_col],
            df_numeric[y_col],
            s=df_numeric[size_col] * 20,
            alpha=0.6,
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    elif chart_type == "Radar":
        _radar(ax, df_numeric)

    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    if labels_flag and chart_type not in ("Pie", "Donut", "Heatmap"):
        for line in ax.get_lines():
            ax.annotate(
                f"{line.get_ydata()[-1]:.1f}",
                (line.get_xdata()[-1], line.get_ydata()[-1]),
            )

    ax.set_title(params_dict.get("title", ""))
    ax.set_xlabel(params_dict.get("xlabel", ""))
    ax.set_ylabel(params_dict.get("ylabel", ""))

    if chart_type not in ("Pie", "Donut") and len(df_numeric.columns) > 1:
        ax.legend()
