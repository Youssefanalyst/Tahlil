"""Interactive Chart Builder dialog.

Allows the user to configure a chart visually:
1. Select chart type (Line, Area, Stacked Bar, etc.).
2. Choose numeric columns to plot (multi-select).
3. (Optional) aggregation choice for Pie/Donut when grouping is needed – reserved for future.
4. Live preview that updates immediately on any change.

This is an initial implementation focusing on basic numeric chart types.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT  # type: ignore
from matplotlib.figure import Figure

from charts import plot_chart, ChartOptions


class ChartBuilderDialog(QDialog):
    """A simple interactive builder that previews the chart live."""

    def __init__(self, df: pd.DataFrame, chart_types: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chart Builder")
        self.resize(900, 600)
        self._df = df
        self._chart_types = chart_types
        self._options = ChartOptions()
        self._preview_df: pd.DataFrame | None = None

        self._build_ui()
        self._refresh_column_list()
        self._update_preview()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Chart type:"))
        self.chart_cmb = QComboBox()
        self.chart_cmb.addItems(self._chart_types)
        self.chart_cmb.currentIndexChanged.connect(self._update_preview)
        ctrl_row.addWidget(self.chart_cmb)

        # Palette
        ctrl_row.addSpacing(10)
        ctrl_row.addWidget(QLabel("Palette:"))
        self.palette_cmb = QComboBox(); self.palette_cmb.addItems(["Default", "Tableau", "Pastel", "Dark"])
        self.palette_cmb.currentIndexChanged.connect(self._update_preview)
        ctrl_row.addWidget(self.palette_cmb)

        # Data labels
        self.labels_chk = QCheckBox("Labels")
        self.labels_chk.stateChanged.connect(self._update_preview)
        ctrl_row.addWidget(self.labels_chk)
        # Dual axis
        self.dual_chk = QCheckBox("Dual Y")
        self.dual_chk.stateChanged.connect(self._update_preview)
        ctrl_row.addWidget(self.dual_chk)

        # Aggregation combobox (hidden unless needed)
        ctrl_row.addSpacing(10)
        ctrl_row.addWidget(QLabel("Agg:"))
        self.agg_cmb = QComboBox()
        self.agg_cmb.addItems(["Mean", "Sum", "Count", "Median", "Min", "Max"])
        self.agg_cmb.currentIndexChanged.connect(self._update_preview)
        self.agg_cmb.hide()
        ctrl_row.addWidget(self.agg_cmb)

        # Category combo (hidden)
        ctrl_row.addSpacing(10)
        ctrl_row.addWidget(QLabel("Group by:"))
        self.cat_cmb = QComboBox()
        self.cat_cmb.currentIndexChanged.connect(self._update_preview)
        self.cat_cmb.hide()
        ctrl_row.addWidget(self.cat_cmb)

        ctrl_row.addSpacing(20)
        ctrl_row.addWidget(QLabel("Columns:"))
        self.col_list = QListWidget()
        self.col_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.col_list.itemSelectionChanged.connect(self._update_preview)
        ctrl_row.addWidget(self.col_list, 1)
        layout.addLayout(ctrl_row)

        # Preview area
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)
        # Navigation toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    def _refresh_column_list(self):
        """Populate column list with columns from *df*.

        We keep a mapping from displayed text -> original column object so that
        we can recover non-string columns such as tuples coming from a MultiIndex.
        """
        self.col_list.clear()
        self._col_map: dict[str, Any] = {}
        for col in self._df.columns:
            display = str(col)
            self._col_map[display] = col
            item = QListWidgetItem(display)
            item.setSelected(True)  # preselect all
            self.col_list.addItem(item)

    # ------------------------------------------------------------------
    def _selected_columns(self) -> List[Any]:
        """Return list of original column objects currently selected."""
        return [self._col_map[i.text()] for i in self.col_list.selectedItems()]

    def _update_preview(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        cols = self._selected_columns()
        if not cols:
            ax.text(0.5, 0.5, "Select columns", ha="center", va="center")
        else:
            chart_type = self.chart_cmb.currentText()
            df_sel = self._df[cols]
            # detect categorical + numeric mix
            num_cols = [col for col, dtype in df_sel.dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]
            cat_cols = [c for c in df_sel.columns if c not in num_cols]
            agg_needed = bool(cat_cols) and bool(num_cols)
            self.agg_cmb.setVisible(agg_needed)
            self.cat_cmb.setVisible(bool(cat_cols))
            if agg_needed:
                # populate category combo with string representation but keep mapping
                self.cat_cmb.blockSignals(True)
                self.cat_cmb.clear()
                self._cat_map: dict[str, Any] = {}
                for c in cat_cols:
                    disp = str(c)
                    self._cat_map[disp] = c
                    self.cat_cmb.addItem(disp)
                self.cat_cmb.blockSignals(False)
                cat_disp = self.cat_cmb.currentText() or self.cat_cmb.itemText(0)
                cat = self._cat_map.get(cat_disp, cat_cols[0])
                agg_name = self.agg_cmb.currentText()
                func_map = {
                    "Mean": "mean",
                    "Sum": "sum",
                    "Median": "median",
                    "Min": "min",
                    "Max": "max",
                }
                func = func_map.get(agg_name, "mean")
                grouped = df_sel.groupby(cat)[num_cols].agg(func)
                df_numeric = grouped
            else:
                self.cat_cmb.setVisible(False)
                # if no numeric data and chart is Pie etc.
                numeric_part = df_sel.select_dtypes(include="number")
                if numeric_part.empty and chart_type in ("Pie", "Donut", "Stacked Bar"):
                    series = df_sel.iloc[:, 0].astype(str).value_counts()
                    df_numeric = pd.DataFrame({series.name or 'count': series})
                else:
                    df_numeric = df_sel.apply(pd.to_numeric, errors="coerce")
            # update params for live update
            self._options = ChartOptions(
                palette=self.palette_cmb.currentText(),
                labels=self.labels_chk.isChecked(),
                dual=self.dual_chk.isChecked(),
            )
            if agg_needed:
                self._options.group_col = cat
                self._options.agg = agg_name
            try:
                plot_chart(ax, chart_type, df_numeric, self._options)
            except Exception as exc:
                ax.text(0.5, 0.5, str(exc), ha="center", va="center")
            # store for return
            self._preview_df = df_numeric.copy()
        ax.set_title(self.chart_cmb.currentText())
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def result(self) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
        """Return chart_type, numeric DataFrame slice, extra params."""
        cols = self._selected_columns()
        if self._preview_df is not None:
            return self.chart_cmb.currentText(), self._preview_df.copy(), self._params.copy()
        else:
            return self.chart_cmb.currentText(), pd.DataFrame(), self._params
