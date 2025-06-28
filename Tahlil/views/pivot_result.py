"""Dialog for displaying a pivot table result with copy / export / chart actions."""
from __future__ import annotations

from typing import Sequence, Optional

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QApplication,
    QFileDialog,
    QTableView,
)

from models.dataframe_model import DataFrameModel
from views.chart_builder import ChartBuilderDialog

_DEFAULT_CHART_TYPES: Sequence[str] = (
    "Bar",
    "Stacked Bar",
    "Line",
    "Area",
    "Pie",
    "Heatmap",
)


class PivotResultDialog(QDialog):
    """Show *pivot_df* in a table view with utility buttons."""

    def __init__(self, pivot_df: pd.DataFrame, parent=None, chart_types: Optional[Sequence[str]] = None):
        super().__init__(parent)
        self._df = pivot_df
        self.setWindowTitle("Pivot Table Result")
        self.resize(800, 600)

        chart_types = chart_types or _DEFAULT_CHART_TYPES

        layout = QVBoxLayout(self)

        # Table view
        self._table = QTableView()
        self._table.setModel(DataFrameModel(pivot_df.reset_index()))
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        copy_btn = QPushButton("Copy")
        export_btn = QPushButton("Export CSV…")
        chart_btn = QPushButton("Chart…")
        close_btn = QPushButton("Close")
        for b in (copy_btn, export_btn, chart_btn, close_btn):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        # Connect actions
        copy_btn.clicked.connect(lambda: self._copy())
        export_btn.clicked.connect(lambda: self._export())
        chart_btn.clicked.connect(lambda: self._open_chart(chart_types))
        close_btn.clicked.connect(self.close)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _copy(self):
        QApplication.clipboard().setText(self._df.to_csv(index=True))

    def _export(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if fname:
            self._df.to_csv(fname, index=True)

    def _open_chart(self, chart_types: Sequence[str]):
        dlg_chart = ChartBuilderDialog(self._df.reset_index(), chart_types, self)
        dlg_chart.exec()
