"""Dialog to configure pivot table parameters before creation.

Extracted from main.py for better modularity and reuse.
"""
from __future__ import annotations

from typing import List
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLabel,
    QComboBox,
)


class PivotTableDialog(QDialog):
    """Simple dialog to choose rows/columns/values/aggregations/filters."""

    def __init__(self, columns: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Pivot Table")
        layout = QVBoxLayout(self)

        lists_layout = QHBoxLayout()
        # Row fields
        row_group = QGroupBox("Row Fields (choose 1+)")
        row_v = QVBoxLayout(row_group)
        self.rows_list = QListWidget()
        self.rows_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.rows_list.setAlternatingRowColors(True)
        for col in columns:
            QListWidgetItem(str(col), self.rows_list)
        row_v.addWidget(self.rows_list)
        lists_layout.addWidget(row_group)

        # Column fields
        col_group = QGroupBox("Column Fields (optional)")
        col_v = QVBoxLayout(col_group)
        self.cols_list = QListWidget()
        self.cols_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.cols_list.setAlternatingRowColors(True)
        for col in columns:
            QListWidgetItem(str(col), self.cols_list)
        col_v.addWidget(self.cols_list)
        lists_layout.addWidget(col_group)
        layout.addLayout(lists_layout)

        # Filters
        filter_group = QGroupBox("Filters (optional, choose 1+)")
        filter_v = QVBoxLayout(filter_group)
        self.filter_list = QListWidget()
        self.filter_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.filter_list.setAlternatingRowColors(True)
        for col in columns:
            QListWidgetItem(str(col), self.filter_list)
        filter_v.addWidget(self.filter_list)
        layout.addWidget(filter_group)

        # Values + aggregations
        val_group = QGroupBox("Values")
        val_layout = QVBoxLayout(val_group)
        val_layout.addWidget(QLabel("Value Fields (choose 1+):"))
        self.values_list = QListWidget()
        self.values_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.values_list.setAlternatingRowColors(True)
        for col in columns:
            QListWidgetItem(str(col), self.values_list)
        if self.values_list.count():
            self.values_list.item(0).setSelected(True)
        val_layout.addWidget(self.values_list)
        val_layout.addWidget(QLabel("Aggregation Functions (choose 1+):"))
        self.agg_list = QListWidget()
        self.agg_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.agg_list.setAlternatingRowColors(True)
        for fn in [
            "sum",
            "mean",
            "count",
            "median",
            "std",
            "var",
            "min",
            "max",
            "nunique",
        ]:
            QListWidgetItem(fn, self.agg_list)
        val_layout.addWidget(self.agg_list)
        layout.addWidget(val_group)

        # Dialog buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    def get_settings(self):
        rows = [i.text() for i in self.rows_list.selectedItems()]
        cols = [i.text() for i in self.cols_list.selectedItems()]
        values = [i.text() for i in self.values_list.selectedItems()]
        filters = [i.text() for i in self.filter_list.selectedItems()]
        aggs = [i.text() for i in self.agg_list.selectedItems()]
        if not aggs:
            aggs = ["sum"]
        return rows, cols, values, aggs, filters
