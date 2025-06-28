"""Dialog to set simple filters on a DataFrame for charting purposes.
Currently supports:
- Selecting a categorical column and choosing allowed values.
- Selecting a numeric column and specifying min/max range.
Multiple filters can be defined by reopening the dialog.
"""
from __future__ import annotations

from typing import Dict, Any

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QDoubleSpinBox,
    QPushButton,
    QDialogButtonBox,
)


class ChartFilterDialog(QDialog):
    def __init__(self, df: pd.DataFrame, existing_filters: Dict[str, Any] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chart Filters")
        self.resize(400, 300)
        self._df = df
        self._filters: Dict[str, Any] = existing_filters.copy() if existing_filters else {}
        self._build_ui()

    # ------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Column selector
        col_row = QHBoxLayout()
        col_row.addWidget(QLabel("Column:"))
        self.col_cmb = QComboBox()
        self.col_cmb.addItems(list(self._df.columns))
        self.col_cmb.currentIndexChanged.connect(self._populate_value_widget)
        col_row.addWidget(self.col_cmb, 1)
        layout.addLayout(col_row)

        # Area for value selection
        self.value_widget_container = QVBoxLayout()
        layout.addLayout(self.value_widget_container, 1)

        # buttons
        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.clicked.connect(self._apply_filter)
        btn_row.addWidget(self.apply_btn)
        self.remove_btn = QPushButton("Remove Filter")
        self.remove_btn.clicked.connect(self._remove_filter)
        btn_row.addWidget(self.remove_btn)
        layout.addLayout(btn_row)

        # OK/Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate_value_widget()
        # if there was an existing filter, preload its values
        self._load_existing()

    # ------------------------------------------------------------
    def _clear_value_widgets(self):
        while self.value_widget_container.count():
            item = self.value_widget_container.takeAt(0)
            if w := item.widget():
                w.setParent(None)

    def _load_existing(self):
        col = self.col_cmb.currentText()
        if not col or col not in self._filters:
            return
        spec = self._filters[col]
        if spec['type'] == 'numeric' and isinstance(self._current_value_widget, tuple):
            min_spin, max_spin = self._current_value_widget
            min_spin.setValue(spec['min'])
            max_spin.setValue(spec['max'])
        elif spec['type'] == 'categorical' and not isinstance(self._current_value_widget, tuple):
            for i in range(self._current_value_widget.count()):  # type: ignore
                item = self._current_value_widget.item(i)
                item.setSelected(item.text() in spec['values'])

    def _populate_value_widget(self):
        self._clear_value_widgets()
        col = self.col_cmb.currentText()
        if col == "":
            return
        series = self._df[col]
        if pd.api.types.is_numeric_dtype(series):
            # numeric range selectors
            row = QHBoxLayout()
            row.addWidget(QLabel("Min:"))
            min_spin = QDoubleSpinBox()
            min_spin.setRange(float(series.min()), float(series.max()))
            min_spin.setValue(float(series.min()))
            row.addWidget(min_spin)
            row.addWidget(QLabel("Max:"))
            max_spin = QDoubleSpinBox()
            max_spin.setRange(float(series.min()), float(series.max()))
            max_spin.setValue(float(series.max()))
            row.addWidget(max_spin)
            self.value_widget_container.addLayout(row)
            self._current_value_widget = (min_spin, max_spin)
        else:
            # categorical list
            list_widget = QListWidget()
            list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
            uniq_vals = list(series.dropna().astype(str).unique())
            for v in uniq_vals:
                item = QListWidgetItem(v)
                item.setSelected(True)
                list_widget.addItem(item)
            self.value_widget_container.addWidget(list_widget)
            self._current_value_widget = list_widget

    # ------------------------------------------------------------
    def _apply_filter(self):
        col = self.col_cmb.currentText()
        if not col:
            return
        series = self._df[col]
        if isinstance(self._current_value_widget, tuple):
            # numeric
            min_spin, max_spin = self._current_value_widget
            self._filters[col] = {
                'type': 'numeric',
                'min': min_spin.value(),
                'max': max_spin.value(),
            }
        else:
            # categorical
            selected = [i.text() for i in self._current_value_widget.selectedItems()]  # type: ignore
            if not selected:
                return
            self._filters[col] = {
                'type': 'categorical',
                'values': selected,
            }

    def _remove_filter(self):
        col = self.col_cmb.currentText()
        self._filters.pop(col, None)

    # ------------------------------------------------------------
    def _on_ok(self):
        # ensure current column filter stored
        self._apply_filter()
        self.accept()

    def filters(self) -> Dict[str, Any]:
        return self._filters.copy()
