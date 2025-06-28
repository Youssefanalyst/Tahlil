"""Dialog to analyse formula dependencies and execution times for a given cell."""
from __future__ import annotations

import time
from typing import Set, Tuple, List

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt

from models.dataframe_model import DataFrameModel


class FormulaAuditDialog(QDialog):
    """Displays dependency tree of a formula along with evaluation time (ms)."""

    def __init__(self, model: DataFrameModel, row: int, col: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Formula Audit")
        self.resize(500, 400)
        self._model = model
        self._root_row = row
        self._root_col = col

        layout = QVBoxLayout(self)
        info_lbl = QLabel(
            f"Auditing cell {model._num_to_col_letter(col)}{row+1}. Times include formula evaluation overhead.")
        layout.addWidget(info_lbl)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Cell", "Value", "Time (ms)"])
        layout.addWidget(self.tree)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        hl = QHBoxLayout(); hl.addStretch(1); hl.addWidget(btn_close)
        layout.addLayout(hl)

        self._visited: Set[Tuple[int, int]] = set()
        self._populate_tree()
        self.tree.expandAll()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _populate_tree(self):
        root_item = QTreeWidgetItem()
        self.tree.addTopLevelItem(root_item)
        self._add_node(root_item, self._root_row, self._root_col)

    def _add_node(self, parent_item: QTreeWidgetItem, row: int, col: int):
        if (row, col) in self._visited:
            return  # avoid circular
        self._visited.add((row, col))

        coord = f"{self._model._num_to_col_letter(col)}{row+1}"
        start = time.perf_counter()
        value = self._model._internal_compute_value(row, col, set())
        elapsed_ms = (time.perf_counter() - start) * 1000

        node = QTreeWidgetItem([coord, str(value), f"{elapsed_ms:.2f}"])
        parent_item.addChild(node)

        # Only expand if original cell contains formula (starts with '=')
        raw = str(self._model._df.iat[row, col])
        if not raw.startswith('='):
            return
        formula = raw[1:]

        # Collect cell references
        refs: List[Tuple[int, int]] = []
        for cell_match in self._model._re_cell.finditer(formula):
            col_letters, row_num = cell_match.groups()
            r = int(row_num) - 1
            c = self._model._col_letter_to_num(col_letters)
            refs.append((r, c))
        # Collect ranges and expand
        for start_ref, end_ref in self._model._re_range.findall(formula):
            start_col_letters = ''.join(filter(str.isalpha, start_ref))
            start_row_num = int(''.join(filter(str.isdigit, start_ref)))
            end_col_letters = ''.join(filter(str.isalpha, end_ref))
            end_row_num = int(''.join(filter(str.isdigit, end_ref)))
            sr = start_row_num - 1; sc = self._model._col_letter_to_num(start_col_letters)
            er = end_row_num - 1; ec = self._model._col_letter_to_num(end_col_letters)
            for r in range(min(sr, er), max(sr, er) + 1):
                for c in range(min(sc, ec), max(sc, ec) + 1):
                    refs.append((r, c))

        # Add child nodes
        for r, c in refs:
            self._add_node(node, r, c)
