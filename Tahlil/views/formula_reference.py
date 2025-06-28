"""Dialog showing list of available spreadsheet formulas with short English description.
Parses `docs/formulas_reference.md` to build the mapping automatically so docs & UI stay in sync.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
)

_DOC_PATH = Path(__file__).resolve().parent.parent / "docs" / "formulas_reference.md"


def _load_reference() -> Dict[str, Tuple[str, str, str]]:
    """Parse markdown file and return mapping {FUNCTION: (description, example)}."""
    mapping: Dict[str, Tuple[str, str, str]] = {}
    if not _DOC_PATH.exists():
        return mapping
    for line in _DOC_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("| `"):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 2:
            continue
        funcs_part = parts[0]  # like `SUM(range)` or `ROLLINGMAX` / ...
        desc = parts[1]
        input_type = ""
        example = ""
        if len(parts) == 3:
            # assume only example provided
            example = parts[2]
        elif len(parts) >= 4:
            input_type = parts[2]
            example = parts[3]
        # clean description (remove trailing backticks etc)
        desc = desc.strip().rstrip("|")
        input_type = input_type.strip().rstrip("|")
        example = example.strip().rstrip("|")
        # split funcs by '/' and slash
        funcs_raw = funcs_part.strip("`")
        for fn in funcs_raw.split("/"):
            fn = fn.strip()
            # remove arguments inside parentheses
            if "(" in fn:
                fn = fn.split("(")[0]
            if fn:
                mapping[fn] = (desc, input_type, example)
    # Ensure every supported function has an entry
    try:
        from models.dataframe_model import DataFrameModel
        for fn in DataFrameModel.SUPPORTED_FUNCTIONS:
            if fn not in mapping:
                mapping[fn] = ("(no description yet)", "", "")
    except Exception:
        pass
    return mapping


class FormulaReferenceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Formula Reference")
        self.resize(600, 500)
        layout = QVBoxLayout(self)

        self._mapping = _load_reference()
        if not self._mapping:
            layout.addWidget(QLabel("Reference document not found."))
            return

        # Search box
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search…")
        layout.addWidget(self.search_edit)

        # Table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Function", "Description", "Input", "Example"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        self._populate_table()
        self.search_edit.textChanged.connect(self._filter)

    # -----------------------------------------------------
    def _populate_table(self):
        self.table.setRowCount(len(self._mapping))
        for row, (fn, (desc, input_type, example)) in enumerate(sorted(self._mapping.items())):
            self.table.setItem(row, 0, QTableWidgetItem(fn))
            item_desc = QTableWidgetItem(desc)
            item_desc.setFlags(item_desc.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 1, item_desc)
            item_in = QTableWidgetItem(input_type)
            item_in.setFlags(item_in.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            if not input_type:
                item_in.setText("–")
            self.table.setItem(row, 2, item_in)
            item_ex = QTableWidgetItem(example)
            item_ex.setFlags(item_ex.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            if not example:
                item_ex.setText("–")
            self.table.setItem(row, 3, item_ex)
        self.table.resizeColumnsToContents()

    def _filter(self, text: str):
        text = text.lower()
        for row in range(self.table.rowCount()):
            fn = self.table.item(row, 0).text().lower()
            desc = self.table.item(row, 1).text().lower()
            inp = self.table.item(row, 2).text().lower()
            example = self.table.item(row, 3).text().lower()
            visible = text in fn or text in desc or text in inp or text in example
            self.table.setRowHidden(row, not visible)
