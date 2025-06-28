"""Interactive dialog to help users construct formulas easily.

Allows selecting a function, entering its arguments in labeled fields, previewing the
resulting formula string, and inserting it into the spreadsheet.

The dialog relies on the same markdown reference file that powers the formula
reference window, ensuring the list of functions/descriptions/inputs is always
in sync with documentation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QFormLayout,
    QWidget,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt

# We reuse the private loader from the reference module instead of duplicating logic.
try:
    from views.formula_reference import _load_reference  # type: ignore
except Exception:
    # Fallback: empty mapping so dialog still opens without docs
    def _load_reference() -> Dict[str, Tuple[str, str, str]]:  # type: ignore
        return {}


class FormulaWizardDialog(QDialog):
    """Dialog that builds a formula string interactively."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Formula Wizard")
        self.resize(420, 380)
        self._mapping = _load_reference()
        if not self._mapping:
            # Should not really happen – docs missing – but degrade gracefully
            self._mapping = {"SUM": ("Sum of numbers", "value1, value2, …", "SUM(1,2)")}

        layout = QVBoxLayout(self)

        # Function picker
        self.fn_combo = QComboBox()
        self.fn_combo.addItems(sorted(self._mapping.keys()))
        layout.addWidget(self.fn_combo)

        # Description label
        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.desc_label)

        # Dynamic form for arguments
        self._args_container = QWidget()
        self._form = QFormLayout(self._args_container)
        layout.addWidget(self._args_container)

        # Preview edit (read-only)
        self.preview_edit = QLineEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setPlaceholderText("=FUNCTION(arg1, …)")
        layout.addWidget(self.preview_edit)

        # Result label
        self.result_label = QLabel("Result: —")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.result_label)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Connections
        self.fn_combo.currentTextChanged.connect(self._rebuild_form)

        # Build first form
        self._arg_edits: List[QLineEdit] = []
        self._rebuild_form()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _rebuild_form(self):
        """Regenerate argument fields when function changes."""
        fn = self.fn_combo.currentText()
        desc, input_type, _example = self._mapping.get(fn, ("", "", ""))
        self.desc_label.setText(desc or "(no description)")

        # Clear old inputs
        while self._form.rowCount():
            self._form.removeRow(0)
        self._arg_edits.clear()

        # Determine parameter names from input_type; fallback to generic arg1, arg2…
        param_names: List[str] = []
        if input_type:
            # Split by comma and strip; ignore ellipsis char
            raw_parts = [p.strip() for p in input_type.split(",") if p.strip()]
            for p in raw_parts:
                # remove trailing ellipsis or variable dots
                p = p.replace("…", "").strip()
                if p:
                    param_names.append(p)
        # Provide at least one argument slot even if function is variadic or takes none.
        if not param_names:
            param_names = ["value"]

        for name in param_names:
            edit = QLineEdit()
            edit.setPlaceholderText(name)
            edit.textChanged.connect(self._update_preview)
            self._form.addRow(f"{name}:", edit)
            self._arg_edits.append(edit)

        self._update_preview()

    def _update_preview(self):
        fn = self.fn_combo.currentText()
        args = [e.text() for e in self._arg_edits if e.text().strip()]
        if args:
            formula = f"={fn}(" + ", ".join(args) + ")"
        else:
            # zero-arg function or user left blanks
            formula = f"={fn}()" if self._arg_edits else f"={fn}()"
        # tidy – remove trailing '()' for built-ins like RAND()
        if formula.endswith("()") and len(args) == 0:
            formula = formula[:-2]
        self.preview_edit.setText(formula)

        # Try evaluating result for preview
        try:
            from models.dataframe_model import DataFrameModel
            import pandas as pd
            temp_model = DataFrameModel(pd.DataFrame([[formula]]))
            val = temp_model._internal_compute_value(0, 0, set())
            self.result_label.setText(f"Result: {val}")
        except Exception:
            self.result_label.setText("Result: —")

    # Public API --------------------------------------------------------
    def formula(self) -> str:
        """Return composed formula string (leading '=' included)."""
        return self.preview_edit.text()
