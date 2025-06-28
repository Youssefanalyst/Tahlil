from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox,
    QPushButton, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, QRegularExpression
import re

class FindReplaceDialog(QDialog):
    """Simple find & replace dialog supporting regex, match-case, replace-all."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find and Replace")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        lay = QVBoxLayout(self)
        # Find
        f_h = QHBoxLayout()
        f_h.addWidget(QLabel("Find:"))
        self.find_edit = QLineEdit(self)
        f_h.addWidget(self.find_edit)
        lay.addLayout(f_h)
        # Replace
        r_h = QHBoxLayout()
        r_h.addWidget(QLabel("Replace:"))
        self.replace_edit = QLineEdit(self)
        r_h.addWidget(self.replace_edit)
        lay.addLayout(r_h)
        # Options
        opt_h = QHBoxLayout()
        self.regex_chk = QCheckBox("Regex", self)
        self.case_chk = QCheckBox("Match case", self)
        self.all_sheets_chk = QCheckBox("All sheets", self)
        opt_h.addWidget(self.regex_chk)
        opt_h.addWidget(self.case_chk)
        opt_h.addWidget(self.all_sheets_chk)
        opt_h.addStretch(1)
        lay.addLayout(opt_h)
        # Buttons
        btn_h = QHBoxLayout()
        self.replace_all_btn = QPushButton("Replace All", self)
        self.cancel_btn = QPushButton("Cancel", self)
        self.replace_all_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        btn_h.addStretch(1)
        btn_h.addWidget(self.replace_all_btn)
        btn_h.addWidget(self.cancel_btn)
        lay.addLayout(btn_h)

    # ------------------------------------------------------------------
    def get_options(self):
        return {
            "find": self.find_edit.text(),
            "replace": self.replace_edit.text(),
            "regex": self.regex_chk.isChecked(),
            "case": self.case_chk.isChecked(),
             "all": self.all_sheets_chk.isChecked(),
        }

    # Convenience static method
    @staticmethod
    def get_find_replace(parent=None):
        dlg = FindReplaceDialog(parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.get_options()
        return None
