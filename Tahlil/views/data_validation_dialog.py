"""Dialog to define data-validation rules on a cell range."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QMessageBox,
)
from PyQt6.QtCore import Qt


class DataValidationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Validation")
        layout = QVBoxLayout(self)

        # ---------------- type -----------------
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Validation type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Number", "List"])
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # Stacked pages for options
        self.pages = QStackedWidget()
        # Number page
        page_num = QVBoxLayout()
        num_widget = QLabel()
        pw = num_widget  # placeholder
        num_layout = QHBoxLayout()
        num_layout.addWidget(QLabel("Operator:"))
        self.op_combo = QComboBox()
        self.op_combo.addItems([">", "<", ">=", "<=", "=", "<>", "between"])
        num_layout.addWidget(self.op_combo)
        num_layout.addWidget(QLabel("Value:"))
        self.val_edit = QLineEdit()
        self.val_edit.setPlaceholderText("e.g. 100")
        num_layout.addWidget(self.val_edit)
        self.val2_edit = QLineEdit()
        self.val2_edit.setPlaceholderText("and value 2")
        self.val2_edit.setEnabled(False)
        num_layout.addWidget(self.val2_edit)
        # enable second value when between selected
        def _toggle_between(text):
            self.val2_edit.setEnabled(text == "between")
        self.op_combo.currentTextChanged.connect(_toggle_between)
        from PyQt6.QtWidgets import QWidget
        container_num = QWidget()
        container_num.setLayout(num_layout)
        self.pages.addWidget(container_num)

        # List page
        list_layout = QVBoxLayout()
        list_layout.addWidget(QLabel("Allowed values (comma separated):"))
        self.list_edit = QPlainTextEdit()
        self.list_edit.setPlaceholderText("e.g. Yes,No,Maybe")
        self.list_edit.setFixedHeight(60)
        list_layout.addWidget(self.list_edit)
        container_list = QWidget()
        container_list.setLayout(list_layout)
        self.pages.addWidget(container_list)

        layout.addWidget(self.pages)

        # Mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("On violation:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Warn", "Block"])
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # buttons
        btn_layout = QHBoxLayout()
        ok = QPushButton("OK"); cancel = QPushButton("Cancel")
        ok.clicked.connect(self._on_ok)
        cancel.clicked.connect(self.reject)
        btn_layout.addWidget(ok); btn_layout.addWidget(cancel)
        layout.addLayout(btn_layout)

        # connections
        self.type_combo.currentIndexChanged.connect(self.pages.setCurrentIndex)

    # -------------------- helpers -------------------
    def _on_ok(self):
        if self.type_combo.currentText() == "Number":
            if not self.val_edit.text().strip():
                QMessageBox.warning(self, "Validation", "Please enter a value for comparison.")
                return
        elif self.type_combo.currentText() == "List":
            if not self.list_edit.toPlainText().strip():
                QMessageBox.warning(self, "Validation", "Please enter at least one allowed value.")
                return
        self.accept()

    # ---------------- public -----------------
    def get_settings(self):
        vtype = self.type_combo.currentText().lower()
        mode = self.mode_combo.currentText().lower()
        if vtype == "number":
            op = self.op_combo.currentText()
            v1 = self.val_edit.text().strip()
            v2 = self.val2_edit.text().strip() if self.val2_edit.isEnabled() else None
            return {
                "type": vtype,
                "op": op,
                "value": v1,
                "value2": v2,
                "mode": mode,
            }
        else:
            allowed = [s.strip() for s in self.list_edit.toPlainText().split(',') if s.strip()]
            return {
                "type": vtype,
                "list": allowed,
                "mode": mode,
            }
