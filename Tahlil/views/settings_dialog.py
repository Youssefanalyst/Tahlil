"""Application settings dialog to edit preferences permanently."""
from __future__ import annotations

from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QSpinBox, QFormLayout
)
from PyQt6.QtCore import Qt

from qt_material import list_themes

from utils.settings_manager import settings


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(400, 250)
        self._build_ui()

    def _build_ui(self):
        main_lay = QVBoxLayout(self)
        form = QFormLayout()
        # Default path
        path_h = QHBoxLayout()
        self.path_edit = QLineEdit(settings.get("default_path"))
        browse_btn = QPushButton("Browse…")
        def _browse():
            directory = QFileDialog.getExistingDirectory(self, "Select default directory", self.path_edit.text())
            if directory:
                self.path_edit.setText(directory)
        browse_btn.clicked.connect(_browse)
        path_h.addWidget(self.path_edit)
        path_h.addWidget(browse_btn)
        form.addRow("Default path:", path_h)
        # Language
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["en", "ar"])
        self.lang_combo.setCurrentText(settings.get("language"))
        form.addRow("Language:", self.lang_combo)
        # Theme
        self.theme_combo = QComboBox()
        for t in list_themes():
            display = t.replace(".xml", "").replace("_", " ").title()
            self.theme_combo.addItem(display, t)
        # Attempt to match current theme
        current_theme_file = settings.get("theme")
        for i in range(self.theme_combo.count()):
            if self.theme_combo.itemText(i) == current_theme_file:
                self.theme_combo.setCurrentIndex(i)
                break
        form.addRow("Theme:", self.theme_combo)
        # Save precision
        self.prec_spin = QSpinBox()
        self.prec_spin.setRange(0, 10)
        self.prec_spin.setValue(int(settings.get("save_precision")))
        form.addRow("Save precision:", self.prec_spin)
        main_lay.addLayout(form)
        # buttons
        btn_lay = QHBoxLayout()
        ok = QPushButton("OK"); cancel = QPushButton("Cancel")
        ok.clicked.connect(self.accept); cancel.clicked.connect(self.reject)
        btn_lay.addStretch(1); btn_lay.addWidget(ok); btn_lay.addWidget(cancel)
        main_lay.addLayout(btn_lay)

    # -------------------
    def get_settings(self):
        return {
            "default_path": self.path_edit.text(),
            "language": self.lang_combo.currentText(),
            "theme": self.theme_combo.currentText(),
            "save_precision": self.prec_spin.value(),
        }
