"""Import Wizard dialog.

Allows user to preview a dataset before importing it into the spreadsheet.
Currently supports CSV / Excel / JSON / SQLite via pandas.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QProgressDialog,
    QApplication,
    QSplitter,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

from io_utils import load_dataframe, preview_dataframe
import time
from concurrent.futures import ThreadPoolExecutor


class ImportWizard(QDialog):
    """Very simple interactive import dialog."""

    def __init__(self, file_path: str | Path, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Import Wizard")
        self.resize(800, 500)
        self._path = Path(file_path)
        self._df_full: pd.DataFrame | None = None
        # background loader future
        self._full_future = None

        self._selected_columns: list[str] = []
        self._build_ui()
        self._start_background_load()
        self._load_preview()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Path label + browse (allow changing)
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit(str(self._path))
        browse_btn = QPushButton("…")
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(QLabel("File:"))
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # Options row
        opts_layout = QHBoxLayout()
        self.header_chk = QCheckBox("First row is header")
        self.header_chk.setChecked(True)
        self.header_chk.stateChanged.connect(self._load_preview)
        opts_layout.addWidget(self.header_chk)

        opts_layout.addStretch()
        opts_layout.addWidget(QLabel("Rows (0 = all):"))
        self.row_spin = QSpinBox()
        self.row_spin.setRange(0, 1_000_000)
        self.row_spin.setValue(0)
        self.row_spin.valueChanged.connect(self._update_preview)
        opts_layout.addWidget(self.row_spin)

        self.random_chk = QCheckBox("Random")
        self.random_chk.setToolTip("Select random rows instead of top N")
        self.random_chk.stateChanged.connect(self._update_preview)
        opts_layout.addWidget(self.random_chk)
        layout.addLayout(opts_layout)

        # Column list + preview inside splitter
        self.col_list = QListWidget()
        self.col_list.itemChanged.connect(self._update_preview)

        col_container = QWidget()
        col_layout = QVBoxLayout(col_container)
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.addWidget(QLabel("Select columns:"))
        col_layout.addWidget(self.col_list)

        self.preview = QTableView()
        self.preview.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview.setAlternatingRowColors(True)
        self.preview.verticalHeader().setVisible(False)
        self.preview.horizontalHeader().setStretchLastSection(True)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(col_container)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter, 1)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _browse(self):
        filters = "CSV (*.csv);;Excel (*.xlsx *.xls);;JSON (*.json);;SQLite (*.db *.sqlite)"
        fname, _ = QFileDialog.getOpenFileName(self, "Select file", str(self._path.parent), filters)
        if fname:
            self._path = Path(fname)
            self.path_edit.setText(fname)
            # restart background load with new file
            self._start_background_load()
            self._load_preview()

    def _start_background_load(self):
        """Kick off background thread to load entire file so it's ready when user clicks OK."""
        # reuse single global executor to avoid creating many threads
        global _IMPORT_EXECUTOR
        try:
            _IMPORT_EXECUTOR
        except NameError:
            _IMPORT_EXECUTOR = ThreadPoolExecutor(max_workers=2)
        self._full_future = _IMPORT_EXECUTOR.submit(load_dataframe, self._path)

    def _load_preview(self):
        """Load file into DataFrame and refresh UI lists."""
        try:
            df = preview_dataframe(self._path, nrows=100, header=self.header_chk.isChecked())
            if not self.header_chk.isChecked():
                df.columns = [chr(65 + i) for i in range(len(df.columns))]
            self._df_full = df
            # populate columns list
            self.col_list.blockSignals(True)
            self.col_list.clear()
            for col in df.columns:
                item = QListWidgetItem(str(col))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self.col_list.addItem(item)
            self.col_list.blockSignals(False)
            self._selected_columns = list(df.columns)
            self._update_preview()
        except Exception as exc:
            self._df_full = None
            self.col_list.clear()
            self._populate_table(pd.DataFrame([{"Error": str(exc)}]))

    def _update_preview(self):
        if self._df_full is None:
            return
        # columns
        self._selected_columns = [self.col_list.item(i).text() for i in range(self.col_list.count()) if self.col_list.item(i).checkState()==Qt.CheckState.Checked]
        df = self._df_full[self._selected_columns] if self._selected_columns else self._df_full.iloc[:, :0]
        # rows limit
        n = self.row_spin.value()
        if n > 0:
            if self.random_chk.isChecked():
                try:
                    full_df = load_dataframe(self._path)
                    if not self.header_chk.isChecked():
                        full_df.columns = [chr(65 + i) for i in range(len(full_df.columns))]
                    source_df = full_df[self._selected_columns] if self._selected_columns else full_df
                    df = source_df.sample(n=min(n, len(source_df)), random_state=0)
                except Exception:
                    # fallback to current df
                    df = df.sample(n=min(n, len(df)), random_state=0)
            else:
                df = df.head(n)
        self._populate_table(df)

    def _populate_table(self, df: pd.DataFrame):
        model = QStandardItemModel(self)
        # headers
        model.setColumnCount(len(df.columns))
        model.setHorizontalHeaderLabels(list(map(str, df.columns)))
        # rows
        for row_idx, (_, row) in enumerate(df.iterrows()):
            items = [QStandardItem(str(v)) for v in row]
            model.appendRow(items)
        self.preview.setModel(model)
        self.preview.resizeColumnsToContents()

    # ------------------------------------------------------------------
    def _accept(self):
        # Show progress if background loading not finished and full import is required
        need_full = self.row_spin.value() == 0 or (self.random_chk.isChecked() and self.row_spin.value() > 0)
        progress: QProgressDialog | None = None
        if need_full:
            # if future not done we'll wait with a spinner
            if self._full_future and not self._full_future.done():
                progress = QProgressDialog("Importing data…", None, 0, 0, self)
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.setCancelButton(None)
                progress.show()
                QApplication.processEvents()
                # simple polling loop
                while not self._full_future.done():
                    QApplication.processEvents()
                    time.sleep(0.05)
        
        if self._df_full is None:
            self.reject()
            return
        header_on = self.header_chk.isChecked()
        n = self.row_spin.value()
        # Try to get preloaded full dataframe if available
        full_df: pd.DataFrame | None = None
        try:
            if self._full_future and self._full_future.done():
                full_df = self._full_future.result()
        except Exception:
            full_df = None
        try:
            if n > 0:
                if self.random_chk.isChecked():
                    if full_df is None:
                        full_df = load_dataframe(self._path)
                    df = full_df
                    if not header_on:
                        df.columns = [chr(65 + i) for i in range(len(df.columns))]
                    df = df.sample(n=min(n, len(df)), random_state=0)
                else:
                    df = preview_dataframe(self._path, nrows=n, header=header_on)
            else:
                if full_df is None:
                    df = load_dataframe(self._path)
                else:
                    df = full_df
            if not header_on:
                df.columns = [chr(65 + i) for i in range(len(df.columns))]
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to load file: {exc}")
            self.reject()
            return
        # apply column selections
        if progress is not None:
            progress.close()
        df = df[self._selected_columns] if self._selected_columns else df.iloc[:, :0]
        self._df_full = df
        self.accept()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def get_dataframe(self) -> pd.DataFrame | None:
        return self._df_full
