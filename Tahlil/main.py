"""Entry point for Spreadsheet PyQt6 application."""
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt, QModelIndex, pyqtSignal, QRegularExpression, QTimer, QStringListModel, QSize
from PyQt6.QtWidgets import (
    QApplication,
    QPlainTextEdit,
    QCompleter,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QTableView,
    QToolBar,
    QHeaderView,
    QLineEdit,
    QVBoxLayout,
    QWidget,
    QInputDialog,
    QSplitter,
    QScrollArea,
    QGridLayout,
    QMenu,
    QColorDialog,
    QDialog,
    QComboBox,
    QFormLayout,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QComboBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QDockWidget,
    QSpinBox,
    QProgressDialog,
    QMenu,
)
from PyQt6.QtGui import QColor, QPainter
from qt_material import apply_stylesheet, list_themes
from utils.settings_manager import settings
from views.find_replace_dialog import FindReplaceDialog
from PyQt6.QtGui import QKeySequence, QAction, QActionGroup, QTextCharFormat, QSyntaxHighlighter, QIcon, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import QRegularExpression, QTimer
from PyQt6.QtSql import QSqlDatabase  # placeholder for future SQL features

# matplotlib for inline charts
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np


class GroupByDialog(QDialog):
    """Dialog to choose grouping and aggregation for categorical charts."""
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Data")
        form = QFormLayout(self)
        from pandas.api.types import CategoricalDtype as _CatDType
        cat_cols = [c for c in df.columns if df[c].dtype == object or isinstance(df[c].dtype, _CatDType)]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.group_cb = QComboBox(); self.group_cb.addItems(cat_cols)
        self.value_cb = QComboBox(); self.value_cb.addItem("--Count--"); self.value_cb.addItems(num_cols)
        self.agg_cb = QComboBox(); self.agg_cb.addItems(["Count", "Sum", "Mean"])
        form.addRow("Group by:", self.group_cb)
        form.addRow("Value column:", self.value_cb)
        form.addRow("Aggregation:", self.agg_cb)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addWidget(buttons)

    def choices(self):
        return self.group_cb.currentText(), self.value_cb.currentText(), self.agg_cb.currentText()


class FormulaHighlighter(QSyntaxHighlighter):
    """Simple syntax highlighter for formula text."""
    def __init__(self, document, functions: list[str]):
        super().__init__(document)
        self.func_pattern = QRegularExpression(r"\b(" + "|".join(functions) + r")\b")
        self.num_pattern = QRegularExpression(r"\b\d+(?:\.\d+)?\b")
        self.op_pattern = QRegularExpression(r"[\+\-\*/\^<>=(),]")
        # formats
        self.func_format = QTextCharFormat(); self.func_format.setForeground(QColor("#006699"))
        self.num_format = QTextCharFormat(); self.num_format.setForeground(QColor("#b58900"))
        self.op_format = QTextCharFormat(); self.op_format.setForeground(QColor("#268bd2"))

    def _apply_format(self, pattern, text, fmt):
        it = pattern.globalMatch(text)
        while it.hasNext():
            match = it.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), fmt)

    def highlightBlock(self, text):
        self._apply_format(self.func_pattern, text, self.func_format)
        self._apply_format(self.num_pattern, text, self.num_format)
        self._apply_format(self.op_pattern, text, self.op_format)

class FormulaEdit(QPlainTextEdit):
    """Single-line text edit with completer and returnPressed signal."""
    returnPressed = pyqtSignal()
    def __init__(self, functions: list[str], parent=None):
        super().__init__(parent)
        self.setMaximumHeight(24)
        # completer
        model = QStringListModel(functions, self)
        completer = QCompleter(model, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer = completer
        completer.setWidget(self)
        completer.activated.connect(self.insertCompletion)
        # syntax highlighter
        self.highlighter = FormulaHighlighter(self.document(), functions)
        # disable word wrap
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

    def insertCompletion(self, text):
        cursor = self.textCursor()
        cursor.select(cursor.SelectionType.WordUnderCursor)
        cursor.insertText(text)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.returnPressed.emit()
            return
        super().keyPressEvent(e)
        # show completer after typing alpha chars
        tc = self.textCursor()
        tc.select(tc.SelectionType.WordUnderCursor)
        prefix = tc.selectedText()
        if len(prefix) >= 1:
            self.completer.setCompletionPrefix(prefix)
            self.completer.complete()
        else:
            self.completer.popup().hide()

    # ---- QLineEdit compatibility helpers ----
    def text(self):
        return self.toPlainText()

    def setText(self, txt: str):
        self.setPlainText(txt)

    def cursorPosition(self):
        return self.textCursor().position()

    def setCursorPosition(self, pos: int):
        tc = self.textCursor()
        tc.setPosition(pos)
        self.setTextCursor(tc)

class SpreadsheetModel(pd.DataFrame):
    """Simple DataFrame-backed model for the MVP."""

    # For now we just subclass DataFrame; later we can implement Qt models.
    pass


class ChartPropertiesWidget(QWidget):
    def __init__(self, ax, canvas, parent=None):
        super().__init__(parent)
        self._ax = ax
        self._canvas = canvas
        lay = QVBoxLayout(self)
        # title
        self.title_edit = QLineEdit(ax.get_title())
        self.title_edit.setPlaceholderText("Chart Title")
        lay.addWidget(QLabel("Title:")); lay.addWidget(self.title_edit)
        # x label
        self.xlabel_edit = QLineEdit(ax.get_xlabel())
        lay.addWidget(QLabel("X Label:")); lay.addWidget(self.xlabel_edit)
        # y label
        self.ylabel_edit = QLineEdit(ax.get_ylabel())
        lay.addWidget(QLabel("Y Label:")); lay.addWidget(self.ylabel_edit)
        # line width
        self.linewidth_spin = QSpinBox(); self.linewidth_spin.setRange(1,10); self.linewidth_spin.setValue(2)
        lay.addWidget(QLabel("Line Width:")); lay.addWidget(self.linewidth_spin)
        # line style
        self.linestyle_combo = QComboBox(); self.linestyle_combo.addItems(["-","--","-.",":"])
        lay.addWidget(QLabel("Line Style:")); lay.addWidget(self.linestyle_combo)
        # color button
        self.color_btn = QPushButton("Choose Color")
        lay.addWidget(self.color_btn)
        lay.addStretch()
        # connect
        self.title_edit.textChanged.connect(self._apply)
        self.xlabel_edit.textChanged.connect(self._apply)
        self.ylabel_edit.textChanged.connect(self._apply)
        self.linewidth_spin.valueChanged.connect(self._apply)
        self.linestyle_combo.currentTextChanged.connect(self._apply)
        self.color_btn.clicked.connect(self._pick_color)
        self._color = None

    def _pick_color(self):
        col = QColorDialog.getColor(parent=self)
        if col.isValid():
            self._color = col.name()
            self._apply()

    def _apply(self):
        self._ax.set_title(self.title_edit.text())
        self._ax.set_xlabel(self.xlabel_edit.text())
        self._ax.set_ylabel(self.ylabel_edit.text())
        for line in self._ax.get_lines():
            line.set_linewidth(self.linewidth_spin.value())
            line.set_linestyle(self.linestyle_combo.currentText())
            if self._color:
                line.set_color(self._color)
        self._canvas.draw_idle()

"""
Legacy duplicate PivotTableDialog class retained for reference after moving to views.pivot_builder.

class PivotTableDialog(QDialog):
    def __init__(self, columns: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Pivot Table")
        layout=QVBoxLayout(self)

        lists_layout = QHBoxLayout()
        # Row fields group
        row_group = QGroupBox("Row Fields (choose 1+)")
        row_v = QVBoxLayout(row_group)
        self.rows_list = QListWidget()
        self.rows_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.rows_list.setAlternatingRowColors(True)
        for col in columns:
            QListWidgetItem(str(col), self.rows_list)
        row_v.addWidget(self.rows_list)
        lists_layout.addWidget(row_group)

        # Column fields group
        col_group = QGroupBox("Column Fields (optional)")
        col_v = QVBoxLayout(col_group)
        self.cols_list = QListWidget(); self.cols_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.cols_list.setAlternatingRowColors(True)
        for col in columns:
            QListWidgetItem(str(col), self.cols_list)
        col_v.addWidget(self.cols_list)
        lists_layout.addWidget(col_group)
        layout.addLayout(lists_layout)

        # Filter fields group
        filter_group = QGroupBox("Filters (optional, choose 1+)")
        filter_v = QVBoxLayout(filter_group)
        self.filter_list = QListWidget(); self.filter_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.filter_list.setAlternatingRowColors(True)
        for col in columns:
            QListWidgetItem(str(col), self.filter_list)
        filter_v.addWidget(self.filter_list)
        layout.addWidget(filter_group)

        # Value + aggs
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
        self.agg_list = QListWidget(); self.agg_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.agg_list.setAlternatingRowColors(True)
        for fn in ["sum","mean","count","median","std","var","min","max","nunique"]:
            QListWidgetItem(fn, self.agg_list)
        val_layout.addWidget(self.agg_list)
        layout.addWidget(val_group)

        # buttons
        btn_layout=QHBoxLayout(); ok=QPushButton("OK"); cancel=QPushButton("Cancel")
        ok.clicked.connect(self.accept); cancel.clicked.connect(self.reject)
        btn_layout.addWidget(ok); btn_layout.addWidget(cancel)
        layout.addLayout(btn_layout)
    def get_settings(self):
        rows=[item.text() for item in self.rows_list.selectedItems()]
        cols=[item.text() for item in self.cols_list.selectedItems()]
        values=[item.text() for item in self.values_list.selectedItems()]
        filters=[item.text() for item in self.filter_list.selectedItems()]
        aggs=[item.text() for item in self.agg_list.selectedItems()]
        if not aggs:
            aggs=["sum"]
        return rows, cols, values, aggs, filters
"""

class ConditionalFormattingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Conditional Formatting")
        layout=QVBoxLayout(self)
        # operator + value
        op_layout=QHBoxLayout()
        op_layout.addWidget(QLabel("Condition:"))
        self.op_combo=QComboBox()
        self.op_combo.addItems([">", "<", ">=", "<=", "=", "<>"])
        op_layout.addWidget(self.op_combo)
        self.value_edit=QLineEdit()
        self.value_edit.setPlaceholderText("Value")
        op_layout.addWidget(self.value_edit)
        layout.addLayout(op_layout)
        # color picker
        color_layout=QHBoxLayout()
        color_layout.addWidget(QLabel("Background:"))
        self.color_btn=QPushButton()
        self._color=QColor("#ffff99")
        self._update_color_btn()
        self.color_btn.clicked.connect(self._pick_color)
        color_layout.addWidget(self.color_btn)
        layout.addLayout(color_layout)
        # buttons
        btn_layout=QHBoxLayout()
        ok_btn=QPushButton("OK"); cancel_btn=QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept); cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn); btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _pick_color(self):
        col=QColorDialog.getColor(self._color, self, "Select Color")
        if col.isValid():
            self._color=col; self._update_color_btn()
    def _update_color_btn(self):
        self.color_btn.setStyleSheet(f"background:{self._color.name()};")
    def get_settings(self):
        return self.op_combo.currentText(), self.value_edit.text(), self._color


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAHLIL")
        # initial size but will be maximized on show
        self.setAcceptDrops(True)  # enable drag-and-drop
        self.resize(1200, 800)

        # Load external extensions (plugins)
        try:
            from extensions import load_extensions  # local package
            self._loaded_extensions = load_extensions(self)
        except Exception as exc:
            print(f"[Main] Failed to load extensions: {exc}")
            self._loaded_extensions = []

        # Central widgets: formula bar + table view + chart
        from views.spreadsheet_view import SpreadsheetView
        self.table = SpreadsheetView()
        # Ensure per-item background brushes are not overridden by stylesheet
        # Some qt-material themes set QTableView::item{ background-color: ... }
        # which hides model-provided BackgroundRole brushes. Reset to 'none'.
        self.table.setStyleSheet("QTableView::item { background: none; }")  # will become first sheet
        self.table.setAlternatingRowColors(True)
        header=self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setDefaultSectionSize(100)  # prevent columns from shrinking too small
        self.table.verticalHeader().setDefaultSectionSize(24)

        # Formula bar with autocomplete & highlighting
        from models.dataframe_model import DataFrameModel
        self.formula_bar = FormulaEdit(DataFrameModel.SUPPORTED_FUNCTIONS, self)
        self.formula_bar.setPlaceholderText("Formula")
        self.formula_bar.returnPressed.connect(self._apply_formula)

        # chart tabs (multiple canvases)  + dashboard area
        from PyQt6.QtWidgets import QTabWidget
        from PyQt6.QtWidgets import QTabWidget
        # Sheet tabs ---------------------------------------------------------
        self.sheet_tabs = QTabWidget()
        self.sheet_tabs.setMovable(True)
        self.sheet_tabs.setTabsClosable(True)
        self.sheet_tabs.setDocumentMode(True)
        self.sheet_tabs.tabCloseRequested.connect(self._delete_sheet)
        self.sheet_tabs.currentChanged.connect(self._on_sheet_changed)
        self.sheet_tabs.tabBarDoubleClicked.connect(self._rename_sheet)
        self.sheet_tabs.addTab(self.table, "Sheet1")

        # Chart tabs ---------------------------------------------------------
        self.chart_tabs = QTabWidget()
        self.chart_tabs.setTabsClosable(True)
        self.chart_tabs.setMovable(True)
        self.chart_tabs.setDocumentMode(True)
        self.chart_tabs.setMinimumWidth(300)
        self.chart_tabs.tabCloseRequested.connect(self._close_chart_tab)
        # live charts list
        self._charts = []  # list of dicts with canvas, ax, rows, cols, type, params, live
        # dashboard
        self._dashboard_mode = False
        self.dashboard_area = QScrollArea()
        self.dashboard_area.setWidgetResizable(True)
        self.dashboard_container = QWidget()
        self.dashboard_grid = QGridLayout(self.dashboard_container)
        self.dashboard_grid.setContentsMargins(4,4,4,4)
        self.dashboard_grid.setSpacing(4)
        self.dashboard_area.setWidget(self.dashboard_container)
        self.dashboard_area.hide()

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(2)
        vbox.addWidget(self.formula_bar)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter = self.splitter
        splitter.addWidget(self.sheet_tabs)
        splitter.addWidget(self.chart_tabs)

        # Initially hide chart panel until first chart is added
        self.chart_tabs.hide()
        splitter.setSizes([1, 0])
        splitter.setSizes([700, 300])
        vbox.addWidget(self.splitter)
        self.setCentralWidget(container)

        # header rename
        self.table.horizontalHeader().sectionDoubleClicked.connect(self._rename_column)

        # status bar with advanced info
        self.status_pos = self.status_bar = self.statusBar()
        self.sel_label = QLabel("Sel: 0×0")
        self.value_label = QLabel("Value: ")
        self.formula_label = QLabel("")
        self.info_label = QLabel("Rows: 0 Cols: 0 Mem: 0 B")
        for lbl in (self.sel_label, self.value_label, self.formula_label, self.info_label):
            self.status_bar.addPermanentWidget(lbl)

        # theme setup before building menus
                # Dynamically build list of available qt-material themes
        def _fmt(xml_name: str) -> str:
            return xml_name.replace('.xml', '').replace('_', ' ').title()
        self._theme_files = {_fmt(t): t for t in list_themes()}
        # pick default theme
        # load theme from persisted settings if available
        self._current_theme = settings.get("theme", "Light Blue" if "Light Blue" in self._theme_files else next(iter(self._theme_files)))
        self._dark_mode = False


        self._create_actions()
        self._create_menu()
        self._create_toolbar()
        # defer connecting after model is set
        QTimer.singleShot(0, self._setup_status_connections)
        self._apply_theme()


        self.current_path: Optional[Path] = None
        self._new_file()

    # ---------------------------------------------------------------------
    # UI helpers
    # ---------------------------------------------------------------------
    def _create_actions(self):
        self.new_act = QAction(QIcon.fromTheme("document-new"), "&New", self)
        self.new_act.setShortcut(QKeySequence.StandardKey.New)
        self.new_act.triggered.connect(self._new_file)

        self.open_act = QAction(QIcon.fromTheme("document-open"), "&Open…", self)
        self.open_act.setShortcut(QKeySequence.StandardKey.Open)
        self.open_act.triggered.connect(self._open_file)

        self.save_act = QAction(QIcon.fromTheme("document-save"), "&Save", self)

        self.chart_act = QAction(QIcon.fromTheme("view-statistics"), "Chart &Builder…", self)

        self.add_cols_act = QAction(QIcon.fromTheme("list-add"), "Add &Columns…", self)
        self.add_rows_act = QAction(QIcon.fromTheme("list-add"), "Add &Rows…", self)
        self.add_cols_act.setShortcut("Ctrl+Shift+C")
        self.add_cols_act.triggered.connect(self._prompt_add_columns)
        self.add_rows_act.setShortcut("Ctrl+Shift+R")
        self.add_rows_act.triggered.connect(self._prompt_add_rows)
        self.chart_act.triggered.connect(self._open_chart_builder)

        # Outlier detection actions
        self.outlier_z_act = QAction("Detect Outliers (Z-Score)", self)
        self.outlier_iqr_act = QAction("Detect Outliers (IQR)", self)
        self.outlier_knn_act = QAction("Detect Outliers (K-NN)", self)
        self.outlier_z_act.triggered.connect(lambda: self._detect_outliers("z"))
        self.outlier_iqr_act.triggered.connect(lambda: self._detect_outliers("iqr"))
        self.outlier_knn_act.triggered.connect(lambda: self._detect_outliers("knn"))
        # chart types list
        # New Sheet action
        self.new_sheet_act = QAction(QIcon.fromTheme("tab-new"), "New &Sheet", self)
        self.new_sheet_act.setShortcut("Ctrl+Shift+N")
        self.new_sheet_act.triggered.connect(self._add_sheet)

        self.rename_sheet_act = QAction("Rename Sheet…", self)
        self.rename_sheet_act.triggered.connect(lambda: self._rename_sheet(self.sheet_tabs.currentIndex()))

        self.delete_sheet_act = QAction("Delete Sheet", self)
        self.delete_sheet_act.triggered.connect(lambda: self._delete_sheet(self.sheet_tabs.currentIndex()))

        self._chart_types = [
            "Line", "Area", "Stacked Bar", "Pie", "Donut", "Bubble", "Waterfall", "Heatmap", "Boxplot", "Radar"
        ]
        self.save_act.setShortcut(QKeySequence.StandardKey.Save)
        self.save_act.triggered.connect(self._save_file)

        self.undo_act = QAction(QIcon.fromTheme("edit-undo"), "&Undo", self)
        self.undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_act.triggered.connect(lambda: self.table.model().undo())

        self.redo_act = QAction(QIcon.fromTheme("edit-redo"), "&Redo", self)
        self.redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_act.triggered.connect(lambda: self.table.model().redo())

        # Find & Replace action
        self.find_replace_act = QAction(QIcon.fromTheme("edit-find-replace"), "Find && Replace…", self)
        self.find_replace_act.setShortcut("Ctrl+H")
        self.find_replace_act.triggered.connect(self._show_find_replace)

        # Formula wizard action
        self.formula_wizard_act = QAction("Insert &Function Wizard…", self)
        self.formula_wizard_act.setShortcut("Ctrl+Shift+F")
        self.formula_wizard_act.triggered.connect(self._open_formula_wizard)

        # Formula reference action
        self.formula_ref_act = QAction("Formula &Reference…", self)
        self.formula_ref_act.setShortcut("F1")
        self.formula_ref_act.triggered.connect(self._open_formula_reference)

        # Formula audit action
        self.formula_audit_act = QAction("Formula &Audit…", self)
        self.formula_audit_act.setShortcut("Ctrl+F1")
        self.formula_audit_act.triggered.connect(self._open_formula_audit)

        self.exit_act = QAction(QIcon.fromTheme("application-exit"), "E&xit", self)
        self.exit_act.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_act.triggered.connect(self.close)

        # Tooltips / status tips
        self.new_act.setStatusTip("New file")
        self.open_act.setStatusTip("Open file")
        self.save_act.setStatusTip("Save file")
        self.undo_act.setStatusTip("Undo")
        self.redo_act.setStatusTip("Redo")
        self.find_replace_act.setStatusTip("Find & Replace")
        self.chart_act.setStatusTip("Chart Builder")
        self.add_cols_act.setStatusTip("Add Columns")
        self.add_rows_act.setStatusTip("Add Rows")
        self.exit_act.setStatusTip("Exit")

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        formulas_menu = menubar.addMenu("&Formulas")
        # Add wizard and reference at top
        formulas_menu.addAction(self.formula_wizard_act)
        formulas_menu.addAction(self.formula_audit_act)
        # categorise functions
        categories = {
            "Math": [
                "SUM","AVERAGE","MIN","MAX","COUNT","PRODUCT","ABS","SQRT","POW","MOD","ROUND","CEILING","FLOOR","LOG","LOG10","LN","QUOTIENT","TRUNC","RANK","INT","FACT","ROUNDUP","ROUNDDOWN"],
            "Logical": ["IF","IFS","IFERROR","AND","OR","NOT","XOR","TRUE","FALSE"],
            "Text": ["LEN","CONCAT","TEXTJOIN","LEFT","RIGHT","UPPER","LOWER","TRIM","SUBSTITUTE","FIND","REPLACE","VALUE","REPT","PROPER","SPLIT","MID"],
            "Date/Time": ["DATE","TODAY","NOW","YEAR","MONTH","DAY","DAYS","NETWORKDAYS"],
            "Statistical": ["MEDIAN","MODE","VAR","STDEV","RAND","RANDBETWEEN","SMALL","LARGE","PERCENTILE","AVERAGEIF","AVERAGEIFS","COUNTIF","COUNTIFS","SUMIF","SUMIFS"],
            "Lookup": ["CHOOSE","COLUMN","UNIQUE","COUNTUNIQUE"],
            "Trigonometry": ["SIN","COS","TAN","ASIN","ACOS","ATAN","ATAN2","SINH","COSH","TANH","SEC","COT","CSC","DEGREES","RADIANS"],
            "Other": []
        }
        # collect into lookup
        category_lookup = {}
        for cat, funcs in categories.items():
            for f in funcs:
                category_lookup[f] = cat
        from models.dataframe_model import DataFrameModel
        SUPPORTED_FUNCTIONS = DataFrameModel.SUPPORTED_FUNCTIONS
        submenus: dict[str,QMenu] = {}
        # Add formula reference action below wizard
        formulas_menu.addAction(self.formula_ref_act)
        formulas_menu.addSeparator()
        for fname in SUPPORTED_FUNCTIONS:
            cat = category_lookup.get(fname, "Other")
            if cat not in submenus:
                submenus[cat] = formulas_menu.addMenu(cat)
            act = QAction(fname, self)
            act.triggered.connect(lambda checked=False, fn=fname: self._insert_function(fn))
            submenus[cat].addAction(act)
        # ensure others recorded
        
        
        self.view_menu = menubar.addMenu("&View")
        data_menu = menubar.addMenu("&Data")
        analysis_menu = menubar.addMenu("&Analysis")
        # Add outlier detection actions
        data_menu.addAction(self.outlier_z_act)
        data_menu.addAction(self.outlier_iqr_act)
        data_menu.addAction(self.outlier_knn_act)
                # Analysis tools
        self.analysis_tools_act = QAction("Statistical &Analysis…", self)
        self.analysis_tools_act.triggered.connect(self._open_analysis_tools)
        analysis_menu.addAction(self.analysis_tools_act)

        self.format_menu = menubar.addMenu("&Format")
        # Preferences / settings action
        self.settings_act = QAction("Settings…", self)
        self.settings_act.triggered.connect(self._open_settings_dialog)
        menubar.addAction(self.settings_act)

        # Themes submenu with instant switching
        # Sheet submenu
        sheet_menu = self.view_menu.addMenu("Sheets")
        sheet_menu.addAction(self.new_sheet_act)
        sheet_menu.addAction(self.rename_sheet_act)
        sheet_menu.addAction(self.delete_sheet_act)

        theme_menu = self.view_menu.addMenu("Themes")
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        for name in self._theme_files.keys():
            act = QAction(name, self, checkable=True)
            if name == self._current_theme:
                act.setChecked(True)
            act.triggered.connect(lambda checked=False, n=name: self._set_theme(n))
            self._theme_group.addAction(act)
            theme_menu.addAction(act)

        self.view_menu.addAction(self.add_cols_act)
        self.view_menu.addAction(self.add_rows_act)
        file_menu.addAction(self.new_act)
        file_menu.addAction(self.open_act)
        file_menu.addAction(self.save_act)
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.undo_act)
        edit_menu.addAction(self.redo_act)
        edit_menu.addAction(self.find_replace_act)

        file_menu.addSeparator()
        file_menu.addAction(self.exit_act)
        # Data menu actions
        self.pivot_act = QAction("Pivot Table…", self)
        self.pivot_act.triggered.connect(self._open_pivot_dialog)
        data_menu.addAction(self.pivot_act)

        # Help menu with reference link (duplicate for discoverability)
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.analysis_tools_act)
        help_menu.addAction(self.formula_wizard_act)
        help_menu.addAction(self.formula_ref_act)

    def _open_formula_wizard(self):
        try:
            from views.formula_wizard import FormulaWizardDialog
            dlg = FormulaWizardDialog(self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                formula = dlg.formula()
                if formula:
                    self.formula_bar.setText(formula)
                    self._apply_formula()
        except Exception as exc:
            QMessageBox.critical(self, "Formula Wizard", f"Failed to open wizard:\n{exc}")

    def _open_formula_reference(self):
        # Lazy import to avoid heavy startup
        try:
            from views.formula_reference import FormulaReferenceDialog
            dlg = FormulaReferenceDialog(self)
            dlg.exec()
        except Exception as exc:
            QMessageBox.critical(self, "Formula Reference", f"Failed to open reference:\n{exc}")

    def _open_formula_audit(self):
        # ensure single cell selected
        sel = self.table.selectionModel()
        if not sel or not sel.hasSelection() or len(sel.selectedIndexes()) != 1:
            QMessageBox.information(self, "Formula Audit", "Please select a single cell to audit.")
            return
        idx = sel.selectedIndexes()[0]
        from views.formula_audit import FormulaAuditDialog
        dlg = FormulaAuditDialog(self.table.model(), idx.row(), idx.column(), self)
        dlg.exec()

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        icon_px = self.style().pixelMetric(self.style().PixelMetric.PM_ToolBarIconSize)
        toolbar.setIconSize(QSize(icon_px, icon_px))
        self.addToolBar(toolbar)
        # file actions
        toolbar.addAction(self.new_act)
        toolbar.addAction(self.open_act)
        toolbar.addAction(self.save_act)
        toolbar.addSeparator()
        # chart and dashboard
        self.dashboard_act = QAction("Dashboard", self, checkable=True)
        self.dashboard_act.toggled.connect(self._toggle_dashboard)
        toolbar.addAction(self.chart_act)
        toolbar.addAction(self.dashboard_act)
        toolbar.addSeparator()
        # undo/redo + find/replace
        toolbar.addAction(self.undo_act)
        toolbar.addAction(self.redo_act)
        toolbar.addAction(self.find_replace_act)
        toolbar.addSeparator()
        # row/col actions
        toolbar.addAction(self.add_cols_act)
        toolbar.addAction(self.add_rows_act)
        toolbar.addSeparator()
        # conditional formatting & data validation
        self.cf_act = QAction("Conditional Formatting…", self)
        self.cf_act.triggered.connect(self._open_cf_dialog)
        self.clear_cf_act = QAction("Clear Formatting", self)
        self.clear_cf_act.triggered.connect(self._clear_cf_rules)
        self.format_menu.addAction(self.cf_act)
        self.format_menu.addAction(self.clear_cf_act)
        toolbar.addAction(self.cf_act)
        toolbar.addAction(self.clear_cf_act)
        # data validation actions
        self.val_act = QAction("Data Validation…", self)
        self.val_act.triggered.connect(self._open_val_dialog)
        self.clear_val_act = QAction("Clear Validation", self)
        self.clear_val_act.triggered.connect(self._clear_val_rules)
        self.format_menu.addAction(self.val_act)
        self.format_menu.addAction(self.clear_val_act)
        toolbar.addAction(self.val_act)
        toolbar.addAction(self.clear_val_act)
        toolbar.addSeparator()
        toolbar.addAction(self.exit_act)
        self.toolbar = toolbar

    # ---------------------------------------------------------------------
    # File operations
    # ---------------------------------------------------------------------
    # File operations
    # ---------------------------------------------------------------------
    def _new_file(self):
        """Create a new empty sheet."""
        df = pd.DataFrame([["" for _ in range(20)] for _ in range(50)])
        self._set_dataframe(df)
        self.current_path = None
        self.statusBar().showMessage("New file")
        self._update_status_info()

    def _open_file(self):
        filters = "CSV (*.csv);;Excel (*.xlsx *.xls);;JSON (*.json);;SQLite (*.db *.sqlite)"
        from utils.settings_manager import settings as _settings
        start_dir = _settings.get("default_path", "")
        fname, _ = QFileDialog.getOpenFileName(self, "Open file", start_dir, filters)
        if not fname:
            return
        try:
            from views.import_wizard import ImportWizard
            wiz = ImportWizard(fname, self)
            if wiz.exec() != wiz.DialogCode.Accepted:
                return
            df = wiz.get_dataframe()
            if df is None:
                return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open file:\n{e}")
            return

        # Show loading indicator while inserting DataFrame
        progress = QProgressDialog("Loading data…", None, 0, 0, self)
        progress.setMinimumDuration(0)
        progress.setWindowTitle("Loading")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()
        QTimer.singleShot(0, lambda d=df, p=progress, f=fname: self._finish_open(d, f, p))
        return

    def _open_file_path(self, fname: str):
        """Open *fname* directly (used by drag-and-drop)."""
        if not fname:
            return
        try:
            from views.import_wizard import ImportWizard
            wiz = ImportWizard(fname, self)
            if wiz.exec() != wiz.DialogCode.Accepted:
                return
            df = wiz.get_dataframe()
            if df is None:
                return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open file:\n{e}")
            return
        progress = QProgressDialog("Loading data…", None, 0, 0, self)
        progress.setMinimumDuration(0)
        progress.setWindowTitle("Loading")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()
        QTimer.singleShot(0, lambda d=df, p=progress, f=fname: self._finish_open(d, f, p))
        return

    
    def _finish_open(self, df: pd.DataFrame, fname: str, progress):
        """Finalize opening: set dataframe and close progress."""
        # Faster incremental reveal using visible row count without DataFrame copies
        from models.dataframe_model import DataFrameModel
        from types import MethodType
        total_rows = len(df)
        progress.setRange(0, total_rows)

        # Model already contains full DataFrame; we expose rows gradually
        model = DataFrameModel(df)
        model._visible_rows = 0  # custom attribute

        def _rowCount(self, parent=QModelIndex()):
            return getattr(self, '_visible_rows', 0)
        model.rowCount = MethodType(_rowCount, model)  # patch instance method

        self.table.setModel(model)
        self.table.resizeColumnsToContents()

        chunk_size = max(5000, total_rows // 20)  # ~20 steps

        def insert_chunk():
            start = model._visible_rows
            if start >= total_rows:
                progress.close()
                self.current_path = Path(fname)
                self.statusBar().showMessage(f"Opened {Path(fname).name}")
                # reconnect listeners after finished
                self.table.selectionModel().selectionChanged.connect(self._update_status)
                self.table.model().dataChanged.connect(self._on_dataframe_changed)
                self._update_status_info()
                return
            end = min(start + chunk_size, total_rows)
            model.beginInsertRows(QModelIndex(), start, end - 1)
            model._visible_rows = end
            model.endInsertRows()
            progress.setValue(end)
            QTimer.singleShot(0, insert_chunk)

        insert_chunk()

    # ---------------------------------------------------------------------
    # File save
    # ---------------------------------------------------------------------
    def _save_file(self):
        if self.current_path is None:
            filters = "CSV (*.csv);;Excel (*.xlsx);;JSON (*.json);;SQLite (*.db *.sqlite)"
            from utils.settings_manager import settings as _settings
            start_dir = str(Path(_settings.get("default_path", "")) / "sheet.csv")
            fname, _ = QFileDialog.getSaveFileName(self, "Save file", start_dir, filters)
            if not fname:
                return
            self.current_path = Path(fname)
        try:
            model: pd.DataFrame = self.table.model()._df  # type: ignore
            from io_utils import save_dataframe
            progress = QProgressDialog("Saving data…", None, 0, 0, self)
            progress.setMinimumDuration(0)
            progress.setWindowTitle("Saving")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setCancelButton(None)
            progress.show()
            QApplication.processEvents()
            try:
                save_dataframe(model, self.current_path)
            finally:
                progress.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save file:\n{e}")
            return
        self.statusBar().showMessage(f"Saved to {self.current_path.name}")

    def _insert_function(self, name: str):
        text = self.formula_bar.text()
        # If no leading '=', start fresh
        if not text.startswith("="):
            new_text = f"={name}()"
            self.formula_bar.setText(new_text)
            # place cursor inside parentheses
            self.formula_bar.setCursorPosition(len(new_text)-1)
            return
        # already has '=', insert after cursor (but not before '=')
        cursor = max(1, self.formula_bar.cursorPosition())
        new_text = text[:cursor] + name + "()" + text[cursor:]
        self.formula_bar.setText(new_text)
        self.formula_bar.setCursorPosition(cursor + len(name) + 1)

    def _prompt_add_rows(self):
        num, ok = QInputDialog.getInt(self, "Add Rows", "Number of rows to add:", value=1, min=1, max=10000)
        if ok:
            model: DataFrameModel = self.table.model()
            model.add_rows(num)
            self.table.resizeRowsToContents()

    def _prompt_add_columns(self):
        num, ok = QInputDialog.getInt(self, "Add Columns", "Number of columns to add:", value=1, min=1, max=1000)
        if ok:
            model: DataFrameModel = self.table.model()
            model.add_columns(num)
            self.table.resizeColumnsToContents()

    def _update_status(self):
        indexes = self.table.selectedIndexes()
        if not indexes:
            self.status_pos.clearMessage()
            return
        idx = indexes[0]
        model = self.table.model()
        try:
            col_letter = model._num_to_col_letter(idx.column())
            raw_val = model._df.iat[idx.row(), idx.column()]
            self.formula_bar.setText(str(raw_val))
        except Exception:
            col_letter = idx.column()
            raw_val = ""
        row_num = idx.row() + 1
        self.status_pos.showMessage(f"{col_letter}{row_num} : {raw_val}")

    def _rename_column(self, section: int):
        current = self.table.model().headerData(section, Qt.Orientation.Horizontal, role=Qt.ItemDataRole.DisplayRole)  # type: ignore
        name, ok = QInputDialog.getText(self, "Rename Column", "New name:", text=str(current))
        if ok and name:
            self.table.model().rename_column(section, name)  # type: ignore

    def _open_chart_builder(self):
        from views.chart_builder import ChartBuilderDialog
        dlg = ChartBuilderDialog(self.table.model()._df.copy(), self._chart_types, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        chart_type, df_numeric, params = dlg.result()
        if df_numeric.empty:
            QMessageBox.warning(self, "Chart", "No numeric data selected.")
            return
        import matplotlib.pyplot as plt
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        try:
            from charts import plot_chart
            plot_chart(ax, chart_type, df_numeric, params)
        except Exception as exc:
            QMessageBox.critical(self, "Chart", str(exc))
            return
        canvas.draw()
        self._add_chart_canvas(canvas)
        # determine column indices for live updates
        col_indices = [self.table.model()._df.columns.get_loc(c) for c in params.get('columns', list(df_numeric.columns))]
        self._register_chart(canvas, ax, [], col_indices, chart_type, params)

    # Legacy selection-based insert chart (still available internally if needed)
    def _insert_chart(self):
        idxs = self.table.selectedIndexes()
        if not idxs:
            QMessageBox.information(self, "Chart", "Select some cells to chart.")
            return
        # choose chart type
        chart_type, ok = QInputDialog.getItem(self, "Chart Type", "Select chart type:", self._chart_types, 0, False)
        if not ok:
            return
        rows = sorted({i.row() for i in idxs})
        cols = sorted({i.column() for i in idxs})
        model = self.table.model()
        df_sel = model._df.iloc[rows, cols]  # type: ignore
        df_numeric = df_sel.apply(pd.to_numeric, errors="coerce")
        if df_numeric.isnull().all().all():
            QMessageBox.warning(self, "Chart", "Selected data is not numeric.")
            return
        import matplotlib.pyplot as plt
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        try:
            if chart_type == "Line":
                df_numeric.plot(ax=ax)
            elif chart_type == "Area":
                df_numeric.plot.area(ax=ax, stacked=False)
            elif chart_type == "Stacked Bar":
                df_numeric.plot.bar(ax=ax, stacked=True)
            elif chart_type in ("Pie", "Donut", "Stacked Bar"):
                dlg = GroupByDialog(model._df.iloc[rows+cols])  # pass full df subset
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                group_col, val_col, agg = dlg.choices()
                base_df = model._df.iloc[rows].copy()
                if val_col == "--Count--":
                    grouped = base_df.groupby(group_col).size()
                else:
                    func = {'Sum': 'sum', 'Mean': 'mean'}.get(agg, 'sum')
                    grouped = base_df.groupby(group_col)[val_col].agg(func)
                if chart_type == "Stacked Bar":
                    grouped.plot.bar(ax=ax)
                else:
                    wedges, texts, autotexts = ax.pie(grouped.values, labels=grouped.index, autopct='%1.1f%%')
                    if chart_type=="Donut":
                        centre_circle = plt.Circle((0,0),0.70,fc='white')
                        ax.add_artist(centre_circle)
                        ax.set_aspect('equal')
            elif chart_type == "Bubble":
                if df_numeric.shape[1] < 3:
                    QMessageBox.warning(self, "Chart", "Select at least 3 numeric columns for Bubble chart (x, y, size).")
                    return
                # Let user pick columns order
                col_names = list(df_numeric.columns)
                x_col, ok = QInputDialog.getItem(self, "Bubble Chart", "X axis column:", col_names, 0, False)
                if not ok:
                    return
                y_col, ok = QInputDialog.getItem(self, "Bubble Chart", "Y axis column:", col_names, 1, False)
                if not ok:
                    return
                size_col, ok = QInputDialog.getItem(self, "Bubble Chart", "Size column:", col_names, 2, False)
                if not ok:
                    return
                data = df_numeric[[x_col, y_col, size_col]].dropna()
                sizes = data[size_col].abs() * 20
                ax.scatter(data[x_col], data[y_col], s=sizes, alpha=0.6, edgecolors='w')
            elif chart_type == "Waterfall":
                vals = df_numeric.iloc[0] if df_numeric.shape[0]==1 else df_numeric.iloc[:,0]
                cumulative = vals.cumsum()-vals
                ax.bar(vals.index, vals, bottom=cumulative)
            elif chart_type == "Boxplot":
                df_numeric.plot.box(ax=ax)
            elif chart_type == "Radar":
                # Prepare radar plot
                categories = list(df_numeric.columns)
                if len(categories) < 3:
                    QMessageBox.warning(self, "Chart", "Radar chart requires at least 3 numeric columns.")
                    return
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]
                ax = fig.add_subplot(111, projection="polar")
                for idx, (_, row) in enumerate(df_numeric.iterrows()):
                    values = row.tolist()
                    values += values[:1]
                    ax.plot(angles, values, label=str(idx))
                    ax.fill(angles, values, alpha=0.1)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.grid(True)
                if len(df_numeric) > 1:
                    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))
            elif chart_type == "Heatmap":
                im = ax.imshow(df_numeric.values, cmap='viridis', aspect='auto')
                ax.set_xticks(range(len(df_numeric.columns)))
                ax.set_xticklabels(df_numeric.columns)
                ax.set_yticks(range(len(df_numeric.index)))
                ax.set_yticklabels(df_numeric.index)
                plt.colorbar(im, ax=ax)
            else:
                QMessageBox.warning(self, "Chart", "Chart type not implemented.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Chart Error", str(e))
            return
        ax.set_title(chart_type)
        canvas.draw()
        # add to tabs
        self._add_chart_canvas(canvas)
        # register for live updates
        extra_params = locals().copy()
        extra_params['df_full'] = model._df
        self._register_chart(canvas, ax, rows, cols, chart_type, extra_params)
        # show properties dock
        if not hasattr(self, "_chart_prop_dock"):
            self._chart_prop_dock = QDockWidget("Chart Properties", self)
            self._chart_prop_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._chart_prop_dock)
        self._chart_prop_dock.setWidget(ChartPropertiesWidget(ax, canvas, self))
        self._chart_prop_dock.show()


    # ------------------------- Chart tracking ------------------------
    def _register_chart(self, canvas, ax, rows, cols, chart_type, params):
        info = {
            'canvas': canvas,
            'ax': ax,
            'rows': rows,
            'cols': cols,
            'chart_type': chart_type,
            'params': params,
            'live': True,
        }
        self._charts.append(info)
        # context menu for detach/attach
        canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        canvas.customContextMenuRequested.connect(lambda pos, c=canvas: self._show_chart_menu(c, pos))

    def _show_chart_menu(self, canvas, pos):
        chart = next((c for c in self._charts if c['canvas'] is canvas), None)
        if chart is None:
            return
        menu = QMenu(canvas)
        act_live = menu.addAction("Detach" if chart['live'] else "Attach")
        act_live.triggered.connect(lambda: self._toggle_chart_live(chart))
        act_filter = menu.addAction("Filter…")
        act_filter.triggered.connect(lambda: self._filter_chart(chart))
        menu.exec(canvas.mapToGlobal(pos))

    def _filter_chart(self, chart):
        from views.chart_filter import ChartFilterDialog
        dlg = ChartFilterDialog(self.table.model()._df, chart['params'].get('filters'), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            chart['params']['filters'] = dlg.filters()
            self._redraw_chart(chart)

    def _toggle_chart_live(self, chart):
        chart['live'] = not chart['live']

    # -----------------------------------------------------------------
    def _add_chart_canvas(self, canvas: QWidget):
        """Place canvas either in tabs or dashboard grid depending on mode."""
        if self._dashboard_mode:
            idx = self.dashboard_grid.count()
            row, col = divmod(idx, 2)
            self.dashboard_grid.addWidget(canvas, row, col)
        else:
            # ensure chart panel visible
            if not self.chart_tabs.isVisible():
                self.chart_tabs.show()
            tab_name = f"Chart {self.chart_tabs.count()+1}"
            self.chart_tabs.addTab(canvas, tab_name)
            self.chart_tabs.setCurrentWidget(canvas)
            self.splitter.setSizes([700, 300])

    def _toggle_dashboard(self, checked: bool):
        # move canvases between container and tabs
        if checked:
            self._dashboard_mode = True
            # move all canvases from tabs to grid
            canvases = []
            while self.chart_tabs.count():
                w = self.chart_tabs.widget(0)
                self.chart_tabs.removeTab(0)
                canvases.append(w)
            for idx, w in enumerate(canvases):
                row, col = divmod(idx, 2)
                self.dashboard_grid.addWidget(w, row, col)
            # replace widget in splitter
            self.splitter.replaceWidget(1, self.dashboard_area)
            self.chart_tabs.hide()
            self.dashboard_area.show()
            # ensure widths reasonable
            self.splitter.setSizes([700, 300])
        else:
            self._dashboard_mode = False
            # move canvases back to tabs
            for i in reversed(range(self.dashboard_grid.count())):
                item = self.dashboard_grid.itemAt(i)
                w = item.widget()
                self.dashboard_grid.removeWidget(w)
                tab_name = f"Chart {self.chart_tabs.count()+1}"
                self.chart_tabs.addTab(w, tab_name)
            self.splitter.replaceWidget(1, self.chart_tabs)
            self.dashboard_area.hide()
            self.chart_tabs.show()
            self.splitter.setSizes([700, 300])

    def _on_dataframe_changed(self, topLeft, bottomRight, roles):
        tr = range(topLeft.row(), bottomRight.row()+1)
        tc = range(topLeft.column(), bottomRight.column()+1)
        for chart in self._charts:
            if not chart['live']:
                continue
            rows = chart['rows']
            cols = chart['cols']
            # Determine intersections
            row_intersect = True if not rows else not (max(rows) < min(tr) or min(rows) > max(tr))
            col_intersect = True if not cols else not (max(cols) < min(tc) or min(cols) > max(tc))
            if row_intersect and col_intersect:
                self._redraw_chart(chart)

    def _redraw_chart(self, chart):
        ax = chart['ax']
        canvas = chart['canvas']
        chart_type = chart['chart_type']
        rows, cols = chart['rows'], chart['cols']
        model = self.table.model()
        params = chart['params']
        filters = params.get('filters')
        def _apply_filters(df):
            if not filters:
                return df
            for col, spec in filters.items():
                if col not in df.columns:
                    continue
                if spec['type'] == 'numeric':
                    df = df[df[col].between(spec['min'], spec['max'])]
                else:
                    df = df[df[col].astype(str).isin(spec['values'])]
            return df

        if params.get('builder'):
            cols_names = params.get('columns', [])
            if not cols_names:
                return
            df_base = _apply_filters(model._df)
            df_sel = df_base[cols_names].copy()
            num_cols = [c for c in df_sel.columns if pd.api.types.is_numeric_dtype(df_sel[c])]
            cat = params.get('groupby')
            agg_name = params.get('agg', 'Mean')
            if cat and cat in df_sel.columns and num_cols:
                func_map = {
                    'Mean': 'mean', 'Sum': 'sum', 'Median': 'median', 'Min': 'min', 'Max': 'max'
                }
                if agg_name == 'Count':
                    df_numeric = pd.DataFrame({'Count': df_sel.groupby(cat).size()})
                else:
                    func = func_map.get(agg_name, 'mean')
                    df_numeric = df_sel.groupby(cat)[num_cols].agg(func)
            else:
                numeric_part = df_sel.select_dtypes(include='number')
                if numeric_part.empty and chart_type in ('Pie','Donut','Stacked Bar') and not df_sel.empty:
                    series = df_sel.iloc[:,0].astype(str).value_counts()
                    df_numeric = pd.DataFrame({series.name or 'count': series})
                else:
                    df_numeric = df_sel.apply(pd.to_numeric, errors='coerce')
        else:
            df_sel = model._df.iloc[rows, cols]
            df_sel = _apply_filters(df_sel)
            df_numeric = df_sel.apply(pd.to_numeric, errors='coerce')
        import charts
        ax.clear()
        charts.plot_chart(ax, chart_type, df_numeric, chart['params'])
        canvas.draw_idle()

    def _plot_chart(self, ax, chart_type, df_numeric, params):
        """Delegate to shared charts utility."""
        import charts
        charts.plot_chart(ax, chart_type, df_numeric, params)
        return

        if chart_type == "Line":
            df_numeric.plot(ax=ax)
        elif chart_type == "Area":
            df_numeric.plot.area(ax=ax, stacked=False)

        elif chart_type == "Stacked Bar":
            df_numeric.plot.bar(ax=ax, stacked=True)
        elif chart_type == "Boxplot":
            df_numeric.plot.box(ax=ax)
        elif chart_type in ("Pie", "Donut"):
            # If chart was created via GroupByDialog, params include 'group_col'
            if 'group_col' in params:
                group_col = params.get('group_col')
                val_col = params.get('val_col')
                agg = params.get('agg') or 'Sum'
                # Use full dataframe rows that were initially selected
                df_full = self.table.model()._df
                base_df = df_full.iloc[params['rows']].copy()
                if val_col == "--Count--" or val_col is None:
                    series = base_df.groupby(group_col).size()
                else:
                    func = {'Sum': 'sum', 'Mean': 'mean'}.get(agg, 'sum')
                    series = base_df.groupby(group_col)[val_col].agg(func)
            else:
                # Fallback to numeric selection aggregate
                if df_numeric.shape[0] == 1:
                    series = df_numeric.iloc[0]
                else:
                    series = df_numeric.mean()
            import numpy as np
            series = series.dropna().astype(float)
            series = series[series > 0]
            if series.empty or np.isclose(series.sum(), 0):
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                return
            wedges, texts, autotexts = ax.pie(series.values, labels=series.index, autopct='%1.1f%%')
            if chart_type == "Donut":
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                ax.add_artist(centre_circle)
                ax.set_aspect('equal')
        elif chart_type == "Bubble":
            x_col = params.get('x_col')
            y_col = params.get('y_col')
            size_col = params.get('size_col')
            if x_col and y_col and size_col and all(c in df_numeric.columns for c in (x_col, y_col, size_col)):
                data = df_numeric[[x_col, y_col, size_col]].dropna()
                sizes = data[size_col].abs() * 20
                ax.scatter(data[x_col], data[y_col], s=sizes, alpha=0.6, edgecolors='w')
            else:
                df_numeric.plot.scatter(x=df_numeric.columns[0], y=df_numeric.columns[1], ax=ax)
        elif chart_type == "Radar":
            categories = list(df_numeric.columns)
            if len(categories) < 3:
                return
            import numpy as np
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            for idx, (_, row) in enumerate(df_numeric.iterrows()):
                values = row.tolist(); values += values[:1]
                ax.plot(angles, values, label=str(idx))
                ax.fill(angles, values, alpha=0.1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.grid(True)
        elif chart_type == "Waterfall":
            vals = df_numeric.iloc[0] if df_numeric.shape[0]==1 else df_numeric.iloc[:,0]
            cumulative = vals.cumsum()-vals
            ax.bar(vals.index, vals, bottom=cumulative)
        elif chart_type == "Heatmap":
            im = ax.imshow(df_numeric.values, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(df_numeric.columns)))
            ax.set_xticklabels(df_numeric.columns)
            ax.set_yticks(range(len(df_numeric.index)))
            ax.set_yticklabels(df_numeric.index)
        else:
            # fallback simple plot
            df_numeric.plot(ax=ax)
        ax.set_title(chart_type)

    # existing methods
    def _close_chart_tab(self, index: int):
        widget = self.chart_tabs.widget(index)
        self.chart_tabs.removeTab(index)
        widget.deleteLater()
        if self.chart_tabs.count() == 0 and not self._dashboard_mode:
            self.chart_tabs.hide()
            # collapse splitter second pane
            self.splitter.setSizes([1, 0])

    def _detect_outliers(self, method: str):
        model = self.table.model()
        # build list of numeric columns
        num_cols_idx = [i for i,c in enumerate(model._df.columns) if pd.api.types.is_numeric_dtype(model._df[c])]
        if not num_cols_idx:
            QMessageBox.warning(self, "Outlier Detection", "No numeric columns found.")
            return
        items = [str(model._df.columns[i]) for i in num_cols_idx]
        items.insert(0, "All numeric columns")
        col_name, ok = QInputDialog.getItem(self, "Outlier Detection", "Choose column:", items, 0, False)
        if not ok:
            return
        rows = list(range(model.rowCount()))
        if col_name == "All numeric columns":
            cols = num_cols_idx
        else:
            col_idx = model._df.columns.get_loc(col_name)
            cols = [col_idx]
        model = self.table.model()
        try:
            n = model.mark_outliers(rows, cols, method)
        except Exception as e:
            QMessageBox.critical(self, "Outlier Detection", str(e))
            return
        if n:
            QMessageBox.information(self, "Outlier Detection", f"Highlighted {n} outlier cells.")
        else:
            QMessageBox.information(self, "Outlier Detection", "No outliers detected in selection.")

    def _apply_formula(self):
        text = self.formula_bar.text()
        idxs = self.table.selectedIndexes()
        if not idxs:
            return
        idx = idxs[0]
        self.table.model().setData(idx, text, role=Qt.ItemDataRole.EditRole)  # type: ignore
        # update status bar with formula result
        value = self.table.model().data(idx, Qt.ItemDataRole.DisplayRole)
        self.formula_label.setText(f"= {value}")
        self._update_status_selection()

    # ---------------------------------------------------------------------
    # Status bar helpers
    # ---------------------------------------------------------------------
    def _setup_status_connections(self):
        if self.table.selectionModel() is None:
            return
        sel_model = self.table.selectionModel()
        model = self.table.model()

        # Lightweight selection size / value update on every selection or data edit
        sel_model.selectionChanged.connect(self._update_status_selection)
        model.dataChanged.connect(lambda *_: self._update_status_selection())

        # Heavier full-info refresh (rows/cols/memory) only when table structure changes
        for sig in (model.rowsInserted, model.rowsRemoved,
                    model.columnsInserted, model.columnsRemoved,
                    model.layoutChanged):
            sig.connect(lambda *_: self._update_status_info())

        # Initial population
        self._update_status_selection()
        self._update_status_info()

    def _update_status_selection(self):
        idxs = self.table.selectedIndexes()
        if not idxs:
            self.sel_label.setText("Sel: 0×0")
            self.value_label.setText("Value: ")
            return
        rows = {i.row() for i in idxs}
        cols = {i.column() for i in idxs}
        self.sel_label.setText(f"Sel: {len(rows)}×{len(cols)}")
        first = idxs[0]
        val = self.table.model().data(first, Qt.ItemDataRole.DisplayRole)
        self.value_label.setText(f"Value: {val}")

    def _update_status_info(self):
        """Update total row/column count and memory usage."""
        try:
            df = self.table.model()._df  # type: ignore
            rows, cols = len(df), len(df.columns)
            mem = df.memory_usage(deep=True).sum()
            unit = "B"
            for u in ("KB", "MB", "GB"):
                if mem < 1024:
                    break
                mem /= 1024
                unit = u
            self.info_label.setText(f"Rows: {rows:,} Cols: {cols:,} Mem: {mem:.1f} {unit}")
        except Exception:
            self.info_label.setText("Rows: ? Cols: ? Mem: ?")

    # ---------------------------------------------------------------------
    # Theme handling
    # ---------------------------------------------------------------------
    # qt-material theme application replaces legacy QSS loader
    def _apply_qss(self, path: str):
        # kept for backward compatibility – delegates to qt-material
        self._apply_material()

    def _apply_material(self, theme_name: str | None = None):
        """Apply qt-material theme to the whole application."""
        try:
            app = QApplication.instance()
            if app is None:
                return
            if theme_name is None:
                # choose based on dark_mode flag if present
                theme_name = "dark_cyan.xml" if getattr(self, "_dark_mode", False) else "light_blue.xml"
            apply_stylesheet(app, theme_name)
        except Exception as e:
            print("qt-material theme failed", e)

    def _apply_theme(self):
        """Apply currently selected qt-material theme file."""
        file_name = self._theme_files.get(self._current_theme)
        if file_name is None:
            file_name = "light_blue.xml"
        self._apply_material(file_name)

    def _set_theme(self, name: str):
        """Switch theme and sync UI actions."""
        if name not in self._theme_files:
            return
        self._current_theme = name
        file_name = self._theme_files[name]
        self._dark_mode = file_name.startswith("dark_")
        if hasattr(self, "dark_mode_act"):
            self.dark_mode_act.setChecked(self._dark_mode)
        if hasattr(self, "_theme_group"):
            for act in self._theme_group.actions():
                act.setChecked(act.text() == name)
        self._apply_theme()

    # ---------------------------------------------------------------------
    # Pivot Table dialog handling
    # ---------------------------------------------------------------------
    def _open_pivot_dialog(self):
        from models.dataframe_model import DataFrameModel
        model: DataFrameModel = self.table.model()  # type: ignore
        cols = list(model._df.columns)
        from views.pivot_builder import PivotTableDialog
        dlg = PivotTableDialog(cols, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        rows, cols_sel, value_fields, aggs, filters = dlg.get_settings()
        if not rows:
            QMessageBox.warning(self, "Pivot", "Select at least one row field.")
            return
        if not value_fields:
            QMessageBox.warning(self, "Pivot", "Select at least one value field.")
            return
        if not rows:
            QMessageBox.warning(self, "Pivot", "Select at least one row field.")
            return
        base_df = model._df.copy()
        # apply filters
        if filters:
            for fcol in filters:
                uniq_vals = sorted(base_df[fcol].dropna().astype(str).unique().tolist())
                if not uniq_vals:
                    continue
                sel, ok = QInputDialog.getItem(self, "Filter", f"Select value for {fcol} (or All):", ["<All>"] + uniq_vals, 0, False)
                if not ok:
                    return
                if sel != "<All>":
                    base_df = base_df[base_df[fcol].astype(str) == sel]

        from analysis.pivot import pivot_table
        try:
            pivot_df = pivot_table(
                base_df,
                index=rows,
                columns=cols_sel,
                values=value_fields,
                aggs=aggs,
            )
        except Exception as e:
            QMessageBox.critical(self, "Pivot Error", str(e))
            return
        from views.pivot_result import PivotResultDialog
        dlg = PivotResultDialog(pivot_df, self)
        dlg.exec()
        return
        """
        # dlg_res = QDialog(self)
        dlg_res.setWindowTitle("Pivot Table Result")
        vlayout = QVBoxLayout(dlg_res)
        table = QTableView()
        table.setModel(DataFrameModel(pivot_df.reset_index()))
        table.horizontalHeader().setStretchLastSection(True)
        vlayout.addWidget(table)

        # Buttons row
        btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy")
        export_btn = QPushButton("Export CSV…")
        chart_btn = QPushButton("Chart…")
        close_btn = QPushButton("Close")
        btn_row.addStretch()
        for b in (copy_btn, export_btn, chart_btn, close_btn):
            btn_row.addWidget(b)
        vlayout.addLayout(btn_row)

        # Handlers
        from PyQt6.QtWidgets import QApplication, QFileDialog
        from views.chart_builder import ChartBuilderDialog
        _chart_types = ["Bar", "Stacked Bar", "Line", "Area", "Pie", "Heatmap"]

        def _copy_pivot():
            QApplication.clipboard().setText(pivot_df.to_csv(index=True))

        def _export_csv():
            fname, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
            if fname:
                pivot_df.to_csv(fname, index=True)

        def _open_pivot_chart():
            dlg_chart = ChartBuilderDialog(pivot_df.reset_index(), _chart_types, self)
            dlg_chart.exec()

        copy_btn.clicked.connect(_copy_pivot)
        export_btn.clicked.connect(_export_csv)
        chart_btn.clicked.connect(_open_pivot_chart)
        close_btn.clicked.connect(dlg_res.close)

        dlg_res.resize(800, 600)
        dlg_res.exec()
"""

    def _open_analysis_tools(self):
        from models.dataframe_model import DataFrameModel
        model: DataFrameModel = self.table.model()  # type: ignore
        from views.analysis_tools import AnalysisDialog
        dlg = AnalysisDialog(model._df.copy(), self)
        dlg.exec()

    def _clear_cf_rules(self):
        from models.dataframe_model import DataFrameModel
        model: DataFrameModel = self.table.model()  # type: ignore
        if not model._cf_rules:
            QMessageBox.information(self, "Conditional Formatting", "No rules to clear.")
            return
        model.clear_cf_rules()
        self.table.viewport().update()
        QMessageBox.information(self, "Conditional Formatting", "All formatting rules cleared.")

    def _open_cf_dialog(self):
        idxs=self.table.selectedIndexes()
        if not idxs:
            QMessageBox.information(self,"Conditional Formatting","Select some cells first.")
            return
        rows=[i.row() for i in idxs]; cols=[i.column() for i in idxs]
        dlg=ConditionalFormattingDialog(self)
        if dlg.exec()==QDialog.DialogCode.Accepted:
            op,val_text,color=dlg.get_settings()
            try:
                val=float(val_text)
            except ValueError:
                QMessageBox.warning(self,"Invalid","Value must be numeric")
                return
            model: DataFrameModel=self.table.model()
            model.add_cf_rule(min(rows),max(rows),min(cols),max(cols),op,val,color)
            self.table.viewport().update()

    def _clear_val_rules(self):
        from models.dataframe_model import DataFrameModel
        model: DataFrameModel = self.table.model()  # type: ignore
        if not model._validation_rules:
            QMessageBox.information(self, "Data Validation", "No rules to clear.")
            return
        model.clear_validation_rules()
        self.table.viewport().update()
        QMessageBox.information(self, "Data Validation", "All validation rules cleared.")

    def _open_val_dialog(self):
        idxs = self.table.selectedIndexes()
        if not idxs:
            QMessageBox.information(self, "Data Validation", "Select some cells first.")
            return
        rows = [i.row() for i in idxs]
        cols = [i.column() for i in idxs]
        from views.data_validation_dialog import DataValidationDialog
        dlg = DataValidationDialog(self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        settings = dlg.get_settings()
        from models.dataframe_model import DataFrameModel
        model: DataFrameModel = self.table.model()  # type: ignore
        if settings['type'] == 'number':
            try:
                v1 = float(settings['value']) if settings['value'] not in (None, "") else None
                v2 = float(settings['value2']) if settings.get('value2') else None
            except ValueError:
                QMessageBox.warning(self, "Invalid", "Values must be numeric.")
                return
            model.add_validation_rule(min(rows), max(rows), min(cols), max(cols), 'number', settings['op'], v1, v2, None, settings['mode'])
        else:
            list_vals = settings['list']
            if not list_vals:
                QMessageBox.warning(self, "Invalid", "List cannot be empty.")
                return
            model.add_validation_rule(min(rows), max(rows), min(cols), max(cols), 'list', None, None, None, list_vals, settings['mode'])
        self.table.viewport().update()

    # ---------------------------------------------------------------------
    # Sheet helpers
    # ---------------------------------------------------------------------
    def _add_sheet(self, name: str | None = None):
        # QAction.triggered passes a boolean (checked) argument; ignore it
        if isinstance(name, bool):
            name = None
        from views.spreadsheet_view import SpreadsheetView
        from models.dataframe_model import DataFrameModel
        import pandas as pd
        if name is None:
            base = "Sheet"
            idx = 1
            existing = {self.sheet_tabs.tabText(i) for i in range(self.sheet_tabs.count())}
            while f"{base}{idx}" in existing:
                idx += 1
            name = f"{base}{idx}"
        view = SpreadsheetView()
        view.setAlternatingRowColors(True)
        hdr = view.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setDefaultSectionSize(100)
        view.verticalHeader().setDefaultSectionSize(24)
        empty_df = pd.DataFrame([["" for _ in range(20)] for _ in range(50)])
        view.setModel(DataFrameModel(empty_df))
        self.sheet_tabs.addTab(view, name)
        self.sheet_tabs.setCurrentWidget(view)

    def _delete_sheet(self, index: int):
        if index < 0 or self.sheet_tabs.count() <= 1:
            return  # keep at least one sheet
        widget = self.sheet_tabs.widget(index)
        self.sheet_tabs.removeTab(index)
        widget.deleteLater()

    def _rename_sheet(self, index: int):
        if index < 0:
            return
        current_name = self.sheet_tabs.tabText(index)
        new_name, ok = QInputDialog.getText(self, "Rename Sheet", "Sheet name:", text=current_name)
        if ok and new_name and new_name not in [self.sheet_tabs.tabText(i) for i in range(self.sheet_tabs.count())]:
            self.sheet_tabs.setTabText(index, new_name)

    def _on_sheet_changed(self, index: int):
        view = self.sheet_tabs.widget(index)
        if view is not None:
            self.table = view
            self._setup_status_connections()

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Find / Replace handler
    # ---------------------------------------------------------------------
    def _open_settings_dialog(self):
        from views.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        new_vals = dlg.get_settings()
        for k, v in new_vals.items():
            settings.set(k, v)
        settings.save()
        # apply theme if changed
        if new_vals.get("theme") != self._current_theme:
            self._set_theme(new_vals["theme"])

    def _show_find_replace(self):
        opts = FindReplaceDialog.get_find_replace(self)
        if not opts:
            return
        total = 0
        if opts.get("all"):
            for i in range(self.sheet_tabs.count()):
                view = self.sheet_tabs.widget(i)
                model = view.model()
                replaced = model.search_replace(opts["find"], opts["replace"], regex=opts["regex"], match_case=opts["case"])
                if replaced:
                    total += replaced
            QMessageBox.information(self, "Find & Replace", f"Replaced {total} occurrence(s) across all sheets.")
        else:
            model = self.table.model()
            replaced = model.search_replace(opts["find"], opts["replace"], regex=opts["regex"], match_case=opts["case"])
            QMessageBox.information(self, "Find & Replace", f"Replaced {replaced} occurrence(s) in current sheet.")

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Drag-and-drop events
    # ---------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            # accept only first local file URL
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self._open_file_path(url.toLocalFile())
                break
        event.acceptProposedAction()

    # Theme toggle legacy (now unused)
    # ---------------------------------------------------------------------
    def _toggle_theme(self, checked):
        # Map legacy toggle to Light/Dark themes
        self._set_theme("Dark" if checked else "Light")

    # ---------------------------------------------------------------------
    # Data handling
    # ---------------------------------------------------------------------
    def _set_dataframe(self, df: pd.DataFrame):
        from models import DataFrameModel

        # use external structured model
        self.table.setModel(DataFrameModel(df))
        self.table.resizeColumnsToContents()
        # connect selection after model is set
        self.table.selectionModel().selectionChanged.connect(self._update_status)
        # connect for live chart updates
        self.table.model().dataChanged.connect(self._on_dataframe_changed)
        self._update_status_info()
        return  # skip legacy inlined model below

        class DataFrameModel(QAbstractTableModel):
            def __init__(self, _df: pd.DataFrame):
                super().__init__()
                self._df = _df

            def rowCount(self, parent: QModelIndex = QModelIndex()):
                return len(self._df.index)

            def columnCount(self, parent: QModelIndex = QModelIndex()):
                return len(self._df.columns)

            def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
                if not index.isValid():
                    return None
                raw = self._df.iat[index.row(), index.column()]
                if role == Qt.ItemDataRole.EditRole:
                    return str(raw)
                if role == Qt.ItemDataRole.DisplayRole:
                    return str(self._compute_value(index.row(), index.column(), set()))
                return None

            def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole):
                if role == Qt.ItemDataRole.EditRole:
                    self._df.iat[index.row(), index.column()] = value
                    self.dataChanged.emit(self.index(0,0), self.index(self.rowCount()-1, self.columnCount()-1))
                    return True
                return False

            def flags(self, index: QModelIndex):
                return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable

            # ------------------------ headers ----------------------------
            def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
                if role != Qt.ItemDataRole.DisplayRole:
                    return None
                if orientation == Qt.Orientation.Horizontal:
                    return self._num_to_col_letter(section)
                else:  # Vertical headers: row numbers starting at 1
                    return str(section + 1)

            @staticmethod
            def _num_to_col_letter(n: int) -> str:
                """Convert 0-based column index to Excel-style letters."""
                result = ""
                while True:
                    n, rem = divmod(n, 26)
                    result = chr(65 + rem) + result
                    if n == 0:
                        break
                    n -= 1  # adjust because Excel has no 0 letter
                return result

            # ---------------------- formula engine ------------------------
            import re
            _re_cell = re.compile(r"([A-Z]+)(\d+)")
            _re_range = re.compile(r"([A-Z]+\d+):([A-Z]+\d+)")

            def _compute_value(self, row: int, col: int, visiting: set):
                key = (row, col)
                if key in visiting:
                    return "#CYCLE"
                visiting.add(key)
                raw = self._df.iat[row, col]
                if isinstance(raw, str) and raw.startswith("="):
                    expr = raw[1:]
                    import math

                    # 1) replace ranges (e.g., A1:B3)
                    def repl_range(m):
                        start, end = m.groups()
                        cells = self._expand_range(start, end)
                        vals = [self._compute_value(r, c, visiting) for r, c in cells]
                        return str(vals)
                    expr = self._re_range.sub(repl_range, expr)

                    # 2) replace single cell refs
                    def repl_cell(match):
                        col_letters, row_num = match.groups()
                        r = int(row_num) - 1
                        c = self._col_letter_to_num(col_letters.upper())
                        return str(self._compute_value(r, c, visiting))
                    safe_expr = self._re_cell.sub(repl_cell, expr)

                    # allowed functions
                    def _flatten(args):
                        for a in args:
                            if isinstance(a, (list, tuple)):
                                yield from _flatten(a)
                            else:
                                yield a
                    def SUM(*args):
                        return sum(_flatten(args))
                    def AVERAGE(*args):
                        flat=list(_flatten(args))
                        return sum(flat)/len(flat) if flat else 0
                    def MIN(*args):
                        return min(_flatten(args))
                    def MAX(*args):
                        return max(_flatten(args))
                    def COUNT(*args):
                        return sum(1 for x in _flatten(args) if isinstance(x,(int,float)))
                    def COUNTA(*args):
                        return sum(1 for x in _flatten(args) if x not in ("", None))
                    def PRODUCT(*args):
                        p=1
                        for x in _flatten(args):
                            p*=x
                        return p
                    def SQRT(x):
                        return math.sqrt(x)
                    def POW(x,y):
                        return math.pow(x,y)
                    allowed = {"SUM":SUM,"AVERAGE":AVERAGE,"MIN":MIN,"MAX":MAX,"COUNT":COUNT,"COUNTA":COUNTA,
                               "PRODUCT":PRODUCT,"ABS":abs,"SQRT":SQRT,"POW":POW}
                    try:
                        value = eval(safe_expr, {"__builtins__": None}, allowed)
                    except Exception:
                        value = "#ERR"
                    visiting.discard(key)
                    return value
                else:
                    visiting.discard(key)
                    return raw

            @staticmethod
            def _col_letter_to_num(letters: str) -> int:
                num = 0
                for ch in letters:
                    num = num * 26 + (ord(ch) - 64)
                return num - 1

            def _expand_range(self, start: str, end: str):
                s_col_letters, s_row = re.match(r"([A-Z]+)(\d+)", start).groups()
                e_col_letters, e_row = re.match(r"([A-Z]+)(\d+)", end).groups()
                r1, r2 = int(s_row)-1, int(e_row)-1
                c1, c2 = self._col_letter_to_num(s_col_letters), self._col_letter_to_num(e_col_letters)
                cells=[]
                for r in range(min(r1,r2), max(r1,r2)+1):
                    for c in range(min(c1,c2), max(c1,c2)+1):
                        cells.append((r,c))
                return cells

        self.table.setModel(DataFrameModel(df))
        self.table.resizeColumnsToContents()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
