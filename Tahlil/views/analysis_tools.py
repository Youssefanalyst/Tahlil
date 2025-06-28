"""Dialogs for advanced statistical analyses: correlation, regression, hypothesis tests."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QTabWidget,
    QTableView,

    QFileDialog,
    QMessageBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # type: ignore
from matplotlib.figure import Figure

# ---------------------------------------------------------------------
# Helper utils
# ---------------------------------------------------------------------

def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# ---------------------------------------------------------------------
class SummaryTab(QDialog):
    """Tab that shows descriptive statistics (count, mean, std, etc.)."""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df
        self.setWindowTitle("Summary Statistics")
        layout = QVBoxLayout(self)

        hl = QHBoxLayout()
        hl.addWidget(QLabel("Columns:"))
        self.col_list = QListWidget()
        self.col_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for col in df.columns:
            col_str = str(col)
            item = QListWidgetItem(str(col))
            if pd.api.types.is_numeric_dtype(df[col]):
                item.setSelected(True)
            self.col_list.addItem(item)
        self.col_list.itemSelectionChanged.connect(self._update_summary)
        hl.addWidget(self.col_list, 1)
        layout.addLayout(hl)

        self.table = QTableView()
        layout.addWidget(self.table, 1)

        # Buttons row
        btn_row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self._copy_clipboard)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.clicked.connect(self._export_csv)
        btn_row.addStretch()
        btn_row.addWidget(self.copy_btn)
        btn_row.addWidget(self.export_btn)
        layout.addLayout(btn_row)

        self._update_summary()

    # --------------------------------------------------------------
    def _copy_clipboard(self):
        if getattr(self, "_summary_df", None) is None:
            return
        QGuiApplication.clipboard().setText(self._summary_df.to_csv(index=False))

    def _export_csv(self):
        if getattr(self, "_summary_df", None) is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "summary.csv", "CSV Files (*.csv)")
        if path:
            try:
                self._summary_df.to_csv(path, index=False)
            except Exception as exc:
                QMessageBox.critical(self, "Export CSV", str(exc))

    def _selected_cols(self):
        return [i.text() for i in self.col_list.selectedItems()]

    def _update_summary(self):
        cols = self._selected_cols()
        if not cols:
            self.table.setModel(None)
            return
        df_sel = self._df[cols]
        desc = df_sel.describe().transpose()
        desc["missing"] = df_sel.isna().sum()
        try:
            desc["skew"] = df_sel.skew()
            desc["kurt"] = df_sel.kurt()
        except Exception:
            pass
        summary = desc.reset_index().rename(columns={'index':'Variable'})
        from models.dataframe_model import DataFrameModel
        self._summary_df = summary  # cache for export
        self.table.setModel(DataFrameModel(summary))
        self.table.horizontalHeader().setStretchLastSection(True)

# ---------------------------------------------------------------------
class DistributionTab(QDialog):
    """Tab for visualising distributions (Histogram or Boxplot)."""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df
        self.setWindowTitle("Distributions")
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Plot type:"))
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(["Histogram", "Boxplot"])
        self.plot_combo.currentIndexChanged.connect(self._update_plot)
        ctrl.addWidget(self.plot_combo)
        ctrl.addSpacing(20)
        ctrl.addWidget(QLabel("Columns:"))
        self.col_list = QListWidget()
        self.col_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for col in _numeric_cols(df):
            item = QListWidgetItem(str(col))
            item.setSelected(True)
            self.col_list.addItem(item)
        self.col_list.itemSelectionChanged.connect(self._update_plot)
        ctrl.addWidget(self.col_list, 1)
        layout.addLayout(ctrl)

        self.fig = Figure(figsize=(5,4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)

        self._update_plot()

    # ------------------------------------------------------------------
    def _selected_cols(self):
        return [i.text() for i in self.col_list.selectedItems()]

    def _update_plot(self):
        cols = self._selected_cols()
        plot_type = self.plot_combo.currentText()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if not cols:
            ax.text(0.5, 0.5, "Select columns", ha="center", va="center")
        else:
            data = self._df[cols].dropna()
            if plot_type == "Histogram":
                for col in cols:
                    ax.hist(data[col].dropna(), bins=30, alpha=0.6, label=col, edgecolor='black')
                ax.legend()
            else:  # Boxplot
                data.boxplot(ax=ax)
                ax.set_xticklabels(cols, rotation=45, ha="right")
        self.canvas.draw_idle()

# ---------------------------------------------------------------------
class CorrelationTab(QDialog):
    """Tab content for correlation matrix analysis."""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df
        self.setWindowTitle("Correlation Matrix")
        layout = QVBoxLayout(self)

        # Column selector
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Numeric columns:"))
        self.col_list = QListWidget()
        self.col_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for col in _numeric_cols(df):
            item = QListWidgetItem(str(col))
            item.setSelected(True)
            self.col_list.addItem(item)
        self.col_list.itemSelectionChanged.connect(self._update_plot)
        hl.addWidget(self.col_list, 1)
        layout.addLayout(hl)

        # Figure
        self.fig = Figure(figsize=(5,4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)

        self._update_plot()

    def _selected(self) -> List[str]:
        return [i.text() for i in self.col_list.selectedItems()]

    def _update_plot(self):
        cols = self._selected()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if len(cols) < 2:
            ax.text(0.5, 0.5, "Select >=2 columns", ha="center", va="center")
        else:
            corr = self._df[cols].corr()
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cols)), labels=cols, rotation=45, ha="right")
            ax.set_yticks(range(len(cols)), labels=cols)
            self.fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
        self.canvas.draw_idle()

# ---------------------------------------------------------------------
class RegressionTab(QDialog):
    """Tab content for multiple linear regression."""

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df
        self.setWindowTitle("Multiple Linear Regression")
        layout = QVBoxLayout(self)

        # Variable selectors
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Dependent (Y):"))
        self.y_combo = QComboBox()
        self.y_combo.addItems(_numeric_cols(df))
        hl.addWidget(self.y_combo)

        hl.addWidget(QLabel("Independent (X):"))
        self.x_list = QListWidget()
        self.x_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for col in _numeric_cols(df):
            item = QListWidgetItem(str(col))
            if col != self.y_combo.currentText():
                item.setSelected(True)
            self.x_list.addItem(item)
        hl.addWidget(self.x_list, 1)
        layout.addLayout(hl)

        # Run button
        run_btn = QPushButton("Run Regression")
        run_btn.clicked.connect(self._run)
        layout.addWidget(run_btn)

        # Results text
        self.result_edit = QTextEdit()
        self.result_edit.setReadOnly(True)
        self.result_edit.setFixedHeight(250)
        layout.addWidget(self.result_edit)

    def _run(self):
        y = self.y_combo.currentText()
        xs = [i.text() for i in self.x_list.selectedItems()]
        if y in xs:
            xs.remove(y)
        if not xs:
            QMessageBox.warning(self, "Regression", "Select at least one independent variable.")
            return
        df = self._df[[y]+xs].dropna()
        y_vals = df[y].values
        X = df[xs].values
        X = np.column_stack([np.ones(len(X)), X])  # add intercept
        try:
            coeffs, *rest = np.linalg.lstsq(X, y_vals, rcond=None)
            y_pred = X.dot(coeffs)
            residuals = y_vals - y_pred
            ssr = np.sum(residuals**2)
            sst = np.sum((y_vals - y_vals.mean())**2)
            r2 = 1 - ssr / sst if sst != 0 else float('nan')
            lines = [
                "Coefficients:",
                "Intercept: {:.4f}".format(coeffs[0]),
            ]
            for name, coef in zip(xs, coeffs[1:]):
                lines.append(f"{name}: {coef:.4f}")
            lines.append("")
            lines.append(f"R²: {r2:.4f}")
            self.result_edit.setText("\n".join(lines))
        except Exception as exc:
            QMessageBox.critical(self, "Regression", str(exc))

# ---------------------------------------------------------------------
class HypothesisTab(QDialog):
    """Tab for basic hypothesis tests (two-sample t-test & correlation significance)."""
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df
        self.setWindowTitle("Hypothesis Testing")
        layout = QVBoxLayout(self)

        # Test selector row
        top = QHBoxLayout()
        top.addWidget(QLabel("Test:"))
        self.test_combo = QComboBox()
        self.test_combo.addItems(["Two-sample t-test", "Correlation p-value"])
        self.test_combo.currentIndexChanged.connect(self._refresh_ui)
        top.addWidget(self.test_combo)
        layout.addLayout(top)

        # Variable selectors
        var_row = QHBoxLayout()
        var_row.addWidget(QLabel("Variable 1:"))
        self.var1_combo = QComboBox(); self.var1_combo.addItems(_numeric_cols(df))
        var_row.addWidget(self.var1_combo)
        var_row.addWidget(QLabel("Variable 2:"))
        self.var2_combo = QComboBox(); self.var2_combo.addItems(_numeric_cols(df))
        var_row.addWidget(self.var2_combo)
        layout.addLayout(var_row)

        # Run button
        run_btn = QPushButton("Run Test")
        run_btn.clicked.connect(self._run_test)
        layout.addWidget(run_btn)

        # Result
        self.result_edit = QTextEdit(); self.result_edit.setReadOnly(True)
        layout.addWidget(self.result_edit, 1)

        self._refresh_ui()

    def _refresh_ui(self):
        test = self.test_combo.currentText()
        # same UI for now – nothing dynamic needed
        self.result_edit.clear()

    def _run_test(self):
        test = self.test_combo.currentText()
        v1 = self.var1_combo.currentText(); v2 = self.var2_combo.currentText()
        if v1 == v2:
            QMessageBox.warning(self, "Hypothesis", "Choose two different variables.")
            return
        import warnings
        import math
        data1 = pd.to_numeric(self._df[v1], errors="coerce").dropna()
        data2 = pd.to_numeric(self._df[v2], errors="coerce").dropna()
        if data1.empty or data2.empty:
            QMessageBox.warning(self, "Hypothesis", "Variables must contain numeric data.")
            return
        lines: list[str] = []
        if test == "Two-sample t-test":
            try:
                from scipy import stats  # type: ignore
                tstat, pval = stats.ttest_ind(data1, data2, nan_policy="omit", equal_var=False)
                lines.append(f"Two-sample t-test (Welch):")
                lines.append(f"t = {tstat:.4f}")
                lines.append(f"p-value = {pval:.4g}")
                mean1 = data1.mean(); mean2 = data2.mean()
                lines.append(f"mean({v1}) = {mean1:.4f}")
                lines.append(f"mean({v2}) = {mean2:.4f}")
            except Exception as exc:
                lines.append("scipy not available or error: " + str(exc))
        elif test == "Correlation p-value":
            try:
                from scipy import stats  # type: ignore
                r, pval = stats.pearsonr(data1, data2)
                lines.append(f"Pearson r = {r:.4f}")
                lines.append(f"p-value = {pval:.4g}")
            except Exception as exc:
                # fallback manual correlation & approximate p assuming large N
                r = data1.corr(data2)
                n = len(data1)
                if n > 2 and abs(r) < 1:
                    t = r * math.sqrt((n-2)/(1-r**2))
                    from scipy import stats as _stats  # type: ignore
                    pval = 2*_stats.t.sf(abs(t), df=n-2) if _stats else float('nan')
                    lines.append(f"Pearson r = {r:.4f}")
                    lines.append(f"p-value ≈ {pval:.4g}")
                else:
                    lines.append("Insufficient data for correlation test.")
        self.result_edit.setText("\n".join(lines))

# ---------------------------------------------------------------------
class AnalysisDialog(QDialog):
    """Main dialog housing multiple analysis tabs."""
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Statistical Analysis Tools")
        self.resize(1000, 700)
        tabs = QTabWidget(self)

        tabs.addTab(SummaryTab(df, self), "Summary")
        tabs.addTab(DistributionTab(df, self), "Distribution")

        tabs.addTab(CorrelationTab(df, self), "Correlation")
        tabs.addTab(RegressionTab(df, self), "Regression")
        tabs.addTab(HypothesisTab(df, self), "Hypothesis Test")

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
