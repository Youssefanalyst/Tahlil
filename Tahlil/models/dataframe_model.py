"""Qt Table model backed by pandas DataFrame with formula evaluation."""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Tuple

import pandas as pd
import numpy as np
from analysis.outliers import mask_outliers
# import extra formulas module if present
try:
    from . import formulas_extra
except ImportError:
    formulas_extra = None
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QBrush, QColor, QFont


class DataFrameModel(QAbstractTableModel):
    """Spreadsheet table model supporting simple formulas.

    Features:
    • Editable cells backed by `pandas.DataFrame`
    • Column headers as Excel-style letters, rows start at 1
    • Formula engine supporting:
        – Arithmetic + - * /
        – Cell references (A1, B3)
        – Ranges (A1:B3)
        – Functions: SUM, AVERAGE, MIN, MAX, COUNT, COUNTA, PRODUCT, ABS, SQRT, POW,
          MOD, ROUND, CEILING, FLOOR, SIN, COS, TAN, LOG, EXP, IF,
          AND, OR, NOT, LEN, CONCAT, LEFT, RIGHT, UPPER, LOWER, DATE,
          MEDIAN, MODE, RAND, RANDBETWEEN, TRIM, SUBSTITUTE, FIND, REPLACE, VALUE, REPT,
          VAR, STDEV, DEGREES, RADIANS, SIGN, PI, EVEN, ODD, INT, FACT,
          ROUNDUP, ROUNDDOWN, TODAY, NOW, YEAR, MONTH, DAY, IFERROR, COUNTBLANK, UNIQUE,
          SUMIF, COUNTIF, AVERAGEIF, MID, PROPER, SMALL, LARGE, PERCENTILE, SUMPRODUCT, CHOOSE,
          ASIN, ACOS, ATAN, ATAN2, SINH, COSH, TANH, SEC, COT, CSC
    """

    # cell reference like A1, AB12 but NOT followed by '(', so it won't match ATAN2(
    # ensure cell ref not part of longer token (no trailing letter/digit) and not before '('
    _re_cell = re.compile(r"([A-Z]+)(\d+)(?![A-Z0-9])(?!(?=\())")
    _re_range = re.compile(r"([A-Z]+\d+):([A-Z]+\d+)")

    # Public list of supported function names for GUI menus
    SUPPORTED_FUNCTIONS = [
    "SUM", "AVERAGE", "MIN", "MAX", "COUNT", "COUNTA", "PRODUCT", "ABS", "SQRT", "POW",
    "MOD", "ROUND", "CEILING", "FLOOR", "SIN", "COS", "TAN", "LOG", "EXP", "IF",
    "AND", "OR", "NOT", "LEN", "CONCAT", "LEFT", "RIGHT", "UPPER", "LOWER", "DATE",
    "MEDIAN", "MODE", "RAND", "RANDBETWEEN", "TRIM", "SUBSTITUTE", "FIND", "REPLACE", "VALUE", "REPT",
    "VAR", "STDEV", "DEGREES", "RADIANS", "SIGN", "PI", "EVEN", "ODD", "INT", "FACT",
    "ROUNDUP", "ROUNDDOWN", "TODAY", "NOW", "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND", "WEEKDAY", "WEEKNUM", "IFERROR", "COUNTBLANK", "UNIQUE",
    "SUMIF", "COUNTIF", "AVERAGEIF", "MID", "PROPER", "SMALL", "LARGE", "PERCENTILE", "SUMPRODUCT", "CHOOSE",
    "ASIN", "ACOS", "ATAN", "ATAN2", "SINH", "COSH", "TANH", "SEC", "COT", "CSC",
    "XOR", "TEXTJOIN", "SPLIT", "TRANSPOSE", "COUNTUNIQUE", "MAXIFS", "MINIFS", "DAYS", "NETWORKDAYS", "COLUMN",
    "SUMIFS", "COUNTIFS", "AVERAGEIFS", "IFNA", "IFS", "RANK", "QUOTIENT", "TRUNC", "LOG10", "LN", "SLN", "DDB", "SYD", "COGS", "GPM", "NPM", "ROA", "ROE", "EPS", "WORKINGCAP",
# newly added
"LOG2", "SQUARE", "CUBE", "GEOMEAN", "HARMEAN", "SECH", "CSCH", "COTH", "ASINH", "ACOSH", "ATANH", "CLAMP", "ABSMAX", "HYPOT", "SQRTPI",
# extra batch
"LOG1P", "EXPM1", "GCD", "LCM", "BITAND", "BITOR", "BITXOR", "BITNOT", "FACTDOUBLE", "COMBIN", "PERMUT", "ROUNDBANKER", "SIGNIF", "CLAMP01", "SINC",
# data analysis batch
"COVAR", "CORREL", "SKEW", "KURT", "ZSCORE", "PERCENTRANK", "MAD", "IQR", "VARP", "STDEVP", "NORMALIZE", "STANDARDIZE", "SLOPE", "INTERCEPT", "RSQUARED", "QUARTILE", "PERCENTILE_INC", "PERCENTILE_EXC", "TTEST", "MODEMULT", "CUMSUM", "CUMPROD", "DIFF", "LAG", "LEAD", "MOVAVG", "MOVING_AVG", "EMAVG", "ROLLINGSUM", "ROLLINGMAX", "ROLLINGMIN", "ROLLINGMEAN", "PCTCHANGE", "STDERR", "CVAR", "SHARPE", "BETA", "ALPHA", "SLOPELINE", "RESID", "DURBINWATSON", "SORTASC", "SORTDESC", "REVERSE", "UNIQUECNT", "FREQUENCY", "REGEX", "VLOOKUP", "HLOOKUP", "INDEX", "MATCH", "XLOOKUP", "ISNUMBER", "ISTEXT", "ISBLANK", "ISEVEN", "ISODD",
    ]
    # extend with any UPPERCASE callables defined in formulas_extra
    if 'formulas_extra' in globals() and formulas_extra:
        SUPPORTED_FUNCTIONS.extend([name for name in dir(formulas_extra)
                                    if name.isupper() and callable(getattr(formulas_extra, name))])
    import formulas as _forms
    for _n in _forms.__all__:
        if _n not in SUPPORTED_FUNCTIONS:
            SUPPORTED_FUNCTIONS.append(_n)

    # ------------------------------------------------------------------
    # Construction & basic overrides
    # ------------------------------------------------------------------
    def __init__(self, dataframe: pd.DataFrame | None = None):
        super().__init__()
        if dataframe is None:
            dataframe = pd.DataFrame([["" for _ in range(10)] for _ in range(100)])
        self._df = dataframe
        self._history: List[pd.DataFrame] = []
        self._future: list[pd.DataFrame] = []
        # conditional formatting rules: list of dicts
        # rule: {'range': (row_start,row_end,col_start,col_end), 'op': '>', 'value':float, 'bg':QColor}
        self._cf_rules: list[dict] = []
        # data validation rules: list of dicts
        # rule: {'range':(r1,r2,c1,c2), 'type':'number'|'list', 'op':str, 'value':any, 'value2':any, 'list':list, 'mode':'warn'|'block'}
        self._validation_rules: list[dict] = []
        # per-cell validation error messages (for warn mode)
        self._cell_validation_errors: dict[tuple[int,int], str] = {}
        # per-cell colors (e.g., outliers)
        self._cell_colors: dict[tuple[int,int], QColor] = {}
        # per-cell font styles
        self._cell_fonts: dict[tuple[int,int], QFont] = {}
        # cache for evaluated formulas to avoid recomputation within a paint cycle
        self._eval_cache: dict[tuple[int,int], object] = {}
        self._last_evaluated_value = None  # optional debug/reference
        # column headers from dataframe
        self._col_headers: List[str] = [
            str(c) if str(c).strip() and not str(c).isdigit() and not str(c).startswith("Unnamed") else self._num_to_col_letter(i)
            for i, c in enumerate(self._df.columns)
        ]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        return len(self._df.index)

    # ------------------------------------------------------------------
    # Column insertion
    # ------------------------------------------------------------------
    def add_columns(self, count: int = 1):
        """Append `count` empty columns to the right."""
        if count <= 0:
            return
        # save state for undo
        self._history.append(self._df.copy(deep=True))
        self._future.clear()

        self.beginResetModel()
        start_idx = self.columnCount()
        for i in range(count):
            col_idx = start_idx + i
            col_letter = self._num_to_col_letter(col_idx)
            self._df[col_letter] = ""
            self._col_headers.append(col_letter)
        self.endResetModel()

    # ------------------------------------------------------------------
    # Row insertion
    # ------------------------------------------------------------------
    def add_rows(self, count: int = 1):
        """Append `count` empty rows to the bottom."""
        if count <= 0:
            return
        self._history.append(self._df.copy(deep=True))
        self._future.clear()

        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount() + count - 1)
        for _ in range(count):
            self._df.loc[len(self._df.index)] = ["" for _ in range(self.columnCount())]
        self.endInsertRows()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if not index.isValid():
            return None
        r = index.row(); c = index.column()
        raw = self._df.iat[r, c]
        # evaluate formula only if needed
        if isinstance(raw, str) and raw.startswith("="):
            key = (r, c)
            if key in self._eval_cache:
                eval_val = self._eval_cache[key]
            else:
                eval_val = self._internal_compute_value(r, c, set())
                self._eval_cache[key] = eval_val
                self._last_evaluated_value = eval_val
        else:
            eval_val = raw
        if role == Qt.ItemDataRole.EditRole:
            return str(raw)
        # Conditional formatting background has priority over other decorations
        if role == Qt.ItemDataRole.BackgroundRole:
            # 1) explicit cell colours (e.g., outliers) override all
            if (r, c) in self._cell_colors:
                return QBrush(self._cell_colors[(r, c)])
            # 2) conditional-formatting rules
            for rule in self._cf_rules:
                rs, re, cs, ce = rule['range']
                if rs <= r <= re and cs <= c <= ce:
                    cell_val = eval_val
                    try:
                        num = float(cell_val)
                    except Exception:
                        continue
                    op = rule['op']
                    val = rule['value']
                    cond = (
                        (op == '>') and (num > val) or
                        (op == '<') and (num < val) or
                        (op == '>=') and (num >= val) or
                        (op == '<=') and (num <= val) or
                        (op == '=') and (num == val) or
                        (op == '<>') and (num != val)
                    )
                    if cond:
                        return QBrush(rule['bg'])
            # 3) highlight hard errors
            if isinstance(eval_val, str) and eval_val.startswith("#"):
                return QBrush(QColor("#ffcccc"))  # light red
            # 4) validation errors
            if (r, c) in self._cell_validation_errors:
                return QBrush(QColor("#ffe0b2"))  # light orange
            # fallthrough
        if role == Qt.ItemDataRole.ToolTipRole:
            if (r, c) in self._cell_validation_errors:
                return self._cell_validation_errors[(r, c)]

            if isinstance(eval_val, str) and eval_val.startswith("#"):
                return f"Error in formula: {eval_val}. Check arguments or ranges."
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if eval_val is None:
                return ""
            if isinstance(eval_val, str):
                return eval_val
            try:
                import numpy as _np
                if isinstance(eval_val, _np.ndarray):
                    return str(eval_val.tolist())
            except Exception:
                pass
            if isinstance(eval_val, (list, tuple, dict, set)):
                return str(eval_val)
            return str(eval_val)
        return None

    def _clear_eval_cache(self, rows: slice | None = None, cols: slice | None = None):
        """Invalidate cached evaluations for affected cells."""
        if rows is None and cols is None:
            self._eval_cache.clear()
            return
        to_delete = []
        for (r, c) in self._eval_cache:
            if (rows is None or rows.start <= r <= rows.stop) and (cols is None or cols.start <= c <= cols.stop):
                to_delete.append((r, c))
        for key in to_delete:
            self._eval_cache.pop(key, None)

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole):  # type: ignore[override]
        if role == Qt.ItemDataRole.EditRole:
            # validation first
            is_ok, msg, mode = self._check_validation(index.row(), index.column(), value)
            if not is_ok and mode == 'block':
                return False  # reject edit

            # save snapshot for undo
            self._history.append(self._df.copy(deep=True))
            self._clear_eval_cache(rows=slice(index.row(), index.row()), cols=slice(index.column(), index.column()))
            self._future.clear()
            col_dtype = self._df.dtypes.iloc[index.column()]
            new_val = value
            try:
                if pd.api.types.is_numeric_dtype(col_dtype):
                    new_val = float(value) if value != "" else np.nan
            except Exception:
                pass
            self._df.iat[index.row(), index.column()] = new_val
            # update validation error store
            cell_key = (index.row(), index.column())
            if is_ok:
                self._cell_validation_errors.pop(cell_key, None)
            else:  # warn mode
                if msg:
                    self._cell_validation_errors[cell_key] = msg
            # emit change for whole model to refresh dependent formulas
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))
            return True
        return False

    def flags(self, index: QModelIndex):  # type: ignore[override]
        return (
            Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsEditable
        )

    # --------------------------------------------------------------
    # Conditional formatting API
    # --------------------------------------------------------------
    def clear_cf_rules(self) -> None:
        """Remove all conditional-formatting rules and refresh view."""
        if not self._cf_rules:
            return
        self._cf_rules.clear()
        self._clear_eval_cache()
        # trigger repaint for whole model
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1), [Qt.ItemDataRole.BackgroundRole])

    # --------------------------------------------------------------
    # Data validation API
    # --------------------------------------------------------------
    def add_validation_rule(
        self,
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
        rule_type: str,
        op: str | None,
        value: str | float | None = None,
        value2: str | float | None = None,
        list_vals: list[str] | None = None,
        mode: str = 'warn',
    ) -> None:
        if rule_type not in {'number', 'list'}:
            raise ValueError('Unsupported validation rule type')
        if mode not in {'warn', 'block'}:
            mode = 'warn'
        self._validation_rules.append({
            'range': (start_row, end_row, start_col, end_col),
            'type': rule_type,
            'op': op,
            'value': value,
            'value2': value2,
            'list': list_vals,
            'mode': mode,
        })
        # reset errors and repaint
        self._cell_validation_errors.clear()
        self.dataChanged.emit(self.index(start_row, start_col), self.index(end_row, end_col), [Qt.ItemDataRole.BackgroundRole])

    def clear_validation_rules(self):
        if not self._validation_rules:
            return
        self._validation_rules.clear()
        self._cell_validation_errors.clear()
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount()-1, self.columnCount()-1), [Qt.ItemDataRole.BackgroundRole])

    def _check_validation(self, row: int, col: int, val) -> tuple[bool, str | None, str]:
        """Return (is_valid, message, mode) for first failing rule covering cell."""
        for rule in self._validation_rules:
            r1, r2, c1, c2 = rule['range']
            if not (r1 <= row <= r2 and c1 <= col <= c2):
                continue
            mode = rule.get('mode', 'warn')
            if rule['type'] == 'number':
                try:
                    num = float(val)
                except Exception:
                    return False, 'Value is not numeric', mode
                op = rule.get('op')
                v = rule.get('value'); v2 = rule.get('value2')
                cond_map = {
                    '>': num > v,
                    '<': num < v,
                    '>=': num >= v,
                    '<=': num <= v,
                    '=': num == v,
                    '<>': num != v,
                    'between': v <= num <= v2 if (v is not None and v2 is not None) else False,
                }
                if not cond_map.get(op, True):
                    return False, f'Number must satisfy {op} {v}', mode
            elif rule['type'] == 'list':
                allowed = rule.get('list', []) or []
                if str(val) not in allowed:
                    return False, f'Value not in allowed list', mode
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(self.rowCount() - 1, self.columnCount() - 1),
            [Qt.ItemDataRole.BackgroundRole],
        )


    def mark_outliers(self, rows: list[int], cols: list[int], method: str = "z") -> int:
        """Detect outliers in given rows/cols and mark them.

        Parameters
        ----------
        rows, cols : list[int]
            Row and column indices to inspect.
        method : {'z','iqr','knn'}
             Detection algorithm.
             'knn' uses k-nearest-neighbor distance (k=5) thresholded at mean+3·std of neighbor distance.

        Returns
        -------
        int
            Number of outlier cells marked.
        """
        self.clear_cell_colors()
        count = 0
        for c in cols:
            series = self._df.iloc[rows, c]
            mask = mask_outliers(series, method)
            for pos_in_sel in mask[mask].index:
                r = rows[pos_in_sel]
                self._cell_colors[(r, c)] = QColor("#ffab91")
                count += 1
        if count:
            self.dataChanged.emit(
                self.index(min(rows), min(cols)),
                self.index(max(rows), max(cols)),
                [Qt.ItemDataRole.BackgroundRole],
            )
        return count

    def add_cf_rule(
        self,
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
        op: str,
        value: float,
        bg_color: QColor,
    ) -> None:
        """Add a conditional-formatting rule.

        Parameters
        ----------
        start_row, end_row : int
            Inclusive row bounds.
        start_col, end_col : int
            Inclusive column bounds.
        op : str
            One of >, <, >=, <=, =, <>.
        value : float
            Numeric threshold to compare against.
        bg_color : QColor
            Background colour when condition is met.
        """
        if op not in {">", "<", ">=", "<=", "=", "<>"}:
            raise ValueError("Unsupported operator for conditional formatting")
        self._cf_rules.append(
            {
                "range": (start_row, end_row, start_col, end_col),
                "op": op,
                "value": value,
                "bg": bg_color,
            }
        )
        # notify view to repaint those cells
        self.dataChanged.emit(
            self.index(start_row, start_col),
            self.index(end_row, end_col),
            [Qt.ItemDataRole.BackgroundRole],
        )

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------
    def undo(self):
        if not self._history:
            return
        self._future.append(self._df.copy(deep=True))
        self._df = self._history.pop()
        self.layoutChanged.emit()

    def redo(self):
        if not self._future:
            return
        self._history.append(self._df.copy(deep=True))
        self._df = self._future.pop()
        self.layoutChanged.emit()

        # ------------------------------------------------------------------
    # External engine delegate
    # ------------------------------------------------------------------
    def _compute_value(self, row: int, col: int, visiting: set):
        """Delegate to formulas.engine.compute_value for evaluation."""
        from formulas.engine import compute_value
        return compute_value(self, row, col, visiting)

    # ------------------------------------------------------------------
    # Column header rename
    # ------------------------------------------------------------------
    def rename_column(self, index: int, name: str):
        if 0 <= index < len(self._col_headers):
            self._col_headers[index] = name
            # also update underlying DataFrame column label
            try:
                self._df.columns = [name if i==index else c for i,c in enumerate(self._df.columns)]
            except Exception:
                pass
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, index, index)

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------
    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            try:
                return self._col_headers[section]
            except IndexError:
                return self._num_to_col_letter(section)
        return str(section + 1)

    # ------------------------------------------------------------------
    # Search / Replace across sheet
    # ------------------------------------------------------------------
    def search_replace(self, find: str, replace: str, regex: bool = False, match_case: bool = False):
        """Replace occurrences of `find` with `replace` across entire DataFrame.
        Returns number of replacements made."""
        if not find:
            return 0
        flags = 0 if match_case else re.IGNORECASE
        pattern = re.compile(find, flags) if regex else None
        count = 0
        self._history.append(self._df.copy(deep=True))
        self._future.clear()
        for r in range(self.rowCount()):
            for c in range(self.columnCount()):
                val = self._df.iat[r, c]
                if isinstance(val, str):
                    if regex:
                        new_val, repl = pattern.subn(replace, val)
                    else:
                        if match_case:
                            repl = val.count(find)
                            new_val = val.replace(find, replace)
                        else:
                            repl = len(re.findall(re.escape(find), val, flags))
                            new_val = re.sub(re.escape(find), replace, val, flags=flags)
                    if repl:
                        self._df.iat[r, c] = new_val
                        count += repl
        if count:
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount()-1, self.columnCount()-1))
        return count

    # ------------------------------------------------------------------
    # Helpers – Excel column/row conversions
    # ------------------------------------------------------------------
    @staticmethod
    def _num_to_col_letter(n: int) -> str:
        result = ""
        while True:
            n, rem = divmod(n, 26)
            result = chr(65 + rem) + result
            if n == 0:
                break
            n -= 1
        return result

    @staticmethod
    def _col_letter_to_num(letters: str) -> int:
        num = 0
        for ch in letters:
            num = num * 26 + (ord(ch) - 64)
        return num - 1

    # ------------------------------------------------------------------
    # Formula engine
    # ------------------------------------------------------------------
    def _internal_compute_value(self, row: int, col: int, visiting: set):
        key = (row, col)
        if key in visiting:
            return "#CYCLE"
        visiting.add(key)
        if row>=self.rowCount() or col>=self.columnCount():
            visiting.discard(key)
            return "#REF"
        raw = self._df.iat[row, col]
        if isinstance(raw, str) and raw.startswith("="):
            expr = raw[1:]

            # 1) replace ranges (e.g. A1:B3)
            def repl_range(m: re.Match[str]):
                start, end = m.groups()
                cells = self._expand_range(start, end)
                vals = [self._internal_compute_value(r, c, visiting) for r, c in cells]
                return str(vals)

            expr = self._re_range.sub(repl_range, expr)

            # 2) replace single cell refs
            def repl_cell(m: re.Match[str]):
                col_letters, row_num = m.groups()
                r = int(row_num) - 1
                c = self._col_letter_to_num(col_letters.upper())
                if r<0 or c<0 or r>=self.rowCount() or c>=self.columnCount():
                    return "'#REF'"
                val = self._internal_compute_value(r, c, visiting)
                if val == "":
                    return "0"
                return str(val)

            safe_expr = self._re_cell.sub(repl_cell, expr)

            # Accounting functions
            def _to_num(x):
                try:
                    return float(x)
                except Exception:
                    raise ValueError

            def SLN(cost, salvage, life):
                try:
                    cost=_to_num(cost); salvage=_to_num(salvage); life=_to_num(life)
                    if life==0:
                        return "#ERR"
                    return (cost-salvage)/life
                except Exception:
                    return "#ERR"

            def DDB(cost, salvage, life, period):
                try:
                    cost=_to_num(cost); salvage=_to_num(salvage); life=_to_num(life); period=_to_num(period)
                    rate=2/life
                    acc=0.0
                    for p in range(1,int(period)):
                        dep=min((cost-acc-salvage)*rate, cost-acc-salvage)
                        acc+=dep
                    dep=min((cost-acc-salvage)*rate, cost-acc-salvage)
                    return dep if dep>=0 else "#ERR"
                except Exception:
                    return "#ERR"

            def SYD(cost, salvage, life, period):
                try:
                    cost=_to_num(cost); salvage=_to_num(salvage); life=int(_to_num(life)); period=int(_to_num(period))
                    if period<1 or period>life:
                        return "#ERR"
                    syd=sum(range(1,life+1))
                    return (cost-salvage)*(life-period+1)/syd
                except Exception:
                    return "#ERR"

            def COGS(units_sold, unit_cost):
                try:
                    return _to_num(units_sold)*_to_num(unit_cost)
                except Exception:
                    return "#ERR"

            def GPM(revenue, cogs):
                try:
                    revenue=_to_num(revenue); cogs=_to_num(cogs)
                    if revenue==0:
                        return "#ERR"
                    return (revenue-cogs)/revenue
                except Exception:
                    return "#ERR"

            def NPM(revenue, net_income):
                try:
                    revenue=_to_num(revenue); net_income=_to_num(net_income)
                    if revenue==0:
                        return "#ERR"
                    return net_income/revenue
                except Exception:
                    return "#ERR"

            def ROA(net_income, total_assets):
                try:
                    return _to_num(net_income)/_to_num(total_assets)
                except Exception:
                    return "#ERR"

            def ROE(net_income, equity):
                try:
                    return _to_num(net_income)/_to_num(equity)
                except Exception:
                    return "#ERR"

            def EPS(net_income, shares):
                try:
                    return _to_num(net_income)/_to_num(shares)
                except Exception:
                    return "#ERR"

            def WORKINGCAP(current_assets, current_liabilities):
                try:
                    return _to_num(current_assets)-_to_num(current_liabilities)
                except Exception:
                    return "#ERR"

            # Lookup & reference functions
            def INDEX(table, row_idx, col_idx):
                try:
                    r = int(_to_num(row_idx)) - 1
                    c = int(_to_num(col_idx)) - 1
                    if isinstance(table, list):
                        return table[r][c]
                    import numpy as _np
                    if isinstance(table, _np.ndarray):
                        return table[r][c]
                    return "#ERR"
                except Exception:
                    return "#ERR"

            def _to_list(tbl):
                if isinstance(tbl, list):
                    return tbl
                import numpy as _np
                if isinstance(tbl, _np.ndarray):
                    return tbl.tolist()
                return [[tbl]]

            def VLOOKUP(val, table, col_idx, approximate=0):
                tbl=_to_list(table)
                try:
                    col=int(_to_num(col_idx))-1
                    for row in tbl:
                        if len(row)>col and row[0]==val:
                            return row[col]
                    return "#N/A" if not approximate else row[col]
                except Exception:
                    return "#ERR"

            def HLOOKUP(val, table, row_idx, approximate=0):
                tbl=_to_list(table)
                try:
                    row=int(_to_num(row_idx))-1
                    if row>=len(tbl):
                        return "#N/A"
                    hdr=tbl[0]
                    for idx, h in enumerate(hdr):
                        if h==val:
                            return tbl[row][idx]
                    return "#N/A"
                except Exception:
                    return "#ERR"

            def MATCH(val, vector, approximate=0):
                vec=_to_list(vector)
                try:
                    for idx, v in enumerate(vec):
                        if v==val:
                            return idx+1
                    return "#N/A"
                except Exception:
                    return "#ERR"

            def XLOOKUP(val, lookup_vec, return_vec, if_not_found="#N/A"):
                try:
                    lv=_to_list(lookup_vec)
                    rv=_to_list(return_vec)
                    for idx, v in enumerate(lv):
                        if v==val:
                            return rv[idx]
                    return if_not_found
                except Exception:
                    return "#ERR"

            # Information functions
            def ISNUMBER(value):
                try:
                    float(value)
                    return True
                except Exception:
                    return False

            def ISTEXT(value):
                return isinstance(value, str)

            def ISBLANK(value):
                return value in ("", None, np.nan)

            def ISEVEN(value):
                try:
                    return int(float(value)) % 2 == 0
                except Exception:
                    return False

            def ISODD(value):
                try:
                    return int(float(value)) % 2 == 1
                except Exception:
                    return False

            # allowed helpers
            def _flatten(args: Iterable):
                for a in args:
                    if isinstance(a, (list, tuple)):
                        yield from _flatten(a)
                    else:
                        yield a

            def _numeric(iterable: Iterable):
                for v in iterable:
                    try:
                        yield _to_num(v)
                    except Exception:
                        # ignore non-numeric
                        continue

            def SUM(*args):
                return sum(_numeric(_flatten(args)))

            def AVERAGE(*args):
                nums=list(_numeric(_flatten(args)))
                return sum(nums)/len(nums) if nums else 0

            def MIN(*args):
                nums=list(_numeric(_flatten(args)))
                return min(nums) if nums else 0

            def MAX(*args):
                nums=list(_numeric(_flatten(args)))
                return max(nums) if nums else 0

            def COUNT(*args):
                return sum(1 for _ in _numeric(_flatten(args)))

            def COUNTA(*args):
                return sum(1 for x in _flatten(args) if x not in ("", None))

            def PRODUCT(*args):
                p = 1
                for x in _flatten(args):
                    p *= x
                return p

            def SQRT(x):
                return math.sqrt(x)

            def POW(x, y):
                return math.pow(x, y)

            # New formulas
            def MOD(a, b):
                return a % b

            def ROUND(x, n: int = 0):
                return round(x, int(n))

            def CEILING(x):
                return math.ceil(x)

            def FLOOR(x):
                return math.floor(x)

            def SIN(x):
                return math.sin(x)

            def COS(x):
                return math.cos(x)

            def TAN(x):
                return math.tan(x)

            def LOG(x, base=math.e):
                return math.log(x, base)

            def EXP(x):
                return math.exp(x)

            def IF(condition, true_val, false_val):
                return true_val if condition else false_val

            # Additional functions
            def AND_(*args):
                return all(bool(x) for x in _flatten(args))
            def OR_(*args):
                return any(bool(x) for x in _flatten(args))
            def NOT_(x):
                return not bool(x)
            def LEN(x):
                return len(str(x))
            def CONCAT(*args):
                return "".join(str(a) for a in args)
            def LEFT(text, n: int):
                return str(text)[: int(n)]
            def RIGHT(text, n: int):
                return str(text)[-int(n):]
            def UPPER(text):
                return str(text).upper()
            def LOWER(text):
                return str(text).lower()
            from datetime import date as _dt
            def DATE(y, m, d):
                return _dt(int(y), int(m), int(d)).isoformat()

            # Further functions
            import random as _rnd
            import statistics as _stat
            def MEDIAN(*args):
                vals=list(_flatten(args))
                return _stat.median(vals)
            def MODE(*args):
                vals=list(_flatten(args))
                try:
                    return _stat.mode(vals)
                except Exception:
                    return "#N/A"
            def RAND():
                return _rnd.random()
            def RANDBETWEEN(a,b):
                return _rnd.randint(int(a), int(b))
            def TRIM(text):
                return str(text).strip()
            def SUBSTITUTE(text, old, new):
                return str(text).replace(str(old), str(new))
            def FIND(sub,text):
                try:
                    return str(text).index(str(sub))+1
                except ValueError:
                    return 0
            def REPLACE(text, start, length, new):
                s=str(text)
                start=int(start)-1
                length=int(length)
                return s[:start]+str(new)+s[start+length:]
            def VALUE(text):
                try:
                    return float(text)
                except Exception:
                    return "#ERR"
            def REPT(text, n):
                return str(text)*int(n)

            # extra functions
            def TEXT(value, fmt):
                """Format *value* using *fmt* string (Excel TEXT)."""
                import datetime as _dtm, numbers as _nums
                try:
                    if isinstance(value, (_dtm.date, _dtm.datetime)):
                        return _dtm.datetime.fromisoformat(str(value)).strftime(str(fmt))
                    if isinstance(value, _nums.Number):
                        return format(float(value), str(fmt))
                    return format(str(value), str(fmt))
                except Exception:
                    return str(value)

            def SEARCH(find_text, within_text, start_num: int = 1):
                """Case-insensitive position (1-based) of *find_text* in *within_text*. Returns 0 if not found."""
                try:
                    pos = str(within_text).lower().find(str(find_text).lower(), int(start_num) - 1)
                    return pos + 1 if pos != -1 else 0
                except Exception:
                    return 0

            def _to_date(val):
                import pandas as _pd
                return _pd.to_datetime(val).date()

            import calendar as _cal
            def EDATE(start_date, months):
                """Return date shifted by *months* months (ISO format)."""
                dt = _to_date(start_date)
                months_total = dt.month - 1 + int(months)
                year = dt.year + months_total // 12
                month = months_total % 12 + 1
                day = min(dt.day, _cal.monthrange(year, month)[1])
                import datetime as _dtmod
                return _dtmod.date(year, month, day).isoformat()

            def EOMONTH(start_date, months=0):
                """End-of-month date after *months* offset."""
                dt = _to_date(start_date)
                months_total = dt.month - 1 + int(months)
                year = dt.year + months_total // 12
                month = months_total % 12 + 1
                last_day = _cal.monthrange(year, month)[1]
                import datetime as _dtmod
                return _dtmod.date(year, month, last_day).isoformat()

            def WORKDAY(start_date, days):
                """Date after *days* workdays (Mon-Fri)."""
                import datetime as _dtm
                dt = _to_date(start_date)
                days = int(days)
                step = 1 if days >= 0 else -1
                remain = abs(days)
                while remain:
                    dt += _dtm.timedelta(days=step)
                    if dt.weekday() < 5:
                        remain -= 1
                return dt.isoformat()
            def VAR(*args):
                vals=[float(v) for v in _flatten(args)]
                return _stat.variance(vals) if len(vals)>=2 else 0
            def STDEV(*args):
                vals=[float(v) for v in _flatten(args)]
                return _stat.stdev(vals) if len(vals)>=2 else 0
            def DEGREES(x):
                return math.degrees(x)
            def RADIANS(x):
                return math.radians(x)
            def SIGN(x):
                x=float(x)
                return (x>0) - (x<0)
            def PI():
                return math.pi
            def EVEN(x):
                x=int(float(x))
                return x if x%2==0 else x+1
            def ODD(x):
                x=int(float(x))
                return x if x%2!=0 else x+1
            def INT(x):
                return int(float(x))
            def FACT(n):
                n=int(n)
                return math.factorial(n)

            # further functions
            def ROUNDUP(x, n=0):
                p=10**int(n)
                return math.ceil(float(x)*p)/p
            def ROUNDDOWN(x, n=0):
                p=10**int(n)
                return math.floor(float(x)*p)/p
            from datetime import date as _d, datetime as _dt
            def TODAY():
                return _d.today().isoformat()
            def NOW():
                return _dt.now().isoformat(timespec='seconds')
            def YEAR(s):
                return _dt.fromisoformat(str(s)).year
            def MONTH(s):
                return _dt.fromisoformat(str(s)).month
            def DAY(s):
                return _dt.fromisoformat(str(s)).day
            def HOUR(s):
                return _dt.fromisoformat(str(s)).hour
            def MINUTE(s):
                return _dt.fromisoformat(str(s)).minute
            def SECOND(s):
                return _dt.fromisoformat(str(s)).second
            import datetime as _dtmod
            def WEEKDAY(s, weekstart:int=1):
                """Return day of week, 1=Sunday if weekstart==1 (Excel style), else 1=Monday."""
                dt=_dtmod.datetime.fromisoformat(str(s))
                wd=dt.weekday()  # Monday=0
                if weekstart==1:
                    return (wd+1)%7+1  # Sun=1, Sat=7
                else:
                    return wd+1  # Mon=1
            def WEEKNUM(s, weekstart:int=1):
                dt=_dtmod.datetime.fromisoformat(str(s))
                return dt.isocalendar().week if weekstart!=1 else int(dt.strftime("%U"))+1
            def IFERROR(value, fallback):
                return fallback if isinstance(value, str) and value.startswith('#') else value
            def COUNTBLANK(*args):
                return sum(1 for v in _flatten(args) if str(v)=='' )
            def UNIQUE(*args):
                return list(dict.fromkeys(_flatten(args)))

            # helper for criteria evaluation
            _ops = {">":lambda a,b: a>b,
                    "<":lambda a,b: a<b,
                    ">=":lambda a,b: a>=b,
                    "<=":lambda a,b: a<=b,
                    "<>":lambda a,b: a!=b,
                    "!=":lambda a,b: a!=b,
                    "=":lambda a,b: a==b}
            def _match(val, crit):
                crit=str(crit)
                for op in (">=","<=",">","<","!=","<>","="):
                    if crit.startswith(op):
                        try:
                            num=float(crit[len(op):])
                            try_val=float(val)
                        except Exception:
                            try_val=str(val)
                            num=str(num)
                        return _ops[op](try_val, num)
                return str(val)==crit

            def SUMIF(range_, criteria, sum_range=None):
                rng=_flatten(range_)
                if sum_range is None:
                    return sum(float(v) for v in rng if _match(v,criteria))
                sums=_flatten(sum_range)
                return sum(float(s) for v,s in zip(rng,sums) if _match(v,criteria))
            def COUNTIF(range_, criteria):
                rng=_flatten(range_)
                return sum(1 for v in rng if _match(v,criteria))
            def AVERAGEIF(range_, criteria, avg_range=None):
                src=_flatten(range_)
                if avg_range is None:
                    vals=[float(v) for v in src if _match(v,criteria)]
                else:
                    vals=[float(a) for v,a in zip(src,_flatten(avg_range)) if _match(v,criteria)]
                return sum(vals)/len(vals) if vals else 0
            def MID(text, start, length):
                return str(text)[int(start)-1:int(start)-1+int(length)]
            def PROPER(text):
                return str(text).title()
            def SMALL(range_, k):
                arr=sorted(float(v) for v in _flatten(range_))
                k=int(k)
                return arr[k-1] if 0<k<=len(arr) else "#ERR"
            def LARGE(range_, k):
                arr=sorted((float(v) for v in _flatten(range_)), reverse=True)
                k=int(k)
                return arr[k-1] if 0<k<=len(arr) else "#ERR"
            def PERCENTILE(range_, p):
                from math import floor, ceil
                arr=sorted(float(v) for v in _flatten(range_))
                if not arr:
                    return "#ERR"
                p=float(p)
                idx=(len(arr)-1)*p
                lo=arr[floor(idx)]
                hi=arr[ceil(idx)]
                return lo if lo==hi else (lo+hi)/2
            def SUMPRODUCT(r1, r2):
                a=list(_flatten(r1))
                b=list(_flatten(r2))
                if len(a)!=len(b):
                    return "#ERR"
                return sum(float(x)*float(y) for x,y in zip(a,b))
            def CHOOSE(index, *options):
                idx=int(index)
                return options[idx-1] if 1<=idx<=len(options) else "#ERR"

            # --- additional functions ---
            def XOR(*args):
                def _to_bool(val):
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, (int, float)):
                        return val != 0
                    if isinstance(val, str):
                        up = val.strip().upper()
                        if up in ("TRUE", "T", "YES", "Y"):
                            return True
                        if up in ("FALSE", "F", "NO", "N", ""):
                            return False
                        # try numeric string
                        try:
                            return float(up) != 0
                        except Exception:
                            return False
                    return bool(val)
                truth_vals = [_to_bool(v) for v in _flatten(args)]
                return sum(truth_vals) % 2 == 1

            # -------- 15 new functions --------
            def LOG2(x):
                return math.log2(_to_num(x))

            def SQRTPI(x):
                return math.sqrt(math.pi * _to_num(x))

            def SQUARE(x):
                v=_to_num(x)
                return v*v

            def CUBE(x):
                v=_to_num(x)
                return v*v*v

            def GEOMEAN(*args):
                vals=[]
                for v in _flatten(args):
                    try:
                        vals.append(_to_num(v))
                    except Exception:
                        continue
                if not vals:
                    return 0
                prod=1.0
                for v in vals:
                    prod*=v
                return prod**(1/len(vals))

            def HARMEAN(*args):
                vals=[]
                for v in _flatten(args):
                    try:
                        num=_to_num(v)
                        if num!=0:
                            vals.append(num)
                    except Exception:
                        continue
                n=len(vals)
                return n/ sum(1/v for v in vals) if vals else 0

            def SECH(x):
                return 1/math.cosh(_to_num(x))

            def CSCH(x):
                v=_to_num(x)
                return 1/math.sinh(v) if v!=0 else math.inf

            def COTH(x):
                v=_to_num(x)
                return math.cosh(v)/math.sinh(v) if v!=0 else math.inf

            def ASINH(x):
                return math.asinh(_to_num(x))

            def ACOSH(x):
                return math.acosh(_to_num(x))

            def ATANH(x):
                return math.atanh(_to_num(x))

            def CLAMP(x, min_val, max_val):
                v=_to_num(x)
                return max(_to_num(min_val), min(v, _to_num(max_val)))

            def ABSMAX(a, b):
                return max(abs(_to_num(a)), abs(_to_num(b)))

            def HYPOT(a, b):
                return math.hypot(_to_num(a), _to_num(b))

            # -------- another 15 functions --------
            def LOG1P(x):
                return math.log1p(_to_num(x))

            def EXPM1(x):
                return math.expm1(_to_num(x))

            def _int(v):
                return int(float(v))

            def GCD(a, b):
                return math.gcd(_int(a), _int(b))

            def LCM(a, b):
                import math as _m
                a=_int(a); b=_int(b)
                return abs(a*b)//_m.gcd(a,b) if a and b else 0

            def BITAND(a, b):
                return _int(a) & _int(b)

            def BITOR(a, b):
                return _int(a) | _int(b)

            def BITXOR(a, b):
                return _int(a) ^ _int(b)

            def BITNOT(a):
                return ~_int(a)

            def FACTDOUBLE(n):
                n=_int(n)
                res=1
                k= n if n%2==n%2 else n-1
                for i in range(k,0,-2):
                    res*=i
                return res

            def COMBIN(n, k):
                from math import comb
                return comb(_int(n), _int(k))

            def PERMUT(n, k):
                n=_int(n); k=_int(k)
                from math import factorial
                if k>n or k<0:
                    return 0
                return factorial(n)//factorial(n-k)

            def ROUNDBANKER(x):
                x=float(x)
                import decimal
                return float(decimal.Decimal(x).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_EVEN))

            def SIGNIF(x, digits=3):
                x=float(x); digits=int(digits)
                if x==0:
                    return 0
                import math
                return round(x, digits - int(math.floor(math.log10(abs(x)))) -1)

            def CLAMP01(x):
                v=float(x)
                return max(0.0, min(1.0, v))

            def SINC(x):
                x=float(x)
                return 1.0 if x==0 else math.sin(x)/x

            # -------- data analysis 20 functions --------
            def _flat_numeric(it):
                for v in _flatten(it):
                    try:
                        yield float(v)
                    except Exception:
                        continue

            def COVAR(r1, r2):
                a=list(_flat_numeric(r1)); b=list(_flat_numeric(r2))
                if len(a)!=len(b) or not a:
                    return "#ERR"
                mean_a=sum(a)/len(a); mean_b=sum(b)/len(b)
                return sum((x-mean_a)*(y-mean_b) for x,y in zip(a,b))/len(a)

            def CORREL(r1, r2):
                a=list(_flat_numeric(r1)); b=list(_flat_numeric(r2))
                if len(a)!=len(b) or not a:
                    return "#ERR"
                mean_a=sum(a)/len(a); mean_b=sum(b)/len(b)
                num=sum((x-mean_a)*(y-mean_b) for x,y in zip(a,b))
                denA=math.sqrt(sum((x-mean_a)**2 for x in a)); denB=math.sqrt(sum((y-mean_b)**2 for y in b))
                return num/(denA*denB) if denA and denB else "#ERR"

            def SKEW(range_):
                vals=list(_flat_numeric(range_))
                n=len(vals)
                if n<3:
                    return "#ERR"
                mean=sum(vals)/n
                s=math.sqrt(sum((v-mean)**2 for v in vals)/n)
                if s==0:
                    return 0
                return (sum((v-mean)**3 for v in vals)/n)/(s**3)

            def KURT(range_):
                vals=list(_flat_numeric(range_))
                n=len(vals)
                if n<4:
                    return "#ERR"
                mean=sum(vals)/n
                s2=sum((v-mean)**2 for v in vals)/n
                if s2==0:
                    return 0
                m4=sum((v-mean)**4 for v in vals)/n
                return m4/(s2**2)-3

            def ZSCORE(x, mean, stdev):
                mean=float(mean); stdev=float(stdev)
                return (float(x)-mean)/stdev if stdev else "#ERR"

            def PERCENTRANK(range_, x):
                vals=sorted(_flat_numeric(range_))
                if not vals:
                    return "#ERR"
                x=float(x)
                count=sum(1 for v in vals if v<=x)
                return (count-1)/(len(vals)-1) if len(vals)>1 else 0

            def MAD(range_):
                vals=list(_flat_numeric(range_))
                if not vals:
                    return 0
                med=sorted(vals)[len(vals)//2]
                return sorted(abs(v-med) for v in vals)[len(vals)//2]

            def IQR(range_):
                vals=sorted(_flat_numeric(range_))
                if not vals:
                    return 0
                q1=vals[int(0.25*(len(vals)-1))]
                q3=vals[int(0.75*(len(vals)-1))]
                return q3-q1

            def VARP(*args):
                vals=list(_flat_numeric(args))
                return _stat.pvariance(vals) if len(vals)>=1 else 0

            def STDEVP(*args):
                vals=list(_flat_numeric(args))
                return _stat.pstdev(vals) if len(vals)>=1 else 0

            def NORMALIZE(x, min_val, max_val):
                x=float(x); min_val=float(min_val); max_val=float(max_val)
                return (x-min_val)/(max_val-min_val) if max_val!=min_val else "#ERR"

            def STANDARDIZE(x, mean, stdev):
                return ZSCORE(x, mean, stdev)

            def SLOPE(ry, rx):
                y=list(_flat_numeric(ry)); x=list(_flat_numeric(rx))
                if len(y)!=len(x) or not y:
                    return "#ERR"
                mean_x=sum(x)/len(x); mean_y=sum(y)/len(y)
                num=sum((xi-mean_x)*(yi-mean_y) for xi,yi in zip(x,y))
                den=sum((xi-mean_x)**2 for xi in x)
                return num/den if den else "#ERR"

            def INTERCEPT(ry, rx):
                slope=SLOPE(ry, rx)
                if slope=="#ERR":
                    return slope
                x=list(_flat_numeric(rx)); y=list(_flat_numeric(ry))
                mean_x=sum(x)/len(x); mean_y=sum(y)/len(y)
                return mean_y - slope*mean_x

            def RSQUARED(ry, rx):
                corr=CORREL(ry, rx)
                return corr*corr if corr!="#ERR" else corr

            def QUARTILE(range_, q):
                q=float(q)
                if q not in (0,1,2,3,4):
                    return "#ERR"
                vals=sorted(_flat_numeric(range_))
                if not vals:
                    return "#ERR"
                idx=int(q*(len(vals)-1)/4)
                return vals[idx]

            def PERCENTILE_INC(range_, p):
                return PERCENTILE(range_, p)

            def PERCENTILE_EXC(range_, p):
                p=float(p)
                if p<=0 or p>=1:
                    return "#ERR"
                vals=sorted(_flat_numeric(range_))
                if not vals:
                    return "#ERR"
                from math import floor, ceil
                idx=p*(len(vals)+1)
                if idx<1:
                    return vals[0]
                if idx>len(vals):
                    return vals[-1]
                lo=vals[floor(idx)-1]; hi=vals[ceil(idx)-1]
                return lo if lo==hi else (lo+hi)/2

            def TTEST(r1, r2):
                a=list(_flat_numeric(r1)); b=list(_flat_numeric(r2))
                import math
                if len(a)<2 or len(b)<2:
                    return "#ERR"
                mean_a=sum(a)/len(a); mean_b=sum(b)/len(b)
                var_a=_stat.pvariance(a); var_b=_stat.pvariance(b)
                n1=len(a); n2=len(b)
                # two-sample t statistic with unequal variances (Welch)
                t=(mean_a-mean_b)/math.sqrt(var_a/n1 + var_b/n2)
                return t

            def MODEMULT(range_):
                vals=list(_flat_numeric(range_))
                from collections import Counter
                if not vals:
                    return []
                c=Counter(vals)
                max_freq=max(c.values())
                return [k for k,v in c.items() if v==max_freq]
            # -------- extra 25 time-series / analysis functions --------
            def _to_list(it):
                return list(_flatten(it))

            def CUMSUM(range_):
                vals=[_to_num(v) for v in _flat_numeric(range_)]
                out=[]; s=0
                for v in vals:
                    s+=v; out.append(s)
                return out

            def CUMPROD(range_):
                vals=[_to_num(v) for v in _flat_numeric(range_)]
                out=[]; p=1
                for v in vals:
                    p*=v; out.append(p)
                return out

            def DIFF(range_, lag:int=1):
                vals=[_to_num(v) for v in _flat_numeric(range_)]
                lag=int(lag)
                return [vals[i]-vals[i-lag] if i>=lag else None for i in range(len(vals))]

            def LAG(range_, n:int=1):
                vals=_to_list(range_); n=int(n)
                return [None]*n + vals[:-n] if n>=0 else vals

            def LEAD(range_, n:int=1):
                vals=_to_list(range_); n=int(n)
                return vals[n:] + [None]*n if n>=0 else vals

            def MOVAVG(range_, window:int=3):
                vals=[_to_num(v) for v in _flat_numeric(range_)]
                w=int(window)
                return [sum(vals[max(0,i-w+1):i+1])/min(i+1,w) for i in range(len(vals))]

            def EMAVG(range_, alpha:float=0.2):
                vals=[_to_num(v) for v in _flat_numeric(range_)]
                alpha=float(alpha)
                ema=[]
                for v in vals:
                    if not ema:
                        ema.append(v)
                    else:
                        ema.append(alpha*v + (1-alpha)*ema[-1])
                return ema

            def ROLLINGSUM(range_, window:int=3):
                vals=[_to_num(v) for v in _flat_numeric(range_)]
                w=int(window)
                out=[]
                for i in range(len(vals)):
                    out.append(sum(vals[max(0,i-w+1):i+1]))
                return out

            def ROLLINGMAX(range_, window:int=3):
                vals=[_to_num(v) for v in _flat_numeric(range_)]
                w=int(window)
                return [max(vals[max(0,i-w+1):i+1]) for i in range(len(vals))]

            def ROLLINGMIN(range_, window:int=3):
                vals=[_to_num(v) for v in _flat_numeric(range_)]
                w=int(window)
                return [min(vals[max(0,i-w+1):i+1]) for i in range(len(vals))]

            def ROLLINGMEAN(range_, window:int=3):
                return MOVAVG(range_, window)

            def PCTCHANGE(range_):
                vals=[_to_num(v) for v in _flat_numeric(range_)]
                return [None]+[(vals[i]-vals[i-1])/vals[i-1] if vals[i-1]!=0 else None for i in range(1,len(vals))]

            def STDERR(*args):
                vals=list(_flat_numeric(args))
                n=len(vals)
                return _stat.stdev(vals)/math.sqrt(n) if n>=2 else 0

            def CVAR(range_):
                vals=list(_flat_numeric(range_))
                mean=sum(vals)/len(vals) if vals else 0
                return (_stat.stdev(vals)/mean) if mean else "#ERR"

            def SHARPE(returns, risk_free=0):
                rets=[_to_num(v) for v in _flat_numeric(returns)]
                mean=sum(rets)/len(rets) if rets else 0
                rf=float(risk_free)
                st=_stat.stdev(rets) if len(rets)>=2 else 0
                return (mean-rf)/st if st else "#ERR"

            def BETA(asset, market):
                return COVAR(asset, market)/VARP(market) if VARP(market)!=0 else "#ERR"

            def ALPHA(asset, market, risk_free=0):
                beta=BETA(asset, market)
                if beta=="#ERR":
                    return beta
                mean_asset=sum(_flat_numeric(asset))/len(_flat_numeric(asset))
                mean_market=sum(_flat_numeric(market))/len(_flat_numeric(market))
                rf=float(risk_free)
                return mean_asset - (rf + beta*(mean_market - rf))

            def SLOPELINE(ry, rx):
                return SLOPE(ry, rx)

            def RESID(ry, rx):
                slope=SLOPE(ry, rx)
                if slope=="#ERR":
                    return []
                intercept=INTERCEPT(ry, rx)
                y=list(_flat_numeric(ry)); x=list(_flat_numeric(rx))
                return [yi - (intercept + slope*xi) for xi,yi in zip(x,y)]

            def DURBINWATSON(resid):
                r=list(_flat_numeric(resid))
                num=sum((r[i]-r[i-1])**2 for i in range(1,len(r)))
                den=sum(v*v for v in r)
                return num/den if den else "#ERR"

            def SORTASC(range_):
                return sorted(_flatten(range_))

            def SORTDESC(range_):
                return sorted(_flatten(range_), reverse=True)

            def REVERSE(range_):
                return list(_flatten(range_))[::-1]

            def UNIQUECNT(range_):
                return len(set(_flatten(range_)))

            def FREQUENCY(range_, bins):
                values=[_to_num(v) for v in _flat_numeric(range_)]
                bins_sorted=sorted([_to_num(b) for b in _flat_numeric(bins)])
                freq=[0]*len(bins_sorted)
                for v in values:
                    for i,b in enumerate(bins_sorted):
                        if v<=b:
                            freq[i]+=1; break
                return freq

            def TEXTJOIN(delim, *args):
                return str(delim).join(str(v) for v in _flatten(args))
            def SPLIT(text, delim):
                return str(text).split(str(delim))
            import re as _re_mod
            def REGEX(text, pattern, group:int=0):
                """Return first regex match group (default 0 entire match). Empty string if no match."""
                m=_re_mod.search(str(text), str(pattern))
                if not m:
                    return ""
                try:
                    return m.group(int(group))
                except IndexError:
                    return ""
            def MOVING_AVG(range_, window:int=3):
                return MOVAVG(range_, window)
            def TRANSPOSE(range_):
                arr=[list(_flatten(r)) if isinstance(r,(list,tuple)) else [r] for r in range_]
                return [list(col) for col in zip(*arr)] if arr else []
            def COUNTUNIQUE(range_):
                return len(set(_flatten(range_)))
            def MAXIFS(range_vals, crit_range, criteria):
                rv=list(_flatten(range_vals))
                cr=list(_flatten(crit_range))
                if len(rv)!=len(cr):
                    return "#ERR"
                nums=[float(v) for v,c in zip(rv,cr) if _match(c,criteria)]
                return max(nums) if nums else "#ERR"
            def MINIFS(range_vals, crit_range, criteria):
                rv=list(_flatten(range_vals))
                cr=list(_flatten(crit_range))
                if len(rv)!=len(cr):
                    return "#ERR"
                nums=[float(v) for v,c in zip(rv,cr) if _match(c,criteria)]
                return min(nums) if nums else "#ERR"
            from datetime import datetime, timedelta
            def _to_date(val):
                if isinstance(val, (datetime,)): return val
                try:
                    return datetime.fromisoformat(str(val))
                except Exception:
                    return None
            def DAYS(end_date, start_date):
                ed=_to_date(end_date); sd=_to_date(start_date)
                if ed and sd:
                    return (ed-sd).days
                return "#ERR"
            def NETWORKDAYS(start_date, end_date):
                sd=_to_date(start_date); ed=_to_date(end_date)
                if not sd or not ed:
                    return "#ERR"
                if sd>ed:
                    sd,ed=ed,sd
                days=0
                cur=sd
                while cur<=ed:
                    if cur.weekday()<5:
                        days+=1
                    cur+=timedelta(days=1)
                return days
            def COLUMN(ref):
                if isinstance(ref,str):
                    m=re.match(r"([A-Z]+)",ref.upper())
                    if m:
                        return self._col_letter_to_num(m.group(1))+1
                return "#ERR"

            # ---- new batch of functions ----
            def IFNA(value, value_if_na):
                return value_if_na if value in ("#ERR", "#N/A", None, "") else value

            def IFS(*args):
                for cond, res in zip(args[::2], args[1::2]):
                    if cond:
                        return res
                return "#N/A"

            def RANK(value, range_vals, order=0):
                try:
                    nums = []
                    for v in _flatten(range_vals):
                        try:
                            nums.append(float(v))
                        except Exception:
                            continue  # skip non-numeric
                    if not nums:
                        return "#ERR"
                    nums_sorted = sorted(nums, reverse=(order==0))
                    return nums_sorted.index(float(value))+1
                except Exception:
                    return "#ERR"

            def QUOTIENT(numerator, denominator):
                try:
                    return int(float(numerator)//float(denominator))
                except Exception:
                    return "#ERR"

            def TRUNC(number, digits=0):
                try:
                    n = float(number)
                    d = int(digits)
                    factor = 10**d
                    return int(n*factor)/factor
                except Exception:
                    return "#ERR"

            def LOG10(value):
                """Return base-10 logarithm. Accepts numeric or numeric‐string; errors => #ERR"""
                try:
                    # handle strings with commas/spaces
                    if isinstance(value, str):
                        value = value.strip().replace(",", "")
                    num = float(value)
                    if num <= 0:
                        return "#ERR"
                    return math.log10(num)
                except Exception:
                    return "#ERR"

            def LN(x):
                try:
                    return math.log(float(x))
                except Exception:
                    return "#ERR"

            def _apply_multi_criteria(sum_range, criteria_pairs, op):
                vals = list(_flatten(sum_range))
                ranges = []
                crits = []
                for i in range(0, len(criteria_pairs), 2):
                    ranges.append(list(_flatten(criteria_pairs[i])))
                    crits.append(criteria_pairs[i+1])
                if any(len(r)!=len(vals) for r in ranges):
                    return "#ERR"
                selected = []
                for i, v in enumerate(vals):
                    if all(_match(ranges[j][i], crits[j]) for j in range(len(crits))):
                        selected.append(v)
                nums = []
                for x in selected:
                    try:
                        nums.append(float(x))
                    except Exception:
                        continue
                return op(nums)

            def SUMIFS(sum_range, *criteria_pairs):
                res = _apply_multi_criteria(sum_range, criteria_pairs, sum)
                return res if res!="#ERR" else res

            def COUNTIFS(*args):
                if len(args)%2!=0:
                    return "#ERR"
                first_range = list(_flatten(args[0]))
                matches = [True]*len(first_range)
                for i in range(0,len(args),2):
                    rng = list(_flatten(args[i]))
                    crit = args[i+1]
                    if len(rng)!=len(first_range):
                        return "#ERR"
                    for idx,val in enumerate(rng):
                        if not _match(val, crit):
                            matches[idx] = False
                return sum(1 for m in matches if m)

            def AVERAGEIFS(avg_range, *criteria_pairs):
                def _avg(nums):
                    return sum(nums)/len(nums) if nums else "#ERR"
                return _apply_multi_criteria(avg_range, criteria_pairs, _avg)

            # trigonometric and hyperbolic additions
            def ASIN(x):
                return math.asin(float(x))
            def ACOS(x):
                return math.acos(float(x))
            def ATAN(x):
                return math.atan(float(x))
            def ATAN2(x_num, y_num):
                x=float(x_num)
                y=float(y_num)
                if x==0 and y==0:
                    return "#DIV/0!"
                return math.atan2(y, x)
            def SINH(x):
                return math.sinh(float(x))
            def COSH(x):
                return math.cosh(float(x))
            def TANH(x):
                return math.tanh(float(x))
            def SEC(x):
                return 1/math.cos(float(x))
            def COT(x):
                return 1/math.tan(float(x))
            def CSC(x):
                return 1/math.sin(float(x))

            allowed = {
                "SUM": SUM,
                "AVERAGE": AVERAGE,
                "MIN": MIN,
                "MAX": MAX,
                "COUNT": COUNT,
                "COUNTA": COUNTA,
                "PRODUCT": PRODUCT,
                "ABS": abs,
                "SQRT": SQRT,
                "POW": POW,
                "MOD": MOD,
                "ROUND": ROUND,
                "CEILING": CEILING,
                "FLOOR": FLOOR,
                "SIN": SIN,
                "COS": COS,
                "TAN": TAN,
                "LOG": LOG,
                "EXP": EXP,
                "IF": IF,
                "AND": AND_,
                "OR": OR_,
                "NOT": NOT_,
                "LEN": LEN,
                "CONCAT": CONCAT,
                "LEFT": LEFT,
                "RIGHT": RIGHT,
                "UPPER": UPPER,
                "LOWER": LOWER,
                "DATE": DATE,
                "MEDIAN": MEDIAN,
                "MODE": MODE,
                "RAND": RAND,
                "RANDBETWEEN": RANDBETWEEN,
                "TRIM": TRIM,
                "SUBSTITUTE": SUBSTITUTE,
                "FIND": FIND,
                "REPLACE": REPLACE,
                "VALUE": VALUE,
                "REPT": REPT,
                "VAR": VAR,
                "STDEV": STDEV,
                "DEGREES": DEGREES,
                "RADIANS": RADIANS,
                "SIGN": SIGN,
                "PI": PI,
                "EVEN": EVEN,
                "ODD": ODD,
                "INT": INT,
                "FACT": FACT,
                "ROUNDUP": ROUNDUP,
                "ROUNDDOWN": ROUNDDOWN,
                "TODAY": TODAY,
                "NOW": NOW,
                "YEAR": YEAR,
                "MONTH": MONTH,
                "DAY": DAY,
                "IFERROR": IFERROR,
                "COUNTBLANK": COUNTBLANK,
                "UNIQUE": UNIQUE,
                "SUMIF": SUMIF,
                "COUNTIF": COUNTIF,
                "AVERAGEIF": AVERAGEIF,
                "MID": MID,
                "PROPER": PROPER,
                "SMALL": SMALL,
                "LARGE": LARGE,
                "PERCENTILE": PERCENTILE,
                "SUMPRODUCT": SUMPRODUCT,
                "CHOOSE": CHOOSE,
                "ASIN": ASIN,
                "ACOS": ACOS,
                "ATAN": ATAN,
                "ATAN2": ATAN2,
                "SINH": SINH,
                "COSH": COSH,
                "TANH": TANH,
                "SEC": SEC,
                "COT": COT,
                "CSC": CSC,
                "XOR": XOR,
                "TEXTJOIN": TEXTJOIN,
                "SPLIT": SPLIT,
                "TRANSPOSE": TRANSPOSE,
                "COUNTUNIQUE": COUNTUNIQUE,
                "MAXIFS": MAXIFS,
                "MINIFS": MINIFS,
                "DAYS": DAYS,
                "NETWORKDAYS": NETWORKDAYS,
                "COLUMN": COLUMN,
                "SUMIFS": SUMIFS,
                "COUNTIFS": COUNTIFS,
                "AVERAGEIFS": AVERAGEIFS,
                "IFNA": IFNA,
                "IFS": IFS,
                "RANK": RANK,
                "QUOTIENT": QUOTIENT,
                "TRUNC": TRUNC,
                "LOG10": LOG10,
                "SLN": SLN,
                "DDB": DDB,
                "SYD": SYD,
                "COGS": COGS,
                "GPM": GPM,
                "NPM": NPM,
                "ROA": ROA,
                "ROE": ROE,
                "EPS": EPS,
                "WORKINGCAP": WORKINGCAP,
                "LN": LN,
                "LOG": LN,
                "TRUE": True,
                "FALSE": False,
            }
            # add lowercase aliases so functions can be called in any case
            # include functions defined in formulas package
            import formulas as _forms
            for _n in _forms.__all__:
                if _n not in allowed:
                    allowed[_n] = getattr(_forms, _n)
            # add extra formulas from external module
            if formulas_extra:
                for _n in dir(formulas_extra):
                    _obj=getattr(formulas_extra,_n)
                    if _n.isupper() and callable(_obj) and _n not in allowed:
                        allowed[_n]=_obj
            # ensure any newly added uppercase callable functions defined locally are included automatically
            for _n,_obj in locals().items():
                if _n.isupper() and callable(_obj) and _n not in allowed:
                    allowed[_n] = _obj
            # add lowercase aliases
            allowed.update({k.lower(): v for k, v in allowed.items()})
            try:
                # create case-insensitive mapping so mixed-case function names resolve
                class _CI(dict):
                    def __getitem__(self, k):
                        return super().__getitem__(str(k).upper())
                    def __contains__(self, k):
                        return super().__contains__(str(k).upper())
                allowed_ci = _CI({k.upper(): v for k, v in allowed.items()})
                value = eval(safe_expr, {"__builtins__": None}, allowed_ci)
            except Exception as e:
                print("Eval error:", safe_expr, e)
                value = "#ERR"
            visiting.discard(key)
            return value
        visiting.discard(key)
        return raw

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _expand_range(self, start: str, end: str) -> List[Tuple[int, int]]:
        s_col_letters, s_row = re.match(r"([A-Z]+)(\d+)", start).groups()
        e_col_letters, e_row = re.match(r"([A-Z]+)(\d+)", end).groups()
        r1, r2 = int(s_row) - 1, int(e_row) - 1
        c1 = self._col_letter_to_num(s_col_letters)
        c2 = self._col_letter_to_num(e_col_letters)
        cells: List[Tuple[int, int]] = []
        for r in range(min(r1, r2), max(r1, r2) + 1):
            for c in range(min(c1, c2), max(c1, c2) + 1):
                cells.append((r, c))
        return cells
