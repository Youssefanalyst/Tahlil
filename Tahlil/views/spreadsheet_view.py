"""Custom QTableView with Excel-like fill handle (drag to copy)."""
from __future__ import annotations

from PyQt6.QtWidgets import QTableView, QApplication
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QPainter, QColor, QBrush
import re


from PyQt6.QtWidgets import QStyledItemDelegate

class _ColorAwareDelegate(QStyledItemDelegate):
    """Delegate that paints background brush explicitly to bypass style sheets."""
    def paint(self, painter, option, index):
        brush = index.data(Qt.ItemDataRole.BackgroundRole)
        if isinstance(brush, QBrush):
            painter.save()
            painter.fillRect(option.rect, brush)
            painter.restore()
        super().paint(painter, option, index)


class SpreadsheetView(QTableView):
    _HANDLE_SIZE = 6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fill_dragging = False
        self._drag_start_index = None
        self._drag_last_pos: QPoint | None = None
        # ---- pattern capture ----
        self._pattern_vals: list = []
        self._pattern_orientation: str | None = None  # 'vertical' | 'horizontal'
        self.setMouseTracking(True)
        # use delegate to ensure background colours not overridden by theme
        self.setItemDelegate(_ColorAwareDelegate(self))
        # Performance tweaks
        # self.setUniformRowHeights(True)  # removed: method not available for QTableView
        self.setAlternatingRowColors(False)  # disable stripes for large datasets
        self.setWordWrap(False)

    # ---- paint handle ----
    def paintEvent(self, event):
        super().paintEvent(event)
        sel = self.selectionModel()
        if not sel or not sel.hasSelection():
            return
        idxs = sel.selectedIndexes()
        if not idxs:
            return
        rows = [i.row() for i in idxs]
        cols = [i.column() for i in idxs]
        top_left = self.model().index(min(rows), min(cols))
        bottom_right = self.model().index(max(rows), max(cols))
        cell_rect = self.visualRect(bottom_right)
        handle_rect = QRect(cell_rect.right() - self._HANDLE_SIZE + 1,
                            cell_rect.bottom() - self._HANDLE_SIZE + 1,
                            self._HANDLE_SIZE, self._HANDLE_SIZE)
        painter = QPainter(self.viewport())
        painter.fillRect(handle_rect, QColor("#0078d7"))

    # ---- mouse events ----
    def _handle_rect_at_index(self, index):
        cell_rect = self.visualRect(index)
        return QRect(cell_rect.right() - self._HANDLE_SIZE + 1,
                     cell_rect.bottom() - self._HANDLE_SIZE + 1,
                     self._HANDLE_SIZE, self._HANDLE_SIZE)

    def mousePressEvent(self, event):
        sel = self.selectionModel()
        if sel and sel.hasSelection():
            index = sel.currentIndex()
            if self._handle_rect_at_index(index).contains(event.pos()):
                # capture source pattern BEFORE selection changes
                self._capture_pattern()
                # start fill drag
                self._fill_dragging = True
                self._drag_start_index = index
                self._drag_last_pos = event.pos()
                QApplication.setOverrideCursor(Qt.CursorShape.CrossCursor)
                return  # don't let base handle
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._fill_dragging:
            self._drag_last_pos = event.pos()
            # highlight temporary range
            idx = self.indexAt(event.pos())
            if idx.isValid():
                sel = self.selectionModel()
                start = self._drag_start_index
                r1, r2 = sorted([start.row(), idx.row()])
                c1, c2 = sorted([start.column(), idx.column()])
                from PyQt6.QtCore import QItemSelection, QItemSelectionModel
                selection = QItemSelection(self.model().index(r1, c1),
                                            self.model().index(r2, c2))
                sel.select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._fill_dragging:
            self._fill_dragging = False
            QApplication.restoreOverrideCursor()
            idx_end = self.indexAt(event.pos())
            if idx_end.isValid():
                self._perform_fill(self._drag_start_index, idx_end)
            self._drag_start_index = None
            return
        super().mouseReleaseEvent(event)

    # ---- pattern helpers & fill logic ----
    def _capture_pattern(self) -> None:
        """Capture current selection values and orientation for later fill."""
        sel = self.selectionModel()
        idxs = sel.selectedIndexes()
        if not idxs:
            self._pattern_vals = []
            self._pattern_orientation = None
            return
        model = self.model()
        cols = {i.column() for i in idxs}
        rows = {i.row() for i in idxs}
        if len(cols) == 1:
            self._pattern_orientation = "vertical"
            col = next(iter(cols))
            ordered_rows = sorted(rows)
            self._pattern_vals = [model.data(model.index(r, col), Qt.ItemDataRole.EditRole) for r in ordered_rows]
        elif len(rows) == 1:
            self._pattern_orientation = "horizontal"
            row = next(iter(rows))
            ordered_cols = sorted(cols)
            self._pattern_vals = [model.data(model.index(row, c), Qt.ItemDataRole.EditRole) for c in ordered_cols]
        else:
            # rectangular selection – fallback to copy behaviour
            self._pattern_orientation = None
            self._pattern_vals = []

    def _generate_sequence(self, length: int):
        """Generate sequence of *length* elements extending captured pattern."""
        if length <= 0 or not self._pattern_vals:
            return []

        vals = self._pattern_vals
        # Try numeric progression first
        try:
            nums = [float(v) for v in vals]
            numeric = True
        except (TypeError, ValueError):
            numeric = False

        seq: list = []
        if numeric and len(nums) >= 2:
            diff = nums[-1] - nums[-2]
            current = nums[-1]
            for _ in range(length):
                current += diff
                # Preserve int if original numbers were ints
                if all(isinstance(v, int) or (isinstance(v, float) and v.is_integer()) for v in vals):
                    seq.append(int(current))
                else:
                    seq.append(current)
        else:
            # Fallback – repeat/cycle the captured pattern
            i = 0
            while len(seq) < length:
                seq.append(vals[i % len(vals)])
                i += 1
        return seq

    # ---------------------------------------------------------------
    # Formula shifting helpers
    # ---------------------------------------------------------------
    _re_cell = re.compile(r"([A-Z]+)(\d+)(?![A-Z0-9])(?!(?=\())")

    @staticmethod
    def _col_letter_to_num(letters: str) -> int:
        num = 0
        for ch in letters:
            num = num * 26 + (ord(ch) - 64)
        return num - 1

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

    def _shift_formula(self, formula: str, row_delta: int, col_delta: int) -> str:
        """Return formula string shifted by given row/col deltas."""
        if not formula.startswith("="):
            return formula
        expr = formula[1:]

        def repl(m: re.Match[str]):
            letters, digits = m.groups()
            new_row = max(1, int(digits) + row_delta)
            new_col_num = max(0, self._col_letter_to_num(letters) + col_delta)
            new_letters = self._num_to_col_letter(new_col_num)
            return f"{new_letters}{new_row}"

        shifted = self._re_cell.sub(repl, expr)
        return "=" + shifted

    def _adjust_value_for_fill(self, value, row_delta: int, col_delta: int):
        if isinstance(value, str) and value.startswith("="):
            return self._shift_formula(value, row_delta, col_delta)
        return value

    def _perform_fill(self, start_idx, end_idx):
        model = self.model()
        if not start_idx.isValid() or not end_idx.isValid():
            return

        # Vertical fill
        if self._pattern_orientation == "vertical" and start_idx.column() == end_idx.column():
            col = start_idx.column()
            start_row = start_idx.row()
            end_row = end_idx.row()
            direction = 1 if end_row > start_row else -1
            rows_range = list(range(start_row + direction, end_row + direction, direction))
            plen = len(self._pattern_vals) if self._pattern_vals else 1
            for idx, r in enumerate(rows_range, 1):
                base_val = self._pattern_vals[(idx - 1) % plen] if self._pattern_vals else model.data(start_idx, Qt.ItemDataRole.EditRole)
                new_val = self._adjust_value_for_fill(base_val, r - start_row, 0)
                model.setData(model.index(r, col), new_val, Qt.ItemDataRole.EditRole)
            return

        # Horizontal fill
        if self._pattern_orientation == "horizontal" and start_idx.row() == end_idx.row():
            row = start_idx.row()
            start_col = start_idx.column()
            end_col = end_idx.column()
            direction = 1 if end_col > start_col else -1
            cols_range = list(range(start_col + direction, end_col + direction, direction))
            plen = len(self._pattern_vals) if self._pattern_vals else 1
            for idx, c in enumerate(cols_range, 1):
                base_val = self._pattern_vals[(idx - 1) % plen] if self._pattern_vals else model.data(start_idx, Qt.ItemDataRole.EditRole)
                new_val = self._adjust_value_for_fill(base_val, 0, c - start_col)
                model.setData(model.index(row, c), new_val, Qt.ItemDataRole.EditRole)
            return

        # Fallback: simple copy as before
        value = model.data(start_idx, Qt.ItemDataRole.EditRole)
        if start_idx.column() == end_idx.column():
            col = start_idx.column()
            start_row = start_idx.row()
            end_row = end_idx.row()
            direction = 1 if end_row > start_row else -1
            for r in range(start_row + direction, end_row + direction, direction):
                new_val = self._adjust_value_for_fill(value, r - start_row, 0)
                model.setData(model.index(r, col), new_val, Qt.ItemDataRole.EditRole)
        elif start_idx.row() == end_idx.row():
            row = start_idx.row()
            start_col = start_idx.column()
            end_col = end_idx.column()
            direction = 1 if end_col > start_col else -1
            for c in range(start_col + direction, end_col + direction, direction):
                new_val = self._adjust_value_for_fill(value, 0, c - start_col)
                model.setData(model.index(row, c), new_val, Qt.ItemDataRole.EditRole)
