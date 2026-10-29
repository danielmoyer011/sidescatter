"""The Boolean gate editor: combine populations with AND / OR, NOT on any input.

Opened from the ribbon's Boolean button (or the sample table's right-click menu)
with the selected populations already filled in, as in FlowJo. "Make all
combinations" builds one AND gate per +/- combination, for cytokine panels.
"""

from PySide6 import QtCore, QtWidgets

from ..core.workspace import ROOT, boolean_name
from . import icons, theme
from .gating import service as gating


def population_label(name, parent):
    return " › ".join([*tuple(parent)[1:], name])


class BooleanDialog(QtWidgets.QDialog):
    """Create a Boolean gate (or several), or edit one with editing=(name, parent_path)."""

    def __init__(self, workspace, inputs=(), editing=None, sample_id=None, parent=None):
        super().__init__(parent)
        self.ws = workspace
        self.editing = editing
        self.sample_id = sample_id
        self.made = []  # (name, parent_path) of each population created or edited
        self._auto_name = True
        self.setWindowTitle("Edit Boolean gate" if editing else "Boolean gate")
        self.setWindowIcon(icons.icon("boolean"))
        self.setMinimumWidth(520)

        self.candidates = self.ws.boolean_candidates(exclude=editing)
        op, rows = "and", [(n, tuple(p), False) for n, p in inputs]
        if editing:
            op, rows = self.ws.boolean_inputs(*editing)

        self.name_edit = QtWidgets.QLineEdit(editing[0] if editing else "")
        self.name_edit.textEdited.connect(self._name_typed)
        self.and_button = QtWidgets.QRadioButton("AND: events in all of them")
        self.or_button = QtWidgets.QRadioButton("OR: events in any of them")
        (self.or_button if op == "or" else self.and_button).setChecked(True)
        ops = QtWidgets.QButtonGroup(self)
        ops.addButton(self.and_button)
        ops.addButton(self.or_button)
        ops.buttonToggled.connect(lambda *_: self._update())
        op_row = QtWidgets.QHBoxLayout()
        op_row.addWidget(self.and_button)
        op_row.addWidget(self.or_button)
        op_row.addStretch()

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["NOT", "Population"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for n, p, neg in rows:
            self.add_row(n, p, neg)
        if self.table.rowCount() == 0:
            self.add_row()
        add = QtWidgets.QPushButton("Add population")
        add.clicked.connect(lambda: self.add_row())
        remove = QtWidgets.QPushButton("Remove")
        remove.clicked.connect(self._remove_rows)
        row_buttons = QtWidgets.QHBoxLayout()
        row_buttons.addWidget(add)
        row_buttons.addWidget(remove)
        row_buttons.addStretch()

        self.placement = QtWidgets.QLabel()
        self.placement.setStyleSheet(f"color: {theme.MUTED};")
        self.placement.setWordWrap(True)

        self.combos_box = QtWidgets.QGroupBox("All combinations")
        self.all_negative = QtWidgets.QCheckBox("Include the all-negative combination")
        combos_button = QtWidgets.QPushButton("Make all combinations")
        combos_button.setToolTip("One AND gate for every +/- combination of these populations, "
                                 "e.g. IFNg+ TNF- IL-2+")
        combos_button.clicked.connect(self.make_combinations)
        combos = QtWidgets.QHBoxLayout(self.combos_box)
        combos.addWidget(combos_button)
        combos.addWidget(self.all_negative)
        combos.addStretch()
        self.combos_box.setVisible(editing is None)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.ok_button = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.ok_button.setText("Save" if editing else "Create gate")
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.name_edit)
        form.addRow("Combine", op_row)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.table, 1)
        layout.addLayout(row_buttons)
        layout.addWidget(self.placement)
        layout.addWidget(self.combos_box)
        layout.addWidget(buttons)
        self._update()

    # ----- inputs ------------------------------------------------------------

    def add_row(self, name=None, parent=ROOT, negated=False):
        r = self.table.rowCount()
        self.table.insertRow(r)
        tick = QtWidgets.QCheckBox()
        tick.setChecked(negated)
        tick.setToolTip("Use the events that are NOT in this population")
        tick.toggled.connect(lambda *_: self._update())
        cell = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(cell)
        box.setContentsMargins(8, 0, 8, 0)
        box.addWidget(tick, 0, QtCore.Qt.AlignCenter)
        self.table.setCellWidget(r, 0, cell)
        combo = QtWidgets.QComboBox()
        for n, p in self.candidates:
            combo.addItem(population_label(n, p), (n, tuple(p)))
        if name is not None:
            i = combo.findData((name, tuple(parent)))
            if i < 0:  # not a usable input (should not happen); keep it visible anyway
                combo.addItem(population_label(name, parent), (name, tuple(parent)))
                i = combo.count() - 1
            combo.setCurrentIndex(i)
        else:
            used = {k[:2] for k in self.inputs()}
            free = [i for i in range(combo.count()) if combo.itemData(i) not in used]
            combo.setCurrentIndex(free[0] if free else -1)
        combo.currentIndexChanged.connect(lambda *_: self._update())
        self.table.setCellWidget(r, 1, combo)
        self._update()

    def _remove_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for r in rows:
            self.table.removeRow(r)
        self._update()

    def inputs(self):
        """[(name, parent_path, negated), ...] as filled in, skipping empty rows."""
        out = []
        for r in range(self.table.rowCount()):
            combo = self.table.cellWidget(r, 1)
            tick = self.table.cellWidget(r, 0)
            if combo is None or combo.currentData() is None:
                continue
            n, p = combo.currentData()
            out.append((n, p, tick.findChild(QtWidgets.QCheckBox).isChecked()))
        return out

    def set_negated(self, row, on):
        self.table.cellWidget(row, 0).findChild(QtWidgets.QCheckBox).setChecked(on)

    @property
    def op(self):
        return "or" if self.or_button.isChecked() else "and"

    # ----- live preview ------------------------------------------------------

    def _name_typed(self, text):
        self._auto_name = not text.strip()
        self._update()

    def _update(self):
        if not hasattr(self, "ok_button"):
            return  # still building
        inputs = self.inputs()
        if self._auto_name and inputs and not self.editing:
            self.name_edit.setText(boolean_name(inputs, self.op))
        problem = None
        if not inputs:
            problem = "Pick the populations to combine."
        elif len(inputs) == 1 and not inputs[0][2]:
            problem = "Add another population, or tick NOT to use everything outside this one."
        elif len({k[:2] for k in inputs}) < len(inputs):
            problem = "Each population can only be used once."
        if problem:
            self.placement.setText(problem)
        else:
            parent = self.ws.boolean_parent(inputs)
            where = "the top level" if parent == ROOT else population_label(parent[-1], parent[:-1])
            text = f"Goes under {where}, in every sample."
            if self.sample_id is not None:
                text += f" {self._preview_count(inputs, parent)}"
            self.placement.setText(text)
        self.ok_button.setEnabled(problem is None)

    def _preview_count(self, inputs, parent):
        sid = self.sample_id
        if not gating().ensure(self.ws, [sid]):
            if not getattr(self, "_waiting", False):
                self._waiting = True
                gating().ready.connect(self._results_ready)
            return "Counting events…"
        try:
            mask = self.ws.membership(sid, *(_split(parent)))
            hits = [self.ws.membership(sid, n, p) ^ neg for n, p, neg in inputs]
        except KeyError:
            return ""
        combined = hits[0].copy()
        for h in hits[1:]:
            combined = (combined | h) if self.op == "or" else (combined & h)
        return f"{int((mask & combined).sum()):,} events in {sid}."

    def _results_ready(self, ws, sample_id):
        if ws is self.ws and sample_id == self.sample_id:
            self._update()

    # ----- results -----------------------------------------------------------

    def apply(self):
        try:
            if self.editing:
                self.made = [self.ws.edit_boolean(*self.editing, self.inputs(), self.op,
                                                  new_name=self.name_edit.text())]
            else:
                self.made = [self.ws.add_boolean(self.inputs(), self.op, self.name_edit.text())]
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Boolean gate", str(exc))
            return
        self.accept()

    def make_combinations(self):
        try:
            self.made = self.ws.boolean_combinations(
                self.inputs(), include_all_negative=self.all_negative.isChecked())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "All combinations", str(exc))
            return
        self.accept()


def confirm_delete(parent, ws, name, path):
    """Ask before deleting a gate, naming the Boolean gates that would go with it."""
    users = ws.boolean_dependents(name, path)
    text = f"Delete {name} and every population inside it, in all samples?"
    if users:
        listed = "\n".join(f"  • {population_label(n, p)}" for n, p in users[:12])
        more = f"\n  … and {len(users) - 12} more" if len(users) > 12 else ""
        text += (f"\n\nThese Boolean gates use it and will be deleted too:\n{listed}{more}")
    answer = QtWidgets.QMessageBox.question(parent, "Delete gate", text)
    return answer == QtWidgets.QMessageBox.Yes


def _split(path):
    path = tuple(path)
    return (None, ROOT) if path == ROOT else (path[-1], path[:-1])
