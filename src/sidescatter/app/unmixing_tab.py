"""The Unmixing tab: raw files in, classified, unmixed, and packaged like SpectroFlo.

Add or drop every raw file of an experiment. Each one is classified as an
unstained or single-color reference (on beads or cells), a multicolor
sample, or left out, and the table shows why. Fix anything that's wrong,
check the signatures and the live preview, then unmix into the workspace or
export a zipped experiment with Raw and Unmixed folders.
"""

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import experiment, fcs, spectral, unmixing
from ..core.experiment import Setup
from .graph_view import BINS, PSEUDOCOLOR

COLS = ["File", "Role", "Fluor", "Marker", "Negative", "Peak", "Positives", "Why"]
C_FILE, C_ROLE, C_FLUOR, C_MARKER, C_NEG, C_PEAK, C_POS, C_WHY = range(len(COLS))
# The six choices in the Role column: (label, role, carrier).
ROLE_CHOICES = [
    ("Unstained (beads)", "unstained", "beads"),
    ("Unstained (cells)", "unstained", "cells"),
    ("Single color (beads)", "single_stain", "beads"),
    ("Single color (cells)", "single_stain", "cells"),
    ("Multicolor sample", "sample", None),
    ("Exclude", "exclude", None),
]
ROLE_COLORS = {"unstained": "#EEF0F6", "single_stain": "#E8F6F6", "sample": "#EEEEFB", "exclude": "#F6F7FB"}
ASINH_COFACTOR = 2000.0
PREVIEW_EVENTS = 50_000
PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
           (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


def _asinh(x):
    return np.arcsinh(np.asarray(x, dtype=float) / ASINH_COFACTOR)


def role_index(spec):
    for i, (_, role, carrier) in enumerate(ROLE_CHOICES):
        if role == spec.role and carrier in (None, spec.carrier):
            return i
    return len(ROLE_CHOICES) - 1


class UnmixingTab(QtWidgets.QWidget):
    unmixed = QtCore.Signal(list)   # new sample ids in the workspace
    changed = QtCore.Signal()       # the setup changed (marks the workspace unsaved)

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.ws = workspace
        self.setups = []
        self.setup = None
        self.samples = {}          # path -> loaded raw Sample, shared by every setup
        self.unmixer = None
        self.results = []
        self.problems = []
        self._preview_cache = {}
        self.setAcceptDrops(True)

        # ----- experiment picker
        self.exp_combo = QtWidgets.QComboBox()
        self.exp_combo.setMinimumWidth(220)
        self.exp_combo.currentIndexChanged.connect(self._pick_setup)
        new_btn = QtWidgets.QPushButton("New experiment")
        new_btn.clicked.connect(lambda: self.new_setup())
        rename_btn = QtWidgets.QPushButton("Rename…")
        rename_btn.clicked.connect(self._rename_setup)
        delete_btn = QtWidgets.QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_setup)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Experiment"))
        top.addWidget(self.exp_combo)
        top.addWidget(new_btn)
        top.addWidget(rename_btn)
        top.addWidget(delete_btn)
        top.addStretch()

        # ----- file table
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.itemSelectionChanged.connect(self._show_control)
        self.table.itemChanged.connect(self._on_item_edited)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)

        add_btn = QtWidgets.QPushButton("Add files…")
        add_btn.clicked.connect(self._add_files)
        folder_btn = QtWidgets.QPushButton("Add folder…")
        folder_btn.clicked.connect(self._add_folder)
        remove_btn = QtWidgets.QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_selected)
        self.drop_hint = QtWidgets.QLabel(
            "Add or drop all the raw files of an experiment: unstained and single-color controls on "
            "beads or cells, and your multicolor samples. Change any role in the Role column; with "
            "several rows selected, the change applies to all of them.")
        self.drop_hint.setWordWrap(True)
        self.drop_hint.setObjectName("Hint")

        self.method = QtWidgets.QComboBox()
        self.method.addItem("Weighted least squares (WLS)", "wls")
        self.method.addItem("Ordinary least squares (OLS)", "ols")
        self.method.currentIndexChanged.connect(self._settings_changed)
        self.af_check = QtWidgets.QCheckBox("Extract autofluorescence from unstained cells")
        self.af_check.setChecked(True)
        self.af_check.toggled.connect(self._settings_changed)
        self.af_count = QtWidgets.QSpinBox()
        self.af_count.setRange(1, 6)
        self.af_count.setPrefix("Signatures: ")
        self.af_count.valueChanged.connect(self._settings_changed)
        options = QtWidgets.QFormLayout()
        options.addRow("Method", self.method)
        af_row = QtWidgets.QHBoxLayout()
        af_row.addWidget(self.af_check)
        af_row.addWidget(self.af_count)
        options.addRow(af_row)

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.RichText)

        left = QtWidgets.QVBoxLayout()
        buttons = QtWidgets.QHBoxLayout()
        for b in (add_btn, folder_btn, remove_btn):
            buttons.addWidget(b)
        buttons.addStretch()
        left.addLayout(buttons)
        left.addWidget(self.drop_hint)
        left.addWidget(self.table, 1)
        left.addLayout(options)
        left.addWidget(self.status)

        # ----- workbench
        self.tabs = QtWidgets.QTabWidget()
        self.sig_plot = pg.PlotWidget(background="w")
        self.sig_plot.addLegend(offset=(-10, 10))
        self.sig_plot.setLabel("left", "Normalised intensity")
        self.tabs.addTab(self.sig_plot, "Signatures")

        self.sim_table = QtWidgets.QTableWidget()
        self.tabs.addTab(self.sim_table, "Similarity")

        self.ctrl_plot = pg.PlotWidget(background="w")
        self.ctrl_plot.setLabel("bottom", "Peak detector (asinh scale)")
        self.ctrl_hint = QtWidgets.QLabel(
            "Select a single-color control. Drag the shaded region to choose which events count as positive.")
        self.ctrl_hint.setWordWrap(True)
        ctrl_box = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(ctrl_box)
        cl.addWidget(self.ctrl_hint)
        cl.addWidget(self.ctrl_plot)
        self.tabs.addTab(ctrl_box, "Control gate")

        preview = QtWidgets.QWidget()
        pl = QtWidgets.QGridLayout(preview)
        self.preview_sample = QtWidgets.QComboBox()
        self.preview_x = QtWidgets.QComboBox()
        self.preview_y = QtWidgets.QComboBox()
        for c in (self.preview_sample, self.preview_x, self.preview_y):
            c.currentIndexChanged.connect(self._draw_preview)
        self.preview_plot = pg.PlotWidget(background="w")
        self.resid_plot = pg.PlotWidget(background="w")
        self.resid_plot.setLabel("left", "Residual RMS")
        self.resid_plot.setMaximumHeight(170)
        pl.addWidget(QtWidgets.QLabel("Sample"), 0, 0)
        pl.addWidget(self.preview_sample, 0, 1)
        pl.addWidget(QtWidgets.QLabel("X"), 0, 2)
        pl.addWidget(self.preview_x, 0, 3)
        pl.addWidget(QtWidgets.QLabel("Y"), 0, 4)
        pl.addWidget(self.preview_y, 0, 5)
        pl.addWidget(self.preview_plot, 1, 0, 1, 6)
        pl.addWidget(self.resid_plot, 2, 0, 1, 6)
        self.tabs.addTab(preview, "Live preview")

        # ----- actions
        self.unmix_btn = QtWidgets.QPushButton("Unmix into workspace")
        self.unmix_btn.setToolTip("Add the unmixed samples (and references) to the workspace for gating")
        self.unmix_btn.clicked.connect(self.unmix_into_workspace)
        self.export_btn = QtWidgets.QPushButton("Export experiment…")
        self.export_btn.setToolTip("Write a zipped experiment with Raw and Unmixed folders, like SpectroFlo")
        self.export_btn.clicked.connect(self.export)
        self.export_btn.setDefault(True)

        split = QtWidgets.QSplitter()
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        split.addWidget(lw)
        split.addWidget(self.tabs)
        split.setSizes([760, 540])

        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(self.unmix_btn)
        bottom.addWidget(self.export_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.addLayout(top)
        layout.addWidget(split, 1)
        layout.addLayout(bottom)

        self.set_workspace(workspace)

    # ----- setups ----------------------------------------------------------

    def set_workspace(self, workspace):
        self.ws = workspace
        self.setups = [Setup.from_dict(d) for d in workspace.experiments]
        if not self.setups:
            self.setups = [Setup("Experiment 1")]
        self._fill_setups(0)

    def store(self):
        """Write the setups into the workspace, so saving the workspace keeps them."""
        self.ws.experiments = [s.to_dict() for s in self.setups if s.specs]

    def _fill_setups(self, current):
        self.exp_combo.blockSignals(True)
        self.exp_combo.clear()
        for s in self.setups:
            self.exp_combo.addItem(s.name)
        self.exp_combo.setCurrentIndex(current)
        self.exp_combo.blockSignals(False)
        self._pick_setup(current)

    def _pick_setup(self, index):
        if index < 0 or index >= len(self.setups):
            return
        self.setup = self.setups[index]
        for w, v in ((self.method, self.method.findData(self.setup.method)),
                     (self.af_count, self.setup.n_autofluorescence)):
            w.blockSignals(True)
            w.setCurrentIndex(v) if w is self.method else w.setValue(v)
            w.blockSignals(False)
        self.af_check.blockSignals(True)
        self.af_check.setChecked(self.setup.autofluorescence)
        self.af_check.blockSignals(False)
        missing = [s for s in self.setup.specs if not self._load(s.path)]
        if missing:
            gone = {id(s) for s in missing}
            self.setup.specs = [s for s in self.setup.specs if id(s) not in gone]
            QtWidgets.QMessageBox.warning(
                self, "Missing files", "These files could not be found and were left out:\n\n"
                + "\n".join(str(s.path) for s in missing[:10]))
        self._fill_table()
        self.recalculate()

    def new_setup(self, name=None):
        names = {s.name for s in self.setups}
        n = len(self.setups) + 1
        while name is None or name in names:
            name, n = f"Experiment {n}", n + 1
        self.setups.append(Setup(name))
        self._fill_setups(len(self.setups) - 1)
        return self.setups[-1]

    def _rename_setup(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename experiment", "Name:", text=self.setup.name)
        if ok and name.strip():
            self.setup.name = name.strip()
            self.exp_combo.setItemText(self.exp_combo.currentIndex(), self.setup.name)
            self._mark_changed()

    def _delete_setup(self):
        i = self.exp_combo.currentIndex()
        del self.setups[i]
        if not self.setups:
            self.setups = [Setup("Experiment 1")]
        self._fill_setups(max(i - 1, 0))
        self._mark_changed()

    # ----- files -----------------------------------------------------------

    @property
    def specs(self):
        return self.setup.specs

    def _load(self, path):
        if path not in self.samples:
            try:
                self.samples[path] = fcs.load_sample(path)
            except (OSError, ValueError):
                return False
        return True

    def _add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add raw files", "", "FCS files (*.fcs *.FCS)")
        if paths:
            self.add_files(paths)

    def _add_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Add a folder of raw files")
        if folder:
            self.add_files([folder])

    def add_files(self, paths):
        """Add files or folders (searched recursively); returns the names skipped."""
        found, skipped = fcs.find_fcs_files(paths)
        known = {s.path for s in self.specs}
        new = [Path(p) for p in found if Path(p) not in known]
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            unreadable = [p for p in new if not self._load(p)]
            new = [p for p in new if p not in unreadable]
            self.specs.extend(unmixing.guess_files(new, self.samples))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        skipped += [p.name for p in unreadable]
        self._fill_table()
        self.recalculate()
        self._mark_changed()
        return skipped

    def remove_selected(self):
        rows = self._selected_rows()
        for row in sorted(rows, reverse=True):
            del self.specs[row]
        if rows:
            self._fill_table()
            self.recalculate()
            self._mark_changed()

    def _selected_rows(self):
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def _fill_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.specs))
        unstained = [s for s in self.specs if s.role == "unstained"]
        for row, spec in enumerate(self.specs):
            color = QtGui.QColor(ROLE_COLORS[spec.role])
            file_item = self._item(spec.path.name, editable=False, tip=str(spec.path))
            self.table.setItem(row, C_FILE, file_item)
            role = QtWidgets.QComboBox()
            for label, _, _ in ROLE_CHOICES:
                role.addItem(label)
            role.setCurrentIndex(role_index(spec))
            role.currentIndexChanged.connect(lambda i, s=spec: self._set_role(s, i))
            self.table.setCellWidget(row, C_ROLE, role)
            single = spec.role == "single_stain"
            self.table.setItem(row, C_FLUOR, self._item(spec.name if single else "", editable=single))
            self.table.setItem(row, C_MARKER, self._item(spec.marker, editable=spec.role != "exclude"))
            negs = [("Auto", "auto"), ("Internal", "internal")]
            negs += [(u.path.name, str(u.path)) for u in unstained]
            neg = QtWidgets.QComboBox()
            for label, value in negs:
                neg.addItem(label, value)
            neg.setCurrentIndex(max(neg.findData(spec.negative), 0))
            neg.setEnabled(single)
            neg.setToolTip("Auto: the unstained file of the same carrier (beads or cells). "
                           "Internal: the dim events in this file.")
            neg.currentIndexChanged.connect(lambda _, s=spec, c=neg: self._set_negative(s, c.currentData()))
            self.table.setCellWidget(row, C_NEG, neg)
            for col in (C_PEAK, C_POS):
                self.table.setItem(row, col, self._item("", editable=False))
            why = "; ".join(spec.reasons)
            self.table.setItem(row, C_WHY, self._item(why, editable=False, tip=why))
            for col in range(len(COLS)):
                item = self.table.item(row, col)
                if item is not None:
                    item.setBackground(color)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(C_FILE, min(self.table.columnWidth(C_FILE), 260))
        self.table.blockSignals(False)
        self.drop_hint.setVisible(not self.specs)

    @staticmethod
    def _item(text, editable=True, tip=None):
        item = QtWidgets.QTableWidgetItem(text)
        if not editable:
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        if tip:
            item.setToolTip(tip)
        return item

    def _set_role(self, spec, index):
        """Set one row's role, or every selected row's when that row is among them."""
        _, role, carrier = ROLE_CHOICES[index]
        row = next(i for i, s in enumerate(self.specs) if s is spec)
        rows = self._selected_rows()
        targets = [self.specs[r] for r in rows] if row in rows else [spec]
        for s in targets:
            s.role = role
            if carrier:
                s.carrier = carrier
            s.guessed_by = "you"
            if role == "single_stain" and not s.name:
                s.name = s.path.stem
        self._fill_table()
        self._select_rows(rows)
        self.recalculate()
        self._mark_changed()

    def _select_rows(self, rows):
        model = self.table.selectionModel()
        flags = QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
        for r in rows:
            model.select(self.table.model().index(r, 0), flags)

    def set_roles(self, rows, index):
        """Give these rows the ROLE_CHOICES[index] role (used by the menu and tests)."""
        _, role, carrier = ROLE_CHOICES[index]
        for r in rows:
            s = self.specs[r]
            s.role, s.guessed_by = role, "you"
            if carrier:
                s.carrier = carrier
        self._fill_table()
        self.recalculate()
        self._mark_changed()

    def _set_negative(self, spec, value):
        spec.negative = value
        self.recalculate()
        self._mark_changed()

    def _on_item_edited(self, item):
        spec = self.specs[item.row()]
        text = item.text().strip()
        if item.column() == C_FLUOR and text:
            spec.name = text
        elif item.column() == C_MARKER:
            spec.marker = text
        else:
            return
        self.recalculate()
        self._mark_changed()

    def _table_menu(self, pos):
        rows = self._selected_rows()
        if not rows:
            return
        menu = QtWidgets.QMenu(self)
        sub = menu.addMenu(f"Set role for {len(rows)} file(s)")
        for i, (label, _, _) in enumerate(ROLE_CHOICES):
            sub.addAction(label, lambda i=i: self.set_roles(rows, i))
        menu.addAction("Remove", self.remove_selected)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _settings_changed(self, *_):
        self.setup.method = self.method.currentData()
        self.setup.autofluorescence = self.af_check.isChecked()
        self.setup.n_autofluorescence = self.af_count.value()
        self.recalculate()
        self._mark_changed()

    def _mark_changed(self):
        self.store()
        self.changed.emit()

    # ----- drag and drop ---------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return
        event.acceptProposedAction()
        skipped = self.add_files(paths)
        if skipped:
            QtWidgets.QMessageBox.warning(
                self, "Not FCS files", f"Skipped {len(skipped)} item(s):\n\n" + "\n".join(skipped[:10]))

    # ----- computing -------------------------------------------------------

    def recalculate(self):
        self.problems = unmixing.check_specs(self.specs) if self.specs else []
        errors = [m for level, m in self.problems if level == "error"]
        self.unmixer, self.results = None, []
        if self.specs and not errors:
            try:
                self.unmixer, self.results = unmixing.build_unmixer(
                    [s for s in self.specs if s.role != "exclude"], method=self.setup.method,
                    n_autofluorescence=self.setup.n_autofluorescence,
                    autofluorescence=self.setup.autofluorescence, samples=self.samples)
            except Exception as exc:
                errors.append(f"Couldn't build the unmixing matrix: {exc}")
        ok = self.unmixer is not None
        self.unmix_btn.setEnabled(ok)
        self.export_btn.setEnabled(ok)
        self._show_status(errors)
        by_path = {r.spec.path: r for r in self.results}
        self.table.blockSignals(True)
        for row, spec in enumerate(self.specs):
            r = by_path.get(spec.path)
            self.table.item(row, C_PEAK).setText(r.peak_detector if r else "")
            self.table.item(row, C_POS).setText(f"{r.n_positive:,}" if r else "")
        self.table.blockSignals(False)
        self.sig_plot.clear()
        self.sim_table.clear()
        self.sim_table.setRowCount(0)
        if ok:
            self._draw_signatures()
            self._draw_similarity()
        self._show_control()
        self._fill_preview()

    def _show_status(self, errors):
        counts = {}
        for s in self.specs:
            counts[s.role] = counts.get(s.role, 0) + 1
        parts = [f"{counts.get('single_stain', 0)} single color", f"{counts.get('unstained', 0)} unstained",
                 f"{counts.get('sample', 0)} samples"]
        if counts.get("exclude"):
            parts.append(f"{counts['exclude']} excluded")
        lines = [", ".join(parts) + "."] if self.specs else []
        if self.unmixer is not None:
            um = self.unmixer
            sim = spectral.similarity_matrix(um.matrix)
            np.fill_diagonal(sim, 0)
            i, j = np.unravel_index(np.argmax(sim), sim.shape)
            text = f"Complexity index {spectral.complexity_index(um.matrix):.1f} with {len(um.names)} components."
            if len(um.names) > 1:
                text += f" Most similar pair: {um.names[i]} and {um.names[j]} ({sim[i, j]:.2f})."
                if sim[i, j] > 0.98:
                    text += " Pairs above 0.98 are very hard to separate."
            lines.append(text)
        lines += [f"<span style='color:#C2412D'>{m}</span>" for m in errors]
        lines += [f"<span style='color:#6A7187'>{m}</span>" for level, m in self.problems if level == "warning"]
        self.status.setText("<br>".join(lines))

    def _draw_signatures(self):
        um = self.unmixer
        for i, ref in enumerate(um.references):
            color = (90, 90, 90) if ref.is_autofluorescence else PALETTE[i % len(PALETTE)]
            style = QtCore.Qt.DashLine if ref.is_autofluorescence else QtCore.Qt.SolidLine
            self.sig_plot.plot(ref.signature, pen=pg.mkPen(color, width=2, style=style), name=ref.name)
        ticks = [(i, d.replace("-A", "")) for i, d in enumerate(um.detectors)]
        self.sig_plot.getAxis("bottom").setTicks([ticks[::2], ticks[1::2]])
        self.sig_plot.setLabel("bottom", "Detector")

    def _draw_similarity(self):
        um = self.unmixer
        sim = spectral.similarity_matrix(um.matrix)
        n = len(um.names)
        self.sim_table.setRowCount(n)
        self.sim_table.setColumnCount(n)
        self.sim_table.setHorizontalHeaderLabels(um.names)
        self.sim_table.setVerticalHeaderLabels(um.names)
        for i in range(n):
            for j in range(n):
                item = QtWidgets.QTableWidgetItem(f"{sim[i, j]:.2f}")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                if i != j:
                    item.setBackground(QtGui.QColor.fromHsvF(0.0, min(max((sim[i, j] - 0.5) * 2, 0), 1), 1.0))
                self.sim_table.setItem(i, j, item)
        self.sim_table.resizeColumnsToContents()

    def _show_control(self):
        self.ctrl_plot.clear()
        rows = self._selected_rows()
        if len(rows) != 1 or self.unmixer is None:
            return
        spec = self.specs[rows[0]]
        result = next((r for r in self.results if r.spec.path == spec.path), None)
        if result is None:
            self.ctrl_hint.setText(f"{spec.path.name} is not a single-color control.")
            return
        peak = self.unmixer.detectors.index(result.peak_detector)
        events = unmixing._detector_events(self.samples[spec.path], self.unmixer.detectors)
        values = _asinh(events[:, peak])
        counts, edges = np.histogram(values, bins=128)
        self.ctrl_plot.plot(edges, counts, stepMode="center", fillLevel=0,
                            brush=(91, 91, 214, 110), pen=pg.mkPen((60, 60, 160)))
        pos_vals = values[result.positive_mask]
        region = pg.LinearRegionItem((float(pos_vals.min()), float(pos_vals.max())), brush=(20, 163, 163, 50))
        self.ctrl_plot.addItem(region)
        self.ctrl_plot.setLabel("bottom", f"{result.peak_detector} (asinh scale)")
        self.ctrl_hint.setText(
            f"{spec.name}: {result.n_positive:,} positive events; negative from "
            f"{result.negative_source}. Drag the shaded region to change the positives.")

        def changed():
            lo, hi = region.getRegion()
            spec.positive_mask = (values >= lo) & (values <= hi)
            if spec.positive_mask.sum() < 10:
                spec.positive_mask = None
            self.recalculate()

        region.sigRegionChangeFinished.connect(changed)

    # ----- preview ---------------------------------------------------------

    def _fill_preview(self):
        keep = self.preview_sample.currentData()
        self.preview_sample.blockSignals(True)
        self.preview_sample.clear()
        for s in self.specs:
            if s.role == "sample":
                self.preview_sample.addItem(s.path.name, str(s.path))
        i = self.preview_sample.findData(keep)
        self.preview_sample.setCurrentIndex(i if i >= 0 else 0)
        self.preview_sample.blockSignals(False)
        names = self.unmixer.names if self.unmixer else []
        for combo, default in ((self.preview_x, 0), (self.preview_y, 1)):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            i = combo.findText(keep)
            combo.setCurrentIndex(i if i >= 0 else min(default, len(names) - 1))
            combo.blockSignals(False)
        self._draw_preview()

    def _draw_preview(self):
        self.preview_plot.clear()
        self.resid_plot.clear()
        path = self.preview_sample.currentData()
        if self.unmixer is None or path is None:
            return
        sample = self.samples[Path(path)]
        key = (path, tuple(self.unmixer.detectors))
        if key not in self._preview_cache:
            try:
                idx = unmixing._detector_index(sample, self.unmixer.detectors)
            except ValueError:
                return
            raw = sample.get_events(source="raw")
            rows = np.random.default_rng(0).choice(len(raw), min(len(raw), PREVIEW_EVENTS), replace=False)
            self._preview_cache[key] = raw[np.sort(rows)][:, idx]
        Y = self._preview_cache[key]
        A = self.unmixer.unmix(Y)
        xi, yi = self.preview_x.currentIndex(), self.preview_y.currentIndex()
        if xi < 0 or yi < 0:
            return
        x, y = _asinh(A[:, xi]), _asinh(A[:, yi])
        lo = min(np.percentile(x, 0.5), np.percentile(y, 0.5), -1)
        hi = max(np.percentile(x, 99.9), np.percentile(y, 99.9))
        h, _, _ = np.histogram2d(x, y, bins=BINS, range=[[lo, hi], [lo, hi]])
        density = np.log1p(h)
        rgba = PSEUDOCOLOR.map(density / (density.max() or 1), mode="byte")
        rgba[h == 0] = (0, 0, 0, 0)
        img = pg.ImageItem(rgba)
        img.setRect(QtCore.QRectF(lo, lo, hi - lo, hi - lo))
        self.preview_plot.addItem(img)
        self.preview_plot.setLabel("bottom", f"{self.unmixer.names[xi]} (asinh)")
        self.preview_plot.setLabel("left", f"{self.unmixer.names[yi]} (asinh)")
        R = self.unmixer.residuals(Y, A)
        rms = np.sqrt((R ** 2).mean(axis=0))
        self.resid_plot.addItem(pg.BarGraphItem(x=np.arange(len(rms)), height=rms, width=0.8,
                                                brush=(242, 109, 91, 170)))
        ticks = [(i, d.replace("-A", "")) for i, d in enumerate(self.unmixer.detectors)]
        self.resid_plot.getAxis("bottom").setTicks([ticks[::3], []])

    # ----- applying --------------------------------------------------------

    def unmix_into_workspace(self):
        """Add the unmixed samples and references to the workspace; raw files stay here."""
        if self.unmixer is None:
            return []
        name = self.setup.name
        um_name, n = name, 2
        while um_name in self.ws.unmixers:
            um_name, n = f"{name} ({n})", n + 1
        self.ws.add_unmixer(um_name, self.unmixer)
        new_ids = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for spec in self.specs:
                if spec.role == "exclude":
                    continue
                group = "Unmixed" if spec.role == "sample" else experiment.REFERENCE_GROUP
                new_ids.append(self.ws.unmix_file(spec.path, um_name, group, self.samples[spec.path]))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.store()
        self.unmixed.emit(new_ids)
        return new_ids

    def export(self, dest=None, name=None, as_zip=None):
        """Ask where to write the experiment (unless told), then write it."""
        if self.unmixer is None:
            return None
        if dest is None:
            dlg = ExportDialog(self.setup.name if not self.setup.name.startswith("Experiment ")
                               else experiment.default_name(self.specs, self.samples), self)
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return None
            dest, name, as_zip = dlg.values()
        progress = QtWidgets.QProgressDialog("Exporting…", "", 0, 1, self)
        progress.setWindowTitle("Export experiment")
        progress.setCancelButton(None)
        progress.setMinimumDuration(300)
        progress.setWindowModality(QtCore.Qt.WindowModal)

        def step(done, total, text):
            progress.setMaximum(total + 1)
            progress.setValue(done)
            progress.setLabelText(text)
            QtWidgets.QApplication.processEvents()

        try:
            result = experiment.export_experiment(dest, name, self.specs, self.unmixer,
                                                  as_zip=as_zip, samples=self.samples,
                                                  unmixer_name=self.setup.name, progress=step)
        except Exception as exc:
            progress.close()
            QtWidgets.QMessageBox.critical(self, "Export experiment", f"The export failed:\n\n{exc}")
            return None
        progress.close()
        msg = (f"Wrote {result.n_references} reference and {result.n_samples} sample file(s), raw and "
               f"unmixed, to\n{result.path}")
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, "Experiment exported", msg,
                                    QtWidgets.QMessageBox.Ok, self)
        show = box.addButton("Show in folder", QtWidgets.QMessageBox.ActionRole)
        box.setModal(False)
        box.buttonClicked.connect(lambda b: b is show and QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(result.path.parent))))
        box.show()
        return result


class ExportDialog(QtWidgets.QDialog):
    """Name, destination folder and zip-or-folder for an exported experiment."""

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export experiment")
        self.name = QtWidgets.QLineEdit(name)
        self.folder = QtWidgets.QLineEdit(str(Path.home()))
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.as_zip = QtWidgets.QRadioButton("Zipped folder (.zip)")
        self.as_zip.setChecked(True)
        plain = QtWidgets.QRadioButton("Plain folder")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        fmt = QtWidgets.QHBoxLayout()
        fmt.addWidget(self.as_zip)
        fmt.addWidget(plain)
        fmt.addStretch()
        layout_hint = QtWidgets.QLabel(
            "Raw/ and Unmixed/, each with Reference Group/ and Samples/, plus the unmixing matrix, "
            "a manifest, a QC report and a Sidescatter workspace.")
        layout_hint.setWordWrap(True)
        layout_hint.setObjectName("Hint")
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form = QtWidgets.QFormLayout(self)
        form.addRow("Experiment name", self.name)
        form.addRow("Save in", row)
        form.addRow("Format", fmt)
        form.addRow(layout_hint)
        form.addRow(buttons)
        self.setMinimumWidth(520)

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Save the experiment in", self.folder.text())
        if d:
            self.folder.setText(d)

    def values(self):
        return Path(self.folder.text()), self.name.text().strip() or "Experiment", self.as_zip.isChecked()
