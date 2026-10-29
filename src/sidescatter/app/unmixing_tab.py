"""The Unmixing tab: raw files in, classified, unmixed, and packaged like SpectroFlo.

Three steps, as on the instrument. 1 Select controls: add or drop every raw
file of an experiment; each one is classified as an unstained or single-color
reference (on beads or cells), a multicolor sample, or left out, and the table
shows why. 2 Identify positives: one row per control with its scatter gate,
the Positive and Negative ranges on its peak detector, and its spectrum.
3 Check and unmix: signatures, similarity and a live preview, then unmix into
the workspace or export a zipped experiment with Raw and Unmixed folders.
"""

import copy
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import (autospectral_io, cleaning, control_cleanup, experiment, fcs, fluor_variants, spectra_library,
                   spectral, unmixing)
from ..core.experiment import Setup
from ..core.workspace import default_tops
from . import theme
from .control_rows import ControlRow
from .jobs import manager as jobs, run_job
from .graph_view import BINS, CLOSE_PEN, DRAW_PEN, GATE_PEN, PSEUDOCOLOR, SNAP_FILL, SNAP_RADIUS_PX
from .spectrum_plot import SignaturesView
from . import qc_report

COLS = ["File", "Role", "Fluor", "Marker", "Negative / AF", "Peak", "Positives", "Why"]
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
STEPS = ("Select controls", "Identify positives", "Check and unmix")


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
        self._control_qc = {}      # (path, settings) -> (keep mask, cleaning.Result)
        self.library = spectra_library.SpectraLibrary()   # spectra you saved from good controls
        self.rows = []
        self._rows_stale = True
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
        self.af_mode = QtWidgets.QComboBox()
        self.af_mode.addItem("Per-cell matching", "per_cell")
        self.af_mode.addItem("AF signatures as fluors", "fluors")
        self.af_mode.setToolTip(
            "Per-cell matching: every cell gets the one AF spectrum from a library that fits it best "
            "(best for tissues with mixed AF). As fluors: a few AF signatures are unmixed like fluors, "
            "as SpectroFlo does.")
        self.af_mode.currentIndexChanged.connect(self._settings_changed)
        self.af_count = QtWidgets.QSpinBox()
        self.af_count.setRange(1, 6)
        self.af_count.setPrefix("Signatures: ")
        self.af_count.valueChanged.connect(self._settings_changed)
        self.clean_check = QtWidgets.QCheckBox("Clean controls first")
        self.clean_check.setToolTip(
            "Remove margin events, clogs and bubbles from each control before its signature is "
            "calculated (uses the Clean data settings). A clog in a control skews its spectrum.")
        self.clean_check.toggled.connect(self._settings_changed)
        self.smart_check = QtWidgets.QCheckBox("Smart control clean-up")
        self.smart_check.setToolTip(
            "Drop saturated events and doublets, use only the brightest positive events, subtract "
            "scatter-matched unstained cells from cell controls, and check every signature against "
            "a reference spectrum. Based on AutoSpectral (Burton et al. 2025).")
        self.smart_check.toggled.connect(self._settings_changed)
        self.variation_check = QtWidgets.QCheckBox("Per-cell fluor variation")
        self.variation_check.setToolTip(
            "Let each cell use a slightly shifted version of a fluor's spectrum, picked from the "
            "variation seen in that fluor's control. Cuts spillover spread, most in big panels (25+ "
            "colours), but unmixing is slower. Based on AutoSpectral (Burton et al. 2025).")
        self.variation_check.toggled.connect(self._settings_changed)
        self.smart_btn = QtWidgets.QPushButton("Settings…")
        self.smart_btn.setAutoDefault(False)
        self.smart_btn.clicked.connect(self._cleanup_settings)
        self.af_lib_size = QtWidgets.QSpinBox()
        self.af_lib_size.setRange(1, 60)
        self.af_lib_size.setValue(30)
        self.af_lib_size.setPrefix("Library: up to ")
        self.af_lib_size.setSuffix(" spectra")
        self.af_lib_size.setToolTip("Most automatic AF spectra per unstained file; near-duplicates "
                                    "(cosine similarity above 0.99) are dropped.")
        self.af_lib_size.valueChanged.connect(self._settings_changed)
        self.af_auto = QtWidgets.QCheckBox("Automatic spectra")
        self.af_auto.setChecked(True)
        self.af_auto.setToolTip("Untick to use only the AF populations you draw in the AF library tab.")
        self.af_auto.toggled.connect(self._settings_changed)
        options = QtWidgets.QFormLayout()
        options.addRow("Method", self.method)
        af_row = QtWidgets.QHBoxLayout()
        af_row.addWidget(self.af_check)
        af_row.addWidget(self.af_mode)
        af_row.addWidget(self.af_count)
        af_row.addWidget(self.af_lib_size)
        af_row.addWidget(self.af_auto)
        af_row.addStretch()
        options.addRow(af_row)
        clean_row = QtWidgets.QHBoxLayout()
        clean_row.addWidget(self.clean_check)
        clean_row.addSpacing(16)
        clean_row.addWidget(self.smart_check)
        clean_row.addWidget(self.smart_btn)
        clean_row.addSpacing(16)
        clean_row.addWidget(self.variation_check)
        clean_row.addSpacing(16)
        self.autospectral_btn = QtWidgets.QToolButton()
        self.autospectral_btn.setText("AutoSpectral files")
        self.autospectral_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(self.autospectral_btn)
        menu.addAction("Import spectra…", self.import_spectra)
        self.stop_import = menu.addAction("Stop using imported spectra", self.clear_imported)
        menu.addSeparator()
        self.export_as = menu.addAction("Export spectra and control file…", self.export_autospectral)
        self.autospectral_btn.setMenu(menu)
        self.autospectral_btn.setToolTip(
            "Exchange spectra with AutoSpectral (R): unmix with spectra it made, or write this "
            "experiment's spectra and an AutoSpectral control file to compare the two")
        clean_row.addWidget(self.autospectral_btn)
        clean_row.addStretch()
        options.addRow(clean_row)

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.RichText)

        select_page = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(select_page)
        left.setContentsMargins(0, 0, 0, 0)
        buttons = QtWidgets.QHBoxLayout()
        for b in (add_btn, folder_btn, remove_btn):
            buttons.addWidget(b)
        buttons.addStretch()
        left.addLayout(buttons)
        left.addWidget(self.drop_hint)
        left.addWidget(self.table, 1)
        left.addLayout(options)
        left.addWidget(self.status)

        # ----- step 2: one row per reference control
        identify_page = QtWidgets.QWidget()
        il = QtWidgets.QVBoxLayout(identify_page)
        il.setContentsMargins(0, 0, 0, 0)
        self.identify_hint = QtWidgets.QLabel()
        self.identify_hint.setWordWrap(True)
        self.identify_hint.setObjectName("Hint")
        self.rows_box = QtWidgets.QWidget()
        self.rows_layout = QtWidgets.QVBoxLayout(self.rows_box)
        self.rows_layout.setContentsMargins(0, 0, 6, 0)
        self.rows_layout.setSpacing(10)
        self.rows_layout.addStretch()
        rows_scroll = QtWidgets.QScrollArea()
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        rows_scroll.setWidget(self.rows_box)
        il.addWidget(self.identify_hint)
        il.addWidget(rows_scroll, 1)

        # ----- step 3: signatures, similarity, live preview
        self.tabs = QtWidgets.QTabWidget()
        self.signatures = SignaturesView()
        self.tabs.addTab(self.signatures, "Signatures")

        self.sim_table = QtWidgets.QTableWidget()
        self.tabs.addTab(self.sim_table, "Similarity")

        hot = QtWidgets.QWidget()
        hl = QtWidgets.QVBoxLayout(hot)
        hl.setContentsMargins(4, 6, 4, 4)
        self.hot_hint = QtWidgets.QLabel()
        self.hot_hint.setWordWrap(True)
        self.hot_hint.setObjectName("Hint")
        self.hot_table = QtWidgets.QTableWidget()
        hl.addWidget(self.hot_hint)
        hl.addWidget(self.hot_table, 1)
        self.tabs.addTab(hot, "Hotspots")

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

        self.af_view = AFLibraryView()
        self.af_view.gates_changed.connect(self._af_gates_changed)
        self.tabs.addTab(self.af_view, "AF library")
        self.af_check_view = AFCheckView()
        self.tabs.addTab(self.af_check_view, "AF check")
        self._af_check = None
        self._af_check_for = None   # the unmixer the AF check job is working on
        self.loading = None         # the job reading raw files, if any
        self.tabs.currentChanged.connect(self._af_tab_shown)

        self.pages = QtWidgets.QStackedWidget()
        for page in (select_page, identify_page, self.tabs):
            self.pages.addWidget(page)

        # ----- step bar
        step_bar = QtWidgets.QWidget()
        step_bar.setObjectName("StepBar")
        step_bar.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        sl = QtWidgets.QHBoxLayout(step_bar)
        sl.setContentsMargins(4, 4, 4, 4)
        sl.setSpacing(2)
        self.step_group = QtWidgets.QButtonGroup(self)
        self.step_buttons = []
        for i, label in enumerate(STEPS):
            if i:
                arrow = QtWidgets.QLabel("›")
                arrow.setObjectName("StepArrow")
                sl.addWidget(arrow)
            b = QtWidgets.QToolButton()
            b.setObjectName("StepButton")
            b.setText(f"{i + 1}   {label}")
            b.setCheckable(True)
            self.step_group.addButton(b, i)
            self.step_buttons.append(b)
            sl.addWidget(b)
        self.step_buttons[0].setChecked(True)
        self.step_group.idClicked.connect(self.go_to_step)

        # ----- actions
        self.unmix_btn = QtWidgets.QPushButton("Unmix into workspace")
        self.unmix_btn.setProperty("tip", "Add the unmixed samples (and references) to the workspace for gating")
        self.unmix_btn.clicked.connect(self.unmix_into_workspace)
        self.export_btn = QtWidgets.QPushButton("Export experiment…")
        self.export_btn.setProperty("tip", "Write a zipped experiment with Raw and Unmixed folders, like SpectroFlo")
        self.export_btn.clicked.connect(self.export)
        self.export_btn.setDefault(True)
        self.back_btn = QtWidgets.QPushButton("‹  Back")
        self.back_btn.clicked.connect(lambda: self.go_to_step(self.step - 1))
        self.next_btn = QtWidgets.QPushButton("Next  ›")
        self.next_btn.clicked.connect(lambda: self.go_to_step(self.step + 1))

        steps_row = QtWidgets.QHBoxLayout()
        steps_row.addStretch()
        steps_row.addWidget(step_bar)
        steps_row.addStretch()

        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.back_btn)
        bottom.addStretch()
        bottom.addWidget(self.unmix_btn)
        bottom.addWidget(self.export_btn)
        bottom.addWidget(self.next_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.addLayout(top)
        layout.addLayout(steps_row)
        layout.addWidget(self.pages, 1)
        layout.addLayout(bottom)

        self.go_to_step(0)
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
        for w, v in ((self.af_check, self.setup.autofluorescence), (self.af_auto, self.setup.af_auto),
                     (self.clean_check, self.setup.clean_controls),
                     (self.smart_check, self.setup.cleanup.enabled),
                     (self.variation_check, self.setup.variation.enabled),
                     (self.af_lib_size, self.setup.af_library_size),
                     (self.af_mode, self.af_mode.findData(self.setup.af_mode))):
            w.blockSignals(True)
            if isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(v)
            elif isinstance(w, QtWidgets.QComboBox):
                w.setCurrentIndex(max(v, 0))
            else:
                w.setValue(v)
            w.blockSignals(False)
        self._show_af_options()
        self.smart_btn.setEnabled(self.setup.cleanup.enabled or self.setup.variation.enabled)
        setup = self.setup
        to_read = list(dict.fromkeys(s.path for s in setup.specs if s.path not in self.samples))
        if not to_read:
            self._fill_table()
            self.recalculate()
            return

        def work(job):
            loaded = {}
            for i, path in enumerate(to_read):
                job.progress(i / len(to_read), f"Reading {Path(path).name}")
                try:
                    loaded[path] = fcs.load_sample(path)
                except (OSError, ValueError):
                    pass
            return loaded

        def done(loaded):
            self.samples.update(loaded)
            missing = [s for s in setup.specs if s.path not in self.samples]
            if missing:
                gone = {id(s) for s in missing}
                setup.specs = [s for s in setup.specs if id(s) not in gone]
                QtWidgets.QMessageBox.warning(
                    self, "Missing files", "These files could not be found and were left out:\n\n"
                    + "\n".join(str(s.path) for s in missing[:10]))
            if setup is self.setup:
                self._fill_table()
                self.recalculate()

        self.table.setRowCount(0)
        self.status.setText(f"Reading {len(to_read)} file{'s' if len(to_read) > 1 else ''}…")
        self.loading = run_job(f"Reading {setup.name}", work, on_done=done, key=("read-setup", id(self)))

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
        """Add files or folders (searched recursively). They're read in the
        background (self.loading is the job); returns the names skipped because
        they aren't FCS files."""
        found, skipped = fcs.find_fcs_files(paths)
        known = {s.path for s in self.specs}
        new = list(dict.fromkeys(Path(p) for p in found if Path(p) not in known))
        if not new:
            return skipped
        setup, to_read = self.setup, [p for p in new if p not in self.samples]

        def work(job):
            loaded, bad = {}, []
            for i, p in enumerate(to_read):
                job.progress(i / len(to_read), f"Reading {p.name}")
                try:
                    loaded[p] = fcs.load_sample(p)
                except (OSError, ValueError):
                    bad.append(p.name)
            return loaded, bad

        def done(out):
            loaded, bad = out
            self.samples.update(loaded)
            have = {s.path for s in setup.specs}
            ok = [p for p in new if p in self.samples and p not in have]
            setup.specs.extend(unmixing.guess_files(ok, self.samples))
            if setup is self.setup:
                self._fill_table()
                self.recalculate()
            self._mark_changed()
            if bad:
                QtWidgets.QMessageBox.warning(
                    self, "Couldn't read files", f"Couldn't read {len(bad)} file(s):\n\n" + "\n".join(bad[:10]))

        self.loading = run_job(f"Reading {len(new)} file{'s' if len(new) > 1 else ''}", work, on_done=done)
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
            neg = QtWidgets.QComboBox()
            if spec.role == "sample":
                # Samples: which unstained cells' AF library they use (per-cell AF).
                neg.addItem("Shared AF library", "shared")
                for u in unstained:
                    if u.carrier == "cells":
                        neg.addItem(f"AF from {u.path.name}", str(u.path))
                neg.setCurrentIndex(max(neg.findData(spec.af_source), 0))
                neg.setEnabled(self.setup.af_mode == "per_cell" and neg.count() > 2)
                neg.setToolTip("Per-cell autofluorescence: match this sample's cells against the AF "
                               "spectra of every unstained file, or only one (e.g. the same tissue's).")
                neg.currentIndexChanged.connect(lambda _, s=spec, c=neg: self._set_af_source(s, c.currentData()))
            else:
                negs = [("Auto", "auto"), ("Internal", "internal")]
                negs += [(u.path.name, str(u.path)) for u in unstained]
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

    def _set_af_source(self, spec, value):
        spec.af_source = value
        self._draw_preview()
        self._mark_changed()

    def _af_gates_changed(self, spec):
        self.recalculate()
        self._mark_changed()

    def _show_af_options(self):
        on = self.af_check.isChecked()
        per_cell = self.af_mode.currentData() == "per_cell"
        self.af_mode.setEnabled(on)
        self.af_count.setVisible(not per_cell)
        self.af_count.setEnabled(on)
        for w in (self.af_lib_size, self.af_auto):
            w.setVisible(per_cell)
            w.setEnabled(on)

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
        self.setup.clean_controls = self.clean_check.isChecked()
        self.setup.cleanup.enabled = self.smart_check.isChecked()
        self.setup.variation.enabled = self.variation_check.isChecked()
        self.smart_btn.setEnabled(self.setup.cleanup.enabled or self.setup.variation.enabled)
        mode_changed = self.setup.af_mode != self.af_mode.currentData()
        self.setup.af_mode = self.af_mode.currentData()
        self.setup.af_library_size = self.af_lib_size.value()
        self.setup.af_auto = self.af_auto.isChecked()
        self._show_af_options()
        if mode_changed:
            self._fill_table()  # the samples' AF source choice depends on the mode
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

    def control_cleaning(self):
        """{path: (keep mask, cleaning.Result)} for the reference controls, when they are cleaned."""
        if not self.setup.clean_controls:
            return {}
        settings = self.ws.qc_settings
        key = repr(sorted(settings.to_dict().items()))
        out = {}
        for spec in self.specs:
            if spec.is_reference and spec.path in self.samples:
                if (spec.path, key) not in self._control_qc:
                    self._control_qc[(spec.path, key)] = unmixing.control_keep(
                        self.samples[spec.path], cleaning.Settings.from_dict(
                            {**settings.to_dict(), "channels": None}))
                out[spec.path] = self._control_qc[(spec.path, key)]
        return out

    def recalculate(self, rebuild_rows=True):
        """Rebuild the matrix from the current files and gates.

        rebuild_rows False keeps the Identify rows (and whatever is being
        dragged in them) and only redraws their histograms and spectra.
        """
        self.problems = unmixing.check_specs(self.specs) if self.specs else []
        errors = [m for level, m in self.problems if level == "error"]
        self.unmixer, self.results = None, []
        cleaned = {}
        if self.specs and not errors:
            try:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    cleaned = self.control_cleaning()
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
                self.unmixer, self.results = unmixing.build_unmixer(
                    [s for s in self.specs if s.role != "exclude"], method=self.setup.method,
                    n_autofluorescence=self.setup.n_autofluorescence,
                    autofluorescence=self.setup.autofluorescence, samples=self.samples,
                    af_mode=self.setup.af_mode, af_library_size=self.setup.af_library_size,
                    af_auto=self.setup.af_auto, keep={p: k for p, (k, _) in cleaned.items()},
                    cleanup=self.setup.cleanup, library=self.library, variation=self.setup.variation,
                    imported=self.setup.imported)
            except Exception as exc:
                errors.append(f"Couldn't build the unmixing matrix: {exc}")
        self._cleaned = cleaned
        ok = self.unmixer is not None
        self._update_buttons()
        self._update_autospectral()
        self._show_status(errors)
        by_path = {r.spec.path: r for r in self.results}
        self.table.blockSignals(True)
        for row, spec in enumerate(self.specs):
            r = by_path.get(spec.path)
            self.table.item(row, C_PEAK).setText(r.peak_detector if r else "")
            self.table.item(row, C_POS).setText(f"{r.n_positive:,}" if r else "")
        self.table.blockSignals(False)
        self.sim_table.clear()
        self.sim_table.setRowCount(0)
        if ok:
            self.signatures.set_signatures(self.unmixer.references, self.unmixer.detectors,
                                           self._comparison_lines())
            self._draw_similarity()
        else:
            self.signatures.clear()
        self._draw_hotspots()
        self._fill_preview()
        if rebuild_rows:
            self._rows_stale = True
            if self.step == 1:
                self._build_rows()
        else:
            for row in self.rows:
                row.update_result(by_path.get(row.spec.path))
        self._af_check = self._af_check_for = None
        unstained = [s for s in self.specs if s.role == "unstained" and s.carrier == "cells"]
        self.af_view.set_data(unstained, self.samples, self.unmixer)
        self._af_tab_shown()

    def _af_tab_shown(self, *_):
        """The AF comparison is the slow part, so it only runs while its tab is open."""
        if self.tabs.currentWidget() is not self.af_check_view:
            return
        if self._af_check is None and self.unmixer is not None and self.setup.autofluorescence:
            if self._af_check_for is not self.unmixer:
                unmixer, specs = self.unmixer, [s for s in self.specs if s.role != "exclude"]
                samples, settings = dict(self.samples), dict(self.setup.af_settings)
                self._af_check_for = unmixer

                def done(check):
                    if unmixer is self.unmixer:
                        self._af_check = check
                        self._af_tab_shown()

                run_job("Checking autofluorescence",
                        lambda job: unmixing.af_check(specs, unmixer, samples, **settings),
                        on_done=done, key=("af-check", id(self)))
            self.af_check_view.show_check(None, None)
            return
        self.af_check_view.show_check(self._af_check, self.setup.af_mode if self.unmixer else None)

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
            if um.per_cell_af:
                text += f" Per-cell AF library: {len(um.af_library)} spectra."
            if len(um.names) > 1:
                text += f" Most similar pair: {um.names[i]} and {um.names[j]} ({sim[i, j]:.2f})."
                if sim[i, j] > 0.98:
                    text += " Pairs above 0.98 are very hard to separate."
            lines.append(text)
            if self.results and all(r.basic_signature is not None for r in self.results):
                now = np.linalg.cond(np.vstack([r.signature for r in self.results]))
                before = np.linalg.cond(np.vstack([r.basic_signature for r in self.results]))
                lines.append(f"Smart control clean-up: fluor matrix condition number {now:.2f} "
                             f"(plain method {before:.2f}; lower is easier to unmix).")
            if self.setup.imported is not None:
                used = sum(r.signature_source == "imported" for r in self.results)
                extra = len(um.names) - len(self.results) - sum(r.is_autofluorescence for r in um.references)
                text = (f"Imported AutoSpectral spectra ({self.setup.imported.path.name}): used for {used} "
                        f"control(s)")
                if extra > 0:
                    text += f", plus {extra} fluor(s) without a control here"
                if len(self.setup.imported.af) and self.setup.autofluorescence:
                    text += f", and its {len(self.setup.imported.af)} AF spectra"
                lines.append(text + ".")
            flagged = [r.spec.name for r in self.results if r.flagged]
            if flagged:
                lines.append(f"<span style='color:#C2412D'>Spectrum check failed for {', '.join(flagged)}: "
                             "see step 2.</span>")
        cleaned = getattr(self, "_cleaned", {})
        if cleaned:
            parts = [(p.name, 100 * (1 - k.mean())) for p, (k, _) in cleaned.items() if len(k)]
            worst = sorted(parts, key=lambda x: -x[1])[:3]
            text = f"Cleaned {len(parts)} control(s)"
            if worst and worst[0][1] > 0:
                text += ": most removed from " + ", ".join(f"{n} ({pct:.1f}%)" for n, pct in worst if pct > 0)
            lines.append(text + ".")
        lines += [f"<span style='color:#C2412D'>{m}</span>" for m in errors]
        lines += [f"<span style='color:#6A7187'>{m}</span>" for level, m in self.problems if level == "warning"]
        self.status.setText("<br>".join(lines))

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

    def _draw_hotspots(self):
        """Error hotspot matrix: spread each fluor's variation is expected to cause in the others."""
        self.hot_table.clear()
        self.hot_table.setRowCount(0)
        self.hot_table.setColumnCount(0)
        hs = unmixing.variant_hotspots(self.unmixer, self.results) if self.unmixer is not None else None
        if hs is None:
            self.hot_hint.setText("Turn on Smart control clean-up or Per-cell fluor variation in step 1 "
                                  "to see which fluors' variation spreads into which.")
            return
        names, H = hs
        self.hot_hint.setText(
            "Expected spread in each column's fluor per unit of the row's fluor, from the variation "
            "in the row fluor's control (darker = more). Hotspots are the pairs to watch; per-cell "
            "fluor variation " + ("is on to reduce them." if self.unmixer.per_cell_variants
                                  else "(step 1) reduces them."))
        n = len(names)
        self.hot_table.setRowCount(n)
        self.hot_table.setColumnCount(n)
        self.hot_table.setHorizontalHeaderLabels(names)
        self.hot_table.setVerticalHeaderLabels(names)
        top = np.nanmax(H) if np.isfinite(H).any() else 1.0
        for i in range(n):
            for j in range(n):
                v = H[i, j]
                item = QtWidgets.QTableWidgetItem("" if not np.isfinite(v) else f"{v:.3f}")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                if np.isfinite(v) and top > 0:
                    item.setBackground(QtGui.QColor.fromHsvF(0.0, min(v / top, 1.0), 1.0))
                self.hot_table.setItem(i, j, item)
        self.hot_table.resizeColumnsToContents()

    # ----- steps ---------------------------------------------------------

    @property
    def step(self):
        return self.pages.currentIndex()

    def go_to_step(self, index):
        index = max(0, min(index, len(STEPS) - 1))
        self.pages.setCurrentIndex(index)
        self.step_buttons[index].setChecked(True)
        self.back_btn.setEnabled(index > 0)
        self.next_btn.setVisible(index < len(STEPS) - 1)
        self.next_btn.setDefault(index < len(STEPS) - 1)
        self.export_btn.setDefault(index == len(STEPS) - 1)
        if index == 1 and self._rows_stale:
            self._build_rows()

    def _negative_for(self, spec):
        """(spec, sample) of the file a single stain takes its negatives from, if any."""
        other = None
        if spec.negative == "auto":
            other = unmixing._matching_unstained(spec, self.specs)
        elif spec.negative != "internal":
            other = next((s for s in self.specs if str(s.path) == spec.negative), None)
        if other is None or other.path not in self.samples:
            return None
        return other, self.samples[other.path]

    def _build_rows(self):
        """One row per reference control, single stains first, then unstained."""
        for row in self.rows:
            row.setParent(None)
            row.deleteLater()
        self.rows = []
        self._rows_stale = False
        if self.unmixer is None:
            self.identify_hint.setText(
                "Add single-color controls and fix the problems listed in step 1 first; "
                "the controls appear here once a matrix can be built.")
            return
        self.identify_hint.setText(
            "For each control, draw the scatter gate around the beads or cells, then drag the "
            "Positive and Negative ranges on its peak detector. The spectrum shows the positive "
            "events in every detector and updates as you go.")
        by_path = {r.spec.path: r for r in self.results}
        refs = [s for s in self.specs if s.role == "single_stain"] + [s for s in self.specs if s.role == "unstained"]
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for spec in refs:
                try:
                    row = ControlRow(spec, self.samples[spec.path], self.unmixer.detectors,
                                     by_path.get(spec.path), self._negative_for(spec))
                except ValueError:
                    continue   # detectors don't match the experiment; check_specs reports it
                row.edited.connect(self._gate_edited)
                row.save_requested.connect(self._save_to_library)
                self.rows_layout.insertWidget(self.rows_layout.count() - 1, row)
                self.rows.append(row)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _gate_edited(self, spec):
        self.recalculate(rebuild_rows=False)
        self._mark_changed()

    # ----- smart control clean-up and the spectra check ---------------------

    def _cleanup_settings(self):
        dialog = CleanupDialog(self.setup.cleanup, self, self.setup.variation)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.setup.cleanup = dialog.values()
            self.setup.variation = dialog.variation(self.setup.variation.enabled)
            self.recalculate()
            self._mark_changed()

    def _save_to_library(self, spec):
        r = next((r for r in self.results if r.spec is spec), None)
        if r is None or self.unmixer is None:
            return
        signature = r.measured_signature if r.measured_signature is not None else r.signature
        self.library.add(spec.name, self.unmixer.detectors, signature, spec.carrier, spec.path.name)
        try:
            self.library.save()
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Couldn't save the spectrum",
                                          f"{self.library.path}\n\n{exc}")
            return
        if spec.signature_choice != "auto":
            spec.signature_choice = "auto"
            self._mark_changed()
        self.recalculate(rebuild_rows=False)

    def import_spectra(self, path=None):
        """Unmix with spectra from an AutoSpectral CSV (its table_spectra folder)."""
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Import AutoSpectral spectra", str(Path.home()), "CSV files (*.csv)")
            if not path:
                return False
        try:
            spectra = autospectral_io.read_spectra(path)
            if self.unmixer is not None:
                spectra.on(self.unmixer.detectors)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Couldn't import the spectra", str(exc))
            return False
        self.setup.imported = spectra
        self._update_autospectral()
        self.recalculate()
        self._mark_changed()
        return True

    def clear_imported(self):
        self.setup.imported = None
        self._update_autospectral()
        self.recalculate()
        self._mark_changed()

    def export_autospectral(self, folder=None):
        """Write the spectra and an AutoSpectral control file for these controls."""
        if self.unmixer is None:
            return None
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Folder for the AutoSpectral files", str(Path.home()))
            if not folder:
                return None
        try:
            written = autospectral_io.export(folder, self.specs, self.unmixer, self.results)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Couldn't write the files", str(exc))
            return None
        if not getattr(self, "_quiet", False):
            QtWidgets.QMessageBox.information(
                self, "AutoSpectral files written",
                "Wrote " + ", ".join(p.name for p in written) + f" to\n{folder}\n\n"
                "The control file names your control files as they are on disk; point AutoSpectral's "
                "control.dir at the folder that holds them.")
        return written

    def _update_autospectral(self):
        self.stop_import.setEnabled(self.setup is not None and self.setup.imported is not None)
        self.export_as.setEnabled(self.unmixer is not None)

    def _comparison_lines(self):
        """Per fluor, the lines the Signatures tab draws behind it: the signature
        before control clean-up and the reference it was checked against."""
        out = {}
        before = pg.mkPen(theme.FAINT, width=1.6, style=QtCore.Qt.DashLine)
        reference = pg.mkPen("#C2412D", width=1.6, style=QtCore.Qt.DotLine)
        for r in self.results:
            lines = []
            if r.basic_signature is not None:
                lines.append((r.basic_signature, before))
            if r.check is not None:
                lines.append((spectral.normalise(r.check.reference), reference))
            if lines:
                out[r.spec.name] = lines
        return out

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
        names = self.unmixer.output_names if self.unmixer else []
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
        spec = next((s for s in self.specs if str(s.path) == path), None)
        af_source = spec.af_source if spec is not None and spec.af_source != "shared" else None
        A = self.unmixer.unmix(Y, af_source)
        xi, yi = self.preview_x.currentIndex(), self.preview_y.currentIndex()
        if xi < 0 or yi < 0:
            return
        names = self.unmixer.output_names
        x, y = (A[:, i] if names[i] == "AF Index" else _asinh(A[:, i]) for i in (xi, yi))
        lo = min(np.percentile(x, 0.5), np.percentile(y, 0.5), -1)
        hi = max(np.percentile(x, 99.9), np.percentile(y, 99.9))
        h, _, _ = np.histogram2d(x, y, bins=BINS, range=[[lo, hi], [lo, hi]])
        density = np.log1p(h)
        rgba = PSEUDOCOLOR.map(density / (density.max() or 1), mode="byte")
        rgba[h == 0] = (0, 0, 0, 0)
        img = pg.ImageItem(rgba)
        img.setRect(QtCore.QRectF(lo, lo, hi - lo, hi - lo))
        self.preview_plot.addItem(img)
        self.preview_plot.setLabel("bottom", names[xi] if names[xi] == "AF Index" else f"{names[xi]} (asinh)")
        self.preview_plot.setLabel("left", names[yi] if names[yi] == "AF Index" else f"{names[yi]} (asinh)")
        R = self.unmixer.residuals(Y, A)
        rms = np.sqrt((R ** 2).mean(axis=0))
        self.resid_plot.addItem(pg.BarGraphItem(x=np.arange(len(rms)), height=rms, width=0.8,
                                                brush=(242, 109, 91, 170)))
        self.resid_plot.getAxis("bottom").setTicks([_detector_ticks(self.unmixer.detectors), []])

    # ----- applying --------------------------------------------------------

    def _update_buttons(self):
        ok = self.unmixer is not None
        self.unmix_btn.setEnabled(ok and not jobs().busy(key=("unmix", id(self), self.setup.name)))
        self.export_btn.setEnabled(ok and not jobs().busy(key=("export-experiment", id(self))))
        for b, what in ((self.unmix_btn, "unmixing"), (self.export_btn, "exporting")):
            busy = ok and not b.isEnabled()
            b.setToolTip(f"Still {what} this experiment; see the status bar" if busy else b.property("tip"))

    def unmix_into_workspace(self):
        """Unmix every file in the background, then add the unmixed samples and
        references to the workspace; raw files stay here. Returns the job."""
        if self.unmixer is None:
            return None
        key = ("unmix", id(self), self.setup.name)
        if jobs().busy(key=key):
            return None
        ws, setup, unmixer = self.ws, self.setup, self.unmixer
        todo = [s for s in self.specs if s.role != "exclude"]
        raws = {s.path: self.samples[s.path] for s in todo}

        def work(job):
            from ..core.unmixing import unmix_sample

            out = []
            for i, spec in enumerate(todo):
                text = f"Unmixing {spec.path.name} ({i + 1} of {len(todo)})"

                def step(done, total, i=i, text=text):
                    job.progress((i + done / max(total, 1)) / len(todo), text)

                step(0, 1)
                group = "Unmixed" if spec.role == "sample" else experiment.REFERENCE_GROUP
                af_source = spec.af_source if spec.role == "sample" else None
                unmixed = unmix_sample(raws[spec.path], unmixer, af_source=af_source, progress=step)
                out.append((spec.path, unmixed, group, af_source, default_tops(unmixed)))
            return out

        def done(out):
            if ws is not self.ws:
                return
            name = setup.name
            um_name, n = name, 2
            while um_name in ws.unmixers:
                um_name, n = f"{name} ({n})", n + 1
            ws.add_unmixer(um_name, unmixer)
            new_ids = [ws.add_unmixed(path, um_name, unmixed, group, af_source, tops)
                       for path, unmixed, group, af_source, tops in out]
            self.store()
            self.unmixed.emit(new_ids)

        job = run_job(f"Unmixing {len(todo)} file{'s' if len(todo) > 1 else ''}", work, on_done=done, key=key)
        job.finished.connect(self._update_buttons)
        self._update_buttons()
        return job

    def export(self, dest=None, name=None, as_zip=None, clean=None):
        """Ask where to write the experiment (unless told), then write it in the
        background. Cancel deletes what was written so far. Returns the job.

        clean: also clean the samples and add the data-cleaning PDF (default:
        when anything in the workspace has been cleaned).
        """
        if self.unmixer is None:
            return None
        if dest is None:
            dlg = ExportDialog(self.setup.name if not self.setup.name.startswith("Experiment ")
                               else experiment.default_name(self.specs, self.samples), self,
                               clean=bool(self.ws.qc))
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return None
            dest, name, as_zip = dlg.values()
            clean = dlg.clean.isChecked()
        elif clean is None:
            clean = bool(self.ws.qc)
        specs, unmixer, samples = list(self.specs), self.unmixer, dict(self.samples)
        results = list(self.results)
        um_name, af_settings = self.setup.name, dict(self.setup.af_settings)
        clean_settings = copy.deepcopy(self.ws.qc_settings) if clean else None

        def work(job):
            return experiment.export_experiment(
                dest, name, specs, unmixer, as_zip=as_zip, samples=samples, unmixer_name=um_name,
                progress=lambda done, total, text: job.progress(done / (total + 1), text),
                af_settings=af_settings, clean_settings=clean_settings,
                cleaning_report=qc_report.write_pdf, control_results=results)

        job = run_job(f"Exporting {name}", work, on_done=self._exported, key=("export-experiment", id(self)))
        job.finished.connect(self._update_buttons)
        self._update_buttons()
        return job

    def _exported(self, result):
        msg = (f"Wrote {result.n_references} reference and {result.n_samples} sample file(s), raw and "
               f"unmixed, to\n{result.path}")
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, "Experiment exported", msg,
                                    QtWidgets.QMessageBox.Ok, self)
        show = box.addButton("Show in folder", QtWidgets.QMessageBox.ActionRole)
        box.setModal(False)
        box.buttonClicked.connect(lambda b: b is show and QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(result.path.parent))))
        box.show()


class CleanupDialog(QtWidgets.QDialog):
    """Settings of the smart control clean-up."""

    def __init__(self, settings, parent=None, variation=None):
        super().__init__(parent)
        self.setWindowTitle("Smart control clean-up")
        self.saturation = QtWidgets.QCheckBox("Drop events at a detector's full scale (saturated)")
        self.saturation.setChecked(settings.saturation)
        self.singlets = QtWidgets.QCheckBox("Drop doublets (high FSC-A / FSC-H)")
        self.singlets.setChecked(settings.singlets)
        self.brightest = QtWidgets.QSpinBox()
        self.brightest.setRange(0, 1_000_000)
        self.brightest.setSingleStep(100)
        self.brightest.setValue(settings.brightest)
        self.brightest.setSpecialValueText("All positive events")
        self.brightest.setSuffix(" brightest positive events")
        self.scatter = QtWidgets.QCheckBox("Scatter-matched negatives for cell controls")
        self.scatter.setChecked(settings.scatter_match)
        self.neighbours = QtWidgets.QSpinBox()
        self.neighbours.setRange(1, 50)
        self.neighbours.setValue(settings.neighbours)
        self.neighbours.setSuffix(" nearest unstained cells")
        self.scatter.toggled.connect(self.neighbours.setEnabled)
        self.neighbours.setEnabled(settings.scatter_match)
        variation = variation or fluor_variants.Settings()
        self.max_variants = QtWidgets.QSpinBox()
        self.max_variants.setRange(4, 400)
        self.max_variants.setValue(variation.max_variants)
        self.max_variants.setSuffix(" variants per fluor")
        self.min_cosine = QtWidgets.QDoubleSpinBox()
        self.min_cosine.setRange(0.8, 0.9999)
        self.min_cosine.setDecimals(3)
        self.min_cosine.setSingleStep(0.005)
        self.min_cosine.setValue(variation.min_cosine)
        self.variant_channels = QtWidgets.QCheckBox("Add a \"<fluor> Variant\" channel per fluor (for checking)")
        self.variant_channels.setChecked(variation.channels)
        note = QtWidgets.QLabel(
            "Each positive cell's background is the average of the unstained cells closest to it in "
            "forward and side scatter, so bigger, brighter-AF cells are compared with cells like "
            "them. Beads keep the median of their unstained beads. Based on AutoSpectral "
            "(Burton et al. 2025, bioRxiv 10.1101/2025.10.27.684855).")
        note.setWordWrap(True)
        note.setObjectName("Hint")
        form = QtWidgets.QFormLayout()
        form.addRow(self.saturation)
        form.addRow(self.singlets)
        form.addRow("Use", self.brightest)
        form.addRow(self.scatter)
        form.addRow("Average of", self.neighbours)
        title = QtWidgets.QLabel("Per-cell fluor variation")
        title.setObjectName("ControlRowTitle")
        form.addRow(title)
        form.addRow("Up to", self.max_variants)
        form.addRow("Keep variants with cosine ≥", self.min_cosine)
        form.addRow(self.variant_channels)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                                             | QtWidgets.QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(self._defaults)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def _defaults(self):
        d = control_cleanup.Settings()
        self.saturation.setChecked(d.saturation)
        self.singlets.setChecked(d.singlets)
        self.brightest.setValue(d.brightest)
        self.scatter.setChecked(d.scatter_match)
        self.neighbours.setValue(d.neighbours)
        v = fluor_variants.Settings()
        self.max_variants.setValue(v.max_variants)
        self.min_cosine.setValue(v.min_cosine)
        self.variant_channels.setChecked(v.channels)

    def variation(self, enabled):
        return fluor_variants.Settings(enabled, self.max_variants.value(), self.min_cosine.value(),
                                       self.variant_channels.isChecked())

    def values(self):
        return control_cleanup.Settings(True, self.saturation.isChecked(), self.singlets.isChecked(),
                                        self.brightest.value(), self.scatter.isChecked(),
                                        self.neighbours.value())


class ExportDialog(QtWidgets.QDialog):
    """Name, destination folder and zip-or-folder for an exported experiment."""

    def __init__(self, name, parent=None, clean=False):
        super().__init__(parent)
        self.setWindowTitle("Export experiment")
        self.clean = QtWidgets.QCheckBox("Clean the samples and add a data-cleaning report (QC/*.pdf)")
        self.clean.setChecked(clean)
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
        form.addRow(self.clean)
        form.addRow(layout_hint)
        form.addRow(buttons)
        self.setMinimumWidth(520)

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Save the experiment in", self.folder.text())
        if d:
            self.folder.setText(d)

    def values(self):
        return Path(self.folder.text()), self.name.text().strip() or "Experiment", self.as_zip.isChecked()


AF_COLORS = [(91, 91, 214), (20, 163, 163), (242, 109, 91), (214, 160, 20), (148, 103, 189),
             (44, 160, 44), (227, 119, 194), (140, 86, 75)]


def _detector_ticks(detectors, labels=14):
    """Every nth detector's name, so a 64-detector axis stays readable."""
    step = max(1, int(np.ceil(len(detectors) / labels)))
    return [(i, d.replace("-A", "")) for i, d in enumerate(detectors)][::step]


class AFLibraryView(QtWidgets.QWidget):
    """The per-cell AF library: unstained cells on two detectors, hand-drawn AF
    populations (AF Explorer style), and every library spectrum.

    Draw a polygon around a population to add its median spectrum to the
    library: click the corners, then click the first corner, double-click,
    right-click or press Enter to close it. Backspace removes the last corner
    and Esc cancels.
    """

    gates_changed = QtCore.Signal(object)   # the ControlSpec whose gates changed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.specs, self.samples, self.unmixer = [], {}, None
        self._events = None
        self._points, self._drawing = [], False
        self._preview = self._close_preview = self._snap_ring = None

        self.file_combo = QtWidgets.QComboBox()
        self.x_combo = QtWidgets.QComboBox()
        self.y_combo = QtWidgets.QComboBox()
        for c in (self.file_combo, self.x_combo, self.y_combo):
            c.currentIndexChanged.connect(self._draw)
        self.draw_btn = QtWidgets.QPushButton("Draw AF population")
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self._toggle_drawing)
        self.gate_list = QtWidgets.QListWidget()
        self.gate_list.setMaximumHeight(90)
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_selected_gate)
        self.plot = pg.PlotWidget(background="w")
        self.plot.viewport().installEventFilter(self)
        self.spectra_plot = pg.PlotWidget(background="w")
        self.spectra_plot.setLabel("left", "Normalised intensity")
        self.spectra_plot.setMaximumHeight(200)
        self.spectra_plot.addLegend(offset=(-10, 5))
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setObjectName("Hint")

        top = QtWidgets.QHBoxLayout()
        for label, w in (("Unstained", self.file_combo), ("X", self.x_combo), ("Y", self.y_combo)):
            top.addWidget(QtWidgets.QLabel(label))
            top.addWidget(w, 1 if w is self.file_combo else 0)
        top.addWidget(self.draw_btn)
        gates = QtWidgets.QHBoxLayout()
        gates.addWidget(self.gate_list, 1)
        gates.addWidget(self.remove_btn, 0, QtCore.Qt.AlignTop)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.plot, 1)
        layout.addLayout(gates)
        layout.addWidget(self.spectra_plot)
        layout.addWidget(self.summary)

        for key, slot in ((QtCore.Qt.Key_Return, self.close_polygon), (QtCore.Qt.Key_Enter, self.close_polygon),
                          (QtCore.Qt.Key_Backspace, self.undo_point), (QtCore.Qt.Key_Escape, self.cancel)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)

    @property
    def spec(self):
        i = self.file_combo.currentIndex()
        return self.specs[i] if 0 <= i < len(self.specs) else None

    def set_data(self, specs, samples, unmixer):
        """specs: the unstained-cells files; unmixer: the current one (or None)."""
        keep = self.file_combo.currentText()
        self.specs, self.samples, self.unmixer = specs, samples, unmixer
        detectors = list(unmixer.detectors) if unmixer is not None else []
        for combo, items in ((self.file_combo, [s.path.name for s in specs]),
                             (self.x_combo, detectors), (self.y_combo, detectors)):
            old = keep if combo is self.file_combo else combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if combo.findText(old) >= 0:
                combo.setCurrentIndex(combo.findText(old))
            combo.blockSignals(False)
        self._pick_default_axes()
        self._draw()

    def _pick_default_axes(self):
        """Start on the detectors where AF is brightest and where its shape varies most."""
        if self.unmixer is None or not self.specs or self.x_combo.property("chosen"):
            return
        Y = self._unstained_events()
        if Y is None:
            return
        med = np.median(Y, axis=0)
        x = int(np.argmax(med))
        shapes = np.clip(Y, 0, None) / np.maximum(np.clip(Y, 0, None).max(axis=1, keepdims=True), 1e-9)
        spread = spectral.robust_sd(shapes)
        spread[x] = -1
        y = int(np.argmax(spread))
        for combo, i in ((self.x_combo, x), (self.y_combo, y)):
            combo.blockSignals(True)
            combo.setCurrentIndex(i)
            combo.blockSignals(False)
        self.x_combo.setProperty("chosen", True)

    def _unstained_events(self):
        spec = self.spec
        if spec is None or self.unmixer is None or spec.path not in self.samples:
            return None
        try:
            Y = unmixing._detector_events(self.samples[spec.path], self.unmixer.detectors)
        except ValueError:
            return None
        if len(Y) > PREVIEW_EVENTS:
            Y = Y[np.sort(np.random.default_rng(0).choice(len(Y), PREVIEW_EVENTS, replace=False))]
        return Y

    def _draw(self, *_):
        self._clear_preview()
        self.plot.clear()
        self.spectra_plot.clear()
        self.gate_list.clear()
        Y = self._unstained_events()
        self._events = Y
        enabled = Y is not None and self.unmixer is not None and self.unmixer.per_cell_af
        self.draw_btn.setEnabled(enabled)
        if self.unmixer is not None and not self.unmixer.per_cell_af:
            self.summary.setText("Choose per-cell matching under the file table to use an AF library.")
        elif not self.specs:
            self.summary.setText("Add unstained cells to build an autofluorescence library.")
        if Y is None:
            return
        xi, yi = self.x_combo.currentIndex(), self.y_combo.currentIndex()
        if xi < 0 or yi < 0:
            return
        x, y = _asinh(Y[:, xi]), _asinh(Y[:, yi])
        lo = [min(np.percentile(v, 0.5), -0.5) for v in (x, y)]
        hi = [np.percentile(v, 99.9) for v in (x, y)]
        h, _, _ = np.histogram2d(x, y, bins=BINS, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
        density = np.log1p(h)
        rgba = PSEUDOCOLOR.map(density / (density.max() or 1), mode="byte")
        rgba[h == 0] = (0, 0, 0, 0)
        img = pg.ImageItem(rgba)
        img.setRect(QtCore.QRectF(lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]))
        self.plot.addItem(img)
        self.plot.setLabel("bottom", f"{self.x_combo.currentText()} (asinh)")
        self.plot.setLabel("left", f"{self.y_combo.currentText()} (asinh)")
        detectors = list(self.unmixer.detectors)
        for i, gate in enumerate(self.spec.af_gates):
            color = AF_COLORS[i % len(AF_COLORS)]
            n = int(unmixing.af_gate_mask(gate, Y, detectors).sum())
            item = QtWidgets.QListWidgetItem(f"{gate['name']}  ({gate['x']} vs {gate['y']}, {n:,} events shown)")
            item.setForeground(QtGui.QColor(*color))
            self.gate_list.addItem(item)
            if gate["x"] == self.x_combo.currentText() and gate["y"] == self.y_combo.currentText():
                v = _asinh(np.asarray(gate["vertices"], float))
                v = np.vstack([v, v[:1]])
                self.plot.plot(v[:, 0], v[:, 1], pen=pg.mkPen(color, width=2))
                label = pg.TextItem(gate["name"], color=color, anchor=(0, 1))
                label.setPos(*v[0])
                self.plot.addItem(label)
        self._draw_library()

    def _draw_library(self):
        um = self.unmixer
        if um is None or not um.per_cell_af:
            return
        n_drawn = sum(len(s.af_gates) for s in self.specs)
        for i, sig in enumerate(um.af_library):
            self.spectra_plot.plot(sig, pen=pg.mkPen((150, 155, 170, 140), width=1))
        # Drawn gates come first in each file's part of the library.
        drawn = 0
        for spec in self.specs:
            rows = [i for i, s in enumerate(um.af_sources or []) if s == str(spec.path)]
            for k, gate in enumerate(spec.af_gates):
                if k < len(rows):
                    self.spectra_plot.plot(um.af_library[rows[k]], name=gate["name"],
                                           pen=pg.mkPen(AF_COLORS[drawn % len(AF_COLORS)], width=2.4))
                drawn += 1
        self.spectra_plot.getAxis("bottom").setTicks([_detector_ticks(um.detectors), []])
        files = len({s for s in (um.af_sources or [])})
        self.summary.setText(
            f"{len(um.af_library)} AF spectra in the library from {files} unstained file(s), "
            f"{n_drawn} of them drawn by hand. Each cell is matched to the one that fits it best; "
            "the AF Index channel says which (1 = first spectrum).")

    # ----- drawing ---------------------------------------------------------

    def _toggle_drawing(self, on):
        self._drawing = on
        self._points = []
        self._clear_preview()
        if on:
            self.plot.viewport().setCursor(QtCore.Qt.CrossCursor)
            self.plot.setFocus()
        else:
            self.plot.viewport().unsetCursor()

    def cancel(self):
        self.draw_btn.setChecked(False)

    def undo_point(self):
        if self._drawing and self._points:
            self._points.pop()
            self._update_preview(self._points[-1] if self._points else None)

    def close_polygon(self):
        if not self._drawing or len(self._points) < 3 or self.spec is None:
            return False
        pts = np.sinh(np.asarray(self._points)) * ASINH_COFACTOR  # back to raw detector units
        names = {g["name"] for g in self.spec.af_gates}
        n = len(self.spec.af_gates) + 1
        while f"AF {n}" in names:
            n += 1
        self.add_gate({"name": f"AF {n}", "x": self.x_combo.currentText(), "y": self.y_combo.currentText(),
                       "vertices": pts.tolist()})
        self.draw_btn.setChecked(False)
        return True

    def add_gate(self, gate):
        self.spec.af_gates.append(gate)
        self.gates_changed.emit(self.spec)

    def remove_selected_gate(self):
        row = self.gate_list.currentRow()
        if self.spec is not None and 0 <= row < len(self.spec.af_gates):
            del self.spec.af_gates[row]
            self.gates_changed.emit(self.spec)

    def _view_point(self, pos):
        p = self.plot.getViewBox().mapSceneToView(self.plot.mapToScene(pos.toPoint()))
        return [float(p.x()), float(p.y())]

    def _near_first(self, pos):
        if len(self._points) < 3:
            return False
        first = self.plot.mapFromScene(self.plot.getViewBox().mapViewToScene(QtCore.QPointF(*self._points[0])))
        return np.hypot(pos.x() - first.x(), pos.y() - first.y()) <= SNAP_RADIUS_PX

    def eventFilter(self, obj, event):
        if obj is not self.plot.viewport() or not self._drawing:
            return super().eventFilter(obj, event)
        t = event.type()
        if t == QtCore.QEvent.MouseButtonDblClick:
            self.close_polygon()
            return True
        if t == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.RightButton:
                if not self.close_polygon():
                    self.cancel()
            elif event.button() == QtCore.Qt.LeftButton:
                if self._near_first(event.position()):
                    self.close_polygon()
                else:
                    pt = self._view_point(event.position())
                    self._points.append(pt)
                    self._update_preview(pt, event.position())
            return True
        if t == QtCore.QEvent.MouseMove and self._points:
            self._update_preview(self._view_point(event.position()), event.position())
            return True
        return super().eventFilter(obj, event)

    def _clear_preview(self):
        for attr in ("_preview", "_close_preview", "_snap_ring"):
            item = getattr(self, attr)
            if item is not None:
                self.plot.removeItem(item)
                setattr(self, attr, None)

    def _update_preview(self, pt, widget_pos=None):
        if not self._points or pt is None:
            self._clear_preview()
            return
        if self._preview is None:
            self._preview = pg.PlotDataItem(pen=DRAW_PEN)
            self._close_preview = pg.PlotDataItem(pen=CLOSE_PEN)
            self._snap_ring = pg.ScatterPlotItem(size=2 * SNAP_RADIUS_PX, symbol="o",
                                                 pen=pg.mkPen(theme.ACCENT, width=2),
                                                 brush=pg.mkBrush(SNAP_FILL))
            for item in (self._preview, self._close_preview, self._snap_ring):
                self.plot.addItem(item)
        xs, ys = zip(*(self._points + [pt]))
        self._preview.setData(np.asarray(xs), np.asarray(ys))
        first = self._points[0]
        snap = widget_pos is not None and self._near_first(widget_pos)
        if len(self._points) >= 2 and not snap:
            self._close_preview.setData([pt[0], first[0]], [pt[1], first[1]])
        else:
            self._close_preview.setData([], [])
        self._snap_ring.setData([first[0]] if snap else [], [first[1]] if snap else [])


class AFCheckView(QtWidgets.QWidget):
    """Unstained cells unmixed with per-cell AF and with AF as fluors, side by side.

    Unstained cells should sit at zero in every fluor channel, so the spread
    left there (and the per-detector residual) shows which AF model fits.
    """

    COLORS = {"per_cell": (91, 91, 214, 200), "fluors": (163, 169, 186, 200)}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(QtCore.Qt.RichText)
        self.spread_plot = pg.PlotWidget(background="w")
        self.spread_plot.setLabel("bottom", "Spread left in each fluor on unstained cells (rSD)")
        self.spread_plot.getViewBox().invertY(True)
        self.resid_plot = pg.PlotWidget(background="w")
        self.resid_plot.setLabel("left", "Residual RMS")
        self.resid_plot.setMaximumHeight(190)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.spread_plot, 1)
        layout.addWidget(self.resid_plot)

    def show_check(self, check, mode):
        self.spread_plot.clear()
        self.resid_plot.clear()
        if check is None:
            self.summary.setText("Add unstained cells and at least one single-color control to compare "
                                 "the two ways of handling autofluorescence.")
            return
        per, flu = check["methods"]["per_cell"], check["methods"]["fluors"]
        n = len(check["fluors"])
        x = np.arange(n)
        for offset, key in ((-0.2, "per_cell"), (0.2, "fluors")):
            self.spread_plot.addItem(pg.BarGraphItem(x0=0, y=x + offset, width=check["methods"][key]["spread"],
                                                     height=0.38, brush=self.COLORS[key], pen=None))
        self.spread_plot.getAxis("left").setTicks([[(i, f) for i, f in enumerate(check["fluors"])]])
        d = len(check["detectors"])
        for offset, key in ((-0.2, "per_cell"), (0.2, "fluors")):
            self.resid_plot.addItem(pg.BarGraphItem(x=np.arange(d) + offset,
                                                    height=check["methods"][key]["residual_rms"],
                                                    width=0.38, brush=self.COLORS[key], pen=None))
        self.resid_plot.getAxis("bottom").setTicks([_detector_ticks(check["detectors"]), []])
        a, b = per["spread"].sum(), flu["spread"].sum()
        better = "per_cell" if a <= b else "fluors"
        change = abs(a - b) / max(a, b, 1e-9) * 100
        using = unmixing.AF_METHOD_LABELS.get(mode, "")
        key = "  ".join(f"<span style='color:rgb{self.COLORS[k][:3]}'>&#9632;</span> "
                        f"{unmixing.AF_METHOD_LABELS[k]}" for k in ("per_cell", "fluors"))
        text = (f"{key}<br>Unstained cells ({check['n_events']:,} events) unmixed both ways. Every fluor should sit "
                f"at 0, so lower bars are better. <b>{unmixing.AF_METHOD_LABELS[better]}</b> leaves "
                f"{change:.0f}% less spread in total.")
        if using and better != mode and change > 10:
            text += f" You are using {using}; consider switching."
        self.summary.setText(text)
