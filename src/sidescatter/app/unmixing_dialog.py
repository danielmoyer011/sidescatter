"""Spectral unmixing workbench: controls -> signatures -> live preview -> unmix samples."""

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import fcs, spectral, unmixing
from .graph_view import BINS, PSEUDOCOLOR

COLS = ["File", "Fluor / marker", "Type", "Carrier", "Negative", "Peak", "Positives"]
ASINH_COFACTOR = 2000.0
PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
           (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


def _asinh(x):
    return np.arcsinh(np.asarray(x, dtype=float) / ASINH_COFACTOR)


class UnmixingDialog(QtWidgets.QDialog):
    unmixed = QtCore.Signal(list)  # new sample ids

    def __init__(self, workspace, parent=None, control_paths=()):
        super().__init__(parent)
        self.setWindowTitle("Spectral unmixing")
        self.resize(1300, 820)
        self.ws = workspace
        self.specs = []
        self.events = {}        # path -> detector events
        self.unmixer = None
        self.results = []
        self._preview_cache = {}

        # ----- controls table
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._show_control)
        self.table.itemChanged.connect(self._on_name_edited)

        add_btn = QtWidgets.QPushButton("Add control files…")
        add_btn.clicked.connect(self._add_files)
        remove_btn = QtWidgets.QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected)

        self.method = QtWidgets.QComboBox()
        self.method.addItem("Weighted least squares (WLS)", "wls")
        self.method.addItem("Ordinary least squares (OLS)", "ols")
        self.method.currentIndexChanged.connect(self.recalculate)
        self.af_check = QtWidgets.QCheckBox("Extract autofluorescence from unstained cells")
        self.af_check.setChecked(True)
        self.af_check.toggled.connect(self.recalculate)
        self.af_count = QtWidgets.QSpinBox()
        self.af_count.setRange(1, 6)
        self.af_count.setPrefix("Signatures: ")
        self.af_count.valueChanged.connect(self.recalculate)

        options = QtWidgets.QFormLayout()
        options.addRow("Method", self.method)
        af_row = QtWidgets.QHBoxLayout()
        af_row.addWidget(self.af_check)
        af_row.addWidget(self.af_count)
        options.addRow(af_row)

        self.complexity = QtWidgets.QLabel("Add single-stain and unstained controls to begin.")
        self.complexity.setWordWrap(True)

        left = QtWidgets.QVBoxLayout()
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(add_btn)
        buttons.addWidget(remove_btn)
        buttons.addStretch()
        left.addLayout(buttons)
        left.addWidget(self.table, 1)
        left.addLayout(options)
        left.addWidget(self.complexity)

        # ----- right side tabs
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
            "Select a control. Drag the shaded region to choose which events count as positive.")
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

        # ----- bottom
        self.unmix_btn = QtWidgets.QPushButton("Unmix samples…")
        self.unmix_btn.setEnabled(False)
        self.unmix_btn.clicked.connect(self._unmix_samples)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.reject)

        split = QtWidgets.QSplitter()
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        split.addWidget(lw)
        split.addWidget(self.tabs)
        split.setSizes([560, 740])

        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(self.unmix_btn)
        bottom.addWidget(close_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(split, 1)
        layout.addLayout(bottom)

        self._fill_preview_samples()
        if control_paths:
            self.add_controls(control_paths)

    # ----- controls --------------------------------------------------------

    def _add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Choose unmixing controls", "", "FCS files (*.fcs *.FCS)")
        if paths:
            self.add_controls(paths)

    def add_controls(self, paths):
        known = {s.path for s in self.specs}
        new = [s for s in unmixing.guess_controls(paths) if s.path not in known]
        for spec in new:
            sample = fcs.load_sample(spec.path)
            self.events[spec.path] = fcs.raw_events(sample, fcs.detector_channels(sample))
        self.specs.extend(new)
        self._fill_table()
        self.recalculate()

    def _remove_selected(self):
        row = self._selected_row()
        if row is not None:
            del self.specs[row]
            self._fill_table()
            self.recalculate()

    def _selected_row(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        return min(rows) if rows else None

    def _fill_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.specs))
        unstained = [s.path.name for s in self.specs if s.role == "unstained"]
        for row, spec in enumerate(self.specs):
            file_item = QtWidgets.QTableWidgetItem(spec.path.name)
            file_item.setFlags(file_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, file_item)
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(spec.name))
            self.table.setCellWidget(row, 2, self._combo(
                [("Single stain", "single_stain"), ("Unstained", "unstained")], spec.role,
                lambda v, s=spec: self._set(s, "role", v)))
            self.table.setCellWidget(row, 3, self._combo(
                [("Beads", "beads"), ("Cells", "cells")], spec.carrier,
                lambda v, s=spec: self._set(s, "carrier", v)))
            negs = [("Auto (matching unstained)", "auto"), ("Internal (dim events)", "internal")]
            negs += [(n, str(next(x.path for x in self.specs if x.path.name == n))) for n in unstained]
            self.table.setCellWidget(row, 4, self._combo(
                negs, spec.negative, lambda v, s=spec: self._set(s, "negative", v),
                enabled=spec.role == "single_stain"))
            for col in (5, 6):
                item = QtWidgets.QTableWidgetItem("")
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def _combo(self, options, current, on_change, enabled=True):
        combo = QtWidgets.QComboBox()
        for label, value in options:
            combo.addItem(label, value)
        i = combo.findData(current)
        combo.setCurrentIndex(max(i, 0))
        combo.setEnabled(enabled)
        combo.currentIndexChanged.connect(lambda _: on_change(combo.currentData()))
        return combo

    def _set(self, spec, attr, value):
        setattr(spec, attr, value)
        if attr == "role":
            self._fill_table()
        self.recalculate()

    def _on_name_edited(self, item):
        if item.column() == 1:
            self.specs[item.row()].name = item.text().strip() or self.specs[item.row()].name
            self.recalculate()

    # ----- computing -------------------------------------------------------

    def recalculate(self):
        singles = [s for s in self.specs if s.role == "single_stain"]
        if not singles:
            self.unmixer = None
            self.unmix_btn.setEnabled(False)
            return
        try:
            self.unmixer, self.results = unmixing.build_unmixer(
                self.specs, method=self.method.currentData(),
                n_autofluorescence=self.af_count.value(), autofluorescence=self.af_check.isChecked())
        except Exception as exc:
            self.complexity.setText(f"Couldn't build the unmixing matrix: {exc}")
            self.unmix_btn.setEnabled(False)
            return
        by_path = {r.spec.path: r for r in self.results}
        self.table.blockSignals(True)
        for row, spec in enumerate(self.specs):
            r = by_path.get(spec.path)
            self.table.item(row, 5).setText(r.peak_detector if r else "")
            self.table.item(row, 6).setText(f"{r.n_positive:,}" if r else "")
        self.table.blockSignals(False)
        self._preview_cache.clear()
        self.unmix_btn.setEnabled(True)
        self._draw_signatures()
        self._draw_similarity()
        self._show_control()
        self._fill_preview_channels()
        self._draw_preview()

    def _draw_signatures(self):
        self.sig_plot.clear()
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
        worst = (0, None)
        for i in range(n):
            for j in range(n):
                v = sim[i, j]
                item = QtWidgets.QTableWidgetItem(f"{v:.2f}")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                if i != j:
                    item.setBackground(QtGui.QColor.fromHsvF(0.0, min(max((v - 0.5) * 2, 0), 1), 1.0))
                    if v > worst[0]:
                        worst = (v, (um.names[i], um.names[j]))
                self.sim_table.setItem(i, j, item)
        self.sim_table.resizeColumnsToContents()
        ci = spectral.complexity_index(um.matrix)
        text = f"Complexity index {ci:.1f} with {n} components."
        if worst[1]:
            text += f" Most similar pair: {worst[1][0]} and {worst[1][1]} ({worst[0]:.2f})."
            if worst[0] > 0.98:
                text += " Pairs above 0.98 are very hard to separate."
        self.complexity.setText(text)

    def _show_control(self):
        self.ctrl_plot.clear()
        row = self._selected_row()
        if row is None or self.unmixer is None:
            return
        spec = self.specs[row]
        result = next((r for r in self.results if r.spec.path == spec.path), None)
        if result is None:
            self.ctrl_hint.setText(f"{spec.path.name} is an unstained control.")
            return
        peak = self.unmixer.detectors.index(result.peak_detector)
        values = _asinh(self.events[spec.path][:, peak])
        counts, edges = np.histogram(values, bins=128)
        self.ctrl_plot.plot(edges, counts, stepMode="center", fillLevel=0,
                            brush=(90, 120, 200, 120), pen=pg.mkPen((40, 60, 140)))
        pos_vals = values[result.positive_mask]
        region = pg.LinearRegionItem((float(pos_vals.min()), float(pos_vals.max())),
                                     brush=(44, 160, 44, 50))
        self.ctrl_plot.addItem(region)
        self.ctrl_plot.setLabel("bottom", f"{result.peak_detector} (asinh scale)")
        self.ctrl_hint.setText(
            f"{spec.name}: {result.n_positive:,} positive events; negative from "
            f"{result.negative_source}. Drag the green region to change the positives.")

        def changed():
            lo, hi = region.getRegion()
            spec.positive_mask = (values >= lo) & (values <= hi)
            if spec.positive_mask.sum() < 10:
                spec.positive_mask = None
            self.recalculate()

        region.sigRegionChangeFinished.connect(changed)

    # ----- preview ---------------------------------------------------------

    def _fill_preview_samples(self):
        self.preview_sample.blockSignals(True)
        self.preview_sample.clear()
        for sid, sample in self.ws.samples.items():
            if fcs.is_spectral(sample) and "unmixed_from" not in self.ws.sources.get(sid, {}):
                self.preview_sample.addItem(sid, sid)
        self.preview_sample.blockSignals(False)

    def _fill_preview_channels(self):
        names = self.unmixer.names
        for combo, default in ((self.preview_x, 0), (self.preview_y, 1)):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            i = combo.findText(keep)
            combo.setCurrentIndex(i if i >= 0 else min(default, len(names) - 1))
            combo.blockSignals(False)

    def _draw_preview(self):
        self.preview_plot.clear()
        self.resid_plot.clear()
        sid = self.preview_sample.currentData()
        if self.unmixer is None or sid is None:
            return
        sample = self.ws.samples[sid]
        by_name = {c.pnn: c for c in fcs.channels(sample)}
        if any(d not in by_name for d in self.unmixer.detectors):
            return
        Y = fcs.raw_events(sample, [by_name[d] for d in self.unmixer.detectors])
        A = self.unmixer.unmix(Y)
        xi, yi = self.preview_x.currentIndex(), self.preview_y.currentIndex()
        x, y = _asinh(A[:, xi]), _asinh(A[:, yi])
        lo = min(np.percentile(x, 0.5), np.percentile(y, 0.5), -1)
        hi = max(np.percentile(x, 99.9), np.percentile(y, 99.9))
        h, xe, ye = np.histogram2d(x, y, bins=BINS, range=[[lo, hi], [lo, hi]])
        density = np.log1p(h)
        rgba = PSEUDOCOLOR.map(density / (density.max() or 1), mode="byte")
        rgba[h == 0] = (0, 0, 0, 0)
        img = pg.ImageItem(rgba)
        img.setRect(QtCore.QRectF(lo, lo, hi - lo, hi - lo))
        self.preview_plot.addItem(img)
        self.preview_plot.setLabel("bottom", f"{self.unmixer.names[xi]} (asinh)")
        self.preview_plot.setLabel("left", f"{self.unmixer.names[yi]} (asinh)")

        R = spectral.residuals(Y, A, self.unmixer.matrix)
        rms = np.sqrt((R ** 2).mean(axis=0))
        bars = pg.BarGraphItem(x=np.arange(len(rms)), height=rms, width=0.8, brush=(214, 39, 40, 160))
        self.resid_plot.addItem(bars)
        ticks = [(i, d.replace("-A", "")) for i, d in enumerate(self.unmixer.detectors)]
        self.resid_plot.getAxis("bottom").setTicks([ticks[::3], []])

    # ----- applying --------------------------------------------------------

    def _unmix_samples(self):
        candidates = [sid for sid, s in self.ws.samples.items()
                      if fcs.is_spectral(s) and "unmixed_from" not in self.ws.sources.get(sid, {})
                      and sid not in {sp.path.name for sp in self.specs}]
        if not candidates:
            QtWidgets.QMessageBox.information(
                self, "No samples", "Add your raw spectral sample files to the workspace first "
                "(File › Add samples), then unmix them here.")
            return
        dlg = _PickSamples(candidates, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        name = "Unmixing 1"
        n = 1
        while name in self.ws.unmixers:
            n += 1
            name = f"Unmixing {n}"
        self.ws.add_unmixer(name, self.unmixer)
        new_ids = [self.ws.unmix(sid, name, group="Unmixed") for sid in dlg.chosen()]
        self.unmixed.emit(new_ids)
        self.accept()


class _PickSamples(QtWidgets.QDialog):
    def __init__(self, sample_ids, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Unmix samples")
        self.list = QtWidgets.QListWidget()
        for sid in sample_ids:
            item = QtWidgets.QListWidgetItem(sid)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.list.addItem(item)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Unmix these samples with the current controls:"))
        layout.addWidget(self.list)
        layout.addWidget(buttons)

    def chosen(self):
        return [self.list.item(i).text() for i in range(self.list.count())
                if self.list.item(i).checkState() == QtCore.Qt.Checked]
