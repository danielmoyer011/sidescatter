"""Clean data: the settings dialog, and the QC window to check and correct the result.

The QC window lists every cleaned sample (amber when it lost more than 20%)
and shows the selected one as Time plots: event rate on top, then each
checked channel, with removed stretches shaded. Stretches can be dragged on
the event-rate plot, added, or deleted; edits are kept when the sample is
cleaned again with the same settings.
"""

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import icons, qc_report, theme
from ..core import cleaning

STEPS = ("Margins", "Flow rate", "Isolation tree", "MAD", "Consecutive windows")
REGION_BRUSH = theme.qcolor(theme.CORAL, 55)
REGION_HOVER = theme.qcolor(theme.CORAL, 95)
PLOT_HEIGHT = 120


class CleanDialog(QtWidgets.QDialog):
    """Preset, advanced PeacoQC settings and the channels to check."""

    def __init__(self, ws, sample_ids, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clean data")
        self.ws = ws
        self.sample_ids = list(sample_ids)
        s = settings or ws.qc_settings
        intro = QtWidgets.QLabel(
            f"Removes margin events, flow-rate jumps and unstable stretches (clogs, bubbles, drift) "
            f"from {len(self.sample_ids)} sample(s), with the same settings. The result is a "
            f"<b>QC passed</b> population at the top of each sample's gates; the files aren't changed.")
        intro.setWordWrap(True)
        intro.setTextFormat(QtCore.Qt.RichText)

        self.preset = QtWidgets.QComboBox()
        self.preset.addItems([*cleaning.PRESETS, "Custom"])
        self.preset.setCurrentText(s.preset if s.preset in cleaning.PRESETS else "Custom")

        self.margins = QtWidgets.QCheckBox("Margin events (piled up at a channel's edge)")
        self.flow = QtWidgets.QCheckBox("Flow rate (clogs, bubbles, pauses)")
        self.signal = QtWidgets.QCheckBox("Signal stability (PeacoQC)")
        for box, on in ((self.margins, s.margins), (self.flow, s.flow_rate), (self.signal, s.signal)):
            box.setChecked(on)

        self.mad = _spin(1, 20, 0.5, s.mad, 1)
        self.it = _spin(0.3, 0.95, 0.01, s.it_limit, 2)
        self.consecutive = QtWidgets.QSpinBox()
        self.consecutive.setRange(1, 50)
        self.consecutive.setValue(int(s.consecutive_bins))
        self.epb = QtWidgets.QSpinBox()
        self.epb.setRange(0, 1_000_000)
        self.epb.setSingleStep(500)
        self.epb.setSpecialValueText("Automatic")
        self.epb.setValue(int(s.events_per_bin))
        self.zeros = QtWidgets.QCheckBox("Ignore zeros (mass cytometry)")
        self.zeros.setChecked(s.remove_zeros)
        self.mad.valueChanged.connect(self._custom)
        self.it.valueChanged.connect(self._custom)
        self.consecutive.valueChanged.connect(self._custom)
        self.preset.currentTextChanged.connect(self._preset_picked)

        adv = QtWidgets.QGroupBox("Advanced (PeacoQC settings)")
        af = QtWidgets.QFormLayout(adv)
        af.addRow("MAD", self.mad)
        af.addRow("Isolation tree limit", self.it)
        af.addRow("Consecutive windows", self.consecutive)
        af.addRow("Events per window", self.epb)
        af.addRow(self.zeros)
        hint = QtWidgets.QLabel("Lower MAD or isolation tree limit removes more.")
        hint.setObjectName("Hint")
        af.addRow(hint)

        first = self.sample_ids[0] if self.sample_ids else None
        self.channels = QtWidgets.QListWidget()
        self.channels.setMaximumHeight(170)
        if first is not None:
            chosen = set(s.channels or ws.default_qc_channels(first))
            for c in ws.channels(first):
                if c.kind == "time":
                    continue
                item = QtWidgets.QListWidgetItem(c.label)
                item.setData(QtCore.Qt.UserRole, c.pnn)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked if c.pnn in chosen else QtCore.Qt.Unchecked)
                self.channels.addItem(item)
        self._default_channels = s.channels is None
        self.channels.itemChanged.connect(lambda *_: setattr(self, "_default_channels", False))
        ch_hint = QtWidgets.QLabel("Default: every fluorescence channel plus FSC and SSC; Time and "
                                   "autofluorescence are left out.")
        ch_hint.setWordWrap(True)
        ch_hint.setObjectName("Hint")

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Clean")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QtWidgets.QFormLayout()
        form.addRow("Strictness", self.preset)
        steps = QtWidgets.QVBoxLayout()
        for b in (self.margins, self.flow, self.signal):
            steps.addWidget(b)
        form.addRow("Check", steps)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(adv)
        layout.addWidget(QtWidgets.QLabel("Channels checked"))
        layout.addWidget(self.channels)
        layout.addWidget(ch_hint)
        layout.addWidget(buttons)
        self.setMinimumWidth(480)

    def _preset_picked(self, name):
        if name in cleaning.PRESETS:
            for w, v in ((self.mad, cleaning.PRESETS[name]["mad"]), (self.it, cleaning.PRESETS[name]["it_limit"]),
                         (self.consecutive, 5)):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)

    def _custom(self, *_):
        name = self.preset.currentText()
        if name in cleaning.PRESETS:
            p = cleaning.PRESETS[name]
            if (self.mad.value(), round(self.it.value(), 2), self.consecutive.value()) != \
                    (p["mad"], p["it_limit"], 5):
                self.preset.blockSignals(True)
                self.preset.setCurrentText("Custom")
                self.preset.blockSignals(False)

    def settings(self):
        chans = None
        if not self._default_channels:
            chans = [self.channels.item(i).data(QtCore.Qt.UserRole) for i in range(self.channels.count())
                     if self.channels.item(i).checkState() == QtCore.Qt.Checked]
        return cleaning.Settings(
            preset=self.preset.currentText(), mad=self.mad.value(), it_limit=round(self.it.value(), 3),
            consecutive_bins=self.consecutive.value(), events_per_bin=self.epb.value(),
            remove_zeros=self.zeros.isChecked(), margins=self.margins.isChecked(),
            flow_rate=self.flow.isChecked(), signal=self.signal.isChecked(), channels=chans)


def _spin(lo, hi, step, value, decimals):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(float(value))
    return w


def run_cleaning(ws, sample_ids, settings, parent=None):
    """Clean with a progress dialog. Returns the results, or None if it failed."""
    progress = QtWidgets.QProgressDialog("Cleaning…", "", 0, len(sample_ids) + 1, parent)
    progress.setWindowTitle("Clean data")
    progress.setCancelButton(None)
    progress.setMinimumDuration(300)
    progress.setWindowModality(QtCore.Qt.WindowModal)

    def step(done, total, sid):
        progress.setValue(done)
        progress.setLabelText(f"Cleaning {sid}")
        QtWidgets.QApplication.processEvents()

    try:
        return ws.clean(sample_ids, settings, step)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(parent, "Clean data", f"Cleaning failed:\n\n{exc}")
        return None
    finally:
        progress.close()


class QCWindow(QtWidgets.QWidget):
    changed = QtCore.Signal()

    COLS = ("Sample", "Events", "Removed", *STEPS, "Notes")

    def __init__(self, ws, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Data cleaning")
        self.setWindowIcon(icons.icon("clean_data"))
        self.ws = ws
        self.sample_id = None
        self._plot = None
        self._regions = []

        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._picked)

        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-size: 12pt; font-weight: 600;")
        self.notes = QtWidgets.QLabel()
        self.notes.setWordWrap(True)
        self.notes.setObjectName("Hint")

        self.add_btn = QtWidgets.QPushButton("Add stretch")
        self.add_btn.setToolTip("Add a removed stretch in the middle of the view, then drag its edges")
        self.add_btn.clicked.connect(self.add_stretch)
        self.del_btn = QtWidgets.QPushButton("Delete stretch")
        self.del_btn.setToolTip("Delete the selected stretch (click one on the event-rate plot first). "
                                "Right-clicking a stretch also deletes it.")
        self.del_btn.clicked.connect(self.delete_selected_stretch)
        self.reset_btn = QtWidgets.QPushButton("Reset to automatic")
        self.reset_btn.clicked.connect(self.reset_stretches)
        edit_row = QtWidgets.QHBoxLayout()
        for b in (self.add_btn, self.del_btn, self.reset_btn):
            edit_row.addWidget(b)
        edit_row.addStretch()

        self.plots = pg.GraphicsLayoutWidget()
        self.plots.setBackground(theme.SURFACE)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.plots)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.title)
        right.addWidget(self.notes)
        right.addLayout(edit_row)
        right.addWidget(self.scroll, 1)
        rw = QtWidgets.QWidget()
        rw.setLayout(right)

        split = QtWidgets.QSplitter()
        split.addWidget(self.table)
        split.addWidget(rw)
        split.setSizes([520, 760])

        again = QtWidgets.QPushButton("Clean again…")
        again.clicked.connect(self.clean_again)
        export = QtWidgets.QPushButton("Export cleaned FCS…")
        export.setToolTip("Write the events that passed as new FCS files (optional; the workspace "
                          "already uses QC passed)")
        export.clicked.connect(self.export_fcs)
        pdf = QtWidgets.QPushButton("Save PDF report…")
        pdf.clicked.connect(self.save_pdf)
        remove = QtWidgets.QPushButton("Remove cleaning")
        remove.setToolTip("Undo cleaning for the selected samples")
        remove.clicked.connect(self.remove_cleaning)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(remove)
        bottom.addStretch()
        for b in (again, export, pdf):
            bottom.addWidget(b)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(split, 1)
        layout.addLayout(bottom)
        self.resize(1300, 820)
        self.refresh()

    def set_workspace(self, ws):
        self.ws = ws
        self.sample_id = None
        self.refresh()

    # ----- summary table -----------------------------------------------------

    def cleaned_ids(self):
        return [sid for sid in self.ws.samples if sid in self.ws.qc]

    def refresh(self):
        keep = self.sample_id
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for sid in self.cleaned_ids():
            res = self.ws.qc[sid]
            pct = res.percent_removed(self.ws.time_values(sid)[0])
            row = self.table.rowCount()
            self.table.insertRow(row)
            cells = [sid, f"{res.n_events:,}", f"{pct:.2f}%"]
            cells += [f"{res.removed_by[s]:.2f}%" if s in res.removed_by else "–" for s in STEPS]
            notes = ("Edited by hand. " if res.edited else "") + " ".join(res.warnings)
            cells.append(notes)
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setData(QtCore.Qt.UserRole, sid)
                if 0 < col < len(cells) - 1:
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                if qc_report.flagged(pct):
                    item.setBackground(theme.qcolor(qc_report.AMBER, 90))
                    item.setToolTip(f"More than {qc_report.AMBER_PCT:.0f}% of events were removed: "
                                    "check this sample's Time plots.")
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        ids = self.cleaned_ids()
        if keep not in ids:
            keep = ids[0] if ids else None
        if keep is not None:
            self.table.selectRow(ids.index(keep))
        self.show_sample(keep)

    def selected_ids(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        return [self.table.item(r, 0).data(QtCore.Qt.UserRole) for r in rows]

    def _picked(self):
        ids = self.selected_ids()
        if ids:
            self.show_sample(ids[0])

    # ----- Time plots -----------------------------------------------------------

    def _scale(self, sid):
        time, _ = self.ws.time_values(sid)
        return (self.ws.time_step(sid) or 1.0) if time is not None else 1.0

    def show_sample(self, sid):
        self.sample_id = sid
        self.plots.clear()
        self._regions = []
        self._selected_region = None
        self._plot = None
        for b in (self.add_btn, self.del_btn, self.reset_btn):
            b.setEnabled(sid is not None)
        if sid is None:
            self.title.setText("No cleaned samples")
            self.notes.setText("Use Clean data on the ribbon to clean the samples of a group.")
            return
        res = self.ws.qc[sid]
        data = self.ws.qc_plot_data(sid)
        self.title.setText(f"{sid}: {data['removed_pct']:.2f}% removed")
        self.title.setStyleSheet("font-size: 12pt; font-weight: 600; color: %s;"
                                 % (qc_report.AMBER if qc_report.flagged(data["removed_pct"]) else theme.INK))
        trend = [] if res.trend.startswith("No ") else [f"{res.trend}."]
        self.notes.setText(" ".join([*trend, *res.warnings,
                                     "Drag the shaded stretches on the event-rate plot to change what "
                                     "is removed; right-click one to delete it."]))
        x0, x1 = data["x_range"]
        rate_plot = self.plots.addPlot(row=0, col=0)
        self._style(rate_plot, "Event rate")
        xs, ys = data["rate"]
        rate_plot.plot(xs, ys, pen=pg.mkPen(theme.TEAL, width=1.4))
        rate_plot.setXRange(x0, x1, padding=0)
        self._plot = rate_plot
        for a, b in data["stretches"]:
            self._add_region(a, b, editable=True)
        prev = rate_plot
        for i, ch in enumerate(data["channels"], start=1):
            plt = self.plots.addPlot(row=i, col=0)
            self._style(plt, ch["name"])
            lo, hi = ch["y_range"]
            img = pg.ImageItem(np.log1p(ch["hist"]))
            img.setLookupTable(qc_lut())
            img.setLevels([0, float(np.log1p(ch["hist"]).max()) or 1.0])
            img.setRect(QtCore.QRectF(x0, lo, x1 - x0, hi - lo))
            plt.addItem(img)
            for txs, tys in ch["peaks"]:
                plt.plot(txs, tys, pen=pg.mkPen(theme.ACCENT_DARK, width=1.2))
            for a, b in data["stretches"]:
                plt.addItem(pg.LinearRegionItem((a, b), movable=False, brush=REGION_BRUSH,
                                                pen=pg.mkPen(None)))
            plt.setXLink(prev)
            plt.setXRange(x0, x1, padding=0)
            plt.setYRange(lo, hi, padding=0)
        last = len(data["channels"])
        self.plots.getItem(last, 0).setLabel("bottom", data["x_label"])
        self.plots.setMinimumHeight(PLOT_HEIGHT * (last + 1))

    @staticmethod
    def _style(plt, title):
        plt.setTitle(title, size="9pt", color=theme.INK)
        plt.setMouseEnabled(x=True, y=False)
        plt.hideButtons()
        plt.getAxis("left").setWidth(48)
        plt.getAxis("left").setStyle(showValues=title == "Event rate")
        plt.setMenuEnabled(False)

    def _add_region(self, a, b, editable):
        region = pg.LinearRegionItem((a, b), brush=REGION_BRUSH, hoverBrush=REGION_HOVER,
                                     movable=editable)
        region.sigRegionChangeFinished.connect(lambda *_: self._commit())
        region.mouseClickEvent = lambda ev, r=region: self._region_clicked(r, ev)
        self._plot.addItem(region)
        self._regions.append(region)
        return region

    def _region_clicked(self, region, ev):
        if ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self._regions.remove(region)
            self._commit()
        elif ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            self._selected_region = region
            for r in self._regions:
                r.setBrush(REGION_HOVER if r is region else REGION_BRUSH)

    def spans(self):
        """The stretches as shown, in Time units."""
        scale = self._scale(self.sample_id)
        return [tuple(v / scale for v in sorted(r.getRegion())) for r in self._regions]

    def _commit(self):
        if self.sample_id is None:
            return
        self.ws.set_qc_stretches(self.sample_id, self.spans())
        self.changed.emit()
        QtCore.QTimer.singleShot(0, self.refresh)

    def add_stretch(self):
        if self._plot is None:
            return
        (v0, v1), _ = self._plot.viewRange()
        mid, half = (v0 + v1) / 2, (v1 - v0) * 0.025
        self._add_region(mid - half, mid + half, True)
        self._commit()

    def delete_selected_stretch(self):
        region = getattr(self, "_selected_region", None)
        if region in self._regions:
            self._regions.remove(region)
            self._commit()

    def reset_stretches(self):
        if self.sample_id is not None:
            self.ws.reset_qc_stretches(self.sample_id)
            self.changed.emit()
            self.refresh()

    # ----- actions ---------------------------------------------------------------

    def clean_again(self):
        ids = self.selected_ids() or self.cleaned_ids()
        if not ids:
            return
        dlg = CleanDialog(self.ws, ids, cleaning.Settings.from_dict(self.ws.qc[ids[0]].settings), self)
        if dlg.exec() == QtWidgets.QDialog.Accepted and run_cleaning(self.ws, ids, dlg.settings(), self):
            self.changed.emit()
            self.refresh()

    def remove_cleaning(self):
        ids = self.selected_ids()
        if not ids:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Remove cleaning", f"Undo cleaning for {len(ids)} sample(s)? Their gates keep working "
                                     "on all events again.")
        if answer == QtWidgets.QMessageBox.Yes:
            self.ws.remove_qc(ids)
            self.changed.emit()
            self.refresh()

    def export_fcs(self):
        ids = self.selected_ids() or self.cleaned_ids()
        if not ids:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Save cleaned FCS files in")
        if not folder:
            return
        written = []
        for sid in ids:
            stem = Path(sid).stem if sid.lower().endswith(".fcs") else sid
            path = Path(folder) / f"{cleaning_safe(stem)}_QC.fcs"
            self.ws.export_cleaned(sid, path)
            written.append(path.name)
        QtWidgets.QMessageBox.information(self, "Export cleaned FCS",
                                          f"Wrote {len(written)} file(s) to {folder}.")

    def report_pages(self, ids=None):
        return [{"sample": sid, "result": self.ws.qc[sid], "plot": self.ws.qc_plot_data(sid)}
                for sid in (ids or self.cleaned_ids())]

    def save_pdf(self, path=None):
        if not self.cleaned_ids():
            return None
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PDF report", "data cleaning.pdf",
                                                            "PDF (*.pdf)")
            if not path:
                return None
        qc_report.write_pdf(path, self.report_pages())
        return path


def cleaning_safe(name):
    return "".join(c if c.isalnum() or c in " ._-()" else "_" for c in name).strip() or "sample"


_LUT = None


def qc_lut():
    global _LUT
    if _LUT is None:
        from .graph_view import PSEUDOCOLOR

        lut = PSEUDOCOLOR.getLookupTable(0.0, 1.0, 256, alpha=True)
        lut[0] = (0, 0, 0, 0)
        _LUT = lut
    return _LUT
